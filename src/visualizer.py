import numpy as np
import pyqtgraph.opengl as gl
from PyQt6.QtGui import QColor, QVector3D
from PyQt6.QtCore import Qt

class Visualizer3D(gl.GLViewWidget):
    def __init__(self, M, N, K, systolic_size=None):
        super().__init__()
        self.M = M
        self.N = N
        self.K = K
        self.systolic_size = systolic_size
        
        # Adjust camera to see A(left), B(right), C(top)
        self.setCameraPosition(distance=max(M, N, K) * 3.5, elevation=45, azimuth=135)
        
        # Set Orthographic Projection (Pseudo)
        # self.opts['fov'] = 1  
        # self.opts['distance'] = max(M, N, K) * 100 
        
        self.setBackgroundColor('w') # White background

        # Colors
        self.color_default = (0.6, 0.6, 0.6, 0.3) 
        self.color_active = (1.0, 0.5, 0.0, 0.9)  # Strong Orange
        self.color_done = (0.0, 0.7, 0.0, 0.7)    # Medium Green
        
        # Projection Colors (Darker base for visibility)
        self.base_color_A = (0.8, 0.0, 0.0, 0.3) 
        self.base_color_B = (0.0, 0.0, 0.8, 0.3) 
        self.base_color_C = (0.0, 0.6, 0.6, 0.3) 
        
        # Highlight Colors
        self.active_color_A = (1.0, 0.0, 0.0, 0.9) # Bright Red (Direct)
        self.active_color_B = (0.0, 0.0, 1.0, 0.9) # Bright Blue (Direct)
        
        self.transferred_color_A = (0.8, 0.5, 0.5, 0.9) # Lighter Red (Transferred)
        self.transferred_color_B = (0.5, 0.5, 0.8, 0.9) # Lighter Blue (Transferred)
        
        self.active_color_C = (0.0, 1.0, 1.0, 0.9) # Bright Cyan
        
        self.setup_volume_grid()
        self.setup_projections()
        self.setup_floor()
        self.setup_ceiling_C()
        
    def setup_volume_grid(self):
        # Optimized vector initialization
        self.indices_map = {}
        
        # Create coordinate grids
        # numpy.indices returns (3, M, N, K) array of coordinates
        # We need (M*N*K, 3)
        grid = np.indices((self.M, self.N, self.K))
        self.positions = grid.reshape(3, -1).T
        
        # Pre-allocate colors array
        count = self.positions.shape[0]
        self.colors = np.tile(self.color_default, (count, 1))
        
        self.scatter = gl.GLScatterPlotItem(
            pos=self.positions,
            color=self.colors,
            size=0.3, # Smaller size for volume grid
            pxMode=False
        )
        self.scatter.setGLOptions('translucent')
        self.addItem(self.scatter)

    def get_volume_index(self, x, y, z):
        return x * (self.N * self.K) + y * self.K + z

    def setup_projections(self):
        # Vectorized setup for A
        # A: x varies (0..M), z varies (0..K), y fixed at -2
        grid_A = np.indices((self.M, self.K))
        
        x_A = grid_A[0].ravel()
        z_A = grid_A[1].ravel()
        y_A = np.full_like(x_A, -2)
        
        self.proj_A_pos = np.stack((x_A, y_A, z_A), axis=1)
        self.proj_A_colors = np.tile(self.base_color_A, (len(self.proj_A_pos), 1))
        
        self.scatter_A = gl.GLScatterPlotItem(
            pos=self.proj_A_pos,
            color=self.proj_A_colors,
            size=0.8,
            pxMode=False
        )
        self.scatter_A.setGLOptions('translucent')
        self.addItem(self.scatter_A)
        
        # Vectorized setup for B
        # B: y varies (0..N), z varies (0..K), x fixed at -2
        grid_B = np.indices((self.N, self.K))
        y_B = grid_B[0].ravel()
        z_B = grid_B[1].ravel()
        x_B = np.full_like(y_B, -2)
        
        self.proj_B_pos = np.stack((x_B, y_B, z_B), axis=1)
        self.proj_B_colors = np.tile(self.base_color_B, (len(self.proj_B_pos), 1))
        
        self.scatter_B = gl.GLScatterPlotItem(
            pos=self.proj_B_pos,
            color=self.proj_B_colors,
            size=0.8,
            pxMode=False
        )
        self.scatter_B.setGLOptions('translucent')
        self.addItem(self.scatter_B)
    
    def setup_ceiling_C(self):
        # Vectorized setup for C (Top Plane)
        # C: x varies (0..M), y varies (0..N), z fixed at K + 1
        grid_C = np.indices((self.M, self.N))
        x_C = grid_C[0].ravel()
        y_C = grid_C[1].ravel()
        z_C = np.full_like(x_C, self.K + 1)
        
        self.proj_C_pos = np.stack((x_C, y_C, z_C), axis=1)
        self.proj_C_colors = np.tile(self.base_color_C, (len(self.proj_C_pos), 1))
        
        self.scatter_C = gl.GLScatterPlotItem(
            pos=self.proj_C_pos,
            color=self.proj_C_colors,
            size=0.8,
            pxMode=False
        )
        self.scatter_C.setGLOptions('translucent')
        self.addItem(self.scatter_C)

    def get_A_index(self, x, z):
        return x * self.K + z
        
    def get_B_index(self, y, z):
        return y * self.K + z

    def get_C_index(self, x, y):
        return x * self.N + y

    def setup_floor(self):
        grid = gl.GLGridItem()
        grid.setSize(self.M, self.N, 1)
        grid.setSpacing(1, 1, 1)
        # grid.setColor((0, 0, 0, 100)) 
        grid.translate(self.M/2 - 0.5, self.N/2 - 0.5, -1)
        self.addItem(grid)

    def update_view(self, active_blocks, completed_blocks):
        # Update persistent state
        for x, y, z in completed_blocks:
            idx = self.get_volume_index(x, y, z)
            if 0 <= idx < len(self.colors):
                self.colors[idx] = self.color_done

        # Prepare frame colors
        current_colors = self.colors.copy()
        
        active_A_indices_direct = []
        active_A_indices_transferred = []
        
        active_B_indices_direct = []
        active_B_indices_transferred = []
        
        active_C_indices = []
        
        for x, y, z in active_blocks:
            idx = self.get_volume_index(x, y, z)
            if 0 <= idx < len(current_colors):
                current_colors[idx] = self.color_active
            
            # Determine source type for A and B
            # A source logic:
            # If j % systolic_size == 0, direct. Else transferred.
            # Default to direct if systolic_size is None
            
            # A (Row flow): depends on column index y (which is j)
            if self.systolic_size and (y % self.systolic_size != 0):
                active_A_indices_transferred.append(self.get_A_index(x, z))
            else:
                active_A_indices_direct.append(self.get_A_index(x, z))
                
            # B (Col flow): depends on row index x (which is i)
            if self.systolic_size and (x % self.systolic_size != 0):
                active_B_indices_transferred.append(self.get_B_index(y, z))
            else:
                active_B_indices_direct.append(self.get_B_index(y, z))

            active_C_indices.append(self.get_C_index(x, y))
                
        self.scatter.setData(color=current_colors)
        
        # Update A Projection
        new_A = np.tile(self.base_color_A, (len(self.proj_A_pos), 1))
        
        if active_A_indices_transferred:
             idx_arr = np.array(active_A_indices_transferred)
             valid_mask = (idx_arr >= 0) & (idx_arr < len(new_A))
             new_A[idx_arr[valid_mask]] = self.transferred_color_A
             
        if active_A_indices_direct:
             idx_arr = np.array(active_A_indices_direct)
             valid_mask = (idx_arr >= 0) & (idx_arr < len(new_A))
             new_A[idx_arr[valid_mask]] = self.active_color_A
             
        self.scatter_A.setData(color=new_A)
        
        # Update B Projection
        new_B = np.tile(self.base_color_B, (len(self.proj_B_pos), 1))
        
        if active_B_indices_transferred:
             idx_arr = np.array(active_B_indices_transferred)
             valid_mask = (idx_arr >= 0) & (idx_arr < len(new_B))
             new_B[idx_arr[valid_mask]] = self.transferred_color_B
             
        if active_B_indices_direct:
             idx_arr = np.array(active_B_indices_direct)
             valid_mask = (idx_arr >= 0) & (idx_arr < len(new_B))
             new_B[idx_arr[valid_mask]] = self.active_color_B
             
        self.scatter_B.setData(color=new_B)

        # Update C Projection
        new_C = np.tile(self.base_color_C, (len(self.proj_C_pos), 1))
        if active_C_indices:
            idx_arr = np.array(active_C_indices)
            valid_mask = (idx_arr >= 0) & (idx_arr < len(new_C))
            valid_idx = idx_arr[valid_mask]
            if len(valid_idx) > 0:
                 new_C[valid_idx] = self.active_color_C
        self.scatter_C.setData(color=new_C)

    def reset_simulation(self):
        # Quick reset using tiling
        self.colors = np.tile(self.color_default, (len(self.positions), 1))
        self.scatter.setData(color=self.colors)
        self.scatter_A.setData(color=np.tile(self.base_color_A, (len(self.proj_A_pos), 1)))
        self.scatter_B.setData(color=np.tile(self.base_color_B, (len(self.proj_B_pos), 1)))
        self.scatter_C.setData(color=np.tile(self.base_color_C, (len(self.proj_C_pos), 1)))
