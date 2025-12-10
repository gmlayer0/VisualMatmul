import numpy as np
import pyqtgraph.opengl as gl
from PyQt6.QtGui import QColor, QVector3D
from PyQt6.QtCore import Qt

class Visualizer3D(gl.GLViewWidget):
    def __init__(self, M, N, K, key_event_callback=None, systolic_size=None):
        super().__init__()
        self.M = M
        self.N = N
        self.K = K
        self.systolic_size = systolic_size
        self.key_event_callback = key_event_callback
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()
        
        # NVIDIA-style camera angle (isometric, looking from front-right-top)
        # Clear view of A (left wall), B (back wall), C (top), and the 3D volume
        self.setCameraPosition(elevation=30, azimuth=45)
        
        # Set view center to the center of the cube
        self.opts['center'] = QVector3D(M / 2 - 0.5, N / 2 - 0.5, K / 2 - 0.75)
        
        # Orthographic projection for cleaner look
        self.opts['fov'] = 1
        self.opts['distance'] = max(M, N, K) * 210
        
        self.setBackgroundColor('#ffffff') # 白色背景

        # Colors (NVIDIA-inspired palette)
        self.color_default = np.array([0.1, 0.1, 0.1, 0.05])    # Very subtle gray
        self.color_active = np.array([0.46, 0.78, 0.0, 0.95])   # NVIDIA Green
        self.color_done = np.array([0.2, 0.5, 0.2, 0.6])        # Darker green
        
        # Projection Colors
        self.base_color_A = np.array([0.6, 0.2, 0.2, 1])    # Dim red
        self.base_color_B = np.array([0.2, 0.3, 0.6, 1])    # Dim blue
        self.base_color_C = np.array([0.2, 0.5, 0.5, 1])    # Dim cyan
        
        # Highlight Colors
        self.active_color_A = np.array([1.0, 0.3, 0.3, 1])  # Bright Red (Direct)
        self.active_color_B = np.array([0.3, 0.5, 1.0, 1])  # Bright Blue (Direct)
        
        # Transferred colors
        self.transferred_color_A = np.array([0.7, 0.5, 0.2, 1])  # Orange
        self.transferred_color_B = np.array([0.2, 0.7, 0.8, 1])  # Teal
        
        self.active_color_C = np.array([0.0, 1.0, 0.8, 1])  # Bright Cyan
        
        # Cube size
        self.cube_size = 0.65
        self.quad_size = 0.95
        
        self.setup_volume_grid()
        self.setup_projections()
        self.setup_ceiling_C()
        
    def create_cube_mesh_data(self, positions, size):
        """
        Vectorized creation of cube mesh data.
        positions: Nx3 array of cube centers
        Returns: vertices, faces, and initial face colors
        """
        n_cubes = len(positions)
        if n_cubes == 0:
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3).astype(np.uint32), np.array([]).reshape(0, 4)
        
        s = size / 2
        
        # Template cube vertices (8 vertices per cube)
        template_verts = np.array([
            [-s, -s, -s],
            [+s, -s, -s],
            [+s, +s, -s],
            [-s, +s, -s],
            [-s, -s, +s],
            [+s, -s, +s],
            [+s, +s, +s],
            [-s, +s, +s],
        ], dtype=np.float32)
        
        # Template faces (12 triangles per cube, 2 per face)
        template_faces = np.array([
            [0, 2, 1], [0, 3, 2],  # bottom (-z)
            [4, 5, 6], [4, 6, 7],  # top (+z)
            [0, 1, 5], [0, 5, 4],  # front (-y)
            [2, 3, 7], [2, 7, 6],  # back (+y)
            [0, 4, 7], [0, 7, 3],  # left (-x)
            [1, 2, 6], [1, 6, 5],  # right (+x)
        ], dtype=np.uint32)
        
        # Broadcast positions to all 8 vertices of each cube
        positions = np.asarray(positions, dtype=np.float32)
        all_verts = positions[:, np.newaxis, :] + template_verts[np.newaxis, :, :]
        all_verts = all_verts.reshape(-1, 3)
        
        # Create face indices with offsets
        offsets = np.arange(n_cubes, dtype=np.uint32)[:, np.newaxis, np.newaxis] * 8
        all_faces = template_faces[np.newaxis, :, :] + offsets
        all_faces = all_faces.reshape(-1, 3)
        
        return all_verts, all_faces

    def create_quad_mesh_data(self, positions, size, normal_axis='z'):
        """
        Create quads (2 triangles each) for projection planes.
        positions: Nx3 array of quad centers
        normal_axis: which axis is perpendicular to the quad ('x', 'y', or 'z')
        """
        n_quads = len(positions)
        if n_quads == 0:
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3).astype(np.uint32)
        
        s = size / 2
        
        # Template quad vertices based on normal axis
        if normal_axis == 'z':
            template_verts = np.array([
                [-s, -s, 0],
                [+s, -s, 0],
                [+s, +s, 0],
                [-s, +s, 0],
            ], dtype=np.float32)
        elif normal_axis == 'y':
            template_verts = np.array([
                [-s, 0, -s],
                [+s, 0, -s],
                [+s, 0, +s],
                [-s, 0, +s],
            ], dtype=np.float32)
        else:  # x
            template_verts = np.array([
                [0, -s, -s],
                [0, +s, -s],
                [0, +s, +s],
                [0, -s, +s],
            ], dtype=np.float32)
        
        # Template faces (2 triangles per quad)
        template_faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
        ], dtype=np.uint32)
        
        positions = np.asarray(positions, dtype=np.float32)
        all_verts = positions[:, np.newaxis, :] + template_verts[np.newaxis, :, :]
        all_verts = all_verts.reshape(-1, 3)
        
        offsets = np.arange(n_quads, dtype=np.uint32)[:, np.newaxis, np.newaxis] * 4
        all_faces = template_faces[np.newaxis, :, :] + offsets
        all_faces = all_faces.reshape(-1, 3)
        
        return all_verts, all_faces

    def setup_volume_grid(self):
        # Create coordinate grids
        grid = np.indices((self.M, self.N, self.K))
        self.positions = grid.reshape(3, -1).T.astype(np.float32)
        
        # Create cube mesh
        self.cube_verts, self.cube_faces = self.create_cube_mesh_data(self.positions, self.cube_size)
        
        # Initial face colors (12 faces per cube, each face has same color)
        n_cubes = len(self.positions)
        self.cube_face_colors = np.tile(self.color_default, (n_cubes * 12, 1))
        
        # Store persistent color state per cube
        self.cube_colors_state = np.tile(self.color_default, (n_cubes, 1))
        
        # Create mesh item
        mesh_data = gl.MeshData(vertexes=self.cube_verts, faces=self.cube_faces)
        mesh_data.setFaceColors(self.cube_face_colors)
        
        self.volume_mesh = gl.GLMeshItem(
            meshdata=mesh_data,
            smooth=False,
            drawFaces=True,    # 是否绘制面 (默认True)
            drawEdges=True,    # 是否绘制边 (默认False)
            edgeColor=(0.5, 0.5, 0.5, 0.2),  # 边的颜色
            shader='edgeHilight',   # 着色器类型，可选 'shaded', 'normalColor', 'edgeHilight', 等
            glOptions='translucent'  # 透明度相关, 也可用 'opaque', 'additive' 等
        )
        self.addItem(self.volume_mesh)

    def get_volume_index(self, x, y, z):
        return x * (self.N * self.K) + y * self.K + z

    def setup_projections(self):
        # A Projection: x-z plane, y fixed at -3 (further from cube)
        grid_A = np.indices((self.M, self.K))
        x_A = grid_A[0].ravel().astype(np.float32)
        z_A = grid_A[1].ravel().astype(np.float32)
        y_A = np.full_like(x_A, -3)
        self.proj_A_pos = np.stack((x_A, y_A, z_A), axis=1)
        
        self.proj_A_verts, self.proj_A_faces = self.create_quad_mesh_data(
            self.proj_A_pos, self.quad_size, normal_axis='y'
        )
        
        n_quads_A = len(self.proj_A_pos)
        self.proj_A_face_colors = np.tile(self.base_color_A, (n_quads_A * 2, 1))
        
        mesh_data_A = gl.MeshData(vertexes=self.proj_A_verts, faces=self.proj_A_faces)
        mesh_data_A.setFaceColors(self.proj_A_face_colors)
        
        self.mesh_A = gl.GLMeshItem(
            meshdata=mesh_data_A,
            smooth=False,
            drawEdges=True,
            edgeColor=(0.6, 0.3, 0.3, 0.6),
            shader=None,
            glOptions='translucent'
        )
        self.addItem(self.mesh_A)
        
        # B Projection: y-z plane, x fixed at -3 (further from cube)
        grid_B = np.indices((self.N, self.K))
        y_B = grid_B[0].ravel().astype(np.float32)
        z_B = grid_B[1].ravel().astype(np.float32)
        x_B = np.full_like(y_B, -3)
        self.proj_B_pos = np.stack((x_B, y_B, z_B), axis=1)
        
        self.proj_B_verts, self.proj_B_faces = self.create_quad_mesh_data(
            self.proj_B_pos, self.quad_size, normal_axis='x'
        )
        
        n_quads_B = len(self.proj_B_pos)
        self.proj_B_face_colors = np.tile(self.base_color_B, (n_quads_B * 2, 1))
        
        mesh_data_B = gl.MeshData(vertexes=self.proj_B_verts, faces=self.proj_B_faces)
        mesh_data_B.setFaceColors(self.proj_B_face_colors)
        
        self.mesh_B = gl.GLMeshItem(
            meshdata=mesh_data_B,
            smooth=False,
            drawEdges=True,
            edgeColor=(0.3, 0.3, 0.6, 0.6),
            shader=None,
            glOptions='translucent'
        )
        self.addItem(self.mesh_B)
    
    def setup_ceiling_C(self):
        # C Projection: x-y plane, z fixed at -3 (below the cube, further away)
        grid_C = np.indices((self.M, self.N))
        x_C = grid_C[0].ravel().astype(np.float32)
        y_C = grid_C[1].ravel().astype(np.float32)
        z_C = np.full_like(x_C, -3)
        self.proj_C_pos = np.stack((x_C, y_C, z_C), axis=1)
        
        self.proj_C_verts, self.proj_C_faces = self.create_quad_mesh_data(
            self.proj_C_pos, self.quad_size, normal_axis='z'
        )
        
        n_quads_C = len(self.proj_C_pos)
        self.proj_C_face_colors = np.tile(self.base_color_C, (n_quads_C * 2, 1))
        
        mesh_data_C = gl.MeshData(vertexes=self.proj_C_verts, faces=self.proj_C_faces)
        mesh_data_C.setFaceColors(self.proj_C_face_colors)
        
        self.mesh_C = gl.GLMeshItem(
            meshdata=mesh_data_C,
            smooth=False,
            drawEdges=True,
            edgeColor=(0.3, 0.5, 0.5, 0.6),
            shader=None,
            glOptions='translucent'
        )
        self.addItem(self.mesh_C)

    def get_A_index(self, x, z):
        return x * self.K + z
        
    def get_B_index(self, y, z):
        return y * self.K + z

    def get_C_index(self, x, y):
        return x * self.N + y

    def update_cube_colors(self, face_colors):
        """Update the volume mesh with new face colors."""
        mesh_data = gl.MeshData(vertexes=self.cube_verts, faces=self.cube_faces)
        mesh_data.setFaceColors(face_colors)
        self.volume_mesh.setMeshData(meshdata=mesh_data)

    def update_mesh_colors(self, mesh_item, verts, faces, face_colors):
        """Update a mesh item with new face colors."""
        mesh_data = gl.MeshData(vertexes=verts, faces=faces)
        mesh_data.setFaceColors(face_colors)
        mesh_item.setMeshData(meshdata=mesh_data)

    def update_view(self, active_blocks, completed_blocks):
        # Update persistent state for completed blocks
        for x, y, z in completed_blocks:
            idx = self.get_volume_index(x, y, z)
            if 0 <= idx < len(self.cube_colors_state):
                self.cube_colors_state[idx] = self.color_done

        # Prepare current frame colors (copy persistent state)
        current_colors = self.cube_colors_state.copy()
        
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
            if self.systolic_size and (y % self.systolic_size != 0):
                active_A_indices_transferred.append(self.get_A_index(x, z))
            else:
                active_A_indices_direct.append(self.get_A_index(x, z))
                
            if self.systolic_size and (x % self.systolic_size != 0):
                active_B_indices_transferred.append(self.get_B_index(y, z))
            else:
                active_B_indices_direct.append(self.get_B_index(y, z))

            active_C_indices.append(self.get_C_index(x, y))
        
        # Expand cube colors to face colors (12 faces per cube)
        cube_face_colors = np.repeat(current_colors, 12, axis=0)
        self.update_cube_colors(cube_face_colors)
        
        # Update A Projection
        n_quads_A = len(self.proj_A_pos)
        new_A_colors = np.tile(self.base_color_A, (n_quads_A, 1))
        
        if active_A_indices_transferred:
            idx_arr = np.array(active_A_indices_transferred)
            valid_mask = (idx_arr >= 0) & (idx_arr < n_quads_A)
            new_A_colors[idx_arr[valid_mask]] = self.transferred_color_A
             
        if active_A_indices_direct:
            idx_arr = np.array(active_A_indices_direct)
            valid_mask = (idx_arr >= 0) & (idx_arr < n_quads_A)
            new_A_colors[idx_arr[valid_mask]] = self.active_color_A
        
        A_face_colors = np.repeat(new_A_colors, 2, axis=0)  # 2 triangles per quad
        self.update_mesh_colors(self.mesh_A, self.proj_A_verts, self.proj_A_faces, A_face_colors)
        
        # Update B Projection
        n_quads_B = len(self.proj_B_pos)
        new_B_colors = np.tile(self.base_color_B, (n_quads_B, 1))
        
        if active_B_indices_transferred:
            idx_arr = np.array(active_B_indices_transferred)
            valid_mask = (idx_arr >= 0) & (idx_arr < n_quads_B)
            new_B_colors[idx_arr[valid_mask]] = self.transferred_color_B
             
        if active_B_indices_direct:
            idx_arr = np.array(active_B_indices_direct)
            valid_mask = (idx_arr >= 0) & (idx_arr < n_quads_B)
            new_B_colors[idx_arr[valid_mask]] = self.active_color_B
        
        B_face_colors = np.repeat(new_B_colors, 2, axis=0)
        self.update_mesh_colors(self.mesh_B, self.proj_B_verts, self.proj_B_faces, B_face_colors)

        # Update C Projection
        n_quads_C = len(self.proj_C_pos)
        new_C_colors = np.tile(self.base_color_C, (n_quads_C, 1))
        
        if active_C_indices:
            idx_arr = np.array(active_C_indices)
            valid_mask = (idx_arr >= 0) & (idx_arr < n_quads_C)
            new_C_colors[idx_arr[valid_mask]] = self.active_color_C
        
        C_face_colors = np.repeat(new_C_colors, 2, axis=0)
        self.update_mesh_colors(self.mesh_C, self.proj_C_verts, self.proj_C_faces, C_face_colors)

    def reset_simulation(self):
        # Reset persistent color state
        n_cubes = len(self.positions)
        self.cube_colors_state = np.tile(self.color_default, (n_cubes, 1))
        
        # Reset cube mesh colors
        cube_face_colors = np.tile(self.color_default, (n_cubes * 12, 1))
        self.update_cube_colors(cube_face_colors)
        
        # Reset projection colors
        n_quads_A = len(self.proj_A_pos)
        A_face_colors = np.tile(self.base_color_A, (n_quads_A * 2, 1))
        self.update_mesh_colors(self.mesh_A, self.proj_A_verts, self.proj_A_faces, A_face_colors)
        
        n_quads_B = len(self.proj_B_pos)
        B_face_colors = np.tile(self.base_color_B, (n_quads_B * 2, 1))
        self.update_mesh_colors(self.mesh_B, self.proj_B_verts, self.proj_B_faces, B_face_colors)
        
        n_quads_C = len(self.proj_C_pos)
        C_face_colors = np.tile(self.base_color_C, (n_quads_C * 2, 1))
        self.update_mesh_colors(self.mesh_C, self.proj_C_verts, self.proj_C_faces, C_face_colors)

    def keyPressEvent(self, event):
        if self.key_event_callback:
            self.key_event_callback(event)
        else:
            super().keyPressEvent(event)
