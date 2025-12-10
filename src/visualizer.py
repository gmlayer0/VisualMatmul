import numpy as np
import pyqtgraph.opengl as gl
from PyQt6.QtGui import QColor, QVector3D
from PyQt6.QtCore import Qt

class Visualizer3D(gl.GLViewWidget):
    def __init__(self, M, N, K):
        super().__init__()
        self.M = M
        self.N = N
        self.K = K
        
        self.setCameraPosition(distance=max(M, N, K) * 3, elevation=30, azimuth=45)
        self.setBackgroundColor('k') # Black background

        # Colors
        self.color_default = (0.2, 0.2, 0.2, 0.1) # Translucent gray
        self.color_active = (1.0, 1.0, 0.0, 0.8)  # Bright Yellow
        self.color_done = (0.0, 1.0, 0.0, 0.6)    # Green
        
        # Projection Colors
        self.color_proj_A = (1.0, 0.0, 0.0, 0.5)  # Red for A (x-z)
        self.color_proj_B = (0.0, 0.0, 1.0, 0.5)  # Blue for B (y-z)
        
        # Data storage for the main 3D volume
        # We will use GLScatterPlotItem with square points as an approximation for voxels 
        # or GLMeshItem if we want true cubes. For speed and simplicity in Python:
        # GLScatterPlotItem with pxMode=False draws squares in 3D space which look like cubes from most angles if flat,
        # but true cubes need GLMeshItem. Let's try GLScatterPlotItem first for performance, 
        # or construct a single Mesh for all cubes.
        
        # Let's use a single GLMeshItem for all cubes to allow distinct coloring.
        # Actually, iterating over individual GLBoxItems is slow.
        # ScatterPlot is fastest. Let's use ScatterPlot with 's' (square) symbol.
        
        self.setup_volume_grid()
        self.setup_projections()
        self.setup_floor()
        
    def setup_volume_grid(self):
        # Generate coordinates
        self.positions = []
        self.colors = []
        self.indices_map = {} # (x,y,z) -> index in arrays
        
        idx = 0
        for x in range(self.M):
            for y in range(self.N):
                for z in range(self.K):
                    self.positions.append((x, y, z))
                    self.colors.append(self.color_default)
                    self.indices_map[(x, y, z)] = idx
                    idx += 1
                    
        self.positions = np.array(self.positions)
        self.colors = np.array(self.colors)
        
        self.scatter = gl.GLScatterPlotItem(
            pos=self.positions,
            color=self.colors,
            size=0.8, # Size relative to unit grid
            pxMode=False
        )
        self.scatter.setGLOptions('translucent')
        self.addItem(self.scatter)

    def setup_projections(self):
        # We need to visualize A on x-z plane (y = -1 or y = N)
        # and B on y-z plane (x = -1 or x = M)
        
        # A Projection (x-z plane), placed at y = -2
        self.proj_A_pos = []
        self.proj_A_colors = []
        self.proj_A_map = {}
        idx = 0
        for x in range(self.M):
            for z in range(self.K):
                # Place A visual at y = -2
                self.proj_A_pos.append((x, -2, z))
                self.proj_A_colors.append((0.5, 0.0, 0.0, 0.2))
                self.proj_A_map[(x, z)] = idx
                idx += 1
        
        self.scatter_A = gl.GLScatterPlotItem(
            pos=np.array(self.proj_A_pos),
            color=np.array(self.proj_A_colors),
            size=0.8,
            pxMode=False
        )
        self.addItem(self.scatter_A)
        
        # B Projection (y-z plane), placed at x = -2
        self.proj_B_pos = []
        self.proj_B_colors = []
        self.proj_B_map = {}
        idx = 0
        for y in range(self.N):
            for z in range(self.K):
                # Place B visual at x = -2
                self.proj_B_pos.append((-2, y, z))
                self.proj_B_colors.append((0.0, 0.0, 0.5, 0.2))
                self.proj_B_map[(y, z)] = idx
                idx += 1
                
        self.scatter_B = gl.GLScatterPlotItem(
            pos=np.array(self.proj_B_pos),
            color=np.array(self.proj_B_colors),
            size=0.8,
            pxMode=False
        )
        self.addItem(self.scatter_B)

    def setup_floor(self):
        # C Projection (x-y plane), placed at z = -1
        # This represents the output accumulator
        grid = gl.GLGridItem()
        grid.setSize(self.M, self.N, 1)
        grid.setSpacing(1, 1, 1)
        # Center the grid
        grid.translate(self.M/2 - 0.5, self.N/2 - 0.5, -1)
        self.addItem(grid)

    def update_view(self, active_blocks, completed_blocks):
        # Reset ephemeral active colors to default or done state is complex without state tracking.
        # Simplification: 
        # 1. Maintain a persistent state array for colors.
        # 2. 'Completed' blocks turn Green permanently.
        # 3. 'Active' blocks are Yellow just for this frame (revert to previous state if not done? No, usually active becomes done).
        # Actually, user wants: "Unupdated"(Default), "Updating"(Active), "Updated"(Done).
        
        # Update main volume
        for x, y, z in completed_blocks:
            idx = self.indices_map.get((x, y, z))
            if idx is not None:
                self.colors[idx] = self.color_done
                
        # Create a temporary color buffer for rendering this frame
        current_colors = self.colors.copy()
        
        # Track active A and B elements for this frame
        active_A_indices = set()
        active_B_indices = set()
        
        for x, y, z in active_blocks:
            idx = self.indices_map.get((x, y, z))
            if idx is not None:
                current_colors[idx] = self.color_active
            
            # Identify projections
            if (x, z) in self.proj_A_map:
                active_A_indices.add(self.proj_A_map[(x, z)])
            if (y, z) in self.proj_B_map:
                active_B_indices.add(self.proj_B_map[(y, z)])
                
        self.scatter.setData(color=current_colors)
        
        # Update Projections
        # Reset projections to base color
        new_A_colors = np.array([(0.5, 0.0, 0.0, 0.2)] * len(self.proj_A_pos))
        new_B_colors = np.array([(0.0, 0.0, 0.5, 0.2)] * len(self.proj_B_pos))
        
        # Highlight active
        for idx in active_A_indices:
            new_A_colors[idx] = (1.0, 0.2, 0.2, 0.9) # Bright Red
            
        for idx in active_B_indices:
            new_B_colors[idx] = (0.2, 0.2, 1.0, 0.9) # Bright Blue
            
        self.scatter_A.setData(color=new_A_colors)
        self.scatter_B.setData(color=new_B_colors)

    def reset_simulation(self):
        self.colors = np.array([self.color_default] * len(self.positions))
        self.scatter.setData(color=self.colors)
        self.scatter_A.setData(color=np.array([(0.5, 0.0, 0.0, 0.2)] * len(self.proj_A_pos)))
        self.scatter_B.setData(color=np.array([(0.0, 0.0, 0.5, 0.2)] * len(self.proj_B_pos)))

