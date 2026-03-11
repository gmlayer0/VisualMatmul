import numpy as np
import pyqtgraph.opengl as gl
from PyQt6.QtGui import QColor, QVector3D
from PyQt6.QtCore import Qt


class ConvVisualizer3D(gl.GLViewWidget):
    """
    3D Visualizer for 2D Convolution on a dot product array.

    The 4D convolution space (OH, OW, KH, KW) is visualized using:
    - X = Output Width (OW) + Kernel Width (KW)
    - Y = Output Height (OH) + Kernel Height (KH)
    - Z = Time/Step or combined Kernel dimension

    Alternatively, we use a flattened 3D representation:
    - X = OW
    - Y = OH
    - Z = KW + KH * KW (kernel flattened)

    Or as two separate 2D grids with connecting lines.
    """

    def __init__(self, H, W, KH, KW, stride=1, padding=0, key_event_callback=None):
        super().__init__()
        self.H = H          # Input height
        self.W = W          # Input width
        self.KH = KH        # Kernel height
        self.KW = KW        # Kernel width
        self.stride = stride
        self.padding = padding

        # Output dimensions
        self.OH = (H + 2 * padding - KH) // stride + 1
        self.OW = (W + 2 * padding - KW) // stride + 1

        self.key_event_callback = key_event_callback
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()

        # Camera setup - isometric view
        self.setCameraPosition(elevation=30, azimuth=45)

        # View center
        max_dim = max(self.OW, self.OH, self.KW * self.KH)
        self.opts['center'] = QVector3D(self.OW / 2, self.OH / 2, (self.KH * self.KW) / 2)

        # Orthographic projection
        self.opts['fov'] = 1
        self.opts['distance'] = max_dim * 120

        self.setBackgroundColor('#ffffff')

        # Color palette (NVIDIA-inspired)
        self.color_default = np.array([0.1, 0.1, 0.1, 0.05])    # Very subtle gray
        self.color_active = np.array([0.46, 0.78, 0.0, 0.95])   # NVIDIA Green
        self.color_done = np.array([0.2, 0.5, 0.2, 0.6])        # Darker green

        # Input (red), Kernel (blue), Output (cyan)
        self.base_color_input = np.array([0.6, 0.2, 0.2, 1])     # Dim red
        self.base_color_kernel = np.array([0.2, 0.3, 0.6, 1])    # Dim blue
        self.base_color_output = np.array([0.2, 0.5, 0.5, 1])    # Dim cyan

        self.active_color_input = np.array([1.0, 0.3, 0.3, 1])   # Bright Red
        self.active_color_kernel = np.array([0.3, 0.5, 1.0, 1])  # Bright Blue
        self.active_color_output = np.array([0.0, 1.0, 0.8, 1])  # Bright Cyan

        # Transferred colors for systolic data flow
        self.transferred_input = np.array([0.7, 0.5, 0.2, 1])    # Orange
        self.transferred_kernel = np.array([0.2, 0.7, 0.8, 1])   # Teal

        self.cube_size = 0.65
        self.quad_size = 0.95

        # Setup all visual elements
        self.setup_volume_grid()
        self.setup_input_plane()
        self.setup_kernel_plane()
        self.setup_output_plane()

    def create_cube_mesh_data(self, positions, size):
        """Vectorized creation of cube mesh data."""
        n_cubes = len(positions)
        if n_cubes == 0:
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3).astype(np.uint32)

        s = size / 2

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

        template_faces = np.array([
            [0, 2, 1], [0, 3, 2],  # bottom (-z)
            [4, 5, 6], [4, 6, 7],  # top (+z)
            [0, 1, 5], [0, 5, 4],  # front (-y)
            [2, 3, 7], [2, 7, 6],  # back (+y)
            [0, 4, 7], [0, 7, 3],  # left (-x)
            [1, 2, 6], [1, 6, 5],  # right (+x)
        ], dtype=np.uint32)

        positions = np.asarray(positions, dtype=np.float32)
        all_verts = positions[:, np.newaxis, :] + template_verts[np.newaxis, :, :]
        all_verts = all_verts.reshape(-1, 3)

        offsets = np.arange(n_cubes, dtype=np.uint32)[:, np.newaxis, np.newaxis] * 8
        all_faces = template_faces[np.newaxis, :, :] + offsets
        all_faces = all_faces.reshape(-1, 3)

        return all_verts, all_faces

    def create_quad_mesh_data(self, positions, size, normal_axis='z'):
        """Create quads for projection planes."""
        n_quads = len(positions)
        if n_quads == 0:
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3).astype(np.uint32)

        s = size / 2

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

    def get_kernel_flat_index(self, kh, kw):
        """Convert 2D kernel position to flat Z coordinate."""
        return kh * self.KW + kw

    def get_volume_index(self, oh, ow, kh, kw):
        """Get linear index into volume grid."""
        k_flat = self.get_kernel_flat_index(kh, kw)
        return oh * (self.OW * self.KH * self.KW) + ow * (self.KH * self.KW) + k_flat

    def get_input_index(self, ih, iw):
        return ih * self.W + iw

    def get_kernel_index(self, kh, kw):
        return kh * self.KW + kw

    def get_output_index(self, oh, ow):
        return oh * self.OW + ow

    def setup_volume_grid(self):
        """
        Setup the main 3D volume grid representing the full convolution space.
        X = OW, Y = OH, Z = flattened kernel (kh*KW + kw)
        """
        positions = []
        for oh in range(self.OH):
            for ow in range(self.OW):
                for kh in range(self.KH):
                    for kw in range(self.KW):
                        k_flat = self.get_kernel_flat_index(kh, kw)
                        positions.append([ow, oh, k_flat])

        self.volume_positions = np.array(positions, dtype=np.float32)

        self.cube_verts, self.cube_faces = self.create_cube_mesh_data(
            self.volume_positions, self.cube_size
        )

        n_cubes = len(self.volume_positions)
        self.cube_face_colors = np.tile(self.color_default, (n_cubes * 12, 1))
        self.cube_colors_state = np.tile(self.color_default, (n_cubes, 1))

        mesh_data = gl.MeshData(vertexes=self.cube_verts, faces=self.cube_faces)
        mesh_data.setFaceColors(self.cube_face_colors)

        self.volume_mesh = gl.GLMeshItem(
            meshdata=mesh_data,
            smooth=False,
            drawFaces=True,
            drawEdges=True,
            edgeColor=(0.5, 0.5, 0.5, 0.2),
            shader='edgeHilight',
            glOptions='translucent'
        )
        self.addItem(self.volume_mesh)

    def setup_input_plane(self):
        """Setup Input plane visualization (left side)."""
        positions = []
        y_pos = -3.0
        for ih in range(self.H):
            for iw in range(self.W):
                positions.append([iw, y_pos, ih])  # x=iw, y=fixed, z=ih

        self.input_positions = np.array(positions, dtype=np.float32)

        self.input_verts, self.input_faces = self.create_quad_mesh_data(
            self.input_positions, self.quad_size, normal_axis='y'
        )

        n_quads = len(self.input_positions)
        self.input_face_colors = np.tile(self.base_color_input, (n_quads * 2, 1))

        mesh_data = gl.MeshData(vertexes=self.input_verts, faces=self.input_faces)
        mesh_data.setFaceColors(self.input_face_colors)

        self.input_mesh = gl.GLMeshItem(
            meshdata=mesh_data,
            smooth=False,
            drawEdges=True,
            edgeColor=(0.6, 0.3, 0.3, 0.6),
            shader=None,
            glOptions='translucent'
        )
        self.addItem(self.input_mesh)

    def setup_kernel_plane(self):
        """Setup Kernel plane visualization (back side)."""
        positions = []
        x_pos = -3.0
        for kh in range(self.KH):
            for kw in range(self.KW):
                positions.append([x_pos, kw, kh])

        self.kernel_positions = np.array(positions, dtype=np.float32)

        self.kernel_verts, self.kernel_faces = self.create_quad_mesh_data(
            self.kernel_positions, self.quad_size, normal_axis='x'
        )

        n_quads = len(self.kernel_positions)
        self.kernel_face_colors = np.tile(self.base_color_kernel, (n_quads * 2, 1))

        mesh_data = gl.MeshData(vertexes=self.kernel_verts, faces=self.kernel_faces)
        mesh_data.setFaceColors(self.kernel_face_colors)

        self.kernel_mesh = gl.GLMeshItem(
            meshdata=mesh_data,
            smooth=False,
            drawEdges=True,
            edgeColor=(0.3, 0.3, 0.6, 0.6),
            shader=None,
            glOptions='translucent'
        )
        self.addItem(self.kernel_mesh)

    def setup_output_plane(self):
        """Setup Output plane visualization (bottom/ceiling)."""
        positions = []
        z_pos = -3.0
        for oh in range(self.OH):
            for ow in range(self.OW):
                positions.append([ow, oh, z_pos])

        self.output_positions = np.array(positions, dtype=np.float32)

        self.output_verts, self.output_faces = self.create_quad_mesh_data(
            self.output_positions, self.quad_size, normal_axis='z'
        )

        n_quads = len(self.output_positions)
        self.output_face_colors = np.tile(self.base_color_output, (n_quads * 2, 1))

        mesh_data = gl.MeshData(vertexes=self.output_verts, faces=self.output_faces)
        mesh_data.setFaceColors(self.output_face_colors)

        self.output_mesh = gl.GLMeshItem(
            meshdata=mesh_data,
            smooth=False,
            drawEdges=True,
            edgeColor=(0.3, 0.5, 0.5, 0.6),
            shader=None,
            glOptions='translucent'
        )
        self.addItem(self.output_mesh)

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

    def update_view(self, step):
        """
        Update visualization based on ConvIterationStep.
        """
        active_blocks = step.active
        completed_blocks = step.completed
        active_input = step.active_input
        active_kernel = step.active_kernel
        active_output = step.active_output

        # Update persistent state for completed blocks
        for oh, ow, kh, kw in completed_blocks:
            idx = self.get_volume_index(oh, ow, kh, kw)
            if 0 <= idx < len(self.cube_colors_state):
                self.cube_colors_state[idx] = self.color_done

        # Prepare current frame colors
        current_colors = self.cube_colors_state.copy()

        active_input_indices = []
        active_kernel_indices = []
        active_output_indices = []

        for oh, ow, kh, kw in active_blocks:
            idx = self.get_volume_index(oh, ow, kh, kw)
            if 0 <= idx < len(current_colors):
                current_colors[idx] = self.color_active

        # Update cube colors
        cube_face_colors = np.repeat(current_colors, 12, axis=0)
        self.update_cube_colors(cube_face_colors)

        # Update Input plane
        n_quads_input = len(self.input_positions)
        new_input_colors = np.tile(self.base_color_input, (n_quads_input, 1))

        for ih, iw in active_input:
            idx = self.get_input_index(ih, iw)
            if 0 <= idx < n_quads_input:
                new_input_colors[idx] = self.active_color_input

        input_face_colors = np.repeat(new_input_colors, 2, axis=0)
        self.update_mesh_colors(self.input_mesh, self.input_verts, self.input_faces, input_face_colors)

        # Update Kernel plane
        n_quads_kernel = len(self.kernel_positions)
        new_kernel_colors = np.tile(self.base_color_kernel, (n_quads_kernel, 1))

        for kh, kw in active_kernel:
            idx = self.get_kernel_index(kh, kw)
            if 0 <= idx < n_quads_kernel:
                new_kernel_colors[idx] = self.active_color_kernel

        kernel_face_colors = np.repeat(new_kernel_colors, 2, axis=0)
        self.update_mesh_colors(self.kernel_mesh, self.kernel_verts, self.kernel_faces, kernel_face_colors)

        # Update Output plane
        n_quads_output = len(self.output_positions)
        new_output_colors = np.tile(self.base_color_output, (n_quads_output, 1))

        for oh, ow in active_output:
            idx = self.get_output_index(oh, ow)
            if 0 <= idx < n_quads_output:
                new_output_colors[idx] = self.active_color_output

        output_face_colors = np.repeat(new_output_colors, 2, axis=0)
        self.update_mesh_colors(self.output_mesh, self.output_verts, self.output_faces, output_face_colors)

    def reset_simulation(self):
        """Reset all visualization colors."""
        n_cubes = len(self.volume_positions)
        self.cube_colors_state = np.tile(self.color_default, (n_cubes, 1))

        cube_face_colors = np.tile(self.color_default, (n_cubes * 12, 1))
        self.update_cube_colors(cube_face_colors)

        n_quads_input = len(self.input_positions)
        input_face_colors = np.tile(self.base_color_input, (n_quads_input * 2, 1))
        self.update_mesh_colors(self.input_mesh, self.input_verts, self.input_faces, input_face_colors)

        n_quads_kernel = len(self.kernel_positions)
        kernel_face_colors = np.tile(self.base_color_kernel, (n_quads_kernel * 2, 1))
        self.update_mesh_colors(self.kernel_mesh, self.kernel_verts, self.kernel_faces, kernel_face_colors)

        n_quads_output = len(self.output_positions)
        output_face_colors = np.tile(self.base_color_output, (n_quads_output * 2, 1))
        self.update_mesh_colors(self.output_mesh, self.output_verts, self.output_faces, output_face_colors)

    def keyPressEvent(self, event):
        if self.key_event_callback:
            self.key_event_callback(event)
        else:
            super().keyPressEvent(event)
