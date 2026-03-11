import numpy as np
import pyqtgraph.opengl as gl
from PyQt6.QtGui import QVector3D
from PyQt6.QtCore import Qt


class ConvVisualizer3D(gl.GLViewWidget):
    """
    3D Visualizer for 2D Convolution on Tensor Core.

    Layout:
        Input Volume    Tensor Core MACs    Output Volume
          [H,W,Ci]  →     [M,N,K]     →    [Oh,Ow,Co]

                        Kernel Volume
                        [Kh,Kw,Ci,Co]
    """

    def __init__(self, H, W, C_in, C_out, KH=3, KW=3, stride=1, padding=0,
                 tc_m=16, tc_n=16, tc_k=16, key_event_callback=None):
        super().__init__()
        self.H = H
        self.W = W
        self.C_in = C_in
        self.C_out = C_out
        self.KH = KH
        self.KW = KW
        self.stride = stride
        self.padding = padding

        self.OH = (H + 2 * padding - KH) // stride + 1
        self.OW = (W + 2 * padding - KW) // stride + 1

        # Tensor Core dimensions (keep small for visualization)
        self.TC_M = min(tc_m, 16)
        self.TC_N = min(tc_n, 4)
        self.TC_K = min(tc_k, 16)

        self.key_event_callback = key_event_callback
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()

        # Camera setup
        self.setCameraPosition(elevation=25, azimuth=40)

        # Calculate layout spacing
        input_width = C_in
        tc_width = self.TC_N
        output_width = C_out
        total_width = input_width + tc_width + output_width + 10

        center_x = total_width / 2
        center_y = max(H, self.OH, self.TC_M, KH * KW) / 2
        center_z = max(W, self.OW, self.TC_K, max(C_in, C_out)) / 2
        self.opts['center'] = QVector3D(center_x, center_y, center_z)

        # Orthographic projection
        self.opts['fov'] = 1
        max_dim = max(H, W, C_in, C_out, self.OH, self.OW, self.TC_M, self.TC_K)
        self.opts['distance'] = max_dim * 200

        self.setBackgroundColor('#ffffff')

        # Color palette (NVIDIA-inspired)
        self.color_default = np.array([0.12, 0.12, 0.12, 0.1])
        self.color_active = np.array([0.46, 0.78, 0.0, 0.95])    # NVIDIA Green
        self.color_done = np.array([0.2, 0.5, 0.2, 0.7])

        # Volume colors - dim base, bright active for contrast
        self.base_color_input = np.array([0.35, 0.15, 0.15, 0.4])    # Dim Red
        self.active_color_input = np.array([1.0, 0.1, 0.1, 1.0])    # Bright Red

        self.base_color_kernel = np.array([0.15, 0.2, 0.45, 0.4])    # Dim Blue
        self.active_color_kernel = np.array([0.1, 0.4, 1.0, 1.0])   # Bright Blue

        self.base_color_output = np.array([0.15, 0.4, 0.4, 0.4])    # Dim Cyan
        self.active_color_output = np.array([0.0, 1.0, 0.9, 1.0])   # Bright Cyan

        self.cube_size = 0.75

        # Setup all visual elements
        self.setup_input_volume()
        self.setup_kernel_volume()
        self.setup_output_volume()
        self.setup_tensor_core_array()

    def create_cube_mesh_data(self, positions, size):
        """Vectorized creation of cube mesh data."""
        n_cubes = len(positions)
        if n_cubes == 0:
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3).astype(np.uint32)

        s = size / 2

        template_verts = np.array([
            [-s, -s, -s], [+s, -s, -s], [+s, +s, -s], [-s, +s, -s],
            [-s, -s, +s], [+s, -s, +s], [+s, +s, +s], [-s, +s, +s],
        ], dtype=np.float32)

        template_faces = np.array([
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
            [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
        ], dtype=np.uint32)

        positions = np.asarray(positions, dtype=np.float32)
        all_verts = positions[:, np.newaxis, :] + template_verts[np.newaxis, :, :]
        all_verts = all_verts.reshape(-1, 3)

        offsets = np.arange(n_cubes, dtype=np.uint32)[:, np.newaxis, np.newaxis] * 8
        all_faces = template_faces[np.newaxis, :, :] + offsets
        all_faces = all_faces.reshape(-1, 3)

        return all_verts, all_faces

    # ========================================================================
    # Input Volume Setup
    # ========================================================================

    def get_input_index(self, h, w, c):
        return h * self.W * self.C_in + w * self.C_in + c

    def setup_input_volume(self):
        """Input Volume: positioned on the left"""
        positions = []
        self.input_coord_map = {}  # (h,w,c) -> index
        idx = 0
        for h in range(self.H):
            for w in range(self.W):
                for c in range(self.C_in):
                    # Layout: x=channel, y=height, z=width
                    x = c
                    y = h
                    z = w
                    positions.append([x, y, z])
                    self.input_coord_map[(h, w, c)] = idx
                    idx += 1

        self.input_positions = np.array(positions, dtype=np.float32)

        self.input_verts, self.input_faces = self.create_cube_mesh_data(
            self.input_positions, self.cube_size
        )

        n_cubes = len(self.input_positions)
        self.input_colors_state = np.tile(self.base_color_input, (n_cubes, 1))
        input_face_colors = np.tile(self.base_color_input, (n_cubes * 12, 1))

        mesh_data = gl.MeshData(vertexes=self.input_verts, faces=self.input_faces)
        mesh_data.setFaceColors(input_face_colors)

        self.input_mesh = gl.GLMeshItem(
            meshdata=mesh_data,
            smooth=False,
            drawFaces=True,
            drawEdges=True,
            edgeColor=(0.6, 0.3, 0.3, 0.3),
            shader='edgeHilight',
            glOptions='translucent'
        )
        self.addItem(self.input_mesh)

    # ========================================================================
    # Kernel Volume Setup
    # ========================================================================

    def get_kernel_index(self, kh, kw, ci, co):
        return kh * self.KW * self.C_in * self.C_out + kw * self.C_in * self.C_out + ci * self.C_out + co

    def setup_kernel_volume(self):
        """Kernel Volume: positioned below everything"""
        positions = []
        self.kernel_coord_map = {}
        idx = 0
        y_base = -max(self.H, self.OH, self.TC_M) - 5
        x_offset = self.C_in + 3

        for kh in range(self.KH):
            for kw in range(self.KW):
                for ci in range(self.C_in):
                    for co in range(self.C_out):
                        x = x_offset + co
                        y = y_base + kh
                        z = kw * (self.C_in + 1) + ci
                        positions.append([x, y, z])
                        self.kernel_coord_map[(kh, kw, ci, co)] = idx
                        idx += 1

        self.kernel_positions = np.array(positions, dtype=np.float32)

        self.kernel_verts, self.kernel_faces = self.create_cube_mesh_data(
            self.kernel_positions, self.cube_size * 0.8
        )

        n_cubes = len(self.kernel_positions)
        self.kernel_colors_state = np.tile(self.base_color_kernel, (n_cubes, 1))
        kernel_face_colors = np.tile(self.base_color_kernel, (n_cubes * 12, 1))

        mesh_data = gl.MeshData(vertexes=self.kernel_verts, faces=self.kernel_faces)
        mesh_data.setFaceColors(kernel_face_colors)

        self.kernel_mesh = gl.GLMeshItem(
            meshdata=mesh_data,
            smooth=False,
            drawFaces=True,
            drawEdges=True,
            edgeColor=(0.3, 0.3, 0.6, 0.3),
            shader='edgeHilight',
            glOptions='translucent'
        )
        self.addItem(self.kernel_mesh)

    # ========================================================================
    # Output Volume Setup
    # ========================================================================

    def get_output_index(self, oh, ow, co):
        return oh * self.OW * self.C_out + ow * self.C_out + co

    def setup_output_volume(self):
        """Output Volume: positioned on the right"""
        positions = []
        self.output_coord_map = {}
        idx = 0
        x_offset = self.C_in + self.TC_N + 6

        for oh in range(self.OH):
            for ow in range(self.OW):
                for co in range(self.C_out):
                    x = x_offset + co
                    y = oh
                    z = ow
                    positions.append([x, y, z])
                    self.output_coord_map[(oh, ow, co)] = idx
                    idx += 1

        self.output_positions = np.array(positions, dtype=np.float32)

        self.output_verts, self.output_faces = self.create_cube_mesh_data(
            self.output_positions, self.cube_size
        )

        n_cubes = len(self.output_positions)
        self.output_colors_state = np.tile(self.base_color_output, (n_cubes, 1))
        output_face_colors = np.tile(self.base_color_output, (n_cubes * 12, 1))

        mesh_data = gl.MeshData(vertexes=self.output_verts, faces=self.output_faces)
        mesh_data.setFaceColors(output_face_colors)

        self.output_mesh = gl.GLMeshItem(
            meshdata=mesh_data,
            smooth=False,
            drawFaces=True,
            drawEdges=True,
            edgeColor=(0.3, 0.5, 0.5, 0.3),
            shader='edgeHilight',
            glOptions='translucent'
        )
        self.addItem(self.output_mesh)

    # ========================================================================
    # Tensor Core MAC Array Setup
    # ========================================================================

    def get_tc_index(self, m, n, k):
        return m * self.TC_N * self.TC_K + n * self.TC_K + k

    def setup_tensor_core_array(self):
        """Tensor Core MAC Array in the center."""
        positions = []
        x_offset = self.C_in + 2

        for m in range(self.TC_M):
            for n in range(self.TC_N):
                for k in range(self.TC_K):
                    x = x_offset + n
                    y = m
                    z = k
                    positions.append([x, y, z])

        self.tc_positions = np.array(positions, dtype=np.float32)

        self.tc_verts, self.tc_faces = self.create_cube_mesh_data(
            self.tc_positions, self.cube_size * 0.85
        )

        n_cubes = len(self.tc_positions)
        self.tc_colors_state = np.tile(self.color_default, (n_cubes, 1))
        tc_face_colors = np.tile(self.color_default, (n_cubes * 12, 1))

        mesh_data = gl.MeshData(vertexes=self.tc_verts, faces=self.tc_faces)
        mesh_data.setFaceColors(tc_face_colors)

        self.tc_mesh = gl.GLMeshItem(
            meshdata=mesh_data,
            smooth=False,
            drawFaces=True,
            drawEdges=True,
            edgeColor=(0.5, 0.5, 0.5, 0.3),
            shader='edgeHilight',
            glOptions='translucent'
        )
        self.addItem(self.tc_mesh)

    # ========================================================================
    # Update Methods
    # ========================================================================

    def update_mesh_colors(self, mesh_item, verts, faces, face_colors):
        mesh_data = gl.MeshData(vertexes=verts, faces=faces)
        mesh_data.setFaceColors(face_colors)
        mesh_item.setMeshData(meshdata=mesh_data)

    def update_view(self, step):
        """Update visualization based on ConvIterationStep."""

        # Update Tensor Core MACs
        for m, n, k in step.completed_macs:
            if 0 <= m < self.TC_M and 0 <= n < self.TC_N and 0 <= k < self.TC_K:
                idx = self.get_tc_index(m, n, k)
                self.tc_colors_state[idx] = self.color_done

        tc_current = self.tc_colors_state.copy()
        for m, n, k in step.active_macs:
            if 0 <= m < self.TC_M and 0 <= n < self.TC_N and 0 <= k < self.TC_K:
                idx = self.get_tc_index(m, n, k)
                tc_current[idx] = self.color_active

        tc_face_colors = np.repeat(tc_current, 12, axis=0)
        self.update_mesh_colors(self.tc_mesh, self.tc_verts, self.tc_faces, tc_face_colors)

        # Update Input Volume - use coord_map for reliability
        for h, w, c in step.active_input:
            key = (h, w, c)
            if key in self.input_coord_map:
                idx = self.input_coord_map[key]
                self.input_colors_state[idx] = self.active_color_input

        input_face_colors = np.repeat(self.input_colors_state, 12, axis=0)
        self.update_mesh_colors(self.input_mesh, self.input_verts, self.input_faces, input_face_colors)

        # Update Kernel Volume
        for kh, kw, ci, co in step.active_kernel:
            key = (kh, kw, ci, co)
            if key in self.kernel_coord_map:
                idx = self.kernel_coord_map[key]
                self.kernel_colors_state[idx] = self.active_color_kernel

        kernel_face_colors = np.repeat(self.kernel_colors_state, 12, axis=0)
        self.update_mesh_colors(self.kernel_mesh, self.kernel_verts, self.kernel_faces, kernel_face_colors)

        # Update Output Volume
        for oh, ow, co in step.active_output:
            key = (oh, ow, co)
            if key in self.output_coord_map:
                idx = self.output_coord_map[key]
                self.output_colors_state[idx] = self.active_color_output

        output_face_colors = np.repeat(self.output_colors_state, 12, axis=0)
        self.update_mesh_colors(self.output_mesh, self.output_verts, self.output_faces, output_face_colors)

    def reset_simulation(self):
        """Reset all visualization colors."""
        n_input = len(self.input_positions)
        self.input_colors_state = np.tile(self.base_color_input, (n_input, 1))
        input_face_colors = np.tile(self.base_color_input, (n_input * 12, 1))
        self.update_mesh_colors(self.input_mesh, self.input_verts, self.input_faces, input_face_colors)

        n_kernel = len(self.kernel_positions)
        self.kernel_colors_state = np.tile(self.base_color_kernel, (n_kernel, 1))
        kernel_face_colors = np.tile(self.base_color_kernel, (n_kernel * 12, 1))
        self.update_mesh_colors(self.kernel_mesh, self.kernel_verts, self.kernel_faces, kernel_face_colors)

        n_output = len(self.output_positions)
        self.output_colors_state = np.tile(self.base_color_output, (n_output, 1))
        output_face_colors = np.tile(self.base_color_output, (n_output * 12, 1))
        self.update_mesh_colors(self.output_mesh, self.output_verts, self.output_faces, output_face_colors)

        n_tc = len(self.tc_positions)
        self.tc_colors_state = np.tile(self.color_default, (n_tc, 1))
        tc_face_colors = np.tile(self.color_default, (n_tc * 12, 1))
        self.update_mesh_colors(self.tc_mesh, self.tc_verts, self.tc_faces, tc_face_colors)

    def keyPressEvent(self, event):
        if self.key_event_callback:
            self.key_event_callback(event)
        else:
            super().keyPressEvent(event)
