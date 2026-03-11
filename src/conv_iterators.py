import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Generator


@dataclass
class ConvIterationStep:
    """
    Represents one step in the Tensor Core convolution visualization.
    Coordinates are for the 3D volumes and MAC array.
    """
    # Active MAC operations in Tensor Core (m, n, k)
    # m = output tile row, n = output tile col, k = reduction dim
    active_macs: List[Tuple[int, int, int]]
    completed_macs: List[Tuple[int, int, int]]

    # Active elements in Input Volume (h, w, c)
    active_input: List[Tuple[int, int, int]]
    # Active elements in Kernel Volume (k_h, k_w, c_in, c_out)
    active_kernel: List[Tuple[int, int, int, int]]
    # Active elements in Output Volume (oh, ow, c_out)
    active_output: List[Tuple[int, int, int]]

    # Tensor Core tile info
    tile_m_start: int = 0
    tile_n_start: int = 0
    tile_k_start: int = 0
    tile_m_size: int = 16
    tile_n_size: int = 16
    tile_k_size: int = 16

    description: str = ""


class TensorCoreConvIterator(ABC):
    """Base class for Tensor Core convolution iterators."""

    def __init__(self, H: int, W: int, C_in: int, C_out: int,
                 KH: int = 3, KW: int = 3, stride: int = 1, padding: int = 0,
                 tc_m: int = 16, tc_n: int = 16, tc_k: int = 16):
        self.H = H          # Input height
        self.W = W          # Input width
        self.C_in = C_in    # Input channels
        self.C_out = C_out  # Output channels
        self.KH = KH        # Kernel height
        self.KW = KW        # Kernel width
        self.stride = stride
        self.padding = padding

        # Tensor Core dimensions
        self.TC_M = tc_m    # Output tile M dimension
        self.TC_N = tc_n    # Output tile N dimension
        self.TC_K = tc_k    # Reduction tile K dimension

        # Calculate output spatial dimensions
        self.OH = (H + 2 * padding - KH) // stride + 1
        self.OW = (W + 2 * padding - KW) // stride + 1

        # Flattened output space for tensor core: M = OH*OW*C_out
        self.M_total = self.OH * self.OW * self.C_out
        # K dimension = KH*KW*C_in
        self.K_total = self.KH * self.KW * self.C_in


class TensorCoreOutputStationaryIterator(TensorCoreConvIterator):
    """
    Tensor Core Output Stationary:
    Each Tensor Core handles one output tile, slides over K dimension.
    This is the classic Tensor Core approach.
    """

    def run(self) -> Generator[ConvIterationStep, None, None]:
        completed_macs = []

        # Iterate over output tiles (M dimension)
        for m_tile in range(0, self.M_total, self.TC_M):
            m_size = min(self.TC_M, self.M_total - m_tile)

            # Iterate over K tiles (reduction dimension)
            for k_tile in range(0, self.K_total, self.TC_K):
                k_size = min(self.TC_K, self.K_total - k_tile)

                active_macs = []
                active_input_set = set()
                active_kernel_set = set()
                active_output_set = set()

                # Within a tile, all MAC operations are active
                # (In real hardware, this happens in one tensor core instruction)
                for m in range(m_tile, m_tile + m_size):
                    # Map m back to output coordinates
                    c_out = m % self.C_out
                    rem = m // self.C_out
                    ow = rem % self.OW
                    oh = rem // self.OW

                    active_output_set.add((oh, ow, c_out))

                    for k in range(k_tile, k_tile + k_size):
                        # Map k back to kernel/input coordinates
                        c_in = k % self.C_in
                        rem = k // self.C_in
                        kw = rem % self.KW
                        kh = rem // self.KW

                        # Input coordinate for this kernel position
                        ih = oh * self.stride - self.padding + kh
                        iw = ow * self.stride - self.padding + kw

                        if 0 <= ih < self.H and 0 <= iw < self.W:
                            active_input_set.add((ih, iw, c_in))

                        active_kernel_set.add((kh, kw, c_in, c_out))

                        # MAC coordinate within tensor core tile
                        tc_m = m - m_tile
                        tc_k = k - k_tile

                        # For simplicity, pair each (m, k) with a dummy n
                        # In a full matmul, n would be another dimension
                        for tc_n in range(min(self.TC_N, 4)):
                            active_macs.append((tc_m, tc_n, tc_k))

                active_input = list(active_input_set)
                active_kernel = list(active_kernel_set)
                active_output = list(active_output_set)

                yield ConvIterationStep(
                    active_macs=active_macs,
                    completed_macs=completed_macs,
                    active_input=active_input,
                    active_kernel=active_kernel,
                    active_output=active_output,
                    tile_m_start=m_tile,
                    tile_n_start=0,
                    tile_k_start=k_tile,
                    tile_m_size=m_size,
                    tile_n_size=min(self.TC_N, 4),
                    tile_k_size=k_size,
                    description=f"TC OS: Output Tile [{m_tile}:{m_tile+m_size}], K Tile [{k_tile}:{k_tile+k_size}]"
                )

                completed_macs = active_macs

        if completed_macs:
            yield ConvIterationStep(
                active_macs=[],
                completed_macs=completed_macs,
                active_input=[],
                active_kernel=[],
                active_output=[],
                description="Done"
            )


class TensorCoreWeightStationaryIterator(TensorCoreConvIterator):
    """
    Tensor Core Weight Stationary:
    Kernel weights are preloaded in registers, input flows through.
    Good for weight reuse.
    """

    def run(self) -> Generator[ConvIterationStep, None, None]:
        completed_macs = []

        # Iterate over kernel tiles first
        for k_tile in range(0, self.K_total, self.TC_K):
            k_size = min(self.TC_K, self.K_total - k_tile)

            # Then iterate over output tiles
            for m_tile in range(0, self.M_total, self.TC_M):
                m_size = min(self.TC_M, self.M_total - m_tile)

                active_macs = []
                active_input = []
                active_kernel = []
                active_output = []

                for k in range(k_tile, k_tile + k_size):
                    c_in = k % self.C_in
                    rem = k // self.C_in
                    kw = rem % self.KW
                    kh = rem // self.KW

                    for m in range(m_tile, m_tile + m_size):
                        c_out = m % self.C_out
                        rem = m // self.C_out
                        ow = rem % self.OW
                        oh = rem // self.OW

                        ih = oh * self.stride - self.padding + kh
                        iw = ow * self.stride - self.padding + kw

                        if 0 <= ih < self.H and 0 <= iw < self.W:
                            active_input.append((ih, iw, c_in))

                        active_kernel.append((kh, kw, c_in, c_out))
                        active_output.append((oh, ow, c_out))

                        tc_m = m - m_tile
                        tc_k = k - k_tile
                        for tc_n in range(min(self.TC_N, 4)):
                            active_macs.append((tc_m, tc_n, tc_k))

                yield ConvIterationStep(
                    active_macs=active_macs,
                    completed_macs=completed_macs,
                    active_input=active_input,
                    active_kernel=active_kernel,
                    active_output=active_output,
                    tile_m_start=m_tile,
                    tile_k_start=k_tile,
                    tile_m_size=m_size,
                    tile_k_size=k_size,
                    description=f"TC WS: Kernel Tile [{k_tile}:{k_tile+k_size}], Output Tile [{m_tile}:{m_tile+m_size}]"
                )

                completed_macs = active_macs

        if completed_macs:
            yield ConvIterationStep(
                active_macs=[],
                completed_macs=completed_macs,
                active_input=[],
                active_kernel=[],
                active_output=[],
                description="Done"
            )


class TensorCoreInputStationaryIterator(TensorCoreConvIterator):
    """
    Tensor Core Input Stationary:
    Input stays, kernels and outputs are scheduled around it.
    Maximizes input reuse.
    """

    def run(self) -> Generator[ConvIterationStep, None, None]:
        completed_macs = []

        # For each input element, find all uses
        # Let's iterate by input channels and spatial regions
        for c_in in range(0, self.C_in, max(1, self.TC_K // (self.KH * self.KW))):
            c_in_end = min(c_in + max(1, self.TC_K // (self.KH * self.KW)), self.C_in)

            active_macs = []
            active_input = []
            active_kernel = []
            active_output = []

            # Find all outputs that use this input channel
            for oh in range(self.OH):
                for ow in range(self.OW):
                    for c_out in range(self.C_out):
                        m = oh * self.OW * self.C_out + ow * self.C_out + c_out
                        tc_m = m % self.TC_M

                        for kh in range(self.KH):
                            for kw in range(self.KW):
                                ih = oh * self.stride - self.padding + kh
                                iw = ow * self.stride - self.padding + kw

                                if 0 <= ih < self.H and 0 <= iw < self.W:
                                    for ci in range(c_in, c_in_end):
                                        k = kh * self.KW * self.C_in + kw * self.C_in + ci
                                        tc_k = (k % self.TC_K)

                                        active_input.append((ih, iw, ci))
                                        active_kernel.append((kh, kw, ci, c_out))
                                        active_output.append((oh, ow, c_out))

                                        for tc_n in range(min(self.TC_N, 4)):
                                            active_macs.append((tc_m, tc_n, tc_k))

            if active_macs:
                yield ConvIterationStep(
                    active_macs=active_macs,
                    completed_macs=completed_macs,
                    active_input=list(set(active_input)),
                    active_kernel=list(set(active_kernel)),
                    active_output=list(set(active_output)),
                    tile_m_start=0,
                    tile_k_start=c_in * self.KH * self.KW,
                    description=f"TC IS: Input Channels [{c_in}:{c_in_end}]"
                )

                completed_macs = active_macs

        if completed_macs:
            yield ConvIterationStep(
                active_macs=[],
                completed_macs=completed_macs,
                active_input=[],
                active_kernel=[],
                active_output=[],
                description="Done"
            )


class TensorCoreSystolicIterator(TensorCoreConvIterator):
    """
    Tensor Core with Systolic Data Flow:
    Wavefront pattern through the Tensor Core array.
    Inputs enter from one side, partial sums accumulate.
    """

    def run(self) -> Generator[ConvIterationStep, None, None]:
        completed_macs = []

        # Wavefront timing
        max_time = self.M_total // self.TC_M + self.K_total // self.TC_K + 10

        for t in range(max_time):
            active_macs = []
            active_input = []
            active_kernel = []
            active_output = []

            # At time t, tile pairs where m_tile_idx + k_tile_idx = t
            for m_tile_idx in range(0, (self.M_total + self.TC_M - 1) // self.TC_M):
                k_tile_idx = t - m_tile_idx

                if k_tile_idx < 0:
                    continue
                if k_tile_idx >= (self.K_total + self.TC_K - 1) // self.TC_K:
                    continue

                m_tile = m_tile_idx * self.TC_M
                k_tile = k_tile_idx * self.TC_K

                if m_tile >= self.M_total or k_tile >= self.K_total:
                    continue

                m_size = min(self.TC_M, self.M_total - m_tile)
                k_size = min(self.TC_K, self.K_total - k_tile)

                for m in range(m_tile, m_tile + m_size):
                    c_out = m % self.C_out
                    rem = m // self.C_out
                    ow = rem % self.OW
                    oh = rem // self.OW

                    active_output.append((oh, ow, c_out))

                    for k in range(k_tile, k_tile + k_size):
                        c_in = k % self.C_in
                        rem = k // self.C_in
                        kw = rem % self.KW
                        kh = rem // self.KW

                        ih = oh * self.stride - self.padding + kh
                        iw = ow * self.stride - self.padding + kw

                        if 0 <= ih < self.H and 0 <= iw < self.W:
                            active_input.append((ih, iw, c_in))

                        active_kernel.append((kh, kw, c_in, c_out))

                        tc_m = m - m_tile
                        tc_k = k - k_tile
                        for tc_n in range(min(self.TC_N, 4)):
                            active_macs.append((tc_m, tc_n, tc_k))

            if active_macs:
                yield ConvIterationStep(
                    active_macs=active_macs,
                    completed_macs=completed_macs,
                    active_input=list(set(active_input)),
                    active_kernel=list(set(active_kernel)),
                    active_output=list(set(active_output)),
                    description=f"TC Systolic: Wavefront t={t}"
                )

                completed_macs = active_macs
            elif completed_macs:
                yield ConvIterationStep(
                    active_macs=[],
                    completed_macs=completed_macs,
                    active_input=[],
                    active_kernel=[],
                    active_output=[],
                    description="Finishing..."
                )
                completed_macs = []

        if completed_macs:
            yield ConvIterationStep(
                active_macs=[],
                completed_macs=completed_macs,
                active_input=[],
                active_kernel=[],
                active_output=[],
                description="Done"
            )
