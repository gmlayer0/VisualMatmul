import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Generator


@dataclass
class ConvIterationStep:
    """
    Represents one step in the convolution visualization.
    Coordinates are (oh, ow, kh, kw) for 4D space,
    where (oh, ow) is output position, (kh, kw) is kernel position.
    """
    active: List[Tuple[int, int, int, int]]      # Currently processing
    completed: List[Tuple[int, int, int, int]]   # Just finished
    active_input: List[Tuple[int, int]]           # Input pixels being read
    active_kernel: List[Tuple[int, int]]          # Kernel weights being read
    active_output: List[Tuple[int, int]]          # Output pixels being written
    description: str = ""


class ConvIterator(ABC):
    """Base class for convolution iterators."""

    def __init__(self, H: int, W: int, KH: int, KW: int, stride: int = 1, padding: int = 0):
        self.H = H          # Input height
        self.W = W          # Input width
        self.KH = KH        # Kernel height
        self.KW = KW        # Kernel width
        self.stride = stride
        self.padding = padding

        # Calculate output dimensions
        self.OH = (H + 2 * padding - KH) // stride + 1
        self.OW = (W + 2 * padding - KW) // stride + 1

    @abstractmethod
    def run(self) -> Generator[ConvIterationStep, None, None]:
        pass

    def get_input_coord(self, oh: int, ow: int, kh: int, kw: int) -> Tuple[int, int]:
        """Get input coordinate from output and kernel position."""
        ih = oh * self.stride - self.padding + kh
        iw = ow * self.stride - self.padding + kw
        return (ih, iw)


# ============================================================================
# Output Stationary (OS) - Output pixel stays fixed, kernel slides over it
# ============================================================================

class OutputStationaryIterator(ConvIterator):
    """
    Output Stationary: One output element is computed completely before
    moving to the next output element. Good for accumulating partial sums.

    Traversal order: oh -> ow -> kh -> kw
    """

    def run(self) -> Generator[ConvIterationStep, None, None]:
        completed_coords = []

        for oh in range(self.OH):
            for ow in range(self.OW):
                # Compute one output pixel completely
                active_coords = []
                active_input = []
                active_kernel = []

                for kh in range(self.KH):
                    for kw in range(self.KW):
                        coord = (oh, ow, kh, kw)
                        active_coords.append(coord)

                        ih, iw = self.get_input_coord(oh, ow, kh, kw)
                        if 0 <= ih < self.H and 0 <= iw < self.W:
                            active_input.append((ih, iw))
                        active_kernel.append((kh, kw))

                # Yield active phase
                yield ConvIterationStep(
                    active=active_coords,
                    completed=completed_coords,
                    active_input=active_input,
                    active_kernel=active_kernel,
                    active_output=[(oh, ow)],
                    description=f"OS: Output[{oh},{ow}] = sum_k Input[{oh*self.stride - self.padding + kh},{ow*self.stride - self.padding + kw}] * Kernel[kh,kw]"
                )

                completed_coords = active_coords

        # Final completion
        if completed_coords:
            yield ConvIterationStep(
                active=[],
                completed=completed_coords,
                active_input=[],
                active_kernel=[],
                active_output=[],
                description="Done"
            )


class OutputStationarySystolicIterator(ConvIterator):
    """
    Output Stationary with systolic array wavefront.
    Multiple output pixels computed in parallel in a wavefront pattern.
    """

    def __init__(self, H: int, W: int, KH: int, KW: int,
                 stride: int = 1, padding: int = 0, array_size: int = 4):
        super().__init__(H, W, KH, KW, stride, padding)
        self.array_size = array_size

    def run(self) -> Generator[ConvIterationStep, None, None]:
        max_t = self.OH + self.OW + self.KH + self.KW - 4
        completed_coords = []

        for t in range(max_t + 1):
            active_coords = []
            active_input = []
            active_kernel = []
            active_output = []

            # Wavefront: oh + ow = t - (kh + kw)
            # Process kernel positions first
            for kh in range(self.KH):
                for kw in range(self.KW):
                    remaining = t - kh - kw
                    if remaining < 0:
                        continue

                    for oh in range(self.OH):
                        ow = remaining - oh
                        if 0 <= ow < self.OW:
                            coord = (oh, ow, kh, kw)
                            active_coords.append(coord)

                            ih, iw = self.get_input_coord(oh, ow, kh, kw)
                            if 0 <= ih < self.H and 0 <= iw < self.W:
                                active_input.append((ih, iw))
                            active_kernel.append((kh, kw))
                            if (oh, ow) not in active_output:
                                active_output.append((oh, ow))

            if active_coords:
                yield ConvIterationStep(
                    active=active_coords,
                    completed=completed_coords,
                    active_input=active_input,
                    active_kernel=active_kernel,
                    active_output=active_output,
                    description=f"OS Systolic: t={t}, Active outputs={len(active_output)}"
                )
                completed_coords = active_coords
            elif completed_coords:
                yield ConvIterationStep(
                    active=[],
                    completed=completed_coords,
                    active_input=[],
                    active_kernel=[],
                    active_output=[],
                    description="Finishing..."
                )
                completed_coords = []

        if completed_coords:
            yield ConvIterationStep(
                active=[],
                completed=completed_coords,
                active_input=[],
                active_kernel=[],
                active_output=[],
                description="Done"
            )


# ============================================================================
# Weight Stationary (WS) - Kernel weights stay in PE registers
# ============================================================================

class WeightStationaryIterator(ConvIterator):
    """
    Weight Stationary: Each PE holds one kernel weight.
    Input is broadcast, partial sums flow through the array.

    Traversal order: kh -> kw -> oh -> ow
    """

    def run(self) -> Generator[ConvIterationStep, None, None]:
        completed_coords = []

        for kh in range(self.KH):
            for kw in range(self.KW):
                # Process one kernel weight across all outputs
                active_coords = []
                active_input = []
                active_kernel = []
                active_output = []

                for oh in range(self.OH):
                    for ow in range(self.OW):
                        coord = (oh, ow, kh, kw)
                        active_coords.append(coord)

                        ih, iw = self.get_input_coord(oh, ow, kh, kw)
                        if 0 <= ih < self.H and 0 <= iw < self.W:
                            if (ih, iw) not in active_input:
                                active_input.append((ih, iw))
                        if (kh, kw) not in active_kernel:
                            active_kernel.append((kh, kw))
                        if (oh, ow) not in active_output:
                            active_output.append((oh, ow))

                yield ConvIterationStep(
                    active=active_coords,
                    completed=completed_coords,
                    active_input=active_input,
                    active_kernel=active_kernel,
                    active_output=active_output,
                    description=f"WS: Kernel[{kh},{kw}] broadcast to all outputs"
                )

                completed_coords = active_coords

        if completed_coords:
            yield ConvIterationStep(
                active=[],
                completed=completed_coords,
                active_input=[],
                active_kernel=[],
                active_output=[],
                description="Done"
            )


class WeightStationarySystolicIterator(ConvIterator):
    """
    Weight Stationary with systolic data flow.
    Kernel weights are pre-loaded, inputs flow from left/top,
    partial sums accumulate as they flow through the array.
    """

    def __init__(self, H: int, W: int, KH: int, KW: int,
                 stride: int = 1, padding: int = 0, array_size: int = 4):
        super().__init__(H, W, KH, KW, stride, padding)
        self.array_size = array_size

    def run(self) -> Generator[ConvIterationStep, None, None]:
        # Timestep based on input arrival
        # Inputs enter at different times based on their position
        max_t = (self.H + self.padding) + (self.W + self.padding) + self.OH + self.OW - 4
        completed_coords = []

        for t in range(max_t + 1):
            active_coords = []
            active_input = []
            active_kernel = []
            active_output = []

            # For each output and kernel position, check if data arrives at this timestep
            for oh in range(self.OH):
                for ow in range(self.OW):
                    for kh in range(self.KH):
                        for kw in range(self.KW):
                            # Input arrives at t = (oh*s - p + kh) + (ow*s - p + kw)
                            ih = oh * self.stride - self.padding + kh
                            iw = ow * self.stride - self.padding + kw
                            arrival_t = ih + iw

                            # Kernel stays, data flows through
                            if arrival_t == t and 0 <= ih < self.H and 0 <= iw < self.W:
                                coord = (oh, ow, kh, kw)
                                active_coords.append(coord)
                                active_input.append((ih, iw))
                                if (kh, kw) not in active_kernel:
                                    active_kernel.append((kh, kw))
                                if (oh, ow) not in active_output:
                                    active_output.append((oh, ow))

            if active_coords:
                yield ConvIterationStep(
                    active=active_coords,
                    completed=completed_coords,
                    active_input=active_input,
                    active_kernel=active_kernel,
                    active_output=active_output,
                    description=f"WS Systolic: t={t}, Inputs flowing"
                )
                completed_coords = active_coords

        if completed_coords:
            yield ConvIterationStep(
                active=[],
                completed=completed_coords,
                active_input=[],
                active_kernel=[],
                active_output=[],
                description="Done"
            )


# ============================================================================
# Input Stationary (IS) - Input pixels stay fixed
# ============================================================================

class InputStationaryIterator(ConvIterator):
    """
    Input Stationary: Input pixel stays in PE, used for different kernel positions
    and output positions. Maximizes input reuse.

    Traversal order: ih -> iw -> kh -> kw -> oh -> ow (filtered)
    """

    def run(self) -> Generator[ConvIterationStep, None, None]:
        completed_coords = []

        for ih in range(self.H):
            for iw in range(self.W):
                # Find all (oh, ow, kh, kw) that use this input
                active_coords = []
                active_input = [(ih, iw)]
                active_kernel = []
                active_output = []

                for kh in range(self.KH):
                    for kw in range(self.KW):
                        # oh = (ih + p - kh) / s
                        # Check if valid
                        for oh in range(self.OH):
                            for ow in range(self.OW):
                                calc_ih = oh * self.stride - self.padding + kh
                                calc_iw = ow * self.stride - self.padding + kw
                                if calc_ih == ih and calc_iw == iw:
                                    coord = (oh, ow, kh, kw)
                                    active_coords.append(coord)
                                    if (kh, kw) not in active_kernel:
                                        active_kernel.append((kh, kw))
                                    if (oh, ow) not in active_output:
                                        active_output.append((oh, ow))

                if active_coords:
                    yield ConvIterationStep(
                        active=active_coords,
                        completed=completed_coords,
                        active_input=active_input,
                        active_kernel=active_kernel,
                        active_output=active_output,
                        description=f"IS: Input[{ih},{iw}] reused across {len(active_output)} outputs"
                    )
                    completed_coords = active_coords

        if completed_coords:
            yield ConvIterationStep(
                active=[],
                completed=completed_coords,
                active_input=[],
                active_kernel=[],
                active_output=[],
                description="Done"
            )


class InputStationarySystolicIterator(ConvIterator):
    """
    Input Stationary with systolic wavefront pattern.
    Inputs are loaded and stay in place, kernels and outputs shift.
    """

    def __init__(self, H: int, W: int, KH: int, KW: int,
                 stride: int = 1, padding: int = 0, array_size: int = 4):
        super().__init__(H, W, KH, KW, stride, padding)
        self.array_size = array_size

    def run(self) -> Generator[ConvIterationStep, None, None]:
        # Wavefront pattern: ih + iw = t
        max_t = (self.H + 2 * self.padding) + (self.W + 2 * self.padding) + self.KH + self.KW - 4
        completed_coords = []

        for t in range(max_t + 1):
            active_coords = []
            active_input = []
            active_kernel = []
            active_output = []

            # Inputs with ih + iw = t are active
            for oh in range(self.OH):
                for ow in range(self.OW):
                    for kh in range(self.KH):
                        for kw in range(self.KW):
                            ih = oh * self.stride - self.padding + kh
                            iw = ow * self.stride - self.padding + kw
                            if 0 <= ih < self.H and 0 <= iw < self.W:
                                if ih + iw == t - kh - kw:
                                    coord = (oh, ow, kh, kw)
                                    active_coords.append(coord)
                                    if (ih, iw) not in active_input:
                                        active_input.append((ih, iw))
                                    if (kh, kw) not in active_kernel:
                                        active_kernel.append((kh, kw))
                                    if (oh, ow) not in active_output:
                                        active_output.append((oh, ow))

            if active_coords:
                yield ConvIterationStep(
                    active=active_coords,
                    completed=completed_coords,
                    active_input=active_input,
                    active_kernel=active_kernel,
                    active_output=active_output,
                    description=f"IS Systolic: t={t}"
                )
                completed_coords = active_coords

        if completed_coords:
            yield ConvIterationStep(
                active=[],
                completed=completed_coords,
                active_input=[],
                active_kernel=[],
                active_output=[],
                description="Done"
            )


# ============================================================================
# Row Stationary (RS) - Specialized for CNNs
# ============================================================================

class RowStationaryIterator(ConvIterator):
    """
    Row Stationary: Optimized for row-wise convolution.
    Each PE handles one row of output and slides across columns.
    Good for 1D convolution and separable filters.

    Traversal order: oh -> kh -> ow -> kw
    """

    def run(self) -> Generator[ConvIterationStep, None, None]:
        completed_coords = []

        for oh in range(self.OH):
            for kh in range(self.KH):
                # Process one output row with one kernel row
                active_coords = []
                active_input = []
                active_kernel = []
                active_output = []

                for ow in range(self.OW):
                    for kw in range(self.KW):
                        coord = (oh, ow, kh, kw)
                        active_coords.append(coord)

                        ih, iw = self.get_input_coord(oh, ow, kh, kw)
                        if 0 <= ih < self.H and 0 <= iw < self.W:
                            active_input.append((ih, iw))
                        active_kernel.append((kh, kw))
                        if (oh, ow) not in active_output:
                            active_output.append((oh, ow))

                if active_coords:
                    yield ConvIterationStep(
                        active=active_coords,
                        completed=completed_coords,
                        active_input=active_input,
                        active_kernel=active_kernel,
                        active_output=active_output,
                        description=f"RS: Output Row[{oh}], Kernel Row[{kh}]"
                    )
                    completed_coords = active_coords

        if completed_coords:
            yield ConvIterationStep(
                active=[],
                completed=completed_coords,
                active_input=[],
                active_kernel=[],
                active_output=[],
                description="Done"
            )


class RowStationarySystolicIterator(ConvIterator):
    """
    Row Stationary with systolic array.
    Rows flow through the array, each PE accumulates its row.
    """

    def __init__(self, H: int, W: int, KH: int, KW: int,
                 stride: int = 1, padding: int = 0, array_size: int = 4):
        super().__init__(H, W, KH, KW, stride, padding)
        self.array_size = array_size

    def run(self) -> Generator[ConvIterationStep, None, None]:
        # Timestep based on column position
        max_t = self.OW + self.W + self.KW + self.padding * 2
        completed_coords = []

        for t in range(max_t + 1):
            active_coords = []
            active_input = []
            active_kernel = []
            active_output = []

            for oh in range(self.OH):
                for kh in range(self.KH):
                    for ow in range(self.OW):
                        for kw in range(self.KW):
                            ih, iw = self.get_input_coord(oh, ow, kh, kw)
                            # Column-based timing
                            if iw == t - ow - kw and 0 <= ih < self.H and 0 <= iw < self.W:
                                coord = (oh, ow, kh, kw)
                                active_coords.append(coord)
                                active_input.append((ih, iw))
                                if (kh, kw) not in active_kernel:
                                    active_kernel.append((kh, kw))
                                if (oh, ow) not in active_output:
                                    active_output.append((oh, ow))

            if active_coords:
                yield ConvIterationStep(
                    active=active_coords,
                    completed=completed_coords,
                    active_input=active_input,
                    active_kernel=active_kernel,
                    active_output=active_output,
                    description=f"RS Systolic: t={t}, Columns flowing"
                )
                completed_coords = active_coords

        if completed_coords:
            yield ConvIterationStep(
                active=[],
                completed=completed_coords,
                active_input=[],
                active_kernel=[],
                active_output=[],
                description="Done"
            )
