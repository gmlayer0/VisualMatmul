from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Generator

@dataclass
class IterationStep:
    active: List[Tuple[int, int, int]]  # List of (x, y, z) coordinates being processed
    completed: List[Tuple[int, int, int]] # List of (x, y, z) coordinates that finished processing
    description: str = ""

class MatrixIterator(ABC):
    def __init__(self, M: int, N: int, K: int):
        self.M = M
        self.N = N
        self.K = K

    @abstractmethod
    def run(self) -> Generator[IterationStep, None, None]:
        pass

class NaiveIterator(MatrixIterator):
    def __init__(self, M: int, N: int, K: int, order: str = "ijk"):
        super().__init__(M, N, K)
        self.order = order.lower()
        if set(self.order) != set("ijk") or len(self.order) != 3:
             raise ValueError("Order must be a permutation of 'ijk'")

    def run(self) -> Generator[IterationStep, None, None]:
        # Map axes names to dimensions
        dims = {'i': self.M, 'j': self.N, 'k': self.K}
        
        # Ranges for each loop
        ranges = [range(dims[c]) for c in self.order]
        
        # Iterate based on order
        for idx0 in ranges[0]:
            for idx1 in ranges[1]:
                for idx2 in ranges[2]:
                    # Map back to i, j, k
                    current_indices = {
                        self.order[0]: idx0,
                        self.order[1]: idx1,
                        self.order[2]: idx2
                    }
                    i, j, k = current_indices['i'], current_indices['j'], current_indices['k']
                    
                    # x=i (M), y=j (N), z=k (K)
                    coord = (i, j, k)
                    
                    # Yield step: this block is active
                    yield IterationStep(active=[coord], completed=[], description=f"Calc C[{i},{j}] += A[{i},{k}]*B[{k},{j}]")
                    
                    # In a simple view, after "processing", it's effectively "done" for this step
                    # But strictly speaking, a block (i,j,k) represents one MAC (Multiply-Accumulate).
                    yield IterationStep(active=[], completed=[coord], description="Done")

class TiledIterator(MatrixIterator):
    def __init__(self, M: int, N: int, K: int, tile_size: int = 2):
        super().__init__(M, N, K)
        self.tile_size = tile_size

    def run(self) -> Generator[IterationStep, None, None]:
        # Iterate over tiles
        for i_base in range(0, self.M, self.tile_size):
            for j_base in range(0, self.N, self.tile_size):
                for k_base in range(0, self.K, self.tile_size):
                    
                    # Collect all coordinates in this tile
                    active_coords = []
                    for i in range(i_base, min(i_base + self.tile_size, self.M)):
                        for j in range(j_base, min(j_base + self.tile_size, self.N)):
                            for k in range(k_base, min(k_base + self.tile_size, self.K)):
                                active_coords.append((i, j, k))
                    
                    if active_coords:
                        yield IterationStep(
                            active=active_coords, 
                            completed=[],
                            description=f"Processing Tile [{i_base}:{i_base+self.tile_size}, {j_base}:{j_base+self.tile_size}, {k_base}:{k_base+self.tile_size}]"
                        )
                        yield IterationStep(
                            active=[], 
                            completed=active_coords,
                            description="Tile Done"
                        )

