import numpy as np
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

class SystolicIterator(MatrixIterator):
    def __init__(self, M: int, N: int, K: int):
        super().__init__(M, N, K)

    def run(self) -> Generator[IterationStep, None, None]:
        # Output Stationary Systolic Array Logic
        # Time step t = i + j + k
        # Max time steps = (M-1) + (N-1) + (K-1)
        max_time = (self.M - 1) + (self.N - 1) + (self.K - 1)
        
        for t in range(max_time + 1):
            active_coords = []
            
            # Find all (i, j, k) such that i + j + k == t
            # Optimization: Iterate over i and j, calculate k = t - i - j
            # Valid if 0 <= k < K
            
            for i in range(self.M):
                for j in range(self.N):
                    k = t - i - j
                    if 0 <= k < self.K:
                        active_coords.append((i, j, k))
            
            if active_coords:
                yield IterationStep(
                    active=active_coords,
                    completed=[],
                    description=f"Systolic Wavefront t={t}"
                )
                
                yield IterationStep(
                    active=[],
                    completed=active_coords,
                    description=f"Wavefront t={t} Done"
                )

class BlockedSystolicIterator(MatrixIterator):
    def __init__(self, M: int, N: int, K: int, array_size: int = 4):
        super().__init__(M, N, K)
        self.array_size = array_size

    def run(self) -> Generator[IterationStep, None, None]:
        S = self.array_size
        
        # Pre-calculate blocks
        blocks = []
        for i_base in range(0, self.M, S):
            for j_base in range(0, self.N, S):
                # K is not blocked (full K used)
                M_curr = min(i_base + S, self.M) - i_base
                N_curr = min(j_base + S, self.N) - j_base
                K_curr = self.K
                
                blocks.append({
                    'i_base': i_base,
                    'j_base': j_base,
                    'M_curr': M_curr,
                    'N_curr': N_curr,
                    'K_curr': K_curr
                })
        
        # Start time for each block
        # Assuming pipeline: Block N+1 starts when Block N's PE(0,0) is free
        current_start_time = 0
        block_schedules = []
        for b in blocks:
            block_schedules.append({
                'block': b,
                'start_time': current_start_time
            })
            current_start_time += self.K 
            
        # Global simulation loop
        if not block_schedules:
            return
            
        last_block = block_schedules[-1]
        last_duration = (last_block['block']['M_curr'] - 1) + (last_block['block']['N_curr'] - 1) + (last_block['block']['K_curr'] - 1)
        max_global_time = last_block['start_time'] + last_duration + 1
        
        for t in range(max_global_time + 1):
            active_coords = []
            active_blocks_desc = []
            
            for schedule in block_schedules:
                start = schedule['start_time']
                b = schedule['block']
                local_t = t - start
                
                duration = (b['M_curr'] - 1) + (b['N_curr'] - 1) + (b['K_curr'] - 1)
                
                if 0 <= local_t <= duration:
                    block_active = False
                    for i_local in range(b['M_curr']):
                        for j_local in range(b['N_curr']):
                            k_local = local_t - i_local - j_local
                            if 0 <= k_local < b['K_curr']:
                                global_coord = (b['i_base'] + i_local, b['j_base'] + j_local, k_local)
                                active_coords.append(global_coord)
                                block_active = True
                    
                    if block_active:
                         active_blocks_desc.append(f"[{b['i_base']}:{b['j_base']}]")

            if active_coords:
                yield IterationStep(
                    active=active_coords,
                    completed=[],
                    description=f"Global t={t}, Active Blocks: {', '.join(active_blocks_desc)}"
                )
                
                yield IterationStep(
                    active=[],
                    completed=active_coords,
                    description=""
                )
