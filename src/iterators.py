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
        dims = {'i': self.M, 'j': self.N, 'k': self.K}
        ranges = [range(dims[c]) for c in self.order]
        for idx0 in ranges[0]:
            for idx1 in ranges[1]:
                for idx2 in ranges[2]:
                    current_indices = {self.order[0]: idx0, self.order[1]: idx1, self.order[2]: idx2}
                    i, j, k = current_indices['i'], current_indices['j'], current_indices['k']
                    coord = (i, j, k)
                    yield IterationStep(active=[coord], completed=[], description=f"Calc C[{i},{j}] += A[{i},{k}]*B[{k},{j}]")
                    yield IterationStep(active=[], completed=[coord], description="Done")

class TiledIterator(MatrixIterator):
    def __init__(self, M: int, N: int, K: int, tile_size: int = 2, tile_k: int = 2):
        super().__init__(M, N, K)
        self.tile_size = tile_size
        self.tile_k = tile_k

    def run(self) -> Generator[IterationStep, None, None]:
        completed_coords = []
        for i_base in range(0, self.M, self.tile_size):
            for j_base in range(0, self.N, self.tile_size):
                for k_base in range(0, self.K, self.tile_k):
                    active_coords = []
                    for i in range(i_base, min(i_base + self.tile_size, self.M)):
                        for j in range(j_base, min(j_base + self.tile_size, self.N)):
                            for k in range(k_base, min(k_base + self.tile_k, self.K)):
                                active_coords.append((i, j, k))
                    if active_coords:
                        yield IterationStep(active=active_coords, completed=completed_coords, description=f"Processing Tile")
                        completed_coords = active_coords
        
        # Flush last
        if completed_coords:
            yield IterationStep(active=[], completed=completed_coords, description="Done")

class SystolicIterator(MatrixIterator):
    def __init__(self, M: int, N: int, K: int):
        super().__init__(M, N, K)

    def run(self) -> Generator[IterationStep, None, None]:
        max_time = (self.M - 1) + (self.N - 1) + (self.K - 1)
        completed_coords = []
        for t in range(max_time + 1):
            active_coords = []
            for i in range(self.M):
                for j in range(self.N):
                    k = t - i - j
                    if 0 <= k < self.K:
                        active_coords.append((i, j, k))
            if active_coords:
                yield IterationStep(active=active_coords, completed=completed_coords, description=f"Systolic Wavefront t={t}")
                completed_coords = active_coords
        
        if completed_coords:
            yield IterationStep(active=[], completed=completed_coords, description="Done")

class BlockedSystolicIterator(MatrixIterator):
    def __init__(self, M: int, N: int, K: int, array_size: int = 4):
        super().__init__(M, N, K)
        self.array_size = array_size

    def run(self) -> Generator[IterationStep, None, None]:
        S = self.array_size
        blocks = []
        for i_base in range(0, self.M, S):
            for j_base in range(0, self.N, S):
                M_curr = min(i_base + S, self.M) - i_base
                N_curr = min(j_base + S, self.N) - j_base
                K_curr = self.K
                blocks.append({'i_base': i_base, 'j_base': j_base, 'M_curr': M_curr, 'N_curr': N_curr, 'K_curr': K_curr})
        
        current_start_time = 0
        block_schedules = []
        for b in blocks:
            block_schedules.append({'block': b, 'start_time': current_start_time})
            current_start_time += self.K 
            
        if not block_schedules:
            return
            
        last_block = block_schedules[-1]
        last_duration = (last_block['block']['M_curr'] - 1) + (last_block['block']['N_curr'] - 1) + (last_block['block']['K_curr'] - 1)
        max_global_time = last_block['start_time'] + last_duration + 1
        
        completed_coords = []
        
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
                    completed=completed_coords, 
                    description=f"Global t={t}, Active Blocks: {', '.join(active_blocks_desc)}"
                )
                completed_coords = active_coords
            elif completed_coords:
                 yield IterationStep(active=[], completed=completed_coords, description="Finishing...")
                 completed_coords = []

class TensorSystolicIterator(MatrixIterator):
    def __init__(self, M: int, N: int, K: int, array_size: int = 4, micro_size: Tuple[int, int, int] = (2, 2, 2)):
        super().__init__(M, N, K)
        self.array_size = array_size
        self.micro_size = micro_size # (M2, N2, K2)

    def run(self) -> Generator[IterationStep, None, None]:
        S = self.array_size
        M2, N2, K2 = self.micro_size
        
        M_macro = (self.M + M2 - 1) // M2
        N_macro = (self.N + N2 - 1) // N2
        K_macro = (self.K + K2 - 1) // K2
        
        blocks = []
        for i_macro_base in range(0, M_macro, S):
            for j_macro_base in range(0, N_macro, S):
                M_curr_macro = min(i_macro_base + S, M_macro) - i_macro_base
                N_curr_macro = min(j_macro_base + S, N_macro) - j_macro_base
                K_curr_macro = K_macro
                
                blocks.append({
                    'i_base': i_macro_base,
                    'j_base': j_macro_base,
                    'M_curr': M_curr_macro,
                    'N_curr': N_curr_macro,
                    'K_curr': K_curr_macro
                })

        current_start_time = 0
        block_schedules = []
        for b in blocks:
            block_schedules.append({'block': b, 'start_time': current_start_time})
            current_start_time += K_macro 
            
        if not block_schedules:
            return

        last_block = block_schedules[-1]
        last_duration = (last_block['block']['M_curr'] - 1) + (last_block['block']['N_curr'] - 1) + (last_block['block']['K_curr'] - 1)
        max_global_time = last_block['start_time'] + last_duration + 1
        
        completed_coords = []
        
        for t in range(max_global_time + 1):
            active_coords = []
            
            for schedule in block_schedules:
                start = schedule['start_time']
                b = schedule['block']
                local_t = t - start
                duration = (b['M_curr'] - 1) + (b['N_curr'] - 1) + (b['K_curr'] - 1)
                
                if 0 <= local_t <= duration:
                    for i_local in range(b['M_curr']):
                        for j_local in range(b['N_curr']):
                            k_local = local_t - i_local - j_local
                            if 0 <= k_local < b['K_curr']:
                                macro_i = b['i_base'] + i_local
                                macro_j = b['j_base'] + j_local
                                macro_k = k_local
                                
                                i_start, i_end = macro_i * M2, min((macro_i + 1) * M2, self.M)
                                j_start, j_end = macro_j * N2, min((macro_j + 1) * N2, self.N)
                                k_start, k_end = macro_k * K2, min((macro_k + 1) * K2, self.K)
                                
                                for x in range(i_start, i_end):
                                    for y in range(j_start, j_end):
                                        for z in range(k_start, k_end):
                                            active_coords.append((x, y, z))

            if active_coords:
                yield IterationStep(active=active_coords, completed=completed_coords, description=f"Tensor Core Step t={t}")
                completed_coords = active_coords
            elif completed_coords:
                yield IterationStep(active=[], completed=completed_coords, description="Finishing...")
                completed_coords = []
