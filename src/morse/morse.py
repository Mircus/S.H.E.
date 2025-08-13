"""
SHE - Finite Morse Theory Extension
Discrete Morse theory implementation for simplicial complexes
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Set, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
import itertools

# Import the base SHE classes (assuming they're available)
# from she_core import SHESimplicialComplex, SHEConfig, SHEEngine

logger = logging.getLogger(__name__)

class CellType(Enum):
    """Type of critical cell in Morse theory"""
    CRITICAL = "critical"
    REGULAR = "regular" 
    PAIRED = "paired"

@dataclass
class MorseFunction:
    """Represents a discrete Morse function on a simplicial complex"""
    values: Dict[Tuple, float] = field(default_factory=dict)
    gradient_pairs: Set[Tuple[Tuple, Tuple]] = field(default_factory=set)
    critical_cells: Dict[int, Set[Tuple]] = field(default_factory=lambda: defaultdict(set))
    
    def __post_init__(self):
        self.cell_types: Dict[Tuple, CellType] = {}
        self._classify_cells()
    
    def _classify_cells(self):
        """Classify cells as critical, regular, or paired"""
        paired_cells = set()
        for pair in self.gradient_pairs:
            paired_cells.update(pair)
            
        for cell in self.values.keys():
            if cell in paired_cells:
                self.cell_types[cell] = CellType.PAIRED
            else:
                self.cell_types[cell] = CellType.CRITICAL
                
        # Regular cells are those that aren't critical but could be paired
        for cell in self.values.keys():
            if cell not in self.cell_types:
                self.cell_types[cell] = CellType.REGULAR

@dataclass
class MorseComplex:
    """The Morse complex derived from a discrete Morse function"""
    critical_cells: Dict[int, List[Tuple]]
    morse_boundaries: Dict[Tuple, List[Tuple]]
    morse_homology: Dict[int, int]
    gradient_paths: Dict[Tuple, List[List[Tuple]]]

class SHEMorseTheory:
    """
    Finite Morse Theory implementation for SHE simplicial complexes
    """
    
    def __init__(self, complex: 'SHESimplicialComplex', config: Optional['SHEConfig'] = None):
        self.complex = complex
        self.config = config
        self.morse_functions: Dict[str, MorseFunction] = {}
        
    def random_morse_function(self, name: str = "random", 
                            noise_level: float = 0.1,
                            seed: Optional[int] = None) -> MorseFunction:
        """Generate a random discrete Morse function"""
        if seed is not None:
            np.random.seed(seed)
            
        morse_func = MorseFunction()
        
        # Assign random values to all cells
        all_cells = []
        for dim in range(self.complex.complex.dim + 1):
            for cell in self.complex.complex.skeleton(dim):
                cell_tuple = tuple(sorted(cell)) if hasattr(cell, '__iter__') else (cell,)
                all_cells.append((cell_tuple, dim))
                morse_func.values[cell_tuple] = np.random.random() + noise_level * np.random.randn()
        
        # Ensure Morse function property: values increase along faces
        self._make_morse_function_valid(morse_func)
        
        self.morse_functions[name] = morse_func
        return morse_func
    
    def height_morse_function(self, name: str = "height", 
                            coordinate: int = 0,
                            noise_level: float = 0.01) -> MorseFunction:
        """Generate Morse function based on coordinate height with small perturbation"""
        morse_func = MorseFunction()
        
        # Get node positions (assuming they exist in metadata or features)
        node_positions = {}
        for node in self.complex.complex.nodes:
            if node in self.complex.node_features and 'position' in self.complex.node_features[node]:
                pos = self.complex.node_features[node]['position']
                node_positions[node] = pos[coordinate] if len(pos) > coordinate else 0.0
            else:
                # Default to random position if not available
                node_positions[node] = np.random.random()
        
        # Assign values to all cells based on average coordinate of vertices
        for dim in range(self.complex.complex.dim + 1):
            for cell in self.complex.complex.skeleton(dim):
                cell_tuple = tuple(sorted(cell)) if hasattr(cell, '__iter__') else (cell,)
                
                if dim == 0:
                    value = node_positions.get(cell_tuple[0], 0.0)
                else:
                    # Average position of vertices in the cell
                    vertices = list(cell_tuple)
                    avg_pos = np.mean([node_positions.get(v, 0.0) for v in vertices])
                    value = avg_pos
                
                # Add small noise to avoid degeneracies
                morse_func.values[cell_tuple] = value + noise_level * np.random.randn()
        
        self._make_morse_function_valid(morse_func)
        self.morse_functions[name] = morse_func
        return morse_func
    
    def custom_morse_function(self, name: str, value_function: Callable) -> MorseFunction:
        """Create Morse function from custom value function"""
        morse_func = MorseFunction()
        
        for dim in range(self.complex.complex.dim + 1):
            for cell in self.complex.complex.skeleton(dim):
                cell_tuple = tuple(sorted(cell)) if hasattr(cell, '__iter__') else (cell,)
                morse_func.values[cell_tuple] = value_function(cell_tuple, dim)
        
        self._make_morse_function_valid(morse_func)
        self.morse_functions[name] = morse_func
        return morse_func
    
    def _make_morse_function_valid(self, morse_func: MorseFunction):
        """Ensure function satisfies discrete Morse function properties"""
        # Sort cells by dimension and value
        cells_by_dim = defaultdict(list)
        for cell, value in morse_func.values.items():
            dim = len(cell) - 1
            cells_by_dim[dim].append((cell, value))
        
        # Sort by value within each dimension
        for dim in cells_by_dim:
            cells_by_dim[dim].sort(key=lambda x: x[1])
        
        # Create gradient vector field by pairing cells
        self._create_gradient_pairs(morse_func, cells_by_dim)
    
    def _create_gradient_pairs(self, morse_func: MorseFunction, cells_by_dim: Dict):
        """Create discrete gradient vector field"""
        paired_cells = set()
        
        # Process dimensions from low to high
        for dim in sorted(cells_by_dim.keys()):
            if dim == 0:
                continue  # No pairing for 0-cells in this simple approach
                
            dim_cells = [cell for cell, _ in cells_by_dim[dim] if cell not in paired_cells]
            
            for cell in dim_cells:
                if cell in paired_cells:
                    continue
                    
                # Find boundary faces that could be paired
                boundary_faces = self._get_boundary_cells(cell)
                unpaired_faces = [face for face in boundary_faces 
                                if face not in paired_cells and 
                                   morse_func.values[face] < morse_func.values[cell]]
                
                if unpaired_faces:
                    # Choose face with highest value (but still less than cell)
                    best_face = max(unpaired_faces, key=lambda f: morse_func.values[f])
                    
                    # Create gradient pair
                    morse_func.gradient_pairs.add((best_face, cell))
                    paired_cells.add(best_face)
                    paired_cells.add(cell)
        
        # Classify remaining unpaired cells as critical
        for cell in morse_func.values.keys():
            if cell not in paired_cells:
                dim = len(cell) - 1
                morse_func.critical_cells[dim].add(cell)
    
    def _get_boundary_cells(self, cell: Tuple) -> List[Tuple]:
        """Get boundary cells of a given cell"""
        if len(cell) <= 1:
            return []
        
        boundary = []
        for i in range(len(cell)):
            face = tuple(cell[:i] + cell[i+1:])
            boundary.append(face)
        
        return boundary
    
    def _get_coboundary_cells(self, cell: Tuple) -> List[Tuple]:
        """Get coboundary cells of a given cell"""
        coboundary = []
        dim = len(cell) - 1
        
        # Look for cells of dimension dim+1 that contain this cell
        if dim + 1 <= self.complex.complex.dim:
            for higher_cell in self.complex.complex.skeleton(dim + 1):
                higher_tuple = tuple(sorted(higher_cell)) if hasattr(higher_cell, '__iter__') else (higher_cell,)
                if set(cell).issubset(set(higher_tuple)):
                    coboundary.append(higher_tuple)
        
        return coboundary
    
    def compute_morse_complex(self, morse_function_name: str) -> MorseComplex:
        """Compute the Morse complex from a discrete Morse function"""
        if morse_function_name not in self.morse_functions:
            raise ValueError(f"Morse function {morse_function_name} not found")
        
        morse_func = self.morse_functions[morse_function_name]
        
        # Extract critical cells by dimension
        critical_cells = {}
        for dim, cells in morse_func.critical_cells.items():
            critical_cells[dim] = list(cells)
        
        # Compute gradient paths between critical cells
        gradient_paths = self._compute_gradient_paths(morse_func)
        
        # Compute Morse boundary operator
        morse_boundaries = self._compute_morse_boundaries(morse_func, gradient_paths)
        
        # Compute Morse homology
        morse_homology = self._compute_morse_homology(critical_cells, morse_boundaries)
        
        return MorseComplex(
            critical_cells=critical_cells,
            morse_boundaries=morse_boundaries,
            morse_homology=morse_homology,
            gradient_paths=gradient_paths
        )
    
    def _compute_gradient_paths(self, morse_func: MorseFunction) -> Dict[Tuple, List[List[Tuple]]]:
        """Compute gradient paths between critical cells"""
        gradient_paths = {}
        
        # Create adjacency structure from gradient pairs
        gradient_graph = defaultdict(list)
        reverse_gradient = defaultdict(list)
        
        for lower, higher in morse_func.gradient_pairs:
            gradient_graph[higher].append(lower)
            reverse_gradient[lower].append(higher)
        
        # For each critical cell, find paths to other critical cells
        for dim in morse_func.critical_cells:
            for crit_cell in morse_func.critical_cells[dim]:
                paths = self._find_paths_from_critical_cell(crit_cell, morse_func, gradient_graph)
                gradient_paths[crit_cell] = paths
        
        return gradient_paths
    
    def _find_paths_from_critical_cell(self, start_cell: Tuple, 
                                     morse_func: MorseFunction,
                                     gradient_graph: Dict) -> List[List[Tuple]]:
        """Find all gradient paths starting from a critical cell"""
        paths = []
        queue = deque([(start_cell, [start_cell])])
        visited = set()
        
        while queue:
            current_cell, path = queue.popleft()
            
            if current_cell in visited:
                continue
            visited.add(current_cell)
            
            # Check if we reached another critical cell
            if current_cell != start_cell and current_cell in morse_func.cell_types:
                if morse_func.cell_types[current_cell] == CellType.CRITICAL:
                    paths.append(path)
                    continue
            
            # Follow gradient flow
            for next_cell in gradient_graph.get(current_cell, []):
                if next_cell not in visited:
                    queue.append((next_cell, path + [next_cell]))
        
        return paths
    
    def _compute_morse_boundaries(self, morse_func: MorseFunction, 
                                gradient_paths: Dict) -> Dict[Tuple, List[Tuple]]:
        """Compute Morse boundary operators"""
        morse_boundaries = {}
        
        # For each critical cell, determine which lower-dimensional critical cells
        # are in its Morse boundary
        for dim in sorted(morse_func.critical_cells.keys(), reverse=True):
            for crit_cell in morse_func.critical_cells[dim]:
                boundary = []
                
                if dim > 0:
                    # Find paths to critical cells of dimension dim-1
                    for path_list in gradient_paths.get(crit_cell, []):
                        for path in path_list:
                            end_cell = path[-1]
                            end_dim = len(end_cell) - 1
                            if (end_dim == dim - 1 and 
                                end_cell in morse_func.critical_cells[end_dim]):
                                boundary.append(end_cell)
                
                morse_boundaries[crit_cell] = boundary
        
        return morse_boundaries
    
    def _compute_morse_homology(self, critical_cells: Dict, 
                              morse_boundaries: Dict) -> Dict[int, int]:
        """Compute Morse homology (Betti numbers)"""
        morse_homology = {}
        
        for dim in critical_cells.keys():
            # Number of critical cells of dimension dim
            num_critical = len(critical_cells[dim])
            
            # Number of boundaries from dimension dim+1
            num_boundaries = 0
            if dim + 1 in critical_cells:
                for higher_cell in critical_cells[dim + 1]:
                    boundary = morse_boundaries.get(higher_cell, [])
                    num_boundaries += len([b for b in boundary if len(b) - 1 == dim])
            
            # Morse homology rank (simplified calculation)
            # In practice, this requires computing the rank of boundary matrices
            morse_homology[dim] = max(0, num_critical - num_boundaries)
        
        return morse_homology
    
    def visualize_morse_function(self, morse_function_name: str) -> Dict[str, Any]:
        """Create visualization data for Morse function"""
        if morse_function_name not in self.morse_functions:
            raise ValueError(f"Morse function {morse_function_name} not found")
        
        morse_func = self.morse_functions[morse_function_name]
        
        viz_data = {
            "function_values": dict(morse_func.values),
            "gradient_pairs": list(morse_func.gradient_pairs),
            "critical_cells": {dim: list(cells) for dim, cells in morse_func.critical_cells.items()},
            "cell_types": {str(cell): cell_type.value for cell, cell_type in morse_func.cell_types.items()},
            "statistics": {
                "num_critical_cells": sum(len(cells) for cells in morse_func.critical_cells.values()),
                "num_gradient_pairs": len(morse_func.gradient_pairs),
                "morse_betti_numbers": {}
            }
        }
        
        return viz_data
    
    def compare_morse_functions(self, name1: str, name2: str) -> Dict[str, Any]:
        """Compare two Morse functions"""
        if name1 not in self.morse_functions or name2 not in self.morse_functions:
            raise ValueError("One or both Morse functions not found")
        
        func1 = self.morse_functions[name1]
        func2 = self.morse_functions[name2]
        
        comparison = {
            "critical_cells_comparison": {},
            "gradient_pairs_comparison": {
                "common_pairs": len(func1.gradient_pairs & func2.gradient_pairs),
                "unique_to_func1": len(func1.gradient_pairs - func2.gradient_pairs),
                "unique_to_func2": len(func2.gradient_pairs - func1.gradient_pairs)
            },
            "value_correlation": self._compute_value_correlation(func1, func2)
        }
        
        # Compare critical cells by dimension
        all_dims = set(func1.critical_cells.keys()) | set(func2.critical_cells.keys())
        for dim in all_dims:
            cells1 = func1.critical_cells.get(dim, set())
            cells2 = func2.critical_cells.get(dim, set())
            comparison["critical_cells_comparison"][dim] = {
                "common": len(cells1 & cells2),
                "unique_to_func1": len(cells1 - cells2),
                "unique_to_func2": len(cells2 - cells1)
            }
        
        return comparison
    
    def _compute_value_correlation(self, func1: MorseFunction, func2: MorseFunction) -> float:
        """Compute correlation between function values"""
        common_cells = set(func1.values.keys()) & set(func2.values.keys())
        if len(common_cells) < 2:
            return 0.0
        
        values1 = [func1.values[cell] for cell in common_cells]
        values2 = [func2.values[cell] for cell in common_cells]
        
        return np.corrcoef(values1, values2)[0, 1] if len(values1) > 1 else 0.0

class SHEMorseAnalyzer:
    """
    High-level analyzer for Morse theory on SHE complexes
    """
    
    def __init__(self, she_engine: 'SHEEngine'):
        self.she_engine = she_engine
        self.morse_theories: Dict[str, SHEMorseTheory] = {}
    
    def create_morse_theory(self, complex_name: str) -> SHEMorseTheory:
        """Create Morse theory analyzer for a complex"""
        if complex_name not in self.she_engine.complexes:
            raise ValueError(f"Complex {complex_name} not found in SHE engine")
        
        complex = self.she_engine.complexes[complex_name]
        morse_theory = SHEMorseTheory(complex, self.she_engine.config)
        self.morse_theories[complex_name] = morse_theory
        
        return morse_theory
    
    def batch_morse_analysis(self, complex_names: List[str], 
                           function_types: List[str] = ["random", "height"]) -> Dict[str, Any]:
        """Perform batch Morse analysis on multiple complexes"""
        results = {}
        
        for complex_name in complex_names:
            try:
                morse_theory = self.create_morse_theory(complex_name)
                complex_results = {}
                
                for func_type in function_types:
                    if func_type == "random":
                        morse_func = morse_theory.random_morse_function(f"{complex_name}_random")
                    elif func_type == "height":
                        morse_func = morse_theory.height_morse_function(f"{complex_name}_height")
                    else:
                        continue
                    
                    morse_complex = morse_theory.compute_morse_complex(f"{complex_name}_{func_type}")
                    viz_data = morse_theory.visualize_morse_function(f"{complex_name}_{func_type}")
                    
                    complex_results[func_type] = {
                        "morse_complex": morse_complex,
                        "visualization": viz_data
                    }
                
                results[complex_name] = complex_results
                
            except Exception as e:
                logger.error(f"Error in Morse analysis for {complex_name}: {e}")
                results[complex_name] = {"error": str(e)}
        
        return results
    
    def compare_homologies(self, complex_name: str) -> Dict[str, Any]:
        """Compare classical and Morse homologies"""
        if complex_name not in self.morse_theories:
            raise ValueError(f"Morse theory for {complex_name} not found")
        
        morse_theory = self.morse_theories[complex_name]
        
        # Get classical homology (Betti numbers) from the complex
        classical_betti = morse_theory.complex.compute_betti_numbers()
        
        # Get Morse homology from different Morse functions
        morse_homologies = {}
        for func_name in morse_theory.morse_functions.keys():
            morse_complex = morse_theory.compute_morse_complex(func_name)
            morse_homologies[func_name] = morse_complex.morse_homology
        
        return {
            "classical_betti_numbers": classical_betti,
            "morse_homologies": morse_homologies,
            "comparison": {
                func_name: {
                    dim: f"Classical: {classical_betti.get(dim, 0)}, Morse: {homol.get(dim, 0)}"
                    for dim in set(classical_betti.keys()) | set(homol.keys())
                }
                for func_name, homol in morse_homologies.items()
            }
        }

# Example usage and demonstration
def demo_morse_theory():
    """Demonstrate Morse theory functionality"""
    print("SHE Finite Morse Theory Demo")
    print("=" * 40)
    
    # This would typically use the actual SHE classes
    # For demo purposes, we'll create a mock setup
    
    try:
        # Create a simple triangular complex for demonstration
        print("Creating demonstration complex...")
        
        # In practice, you would:
        # 1. Load or create a SHESimplicialComplex
        # 2. Register it with SHEEngine
        # 3. Create SHEMorseAnalyzer
        # 4. Perform Morse analysis
        
        print("Morse theory extension loaded successfully!")
        print("\nKey features:")
        print("- Random and height-based Morse functions")
        print("- Discrete gradient vector fields")
        print("- Critical cell identification")
        print("- Morse complex computation")
        print("- Morse homology calculation")
        print("- Visualization and comparison tools")
        
    except Exception as e:
        print(f"Demo error: {e}")

if __name__ == "__main__":
    demo_morse_theory()
