"""
Physics Visualization Component

This module provides interactive 3D visualization capabilities for physics data,
including phase diagrams, energy landscapes, molecular structures, and quantum states.
"""

import json
import time
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class VisualizationType(Enum):
    """Types of physics visualizations."""
    PHASE_DIAGRAM = "phase_diagram"
    ENERGY_LANDSCAPE = "energy_landscape"
    MOLECULAR_STRUCTURE = "molecular_structure"
    QUANTUM_STATE = "quantum_state"
    WAVE_FUNCTION = "wave_function"
    PARTICLE_TRAJECTORY = "particle_trajectory"
    FIELD_VISUALIZATION = "field_visualization"
    STATISTICAL_DISTRIBUTION = "statistical_distribution"
    TIME_SERIES = "time_series"
    THREE_D_PLOT = "3d_plot"

@dataclass
class VisualizationConfig:
    """Configuration for physics visualizations."""
    width: int = 800
    height: int = 600
    color_scheme: str = "viridis"
    interactive: bool = True
    animation: bool = False
    export_format: str = "png"
    quality: str = "high"  # low, medium, high, ultra
    background_color: str = "#ffffff"
    grid: bool = True
    axes_labels: bool = True

@dataclass
class DataPoint:
    """Single data point for visualization."""
    x: float
    y: float
    z: Optional[float] = None
    value: Optional[float] = None
    uncertainty: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class PhysicsVisualization:
    """
    Physics Data Visualization System
    
    Provides advanced visualization capabilities for physics research data including
    3D plots, interactive diagrams, molecular structures, and quantum state representations.
    """
    
    def __init__(self):
        self.visualizations: Dict[str, Dict[str, Any]] = {}
        self.default_config = VisualizationConfig()
        self.supported_types = list(VisualizationType)
        
    def create_phase_diagram(self, data: List[DataPoint], 
                           temperature_range: Tuple[float, float],
                           pressure_range: Tuple[float, float],
                           config: Optional[VisualizationConfig] = None) -> str:
        """Create a phase diagram visualization."""
        try:
            viz_id = f"phase_diagram_{int(time.time() * 1000)}"
            config = config or self.default_config
            
            # Process data for phase diagram
            phases = self._identify_phases(data, temperature_range, pressure_range)
            
            visualization = {
                'id': viz_id,
                'type': VisualizationType.PHASE_DIAGRAM.value,
                'data': {
                    'points': [self._datapoint_to_dict(point) for point in data],
                    'phases': phases,
                    'temperature_range': temperature_range,
                    'pressure_range': pressure_range
                },
                'config': self._config_to_dict(config),
                'metadata': {
                    'created_at': time.time(),
                    'data_points': len(data),
                    'phase_count': len(phases)
                },
                'render_data': self._generate_phase_diagram_data(data, phases, config)
            }
            
            self.visualizations[viz_id] = visualization
            logger.info(f"Created phase diagram visualization: {viz_id}")
            
            return viz_id
            
        except Exception as e:
            logger.error(f"Error creating phase diagram: {e}")
            raise
    
    def create_energy_landscape(self, potential_function: callable,
                              x_range: Tuple[float, float],
                              y_range: Tuple[float, float],
                              resolution: int = 100,
                              config: Optional[VisualizationConfig] = None) -> str:
        """Create an energy landscape visualization."""
        try:
            viz_id = f"energy_landscape_{int(time.time() * 1000)}"
            config = config or self.default_config
            
            # Generate energy landscape data
            x = np.linspace(x_range[0], x_range[1], resolution)
            y = np.linspace(y_range[0], y_range[1], resolution)
            X, Y = np.meshgrid(x, y)
            
            # Calculate potential energy at each point
            Z = np.zeros_like(X)
            for i in range(resolution):
                for j in range(resolution):
                    try:
                        Z[i, j] = potential_function(X[i, j], Y[i, j])
                    except:
                        Z[i, j] = float('inf')  # Handle invalid regions
            
            # Find critical points (minima, maxima, saddle points)
            critical_points = self._find_critical_points(X, Y, Z)
            
            visualization = {
                'id': viz_id,
                'type': VisualizationType.ENERGY_LANDSCAPE.value,
                'data': {
                    'x_grid': X.tolist(),
                    'y_grid': Y.tolist(),
                    'energy_grid': Z.tolist(),
                    'x_range': x_range,
                    'y_range': y_range,
                    'resolution': resolution,
                    'critical_points': critical_points
                },
                'config': self._config_to_dict(config),
                'metadata': {
                    'created_at': time.time(),
                    'min_energy': float(np.min(Z[np.isfinite(Z)])),
                    'max_energy': float(np.max(Z[np.isfinite(Z)])),
                    'critical_points_count': len(critical_points)
                },
                'render_data': self._generate_energy_landscape_data(X, Y, Z, config)
            }
            
            self.visualizations[viz_id] = visualization
            logger.info(f"Created energy landscape visualization: {viz_id}")
            
            return viz_id
            
        except Exception as e:
            logger.error(f"Error creating energy landscape: {e}")
            raise
    
    def create_molecular_structure(self, atoms: List[Dict[str, Any]],
                                 bonds: List[Tuple[int, int]],
                                 config: Optional[VisualizationConfig] = None) -> str:
        """Create a molecular structure visualization."""
        try:
            viz_id = f"molecular_structure_{int(time.time() * 1000)}"
            config = config or self.default_config
            
            # Process molecular data
            atom_data = []
            for i, atom in enumerate(atoms):
                atom_info = {
                    'id': i,
                    'element': atom.get('element', 'C'),
                    'position': atom.get('position', [0, 0, 0]),
                    'charge': atom.get('charge', 0),
                    'radius': self._get_atomic_radius(atom.get('element', 'C')),
                    'color': self._get_atomic_color(atom.get('element', 'C'))
                }
                atom_data.append(atom_info)
            
            # Process bond data
            bond_data = []
            for bond in bonds:
                if len(bond) >= 2:
                    bond_info = {
                        'atom1': bond[0],
                        'atom2': bond[1],
                        'order': bond[2] if len(bond) > 2 else 1,
                        'length': self._calculate_bond_length(
                            atoms[bond[0]]['position'], 
                            atoms[bond[1]]['position']
                        )
                    }
                    bond_data.append(bond_info)
            
            # Calculate molecular properties
            molecular_properties = self._calculate_molecular_properties(atom_data, bond_data)
            
            visualization = {
                'id': viz_id,
                'type': VisualizationType.MOLECULAR_STRUCTURE.value,
                'data': {
                    'atoms': atom_data,
                    'bonds': bond_data,
                    'properties': molecular_properties
                },
                'config': self._config_to_dict(config),
                'metadata': {
                    'created_at': time.time(),
                    'atom_count': len(atom_data),
                    'bond_count': len(bond_data),
                    'molecular_formula': molecular_properties.get('formula', 'Unknown')
                },
                'render_data': self._generate_molecular_structure_data(atom_data, bond_data, config)
            }
            
            self.visualizations[viz_id] = visualization
            logger.info(f"Created molecular structure visualization: {viz_id}")
            
            return viz_id
            
        except Exception as e:
            logger.error(f"Error creating molecular structure: {e}")
            raise
    
    def create_quantum_state_visualization(self, state_vector: np.ndarray,
                                         basis_labels: List[str],
                                         config: Optional[VisualizationConfig] = None) -> str:
        """Create a quantum state visualization."""
        try:
            viz_id = f"quantum_state_{int(time.time() * 1000)}"
            config = config or self.default_config
            
            # Calculate quantum state properties
            probabilities = np.abs(state_vector) ** 2
            phases = np.angle(state_vector)
            
            # Calculate entanglement measures if multi-qubit system
            entanglement_data = self._calculate_entanglement_measures(state_vector)
            
            # Generate Bloch sphere data for single qubit states
            bloch_data = None
            if len(state_vector) == 2:  # Single qubit
                bloch_data = self._calculate_bloch_coordinates(state_vector)
            
            visualization = {
                'id': viz_id,
                'type': VisualizationType.QUANTUM_STATE.value,
                'data': {
                    'state_vector': {
                        'real': state_vector.real.tolist(),
                        'imag': state_vector.imag.tolist()
                    },
                    'probabilities': probabilities.tolist(),
                    'phases': phases.tolist(),
                    'basis_labels': basis_labels,
                    'entanglement': entanglement_data,
                    'bloch_sphere': bloch_data
                },
                'config': self._config_to_dict(config),
                'metadata': {
                    'created_at': time.time(),
                    'dimension': len(state_vector),
                    'is_normalized': abs(np.linalg.norm(state_vector) - 1.0) < 1e-10,
                    'max_probability': float(np.max(probabilities))
                },
                'render_data': self._generate_quantum_state_data(state_vector, probabilities, phases, config)
            }
            
            self.visualizations[viz_id] = visualization
            logger.info(f"Created quantum state visualization: {viz_id}")
            
            return viz_id
            
        except Exception as e:
            logger.error(f"Error creating quantum state visualization: {e}")
            raise
    
    def create_field_visualization(self, field_data: np.ndarray,
                                 field_type: str = "vector",
                                 coordinates: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
                                 config: Optional[VisualizationConfig] = None) -> str:
        """Create a field visualization (vector or scalar)."""
        try:
            viz_id = f"field_{field_type}_{int(time.time() * 1000)}"
            config = config or self.default_config
            
            # Process field data
            if coordinates is None:
                # Generate default coordinate system
                shape = field_data.shape[:3] if len(field_data.shape) >= 3 else field_data.shape[:2] + (1,)
                x = np.linspace(-1, 1, shape[0])
                y = np.linspace(-1, 1, shape[1])
                z = np.linspace(-1, 1, shape[2]) if len(shape) > 2 else np.array([0])
                coordinates = np.meshgrid(x, y, z)
            
            # Calculate field properties
            field_properties = self._analyze_field_properties(field_data, field_type)
            
            visualization = {
                'id': viz_id,
                'type': VisualizationType.FIELD_VISUALIZATION.value,
                'data': {
                    'field_data': field_data.tolist(),
                    'field_type': field_type,
                    'coordinates': [coord.tolist() for coord in coordinates],
                    'properties': field_properties
                },
                'config': self._config_to_dict(config),
                'metadata': {
                    'created_at': time.time(),
                    'field_shape': field_data.shape,
                    'field_type': field_type,
                    'magnitude_range': field_properties.get('magnitude_range', [0, 1])
                },
                'render_data': self._generate_field_visualization_data(field_data, coordinates, field_type, config)
            }
            
            self.visualizations[viz_id] = visualization
            logger.info(f"Created field visualization: {viz_id}")
            
            return viz_id
            
        except Exception as e:
            logger.error(f"Error creating field visualization: {e}")
            raise
    
    def get_visualization(self, viz_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific visualization by ID."""
        return self.visualizations.get(viz_id)
    
    def list_visualizations(self, viz_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all visualizations, optionally filtered by type."""
        visualizations = []
        
        for viz in self.visualizations.values():
            if viz_type is None or viz['type'] == viz_type:
                # Return summary without large render data
                summary = {
                    'id': viz['id'],
                    'type': viz['type'],
                    'metadata': viz['metadata'],
                    'config': viz['config']
                }
                visualizations.append(summary)
        
        return visualizations
    
    def update_visualization_config(self, viz_id: str, new_config: Dict[str, Any]) -> bool:
        """Update visualization configuration and regenerate render data."""
        try:
            if viz_id not in self.visualizations:
                return False
            
            viz = self.visualizations[viz_id]
            
            # Update config
            for key, value in new_config.items():
                if hasattr(self.default_config, key):
                    viz['config'][key] = value
            
            # Regenerate render data based on visualization type
            viz_type = viz['type']
            if viz_type == VisualizationType.PHASE_DIAGRAM.value:
                viz['render_data'] = self._generate_phase_diagram_data(
                    viz['data']['points'], viz['data']['phases'], 
                    self._dict_to_config(viz['config'])
                )
            elif viz_type == VisualizationType.ENERGY_LANDSCAPE.value:
                viz['render_data'] = self._generate_energy_landscape_data(
                    np.array(viz['data']['x_grid']), 
                    np.array(viz['data']['y_grid']),
                    np.array(viz['data']['energy_grid']),
                    self._dict_to_config(viz['config'])
                )
            # Add other visualization types as needed
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating visualization config: {e}")
            return False
    
    def delete_visualization(self, viz_id: str) -> bool:
        """Delete a visualization."""
        try:
            if viz_id in self.visualizations:
                del self.visualizations[viz_id]
                logger.info(f"Deleted visualization: {viz_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error deleting visualization: {e}")
            return False
    
    def export_visualization(self, viz_id: str, format_type: str = "json") -> Optional[Dict[str, Any]]:
        """Export visualization data in specified format."""
        try:
            if viz_id not in self.visualizations:
                return None
            
            viz = self.visualizations[viz_id]
            
            export_data = {
                'visualization': viz,
                'export_format': format_type,
                'export_timestamp': time.time(),
                'metadata': {
                    'version': '1.0.0',
                    'generator': 'physics_visualization'
                }
            }
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting visualization: {e}")
            return None
    
    # Helper methods
    
    def _datapoint_to_dict(self, point: DataPoint) -> Dict[str, Any]:
        """Convert DataPoint to dictionary."""
        return {
            'x': point.x,
            'y': point.y,
            'z': point.z,
            'value': point.value,
            'uncertainty': point.uncertainty,
            'metadata': point.metadata
        }
    
    def _config_to_dict(self, config: VisualizationConfig) -> Dict[str, Any]:
        """Convert VisualizationConfig to dictionary."""
        return {
            'width': config.width,
            'height': config.height,
            'color_scheme': config.color_scheme,
            'interactive': config.interactive,
            'animation': config.animation,
            'export_format': config.export_format,
            'quality': config.quality,
            'background_color': config.background_color,
            'grid': config.grid,
            'axes_labels': config.axes_labels
        }
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> VisualizationConfig:
        """Convert dictionary to VisualizationConfig."""
        return VisualizationConfig(**config_dict)
    
    def _identify_phases(self, data: List[DataPoint], temp_range: Tuple[float, float], 
                        press_range: Tuple[float, float]) -> List[Dict[str, Any]]:
        """Identify phases in the data."""
        # Simplified phase identification
        phases = [
            {'name': 'solid', 'color': '#0066cc', 'region': 'low_temp_high_press'},
            {'name': 'liquid', 'color': '#3399ff', 'region': 'mid_temp_mid_press'},
            {'name': 'gas', 'color': '#99ccff', 'region': 'high_temp_low_press'}
        ]
        return phases
    
    def _find_critical_points(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> List[Dict[str, Any]]:
        """Find critical points in energy landscape."""
        critical_points = []
        # Simplified critical point detection
        try:
            # Find local minima (very basic detection)
            for i in range(1, Z.shape[0] - 1):
                for j in range(1, Z.shape[1] - 1):
                    if (Z[i, j] < Z[i-1, j] and Z[i, j] < Z[i+1, j] and 
                        Z[i, j] < Z[i, j-1] and Z[i, j] < Z[i, j+1]):
                        critical_points.append({
                            'type': 'minimum',
                            'x': float(X[i, j]),
                            'y': float(Y[i, j]),
                            'energy': float(Z[i, j])
                        })
        except:
            pass
        return critical_points
    
    def _get_atomic_radius(self, element: str) -> float:
        """Get atomic radius for element."""
        radii = {'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'P': 1.07, 'S': 1.05}
        return radii.get(element, 1.0)
    
    def _get_atomic_color(self, element: str) -> str:
        """Get color for atomic element."""
        colors = {'H': '#ffffff', 'C': '#909090', 'N': '#3050f8', 'O': '#ff0d0d', 'P': '#ff8000', 'S': '#ffff30'}
        return colors.get(element, '#ff1493')
    
    def _calculate_bond_length(self, pos1: List[float], pos2: List[float]) -> float:
        """Calculate distance between two positions."""
        return float(np.linalg.norm(np.array(pos1) - np.array(pos2)))
    
    def _calculate_molecular_properties(self, atoms: List[Dict], bonds: List[Dict]) -> Dict[str, Any]:
        """Calculate basic molecular properties."""
        # Count elements
        element_count = {}
        for atom in atoms:
            element = atom['element']
            element_count[element] = element_count.get(element, 0) + 1
        
        # Generate molecular formula
        formula = ''.join([f"{elem}{count if count > 1 else ''}" 
                          for elem, count in sorted(element_count.items())])
        
        return {
            'formula': formula,
            'atom_count': len(atoms),
            'bond_count': len(bonds),
            'elements': list(element_count.keys()),
            'molecular_weight': sum([self._get_atomic_weight(atom['element']) for atom in atoms])
        }
    
    def _get_atomic_weight(self, element: str) -> float:
        """Get atomic weight for element."""
        weights = {'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'P': 30.974, 'S': 32.065}
        return weights.get(element, 12.0)
    
    def _calculate_entanglement_measures(self, state_vector: np.ndarray) -> Dict[str, Any]:
        """Calculate quantum entanglement measures."""
        # Simplified entanglement calculation
        return {
            'entanglement_entropy': 0.0,
            'concurrence': 0.0,
            'schmidt_rank': 1
        }
    
    def _calculate_bloch_coordinates(self, state_vector: np.ndarray) -> Dict[str, float]:
        """Calculate Bloch sphere coordinates for single qubit."""
        # For a qubit state |ψ⟩ = α|0⟩ + β|1⟩
        alpha, beta = state_vector[0], state_vector[1]
        
        # Pauli expectation values
        x = 2 * (alpha.conjugate() * beta).real
        y = 2 * (alpha.conjugate() * beta).imag
        z = abs(alpha)**2 - abs(beta)**2
        
        return {'x': float(x), 'y': float(y), 'z': float(z)}
    
    def _analyze_field_properties(self, field_data: np.ndarray, field_type: str) -> Dict[str, Any]:
        """Analyze properties of field data."""
        properties = {}
        
        if field_type == "vector":
            # Calculate magnitude
            if len(field_data.shape) >= 4:  # Vector field
                magnitudes = np.linalg.norm(field_data, axis=-1)
                properties['magnitude_range'] = [float(np.min(magnitudes)), float(np.max(magnitudes))]
                properties['mean_magnitude'] = float(np.mean(magnitudes))
        else:  # scalar field
            properties['value_range'] = [float(np.min(field_data)), float(np.max(field_data))]
            properties['mean_value'] = float(np.mean(field_data))
        
        return properties
    
    def _generate_phase_diagram_data(self, points: List[Dict], phases: List[Dict], 
                                   config: VisualizationConfig) -> Dict[str, Any]:
        """Generate render data for phase diagram."""
        return {
            'plot_type': 'phase_diagram',
            'traces': phases,
            'layout': {
                'width': config.width,
                'height': config.height,
                'title': 'Phase Diagram',
                'xaxis': {'title': 'Temperature (K)'},
                'yaxis': {'title': 'Pressure (Pa)'}
            }
        }
    
    def _generate_energy_landscape_data(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                                      config: VisualizationConfig) -> Dict[str, Any]:
        """Generate render data for energy landscape."""
        return {
            'plot_type': 'surface',
            'x': X.tolist(),
            'y': Y.tolist(),
            'z': Z.tolist(),
            'layout': {
                'width': config.width,
                'height': config.height,
                'title': 'Energy Landscape',
                'scene': {
                    'xaxis': {'title': 'Position X'},
                    'yaxis': {'title': 'Position Y'},
                    'zaxis': {'title': 'Energy'}
                }
            }
        }
    
    def _generate_molecular_structure_data(self, atoms: List[Dict], bonds: List[Dict],
                                         config: VisualizationConfig) -> Dict[str, Any]:
        """Generate render data for molecular structure."""
        return {
            'plot_type': 'molecular',
            'atoms': atoms,
            'bonds': bonds,
            'layout': {
                'width': config.width,
                'height': config.height,
                'title': 'Molecular Structure',
                'scene': {
                    'xaxis': {'title': 'X (Å)'},
                    'yaxis': {'title': 'Y (Å)'},
                    'zaxis': {'title': 'Z (Å)'}
                }
            }
        }
    
    def _generate_quantum_state_data(self, state_vector: np.ndarray, probabilities: np.ndarray,
                                   phases: np.ndarray, config: VisualizationConfig) -> Dict[str, Any]:
        """Generate render data for quantum state."""
        return {
            'plot_type': 'quantum_state',
            'probabilities': probabilities.tolist(),
            'phases': phases.tolist(),
            'layout': {
                'width': config.width,
                'height': config.height,
                'title': 'Quantum State Visualization'
            }
        }
    
    def _generate_field_visualization_data(self, field_data: np.ndarray, coordinates: Tuple,
                                         field_type: str, config: VisualizationConfig) -> Dict[str, Any]:
        """Generate render data for field visualization."""
        return {
            'plot_type': f'field_{field_type}',
            'field_data': field_data.tolist(),
            'coordinates': [coord.tolist() for coord in coordinates],
            'layout': {
                'width': config.width,
                'height': config.height,
                'title': f'{field_type.title()} Field Visualization'
            }
        }

# Global visualization instance
physics_visualization = PhysicsVisualization()