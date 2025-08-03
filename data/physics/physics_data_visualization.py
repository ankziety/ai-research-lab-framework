"""
Physics Data Visualization for AI Research Lab Framework.

Provides specialized visualization capabilities for physics data including
2D/3D scientific plots, interactive visualizations, and physics-specific chart types.
"""

import logging
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass

# Try to import visualization dependencies
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    patches = None

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    sns = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    px = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

logger = logging.getLogger(__name__)

@dataclass
class PlotConfig:
    """Configuration for physics plots."""
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    zlabel: str = ""
    width: int = 800
    height: int = 600
    dpi: int = 100
    style: str = "default"
    color_scheme: str = "viridis"
    grid: bool = True
    legend: bool = True
    save_format: str = "png"
    interactive: bool = False

class PhysicsDataVisualization:
    """Provides physics-specific data visualization capabilities."""
    
    def __init__(self, config: Dict[str, Any], output_dir: Optional[str] = None):
        """
        Initialize the Physics Data Visualization system.
        
        Args:
            config: Configuration dictionary
            output_dir: Directory for saving visualizations
        """
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else Path.home() / ".ai_research_lab" / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default plot configuration
        self.default_config = PlotConfig()
        
        # Initialize matplotlib style if available
        if MATPLOTLIB_AVAILABLE:
            self._setup_matplotlib_style()
        
        # Check available backends
        self.backends = self._check_available_backends()
        
        logger.info(f"PhysicsDataVisualization initialized with backends: {list(self.backends.keys())}")
    
    def _check_available_backends(self) -> Dict[str, bool]:
        """Check which visualization backends are available."""
        return {
            'matplotlib': MATPLOTLIB_AVAILABLE,
            'seaborn': SEABORN_AVAILABLE,
            'plotly': PLOTLY_AVAILABLE
        }
    
    def _setup_matplotlib_style(self) -> None:
        """Setup matplotlib style for physics plots."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        # Set up LaTeX-style fonts for scientific notation
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'lines.linewidth': 2,
            'lines.markersize': 6,
            'grid.alpha': 0.3,
            'axes.grid': True
        })
    
    def visualize_physics_data(self, data: Dict[str, Any], plot_type: str, **kwargs) -> Optional[str]:
        """
        Create physics data visualization.
        
        Args:
            data: Physics data to visualize
            plot_type: Type of plot to create
            **kwargs: Additional plotting parameters
        
        Returns:
            Path to saved visualization or None if failed
        """
        try:
            # Merge configuration
            plot_config = PlotConfig(**{**self.default_config.__dict__, **kwargs})
            
            # Choose backend
            backend = kwargs.get('backend', 'matplotlib')
            if backend not in self.backends or not self.backends[backend]:
                logger.warning(f"Backend {backend} not available, using matplotlib")
                backend = 'matplotlib'
            
            # Create visualization based on plot type
            if plot_type in ['line', 'scatter', 'bar', 'histogram']:
                return self._create_basic_plot(data, plot_type, plot_config, backend)
            elif plot_type in ['phase_diagram', 'phase_space']:
                return self._create_phase_diagram(data, plot_config, backend)
            elif plot_type in ['contour', 'heatmap', 'field']:
                return self._create_field_plot(data, plot_config, backend)
            elif plot_type in ['trajectory', 'path']:
                return self._create_trajectory_plot(data, plot_config, backend)
            elif plot_type in ['energy_levels', 'spectrum']:
                return self._create_energy_diagram(data, plot_config, backend)
            elif plot_type in ['crystal_structure', '3d_structure']:
                return self._create_3d_structure(data, plot_config, backend)
            elif plot_type in ['distribution', 'probability']:
                return self._create_distribution_plot(data, plot_config, backend)
            elif plot_type in ['vector_field', 'field_lines']:
                return self._create_vector_field(data, plot_config, backend)
            elif plot_type in ['wave', 'oscillation']:
                return self._create_wave_plot(data, plot_config, backend)
            elif plot_type in ['correlation', 'matrix']:
                return self._create_correlation_plot(data, plot_config, backend)
            else:
                logger.error(f"Unknown plot type: {plot_type}")
                return None
        
        except Exception as e:
            logger.error(f"Failed to create visualization: {str(e)}")
            return None
    
    def _create_basic_plot(self, data: Dict[str, Any], plot_type: str, config: PlotConfig, backend: str) -> Optional[str]:
        """Create basic plots (line, scatter, bar, histogram)."""
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib required for basic plots")
            return None
        
        fig, ax = plt.subplots(figsize=(config.width/100, config.height/100), dpi=config.dpi)
        
        try:
            if plot_type == 'line':
                x = data.get('x', range(len(data.get('y', []))))
                y = data.get('y', [])
                ax.plot(x, y, linewidth=plt.rcParams.get('lines.linewidth', 2))
                
            elif plot_type == 'scatter':
                x = data.get('x', [])
                y = data.get('y', [])
                c = data.get('color', None)
                s = data.get('size', 50)
                ax.scatter(x, y, c=c, s=s, alpha=0.7, cmap=config.color_scheme)
                
            elif plot_type == 'bar':
                x = data.get('x', [])
                y = data.get('y', [])
                ax.bar(x, y, alpha=0.8)
                
            elif plot_type == 'histogram':
                values = data.get('values', [])
                bins = data.get('bins', 'auto')
                ax.hist(values, bins=bins, alpha=0.7, density=True)
                ax.set_ylabel('Probability Density')
            
            # Set labels and title
            ax.set_xlabel(config.xlabel)
            ax.set_ylabel(config.ylabel)
            ax.set_title(config.title)
            
            if config.grid:
                ax.grid(True, alpha=0.3)
            
            if config.legend and 'legend' in data:
                ax.legend(data['legend'])
            
            # Save plot
            filename = f"{plot_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{config.save_format}"
            output_path = self.output_dir / filename
            plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            plt.close()
            raise e
    
    def _create_phase_diagram(self, data: Dict[str, Any], config: PlotConfig, backend: str) -> Optional[str]:
        """Create phase diagram plots."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        fig, ax = plt.subplots(figsize=(config.width/100, config.height/100), dpi=config.dpi)
        
        try:
            x = data.get('x', [])
            y = data.get('y', [])
            
            if 'regions' in data:
                # Plot phase regions
                for region in data['regions']:
                    vertices = region.get('vertices', [])
                    label = region.get('label', 'Unknown Phase')
                    color = region.get('color', 'lightblue')
                    
                    if vertices:
                        polygon = patches.Polygon(vertices, alpha=0.5, color=color, label=label)
                        ax.add_patch(polygon)
            
            # Plot phase boundaries
            if 'boundaries' in data:
                for boundary in data['boundaries']:
                    x_bound = boundary.get('x', [])
                    y_bound = boundary.get('y', [])
                    label = boundary.get('label', '')
                    ax.plot(x_bound, y_bound, 'k-', linewidth=2, label=label)
            
            # Plot data points
            if x and y:
                ax.scatter(x, y, c='red', s=50, zorder=5, label='Data Points')
            
            ax.set_xlabel(config.xlabel or 'Temperature')
            ax.set_ylabel(config.ylabel or 'Pressure')
            ax.set_title(config.title or 'Phase Diagram')
            
            if config.legend:
                ax.legend()
            
            if config.grid:
                ax.grid(True, alpha=0.3)
            
            filename = f"phase_diagram_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{config.save_format}"
            output_path = self.output_dir / filename
            plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            plt.close()
            raise e
    
    def _create_field_plot(self, data: Dict[str, Any], config: PlotConfig, backend: str) -> Optional[str]:
        """Create field plots (contour, heatmap)."""
        if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
            return None
        
        fig, ax = plt.subplots(figsize=(config.width/100, config.height/100), dpi=config.dpi)
        
        try:
            X = data.get('X', [])
            Y = data.get('Y', [])
            Z = data.get('Z', [])
            
            if not all([len(X), len(Y), len(Z)]):
                raise ValueError("X, Y, Z data required for field plots")
            
            X, Y, Z = np.array(X), np.array(Y), np.array(Z)
            
            if 'contour' in data.get('plot_type', ''):
                contour = ax.contour(X, Y, Z, levels=data.get('levels', 10))
                ax.clabel(contour, inline=True, fontsize=8)
                contourf = ax.contourf(X, Y, Z, levels=data.get('levels', 10), 
                                     cmap=config.color_scheme, alpha=0.7)
                plt.colorbar(contourf, ax=ax, label=config.zlabel)
            else:
                im = ax.imshow(Z, extent=[X.min(), X.max(), Y.min(), Y.max()],
                             cmap=config.color_scheme, origin='lower', aspect='auto')
                plt.colorbar(im, ax=ax, label=config.zlabel)
            
            ax.set_xlabel(config.xlabel)
            ax.set_ylabel(config.ylabel)
            ax.set_title(config.title)
            
            filename = f"field_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{config.save_format}"
            output_path = self.output_dir / filename
            plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            plt.close()
            raise e
    
    def _create_trajectory_plot(self, data: Dict[str, Any], config: PlotConfig, backend: str) -> Optional[str]:
        """Create trajectory plots for particle motion."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        # Determine if 2D or 3D
        is_3d = 'z' in data
        
        if is_3d:
            fig = plt.figure(figsize=(config.width/100, config.height/100), dpi=config.dpi)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots(figsize=(config.width/100, config.height/100), dpi=config.dpi)
        
        try:
            x = data.get('x', [])
            y = data.get('y', [])
            
            if is_3d:
                z = data.get('z', [])
                ax.plot(x, y, z, linewidth=2, alpha=0.8)
                
                # Mark start and end points
                if len(x) > 0:
                    ax.scatter([x[0]], [y[0]], [z[0]], c='green', s=100, label='Start')
                    ax.scatter([x[-1]], [y[-1]], [z[-1]], c='red', s=100, label='End')
                
                ax.set_zlabel(config.zlabel or 'Z')
            else:
                ax.plot(x, y, linewidth=2, alpha=0.8)
                
                # Mark start and end points
                if len(x) > 0:
                    ax.scatter([x[0]], [y[0]], c='green', s=100, label='Start')
                    ax.scatter([x[-1]], [y[-1]], c='red', s=100, label='End')
                
                # Add velocity vectors if available
                if 'vx' in data and 'vy' in data:
                    vx, vy = data['vx'], data['vy']
                    # Sample velocity vectors (every nth point)
                    step = max(1, len(x) // 20)
                    ax.quiver(x[::step], y[::step], vx[::step], vy[::step], 
                            alpha=0.6, scale=50, color='blue')
            
            ax.set_xlabel(config.xlabel or 'X')
            ax.set_ylabel(config.ylabel or 'Y')
            ax.set_title(config.title or 'Particle Trajectory')
            
            if config.legend:
                ax.legend()
            
            if config.grid:
                ax.grid(True, alpha=0.3)
            
            filename = f"trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{config.save_format}"
            output_path = self.output_dir / filename
            plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            plt.close()
            raise e
    
    def _create_energy_diagram(self, data: Dict[str, Any], config: PlotConfig, backend: str) -> Optional[str]:
        """Create energy level diagrams."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        fig, ax = plt.subplots(figsize=(config.width/100, config.height/100), dpi=config.dpi)
        
        try:
            energy_levels = data.get('energy_levels', [])
            labels = data.get('labels', [f'Level {i}' for i in range(len(energy_levels))])
            degeneracies = data.get('degeneracies', [1] * len(energy_levels))
            
            for i, (energy, label, deg) in enumerate(zip(energy_levels, labels, degeneracies)):
                # Draw energy level line
                line_width = max(1, deg)  # Line width represents degeneracy
                ax.hlines(energy, i-0.4, i+0.4, linewidth=line_width*3, color='blue')
                
                # Add label
                ax.text(i+0.5, energy, label, va='center', fontsize=10)
            
            # Add transitions if provided
            if 'transitions' in data:
                for transition in data['transitions']:
                    from_level = transition.get('from', 0)
                    to_level = transition.get('to', 1)
                    if from_level < len(energy_levels) and to_level < len(energy_levels):
                        from_energy = energy_levels[from_level]
                        to_energy = energy_levels[to_level]
                        
                        # Draw transition arrow
                        ax.annotate('', xy=(to_level, to_energy), xytext=(from_level, from_energy),
                                  arrowprops=dict(arrowstyle='->', color='red', lw=2))
            
            ax.set_xlabel('Energy Levels')
            ax.set_ylabel(config.ylabel or 'Energy')
            ax.set_title(config.title or 'Energy Level Diagram')
            ax.set_xticks(range(len(energy_levels)))
            ax.set_xticklabels([f'n={i}' for i in range(len(energy_levels))])
            
            if config.grid:
                ax.grid(True, alpha=0.3, axis='y')
            
            filename = f"energy_diagram_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{config.save_format}"
            output_path = self.output_dir / filename
            plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            plt.close()
            raise e
    
    def _create_3d_structure(self, data: Dict[str, Any], config: PlotConfig, backend: str) -> Optional[str]:
        """Create 3D structure visualizations."""
        if backend == 'plotly' and PLOTLY_AVAILABLE:
            return self._create_3d_structure_plotly(data, config)
        elif MATPLOTLIB_AVAILABLE:
            return self._create_3d_structure_matplotlib(data, config)
        else:
            logger.error("No 3D visualization backend available")
            return None
    
    def _create_3d_structure_plotly(self, data: Dict[str, Any], config: PlotConfig) -> Optional[str]:
        """Create 3D structure using Plotly."""
        try:
            atoms = data.get('atoms', [])
            bonds = data.get('bonds', [])
            
            # Create atom traces
            atom_traces = []
            for atom in atoms:
                x, y, z = atom.get('position', [0, 0, 0])
                element = atom.get('element', 'C')
                color = atom.get('color', 'blue')
                size = atom.get('size', 10)
                
                trace = go.Scatter3d(
                    x=[x], y=[y], z=[z],
                    mode='markers',
                    marker=dict(size=size, color=color),
                    name=element,
                    showlegend=False
                )
                atom_traces.append(trace)
            
            # Create bond traces
            bond_traces = []
            for bond in bonds:
                atom1_idx = bond.get('atom1', 0)
                atom2_idx = bond.get('atom2', 1)
                
                if atom1_idx < len(atoms) and atom2_idx < len(atoms):
                    pos1 = atoms[atom1_idx].get('position', [0, 0, 0])
                    pos2 = atoms[atom2_idx].get('position', [0, 0, 0])
                    
                    trace = go.Scatter3d(
                        x=[pos1[0], pos2[0]], 
                        y=[pos1[1], pos2[1]], 
                        z=[pos1[2], pos2[2]],
                        mode='lines',
                        line=dict(color='gray', width=5),
                        showlegend=False
                    )
                    bond_traces.append(trace)
            
            # Create figure
            fig = go.Figure(data=atom_traces + bond_traces)
            
            fig.update_layout(
                title=config.title or '3D Molecular Structure',
                scene=dict(
                    xaxis_title=config.xlabel or 'X',
                    yaxis_title=config.ylabel or 'Y',
                    zaxis_title=config.zlabel or 'Z'
                ),
                width=config.width,
                height=config.height
            )
            
            # Save as HTML
            filename = f"3d_structure_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            output_path = self.output_dir / filename
            fig.write_html(str(output_path))
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to create 3D structure with Plotly: {str(e)}")
            return None
    
    def _create_3d_structure_matplotlib(self, data: Dict[str, Any], config: PlotConfig) -> Optional[str]:
        """Create 3D structure using Matplotlib."""
        fig = plt.figure(figsize=(config.width/100, config.height/100), dpi=config.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        try:
            atoms = data.get('atoms', [])
            bonds = data.get('bonds', [])
            
            # Plot atoms
            for atom in atoms:
                x, y, z = atom.get('position', [0, 0, 0])
                element = atom.get('element', 'C')
                color = atom.get('color', 'blue')
                size = atom.get('size', 100)
                
                ax.scatter([x], [y], [z], c=color, s=size, alpha=0.8, label=element)
            
            # Plot bonds
            for bond in bonds:
                atom1_idx = bond.get('atom1', 0)
                atom2_idx = bond.get('atom2', 1)
                
                if atom1_idx < len(atoms) and atom2_idx < len(atoms):
                    pos1 = atoms[atom1_idx].get('position', [0, 0, 0])
                    pos2 = atoms[atom2_idx].get('position', [0, 0, 0])
                    
                    ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]], 
                           color='gray', linewidth=2, alpha=0.7)
            
            ax.set_xlabel(config.xlabel or 'X')
            ax.set_ylabel(config.ylabel or 'Y')
            ax.set_zlabel(config.zlabel or 'Z')
            ax.set_title(config.title or '3D Molecular Structure')
            
            filename = f"3d_structure_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{config.save_format}"
            output_path = self.output_dir / filename
            plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            plt.close()
            raise e
    
    def _create_distribution_plot(self, data: Dict[str, Any], config: PlotConfig, backend: str) -> Optional[str]:
        """Create probability distribution plots."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        fig, ax = plt.subplots(figsize=(config.width/100, config.height/100), dpi=config.dpi)
        
        try:
            x = data.get('x', [])
            y = data.get('y', [])
            
            if 'probability' in data:
                # Plot probability distribution
                prob = data['probability']
                ax.plot(x, prob, linewidth=2, label='Probability')
                ax.fill_between(x, prob, alpha=0.3)
                ax.set_ylabel('Probability Density')
            elif 'wavefunction' in data:
                # Plot wave function
                psi = data['wavefunction']
                if NUMPY_AVAILABLE:
                    psi = np.array(psi)
                    # Plot real and imaginary parts if complex
                    if np.iscomplexobj(psi):
                        ax.plot(x, psi.real, label='Re(ψ)', linewidth=2)
                        ax.plot(x, psi.imag, label='Im(ψ)', linewidth=2)
                        ax.plot(x, np.abs(psi)**2, label='|ψ|²', linewidth=2)
                    else:
                        ax.plot(x, psi, label='ψ', linewidth=2)
                        ax.plot(x, psi**2, label='ψ²', linewidth=2)
                else:
                    ax.plot(x, psi, label='ψ', linewidth=2)
                ax.set_ylabel('Wave Function')
            else:
                # Regular plot
                ax.plot(x, y, linewidth=2)
            
            ax.set_xlabel(config.xlabel or 'Position')
            ax.set_title(config.title or 'Probability Distribution')
            
            if config.legend:
                ax.legend()
            
            if config.grid:
                ax.grid(True, alpha=0.3)
            
            filename = f"distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{config.save_format}"
            output_path = self.output_dir / filename
            plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            plt.close()
            raise e
    
    def _create_vector_field(self, data: Dict[str, Any], config: PlotConfig, backend: str) -> Optional[str]:
        """Create vector field plots."""
        if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
            return None
        
        fig, ax = plt.subplots(figsize=(config.width/100, config.height/100), dpi=config.dpi)
        
        try:
            X = np.array(data.get('X', []))
            Y = np.array(data.get('Y', []))
            U = np.array(data.get('U', []))  # x-component of vector field
            V = np.array(data.get('V', []))  # y-component of vector field
            
            # Create vector field plot
            Q = ax.quiver(X, Y, U, V, alpha=0.8, scale=data.get('scale', 50))
            
            # Add field lines if provided
            if 'field_lines' in data:
                for line in data['field_lines']:
                    x_line = line.get('x', [])
                    y_line = line.get('y', [])
                    ax.plot(x_line, y_line, 'k-', alpha=0.6, linewidth=1)
            
            # Add equipotential lines if provided
            if 'equipotentials' in data:
                Z = data['equipotentials']
                contour = ax.contour(X, Y, Z, colors='red', alpha=0.6, linewidths=1)
                ax.clabel(contour, inline=True, fontsize=8)
            
            ax.set_xlabel(config.xlabel or 'X')
            ax.set_ylabel(config.ylabel or 'Y')
            ax.set_title(config.title or 'Vector Field')
            
            if config.grid:
                ax.grid(True, alpha=0.3)
            
            filename = f"vector_field_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{config.save_format}"
            output_path = self.output_dir / filename
            plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            plt.close()
            raise e
    
    def _create_wave_plot(self, data: Dict[str, Any], config: PlotConfig, backend: str) -> Optional[str]:
        """Create wave and oscillation plots."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        fig, ax = plt.subplots(figsize=(config.width/100, config.height/100), dpi=config.dpi)
        
        try:
            t = data.get('t', [])
            x = data.get('x', [])
            
            if 'amplitude' in data:
                # Plot wave with amplitude envelope
                amplitude = data['amplitude']
                frequency = data.get('frequency', 1)
                phase = data.get('phase', 0)
                
                if NUMPY_AVAILABLE:
                    t = np.array(t)
                    wave = amplitude * np.sin(2 * np.pi * frequency * t + phase)
                    ax.plot(t, wave, 'b-', linewidth=2, label='Wave')
                    
                    # Plot envelope if amplitude varies
                    if isinstance(amplitude, (list, np.ndarray)):
                        ax.plot(t, amplitude, 'r--', alpha=0.7, label='Envelope')
                        ax.plot(t, -amplitude, 'r--', alpha=0.7)
            else:
                # Direct plot
                ax.plot(t, x, linewidth=2)
            
            # Add harmonics if provided
            if 'harmonics' in data:
                colors = ['red', 'green', 'orange', 'purple']
                for i, harmonic in enumerate(data['harmonics']):
                    t_harm = harmonic.get('t', t)
                    x_harm = harmonic.get('x', [])
                    color = colors[i % len(colors)]
                    ax.plot(t_harm, x_harm, '--', color=color, alpha=0.7, 
                           label=f'Harmonic {i+1}')
            
            ax.set_xlabel(config.xlabel or 'Time')
            ax.set_ylabel(config.ylabel or 'Amplitude')
            ax.set_title(config.title or 'Wave')
            
            if config.legend:
                ax.legend()
            
            if config.grid:
                ax.grid(True, alpha=0.3)
            
            filename = f"wave_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{config.save_format}"
            output_path = self.output_dir / filename
            plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            plt.close()
            raise e
    
    def _create_correlation_plot(self, data: Dict[str, Any], config: PlotConfig, backend: str) -> Optional[str]:
        """Create correlation and matrix plots."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        fig, ax = plt.subplots(figsize=(config.width/100, config.height/100), dpi=config.dpi)
        
        try:
            if 'correlation_matrix' in data:
                # Plot correlation matrix
                matrix = np.array(data['correlation_matrix'])
                labels = data.get('labels', [f'Var{i}' for i in range(matrix.shape[0])])
                
                im = ax.imshow(matrix, cmap=config.color_scheme, aspect='auto', 
                             vmin=-1, vmax=1)
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Correlation Coefficient')
                
                # Set ticks and labels
                ax.set_xticks(range(len(labels)))
                ax.set_yticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45)
                ax.set_yticklabels(labels)
                
                # Add correlation values as text
                for i in range(len(labels)):
                    for j in range(len(labels)):
                        text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                                     ha="center", va="center", color="w" if abs(matrix[i, j]) > 0.5 else "k")
            
            elif 'x' in data and 'y' in data:
                # Scatter plot with correlation
                x = data['x']
                y = data['y']
                ax.scatter(x, y, alpha=0.6)
                
                # Add trend line if requested
                if data.get('add_trendline', True) and NUMPY_AVAILABLE:
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    ax.plot(x, p(x), "r--", alpha=0.8, label=f'Trend line')
                    
                    # Calculate correlation coefficient
                    corr = np.corrcoef(x, y)[0, 1]
                    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                           bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
            
            ax.set_xlabel(config.xlabel)
            ax.set_ylabel(config.ylabel)
            ax.set_title(config.title or 'Correlation Plot')
            
            if config.legend and 'add_trendline' in data:
                ax.legend()
            
            filename = f"correlation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{config.save_format}"
            output_path = self.output_dir / filename
            plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            plt.close()
            raise e
    
    # Interactive visualization methods
    def create_interactive_plot(self, data: Dict[str, Any], plot_type: str, **kwargs) -> Optional[str]:
        """Create interactive plots using Plotly."""
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly required for interactive plots")
            return None
        
        try:
            if plot_type == 'scatter':
                return self._create_interactive_scatter(data, **kwargs)
            elif plot_type == '3d_scatter':
                return self._create_interactive_3d_scatter(data, **kwargs)
            elif plot_type == 'surface':
                return self._create_interactive_surface(data, **kwargs)
            elif plot_type == 'animation':
                return self._create_animation(data, **kwargs)
            else:
                logger.error(f"Interactive plot type {plot_type} not supported")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create interactive plot: {str(e)}")
            return None
    
    def _create_interactive_scatter(self, data: Dict[str, Any], **kwargs) -> Optional[str]:
        """Create interactive scatter plot with Plotly."""
        x = data.get('x', [])
        y = data.get('y', [])
        text = data.get('text', [])
        color = data.get('color', [])
        
        fig = px.scatter(x=x, y=y, text=text, color=color,
                        title=kwargs.get('title', 'Interactive Scatter Plot'),
                        labels={'x': kwargs.get('xlabel', 'X'),
                               'y': kwargs.get('ylabel', 'Y')})
        
        filename = f"interactive_scatter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        output_path = self.output_dir / filename
        fig.write_html(str(output_path))
        
        return str(output_path)
    
    def _create_interactive_3d_scatter(self, data: Dict[str, Any], **kwargs) -> Optional[str]:
        """Create interactive 3D scatter plot."""
        x = data.get('x', [])
        y = data.get('y', [])
        z = data.get('z', [])
        color = data.get('color', [])
        
        fig = px.scatter_3d(x=x, y=y, z=z, color=color,
                           title=kwargs.get('title', 'Interactive 3D Scatter'),
                           labels={'x': kwargs.get('xlabel', 'X'),
                                  'y': kwargs.get('ylabel', 'Y'),
                                  'z': kwargs.get('zlabel', 'Z')})
        
        filename = f"interactive_3d_scatter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        output_path = self.output_dir / filename
        fig.write_html(str(output_path))
        
        return str(output_path)
    
    def _create_interactive_surface(self, data: Dict[str, Any], **kwargs) -> Optional[str]:
        """Create interactive surface plot."""
        z = data.get('z', [])
        x = data.get('x', None)
        y = data.get('y', None)
        
        fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
        
        fig.update_layout(
            title=kwargs.get('title', 'Interactive Surface'),
            scene=dict(
                xaxis_title=kwargs.get('xlabel', 'X'),
                yaxis_title=kwargs.get('ylabel', 'Y'),
                zaxis_title=kwargs.get('zlabel', 'Z')
            )
        )
        
        filename = f"interactive_surface_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        output_path = self.output_dir / filename
        fig.write_html(str(output_path))
        
        return str(output_path)
    
    def _create_animation(self, data: Dict[str, Any], **kwargs) -> Optional[str]:
        """Create animated plots."""
        # This would create animations showing time evolution
        # Implementation depends on specific data format
        frames = data.get('frames', [])
        
        if not frames:
            logger.error("No frames provided for animation")
            return None
        
        # Create basic animated scatter plot
        if 'x' in frames[0] and 'y' in frames[0]:
            fig = px.scatter(frames[0], x='x', y='y',
                           title=kwargs.get('title', 'Animation'),
                           animation_frame='frame' if 'frame' in frames[0] else None)
            
            filename = f"animation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            output_path = self.output_dir / filename
            fig.write_html(str(output_path))
            
            return str(output_path)
        
        return None
    
    # Utility methods
    def get_supported_plot_types(self) -> Dict[str, List[str]]:
        """Get list of supported plot types by backend."""
        return {
            'matplotlib': [
                'line', 'scatter', 'bar', 'histogram', 'phase_diagram',
                'contour', 'heatmap', 'trajectory', 'energy_levels',
                '3d_structure', 'distribution', 'vector_field', 'wave',
                'correlation'
            ],
            'plotly': [
                'interactive_scatter', '3d_scatter', 'surface', 'animation',
                '3d_structure'
            ],
            'seaborn': [
                'distribution', 'correlation', 'heatmap'
            ]
        }
    
    def save_plot_config(self, config: PlotConfig, filename: str) -> bool:
        """Save plot configuration to file."""
        try:
            config_path = self.output_dir / f"{filename}.json"
            with open(config_path, 'w') as f:
                json.dump(config.__dict__, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save plot config: {str(e)}")
            return False
    
    def load_plot_config(self, filename: str) -> Optional[PlotConfig]:
        """Load plot configuration from file."""
        try:
            config_path = self.output_dir / f"{filename}.json"
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            return PlotConfig(**config_dict)
        except Exception as e:
            logger.error(f"Failed to load plot config: {str(e)}")
            return None