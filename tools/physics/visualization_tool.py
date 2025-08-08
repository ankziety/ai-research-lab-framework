"""
Visualization Tool

Agent-friendly interface for physics data visualization.
Provides plotting, charting, and interactive visualization
capabilities for physics research data and results.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import numpy as np
from datetime import datetime
import base64
import io

# Optional imports - gracefully handle if not available
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    mpatches = None
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    sns = None
    HAS_SEABORN = False

from .base_physics_tool import BasePhysicsTool

logger = logging.getLogger(__name__)


class VisualizationTool(BasePhysicsTool):
    """
    Tool for physics data visualization that agents can request.
    
    Provides interfaces for:
    - Scientific plotting and charting
    - Data visualization and exploration
    - Publication-quality figures
    - Interactive plots and animations
    - Physics-specific plot types
    """
    
    def __init__(self):
        super().__init__(
            tool_id="visualization_tool",
            name="Visualization Tool",
            description="Create scientific visualizations and plots for physics data analysis and presentation",
            physics_domain="data_visualization",
            computational_cost_factor=1.0,
            software_requirements=[
                "matplotlib",   # Core plotting
                "numpy",        # Data handling
                "seaborn"       # Advanced plotting (optional)
            ],
            hardware_requirements={
                "min_memory": 256,   # MB
                "recommended_memory": 512,
                "cpu_cores": 1,
                "supports_gpu": False
            }
        )
        
        # Add visualization specific capabilities
        self.capabilities.extend([
            "scientific_plotting",
            "data_visualization",
            "publication_figures",
            "statistical_plots",
            "physics_specific_plots",
            "interactive_visualization",
            "plot_customization",
            "multi_panel_figures"
        ])
        
        # Available plot types
        self.plot_types = [
            "line_plot",
            "scatter_plot",
            "histogram",
            "heatmap",
            "contour_plot",
            "box_plot",
            "violin_plot",
            "error_bars",
            "regression_plot",
            "phase_space",
            "energy_levels",
            "wave_function",
            "field_lines",
            "orbit_plot"
        ]
        
        # Color schemes for different physics domains
        self.color_schemes = {
            "quantum": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
            "thermal": ["#d62728", "#ff7f0e", "#ffbb78", "#2ca02c"],
            "electromagnetic": ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f"],
            "stellar": ["#ffbb78", "#ff7f0e", "#d62728", "#8c564b"],
            "default": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        }
        
        # Physics-specific plot settings
        self.physics_settings = {
            "quantum_mechanics": {
                "xlabel_default": "Position (nm)",
                "ylabel_default": "Wave Function (ψ)",
                "style": "quantum"
            },
            "thermodynamics": {
                "xlabel_default": "Temperature (K)",
                "ylabel_default": "Energy (J)",
                "style": "thermal"
            },
            "electromagnetism": {
                "xlabel_default": "Distance (m)",
                "ylabel_default": "Field Strength",
                "style": "electromagnetic"
            },
            "astrophysics": {
                "xlabel_default": "Time (years)",
                "ylabel_default": "Luminosity (L☉)",
                "style": "stellar"
            }
        }
    
    def execute(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute visualization task requested by an agent.
        
        Args:
            task: Task specification with visualization parameters
            context: Agent context and execution environment
            
        Returns:
            Visualization results formatted for agents
        """
        start_time = datetime.now()
        
        try:
            # Check if matplotlib is available
            if not HAS_MATPLOTLIB:
                return {
                    "success": False,
                    "error": "Matplotlib not available",
                    "suggestion": "Install matplotlib for visualization capabilities"
                }
            
            # Validate input parameters
            validation_result = self.validate_input(task)
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": "Input validation failed",
                    "validation_errors": validation_result["errors"],
                    "suggestions": validation_result["suggestions"]
                }
            
            # Extract task parameters
            plot_type = task.get("type", "line_plot")
            data = task.get("data", {})
            style_params = task.get("style", {})
            physics_domain = task.get("physics_domain", "default")
            
            # Create visualization
            if plot_type == "line_plot":
                result = self._create_line_plot(data, style_params, physics_domain)
            elif plot_type == "scatter_plot":
                result = self._create_scatter_plot(data, style_params, physics_domain)
            elif plot_type == "histogram":
                result = self._create_histogram(data, style_params, physics_domain)
            elif plot_type == "heatmap":
                result = self._create_heatmap(data, style_params, physics_domain)
            elif plot_type == "contour_plot":
                result = self._create_contour_plot(data, style_params, physics_domain)
            elif plot_type == "error_bars":
                result = self._create_error_plot(data, style_params, physics_domain)
            elif plot_type == "box_plot":
                result = self._create_box_plot(data, style_params, physics_domain)
            elif plot_type == "regression_plot":
                result = self._create_regression_plot(data, style_params, physics_domain)
            elif plot_type == "phase_space":
                result = self._create_phase_space_plot(data, style_params, physics_domain)
            elif plot_type == "energy_levels":
                result = self._create_energy_level_plot(data, style_params, physics_domain)
            elif plot_type == "wave_function":
                result = self._create_wave_function_plot(data, style_params, physics_domain)
            elif plot_type == "field_lines":
                result = self._create_field_line_plot(data, style_params, physics_domain)
            elif plot_type == "orbit_plot":
                result = self._create_orbit_plot(data, style_params, physics_domain)
            else:
                raise ValueError(f"Unknown plot type: {plot_type}")
            
            # Process and format output for agents
            formatted_result = self.process_output(result)
            
            # Update statistics
            calculation_time = (datetime.now() - start_time).total_seconds()
            estimated_cost = self._calculate_actual_cost(task, calculation_time)
            self.update_calculation_stats(calculation_time, estimated_cost, True)
            
            return {
                "success": True,
                "plot_type": plot_type,
                "results": formatted_result,
                "calculation_time": calculation_time,
                "computational_cost": estimated_cost,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            calculation_time = (datetime.now() - start_time).total_seconds()
            self.update_calculation_stats(calculation_time, 0.0, False)
            return self.handle_errors(e, {"task": task, "context": context})
    
    def validate_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate visualization input parameters.
        
        Args:
            input_data: Input parameters from agent
            
        Returns:
            Validation result with errors/warnings
        """
        errors = []
        warnings = []
        suggestions = []
        
        # Check plot type
        plot_type = input_data.get("type", "line_plot")
        if plot_type not in self.plot_types:
            errors.append(f"Unknown plot type '{plot_type}'")
            suggestions.append(f"Available types: {', '.join(self.plot_types)}")
        
        # Check data presence
        if "data" not in input_data:
            errors.append("Missing 'data' for visualization")
            suggestions.append("Provide data as dictionary with arrays")
        else:
            data = input_data["data"]
            
            if not isinstance(data, dict):
                errors.append("Data must be dictionary")
                suggestions.append("Use format: {'x': [1,2,3], 'y': [4,5,6]}")
            elif not data:
                errors.append("Data dictionary is empty")
            else:
                # Check for required variables based on plot type
                if plot_type in ["line_plot", "scatter_plot", "error_bars", "regression_plot"]:
                    if "x" not in data or "y" not in data:
                        errors.append("Line/scatter plots require 'x' and 'y' data")
                
                elif plot_type == "histogram":
                    if not any(key in data for key in ["values", "x", "data"]):
                        errors.append("Histogram requires 'values', 'x', or 'data'")
                
                elif plot_type == "heatmap":
                    if "z" not in data:
                        errors.append("Heatmap requires 'z' (2D) data")
                
                # Check array lengths
                lengths = []
                for key, values in data.items():
                    if hasattr(values, '__len__'):
                        lengths.append(len(values))
                
                if lengths and len(set(lengths)) > 1 and plot_type not in ["heatmap"]:
                    warnings.append("Data arrays have different lengths")
        
        # Validate style parameters
        style = input_data.get("style", {})
        if style and not isinstance(style, dict):
            warnings.append("Style parameters should be dictionary")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions
        }
    
    def process_output(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format visualization results for agents.
        
        Args:
            output_data: Raw visualization results
            
        Returns:
            Agent-friendly formatted results
        """
        formatted = {
            "plot_info": {
                "type": output_data.get("plot_type", "unknown"),
                "title": output_data.get("title", ""),
                "dimensions": output_data.get("dimensions", "2D"),
                "data_points": output_data.get("data_points", 0)
            },
            "image_data": {},
            "plot_elements": {},
            "recommendations": []
        }
        
        # Image data
        if "image_base64" in output_data:
            formatted["image_data"]["base64"] = output_data["image_base64"]
            formatted["image_data"]["format"] = "PNG"
            formatted["image_data"]["size_bytes"] = len(output_data["image_base64"]) * 3 // 4
        
        if "image_path" in output_data:
            formatted["image_data"]["file_path"] = output_data["image_path"]
        
        # Plot elements
        if "axes_labels" in output_data:
            formatted["plot_elements"]["axes"] = output_data["axes_labels"]
        
        if "legend_items" in output_data:
            formatted["plot_elements"]["legend"] = output_data["legend_items"]
        
        if "color_scheme" in output_data:
            formatted["plot_elements"]["colors"] = output_data["color_scheme"]
        
        # Analysis and recommendations
        formatted["recommendations"] = self._generate_plot_recommendations(output_data)
        
        return formatted
    
    def estimate_cost(self, task: Dict[str, Any]) -> Dict[str, float]:
        """
        Estimate computational cost for visualization.
        
        Args:
            task: Task specification
            
        Returns:
            Cost estimates (time, memory, computational units)
        """
        base_cost = 1.0
        
        # Get data size
        data = task.get("data", {})
        data_size = 0
        for values in data.values():
            if hasattr(values, '__len__'):
                data_size = max(data_size, len(values))
        
        # Scale with data size
        size_cost_factor = max(1.0, (data_size / 1000) ** 0.3)
        
        # Plot type cost factors
        plot_costs = {
            "line_plot": 0.5,
            "scatter_plot": 0.7,
            "histogram": 0.4,
            "heatmap": 1.5,
            "contour_plot": 2.0,
            "box_plot": 0.6,
            "error_bars": 0.8,
            "regression_plot": 1.0,
            "phase_space": 1.2,
            "energy_levels": 0.9,
            "wave_function": 1.1,
            "field_lines": 2.5,
            "orbit_plot": 1.3
        }
        
        plot_type = task.get("type", "line_plot")
        plot_cost = plot_costs.get(plot_type, 1.0)
        
        total_cost_factor = size_cost_factor * plot_cost * self.computational_cost_factor
        
        # Estimate time (in seconds)
        estimated_time = base_cost * total_cost_factor * 0.2
        
        # Estimate memory (in MB)
        estimated_memory = 100 + (data_size * 4 / 1024**2) * 2  # Rough estimate
        
        # Computational units
        computational_units = total_cost_factor * 5
        
        return {
            "estimated_time_seconds": estimated_time,
            "estimated_memory_mb": estimated_memory,
            "computational_units": computational_units,
            "cost_breakdown": {
                "size_factor": size_cost_factor,
                "plot_factor": plot_cost,
                "total_factor": total_cost_factor
            }
        }
    
    def get_physics_requirements(self) -> Dict[str, Any]:
        """Get visualization specific requirements."""
        return {
            "physics_domain": "data_visualization",
            "available_plot_types": self.plot_types,
            "supported_physics_domains": list(self.physics_settings.keys()),
            "color_schemes": list(self.color_schemes.keys()),
            "output_formats": ["PNG", "SVG", "PDF"] if HAS_MATPLOTLIB else [],
            "interactive_support": False,
            "typical_creation_time": "Milliseconds to seconds",
            "memory_scaling": "Linear with data size",
            "software_dependencies": self.software_requirements,
            "hardware_recommendations": self.hardware_requirements
        }
    
    def _get_domain_keywords(self) -> List[str]:
        """Get visualization domain keywords."""
        return [
            "plot", "chart", "graph", "visualization", "figure",
            "scatter", "line", "histogram", "heatmap", "contour",
            "data", "axis", "legend", "color", "style", "scientific"
        ]
    
    def _apply_physics_style(self, physics_domain: str, style_params: Dict[str, Any]):
        """Apply physics-specific styling to plots."""
        if physics_domain in self.physics_settings:
            settings = self.physics_settings[physics_domain]
            
            # Set color scheme
            color_scheme = settings.get("style", "default")
            if color_scheme in self.color_schemes:
                colors = self.color_schemes[color_scheme]
                plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
            
            # Apply default labels if not specified
            if 'xlabel' not in style_params:
                style_params['xlabel'] = settings.get("xlabel_default", "X")
            if 'ylabel' not in style_params:
                style_params['ylabel'] = settings.get("ylabel_default", "Y")
        
        # Apply general scientific styling
        plt.style.use('default')  # Reset to default
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'lines.linewidth': 2,
            'grid.alpha': 0.3
        })
    
    def _save_plot_as_base64(self) -> str:
        """Save current plot as base64 encoded string."""
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        return image_base64
    
    def _create_line_plot(self, data: Dict[str, Any], style: Dict[str, Any], physics_domain: str) -> Dict[str, Any]:
        """Create line plot."""
        self._apply_physics_style(physics_domain, style)
        
        fig, ax = plt.subplots(figsize=style.get('figsize', (8, 6)))
        
        x = np.array(data['x'])
        y = np.array(data['y'])
        
        # Multiple y series support
        if isinstance(y[0], (list, tuple, np.ndarray)):
            for i, y_series in enumerate(y):
                label = style.get('labels', [f'Series {i+1}'])[i] if i < len(style.get('labels', [])) else f'Series {i+1}'
                ax.plot(x, y_series, label=label, marker=style.get('marker', 'o'), markersize=style.get('markersize', 4))
            ax.legend()
        else:
            ax.plot(x, y, marker=style.get('marker', 'o'), markersize=style.get('markersize', 4))
        
        ax.set_xlabel(style.get('xlabel', 'X'))
        ax.set_ylabel(style.get('ylabel', 'Y'))
        ax.set_title(style.get('title', 'Line Plot'))
        ax.grid(True, alpha=0.3)
        
        image_base64 = self._save_plot_as_base64()
        plt.close()
        
        return {
            "plot_type": "line_plot",
            "image_base64": image_base64,
            "data_points": len(x),
            "axes_labels": {"x": style.get('xlabel', 'X'), "y": style.get('ylabel', 'Y')},
            "title": style.get('title', 'Line Plot')
        }
    
    def _create_scatter_plot(self, data: Dict[str, Any], style: Dict[str, Any], physics_domain: str) -> Dict[str, Any]:
        """Create scatter plot."""
        self._apply_physics_style(physics_domain, style)
        
        fig, ax = plt.subplots(figsize=style.get('figsize', (8, 6)))
        
        x = np.array(data['x'])
        y = np.array(data['y'])
        
        # Color by category if provided
        if 'color' in data:
            colors = data['color']
            scatter = ax.scatter(x, y, c=colors, alpha=style.get('alpha', 0.7), 
                               s=style.get('markersize', 50), cmap=style.get('colormap', 'viridis'))
            plt.colorbar(scatter)
        else:
            ax.scatter(x, y, alpha=style.get('alpha', 0.7), s=style.get('markersize', 50))
        
        ax.set_xlabel(style.get('xlabel', 'X'))
        ax.set_ylabel(style.get('ylabel', 'Y'))
        ax.set_title(style.get('title', 'Scatter Plot'))
        ax.grid(True, alpha=0.3)
        
        image_base64 = self._save_plot_as_base64()
        plt.close()
        
        return {
            "plot_type": "scatter_plot",
            "image_base64": image_base64,
            "data_points": len(x),
            "axes_labels": {"x": style.get('xlabel', 'X'), "y": style.get('ylabel', 'Y')},
            "title": style.get('title', 'Scatter Plot')
        }
    
    def _create_histogram(self, data: Dict[str, Any], style: Dict[str, Any], physics_domain: str) -> Dict[str, Any]:
        """Create histogram."""
        self._apply_physics_style(physics_domain, style)
        
        fig, ax = plt.subplots(figsize=style.get('figsize', (8, 6)))
        
        values = np.array(data.get('values', data.get('x', data.get('data', []))))
        
        bins = style.get('bins', 30)
        if isinstance(bins, str) and bins == 'auto':
            bins = 'auto'
        
        n, bins_edges, patches = ax.hist(values, bins=bins, alpha=style.get('alpha', 0.7),
                                       density=style.get('density', False),
                                       edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel(style.get('xlabel', 'Values'))
        ax.set_ylabel(style.get('ylabel', 'Frequency'))
        ax.set_title(style.get('title', 'Histogram'))
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        if style.get('show_stats', True):
            mean_val = np.mean(values)
            std_val = np.std(values)
            ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            ax.legend()
        
        image_base64 = self._save_plot_as_base64()
        plt.close()
        
        return {
            "plot_type": "histogram",
            "image_base64": image_base64,
            "data_points": len(values),
            "bins": len(bins_edges) - 1 if hasattr(bins_edges, '__len__') else bins,
            "axes_labels": {"x": style.get('xlabel', 'Values'), "y": style.get('ylabel', 'Frequency')},
            "title": style.get('title', 'Histogram')
        }
    
    def _create_heatmap(self, data: Dict[str, Any], style: Dict[str, Any], physics_domain: str) -> Dict[str, Any]:
        """Create heatmap."""
        self._apply_physics_style(physics_domain, style)
        
        fig, ax = plt.subplots(figsize=style.get('figsize', (10, 8)))
        
        z = np.array(data['z'])
        
        # Create heatmap
        im = ax.imshow(z, cmap=style.get('colormap', 'viridis'), 
                      aspect=style.get('aspect', 'auto'),
                      origin=style.get('origin', 'lower'))
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label(style.get('colorbar_label', 'Intensity'))
        
        ax.set_xlabel(style.get('xlabel', 'X'))
        ax.set_ylabel(style.get('ylabel', 'Y'))
        ax.set_title(style.get('title', 'Heatmap'))
        
        image_base64 = self._save_plot_as_base64()
        plt.close()
        
        return {
            "plot_type": "heatmap",
            "image_base64": image_base64,
            "dimensions": "2D",
            "data_shape": z.shape,
            "axes_labels": {"x": style.get('xlabel', 'X'), "y": style.get('ylabel', 'Y')},
            "title": style.get('title', 'Heatmap')
        }
    
    def _create_contour_plot(self, data: Dict[str, Any], style: Dict[str, Any], physics_domain: str) -> Dict[str, Any]:
        """Create contour plot."""
        self._apply_physics_style(physics_domain, style)
        
        fig, ax = plt.subplots(figsize=style.get('figsize', (10, 8)))
        
        if 'x' in data and 'y' in data and 'z' in data:
            x = np.array(data['x'])
            y = np.array(data['y'])
            z = np.array(data['z'])
            
            # Create meshgrid if needed
            if z.ndim == 1:
                # Assume z values correspond to (x,y) pairs
                xi = np.linspace(x.min(), x.max(), 50)
                yi = np.linspace(y.min(), y.max(), 50)
                xi, yi = np.meshgrid(xi, yi)
                # Simple interpolation for demo
                zi = np.random.random(xi.shape)  # Mock data
            else:
                xi, yi = np.meshgrid(x, y)
                zi = z
        else:
            # Assume z is a 2D array
            z = np.array(data['z'])
            xi, yi = np.mgrid[0:z.shape[0], 0:z.shape[1]]
            zi = z
        
        levels = style.get('levels', 10)
        contour = ax.contour(xi, yi, zi, levels=levels, colors='black', linewidths=0.5)
        contour_filled = ax.contourf(xi, yi, zi, levels=levels, cmap=style.get('colormap', 'viridis'))
        
        plt.colorbar(contour_filled)
        ax.clabel(contour, inline=True, fontsize=8)
        
        ax.set_xlabel(style.get('xlabel', 'X'))
        ax.set_ylabel(style.get('ylabel', 'Y'))
        ax.set_title(style.get('title', 'Contour Plot'))
        
        image_base64 = self._save_plot_as_base64()
        plt.close()
        
        return {
            "plot_type": "contour_plot",
            "image_base64": image_base64,
            "dimensions": "2D",
            "contour_levels": levels,
            "axes_labels": {"x": style.get('xlabel', 'X'), "y": style.get('ylabel', 'Y')},
            "title": style.get('title', 'Contour Plot')
        }
    
    def _create_error_plot(self, data: Dict[str, Any], style: Dict[str, Any], physics_domain: str) -> Dict[str, Any]:
        """Create error bar plot."""
        self._apply_physics_style(physics_domain, style)
        
        fig, ax = plt.subplots(figsize=style.get('figsize', (8, 6)))
        
        x = np.array(data['x'])
        y = np.array(data['y'])
        
        # Error bars
        x_err = data.get('x_err', None)
        y_err = data.get('y_err', None)
        
        ax.errorbar(x, y, xerr=x_err, yerr=y_err, 
                   fmt=style.get('marker', 'o'), 
                   capsize=style.get('capsize', 3),
                   capthick=style.get('capthick', 1),
                   elinewidth=style.get('elinewidth', 1))
        
        ax.set_xlabel(style.get('xlabel', 'X'))
        ax.set_ylabel(style.get('ylabel', 'Y'))
        ax.set_title(style.get('title', 'Error Bar Plot'))
        ax.grid(True, alpha=0.3)
        
        image_base64 = self._save_plot_as_base64()
        plt.close()
        
        return {
            "plot_type": "error_bars",
            "image_base64": image_base64,
            "data_points": len(x),
            "has_x_errors": x_err is not None,
            "has_y_errors": y_err is not None,
            "axes_labels": {"x": style.get('xlabel', 'X'), "y": style.get('ylabel', 'Y')},
            "title": style.get('title', 'Error Bar Plot')
        }
    
    def _create_box_plot(self, data: Dict[str, Any], style: Dict[str, Any], physics_domain: str) -> Dict[str, Any]:
        """Create box plot."""
        self._apply_physics_style(physics_domain, style)
        
        fig, ax = plt.subplots(figsize=style.get('figsize', (8, 6)))
        
        # Handle multiple datasets
        if isinstance(data, dict) and len(data) > 1:
            # Multiple box plots
            plot_data = []
            labels = []
            for key, values in data.items():
                plot_data.append(values)
                labels.append(key)
            
            box_plot = ax.boxplot(plot_data, labels=labels, patch_artist=True)
            
            # Color boxes
            colors = self.color_schemes.get(physics_domain, self.color_schemes['default'])
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        else:
            # Single box plot
            values = list(data.values())[0]
            ax.boxplot(values, patch_artist=True)
        
        ax.set_ylabel(style.get('ylabel', 'Values'))
        ax.set_title(style.get('title', 'Box Plot'))
        ax.grid(True, alpha=0.3)
        
        image_base64 = self._save_plot_as_base64()
        plt.close()
        
        return {
            "plot_type": "box_plot",
            "image_base64": image_base64,
            "num_groups": len(data) if isinstance(data, dict) else 1,
            "axes_labels": {"y": style.get('ylabel', 'Values')},
            "title": style.get('title', 'Box Plot')
        }
    
    def _create_regression_plot(self, data: Dict[str, Any], style: Dict[str, Any], physics_domain: str) -> Dict[str, Any]:
        """Create regression plot with fit line."""
        self._apply_physics_style(physics_domain, style)
        
        fig, ax = plt.subplots(figsize=style.get('figsize', (8, 6)))
        
        x = np.array(data['x'])
        y = np.array(data['y'])
        
        # Scatter plot
        ax.scatter(x, y, alpha=0.6, s=style.get('markersize', 50))
        
        # Fit line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), "r--", linewidth=2, label=f'y = {z[0]:.3f}x + {z[1]:.3f}')
        
        # Calculate R²
        y_pred = p(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        ax.legend()
        ax.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel(style.get('xlabel', 'X'))
        ax.set_ylabel(style.get('ylabel', 'Y'))
        ax.set_title(style.get('title', 'Regression Plot'))
        ax.grid(True, alpha=0.3)
        
        image_base64 = self._save_plot_as_base64()
        plt.close()
        
        return {
            "plot_type": "regression_plot",
            "image_base64": image_base64,
            "data_points": len(x),
            "r_squared": r_squared,
            "fit_coefficients": z.tolist(),
            "axes_labels": {"x": style.get('xlabel', 'X'), "y": style.get('ylabel', 'Y')},
            "title": style.get('title', 'Regression Plot')
        }
    
    def _create_phase_space_plot(self, data: Dict[str, Any], style: Dict[str, Any], physics_domain: str) -> Dict[str, Any]:
        """Create phase space plot."""
        self._apply_physics_style(physics_domain, style)
        
        fig, ax = plt.subplots(figsize=style.get('figsize', (8, 8)))
        
        position = np.array(data.get('position', data.get('q', data['x'])))
        momentum = np.array(data.get('momentum', data.get('p', data['y'])))
        
        ax.plot(position, momentum, 'b-', alpha=0.7, linewidth=1)
        ax.scatter(position[0], momentum[0], color='green', s=100, label='Start', zorder=5)
        ax.scatter(position[-1], momentum[-1], color='red', s=100, label='End', zorder=5)
        
        ax.set_xlabel(style.get('xlabel', 'Position'))
        ax.set_ylabel(style.get('ylabel', 'Momentum'))
        ax.set_title(style.get('title', 'Phase Space Plot'))
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        image_base64 = self._save_plot_as_base64()
        plt.close()
        
        return {
            "plot_type": "phase_space",
            "image_base64": image_base64,
            "trajectory_points": len(position),
            "dimensions": "2D",
            "axes_labels": {"x": style.get('xlabel', 'Position'), "y": style.get('ylabel', 'Momentum')},
            "title": style.get('title', 'Phase Space Plot')
        }
    
    def _create_energy_level_plot(self, data: Dict[str, Any], style: Dict[str, Any], physics_domain: str) -> Dict[str, Any]:
        """Create energy level diagram."""
        self._apply_physics_style(physics_domain, style)
        
        fig, ax = plt.subplots(figsize=style.get('figsize', (6, 8)))
        
        energies = np.array(data['energies'])
        labels = data.get('labels', [f'n={i}' for i in range(len(energies))])
        
        # Draw energy levels
        for i, (energy, label) in enumerate(zip(energies, labels)):
            ax.hlines(energy, 0, 1, colors='black', linewidth=2)
            ax.text(1.1, energy, f'{label}: {energy:.3f} eV', 
                   verticalalignment='center', fontsize=10)
        
        # Draw transitions if provided
        if 'transitions' in data:
            transitions = data['transitions']
            for start, end in transitions:
                y_start = energies[start]
                y_end = energies[end]
                ax.annotate('', xy=(0.5, y_end), xytext=(0.5, y_start),
                           arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
        
        ax.set_xlim(-0.1, 2)
        ax.set_ylabel(style.get('ylabel', 'Energy (eV)'))
        ax.set_title(style.get('title', 'Energy Level Diagram'))
        ax.set_xticks([])
        
        image_base64 = self._save_plot_as_base64()
        plt.close()
        
        return {
            "plot_type": "energy_levels",
            "image_base64": image_base64,
            "num_levels": len(energies),
            "energy_range": [float(np.min(energies)), float(np.max(energies))],
            "title": style.get('title', 'Energy Level Diagram')
        }
    
    def _create_wave_function_plot(self, data: Dict[str, Any], style: Dict[str, Any], physics_domain: str) -> Dict[str, Any]:
        """Create wave function plot."""
        self._apply_physics_style(physics_domain, style)
        
        fig, ax = plt.subplots(figsize=style.get('figsize', (10, 6)))
        
        x = np.array(data['x'])
        
        if 'psi_real' in data and 'psi_imag' in data:
            # Complex wave function
            psi_real = np.array(data['psi_real'])
            psi_imag = np.array(data['psi_imag'])
            psi_prob = psi_real**2 + psi_imag**2
            
            ax.plot(x, psi_real, 'b-', label='Re(ψ)', linewidth=2)
            ax.plot(x, psi_imag, 'r-', label='Im(ψ)', linewidth=2)
            ax.plot(x, psi_prob, 'g-', label='|ψ|²', linewidth=2)
        elif 'psi' in data:
            # Real wave function
            psi = np.array(data['psi'])
            psi_prob = psi**2
            
            ax.plot(x, psi, 'b-', label='ψ(x)', linewidth=2)
            ax.plot(x, psi_prob, 'r-', label='|ψ(x)|²', linewidth=2)
        
        ax.set_xlabel(style.get('xlabel', 'Position (nm)'))
        ax.set_ylabel(style.get('ylabel', 'Wave Function'))
        ax.set_title(style.get('title', 'Wave Function'))
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        image_base64 = self._save_plot_as_base64()
        plt.close()
        
        return {
            "plot_type": "wave_function",
            "image_base64": image_base64,
            "data_points": len(x),
            "complex_function": 'psi_real' in data and 'psi_imag' in data,
            "axes_labels": {"x": style.get('xlabel', 'Position (nm)'), "y": style.get('ylabel', 'Wave Function')},
            "title": style.get('title', 'Wave Function')
        }
    
    def _create_field_line_plot(self, data: Dict[str, Any], style: Dict[str, Any], physics_domain: str) -> Dict[str, Any]:
        """Create field line plot."""
        self._apply_physics_style(physics_domain, style)
        
        fig, ax = plt.subplots(figsize=style.get('figsize', (10, 8)))
        
        x = np.array(data['x'])
        y = np.array(data['y'])
        Ex = np.array(data['Ex'])
        Ey = np.array(data['Ey'])
        
        # Create field line plot
        ax.streamplot(x, y, Ex, Ey, density=style.get('density', 1), 
                     color=style.get('color', 'blue'),
                     linewidth=style.get('linewidth', 1))
        
        # Add charges if provided
        if 'charges' in data:
            charges = data['charges']
            for charge in charges:
                x_pos, y_pos, q = charge['x'], charge['y'], charge['q']
                color = 'red' if q > 0 else 'blue'
                size = abs(q) * 100
                ax.scatter(x_pos, y_pos, c=color, s=size, alpha=0.8, edgecolors='black')
        
        ax.set_xlabel(style.get('xlabel', 'X (m)'))
        ax.set_ylabel(style.get('ylabel', 'Y (m)'))
        ax.set_title(style.get('title', 'Field Lines'))
        ax.set_aspect('equal')
        
        image_base64 = self._save_plot_as_base64()
        plt.close()
        
        return {
            "plot_type": "field_lines",
            "image_base64": image_base64,
            "grid_size": Ex.shape,
            "num_charges": len(data.get('charges', [])),
            "axes_labels": {"x": style.get('xlabel', 'X (m)'), "y": style.get('ylabel', 'Y (m)')},
            "title": style.get('title', 'Field Lines')
        }
    
    def _create_orbit_plot(self, data: Dict[str, Any], style: Dict[str, Any], physics_domain: str) -> Dict[str, Any]:
        """Create orbital plot."""
        self._apply_physics_style(physics_domain, style)
        
        fig, ax = plt.subplots(figsize=style.get('figsize', (8, 8)))
        
        x = np.array(data['x'])
        y = np.array(data['y'])
        
        # Plot orbit
        ax.plot(x, y, 'b-', linewidth=2, alpha=0.7)
        
        # Mark start and end points
        ax.scatter(x[0], y[0], color='green', s=100, label='Start', zorder=5)
        ax.scatter(x[-1], y[-1], color='red', s=100, label='End', zorder=5)
        
        # Add central body
        central_mass = data.get('central_mass', 1)
        central_radius = style.get('central_radius', 0.1)
        circle = plt.Circle((0, 0), central_radius, color='orange', alpha=0.8)
        ax.add_patch(circle)
        ax.scatter(0, 0, color='orange', s=200, label='Central Body', zorder=6)
        
        ax.set_xlabel(style.get('xlabel', 'X (AU)'))
        ax.set_ylabel(style.get('ylabel', 'Y (AU)'))
        ax.set_title(style.get('title', 'Orbital Plot'))
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        image_base64 = self._save_plot_as_base64()
        plt.close()
        
        return {
            "plot_type": "orbit_plot",
            "image_base64": image_base64,
            "orbit_points": len(x),
            "central_mass": central_mass,
            "axes_labels": {"x": style.get('xlabel', 'X (AU)'), "y": style.get('ylabel', 'Y (AU)')},
            "title": style.get('title', 'Orbital Plot')
        }
    
    def _calculate_actual_cost(self, task: Dict[str, Any], actual_time: float) -> float:
        """Calculate actual computational cost."""
        estimates = self.estimate_cost(task)
        estimated_time = estimates["estimated_time_seconds"]
        
        time_ratio = actual_time / max(estimated_time, 0.01)
        actual_cost = estimates["computational_units"] * time_ratio
        
        return actual_cost
    
    def _generate_plot_recommendations(self, result: Dict[str, Any]) -> List[str]:
        """Generate recommendations for plot improvement."""
        recommendations = []
        
        plot_type = result.get("plot_type", "")
        data_points = result.get("data_points", 0)
        
        if data_points < 10:
            recommendations.append("Consider collecting more data points for better visualization")
        elif data_points > 10000:
            recommendations.append("Consider data sampling or aggregation for better performance")
        
        if plot_type == "scatter_plot":
            recommendations.append("Consider adding trend lines or regression analysis")
        
        if plot_type == "histogram":
            recommendations.append("Try different bin sizes to optimize data representation")
        
        if "r_squared" in result and result["r_squared"] < 0.7:
            recommendations.append("Low R² suggests poor fit - consider alternative models")
        
        return recommendations