"""
Scientific Results Visualizer

A module for generating matplotlib plots from experiment result dictionaries.
Designed for headless environments and scientific data visualization.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional
import os

# Configure matplotlib for headless environments
matplotlib.use('Agg')


def visualize(results: List[Dict], out_path: str) -> None:
    """
    Generate and save visualization plots from experiment results.
    
    This function creates multiple plot types to summarize key result fields:
    - Bar chart of numeric metrics
    - Line plot showing trends over experiments
    - Summary statistics table
    
    Args:
        results: List of dictionaries containing experiment results
        out_path: Path where the visualization will be saved (should include file extension)
    
    Raises:
        ValueError: If results list is empty or invalid
        IOError: If unable to save the file
    """
    if not results:
        raise ValueError("Results list cannot be empty")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Experiment Results Summary', fontsize=16, fontweight='bold')
    
    # Extract common numeric fields from results
    numeric_fields = _extract_numeric_fields(results)
    
    if numeric_fields:
        # Create bar chart of average values
        _create_bar_chart(axes[0, 0], results, numeric_fields)
        
        # Create line plot showing trends
        _create_line_plot(axes[0, 1], results, numeric_fields)
    
    # Create summary statistics
    _create_summary_stats(axes[1, 0], results)
    
    # Create experiment count by status/type
    _create_experiment_summary(axes[1, 1], results)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Ensure directory exists if path contains directories
    dir_path = os.path.dirname(out_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    try:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        raise IOError(f"Failed to save visualization to {out_path}: {str(e)}")


def _extract_numeric_fields(results: List[Dict]) -> List[str]:
    """
    Extract field names that contain numeric values from the results.
    
    Args:
        results: List of experiment result dictionaries
    
    Returns:
        List of field names that contain numeric values
    """
    if not results:
        return []
    
    numeric_fields = []
    sample_result = results[0]
    
    for key, value in sample_result.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            # Check if most results have this field as numeric
            numeric_count = sum(1 for r in results 
                              if key in r and isinstance(r[key], (int, float)) 
                              and not isinstance(r[key], bool))
            if numeric_count > len(results) * NUMERIC_FIELD_THRESHOLD:  # At least 50% should be numeric
                numeric_fields.append(key)
    
    return numeric_fields


def _create_bar_chart(ax, results: List[Dict], numeric_fields: List[str]) -> None:
    """
    Create a bar chart showing average values of numeric fields.
    
    Args:
        ax: Matplotlib axis object
        results: List of experiment result dictionaries
        numeric_fields: List of field names to plot
    """
    if not numeric_fields:
        ax.text(0.5, 0.5, 'No numeric fields found', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Numeric Metrics (Bar Chart)')
        return
    
    # Calculate averages for each numeric field
    averages = {}
    for field in numeric_fields:
        values = [r.get(field, 0) for r in results if isinstance(r.get(field), (int, float))]
        if values:
            averages[field] = np.mean(values)
    
    if not averages:
        ax.text(0.5, 0.5, 'No valid numeric data', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Numeric Metrics (Bar Chart)')
        return
    
    # Create bar chart
    fields = list(averages.keys())
    values = list(averages.values())
    
    bars = ax.bar(fields, values, color='skyblue', alpha=0.7)
    ax.set_title('Average Values by Metric')
    ax.set_ylabel('Average Value')
    ax.set_xlabel('Metric')
    
    # Rotate x-axis labels if too long
    if max(len(f) for f in fields) > 10:
        ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                f'{value:.2f}', ha='center', va='bottom')


def _create_line_plot(ax, results: List[Dict], numeric_fields: List[str]) -> None:
    """
    Create a line plot showing trends across experiments.
    
    Args:
        ax: Matplotlib axis object
        results: List of experiment result dictionaries
        numeric_fields: List of field names to plot
    """
    if not numeric_fields:
        ax.text(0.5, 0.5, 'No numeric fields found', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Trends Over Experiments')
        return
    
    # Select fields with highest variance if max_fields is not None
    if max_fields is not None:
        field_variances = {field: np.var([r.get(field) for r in results if isinstance(r.get(field), (int, float))]) for field in numeric_fields}
        selected_fields = sorted(field_variances, key=field_variances.get, reverse=True)[:max_fields]
    else:
        selected_fields = numeric_fields
    
    # Plot each selected numeric field as a line
    for field in selected_fields:
        values = []
        for r in results:
            value = r.get(field)
            if isinstance(value, (int, float)):
                values.append(value)
            else:
                values.append(None)
        
        # Filter out None values
        valid_indices = [i for i, v in enumerate(values) if v is not None]
        valid_values = [values[i] for i in valid_indices]
        
        if valid_values:
            ax.plot(valid_indices, valid_values, marker='o', label=field, linewidth=2)
    
    ax.set_title('Trends Over Experiments')
    ax.set_xlabel('Experiment Index')
    ax.set_ylabel('Metric Value')
    ax.legend()
    ax.grid(True, alpha=0.3)


def _create_summary_stats(ax, results: List[Dict]) -> None:
    """
    Create a summary statistics table.
    
    Args:
        ax: Matplotlib axis object
        results: List of experiment result dictionaries
    """
    ax.axis('off')
    
    # Calculate basic statistics
    total_experiments = len(results)
    
    # Count experiments with different statuses
    status_counts = {}
    for r in results:
        status = r.get('status', 'unknown')
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Create summary text
    summary_text = f"""
    Experiment Summary:
    
    Total Experiments: {total_experiments}
    
    Status Breakdown:
    """
    
    for status, count in status_counts.items():
        percentage = (count / total_experiments) * 100
        summary_text += f"\n{status}: {count} ({percentage:.1f}%)"
    
    # Add some common fields if they exist
    common_fields = ['model', 'dataset', 'method', 'version']
    for field in common_fields:
        unique_values = set()
        for r in results:
            if field in r:
                unique_values.add(str(r[field]))
        if unique_values:
            summary_text += f"\n\nUnique {field}s: {len(unique_values)}"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))


def _create_experiment_summary(ax, results: List[Dict]) -> None:
    """
    Create a pie chart showing experiment distribution.
    
    Args:
        ax: Matplotlib axis object
        results: List of experiment result dictionaries
    """
    # Count experiments by status
    status_counts = {}
    for r in results:
        status = r.get('status', 'unknown')
        status_counts[status] = status_counts.get(status, 0) + 1
    
    if not status_counts:
        ax.text(0.5, 0.5, 'No status data available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Experiment Distribution')
        return
    
    # Create pie chart
    labels = list(status_counts.keys())
    sizes = list(status_counts.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                       colors=colors, startangle=90)
    ax.set_title('Experiment Distribution by Status')
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')