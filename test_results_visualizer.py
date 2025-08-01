"""
Tests for the results_visualizer module.

This module tests the visualization functionality with various types of experiment data.
"""

import pytest
import os
import tempfile
import shutil
from results_visualizer import visualize


class TestResultsVisualizer:
    """Test class for the results visualizer functionality."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test environment after each test."""
        # Remove temporary directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_visualize_with_numeric_data(self):
        """Test visualization with numeric experiment data."""
        # Create dummy experiment results with numeric fields
        results = [
            {
                'accuracy': 0.85,
                'loss': 0.15,
                'precision': 0.82,
                'recall': 0.88,
                'status': 'completed',
                'model': 'resnet50',
                'dataset': 'cifar10'
            },
            {
                'accuracy': 0.87,
                'loss': 0.13,
                'precision': 0.84,
                'recall': 0.86,
                'status': 'completed',
                'model': 'resnet50',
                'dataset': 'cifar10'
            },
            {
                'accuracy': 0.89,
                'loss': 0.11,
                'precision': 0.86,
                'recall': 0.90,
                'status': 'completed',
                'model': 'resnet50',
                'dataset': 'cifar10'
            }
        ]
        
        out_path = os.path.join(self.test_dir, 'test_numeric.png')
        
        # Should not raise any exceptions
        visualize(results, out_path)
        
        # Check that file was created
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 0
    
    def test_visualize_with_mixed_data(self):
        """Test visualization with mixed data types."""
        results = [
            {
                'accuracy': 0.75,
                'model': 'vgg16',
                'status': 'completed',
                'dataset': 'imagenet',
                'epochs': 100,
                'batch_size': 32
            },
            {
                'accuracy': 0.78,
                'model': 'vgg16',
                'status': 'failed',
                'dataset': 'imagenet',
                'epochs': 100,
                'batch_size': 64
            },
            {
                'accuracy': 0.80,
                'model': 'vgg16',
                'status': 'completed',
                'dataset': 'imagenet',
                'epochs': 150,
                'batch_size': 32
            }
        ]
        
        out_path = os.path.join(self.test_dir, 'test_mixed.png')
        
        visualize(results, out_path)
        
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 0
    
    def test_visualize_with_string_only_data(self):
        """Test visualization with only string/categorical data."""
        results = [
            {
                'model': 'resnet50',
                'status': 'completed',
                'dataset': 'cifar10',
                'method': 'supervised'
            },
            {
                'model': 'vgg16',
                'status': 'failed',
                'dataset': 'imagenet',
                'method': 'supervised'
            },
            {
                'model': 'resnet50',
                'status': 'completed',
                'dataset': 'cifar10',
                'method': 'self-supervised'
            }
        ]
        
        out_path = os.path.join(self.test_dir, 'test_string_only.png')
        
        # Should work even without numeric data
        visualize(results, out_path)
        
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 0
    
    def test_visualize_empty_results(self):
        """Test that empty results list raises ValueError."""
        results = []
        out_path = os.path.join(self.test_dir, 'test_empty.png')
        
        with pytest.raises(ValueError, match="Results list cannot be empty"):
            visualize(results, out_path)
    
    def test_visualize_single_result(self):
        """Test visualization with a single experiment result."""
        results = [
            {
                'accuracy': 0.85,
                'loss': 0.15,
                'status': 'completed',
                'model': 'resnet50'
            }
        ]
        
        out_path = os.path.join(self.test_dir, 'test_single.png')
        
        visualize(results, out_path)
        
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 0
    
    def test_visualize_large_dataset(self):
        """Test visualization with a larger dataset."""
        results = []
        for i in range(20):
            results.append({
                'accuracy': 0.70 + (i * 0.01),
                'loss': 0.30 - (i * 0.005),
                'precision': 0.68 + (i * 0.008),
                'recall': 0.72 + (i * 0.012),
                'status': 'completed' if i % 3 != 0 else 'failed',
                'model': f'model_{i % 3}',
                'dataset': 'test_dataset'
            })
        
        out_path = os.path.join(self.test_dir, 'test_large.png')
        
        visualize(results, out_path)
        
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 0
    
    def test_visualize_with_missing_fields(self):
        """Test visualization with inconsistent field presence."""
        results = [
            {
                'accuracy': 0.85,
                'loss': 0.15,
                'status': 'completed'
            },
            {
                'accuracy': 0.87,
                'precision': 0.84,  # Missing loss
                'status': 'completed'
            },
            {
                'loss': 0.11,  # Missing accuracy
                'precision': 0.86,
                'status': 'completed'
            }
        ]
        
        out_path = os.path.join(self.test_dir, 'test_missing_fields.png')
        
        visualize(results, out_path)
        
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 0
    
    def test_visualize_with_boolean_fields(self):
        """Test visualization with boolean fields (should be ignored for numeric plots)."""
        results = [
            {
                'accuracy': 0.85,
                'is_trained': True,
                'is_evaluated': False,
                'status': 'completed'
            },
            {
                'accuracy': 0.87,
                'is_trained': True,
                'is_evaluated': True,
                'status': 'completed'
            }
        ]
        
        out_path = os.path.join(self.test_dir, 'test_boolean.png')
        
        visualize(results, out_path)
        
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 0
    
    def test_visualize_file_extension_handling(self):
        """Test visualization with different file extensions."""
        results = [
            {
                'accuracy': 0.85,
                'status': 'completed'
            }
        ]
        
        # Test different file extensions
        extensions = ['.png', '.jpg', '.pdf', '.svg']
        
        for ext in extensions:
            out_path = os.path.join(self.test_dir, f'test{ext}')
            visualize(results, out_path)
            assert os.path.exists(out_path)
            assert os.path.getsize(out_path) > 0
    
    def test_visualize_directory_creation(self):
        """Test that the function creates directories if they don't exist."""
        results = [
            {
                'accuracy': 0.85,
                'status': 'completed'
            }
        ]
        
        # Create a path with non-existent subdirectories
        nested_dir = os.path.join(self.test_dir, 'nested', 'subdirectory')
        out_path = os.path.join(nested_dir, 'test.png')
        
        visualize(results, out_path)
        
        # Check that directory was created and file was saved
        assert os.path.exists(nested_dir)
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 0


if __name__ == "__main__":
    pytest.main([__file__])