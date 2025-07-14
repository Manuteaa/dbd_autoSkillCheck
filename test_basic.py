"""
Basic tests for DBD Auto Skill Check application.

These tests validate the core functionality and improvements made to the codebase.
"""

import unittest
import os
import sys
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestConfiguration(unittest.TestCase):
    """Test configuration management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_config.json')
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        os.rmdir(self.temp_dir)
    
    def test_config_defaults(self):
        """Test that default configuration values are reasonable."""
        from config import AppConfig
        
        config = AppConfig()
        
        # Test reasonable defaults
        self.assertEqual(config.models_folder, "models")
        self.assertEqual(config.default_cpu_threads, 4)
        self.assertGreaterEqual(config.target_fps, 30)
        self.assertLessEqual(config.target_fps, 120)
        self.assertGreaterEqual(config.default_ante_delay, 0)
    
    def test_config_validation(self):
        """Test configuration validation."""
        from config import AppConfig
        
        # Test invalid CPU threads
        with self.assertRaises(ValueError):
            AppConfig(default_cpu_threads=0)
        
        with self.assertRaises(ValueError):
            AppConfig(default_cpu_threads=20)
        
        # Test invalid ante delay
        with self.assertRaises(ValueError):
            AppConfig(default_ante_delay=-5)
        
        with self.assertRaises(ValueError):
            AppConfig(default_ante_delay=100)
    
    def test_config_manager_save_load(self):
        """Test configuration saving and loading."""
        from config import ConfigManager, AppConfig
        
        # Create config manager with test path
        manager = ConfigManager(self.config_path)
        
        # Modify some values
        manager.config.default_cpu_threads = 6
        manager.config.target_fps = 90
        
        # Save configuration
        self.assertTrue(manager.save_config())
        self.assertTrue(os.path.exists(self.config_path))
        
        # Load configuration in new manager
        new_manager = ConfigManager(self.config_path)
        
        # Verify values were loaded correctly
        self.assertEqual(new_manager.config.default_cpu_threads, 6)
        self.assertEqual(new_manager.config.target_fps, 90)


class TestMonitorUtils(unittest.TestCase):
    """Test monitor utility functions."""
    
    @patch('dbd.utils.monitor.mss')
    def test_get_monitors(self, mock_mss):
        """Test monitor detection."""
        from dbd.utils.monitor import get_monitors
        
        # Mock MSS response
        mock_sct = Mock()
        mock_sct.monitors = [
            {},  # "All in One" monitor (skipped)
            {'width': 1920, 'height': 1080},
            {'width': 2560, 'height': 1440}
        ]
        mock_mss.return_value.__enter__.return_value = mock_sct
        
        monitors = get_monitors()
        
        # Should return 2 monitors (skipping the first one)
        self.assertEqual(len(monitors), 2)
        self.assertIn("1920x1080", monitors[0][0])
        self.assertIn("2560x1440", monitors[1][0])
    
    @patch('dbd.utils.monitor.mss')
    def test_get_monitor_attributes(self, mock_mss):
        """Test monitor attributes calculation."""
        from dbd.utils.monitor import get_monitor_attributes
        
        # Mock MSS response
        mock_sct = Mock()
        mock_sct.monitors = [
            {},  # "All in One" monitor
            {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
        ]
        mock_mss.return_value.__enter__.return_value = mock_sct
        
        attrs = get_monitor_attributes(monitor_id=1, crop_size=224)
        
        # Verify attributes are calculated correctly
        self.assertIn('top', attrs)
        self.assertIn('left', attrs)
        self.assertIn('width', attrs)
        self.assertIn('height', attrs)
        
        # Verify centering (approximately)
        expected_size = int(224 / 1080 * 1080)  # Should be 224
        self.assertEqual(attrs['width'], expected_size)
        self.assertEqual(attrs['height'], expected_size)
    
    def test_invalid_monitor_id(self):
        """Test error handling for invalid monitor ID."""
        from dbd.utils.monitor import get_monitor_attributes
        
        with self.assertRaises(ValueError):
            get_monitor_attributes(monitor_id=0)


class TestAIModelBasics(unittest.TestCase):
    """Test basic AI model functionality (without actual model loading)."""
    
    def test_prediction_dict_completeness(self):
        """Test that prediction dictionary is complete."""
        from dbd.AI_model import AI_model
        
        # Verify all expected categories are present
        expected_categories = 11  # 0-10
        self.assertEqual(len(AI_model.pred_dict), expected_categories)
        
        # Verify all entries have required fields
        for i in range(expected_categories):
            self.assertIn(i, AI_model.pred_dict)
            self.assertIn('desc', AI_model.pred_dict[i])
            self.assertIn('hit', AI_model.pred_dict[i])
            self.assertIsInstance(AI_model.pred_dict[i]['hit'], bool)
    
    def test_normalization_constants(self):
        """Test that normalization constants are reasonable."""
        from dbd.AI_model import AI_model
        import numpy as np
        
        # ImageNet normalization constants should be in reasonable ranges
        self.assertEqual(len(AI_model.MEAN), 3)  # RGB channels
        self.assertEqual(len(AI_model.STD), 3)
        
        # Values should be between 0 and 1
        self.assertTrue(np.all(AI_model.MEAN >= 0))
        self.assertTrue(np.all(AI_model.MEAN <= 1))
        self.assertTrue(np.all(AI_model.STD > 0))
        self.assertTrue(np.all(AI_model.STD <= 1))
    
    def test_softmax_function(self):
        """Test softmax implementation."""
        from dbd.AI_model import AI_model
        import numpy as np
        
        model = AI_model.__new__(AI_model)  # Create without __init__
        
        # Test softmax with simple input
        logits = np.array([1.0, 2.0, 3.0])
        probs = model.softmax(logits)
        
        # Probabilities should sum to 1
        self.assertAlmostEqual(np.sum(probs), 1.0, places=6)
        
        # All probabilities should be positive
        self.assertTrue(np.all(probs > 0))
        
        # Largest logit should have highest probability
        self.assertEqual(np.argmax(probs), 2)


class TestInputValidation(unittest.TestCase):
    """Test input validation throughout the application."""
    
    def test_config_validation_edge_cases(self):
        """Test configuration validation with edge cases."""
        from config import AppConfig
        
        # Test boundary values
        AppConfig(default_cpu_threads=1)  # Should work
        AppConfig(default_cpu_threads=8)  # Should work
        AppConfig(default_ante_delay=0)   # Should work
        AppConfig(default_ante_delay=50)  # Should work
        
        # Test just outside boundaries
        with self.assertRaises(ValueError):
            AppConfig(default_cpu_threads=9)
        
        with self.assertRaises(ValueError):
            AppConfig(default_ante_delay=51)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and logging."""
    
    @patch('dbd.utils.monitor.mss')
    def test_monitor_error_handling(self, mock_mss):
        """Test error handling in monitor utilities."""
        from dbd.utils.monitor import get_monitors, get_frame
        
        # Test MSS failure
        mock_mss.side_effect = Exception("MSS failed")
        
        with self.assertRaises(RuntimeError):
            get_monitors()
    
    def test_file_not_found_handling(self):
        """Test handling of missing files."""
        from dbd.AI_model import AI_model
        
        # Test with non-existent model file
        with self.assertRaises(FileNotFoundError):
            AI_model("non_existent_model.onnx")


class TestResourceCleanup(unittest.TestCase):
    """Test proper resource cleanup."""
    
    def test_ai_model_context_manager(self):
        """Test AI model context manager cleanup."""
        from dbd.AI_model import AI_model
        
        # Mock the AI_model to avoid actual model loading
        with patch.object(AI_model, '__init__', return_value=None):
            with patch.object(AI_model, 'cleanup') as mock_cleanup:
                model = AI_model.__new__(AI_model)
                model.cleanup = mock_cleanup
                
                with model:
                    pass  # Use model in context
                
                # Cleanup should have been called
                mock_cleanup.assert_called_once()


def run_basic_tests():
    """Run basic tests to validate the improvements."""
    print("Running basic functionality tests...")
    
    # Test imports
    try:
        import dbd.AI_model
        import dbd.utils.monitor
        import config
        print("✓ All modules can be imported")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    # Test configuration
    try:
        from config import AppConfig, ConfigManager
        config = AppConfig()
        config.validate()
        print("✓ Configuration system works")
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False
    
    # Test monitor utilities (with mocking to avoid system dependencies)
    try:
        from unittest.mock import patch
        with patch('dbd.utils.monitor.mss'):
            from dbd.utils.monitor import get_monitor_attributes
            print("✓ Monitor utilities can be imported")
    except Exception as e:
        print(f"✗ Monitor utilities error: {e}")
        return False
    
    print("✓ All basic tests passed")
    return True


if __name__ == "__main__":
    # Run basic tests first
    if run_basic_tests():
        print("\nRunning detailed unit tests...")
        unittest.main(verbosity=2)
    else:
        print("Basic tests failed, skipping unit tests")
        sys.exit(1)