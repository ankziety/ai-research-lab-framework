"""
Unit tests for the SpecialistRegistry module.

This module contains comprehensive tests for the SpecialistRegistry class,
testing all methods and edge cases.
"""

import unittest
from unittest.mock import Mock
from specialist_registry import SpecialistRegistry


class TestSpecialistRegistry(unittest.TestCase):
    """Test cases for the SpecialistRegistry class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.registry = SpecialistRegistry()
    
    def test_initialization(self):
        """Test that registry initializes with empty state."""
        self.assertEqual(self.registry.list_roles(), [])
    
    def test_register_valid_handler(self):
        """Test registering a valid callable handler."""
        def test_handler():
            return "test"
        
        self.registry.register("researcher", test_handler)
        self.assertIn("researcher", self.registry.list_roles())
    
    def test_register_lambda_handler(self):
        """Test registering a lambda function as handler."""
        lambda_handler = lambda x: x * 2
        
        self.registry.register("calculator", lambda_handler)
        self.assertEqual(self.registry.get("calculator"), lambda_handler)
    
    def test_register_mock_handler(self):
        """Test registering a mock object as handler."""
        mock_handler = Mock()
        
        self.registry.register("mock_role", mock_handler)
        self.assertEqual(self.registry.get("mock_role"), mock_handler)
    
    def test_register_class_method(self):
        """Test registering a class method as handler."""
        class TestClass:
            def method(self):
                return "method_result"
        
        test_instance = TestClass()
        self.registry.register("class_method", test_instance.method)
        
        retrieved_handler = self.registry.get("class_method")
        self.assertEqual(retrieved_handler(), "method_result")
    
    def test_register_empty_role_raises_error(self):
        """Test that registering with empty role raises ValueError."""
        def test_handler():
            pass
        
        with self.assertRaises(ValueError) as context:
            self.registry.register("", test_handler)
        
        self.assertIn("non-empty string", str(context.exception))
    
    def test_register_none_role_raises_error(self):
        """Test that registering with None role raises ValueError."""
        def test_handler():
            pass
        
        with self.assertRaises(ValueError) as context:
            self.registry.register(None, test_handler)
        
        self.assertIn("non-empty string", str(context.exception))
    
    def test_register_non_string_role_raises_error(self):
        """Test that registering with non-string role raises ValueError."""
        def test_handler():
            pass
        
        with self.assertRaises(ValueError) as context:
            self.registry.register(123, test_handler)
        
        self.assertIn("non-empty string", str(context.exception))
    
    def test_register_non_callable_raises_error(self):
        """Test that registering non-callable handler raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.registry.register("role", "not_callable")
        
        self.assertIn("must be callable", str(context.exception))
    
    def test_register_none_handler_raises_error(self):
        """Test that registering None handler raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.registry.register("role", None)
        
        self.assertIn("must be callable", str(context.exception))
    
    def test_get_existing_role(self):
        """Test retrieving an existing registered role."""
        def test_handler():
            return "success"
        
        self.registry.register("researcher", test_handler)
        retrieved_handler = self.registry.get("researcher")
        
        self.assertEqual(retrieved_handler, test_handler)
        self.assertEqual(retrieved_handler(), "success")
    
    def test_get_nonexistent_role_raises_error(self):
        """Test that getting non-existent role raises KeyError."""
        with self.assertRaises(KeyError) as context:
            self.registry.get("nonexistent")
        
        self.assertIn("No specialist registered for role: nonexistent", str(context.exception))
    
    def test_get_empty_role_raises_error(self):
        """Test that getting with empty role raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.registry.get("")
        
        self.assertIn("non-empty string", str(context.exception))
    
    def test_get_none_role_raises_error(self):
        """Test that getting with None role raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.registry.get(None)
        
        self.assertIn("non-empty string", str(context.exception))
    
    def test_get_non_string_role_raises_error(self):
        """Test that getting with non-string role raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.registry.get(123)
        
        self.assertIn("non-empty string", str(context.exception))
    
    def test_list_roles_empty(self):
        """Test listing roles when registry is empty."""
        self.assertEqual(self.registry.list_roles(), [])
    
    def test_list_roles_single(self):
        """Test listing roles with single registered role."""
        def test_handler():
            pass
        
        self.registry.register("researcher", test_handler)
        self.assertEqual(self.registry.list_roles(), ["researcher"])
    
    def test_list_roles_multiple_sorted(self):
        """Test that multiple roles are returned sorted alphabetically."""
        def handler1():
            pass
        def handler2():
            pass
        def handler3():
            pass
        
        # Register in non-alphabetical order
        self.registry.register("zebra", handler1)
        self.registry.register("alpha", handler2)
        self.registry.register("beta", handler3)
        
        roles = self.registry.list_roles()
        self.assertEqual(roles, ["alpha", "beta", "zebra"])
    
    def test_overwrite_existing_role(self):
        """Test that registering same role overwrites previous handler."""
        def handler1():
            return "first"
        def handler2():
            return "second"
        
        self.registry.register("researcher", handler1)
        self.registry.register("researcher", handler2)
        
        # Should have overwritten with second handler
        retrieved = self.registry.get("researcher")
        self.assertEqual(retrieved(), "second")
        
        # Should still only have one role
        self.assertEqual(len(self.registry.list_roles()), 1)
    
    def test_multiple_independent_registries(self):
        """Test that multiple registry instances are independent."""
        registry1 = SpecialistRegistry()
        registry2 = SpecialistRegistry()
        
        def handler1():
            return "registry1"
        def handler2():
            return "registry2"
        
        registry1.register("role", handler1)
        registry2.register("role", handler2)
        
        # Each registry should have its own handler
        self.assertEqual(registry1.get("role")(), "registry1")
        self.assertEqual(registry2.get("role")(), "registry2")
        
        # Changes to one shouldn't affect the other
        registry1.register("new_role", handler1)
        self.assertIn("new_role", registry1.list_roles())
        self.assertNotIn("new_role", registry2.list_roles())
    
    def test_no_global_state(self):
        """Test that registry instances don't share global state."""
        # Create first registry and register a role
        registry1 = SpecialistRegistry()
        def handler1():
            return "first"
        registry1.register("shared_role", handler1)
        
        # Create second registry - should be empty
        registry2 = SpecialistRegistry()
        self.assertEqual(registry2.list_roles(), [])
        
        # Registering in second registry shouldn't affect first
        def handler2():
            return "second"
        registry2.register("shared_role", handler2)
        
        # Each should maintain its own state
        self.assertEqual(registry1.get("shared_role")(), "first")
        self.assertEqual(registry2.get("shared_role")(), "second")


if __name__ == "__main__":
    unittest.main()