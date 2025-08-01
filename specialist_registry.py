"""
Specialist Registry Module

This module provides a registry for managing callable specialist agents by role.
The SpecialistRegistry class allows registration, retrieval, and listing of
specialist handlers without using global state.
"""

from typing import Callable, Dict, List


class SpecialistRegistry:
    """
    A registry for managing callable specialist agents by role.
    
    This class allows multiple independent registry instances to be created,
    each maintaining its own set of registered specialists without global state.
    """
    
    def __init__(self) -> None:
        """Initialize an empty specialist registry."""
        self._specialists: Dict[str, Callable] = {}
    
    def register(self, role: str, handler: Callable) -> None:
        """
        Register a callable specialist handler for a specific role.
        
        Args:
            role: The role identifier for the specialist
            handler: The callable handler/function for this role
            
        Raises:
            ValueError: If role is empty or handler is not callable
        """
        if not role or not isinstance(role, str):
            raise ValueError("Role must be a non-empty string")
        
        if not callable(handler):
            raise ValueError("Handler must be callable")
            
        self._specialists[role] = handler
    
    def get(self, role: str) -> Callable:
        """
        Retrieve the callable handler for a specific role.
        
        Args:
            role: The role identifier to retrieve
            
        Returns:
            The callable handler registered for this role
            
        Raises:
            KeyError: If the role is not registered
            ValueError: If role is empty
        """
        if not role or not isinstance(role, str):
            raise ValueError("Role must be a non-empty string")
            
        if role not in self._specialists:
            raise KeyError(f"No specialist registered for role: {role}")
            
        return self._specialists[role]
    
    def list_roles(self) -> List[str]:
        """
        Get a list of all registered role identifiers.
        
        Returns:
            A list of all registered role strings, sorted alphabetically
        """
        return sorted(self._specialists.keys())