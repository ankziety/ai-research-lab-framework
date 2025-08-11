#!/usr/bin/env python3
"""
Gradio Client Utils Patch

This module provides a monkey patch to fix the JSON schema parsing issue
in gradio_client.utils.py where boolean values in additionalProperties
are not handled correctly.
"""

import gradio_client.utils as client_utils
from typing import Any

def patched_get_type(schema: Any) -> str:
    """
    Patched version of get_type that handles boolean values in JSON Schema.
    
    The original function expects schema to be a dict, but JSON Schema allows
    boolean values for properties like additionalProperties: true.
    """
    # Handle boolean values (valid in JSON Schema)
    if isinstance(schema, bool):
        return "boolean"
    
    # Handle dict values (original behavior)
    if isinstance(schema, dict):
        if "const" in schema:
            return "const"
        if "enum" in schema:
            return "enum"
        elif "type" in schema:
            return schema["type"]
        elif schema.get("$ref"):
            return "$ref"
        elif schema.get("oneOf"):
            return "oneOf"
        elif schema.get("anyOf"):
            return "anyOf"
        elif schema.get("allOf"):
            return "allOf"
        elif "type" not in schema:
            return {}
        else:
            raise client_utils.APIInfoParseError(f"Cannot parse type for {schema}")
    
    # Handle unexpected types
    raise client_utils.APIInfoParseError(f"Unexpected schema type: {type(schema)} with value {schema}")

def apply_gradio_patch():
    """
    Apply the monkey patch to fix the gradio_client utils issue.
    
    This should be called before creating any Gradio interfaces.
    """
    # Replace the problematic function
    client_utils.get_type = patched_get_type
    print("✅ Applied Gradio client utils patch for boolean schema handling")

def remove_gradio_patch():
    """
    Remove the monkey patch (restore original function).
    
    Note: This would require reloading the module to fully restore.
    """
    # We can't easily restore the original without reloading the module
    print("⚠️  Patch removal requires module reload - not implemented")
