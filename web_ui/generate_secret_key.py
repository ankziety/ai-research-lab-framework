#!/usr/bin/env python3
"""
Generate a secure SECRET_KEY for Flask application.
Use this script to generate a secret key for production deployment.
"""

import secrets
import sys

def generate_secret_key():
    """Generate a secure random secret key."""
    return secrets.token_urlsafe(32)

def main():
    """Generate and display a secret key."""
    print("Generating secure SECRET_KEY for Flask application...")
    print()
    
    secret_key = generate_secret_key()
    
    print("Generated SECRET_KEY:")
    print(f"export SECRET_KEY='{secret_key}'")
    print()
    print("To use this in production:")
    print("1. Copy the export command above")
    print("2. Add it to your environment or .env file")
    print("3. Restart your Flask application")
    print()
    print("For .env file, add this line:")
    print(f"SECRET_KEY={secret_key}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 