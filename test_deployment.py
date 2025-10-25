#!/usr/bin/env python3
"""
Test script to verify deployment configuration.
Run this locally to test if the app can be imported and started.
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported."""
    try:
        print("Testing imports...")
        import flask
        import gunicorn
        import numpy
        import pandas
        import sklearn
        import matplotlib
        import seaborn
        import joblib
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_app_import():
    """Test if the app can be imported."""
    try:
        print("Testing app import...")
        from app import app
        print("✓ App imported successfully")
        return True
    except Exception as e:
        print(f"✗ App import error: {e}")
        return False

def test_wsgi_import():
    """Test if wsgi can be imported."""
    try:
        print("Testing wsgi import...")
        from wsgi import app
        print("✓ WSGI imported successfully")
        return True
    except Exception as e:
        print(f"✗ WSGI import error: {e}")
        return False

if __name__ == "__main__":
    print("=== Deployment Test ===")
    
    success = True
    success &= test_imports()
    success &= test_app_import()
    success &= test_wsgi_import()
    
    if success:
        print("\n✓ All tests passed! Deployment should work.")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. Check the errors above.")
        sys.exit(1)
