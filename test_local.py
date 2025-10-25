#!/usr/bin/env python3
"""
Test script to run the app locally and check for issues.
"""

import os
import sys

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_app_startup():
    """Test if the app can start up properly."""
    try:
        print("Testing app startup...")
        from app import app, model, get_plot_urls
        
        print(f"✓ App imported successfully")
        print(f"✓ Model loaded: {model is not None}")
        
        # Test plots
        plot_urls = get_plot_urls()
        print(f"✓ Found {len(plot_urls)} plot URLs")
        
        # Test health endpoint
        with app.test_client() as client:
            response = client.get('/health')
            if response.status_code == 200:
                print("✓ Health endpoint working")
                print(f"Health data: {response.get_json()}")
            else:
                print(f"✗ Health endpoint failed: {response.status_code}")
        
        print("\n✓ All tests passed! App should work in deployment.")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_app_startup()
    sys.exit(0 if success else 1)
