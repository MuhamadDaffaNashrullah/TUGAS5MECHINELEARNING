#!/usr/bin/env python3
"""
Test deployment configuration.
"""

import os
import sys

def test_deployment():
    """Test deployment configuration."""
    print("=== Deployment Test ===")
    
    # Test 1: Check if app can be imported
    try:
        print("Testing app import...")
        from app import app
        print("✓ App imported successfully")
    except Exception as e:
        print(f"✗ App import failed: {e}")
        return False
    
    # Test 2: Check if wsgi can be imported
    try:
        print("Testing wsgi import...")
        from wsgi import app as wsgi_app
        print("✓ WSGI imported successfully")
    except Exception as e:
        print(f"✗ WSGI import failed: {e}")
        return False
    
    # Test 3: Check if Flask app is callable
    try:
        print("Testing Flask app...")
        with app.test_client() as client:
            response = client.get('/')
            print(f"✓ Flask app working (status: {response.status_code})")
    except Exception as e:
        print(f"✗ Flask app test failed: {e}")
        return False
    
    # Test 4: Check Procfile
    if os.path.exists('Procfile'):
        with open('Procfile', 'r') as f:
            procfile_content = f.read().strip()
        print(f"✓ Procfile found: {procfile_content}")
    else:
        print("✗ Procfile not found")
        return False
    
    # Test 5: Check requirements
    if os.path.exists('requirements.txt'):
        with open('requirements.txt', 'r') as f:
            requirements = f.read().strip().split('\n')
        print(f"✓ Requirements found: {len(requirements)} packages")
    else:
        print("✗ requirements.txt not found")
        return False
    
    print("\n✓ All deployment tests passed!")
    return True

if __name__ == "__main__":
    success = test_deployment()
    sys.exit(0 if success else 1)
