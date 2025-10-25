#!/usr/bin/env python3
"""
Startup script to pre-load the model and verify everything is working.
This can be run during deployment to ensure the model loads correctly.
"""

import os
import sys

def startup_check():
    """Perform startup checks and pre-load the model."""
    print("=== Startup Check ===")
    
    # Check if we're in the right directory
    print(f"Working directory: {os.getcwd()}")
    print(f"Files in directory: {os.listdir('.')}")
    
    # Check model file
    model_path = 'decision_tree_modelB.joblib'
    if os.path.exists(model_path):
        print(f"✓ Model file found: {model_path}")
    else:
        print(f"✗ Model file not found: {model_path}")
        return False
    
    # Check static directory
    static_dir = 'static'
    if os.path.exists(static_dir):
        print(f"✓ Static directory found: {static_dir}")
    else:
        print(f"✗ Static directory not found: {static_dir}")
        return False
    
    # Check plots directory
    plots_dir = os.path.join(static_dir, 'plots')
    if os.path.exists(plots_dir):
        print(f"✓ Plots directory found: {plots_dir}")
        plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
        print(f"✓ Found {len(plot_files)} plot files")
    else:
        print(f"✗ Plots directory not found: {plots_dir}")
        return False
    
    # Try to import and load the model
    try:
        print("Testing model loading...")
        import joblib
        model = joblib.load(model_path)
        print("✓ Model loaded successfully")
        
        # Test a simple prediction
        import numpy as np
        test_data = np.array([[57, 1, 0, 130, 250, 0, 1, 150, 0, 1.0, 1, 0, 2]])
        prediction = model.predict(test_data)
        print(f"✓ Model prediction test successful: {prediction[0]}")
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False
    
    print("=== Startup Check Complete ===")
    return True

if __name__ == "__main__":
    success = startup_check()
    sys.exit(0 if success else 1)
