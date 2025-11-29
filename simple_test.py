#!/usr/bin/env python3
"""
Simple test script to verify dskit installation and basic functionality.
Run this to check if dskit is working correctly on your system.
"""

import sys
import pandas as pd
import numpy as np

def test_basic_imports():
    """Test basic dskit imports."""
    print("🔍 Testing dskit imports...")
    try:
        from dskit import dskit, load, fix_dtypes
        print("✅ Basic imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic dskit functionality with sample data."""
    print("\n🧪 Testing basic functionality...")
    try:
        from dskit import dskit
        
        # Create sample data
        np.random.seed(42)
        data = {
            'numeric_col': np.random.normal(100, 15, 100),
            'category_col': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.choice([0, 1], 100)
        }
        df = pd.DataFrame(data)
        
        # Test dskit operations
        kit = dskit(df)
        print(f"✅ Created dskit object with shape: {kit.df.shape}")
        
        # Test basic operations
        kit = kit.fix_dtypes()
        print("✅ fix_dtypes() works")
        
        health_score = kit.data_health_check()
        print(f"✅ data_health_check() works: {health_score:.1f}/100")
        
        kit = kit.fill_missing(strategy='mean')
        print("✅ fill_missing() works")
        
        return True
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def test_advanced_features():
    """Test some advanced features."""
    print("\n🚀 Testing advanced features...")
    try:
        from dskit import dskit
        
        # Sample data with date
        data = {
            'value': np.random.normal(50, 10, 50),
            'category': np.random.choice(['X', 'Y'], 50),
            'date_col': pd.date_range('2023-01-01', periods=50),
            'target': np.random.choice([0, 1], 50)
        }
        df = pd.DataFrame(data)
        kit = dskit(df)
        
        # Test date features
        kit = kit.create_date_features(['date_col'])
        print("✅ create_date_features() works")
        
        # Test EDA
        health = kit.data_health_check()
        print(f"✅ Advanced data health check: {health:.1f}/100")
        
        return True
    except Exception as e:
        print(f"⚠️ Some advanced features may not work: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 dskit Installation Test")
    print("=" * 40)
    
    tests = [
        test_basic_imports,
        test_basic_functionality,
        test_advanced_features
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed >= 2:
        print("🎉 dskit is working! You can start using it.")
        print("\n📖 Try this simple example:")
        print("""
from dskit import dskit
import pandas as pd
import numpy as np

# Create sample data
data = {'x': np.random.normal(0, 1, 100), 'y': np.random.normal(0, 1, 100)}
df = pd.DataFrame(data)

# Use dskit
kit = dskit(df)
health = kit.data_health_check()
print(f"Data health: {health}/100")
        """)
    else:
        print("⚠️ dskit may have issues. Check the error messages above.")

if __name__ == "__main__":
    main()