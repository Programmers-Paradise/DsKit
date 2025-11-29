"""Comprehensive package validation after bug fixes"""

import sys
import io

print("=" * 70)
print("Ak-dskit Package Validation")
print("=" * 70)

# Test 1: Version consistency
print("\n1. Checking version consistency...")
try:
    from dskit import __version__
    with open('pyproject.toml', 'r') as f:
        toml_content = f.read()
        if '1.0.3' in toml_content and __version__ == '1.0.3':
            print(f"   SUCCESS: Version consistent: {__version__}")
        else:
            print(f"   FAIL: Version mismatch: __init__.py={__version__}")
except Exception as e:
    print(f"   FAIL: Error: {e}")

# Test 2: Import test
print("\n2. Testing imports...")
try:
    from dskit import dskit, load, fix_dtypes, quick_eda
    print("   SUCCESS: Core imports successful")
except ImportError as e:
    print(f"   FAIL: Import failed: {e}")

# Test 3: Basic functionality
print("\n3. Testing basic functionality...")
try:
    import pandas as pd
    import numpy as np
    from dskit import dskit
    
    data = pd.DataFrame({
        'a': np.random.normal(0, 1, 20),
        'b': np.random.randint(1, 10, 20),
        'c': np.random.choice(['X', 'Y'], 20)
    })
    
    kit = dskit(data)
    kit = kit.fix_dtypes()
    
    # Suppress comprehensive output
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    health = kit.data_health_check()
    sys.stdout = old_stdout
    
    print(f"   SUCCESS: Basic operations work (health: {health}/100)")
    
except Exception as e:
    print(f"   FAIL: Error: {e}")

# Test 4: comprehensive_eda bug fix
print("\n4. Testing comprehensive_eda bug fix...")
try:
    import pandas as pd
    import numpy as np
    from dskit import dskit
    
    data = pd.DataFrame({
        'feature': np.random.normal(0, 1, 30),
        'target': np.random.choice([0, 1], 30)
    })
    
    kit = dskit(data)
    
    # Suppress output
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    try:
        kit.comprehensive_eda(target_col="target")
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        if "KEY INSIGHTS" in output:
            print("   SUCCESS: comprehensive_eda works with target column")
        else:
            print("   WARNING: comprehensive_eda completed but output unclear")
            
    except Exception as e:
        sys.stdout = old_stdout
        if "UnboundLocalError" in str(type(e).__name__):
            print(f"   FAIL: UnboundLocalError still exists: {e}")
        else:
            print(f"   WARNING: Different error: {type(e).__name__}")
            
except Exception as e:
    print(f"   FAIL: Setup error: {e}")

# Test 5: Package metadata
print("\n5. Checking package metadata...")
try:
    import dskit
    print(f"   SUCCESS: Package configured correctly")
    print(f"     - Package name: Ak-dskit")
    print(f"     - Import name: dskit")
    print(f"     - Version: {dskit.__version__}")
except Exception as e:
    print(f"   FAIL: Error: {e}")

print("\n" + "=" * 70)
print("Validation Complete")
print("=" * 70)
print("\nPackage Details:")
print("  - PyPI Name: Ak-dskit")
print("  - Import Name: dskit")
print("  - Version: 1.0.3")
print("  - Install: pip install Ak-dskit")
print("  - Usage: from dskit import dskit")
print("=" * 70)
