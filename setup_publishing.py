#!/usr/bin/env python3
"""
Setup script for dskit PyPI publishing.
Installs required tools and validates the package.
"""

import subprocess
import sys

def install_tools():
    """Install required publishing tools."""
    print("📦 Installing PyPI publishing tools...")
    
    tools = [
        "build",
        "twine",
        "wheel",
        "setuptools>=61.0"
    ]
    
    for tool in tools:
        try:
            print(f"Installing {tool}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", tool])
            print(f"✅ {tool} installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {tool}: {e}")
            return False
    
    return True

def main():
    """Main setup function."""
    print("🚀 dskit PyPI Publishing Setup")
    print("=" * 40)
    
    if install_tools():
        print("\n✅ All tools installed successfully!")
        print("\nNext steps:")
        print("1. Run validation: python validate_package_fixed.py")
        print("2. Build package: python -m build")
        print("3. Test upload: python -m twine upload --repository testpypi dist/*")
        print("4. Production upload: python -m twine upload dist/*")
        print("\nSee PUBLISHING_GUIDE.md for detailed instructions.")
    else:
        print("\n❌ Setup failed. Please check the errors above.")

if __name__ == "__main__":
    main()