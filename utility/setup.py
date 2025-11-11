#!/usr/bin/env python3
"""
setup.py
-------
Initial setup script for the cryo-EM processing pipeline.
Creates directory structure and validates environment.
"""

import os
import sys
import subprocess


def check_python_version():
    """Ensure Python 3.7+"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✓ Python version: {sys.version.split()[0]}")
    return True


def check_dependencies():
    """Check if required packages are installed"""
    required = ["numpy", "pandas", "mrcfile", "open3d", "sklearn"]
    missing = []
    
    for package in required:
        try:
            if package == "sklearn":
                __import__("sklearn")
            else:
                __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            print(f"✗ {package} not found")
            missing.append(package)
    
    return missing


def install_dependencies():
    """Install missing dependencies"""
    print("\n[INFO] Installing dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False


def create_directory_structure():
    """Create necessary directories"""
    directories = [
        "maps",
        "output",
        ".cache",
    ]
    
    print("\n[INFO] Creating directory structure...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created: {directory}/")
    
    return True


def create_example_config():
    """Create example configuration if needed"""
    config_file = "pipeline_config.py"
    if not os.path.exists(config_file):
        print(f"\n❌ {config_file} not found!")
        print("   Please ensure all pipeline files are present")
        return False
    print(f"✓ Configuration file: {config_file}")
    return True


def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Virtual environments
venv/
env/
ENV/

# Pipeline specific
.cache/
output/
maps/*.map
maps/*.mrc

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    print("✓ Created: .gitignore")


def print_next_steps():
    """Print instructions for next steps"""
    print("\n" + "="*70)
    print("SETUP COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Place your .map or .mrc files in the 'maps/' directory")
    print("  2. Run the pipeline:")
    print("     python run_pipeline.py")
    print("\nOptional:")
    print("  • Edit pipeline_config.py to customize parameters")
    print("  • Run python utils.py <dataset.csv> for analysis tools")
    print("="*70)


def main():
    print("="*70)
    print("CRYO-EM PIPELINE SETUP")
    print("="*70)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    print("\n[INFO] Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"\n[INFO] Missing packages: {', '.join(missing)}")
        response = input("Install missing dependencies? [Y/n]: ").strip().lower()
        
        if response in ['', 'y', 'yes']:
            if not install_dependencies():
                sys.exit(1)
        else:
            print("\n[INFO] Please install dependencies manually:")
            print("   pip install -r requirements.txt")
            sys.exit(1)
    
    # Create directories
    if not create_directory_structure():
        sys.exit(1)
    
    # Check config
    if not create_example_config():
        sys.exit(1)
    
    # Create .gitignore
    create_gitignore()
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)