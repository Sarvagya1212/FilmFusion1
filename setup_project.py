#!/usr/bin/env python3
"""
Initial project setup script for FilmFusion
Location: scripts/setup_project.py
"""

import os
import subprocess
import sys
from pathlib import Path

def create_directory_structure():
    """Create the complete FilmFusion directory structure"""
    
    directories = [
        "config", "data/raw", "data/processed", "data/external", "data/schemas",
        "src/data_ingestion", "src/data_processing", "src/database", "src/utils",
        "notebooks", "tests", "airflow/dags", "airflow/plugins", "airflow/config",
        "scripts", "docs", "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        # Create __init__.py files for Python packages
        if directory.startswith('src/'):
            (Path(directory) / '__init__.py').touch()
    
    print("✅ Directory structure created successfully!")

def setup_git_hooks():
    """Setup Git hooks and branching strategy"""
    
    git_commands = [
        "git flow init -d",  # Initialize GitFlow with defaults
        "git branch develop",
        "git checkout develop"
    ]
    
    for cmd in git_commands:
        try:
            subprocess.run(cmd.split(), check=True)
        except subprocess.CalledProcessError:
            print(f"⚠️  Git command failed: {cmd}")

if __name__ == "__main__":
    create_directory_structure()
    setup_git_hooks()
    print("🚀 FilmFusion project setup completed!")
