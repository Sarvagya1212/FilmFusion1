#!/usr/bin/env python3
"""
Quick fix script for Windows encoding issues
"""

import re
import os
from pathlib import Path

def replace_emojis_in_file(file_path):
    """Replace emojis with text equivalents in a Python file"""
    
    emoji_replacements = {
        '[SUCCESS]': '[SUCCESS]',
        '[ERROR]': '[ERROR]',
        '[SAVED]': '[SAVED]',
        '[INFO]': '[INFO]',
        '[TIMING]': '[TIMING]',
        '[START]': '[START]',
        '[COMPLETE]': '[COMPLETE]',
        '[CHECK]': '[CHECK]',  
        '[STATS]': '[STATS]',
        '[COLD]': '[COLD]',
        '[WARNING]': '[WARNING]',
        '[TEST]': '[TEST]',
        '[DATABASE]': '[DATABASE]',
        '[PIPELINE]': '[PIPELINE]',
        '[NEXT]': '[NEXT]'
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        for emoji, replacement in emoji_replacements.items():
            content = content.replace(emoji, replacement)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed emojis in: {file_path}")
            return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False
    
    return False

def main():
    """Fix encoding issues in all Python files"""
    
    project_root = Path('.')
    python_files = []
    
    # Find all Python files
    for pattern in ['src/**/*.py', 'scripts/*.py', 'tests/*.py']:
        python_files.extend(project_root.glob(pattern))
    
    print("Fixing encoding issues in Python files...")
    
    fixed_files = 0
    for file_path in python_files:
        if replace_emojis_in_file(file_path):
            fixed_files += 1
    
    print(f"Fixed {fixed_files} files with emoji replacements")
    
    # Also fix the logging config
    logging_config_path = Path('config/logging_config.py')
    if logging_config_path.exists():
        print("Updating logging configuration for Windows compatibility...")
        # The logging config update would go here
        print("Please manually update config/logging_config.py with the provided code")
    
    print("Encoding fixes completed!")

if __name__ == "__main__":
    main()
