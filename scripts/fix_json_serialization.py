#!/usr/bin/env python3
"""
Quick fix for JSON serialization issues
Run this script to patch the ETL pipeline
"""

import sys
from pathlib import Path
import re

def patch_etl_pipeline():
    """Patch the ETL pipeline file"""
    etl_file = Path("src/data_processing/etl_pipeline.py")
    
    if not etl_file.exists():
        print("❌ ETL pipeline file not found")
        return False
    
    # Read the current file
    with open(etl_file, 'r') as f:
        content = f.read()
    
    # Add import for numpy at the top
    if "import numpy as np" not in content:
        content = content.replace("import pandas as pd", "import pandas as pd\nimport numpy as np")
    
    # Add the JSON serialization method
    json_method = '''
    def _make_json_serializable(self, obj):
        """Convert numpy/pandas data types to JSON serializable types"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj
'''
    
    if "_make_json_serializable" not in content:
        # Add before the last method
        content = content.replace("    def run_full_pipeline(self)", json_method + "\n    def run_full_pipeline(self)")
    
    # Update the problematic JSON dumps
    content = content.replace(
        "json.dump(self.genre_normalizer.get_genre_statistics(), f, indent=2)",
        "json.dump(self._make_json_serializable(self.genre_normalizer.get_genre_statistics()), f, indent=2)"
    )
    
    content = content.replace(
        "json.dump(quality_report, f, indent=2)",
        "json.dump(self._make_json_serializable(quality_report), f, indent=2)"
    )
    
    # Write back the patched content
    with open(etl_file, 'w') as f:
        f.write(content)
    
    print("✅ ETL pipeline patched successfully")
    return True

if __name__ == "__main__":
    success = patch_etl_pipeline()
    exit(0 if success else 1)
