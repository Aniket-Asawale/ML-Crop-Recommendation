"""
Clear Streamlit cache and run the app
This ensures the app loads the newly retrained models
"""
import shutil
import os
import subprocess
from pathlib import Path

print("=" * 80)
print("CLEARING STREAMLIT CACHE AND RESTARTING APP")
print("=" * 80)

# Find and remove Streamlit cache directory
cache_dir = Path.home() / '.streamlit' / 'cache'
if cache_dir.exists():
    print(f"\n🗑️  Removing cache directory: {cache_dir}")
    try:
        shutil.rmtree(cache_dir)
        print("✅ Cache cleared successfully!")
    except Exception as e:
        print(f"⚠️  Could not remove cache: {e}")
else:
    print(f"\n✅ No cache directory found at {cache_dir}")

# Also check for local .streamlit directory
local_cache = Path('.streamlit')
if local_cache.exists():
    print(f"\n🗑️  Removing local .streamlit directory: {local_cache}")
    try:
        shutil.rmtree(local_cache)
        print("✅ Local cache cleared successfully!")
    except Exception as e:
        print(f"⚠️  Could not remove local cache: {e}")

print("\n" + "=" * 80)
print("INSTRUCTIONS TO RUN THE APP")
print("=" * 80)
print("\n1. Open a new terminal (to ensure clean environment)")
print("2. Run: streamlit run app.py")
print("3. When the app opens, press 'C' in the terminal to clear cache")
print("4. Or click 'Clear cache' in the app menu (top right ⋮ → Clear cache)")
print("\n💡 TIP: If models still seem stuck, try:")
print("   - Close the browser tab")
print("   - Stop the Streamlit server (Ctrl+C)")
print("   - Run this script again")
print("   - Restart Streamlit")
print("\n" + "=" * 80)

