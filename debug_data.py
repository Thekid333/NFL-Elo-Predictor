# debug_data.py
import os
import sys

print("--- DIAGNOSTIC START ---")

# 1. Test Folder Permissions
print("[1] Checking 'data' folder...")
try:
    if not os.path.exists("data"):
        os.makedirs("data")
        print("    -> Created 'data' folder.")
    else:
        print("    -> 'data' folder exists.")
    
    # Try writing a test file
    with open("data/test_write.txt", "w") as f:
        f.write("Permissions OK")
    print("    -> Write permissions OK.")
    os.remove("data/test_write.txt")
except Exception as e:
    print(f"    -> ERROR: File system issue: {e}")

# 2. Test nfl_data_py and pyarrow
print("[2] Testing nfl_data_py library...")
try:
    import nfl_data_py as nfl
    print("    -> Library imported.")
    
    print("    -> Attempting to fetch 1 season (2023)...")
    # This often fails if pyarrow is missing
    df = nfl.import_schedules(years=[2023])
    
    if df.empty:
        print("    -> WARNING: Downloaded DataFrame is empty.")
    else:
        print(f"    -> Success: Downloaded {len(df)} games.")
        
except ImportError as e:
    print(f"    -> CRITICAL ERROR: {e}")
    if "pyarrow" in str(e) or "fastparquet" in str(e):
        print("       SOLUTION: Run 'pip install pyarrow'")
except Exception as e:
    print(f"    -> ERROR during download: {e}")

# 3. Test Scraper Import
print("[3] Testing scraper.py...")
try:
    from scraper import scrape_2025_schedule
    print("    -> Scraper module imported successfully.")
    
    print("    -> Running test scrape (ESPN)...")
    games = scrape_2025_schedule()
    print(f"    -> Scrape result: {len(games)} games found.")
    
except Exception as e:
    print(f"    -> ERROR in scraper.py: {e}")

print("--- DIAGNOSTIC COMPLETE ---")