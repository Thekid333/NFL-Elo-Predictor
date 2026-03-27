import pandas as pd
import nfl_data_py as nfl
import os
import time
from scraper import scrape_live_schedule, scrape_live_player_stats, scrape_live_injuries

CACHE_DIR = "data"
CURRENT_SEASON = 2025  # Force this to be the current year

def get_data(start_year=2015):
    if not os.path.exists(CACHE_DIR): os.makedirs(CACHE_DIR)
    
    games_path = f"{CACHE_DIR}/games_combined.csv"
    players_path = f"{CACHE_DIR}/players_combined.csv"

    # --- 1. GAMES ---
    print("Checking Game Data...")
    # (Optional: Delete cache if you want to force refresh)
    # if os.path.exists(games_path): os.remove(games_path)

    if os.path.exists(games_path):
        print("   > Loading from cache (delete 'data/' folder to refresh).")
        games = pd.read_csv(games_path)
    else:
        print(f"   > Downloading History ({start_year}-{CURRENT_SEASON-1})...")
        cols = ['season', 'week', 'home_team', 'away_team', 'result', 'game_type', 'div_game']
        
        # 1. Get History (Up to LAST season)
        try:
            # Range stops BEFORE the second number, so (2015, 2025) gets 2015-2024
            years = list(range(start_year, CURRENT_SEASON)) 
            history = nfl.import_schedules(years=years)[cols]
            print(f"   > Library found {len(history)} historical games.")
        except Exception as e:
            print(f"   > Library Error: {e}")
            history = pd.DataFrame(columns=cols)

        # 2. Scrape CURRENT Season (2025)
        print(f"   > Scraping Live Schedule ({CURRENT_SEASON})...")
        try:
            # Pass the current season explicitly
            future = scrape_live_schedule(season=CURRENT_SEASON)
            print(f"   > Scraper found {len(future)} 2025 games.")
        except Exception as e:
            print(f"   > Scraper Error: {e}")
            future = pd.DataFrame(columns=cols)

        # 3. Merge
        games = pd.concat([history, future], ignore_index=True)
        # Drop duplicates in case of overlap
        games = games.drop_duplicates(subset=['season', 'week', 'home_team', 'away_team'], keep='last')
        
        # Filter out empty rows
        games = games[games['home_team'].notna()]
        
        print(f"   > Saving {len(games)} games to {games_path}")
        games.to_csv(games_path, index=False)

    # --- 2. PLAYERS ---
    print("Checking Player Data...")
    if os.path.exists(players_path):
        print("   > Loading from cache.")
        players = pd.read_csv(players_path)
    else:
        print(f"   > Downloading Player Stats ({start_year}-{CURRENT_SEASON-1})...")
        cols = ['season', 'week', 'player_id', 'player_display_name', 'position', 
                'passing_epa', 'rushing_epa', 'receiving_epa']
        
        # 1. Get History
        try:
            years = list(range(start_year, CURRENT_SEASON))
            history = nfl.import_weekly_data(years=years, columns=cols)
            history = history.fillna(0)
            print(f"   > Library found {len(history)} historical records.")
        except Exception as e:
            print(f"   > Library Error: {e}")
            history = pd.DataFrame(columns=cols)

        # 2. Scrape LIVE Stats (2025)
        print(f"   > Scraping Live Stats ({CURRENT_SEASON})...")
        try:
            live_stats = scrape_live_player_stats(season=CURRENT_SEASON)
            # Align columns
            for col in cols:
                if col not in live_stats.columns: live_stats[col] = 0
            print(f"   > Scraper found {len(live_stats)} 2025 records.")
        except Exception:
            live_stats = pd.DataFrame(columns=cols)

        # 3. Merge
        players = pd.concat([history, live_stats], ignore_index=True)
        players = players.drop_duplicates(subset=['season', 'week', 'player_id'], keep='last')
        
        print(f"   > Saving {len(players)} records to {players_path}")
        players.to_csv(players_path, index=False)

    # --- 3. FIX EPA ---
    print("Verifying EPA...")
    cols_to_sum = ['passing_epa', 'rushing_epa', 'receiving_epa']
    for col in cols_to_sum:
        if col not in players.columns: players[col] = 0.0
    players[cols_to_sum] = players[cols_to_sum].fillna(0.0)
    
    # Recalculate Total EPA if missing
    if 'total_epa' not in players.columns:
        players['total_epa'] = players[cols_to_sum].sum(axis=1)
    
    # Fill remaining NaNs in total_epa
    players['total_epa'] = players['total_epa'].fillna(players[cols_to_sum].sum(axis=1))
        
    return games, players

def get_injury_data(start_year=2015):
    if not os.path.exists(CACHE_DIR): os.makedirs(CACHE_DIR)
    path = f"{CACHE_DIR}/injuries_combined.csv"
    
    print("Fetching Injuries...")
    # Fetch History
    try:
        years = list(range(start_year, CURRENT_SEASON))
        history = nfl.import_injuries(years=years)
    except:
        history = pd.DataFrame()

    # Fetch Live
    live = scrape_live_injuries(season=CURRENT_SEASON)
    
    combined = pd.concat([history, live], ignore_index=True)
    combined.to_csv(path, index=False)
    return combined