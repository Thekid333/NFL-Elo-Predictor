import pandas as pd
import requests
import time
import json
import os

# Headers to mimic a browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.espn.com/"
}

def get_val_robust(keys, values, target_key):
    """
    Helper: Finds a value in the list by key, handling case-sensitivity.
    """
    target = target_key.upper()
    keys_upper = [k.upper() for k in keys]
    
    if target in keys_upper:
        idx = keys_upper.index(target)
        try:
            val_str = str(values[idx])
            # Remove commas (e.g. "1,200" -> "1200") and handle "--"
            val_clean = val_str.replace(',', '').replace('--', '0')
            return float(val_clean)
        except (ValueError, IndexError):
            return 0.0
    return 0.0

def scrape_live_schedule(season=2025):
    """
    Fetches the schedule for the specific season from ESPN.
    """
    start_date = f"{season}0901"
    end_date = f"{season+1}0301"
    
    url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?limit=1000&dates={start_date}-{end_date}"
    print(f"   > Scraping ESPN Schedule for {season}...")
    
    try:
        response = requests.get(url, headers=HEADERS)
        data = response.json()
        games = []
        
        for event in data.get('events', []):
            try:
                if event['season']['year'] != season: continue
                
                game_id = event['id']
                week = event['week']['number']
                game_type = 'POST' if event['season']['type'] == 3 else 'REG'
                
                competitors = event['competitions'][0]['competitors']
                home = next(c for c in competitors if c['homeAway'] == 'home')
                away = next(c for c in competitors if c['homeAway'] == 'away')
                
                h_score = float(home.get('score', 0)) if home.get('score') else None
                a_score = float(away.get('score', 0)) if away.get('score') else None
                
                result = None
                if h_score is not None and a_score is not None:
                    result = h_score - a_score
                
                games.append({
                    'season': season, 
                    'week': week,
                    'game_id': game_id,
                    'home_team': home['team']['abbreviation'],
                    'away_team': away['team']['abbreviation'],
                    'home_score': h_score, 
                    'away_score': a_score,
                    'result': result, 
                    'game_type': game_type
                })
            except: continue
                
        return pd.DataFrame(games)
    except Exception as e:
        print(f"   > Schedule Scrape Error: {e}")
        return pd.DataFrame()

def scrape_live_player_stats(season=2025):
    """
    Fetches Player Stats from ESPN Game Summaries (Boxscores).
    FIXED: Now uses the CORRECT keys from your debug dump (passingYards, etc).
    """
    print(f"   > Scraping ESPN Player Stats ({season})...")
    
    schedule = scrape_live_schedule(season)
    if schedule.empty: 
        print("   > No schedule found for 2025.")
        return pd.DataFrame()
    
    played_games = schedule[schedule['result'].notna()]
    print(f"   > Found {len(played_games)} games with stats. Fetching details...")
    
    all_stats = []
    
    count = 0
    for _, game in played_games.iterrows():
        game_id = game['game_id']
        week = game['week']
        
        url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={game_id}"
        
        try:
            time.sleep(0.05) # Rate limit protection
            resp = requests.get(url, headers=HEADERS).json()
            
            if 'boxscore' not in resp or 'players' not in resp['boxscore']:
                continue
                
            for team_box in resp['boxscore']['players']:
                for stat_group in team_box.get('statistics', []):
                    stat_type = stat_group['name'].lower()
                    keys = stat_group['keys']
                    
                    for athlete in stat_group.get('athletes', []):
                        try:
                            ath_obj = athlete.get('athlete', {})
                            name = ath_obj.get('displayName', 'Unknown')
                            pid = f"{season}_{name.replace(' ', '')}"
                            
                            # Improved Position Detection
                            pos = "UNK"
                            if 'position' in ath_obj:
                                if isinstance(ath_obj['position'], dict):
                                    pos = ath_obj['position'].get('abbreviation', 'UNK')
                                else:
                                    pos = str(ath_obj['position'])
                            
                            p_row = {
                                'season': season,
                                'week': week,
                                'player_id': pid,
                                'player_display_name': name,
                                'position': pos,
                                'passing_epa': 0.0,
                                'rushing_epa': 0.0,
                                'receiving_epa': 0.0,
                                'total_epa': 0.0
                            }
                            
                            stats_vals = athlete['stats']
                            
                            # --- FIXED EXTRACTION LOGIC ---
                            # Uses the EXACT keys from your debug file
                            
                            if 'passing' in stat_type:
                                yds = get_val_robust(keys, stats_vals, 'passingYards')
                                tds = get_val_robust(keys, stats_vals, 'passingTouchdowns')
                                ints = get_val_robust(keys, stats_vals, 'interceptions')
                                # Simple EPA approx
                                p_row['passing_epa'] = (yds / 15.0) + (tds * 4.0) - (ints * 2.0)
                                
                            elif 'rushing' in stat_type:
                                yds = get_val_robust(keys, stats_vals, 'rushingYards')
                                tds = get_val_robust(keys, stats_vals, 'rushingTouchdowns')
                                p_row['rushing_epa'] = (yds / 10.0) + (tds * 6.0)
                                
                            elif 'receiving' in stat_type:
                                yds = get_val_robust(keys, stats_vals, 'receivingYards')
                                tds = get_val_robust(keys, stats_vals, 'receivingTouchdowns')
                                p_row['receiving_epa'] = (yds / 10.0) + (tds * 6.0)
                            
                            all_stats.append(p_row)
                            
                        except Exception: continue
        except Exception: continue
            
        count += 1
        if count % 20 == 0: print(f"     > Processed {count}/{len(played_games)} games...")

    # 3. Aggregate (Merge stats if a player had both Rushing and Receiving)
    if not all_stats: return pd.DataFrame()
    
    df = pd.DataFrame(all_stats)
    df_final = df.groupby(['season', 'week', 'player_id', 'player_display_name', 'position'], as_index=False).sum()
    df_final['total_epa'] = df_final['passing_epa'] + df_final['rushing_epa'] + df_final['receiving_epa']
    
    return df_final

def scrape_live_injuries(season=2025):
    # CBS Injuries
    url = "https://www.cbssports.com/nfl/injuries/"
    try:
        tables = pd.read_html(url)
        injuries = []
        for table in tables:
            if 'Player' not in table.columns: continue
            for _, row in table.iterrows():
                name_full = str(row['Player'])
                parts = name_full.split()
                name = " ".join(parts[:-1]) if len(parts) > 1 else name_full
                pos = parts[-1] if len(parts) > 1 else "UNK"
                
                injuries.append({
                    'season': season, 'week': 20, 'team': 'UNK',
                    'full_name': name, 'position': pos,
                    'report_status': str(row['Injury Status'])
                })
        return pd.DataFrame(injuries)
    except: return pd.DataFrame()