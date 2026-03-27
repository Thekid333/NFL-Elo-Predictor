import pandas as pd
from src.data_loader import get_data, get_injury_data
from src.model import NFLModel

# --- CONFIGURATION ---
TARGET_SEASON = 2025    # The season we are currently simulating
START_WEEK = 10         # We have data up to Week 9; we want to predict Week 10 onward
HFA = 24.55
TEAM_K = 14.56
PLAYER_K = 10.0

# --- INJURY SCENARIOS ---
# Add your "What If" injuries here.
# The model will verify the player's ELO and apply the math dynamically.
SIMULATED_INJURIES = [
    # (Week, Team, Player Name, Position)
    (11, 'BUF', 'Josh Allen', 'QB'),   # Example: Josh Allen misses Week 11
    (14, 'KC', 'Patrick Mahomes', 'QB') # Example: Mahomes misses Week 14
]

def main():
    print(f"--- Training on History (2015-{TARGET_SEASON} Week {START_WEEK-1}) ---")

    # 1. Load All Data (History + Live Scrape)
    games, players = get_data(start_year=2015)
    real_injuries = get_injury_data(start_year=2015)

    # 2. SEPARATE PAST & FUTURE
    # Training Set: All past seasons + Current season up to the start week
    past_games = games[
        (games['season'] < TARGET_SEASON) | 
        ((games['season'] == TARGET_SEASON) & (games['week'] < START_WEEK))
    ]
    
    # Future Set: The games we want to predict
    future_games = games[
        (games['season'] == TARGET_SEASON) & 
        (games['week'] >= START_WEEK)
    ]

    print(f"  > Loaded {len(past_games)} historical games.")
    print(f"  > Found {len(future_games)} future games to simulate.")

    # 3. TRAIN THE MODEL (Batch Process)
    # We run this once to get the Elos exactly right as of Week 10
    model = NFLModel(hfa=HFA, team_k=TEAM_K, player_k=PLAYER_K)
    model.run_season(past_games, players)
    
    print("\n--- Training Complete. Ratings are set. Starting Simulation... ---")

    # 4. PREDICT THE FUTURE (Week by Week)
    correct = 0
    total = 0
    
    # Loop through the remaining weeks in order
    sorted_weeks = sorted(future_games['week'].unique())
    
    for week in sorted_weeks:
        print(f"\n[Week {week}]")
        
        # A. Get Games for this week
        weekly_games = future_games[future_games['week'] == week]
        if weekly_games.empty: continue

        # B. Construct the Injury Report
        # 1. Start with any REAL injuries we scraped for this week
        current_injuries = real_injuries[
            (real_injuries['season'] == TARGET_SEASON) & 
            (real_injuries['week'] == week)
        ].copy()
        
        # 2. Add your SIMULATED injuries
        active_sims = [x for x in SIMULATED_INJURIES if x[0] == week]
        if active_sims:
            print(f"   [!] Injecting {len(active_sims)} simulated injuries...")
            new_rows = []
            for _, team, name, pos in active_sims:
                new_rows.append({
                    'season': TARGET_SEASON,
                    'week': week,
                    'team': team,
                    'full_name': name,
                    'position': pos,
                    'report_status': 'Out'
                })
            current_injuries = pd.concat([current_injuries, pd.DataFrame(new_rows)])

        # C. Predict Each Game
        for _, game in weekly_games.iterrows():
            home = game['home_team']
            away = game['away_team']
            
            # Calculate Injury Penalties for these two teams
            home_pen = model.get_injury_impact(home, current_injuries)
            away_pen = model.get_injury_impact(away, current_injuries)
            
            # Get Elos (Adjusted)
            home_elo = model.teams[home].elo - home_pen
            away_elo = model.teams[away].elo - away_pen
            
            # Win Probability
            elo_diff = home_elo + model.hfa - away_elo
            prob = 1 / (1 + 10 ** (-elo_diff / 400))
            
            # Pick Winner
            pred_winner = home if prob > 0.5 else away
            
            # Check Result (if the game has already happened in real life)
            result_text = "Pending"
            actual_result = game.get('result', None)
            
            if pd.notna(actual_result):
                actual_winner = home if actual_result > 0 else (away if actual_result < 0 else "Tie")
                is_correct = (pred_winner == actual_winner)
                result_text = f"Actual: {actual_winner} {'✅' if is_correct else '❌'}"
                if actual_winner != "Tie":
                    total += 1
                    if is_correct: correct += 1
            
            # Print Prediction
            # Only print details if it's a close game OR has injuries
            if home_pen > 0 or away_pen > 0 or 0.45 < prob < 0.55:
                print(f"   {away} @ {home}")
                if home_pen > 0: print(f"      {home} Injury Pen: -{home_pen:.0f}")
                if away_pen > 0: print(f"      {away} Injury Pen: -{away_pen:.0f}")
                print(f"      Pick: {pred_winner} ({prob:.1%}) | {result_text}")
        
        # D. IMPORTANT: Update the model with the REAL results before next week?
        # If you want the model to 'learn' from the games as they happen, uncomment this:
        # model.run_season(weekly_games, players)

    if total > 0:
        print(f"\nSimulation Accuracy: {correct}/{total} ({correct/total:.1%})")

if __name__ == "__main__":
    main()