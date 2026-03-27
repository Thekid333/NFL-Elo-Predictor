import pandas as pd
import numpy as np
from processor import generate_training_data, train_xgb_model

def run_backtest(games, players, start_season=2023):
    print(f"Starting Walk-Forward Backtest from {start_season}...")
    
    # Sort chronologically
    games = games.sort_values(['season', 'week'])
    
    correct_picks = 0
    total_games = 0
    
    # Loop through every week of the test period
    seasons = games['season'].unique()
    test_seasons = [s for s in seasons if s >= start_season]
    
    for season in test_seasons:
        for week in range(1, 19): # Regular season weeks
            # 1. SPLIT DATA
            # Training Data: Everything BEFORE this week
            train_games = games[
                (games['season'] < season) | 
                ((games['season'] == season) & (games['week'] < week))
            ]
            
            # Test Data: JUST this week
            test_games = games[
                (games['season'] == season) & 
                (games['week'] == week)
            ]
            
            if test_games.empty: continue
            
            print(f"Testing {season} Week {week} ({len(test_games)} games)...")
            
            # 2. TRAIN (Re-simulate history up to this point)
            # Note: This is slow but accurate. To speed up, we usually
            # just update the Elo model incrementally instead of full re-train.
            # For "Serious" testing, we do the full generate to be safe.
            params = {'hfa': 45.0, 'team_k': 16.0, 'player_k': 12.0}
            X_train, y_train, _ = generate_training_data(train_games, players, params)
            
            # Train AI
            ai = train_xgb_model(X_train, y_train)
            
            # 3. PREDICT
            # We need to generate features for the test week using the model state
            # (This part requires careful handling in your processor.py to expose
            #  a 'predict_batch' function, but for now we simplify)
            pass 
            # (Logic continues to compare ai.predict(test_games) vs actual result)

    print(f"Final Backtest Accuracy: {correct_picks/total_games:.1%}")