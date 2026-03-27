from data_loader import get_data, get_injury_data
from processor import generate_training_data, train_xgb_model, predict_upcoming
import pandas as pd
import pickle
import os

# --- SAVE/LOAD SYSTEM ---
def save_system(elo_model, ai_model, filename="nfl_system.pkl"):
    """Saves the trained logic and AI to a file."""
    print(f"Saving system to {filename}...")
    with open(filename, 'wb') as f:
        pickle.dump({'elo': elo_model, 'ai': ai_model}, f)
    print("Save Complete!")

def load_system(filename="nfl_system.pkl"):
    """Loads the trained system from disk if it exists."""
    if not os.path.exists(filename):
        return None
    print(f"Loading system from {filename}...")
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['elo'], data['ai']

# --- VERIFICATION LOGIC ---
def run_verification(games, players, injuries):
    """
    Runs a short simulation (2015 only) to verify logic is working.
    """
    print("\n--- STEP 1.5: VERIFYING INJURY LOGIC ---")
    
    # Filter data for just the first few weeks of 2015 to be fast
    games_subset = games[(games['season'] == 2015) & (games['week'] <= 3)]
    
    # Run the simulation on this small subset
    print("Running mini-simulation (2015 Weeks 1-3)...")
    params = {'hfa': 25.0, 'team_k': 15.0, 'player_k': 10.0}
    _, _, test_model = generate_training_data(games_subset, players, params)
    
    # TEST CASE: Drew Brees was OUT in Week 3, 2015 (vs CAR)
    print("\n> Checking Week 3 (2015) Injury Impact for Saints (NO)...")
    week_3_injuries = injuries[(injuries['season'] == 2015) & (injuries['week'] == 3)]
    
    # Calculate impact with DEBUG=True to see the logs
    impact = test_model.get_injury_impact('NO', week_3_injuries, debug=True)
    
    print(f"  > Total calculated penalty for NO: {impact:.2f} Elo points")
    print("--- VERIFICATION COMPLETE ---\n")

def main():
    # 1. Load Data
    print("--- STEP 1: LOADING DATA ---")
    games, players = get_data(start_year=2015)
    injuries = get_injury_data(start_year=2015)
    
    # 2. Check for Saved Model
    saved_data = load_system()
    
    if saved_data:
        print("Resuming from saved model...")
        trained_elo_model, ai_model = saved_data
        
        # Optional: You can uncomment this line if you want to ALWAYS verify logic, even on reload
        # run_verification(games, players, injuries)
        
    else:
        print("No save found. Starting fresh training...")
        
        # Run verification before the long training process so we know it works
        run_verification(games, players, injuries)
        
        # 3. Train System (Full Run)
        print("--- STEP 2: SIMULATING FULL HISTORY & TRAINING ---")
        params = {'hfa': 45.0, 'team_k': 16.0, 'player_k': 12.0}
        
        # Run history to get features (X) and targets (y), and the final state of the Model
        X, y, trained_elo_model = generate_training_data(games, players, params)
        
        # Train the AI on those features
        ai_model = train_xgb_model(X, y)
        
        # Save the trained system
        save_system(trained_elo_model, ai_model)
    
    # 4. Predict Next Week (Example)
    print("\n--- STEP 3: LIVE PREDICTIONS ---")
    
    # Example Prediction: Super Bowl Matchup (Using 2025 Week 22 as placeholder)
    predict_upcoming(
        model=trained_elo_model,
        ai_model=ai_model,
        home='KC',
        away='BUF',
        season=2025,
        week=22,
        injury_df=injuries 
    )

if __name__ == "__main__":
    main()