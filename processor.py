import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss
from model import NFLModel

def generate_training_data(games_df: pd.DataFrame, players_df: pd.DataFrame, params: dict):
    """
    Simulates history chronologically to generate features for the AI.
    """
    print("Initializing Simulation & Feature Engineering...")
    
    model = NFLModel(**params)
    features = []
    targets = []
    
    # 1. Index player stats
    stats_lookup = {}
    for _, row in players_df.iterrows():
        key = (row['season'], row['week'])
        if key not in stats_lookup: stats_lookup[key] = []
        stats_lookup[key].append(row)

    # 2. Track Rolling Stats
    team_history = {} 

    # 3. Sort Chronologically
    games_df = games_df.sort_values(['season', 'week'])
    
    print("Running Historical Simulation...")
    
    for _, game in games_df.iterrows():
        if pd.isna(game['result']): continue 

        season = game['season']
        week = game['week']
        home = game['home_team']
        away = game['away_team']
        
        # --- A. FEATURE EXTRACTION ---
        t_home = model.get_team(home)
        t_away = model.get_team(away)
        
        h_hist = team_history.get(home, [])
        a_hist = team_history.get(away, [])
        
        # Rolling Average Score (Last 5 games)
        h_recent = np.mean([x['score'] for x in h_hist[-5:]]) if h_hist else 22.0
        a_recent = np.mean([x['score'] for x in a_hist[-5:]]) if a_hist else 22.0

        feature_row = {
            'elo_diff': (t_home.elo + model.hfa) - t_away.elo,
            'home_elo': t_home.elo,
            'away_elo': t_away.elo,
            'home_def_elo': t_home.def_elo,
            'away_def_elo': t_away.def_elo,
            'home_recent_score': h_recent,
            'away_recent_score': a_recent,
            'week': week,
            'is_postseason': 1 if game['game_type'] == 'POST' else 0,
            'hfa_static': model.hfa
        }
        features.append(feature_row)
        targets.append(1 if game['result'] > 0 else 0)
        
        # --- B. UPDATE HISTORY ---
        if home not in team_history: team_history[home] = []
        if away not in team_history: team_history[away] = []
        
        team_history[home].append({'score': game['home_score']})
        team_history[away].append({'score': game['away_score']})

        # --- C. UPDATE MODEL ---
        current_stats = pd.DataFrame(stats_lookup.get((season, week), []))
        model.update_game(game, current_stats)

    return pd.DataFrame(features), pd.Series(targets), model

def train_xgb_model(X: pd.DataFrame, y: pd.Series):
    """
    Trains the XGBoost classifier.
    FIXED: early_stopping_rounds moved to constructor for XGBoost 2.0+
    """
    print(f"\n--- Training XGBoost AI on {len(X)} games ---")
    
    split_idx = int(len(X) * 0.85)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Initialize Model
    # Note: early_stopping_rounds is passed HERE now
    clf = XGBClassifier(
        n_estimators=800,
        learning_rate=0.01,
        max_depth=3,
        subsample=0.7,
        colsample_bytree=0.7,
        objective='binary:logistic',
        random_state=42,
        early_stopping_rounds=50 
    )
    
    # Train
    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Evaluate
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test Set Accuracy: {acc:.1%}")
    
    # Retrain on full data (without early stopping since we have no validation set)
    print("Retraining on FULL dataset for live use...")
    final_clf = XGBClassifier(
        n_estimators=clf.best_iteration if hasattr(clf, 'best_iteration') else 500,
        learning_rate=0.01,
        max_depth=3,
        objective='binary:logistic',
        random_state=42
    )
    final_clf.fit(X, y)
    
    return final_clf

def predict_upcoming(model: NFLModel, ai_model: XGBClassifier, 
                     home: str, away: str, season: int, week: int, 
                     injury_df: pd.DataFrame = None):
    # (Same as before - kept for completeness)
    t_home = model.get_team(home)
    t_away = model.get_team(away)
    
    h_pen = model.get_injury_impact(home, injury_df, season, week)
    a_pen = model.get_injury_impact(away, injury_df, season, week)
    
    features = pd.DataFrame([{
        'elo_diff': (t_home.elo - h_pen + model.hfa) - (t_away.elo - a_pen),
        'home_elo': t_home.elo - h_pen,
        'away_elo': t_away.elo - a_pen,
        'home_def_elo': t_home.def_elo,
        'away_def_elo': t_away.def_elo,
        'home_recent_score': 23.0, 
        'away_recent_score': 23.0,
        'week': week,
        'is_postseason': 0,
        'hfa_static': model.hfa
    }])
    
    prob_home = ai_model.predict_proba(features)[0][1]
    
    print(f"\n--- Prediction: {away} @ {home} ---")
    print(f"Win Probability ({home}): {prob_home:.1%}")
    print(f"Pick: {home if prob_home > 0.5 else away}")
    
    return prob_home