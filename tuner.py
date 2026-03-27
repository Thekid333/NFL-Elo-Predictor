from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier
from data_loader import get_data
from processor import generate_training_data

def tune_ai():
    print("Generating Data for Tuning...")
    games, players = get_data()
    params = {'hfa': 45.0, 'team_k': 16.0, 'player_k': 12.0}
    X, y, _ = generate_training_data(games, players, params)
    
    # Define what to test
    # These are the "knobs" we are twisting
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5],
        'n_estimators': [100, 300, 500],
        'subsample': [0.8, 1.0]
    }
    
    print("Starting Grid Search (This takes time)...")
    xgb = XGBClassifier(objective='binary:logistic')
    
    # Use TimeSeriesSplit so we don't train on future data
    tscv = TimeSeriesSplit(n_splits=5)
    
    grid = GridSearchCV(xgb, param_grid, cv=tscv, scoring='accuracy', verbose=1)
    grid.fit(X, y)
    
    print("\n--- BEST PARAMETERS FOUND ---")
    print(grid.best_params_)
    print(f"Best Accuracy: {grid.best_score_:.1%}")

if __name__ == "__main__":
    tune_ai()