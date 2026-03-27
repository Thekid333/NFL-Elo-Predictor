import math
import pandas as pd
from typing import Dict

class Team:
    def __init__(self, name: str):
        self.name = name
        self.elo = 1500.0      # Overall Strength
        self.def_elo = 1500.0  # Defensive Strength specifically
        self.wins = 0
        self.losses = 0

    def record_game(self, won: bool):
        if won: self.wins += 1
        else: self.losses += 1

class Player:
    def __init__(self, player_id: str, name: str, position: str):
        self.id = player_id
        self.name = name
        self.position = position
        self.elo = 1500.0
        self.games_played = 0

    def update_elo(self, epa: float, k_factor: float):
        # Cap EPA to prevent single-game outliers (e.g. 7 TD game)
        capped_epa = max(min(epa, 15.0), -15.0)
        
        # 1.0 EPA approx = 2.5 Elo points
        perf_shift = (capped_epa * 2.5) * (k_factor / 10.0)
        
        # Soft Decay toward 1500 (Regress to mean)
        self.elo = (self.elo * 0.99) + (1500 * 0.01) + perf_shift
        self.games_played += 1

class NFLModel:
    def __init__(self, hfa=45.0, team_k=16.0, player_k=12.0):
        self.hfa = hfa
        self.team_k = team_k
        self.player_k = player_k
        self.teams: Dict[str, Team] = {}
        self.players: Dict[str, Player] = {}
        
        # Elo Floor for replacement players
        self.REPLACEMENT_LEVEL = {
            'QB': 1350, 'WR': 1450, 'RB': 1450, 'TE': 1450, 'UNK': 1450
        }

    def get_team(self, name: str) -> Team:
        if name not in self.teams: self.teams[name] = Team(name)
        return self.teams[name]

    def get_player(self, pid: str, name: str, position: str) -> Player:
        if pid not in self.players: self.players[pid] = Player(pid, name, position)
        return self.players[pid]

    def update_game(self, game_row, weekly_player_stats):
        """Processes game result and updates ratings."""
        home = self.get_team(game_row['home_team'])
        away = self.get_team(game_row['away_team'])
        
        # Result: >0 Home Win, <0 Away Win
        result = game_row['result']
        if pd.isna(result): return # Skip future games

        # 1. Team Elo Update (Overall)
        elo_diff = home.elo + self.hfa - away.elo
        prob = 1 / (1 + 10 ** (-elo_diff / 400))
        actual = 1.0 if result > 0 else (0.0 if result < 0 else 0.5)
        
        # Margin of Victory Multiplier (MoV)
        # Prevents small wins from shifting Elo too much, rewards blowouts logarithmically
        mov_mult = math.log(max(abs(result), 1) + 1) * 2.2 / ((abs(elo_diff)*0.001) + 2.2)
        shift = (actual - prob) * self.team_k * mov_mult

        home.elo += shift
        away.elo -= shift
        
        # 2. Defensive Elo Update (Inverse of score)
        # If Home scores 30, Away Defense played poorly.
        # This is a simplified heuristic: adjust defense based on points allowed vs league avg (approx 22)
        h_score = game_row.get('home_score', 22)
        a_score = game_row.get('away_score', 22)
        
        # Update Home Defense (based on Away Score)
        home.def_elo -= (a_score - 22.0) * (self.team_k / 20.0)
        # Update Away Defense (based on Home Score)
        away.def_elo -= (h_score - 22.0) * (self.team_k / 20.0)

        home.record_game(result > 0)
        away.record_game(result < 0)

        # 3. Player Elo Update
        for _, p_row in weekly_player_stats.iterrows():
            pid = p_row.get('player_id')
            if pd.isna(pid): continue
            
            player = self.get_player(
                str(pid), 
                p_row.get('player_display_name', 'Unknown'), 
                p_row.get('position', 'UNK')
            )
            player.update_elo(p_row.get('total_epa', 0.0), self.player_k)

    def get_injury_impact(self, team_name: str, injury_df: pd.DataFrame, season: int = None, week: int = None, debug=False) -> float:
        """Calculates penalty for 'Out' players."""
        if injury_df is None or injury_df.empty: return 0.0

        impact = 0.0
        mask = (injury_df['team'] == team_name) & \
               (injury_df['report_status'].str.contains('Out|IR|Res|Doubt', case=False, na=False))
        
        if season is not None: mask &= (injury_df['season'] == season)
        if week is not None: mask &= (injury_df['week'] == week)
            
        team_injuries = injury_df[mask]

        if debug and not team_injuries.empty:
            print(f"Checking injuries for {team_name} (S{season} W{week})...")

        for _, row in team_injuries.iterrows():
            name = row['full_name']
            
            # Simple fuzzy match attempt
            found_player = None
            for p in self.players.values():
                if p.name == name or (name in p.name):
                    found_player = p
                    break
            
            if found_player:
                baseline = self.REPLACEMENT_LEVEL.get(found_player.position, 1450)
                # QB injuries are 3x more impactful
                weight = 3.0 if found_player.position == 'QB' else 1.0
                vor = (found_player.elo - baseline) * weight
                
                if vor > 0:
                    impact += vor
                    if debug: print(f"  > PENALTY: {name} (-{vor/15.0:.1f} pts)")

        return impact / 15.0 # Convert Elo diff to approx Point Spread points