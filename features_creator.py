import pandas as pd
import numpy as np
from pathlib import Path


class FeaturesCreator:
    """
    Creates advanced features for soccer match prediction based on team performance metrics.
    Features are calculated for both home_team and away_team using efficient vectorized operations.
    """
    
    def __init__(self, csv_path):
        """
        Initialize the FeaturesCreator with a CSV file.
        
        Args:
            csv_path: Path to the ranked_database.csv file
        """
        self.df = pd.read_csv(csv_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)
        self.df_with_features = None
        
    def calculate_points_won(self, home_score, away_score):
        """
        Calculate points won based on match result.
        Returns: 3 for win, 1 for draw, 0 for loss
        """
        if home_score > away_score:
            return 3
        elif home_score == away_score:
            return 1
        else:
            return 0
    
    def create_all_features(self):
        """
        Create all requested features for both home and away teams using efficient vectorized operations.
        Returns the dataframe with new feature columns.
        """
        # Create a copy to avoid modifying original
        df = self.df.copy()
        df = df.reset_index(drop=True)
        
        # ==================== BASIC POINTS FEATURE ====================
        df['home_points_won'] = df.apply(
            lambda row: self.calculate_points_won(row['home_score'], row['away_score']),
            axis=1
        )
        df['away_points_won'] = df.apply(
            lambda row: self.calculate_points_won(row['away_score'], row['home_score']),
            axis=1
        )
        
        # ==================== WEIGHTED POINTS (BY OPPONENT RANKING) ====================
        df['home_points_weighted'] = df['home_points_won'] / (1 + df['rank_away'] / 100)
        df['away_points_weighted'] = df['away_points_won'] / (1 + df['rank_home'] / 100)
        
        # ==================== RANK DIFFERENCE ====================
        df['rank_dif'] = df['rank_home'] - df['rank_away']
        
        # ==================== BUILD FEATURES FOR MOVING AVERAGES ====================
        print("Calculating moving averages...")
        df = self._add_all_moving_averages_efficient(df)
        
        self.df_with_features = df
        return df
    
    def _add_all_moving_averages_efficient(self, df):
        """
        Efficiently calculate all moving averages for home and away teams.
        Creates generic columns: home_points_won_ma_5, home_points_won_ma_3, away_points_won_ma_5, etc.
        """
        # List to store features for each row
        features_list = []
        
        for idx, row in df.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            current_date = row['date']
            
            # Get features for home team based on prior games
            home_features = self._calculate_team_features_at_date(
                df, home_team, current_date
            )
            
            # Get features for away team based on prior games
            away_features = self._calculate_team_features_at_date(
                df, away_team, current_date
            )
            
            # Create row with home_ and away_ prefixes
            row_features = {}
            for key, val in home_features.items():
                row_features[f'home_{key}'] = val
            for key, val in away_features.items():
                row_features[f'away_{key}'] = val
                
            features_list.append(row_features)
        
        # Convert list of dictionaries to dataframe and concatenate
        features_df = pd.DataFrame(features_list)
        df = pd.concat([df, features_df], axis=1)
        
        return df
    
    def _calculate_team_features_at_date(self, df, team, current_date):
        """
        Calculate all features for a team at a specific date using only prior games.
        Returns a dictionary with feature keys (without team name prefix).
        """
        # Get all games for this team before current_date
        team_games = self._get_team_games_before_date(df, team, current_date)
        
        features = {}
        
        if len(team_games) == 0:
            # No prior games, fill with NaN
            features['points_won_ma_5'] = np.nan
            features['points_won_ma_3'] = np.nan
            features['points_weighted_ma_5'] = np.nan
            features['points_weighted_ma_3'] = np.nan
            features['goals_ma_5'] = np.nan
            features['goals_ma_3'] = np.nan
            features['goals_suffered_ma_5'] = np.nan
            features['goals_suffered_ma_3'] = np.nan
            features['goals_weighted_ma_5'] = np.nan
            features['goals_weighted_ma_3'] = np.nan
            features['goals_suffered_weighted_ma_5'] = np.nan
            features['goals_suffered_weighted_ma_3'] = np.nan
        else:
            # Extract values for calculations
            points_won = team_games['points_won'].values
            points_weighted = team_games['points_weighted'].values
            goals = team_games['goals'].values
            goals_suffered = team_games['goals_suffered'].values
            goals_weighted = team_games['goals_weighted'].values
            goals_suffered_weighted = team_games['goals_suffered_weighted'].values
            
            # Calculate moving averages
            features['points_won_ma_5'] = np.mean(points_won[-5:]) if len(points_won) > 0 else np.nan
            features['points_won_ma_3'] = np.mean(points_won[-3:]) if len(points_won) > 0 else np.nan
            
            features['points_weighted_ma_5'] = np.mean(points_weighted[-5:]) if len(points_weighted) > 0 else np.nan
            features['points_weighted_ma_3'] = np.mean(points_weighted[-3:]) if len(points_weighted) > 0 else np.nan
            
            features['goals_ma_5'] = np.mean(goals[-5:]) if len(goals) > 0 else np.nan
            features['goals_ma_3'] = np.mean(goals[-3:]) if len(goals) > 0 else np.nan
            
            features['goals_suffered_ma_5'] = np.mean(goals_suffered[-5:]) if len(goals_suffered) > 0 else np.nan
            features['goals_suffered_ma_3'] = np.mean(goals_suffered[-3:]) if len(goals_suffered) > 0 else np.nan
            
            features['goals_weighted_ma_5'] = np.mean(goals_weighted[-5:]) if len(goals_weighted) > 0 else np.nan
            features['goals_weighted_ma_3'] = np.mean(goals_weighted[-3:]) if len(goals_weighted) > 0 else np.nan
            
            features['goals_suffered_weighted_ma_5'] = np.mean(goals_suffered_weighted[-5:]) if len(goals_suffered_weighted) > 0 else np.nan
            features['goals_suffered_weighted_ma_3'] = np.mean(goals_suffered_weighted[-3:]) if len(goals_suffered_weighted) > 0 else np.nan
        
        return features
    
    def _get_team_games_before_date(self, df, team, current_date):
        """
        Get all games for a team before a specific date with calculated metrics.
        """
        # Get home games
        home_games = df[(df['home_team'] == team) & (df['date'] < current_date)].copy()
        home_games['points_won'] = home_games.apply(
            lambda row: self.calculate_points_won(row['home_score'], row['away_score']),
            axis=1
        )
        home_games['points_weighted'] = home_games['points_won'] / (1 + home_games['rank_away'] / 100)
        home_games['goals'] = home_games['home_score']
        home_games['goals_suffered'] = home_games['away_score']
        home_games['goals_weighted'] = home_games['goals'] / (1 + home_games['rank_away'] / 100)
        home_games['goals_suffered_weighted'] = home_games['goals_suffered'] / (1 + home_games['rank_away'] / 100)
        
        # Get away games
        away_games = df[(df['away_team'] == team) & (df['date'] < current_date)].copy()
        away_games['points_won'] = away_games.apply(
            lambda row: self.calculate_points_won(row['away_score'], row['home_score']),
            axis=1
        )
        away_games['points_weighted'] = away_games['points_won'] / (1 + away_games['rank_home'] / 100)
        away_games['goals'] = away_games['away_score']
        away_games['goals_suffered'] = away_games['home_score']
        away_games['goals_weighted'] = away_games['goals'] / (1 + away_games['rank_home'] / 100)
        away_games['goals_suffered_weighted'] = away_games['goals_suffered'] / (1 + away_games['rank_home'] / 100)
        
        # Combine and sort
        team_games = pd.concat([home_games, away_games], ignore_index=True)
        team_games = team_games.sort_values('date').reset_index(drop=True)
        
        return team_games
    
    def save_to_csv(self, output_path):
        """Save the features dataframe to CSV."""
        self.df_with_features.to_csv(output_path, index=False)
        print(f"Features saved to {output_path}")
    
    def get_features_dataframe(self):
        """Return the dataframe with all features."""
        return self.df_with_features


if __name__ == "__main__":
    # Example usage
    csv_path = "data/ranked_database.csv"
    
    # Initialize the features creator
    creator = FeaturesCreator(csv_path)
    
    # Create all features
    print("Creating features...")
    df_features = creator.create_all_features()
    
    # Store the result
    creator.df_with_features = df_features
    
    # Save to CSV
    output_path = "data/ranked_database_with_features.csv"
    creator.save_to_csv(output_path)
    
    # Display sample of new features
    print("\nSample of created features:")
    feature_cols = [col for col in df_features.columns if any(
        x in col for x in ['ma_', '_weighted', '_won', '_goals']
    )]
    print(df_features[['date', 'home_team', 'away_team'] + feature_cols[:10]].head(10))
