import pandas as pd
import numpy as np
from utils import game_settings
from pathlib import Path

def clean_chess_dataframe(df):
    """
    Clean chess DataFrame by:
    1. Dropping columns that are all empty (NaN or empty strings)
    2. Verifying move counts match PlyCount
    """
    print("\nInitial DataFrame shape:", df.shape)
    
    # Get list of move columns (W1, B1, W2, B2, etc.)
    move_columns = [col for col in df.columns if col.startswith(('W', 'B'))]
    
    # Function to check if a cell is effectively empty
    def is_empty(cell):
        if pd.isna(cell):  # Checks for NaN
            return True
        if isinstance(cell, str) and cell.strip() == '':  # Checks for empty strings
            return True
        return False
    
    # Drop columns that are all empty (NaN or empty strings)
    non_empty_cols = [col for col in move_columns 
                     if not df[col].apply(is_empty).all()]
    
    non_move_cols = [col for col in df.columns if not col.startswith(('W', 'B'))]
    columns_to_keep = non_move_cols + non_empty_cols
    
    # Create new DataFrame with only non-empty columns
    df_clean = df[columns_to_keep].copy()
    
    print("\nColumns dropped:", set(df.columns) - set(df_clean.columns))
    print("New DataFrame shape:", df_clean.shape)
    
    # Count actual moves and compare with PlyCount
    move_cols_clean = [col for col in df_clean.columns if col.startswith(('W', 'B'))]

    def count_not_empty_cells(row):
        return sum(not is_empty(cell) for cell in row)

    df_clean['ActualMoveCount'] = df_clean[move_cols_clean].apply(count_not_empty_cells, axis=1)
    df_clean['MovesMatch'] = df_clean['PlyCount'] == df_clean['ActualMoveCount']

    # Print summary
    mismatches = df_clean[~df_clean['MovesMatch']]
    print(f"\nFound {len(mismatches)} games with move count mismatches")
    
    if len(mismatches) > 0:
        print("\nSample of mismatched games:")
        sample_size = min(5, len(mismatches))
        for idx, row in mismatches.head(sample_size).iterrows():
            print(f"\nGame {idx}:")
            print(f"PlyCount: {row['PlyCount']}")
            print(f"Actual Moves: {row['ActualMoveCount']}")
            
            # Print some sample moves to help debug
            print("Sample of moves from this game:")
            move_cols = [col for col in row.index if col.startswith(('W', 'B'))]
            for col in move_cols[:5]:  # Show first 5 moves
                print(f"{col}: {row[col]} (type: {type(row[col])})")
    
    return df_clean

def main():
    pkl_file = game_settings.chess_games_filepath_part_2
    print(f"\n📊 Processing begins for {pkl_file}")
    
    # Load DataFrame
    chess_data = pd.read_pickle(pkl_file, compression='zip')
    chess_data = chess_data.head(1000)  # For testing
    
    # Clean and validate the DataFrame
    chess_data_clean = clean_chess_dataframe(chess_data)
    print("\nCleaned DataFrame:")
    print(chess_data_clean.head())
    print(chess_data.info())
    
    # # Save cleaned DataFrame
    # output_path = Path(pkl_file).parent / f"{Path(pkl_file).stem}_cleaned.pkl"
    # chess_data_clean.to_pickle(output_path, compression='zip')
    # print(f"\nCleaned DataFrame saved to: {output_path}")

if __name__ == '__main__':
    main()