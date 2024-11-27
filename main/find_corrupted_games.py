import pandas as pd
import chess
from utils import program_settings
import multiprocessing as mp
from pathlib import Path
import numpy as np
from tabulate import tabulate  # For prettier console output

def validate_single_game(game_row):
    """Validate a single chess game."""
    board = chess.Board()
    is_corrupted = False
    error_type = None
    error_detail = None
    move_sequence = []
    
    try:
        # Extract moves from columns
        moves = []
        max_moves = (game_row['PlyCount'] + 1) // 2
        for i in range(1, max_moves + 1):
            w_col = f'W{i}'
            b_col = f'B{i}'
            
            # Only append white's move if it exists and is not NaN
            if w_col in game_row and pd.notna(game_row[w_col]):
                moves.append(str(game_row[w_col]))
                
            # Only append black's move if it exists and is not NaN
            if b_col in game_row and pd.notna(game_row[b_col]):
                moves.append(str(game_row[b_col]))
        
        # Validate each move
        for move_idx, move_san in enumerate(moves, 1):
            try:
                if not move_san:  # Skip empty moves
                    continue
                    
                move = board.parse_san(move_san)
                move_sequence.append(move_san)
                board.push(move)
                
            except ValueError as e:
                is_corrupted = True
                error_type = "Invalid notation"
                error_detail = f"Move {move_idx}: {move_san} - {str(e)}\nFEN: {board.fen()}"
                break
            except Exception as e:
                is_corrupted = True
                error_type = "Move processing error"
                error_detail = f"Move {move_idx}: {move_san} - {str(e)}\nFEN: {board.fen()}"
                break
        
        # Only check move count if no other errors were found
        if not is_corrupted:
            valid_moves = len(move_sequence)  # Use actual processed moves instead
            if valid_moves != game_row['PlyCount']:
                is_corrupted = True
                error_type = "Move count mismatch"
                error_detail = f"Expected {game_row['PlyCount']} moves, found {valid_moves}"
            
    except Exception as e:
        is_corrupted = True
        error_type = "Unexpected error"
        error_detail = f"Error processing game: {str(e)}"
    
    return {
        'game_id': game_row.name,
        'is_corrupted': is_corrupted,
        'error_type': error_type,
        'error_detail': error_detail,
        'final_position': board.fen() if not is_corrupted else None
    }

def process_chunk(chunk_df):
    """Process a chunk of games."""
    results = []
    for _, row in chunk_df.iterrows():
        result = validate_single_game(row)
        results.append(result)
    return results

def validate_dataframe_parallel(df, num_processes=None):
    """Validate a DataFrame of chess games using parallel processing."""
    if num_processes is None:
        num_processes = mp.cpu_count()
        
    chunk_size = max(1, len(df) // (num_processes * 4))
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_chunk, chunks)
    
    flat_results = [item for sublist in results for item in sublist]
    return pd.DataFrame(flat_results)


def main():    
    pkl_file = program_settings.chess_games_filepath_part_2

    print(f"\n📊 Processing begins...")
    chess_data = pd.read_pickle(pkl_file, compression='zip')
    chess_data = chess_data.head(1000)
    
    # Add some debug info
    print(f"\nDataFrame Info:")
    print("-" * 40)
    print(chess_data.info())
    print("\nSample of first game:")
    print("-" * 40)
    first_game = chess_data.iloc[0]
    for col in chess_data.columns:
        if pd.notna(first_game[col]):
            print(f"{col}: {first_game[col]}")
    
    # Validate games
    results_df = validate_dataframe_parallel(chess_data)
    
    # Enhanced reporting
    corrupted_games = results_df[results_df['is_corrupted']]
    print("\n📑 Corruption Analysis")
    print("=" * 40)
    print(f"Total games analyzed: {len(chess_data)}")
    print(f"Total corrupted games: {len(corrupted_games)}")
    
    if not corrupted_games.empty:
        print("\nError Type Distribution:")
        print(corrupted_games['error_type'].value_counts())
        print("\nSample of corrupted games:")
        sample_size = min(5, len(corrupted_games))

        for _, row in corrupted_games.head(sample_size).iterrows():
            print("\nGame ID:", row['game_id'])
            print("Error Type:", row['error_type'])
            print("Error Detail:", row['error_detail'])
            print("-" * 40)

if __name__ == '__main__':
    main()