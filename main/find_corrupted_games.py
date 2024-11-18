import pandas as pd
import chess
from utils import game_settings
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
        # Extract moves from columns (modified to be more robust)
        moves = []
        max_moves = (game_row['PlyCount'] + 1) // 2
        for i in range(1, max_moves + 1):
            w_col = f'W{i}'
            b_col = f'B{i}'
            
            # Check white's move
            if w_col in game_row and pd.notna(game_row[w_col]):
                white_move = str(game_row[w_col]).strip()  # Convert to string and strip whitespace
                if white_move:  # Only append if move isn't empty
                    moves.append(white_move)
            
            # Check black's move
            if b_col in game_row and pd.notna(game_row[b_col]):
                black_move = str(game_row[b_col]).strip()  # Convert to string and strip whitespace
                if black_move:  # Only append if move isn't empty
                    moves.append(black_move)
        
        # Validate each move with better error handling
        for move_idx, move_san in enumerate(moves, 1):
            try:
                # Skip empty moves
                if not move_san or pd.isna(move_san):
                    continue
                    
                # Handle special case moves
                if move_san in ['O-O', 'O-O-O']:  # Castle moves
                    if move_san == 'O-O':
                        move = board.parse_san('O-O')
                    else:
                        move = board.parse_san('O-O-O')
                else:
                    move = board.parse_san(move_san)
                
                # Check if move is legal
                if move not in board.legal_moves:
                    is_corrupted = True
                    error_type = "Illegal move"
                    error_detail = f"Move {move_idx}: {move_san} is not legal in position\nFEN: {board.fen()}"
                    break
                
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
            actual_moves = len([m for m in moves if m and not pd.isna(m)])
            if actual_moves != game_row['PlyCount']:
                is_corrupted = True
                error_type = "Move count mismatch"
                error_detail = f"Expected {game_row['PlyCount']} moves, found {actual_moves}"
            
    except Exception as e:
        is_corrupted = True
        error_type = "Unexpected error"
        error_detail = f"Error processing game: {str(e)}"
    
    return {
        'game_id': game_row.name,
        'is_corrupted': is_corrupted,
        'error_type': error_type,
        'error_detail': error_detail,
        'moves_until_error': ' '.join(move_sequence),
        'final_position': board.fen() if not is_corrupted else None
    }

def process_chunk(chunk_df):
    """Process a chunk of games."""
    results = []
    for idx, row in chunk_df.iterrows():
        result = validate_single_game(row)
        results.append(result)
    return results

def validate_dataframe_parallel(df, num_processes=None):
    """Validate a DataFrame of chess games using parallel processing."""
    if num_processes is None:
        num_processes = mp.cpu_count()
        
    chunk_size = len(df) // (num_processes * 4)
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_chunk, chunks)
    
    flat_results = [item for sublist in results for item in sublist]
    return pd.DataFrame(flat_results)

def print_corruption_report(results_df, file_name):
    """Print a detailed report of corrupted games."""
    corrupted_df = results_df[results_df['is_corrupted']].copy()
    
    if len(corrupted_df) == 0:
        print(f"\n✅ No corrupted games found in {file_name}")
        return
        
    print(f"\n🔍 Corruption Report for {file_name}")
    print("=" * 80)
    
    # Summary statistics
    print(f"Total games analyzed: {len(results_df)}")
    print(f"Corrupted games found: {len(corrupted_df)} ({(len(corrupted_df)/len(results_df))*100:.2f}%)")
    
    # Error type distribution
    print("\nError Type Distribution:")
    print("-" * 40)
    error_dist = corrupted_df['error_type'].value_counts()
    for error_type, count in error_dist.items():
        print(f"{error_type}: {count} games")
    
    # Detailed corruption report
    print("\nDetailed Corruption Report:")
    print("-" * 80)
    
    corruption_details = []
    for idx, row in corrupted_df.iterrows():
        corruption_details.append([
            row['game_id'],
            row['error_type'],
            row['error_detail'],
            row['moves_until_error'][:50] + "..." if len(row['moves_until_error']) > 50 else row['moves_until_error']
        ])
    
    print(tabulate(
        corruption_details,
        headers=['Game ID', 'Error Type', 'Error Detail', 'Moves Until Error'],
        tablefmt='grid'
    ))
    
    # Save detailed report to file
    report_file = f"{file_name}_corruption_report.txt"
    with open(report_file, 'w') as f:
        f.write(f"Corruption Report for {file_name}\n")
        f.write("=" * 80 + "\n")
        f.write(tabulate(corruption_details, headers=['Game ID', 'Error Type', 'Error Detail', 'Moves Until Error'], tablefmt='grid'))
    
    print(f"\n📝 Detailed report saved to {report_file}")


def main():    
    pkl_file = game_settings.chess_games_filepath_part_2

    print(f"\n📊 Processing begins...")
    chess_data = pd.read_pickle(pkl_file, compression='zip')
    chess_data = chess_data.head(10000)
    
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
            print("Moves until error:", row['moves_until_error'])
            print("-" * 40)

if __name__ == '__main__':
    main()