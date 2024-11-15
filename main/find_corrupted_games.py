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
        # Extract moves from columns
        moves = []
        for i in range(1, (game_row['PlyCount'] + 1) // 2 + 1):
            w_col = f'W{i}'
            b_col = f'B{i}'
            if w_col in game_row:
                moves.append(game_row[w_col])
            if b_col in game_row and pd.notna(game_row[b_col]):
                moves.append(game_row[b_col])
        
        # Validate each move
        for move_idx, move_san in enumerate(moves, 1):
            try:
                move = board.parse_san(move_san)
                if move not in board.legal_moves:
                    is_corrupted = True
                    error_type = "Illegal move"
                    error_detail = f"Move {move_idx}: {move_san} is not legal in position"
                    break
                move_sequence.append(move_san)
                board.push(move)
            except ValueError as e:
                is_corrupted = True
                error_type = "Invalid notation"
                error_detail = f"Move {move_idx}: {move_san} - {str(e)}"
                break
            
        # Validate game length
        if not is_corrupted and len(moves) != game_row['PlyCount']:
            is_corrupted = True
            error_type = "Move count mismatch"
            error_detail = f"Expected {game_row['PlyCount']} moves, found {len(moves)}"
            
    except Exception as e:
        is_corrupted = True
        error_type = "Unexpected error"
        error_detail = str(e)
    
    return {
        'game_id': game_row.name,
        'is_corrupted': is_corrupted,
        'error_type': error_type,
        'error_detail': error_detail,
        'moves_until_error': ' '.join(move_sequence)
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
    pkl_dir = game_settings
    
    total_corrupted = 0
    total_games = 0
    
    for pkl_file in pkl_dir.glob('*.pkl'):
        print(f"\n📊 Processing {pkl_file}")
        
        # Load DataFrame
        df = pd.read_pickle(pkl_file)
        total_games += len(df)
        
        # Validate games
        results_df = validate_dataframe_parallel(df)
        
        # Print detailed report
        print_corruption_report(results_df, pkl_file.name)
        
        # Update totals
        total_corrupted += results_df['is_corrupted'].sum()
    
    # Print final summary
    print("\n📑 Final Summary")
    print("=" * 40)
    print(f"Total files processed: {len(list(pkl_dir.glob('*.pkl')))}")
    print(f"Total games analyzed: {total_games}")
    print(f"Total corrupted games: {total_corrupted}")
    print(f"Overall corruption rate: {(total_corrupted/total_games)*100:.2f}%")

if __name__ == '__main__':
    main()