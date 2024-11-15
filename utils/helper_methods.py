from utils import constants

def agent_selects_and_plays_chess_move(chess_agent, environ, chess_data) -> str:
    curr_state = environ.get_curr_state() 
    chess_move = chess_agent.choose_action(chess_data, curr_state, curr_game)
    environ.load_chessboard(chess_move)
    environ.update_curr_state()
    return chess_move

def is_game_over(environ) -> bool:
    return (
        environ.board.is_game_over() or
        environ.turn_index >= constants.max_turn_index or
        (len(environ.get_legal_moves()) == 0)
    )

def get_game_outcome(environ) -> str:
    return environ.board.outcome().result()

def get_game_termination_reason(environ) -> str:
    return str(environ.board.outcome().termination)