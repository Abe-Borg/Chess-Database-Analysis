import chess

class Environ:
    def __init__(self):
        self.board: chess.Board = chess.Board()            

    def get_curr_state(self):
        legal_moves = [self.board.san(move) for move in self.board.legal_moves]
        return {'legal_moves': legal_moves}
    
    def reset_environ(self) -> None:
        self.board.reset()
