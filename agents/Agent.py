class Agent:
    def __init__(self, color: str):
        self.color = color

    def choose_action(self, chess_data, environ_state, curr_game):
        return chess_data.at[curr_game, environ_state["curr_turn"]]