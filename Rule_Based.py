import sys
import numpy as np
from loguru import logger


import itertools

BOARD_SIZE = 19

def is_valid(c, r):
    """Check if coordinates are within the board bounds."""
    return 0 <= c < BOARD_SIZE and 0 <= r < BOARD_SIZE

def find_patterns(board, player):
    """
    Return all the patterns found in this board.
    Patterns are defined as: (start_point, direction, to_fill)
    Means that starting from start_point, follow direction, filling to_fill will generate a winning pattern
    Of course, inpossible patterns will not be counted. That means that there will be no other players move in this pattern
    """
    # logger.debug(f"Finding patterns for player {player}")
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    patterns = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            for dir in directions:
                to_fill = []
                for i in range(6):
                    nr, nc = r + i * dir[0], c + i * dir[1]
                    if not is_valid(nr, nc) or board[nr][nc] == 3 - player:
                        break
                    if board[nr][nc] == 0:
                        to_fill.append((nr, nc))
                    if i == 5:
                        patterns.append(((r, c), dir, to_fill))
    return patterns

def generate_dangerous_positions(board, player, depth=2):
    # logger.debug("Findind dangerous positions")
    patterns = find_patterns(board, player)
    dangerous_patterns = [p for p in patterns if len(p[2]) <= depth]
    # logger.debug(dangerous_patterns)
    dangerous_positions = []
    for p in dangerous_patterns:
        dangerous_positions.extend(p[2])
    return dangerous_positions


def generate_winning_move(board):
    player, move = get_player_and_moves(board)
    dangerous_positions = generate_dangerous_positions(board, player, move)

    if not dangerous_positions:
        return None
    
    # logger.debug(f"Winning! {dangerous_positions[0]}")
    return dangerous_positions[0]
    

def generate_defending_move(board):
    """
    Finds a move for Black (player 1) to block an immediate win by White (player 2).
    This assumes Black cannot win on the current turn and plays 2 stones.

    Args:
        board: A 2D list representing the game board (e.g., 19x19).
               0 represents empty, 1 represents Black, 2 represents White.

    Returns:
        - ((r1, c1), (r2, c2)): The pair of coordinates Black should play to
                                 block White's immediate win.
        - None: If White does not have an immediate winning move to block.
    """
    dangerous_positions = generate_dangerous_positions(board, 2)

    if not dangerous_positions:
        return None
    
    # logger.debug("Defending!", dangerous_positions)
    
    for move1 in dangerous_positions:
        r, c = move1
        board[r][c] = 1
        d = generate_dangerous_positions(board, 2)
        if not d:
            board[r][c] = 0
            return move1
        board[r][c] = 0


    for move1, move2 in itertools.combinations(dangerous_positions, 2):
        r1, c1 = move1
        r2, c2 = move2

        board[r1][c1] = 1
        board[r2][c2] = 1

        d = generate_dangerous_positions(board, 2)
        if not d:
            board[r1][c1] = 0
            board[r2][c2] = 0
            return move1
        
        board[r1][c1] = 0
        board[r2][c2] = 0
    
    # logger.debug("GG, losing")
    return dangerous_positions[0]

def get_player_and_moves(board):
    count_1, count_2 = 0, 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == 1:
                count_1 += 1
            elif board[r][c] == 2:
                count_2 += 1

    # logger.debug(f"Player 1: {count_1}, Player 2: {count_2}")

    if count_1 == count_2:
        return 1, 1 # Black's first move
    elif count_1 > count_2:
        return 2, 2 # White's turn
    else: # count_1 == count_2 and count_1 > 0
        return 1, 2 # Black's turn

def get_empty_cells(board):
    return [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if board[r][c] == 0]

def generate_best_move(board):
    """
    Generates the best move using MCTS when no immediate win/loss is apparent.
    """
    dx = [1, 0, 1, 1, -1, 0, -1, -1]
    dy = [0, 1, 1, -1, 1, -1, 0, -1]

    def has_neighbor(r, c):
        for d in range(8):
            nr, nc = r + dx[d], c + dy[d]
            if is_valid(nr, nc) and board[nr][nc] != 0:
                return True
        return False

    empty = get_empty_cells(board)
    player, move = get_player_and_moves(board)
    
    best_score = -10000000
    best_move = (5, 5)

    if move == 1:
        for r, c in empty:
            if not has_neighbor(r, c):
                continue
            board[r][c] = 1
            score = evaluate_board(board)
            board[r][c] = 0
            if score > best_score:
                best_score = score
                best_move = (r, c)
    
    elif move == 2:
        for move1, move2 in itertools.combinations(empty, 2):
            r1, c1 = move1
            r2, c2 = move2
            if not has_neighbor(r1, c1) or not has_neighbor(r2, c2):
                continue
            board[r1][c1] = 1
            board[r2][c2] = 1
            score = evaluate_board(board)
            if score > best_score:
                best_score = score
                best_move = (r1, c1)
            board[r1][c1] = 0
            board[r2][c2] = 0
    
    return best_move        


def evaluate_board(board):
    black_patterns = find_patterns(board, 1)
    white_patterns = find_patterns(board, 2)
    
    score = 0
    for pattern in black_patterns:
        length = 6 - len(pattern[2])
        score += 15 ** (length * 2)
    
    for pattern in white_patterns:
        length = 6 - len(pattern[2])
        score -= 15 ** (1 + length * 2)
    return score


def select_move(board, color):
    if color == "W":
        board_cp = board.copy()
        for c in range(BOARD_SIZE):
            for r in range(BOARD_SIZE):
                if board_cp[c][r] != 0:
                    board_cp[c][r] = 3 - board_cp[c][r]
        return select_move(board_cp, "B")
    
    player, move = get_player_and_moves(board)
    # logger.debug(f"Player {player}, Move {move}")
    
    # logger.debug("Finding winning move...")
    winning_move = generate_winning_move(board)
    if winning_move:
        return winning_move
    
    # logger.debug("Finding defending move...")
    defending_move = generate_defending_move(board)
    if defending_move:
        return defending_move
    
    # logger.debug("No winning or defending, using MCTS")
    best_move = generate_best_move(board)    
    return best_move


class Connect6Game:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0: Empty, 1: Black, 2: White
        self.turn = 1  # 1: Black, 2: White
        self.game_over = False

    def reset_board(self):
        """Clears the board and resets the game."""
        self.board.fill(0)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)

    def set_board_size(self, size):
        """Sets the board size and resets the game."""
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)

    def check_win(self):
        """Checks if a player has won.
        Returns:
        0 - No winner yet
        1 - Black wins
        2 - White wins
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0:
                    current_color = self.board[r, c]
                    for dr, dc in directions:
                        prev_r, prev_c = r - dr, c - dc
                        if 0 <= prev_r < self.size and 0 <= prev_c < self.size and self.board[prev_r, prev_c] == current_color:
                            continue
                        count = 0
                        rr, cc = r, c
                        while 0 <= rr < self.size and 0 <= cc < self.size and self.board[rr, cc] == current_color:
                            count += 1
                            rr += dr
                            cc += dc
                        if count >= 6:
                            return current_color
        return 0

    def index_to_label(self, col):
        """Converts column index to letter (skipping 'I')."""
        return chr(ord('A') + col + (1 if col >= 8 else 0))  # Skips 'I'

    def label_to_index(self, col_char):
        """Converts letter to column index (accounting for missing 'I')."""
        col_char = col_char.upper()
        if col_char >= 'J':  # 'I' is skipped
            return ord(col_char) - ord('A') - 1
        else:
            return ord(col_char) - ord('A')

    def play_move(self, color, move):
        """Places stones and checks the game status."""
        if self.game_over:
            print("? Game over")
            return

        stones = move.split(',')
        positions = []

        for stone in stones:
            stone = stone.strip()
            if len(stone) < 2:
                print("? Invalid format")
                return
            col_char = stone[0].upper()
            if not col_char.isalpha():
                print("? Invalid format")
                return
            col = self.label_to_index(col_char)
            try:
                row = int(stone[1:]) - 1
            except ValueError:
                print("? Invalid format")
                return
            if not (0 <= row < self.size and 0 <= col < self.size):
                print("? Move out of board range")
                return
            if self.board[row, col] != 0:
                print("? Position already occupied")
                return
            positions.append((row, col))

        for row, col in positions:
            self.board[row, col] = 1 if color.upper() == 'B' else 2

        self.turn = 3 - self.turn
        print('= ', end='', flush=True)

    def generate_move(self, color):
        """Generates a random move for the computer."""
        if self.game_over:
            print("? Game over")
            return

        from contextlib import redirect_stdout
        with redirect_stdout(sys.stderr):
            selected = [select_move(self.board, color)]
        
        move_str = ",".join(f"{self.index_to_label(c)}{r+1}" for r, c in selected)
        
        self.play_move(color, move_str)

        print(f"{move_str}\n\n", end='', flush=True)
        print(move_str, file=sys.stderr)
        return

    def show_board(self):
        """Displays the board as text."""
        print("= ")
        for row in range(self.size - 1, -1, -1):
            line = f"{row+1:2} " + " ".join("X" if self.board[row, col] == 1 else "O" if self.board[row, col] == 2 else "." for col in range(self.size))
            print(line)
        col_labels = "   " + " ".join(self.index_to_label(i) for i in range(self.size))
        print(col_labels)
        print(flush=True)

    def list_commands(self):
        """Lists all available commands."""
        print("= ", flush=True)  

    def process_command(self, command):
        """Parses and executes GTP commands."""
        command = command.strip()
        if command == "get_conf_str env_board_size:":
            print("env_board_size=19", flush=True)

        if not command:
            return
        
        parts = command.split()
        cmd = parts[0].lower()

        if cmd == "boardsize":
            try:
                size = int(parts[1])
                self.set_board_size(size)
            except ValueError:
                print("? Invalid board size")
        elif cmd == "clear_board":
            self.reset_board()
        elif cmd == "play":
            if len(parts) < 3:
                print("? Invalid play command format")
            else:
                self.play_move(parts[1], parts[2])
                print('', flush=True)
        elif cmd == "genmove":
            if len(parts) < 2:
                print("? Invalid genmove command format")
            else:
                self.generate_move(parts[1])
        elif cmd == "showboard":
            self.show_board()
        elif cmd == "list_commands":
            self.list_commands()
        elif cmd == "quit":
            print("= ", flush=True)
            sys.exit(0)
        else:
            print("? Unsupported command")

    def run(self):
        """Main loop that reads GTP commands from standard input."""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                self.process_command(line)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"? Error: {str(e)}")

if __name__ == "__main__":
    game = Connect6Game()
    game.run()
