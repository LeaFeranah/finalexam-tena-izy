import tkinter as tk
from tkinter import messagebox
import csv
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# === Classe pour gérer le dataset et le modèle ML ===
class GameData:
    def __init__(self, filename="tictactoe_dataset.csv"):
        self.filename = filename
        self.X = []
        self.y = []
        self.model = None
        self.load_data()
        self.train_model()

    def encode_board(self, board):
        encoding = {' ': 0, 'X': 1, 'O': -1}
        return [encoding[s] for s in board]

    def add_sample(self, board, move):
        self.X.append(self.encode_board(board))
        self.y.append(move)
        self.save_sample(self.X[-1], self.y[-1])
        self.train_model()

    def save_sample(self, x, y):
        file_exists = os.path.isfile(self.filename)
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([f'x{i}' for i in range(9)] + ['y'])
            writer.writerow(x + [y])

    def load_data(self):
        if not os.path.isfile(self.filename):
            return
        with open(self.filename, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.X.append([int(x) for x in row[:-1]])
                self.y.append(int(row[-1]))

    def train_model(self):
        if len(self.X) < 10:
            return
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model = RandomForestClassifier()
        self.model.fit(X_train, y_train)
        acc = accuracy_score(y_test, self.model.predict(X_test))
        print(f" Modèle entraîné avec une précision de {acc:.2f}")

    def predict_move(self, board):
        if self.model is None:
            return None
        encoded = np.array([self.encode_board(board)])
        pred = self.model.predict(encoded)[0]
        if board[pred] == ' ':
            return pred
        else:
            available = [i for i, s in enumerate(board) if s == ' ']
            return np.random.choice(available) if available else None


# === Logique du jeu ===
def check_winner(board):
    wins = [
        (0,1,2), (3,4,5), (6,7,8),
        (0,3,6), (1,4,7), (2,5,8),
        (0,4,8), (2,4,6)
    ]
    for a,b,c in wins:
        if board[a] == board[b] == board[c] != ' ':
            return board[a]
    return None

def is_full(board):
    return all(s != ' ' for s in board)

def minimax(board, is_ai_turn, alpha, beta):
    winner = check_winner(board)
    if winner == 'O':
        return 1, None
    elif winner == 'X':
        return -1, None
    elif is_full(board):
        return 0, None

    if is_ai_turn:
        max_eval = -float('inf')
        best_move = None
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'O'
                eval, _ = minimax(board, False, alpha, beta)
                board[i] = ' '
                if eval > max_eval:
                    max_eval = eval
                    best_move = i
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        best_move = None
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'X'
                eval, _ = minimax(board, True, alpha, beta)
                board[i] = ' '
                if eval < min_eval:
                    min_eval = eval
                    best_move = i
                beta = min(beta, eval)
                if beta <= alpha:
                    break
        return min_eval, best_move


# === Interface graphique ===
class TicTacToeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic-Tac-Toe IA + Dataset ML")
        self.root.configure(bg="#2c2f33")
        self.buttons = []
        self.board = [' '] * 9
        self.player_turn = None
        self.score_x = 0
        self.score_o = 0
        self.score_draw = 0
        self.data = GameData()
        self.ai_moves_for_dataset = []

        self.create_widgets()
        self.show_starter_buttons()
        self.disable_all_buttons()

    def create_widgets(self):
        self.title = tk.Label(self.root, text="Tic-Tac-Toe IA", font=("Helvetica", 20, "bold"), fg="#ffffff", bg="#2c2f33")
        self.title.pack(pady=(10, 5))

        self.score_label = tk.Label(self.root, text=self.get_score_text(), font=("Segoe UI", 12), fg="#ffffff", bg="#2c2f33")
        self.score_label.pack(pady=(0, 10))

        self.button_frame = tk.Frame(self.root, bg="#2c2f33")
        self.button_frame.pack()

        for i in range(9):
            btn = tk.Button(self.button_frame, text=' ', font=('Helvetica', 20, 'bold'), width=3, height=1,
                            bg="#ffffff", fg="#000000", activebackground="#7289da",
                            command=lambda i=i: self.player_move(i))
            btn.grid(row=i//3, column=i%3, padx=4, pady=4)
            self.buttons.append(btn)

        self.reset_btn = tk.Button(self.root, text="🔁 Recommencer", font=("Segoe UI", 12), bg="#7289da",
                                   fg="#ffffff", activebackground="#99aab5", relief="ridge", bd=3,
                                   padx=8, pady=4, command=self.reset_game)
        self.reset_btn.pack(pady=10)

        self.starter_frame = tk.Frame(self.root, bg="#2c2f33")
        self.starter_label = tk.Label(self.starter_frame, text="Qui commence ?", font=("Segoe UI", 11), fg="white", bg="#2c2f33")
        self.starter_label.pack(pady=(0,5))

        self.player_first_btn = tk.Button(self.starter_frame, text="🎮 Joueur (X)", bg="#43b581", fg="white",
                                          font=("Segoe UI", 11), command=self.set_player_first)
        self.player_first_btn.pack(side="left", padx=10)

        self.ai_first_btn = tk.Button(self.starter_frame, text="🤖 IA (O)", bg="#f04747", fg="white",
                                      font=("Segoe UI", 11), command=self.set_ai_first)
        self.ai_first_btn.pack(side="right", padx=10)

    def get_score_text(self):
        return f"🎮 Joueur (X): {self.score_x}   🤖 IA (O): {self.score_o}   ➖ Nuls: {self.score_draw}"

    def update_score_label(self):
        self.score_label.config(text=self.get_score_text())

    def show_starter_buttons(self):
        self.starter_frame.pack(pady=10)

    def hide_starter_buttons(self):
        self.starter_frame.pack_forget()

    def set_player_first(self):
        self.player_turn = True
        self.hide_starter_buttons()
        self.enable_all_buttons()

    def set_ai_first(self):
        self.player_turn = False
        self.hide_starter_buttons()
        self.enable_all_buttons()
        self.root.after(300, self.ai_move)

    def player_move(self, index):
        if self.board[index] == ' ' and self.player_turn:
            self.board[index] = 'X'
            self.buttons[index]['text'] = 'X'
            self.buttons[index]['fg'] = '#e74c3c'
            self.player_turn = False
            self.ai_moves_for_dataset.append((self.board.copy(), index))
            if self.check_game_end():
                return
            self.root.after(300, self.ai_move)

    def ai_move(self):
        move = self.data.predict_move(self.board)
        if move is None:
            _, move = minimax(self.board, True, -float('inf'), float('inf'))
        if move is not None:
            self.board[move] = 'O'
            self.buttons[move]['text'] = 'O'
            self.buttons[move]['fg'] = '#3498db'
        self.player_turn = True
        self.check_game_end()

    def check_game_end(self):
        winner = check_winner(self.board)
        if winner:
            if winner == 'X':
                self.score_x += 1
            elif winner == 'O':
                self.score_o += 1
            self.update_score_label()
            messagebox.showinfo("Fin du jeu", f"Le joueur {winner} a gagné !")
            self.disable_all_buttons()
            self.show_starter_buttons()
            self.save_game_data()
            return True
        elif is_full(self.board):
            self.score_draw += 1
            self.update_score_label()
            messagebox.showinfo("Fin du jeu", "Match nul !")
            self.disable_all_buttons()
            self.show_starter_buttons()
            self.save_game_data()
            return True
        return False

    def save_game_data(self):
        for board_state, move_played in self.ai_moves_for_dataset:
            self.data.add_sample(board_state, move_played)
        self.ai_moves_for_dataset.clear()

    def disable_all_buttons(self):
        for btn in self.buttons:
            btn.config(state=tk.DISABLED)

    def enable_all_buttons(self):
        for i, btn in enumerate(self.buttons):
            if self.board[i] == ' ':
                btn.config(state=tk.NORMAL)

    def reset_game(self):
        self.board = [' '] * 9
        for btn in self.buttons:
            btn.config(text=' ', state=tk.DISABLED, fg="#000000")
        self.player_turn = None
        self.show_starter_buttons()

# === Lancement de l'application ===
if __name__ == "__main__":
    root = tk.Tk()
    app = TicTacToeGUI(root)
    root.mainloop()
