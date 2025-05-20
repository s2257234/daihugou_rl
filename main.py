from game.player import Player
from game.game import Game


def main():
    players = [Player(i) for i in range(4)]
    game = Game(players)
    game.play_game()
    for pid, action in game.history:
        print(f"Player {pid} played: {action}")

if __name__ == "__main__":
    main()
