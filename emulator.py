import random

ROWS = 24
COLUMNS = 32


class SnakeEmulator:
    """Headless re-implementation of the browser Snake game (script.js).

    No rendering, no websocket: replicates the game rules exactly so the
    agent can train against it without a browser tab.
    """

    def __init__(self):
        self.reset(initial_start=True)

    def place_food(self):
        while True:
            row = random.randrange(ROWS)
            col = random.randrange(COLUMNS)
            if not any(s['row'] == row and s['col'] == col for s in self.snake):
                self.food = {'row': row, 'col': col}
                return

    def reset(self, initial_start=False):
        if initial_start:
            self.snake = [{'row': 12, 'col': 16}]
        else:
            self.snake = [{'row': 5, 'col': 5}]
        self.direction = {'row': 0, 'col': 1}
        self.score = 0
        self.reward = 0
        self.is_game_over = False
        self.place_food()

    def move(self):
        self.reward = 0

        head = self.snake[0]
        new_head = {
            'row': head['row'] + self.direction['row'],
            'col': head['col'] + self.direction['col'],
        }

        if (
            new_head['row'] < 0 or
            new_head['row'] >= ROWS or
            new_head['col'] < 0 or
            new_head['col'] >= COLUMNS or
            any(s['row'] == new_head['row'] and s['col'] == new_head['col'] for s in self.snake)
        ):
            self.reward = -10
            self.is_game_over = True
            return

        self.snake.insert(0, new_head)

        if new_head['row'] == self.food['row'] and new_head['col'] == self.food['col']:
            self.place_food()
            self.score += 1
            self.reward = 10
        else:
            self.snake.pop()

    def get_state(self):
        return {
            'rows': ROWS,
            'columns': COLUMNS,
            'snake': [{'row': s['row'], 'col': s['col']} for s in self.snake],
            'direction': {'row': self.direction['row'], 'col': self.direction['col']},
            'food': {'row': self.food['row'], 'col': self.food['col']},
            'score': self.score,
            'reward': self.reward,
            'isGameOver': self.is_game_over,
        }