import asyncio
import websockets
import json
import torch
import random
import numpy as np
import os
from collections import deque
from helper import plot
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

EPSILON_START = 1.0
EPSILON_MIN = 0.001
EPSILON_DECAY = 0.975

MODEL_WEIGHTS_FILE = 'model.pth'
CHECKPOINT_FILE = 'checkpoint.pth'

class Agent:

    def _load_progress_if_available(self):
        meta = self.trainer.load_checkpoint(CHECKPOINT_FILE)
        if meta is not None:
            self.n_games = int(meta.get('n_games', self.n_games))
            self.record = int(meta.get('record', self.record))
            self.total_score = int(meta.get('total_score', self.total_score))
            return True

        loaded_weights = self.model.load(MODEL_WEIGHTS_FILE)
        return bool(loaded_weights)

    def _save_progress(self):
        meta = {
            'n_games': self.n_games,
            'record': self.record,
            'total_score': self.total_score,
        }
        self.trainer.save_checkpoint(CHECKPOINT_FILE, meta=meta)
        self.model.save(MODEL_WEIGHTS_FILE)

    def _is_collision(self, pt, snake, rows, cols):
        if pt['row'] < 0 or pt['row'] >= rows or pt['col'] < 0 or pt['col'] >= cols:
            return True

        if any(seg['row'] == pt['row'] and seg['col'] == pt['col'] for seg in snake[1:]):
            return True

        return False
    
    def _rotate_left(self, dir):
        return {'row': -dir['col'], 'col': dir['row']}

    def _rotate_right(self, dir):
        return {'row': dir['col'], 'col': -dir['row']}

    def _flood_fill(self, start, snake, rows, cols):
        # treat all body segments except the tail as obstacles (tail vacates on move)
        occupied = set((s['row'], s['col']) for s in snake[:-1])
        if (start['row'] < 0 or start['row'] >= rows or
                start['col'] < 0 or start['col'] >= cols or
                (start['row'], start['col']) in occupied):
            return 0
        visited = set()
        queue = deque([(start['row'], start['col'])])
        visited.add((start['row'], start['col']))
        while queue:
            r, c = queue.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (0 <= nr < rows and 0 <= nc < cols
                        and (nr, nc) not in occupied
                        and (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return len(visited)


    def _get_state(self, rows, cols, snake, direction, food):

        head = snake[0]

        point_l = {'row': head['row'], 'col': head['col'] - 1}
        point_r = {'row': head['row'], 'col': head['col'] + 1}
        point_u = {'row': head['row'] - 1, 'col': head['col']}
        point_d = {'row': head['row'] + 1, 'col': head['col']}

        dir_l = direction == {'row': 0, 'col': -1}
        dir_r = direction == {'row': 0, 'col': 1}
        dir_u = direction == {'row': -1, 'col': 0}
        dir_d = direction == {'row': 1, 'col': 0}

        state = [
            (dir_r and self._is_collision(point_r, snake, rows, cols)) or
            (dir_l and self._is_collision(point_l, snake, rows, cols)) or
            (dir_u and self._is_collision(point_u, snake, rows, cols)) or
            (dir_d and self._is_collision(point_d, snake, rows, cols)),

            (dir_u and self._is_collision(point_r, snake, rows, cols)) or
            (dir_d and self._is_collision(point_l, snake, rows, cols)) or
            (dir_l and self._is_collision(point_u, snake, rows, cols)) or
            (dir_r and self._is_collision(point_d, snake, rows, cols)),

            (dir_d and self._is_collision(point_r, snake, rows, cols)) or
            (dir_u and self._is_collision(point_l, snake, rows, cols)) or
            (dir_r and self._is_collision(point_u, snake, rows, cols)) or
            (dir_l and self._is_collision(point_d, snake, rows, cols)),

            dir_l,
            dir_r,
            dir_u,
            dir_d,

            food['col'] < head['col'],  # food left
            food['col'] > head['col'],  # food right
            food['row'] < head['row'],  # food up
            food['row'] > head['row'],  # food down
        ]

        # flood fill: free area reachable after each possible action (normalised 0-1)
        total_cells = rows * cols
        dir_straight = direction
        dir_right = self._rotate_right(direction)
        dir_left = self._rotate_left(direction)
        next_straight = {'row': head['row'] + dir_straight['row'], 'col': head['col'] + dir_straight['col']}
        next_right    = {'row': head['row'] + dir_right['row'],    'col': head['col'] + dir_right['col']}
        next_left     = {'row': head['row'] + dir_left['row'],     'col': head['col'] + dir_left['col']}
        state += [
            self._flood_fill(next_straight, snake, rows, cols) / total_cells,
            self._flood_fill(next_right,    snake, rows, cols) / total_cells,
            self._flood_fill(next_left,     snake, rows, cols) / total_cells,
        ]

        return np.array(state, dtype=float)


    def _convert_action_to_direction(self, action, current_direction):
        clock_wise = [
            {'row': 0, 'col': 1},   # RIGHT
            {'row': 1, 'col': 0},   # DOWN
            {'row': 0, 'col': -1},  # LEFT
            {'row': -1, 'col': 0},  # UP
        ]

        idx = clock_wise.index(current_direction)

        if action == [1, 0, 0]:  # straight
            new_dir = clock_wise[idx]

        elif action == [0, 1, 0]:  # right turn
            new_dir = clock_wise[(idx + 1) % 4]

        else:  # left turn [0,0,1]
            new_dir = clock_wise[(idx - 1) % 4]

        return new_dir


    async def _handler(self, websocket):
        async for message in websocket:
            raw_state = json.loads(message)
            state_new  = self._get_state(
                raw_state['rows'],
                raw_state['columns'],
                raw_state['snake'],
                raw_state['direction'],
                raw_state['food'],
            )
            reward = raw_state.get('reward', 0)
            is_game_over = raw_state.get('isGameOver', False)
            score = raw_state.get('score', 0)
            

            current_direction = raw_state['direction']

            # print(f"State updated: {state_new}")
            
            if self.prev_state is not None:
                self.train_short_memory(
                    self.prev_state,
                    self.prev_action,
                    reward,
                    state_new,
                    is_game_over
                )

                self.remember(
                    self.prev_state,
                    self.prev_action,
                    reward,
                    state_new,
                    is_game_over
                )
        
            action = self.get_action(state_new)

            new_direction = self._convert_action_to_direction(action, current_direction)

            await websocket.send(json.dumps({"direction": new_direction}))

            self.prev_state = state_new
            self.prev_action = action

            if is_game_over:
                self.n_games += 1
                self.train_long_memory()

                self.prev_state = None
                self.prev_action = None

                if score > self.record:
                    self.record = score

                self._save_progress()

                print('Game', self.n_games, 'Score', score, 'Record:', self.record)

                self.plot_scores.append(score)
                self.total_score += score
                mean_score = self.total_score / self.n_games
                self.plot_mean_scores.append(mean_score)
                self.plot_epsilons.append(self.epsilon)
                plot(self.plot_scores, self.plot_mean_scores, epsilons=self.plot_epsilons, total_games=self.n_games)

                await websocket.send(json.dumps({"reset": True}))


    async def _init_websocket(self):
        server = await websockets.serve(self._handler, "localhost", 8765)
        print("Python agent running. Press Ctrl+C to stop.")
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            pass
        finally:
            server.close()
            await server.wait_closed()
            print("Server stopped gracefully.")


    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(14, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.prev_state = None
        self.prev_action = None
        self.plot_scores = []
        self.plot_mean_scores = []
        self.plot_epsilons = []
        self.total_score = 0
        self.record = 0

        resumed = self._load_progress_if_available()
        if resumed:
            print(f"Resumed training from saved model/checkpoint (games={self.n_games}, record={self.record}).")
        else:
            print("No saved model found; starting fresh training.")

        try:
            asyncio.run(self._init_websocket())
        except KeyboardInterrupt:
            print("\nAgent stopped by user.")
        finally:
            try:
                self._save_progress()
            except Exception:
                pass

    def remember(self, prev_state, prev_action, reward, state_new, is_game_over):
        self.memory.append((prev_state, prev_action, reward, state_new, is_game_over)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        prev_state, prev_action, rewards, state_new, is_game_over = zip(*mini_sample)
        self.trainer.train_step(prev_state, prev_action, rewards, state_new, is_game_over)
        # for prev_state, prev_action, reward, state_new, is_game_over in mini_sample:
        #     self.trainer.train_step(prev_state, prev_action, reward, state_new, is_game_over)

    def train_short_memory(self, prev_state, prev_action, reward, state_new, is_game_over):
        self.trainer.train_step(prev_state, prev_action, reward, state_new, is_game_over)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = max(EPSILON_MIN, EPSILON_START * (EPSILON_DECAY ** self.n_games))
        action = [0,0,0]
        if random.random() < (self.epsilon - 1):
            move = random.randint(0, 2)
            action[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            action[move] = 1

        return action


if __name__ == "__main__":
    Agent()