import asyncio
import argparse
import websockets
import json
import torch
import random
import numpy as np
import os
from collections import deque
from helper import plot
from model import Linear_QNet, QTrainer
from emulator import SnakeEmulator

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

EPSILON_START = 1.0
EPSILON_MIN = 0.001
EPSILON_DECAY = 0.975

MODEL_WEIGHTS_FILE = 'model.pth'
CHECKPOINT_FILE = 'checkpoint.pth'
CONFIG_FILE = 'config.json'


def _load_config_file(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print(f"Warning: {path} must be a JSON object; ignoring.")
            return {}
        return data
    except Exception as e:
        print(f"Warning: could not read {path}: {e}")
        return {}

class Agent:

    def _load_progress_if_available(self):
        try:
            meta = self.trainer.load_checkpoint(CHECKPOINT_FILE)
        except Exception:
            print("Saved checkpoint is incompatible with current model architecture; starting fresh.")
            return False

        if meta is not None:
            self.n_games = int(meta.get('n_games', self.n_games))
            self.record = int(meta.get('record', self.record))
            self.total_score = int(meta.get('total_score', self.total_score))
            return True

        try:
            loaded_weights = self.model.load(MODEL_WEIGHTS_FILE)
        except Exception:
            print("Saved model weights are incompatible with current model architecture; starting fresh.")
            return False

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
            _, new_direction = self._process_state(raw_state)
            await websocket.send(json.dumps({"direction": new_direction}))
            if raw_state.get('isGameOver', False):
                await websocket.send(json.dumps({"reset": True}))


    def _process_state(self, raw_state):
        """Shared per-step logic for websocket and headless modes.
        Returns (action, new_direction)."""
        state_new = self._get_state(
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

        self.prev_state = state_new
        self.prev_action = action

        if is_game_over:
            self._on_game_over(score)

        return action, new_direction

    def _on_game_over(self, score):
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
        if self.enable_plot:
            plot(self.plot_scores, self.plot_mean_scores, epsilons=self.plot_epsilons, total_games=self.n_games)


    async def _run_emulation(self, speed):
        emulator = SnakeEmulator()
        step_delay = 1.0 / speed
        print(f"Headless mode: training against in-process game emulation ({speed} moves/sec).")
        print("Press Ctrl+C to stop.")
        while True:
            _, new_direction = self._process_state(emulator.get_state())
            emulator.direction = new_direction
            if emulator.is_game_over:
                emulator.reset()
            else:
                emulator.move()
            await asyncio.sleep(step_delay)


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


    def __init__(self, headless=False, speed=10.0, enable_plot=True, explore=True,
                 max_memory=MAX_MEMORY, batch_size=BATCH_SIZE, lr=LR,
                 eps_start=EPSILON_START, eps_min=EPSILON_MIN, eps_decay=EPSILON_DECAY):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.explore = explore
        self.eps_start = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=max_memory) # popleft()
        self.model = Linear_QNet(14, 256, 3)
        self.trainer = QTrainer(self.model, lr=lr, gamma=self.gamma)
        self.prev_state = None
        self.prev_action = None
        self.plot_scores = []
        self.plot_mean_scores = []
        self.plot_epsilons = []
        self.total_score = 0
        self.record = 0
        self.enable_plot = enable_plot

        resumed = self._load_progress_if_available()
        if resumed:
            print(f"Resumed training from saved model/checkpoint (games={self.n_games}, record={self.record}).")
        else:
            print("No saved model found; starting fresh training.")

        try:
            if headless:
                asyncio.run(self._run_emulation(speed))
            else:
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
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size) # list of tuples
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
        if self.explore:
            self.epsilon = max(self.eps_min, self.eps_start * (self.eps_decay ** self.n_games))
        else:
            self.epsilon = 0.0
        action = [0,0,0]
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            action[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            action[move] = 1

        return action


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Snake RL agent")
    parser.add_argument('--config', default=CONFIG_FILE,
                        help=f'path to JSON config file (default: {CONFIG_FILE})')
    parser.add_argument('--headless', action='store_true',
                        help='train against the built-in headless game emulation '
                             '(no browser/websocket needed)')
    parser.add_argument('--speed', type=float,
                        help='game speed in moves per second for headless mode '
                             '(default: 10.0, same as the browser default)')
    parser.add_argument('--no-plot', action='store_true',
                        help='disable the matplotlib training plot')
    parser.add_argument('--no-explore', action='store_true',
                        help='disable random exploration (always use the model)')
    parser.add_argument('--eps-start', type=float,
                        help='initial exploration rate (0-1)')
    parser.add_argument('--eps-min', type=float,
                        help='minimum exploration rate (0-1)')
    parser.add_argument('--eps-decay', type=float,
                        help='exploration decay factor per game (0-1)')
    parser.add_argument('--lr', type=float,
                        help='learning rate')
    parser.add_argument('--batch-size', type=int,
                        help='replay batch size')
    parser.add_argument('--memory', type=int,
                        help='replay buffer size')
    args = parser.parse_args()

    config = _load_config_file(args.config)

    def pick(name, cli_value, default):
        return cli_value if cli_value is not None else config.get(name, default)

    speed = pick('speed', args.speed, 10.0)
    if speed <= 0:
        parser.error('--speed must be > 0')

    lr = pick('lr', args.lr, LR)
    if lr <= 0:
        parser.error('--lr must be > 0')

    batch_size = pick('batch_size', args.batch_size, BATCH_SIZE)
    if batch_size <= 0:
        parser.error('--batch-size must be > 0')

    max_memory = pick('max_memory', args.memory, MAX_MEMORY)
    if max_memory <= 0:
        parser.error('--memory must be > 0')

    eps_start = pick('eps_start', args.eps_start, EPSILON_START)
    eps_min = pick('eps_min', args.eps_min, EPSILON_MIN)
    eps_decay = pick('eps_decay', args.eps_decay, EPSILON_DECAY)
    for name, value in (('--eps-start', eps_start), ('--eps-min', eps_min), ('--eps-decay', eps_decay)):
        if not 0 <= value <= 1:
            parser.error(f'{name} must be between 0 and 1')

    headless = args.headless or bool(config.get('headless', False))
    enable_plot = not args.no_plot and bool(config.get('plot', True))
    explore = not args.no_explore and bool(config.get('explore', True))

    Agent(
        headless=headless,
        speed=speed,
        enable_plot=enable_plot,
        explore=explore,
        max_memory=max_memory,
        batch_size=batch_size,
        lr=lr,
        eps_start=eps_start,
        eps_min=eps_min,
        eps_decay=eps_decay,
    )