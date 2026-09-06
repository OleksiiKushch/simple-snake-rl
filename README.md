# Snake RL

Simple Snake game with a reinforcement learning agent.

- Game (frontend): JavaScript
- Agent/Model: Python
- Communication: WebSocket

## Setup

```bash
pip install torch websockets numpy matplotlib
```

## Run

```bash
python agent.py
```

Then open `index.html` in a browser.

### Controls

- Arrow keys: move the snake
- `+` / `-` (or Numpad `+` / `-`): change game speed

## Headless training (no browser needed)

```bash
python agent.py --headless --speed 200 --no-plot
```

- `--headless`: run against the in-process game emulation instead of the browser/websocket.
- `--speed <n>`: game speed in moves per second (default: `10`).
- `--no-plot`: disable the matplotlib training plot (useful for background training).
