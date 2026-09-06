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

## Configuration

Settings are loaded from `config.json`; CLI flags override them.

```bash
python agent.py --no-explore --eps-start 0.1 --lr 0.01
```

- `--no-explore`: disable random actions (always use the model).
- `--eps-start` / `--eps-min` / `--eps-decay`: exploration rate. `eps` = probability that the agent makes a random move instead of asking the model. It starts at `--eps-start` (default `1.0` = 100% random) and is multiplied by `--eps-decay` (default `0.975`) after every game, down to `--eps-min` (default `0.001`): `eps = max(eps-min, eps-start * eps-decay^games)`. So after ~30 games it is ~0.47, after ~100 games ~0.08, after ~300 games ~0.0005.
- `--lr`: learning rate, the step size of weight updates (default `0.001`). How far each training step nudges the network weights toward the target value. Larger steps learn faster but risk overshooting: `0.1` often diverges (loss becomes NaN, model never learns). Smaller steps are safer but slower: `0.0001` still works, just takes longer. `0.001` is a solid default for the Adam optimizer; try `0.01`-`0.0001` when tuning.
- `--memory`: replay buffer size (default `100000`).
- `--batch-size`: replay batch size (default `1000`). Used by `train_long_memory`.
- `--config <path>`: use a different JSON config file.
