const ROWS = 24;
const COLUMNS = 32;
const UPDATE_INTERVAL = 100;
const board = document.getElementById('board');

const MIN_UPDATE_INTERVAL = 5;
const MAX_UPDATE_INTERVAL = 200;
const UPDATE_INTERVAL_STEP = 5;

let snake = [{ row: 12, col: 16 }];
let direction = { row: 0, col: 1 };
let food = { row: 0, col: 0 };

let score = 0;
let reward = 0;
let isGameOver = false;

let currentUpdateInterval = UPDATE_INTERVAL;
let pendingUpdateInterval = null;
let gameInterval = null;

function createBoard() {
    if (!board) throw new Error('Board element not found');

    for (let r = 0; r < ROWS; r++) {
        const tr = document.createElement('tr');
        for (let c = 0; c < COLUMNS; c++) {
            const td = document.createElement('td');
            td.id = `cell-${r}-${c}`;
            tr.appendChild(td);
        }
        board.appendChild(tr);
    }
}

function draw() {
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLUMNS; c++) {
            const cell = document.getElementById(`cell-${r}-${c}`);
            if (cell) cell.className = '';
        }
    }

    snake.forEach((segment, index) => {
        const cell = document.getElementById(`cell-${segment.row}-${segment.col}`);
        if (cell) cell.classList.add(index === 0 ? 'head' : 'snake');
    });

    const foodCell = document.getElementById(`cell-${food.row}-${food.col}`);
    if (foodCell) foodCell.classList.add('food');

    updateScore();
}

function updateScore() {
    const scoreElement = document.getElementById('score');
    if (scoreElement) {
        scoreElement.textContent = String(score);
    }
}

function updateSpeed() {
    const speedElement = document.getElementById('speed');
    if (speedElement) {
        speedElement.textContent = String((1000 / currentUpdateInterval).toFixed(1));
    }
}

function startGameLoop() {
    if (gameInterval) clearInterval(gameInterval);
    gameInterval = setInterval(move, currentUpdateInterval);
    updateSpeed();
}

function clampUpdateInterval(value) {
    return Math.max(MIN_UPDATE_INTERVAL, Math.min(MAX_UPDATE_INTERVAL, value));
}

function queueSpeedChange(delta) {
    const base = pendingUpdateInterval ?? currentUpdateInterval;
    pendingUpdateInterval = clampUpdateInterval(base + delta);
}

function placeFood() {
    let valid = false;
    while (!valid) {
        const newRow = Math.floor(Math.random() * ROWS);
        const newCol = Math.floor(Math.random() * COLUMNS);
        valid = !snake.some((segment) => segment.row === newRow && segment.col === newCol);
        if (valid) {
            food = { row: newRow, col: newCol };
        }
    }
}

function move() {
    reward = 0;

    const head = snake[0];
    const newHead = {
        row: head.row + direction.row,
        col: head.col + direction.col,
    };

    if (
        newHead.row < 0 ||
        newHead.row >= ROWS ||
        newHead.col < 0 ||
        newHead.col >= COLUMNS ||
        snake.some((segment) => segment.row === newHead.row && segment.col === newHead.col)
    ) {
        clearInterval(gameInterval);
        reward = -10;
        isGameOver = true;
        publishState();
        return;
    }

    snake.unshift(newHead);

    if (newHead.row === food.row && newHead.col === food.col) {
        placeFood();
        score++;
        reward = 10;
    } else {
        snake.pop();
    }

    draw();

    publishState();

    if (pendingUpdateInterval !== null && pendingUpdateInterval !== currentUpdateInterval) {
        currentUpdateInterval = pendingUpdateInterval;
        pendingUpdateInterval = null;
        startGameLoop();
    }
}

function publishState() {
    if (!window.socket) {
        window.socket = new WebSocket('ws://localhost:8765');

        window.socket.onopen = () => {
            console.log('Connected to Python agent');
            window.socket.send(JSON.stringify(getGameState()));
        };

        window.socket.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.reset) {
                snake = [{ row: 5, col: 5 }];
                direction = { row: 0, col: 1 };
                score = 0;
                isGameOver = false;
                startGameLoop();
                return;

                // window.location.reload();
            }

            if (data.direction) {
                direction = data.direction;
            }
        };

        window.socket.onclose = () => console.log('Socket closed');
        window.socket.onerror = (error) => console.log('Socket error', error);
    } else if (window.socket.readyState === WebSocket.OPEN) {
        window.socket.send(JSON.stringify(getGameState()));
    }
}

function getGameState() {
    return {
        rows: ROWS,
        columns: COLUMNS,
        snake: snake.map((s) => ({ row: s.row, col: s.col })),
        direction: { row: direction.row, col: direction.col },
        food: { row: food.row, col: food.col },
        score: score,
        reward: reward,
        isGameOver: isGameOver,
    };
}

document.addEventListener('keydown', (event) => {
    if (event.key === 'ArrowUp' && direction.row !== 1) direction = { row: -1, col: 0 };
    if (event.key === 'ArrowDown' && direction.row !== -1) direction = { row: 1, col: 0 };
    if (event.key === 'ArrowLeft' && direction.col !== 1) direction = { row: 0, col: -1 };
    if (event.key === 'ArrowRight' && direction.col !== -1) direction = { row: 0, col: 1 };

    if (event.code === 'NumpadSubtract' || event.key === '-') {
        queueSpeedChange(UPDATE_INTERVAL_STEP);
    }

    if (event.code === 'NumpadAdd' || event.key === '=' || event.key === '+') {
        queueSpeedChange(-UPDATE_INTERVAL_STEP);
    }
});

createBoard();
placeFood();
draw();

updateSpeed();
startGameLoop();
