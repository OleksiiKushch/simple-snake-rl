import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


def _model_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = _model_dir()
        _ensure_dir(model_folder_path)

        file_path = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_path)

    def load(self, file_name: str = 'model.pth', map_location='cpu', strict: bool = True) -> bool:
        file_path = os.path.join(_model_dir(), file_name)
        if not os.path.exists(file_path):
            return False

        state_dict = torch.load(file_path, map_location=map_location)
        self.load_state_dict(state_dict, strict=strict)
        return True


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def save_checkpoint(self, file_name: str = 'checkpoint.pth', meta: dict | None = None) -> str:
        model_folder_path = _model_dir()
        _ensure_dir(model_folder_path)

        file_path = os.path.join(model_folder_path, file_name)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr': self.lr,
            'gamma': self.gamma,
            'meta': meta or {},
        }
        torch.save(checkpoint, file_path)
        return file_path

    def load_checkpoint(self, file_name: str = 'checkpoint.pth', map_location='cpu', strict: bool = True) -> dict | None:
        file_path = os.path.join(_model_dir(), file_name)
        if not os.path.exists(file_path):
            return None

        checkpoint = torch.load(file_path, map_location=map_location)
        if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
            return None

        self.model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception:
                pass

        meta = checkpoint.get('meta')
        return meta if isinstance(meta, dict) else {}

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()