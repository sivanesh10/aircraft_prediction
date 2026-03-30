from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


@dataclass
class LSTMConfig:
    """
    Configuration for the LSTM RUL model.

    Defaults are tuned for CPU-only training.
    """

    input_size: int = 21 + 3  # sensors + settings (default FD001)
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2

    batch_size: int = 64
    learning_rate: float = 1e-3
    num_epochs: int = 20
    weight_decay: float = 1e-4

    model_dir: Path = Path("models")
    model_filename: str = "lstm_fd001.pt"

    device: str = "cpu"

    def ensure_model_dir(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.model_filename


class RULSequenceDataset(Dataset):
    """
    Simple Dataset for (sequence, RUL) pairs.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class LSTMRULModel(nn.Module):
    """
    Compact LSTM model for RUL regression.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        return self.fc(last_hidden).squeeze(-1)


class LSTMRULTrainer:
    """
    Helper for training and evaluating the LSTM RUL model on CPU.
    """

    def __init__(self, config: Optional[LSTMConfig] = None) -> None:
        self.config = config or LSTMConfig()
        self.device = torch.device(self.config.device)

        self.model = LSTMRULModel(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        ).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _make_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = RULSequenceDataset(X, y)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=0,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the LSTM model for a fixed number of epochs.
        """
        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                preds = self.model(batch_X)
                loss = self.criterion(preds, batch_y)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * batch_X.size(0)

            epoch_loss /= len(train_loader.dataset)
            history["train_loss"].append(epoch_loss)

            if X_val is not None and y_val is not None:
                val_loss = self.evaluate_loss(X_val, y_val)
                history["val_loss"].append(val_loss)

        return history

    def evaluate_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute MSE loss on a given dataset.
        """
        loader = self._make_loader(X, y, shuffle=False)
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                preds = self.model(batch_X)
                loss = self.criterion(preds, batch_y)
                total_loss += loss.item() * batch_X.size(0)

        return total_loss / len(loader.dataset)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Run inference on a batch of sequences.
        """
        loader = self._make_loader(X, np.zeros((X.shape[0],), dtype=np.float32), shuffle=False)
        self.model.eval()
        preds_list = []

        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(self.device)
                preds = self.model(batch_X)
                preds_list.append(preds.cpu().numpy())

        return np.concatenate(preds_list, axis=0)

    def save(self, path: Optional[Path] = None) -> None:
        """
        Save model weights and config to disk.
        """
        target_path = path or self.config.model_path
        self.config.ensure_model_dir()

        state = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
        }
        torch.save(state, target_path)

    @classmethod
    def load(cls, path: Optional[Path] = None, config: Optional[LSTMConfig] = None) -> "LSTMRULTrainer":
        """
        Load model weights and config from disk.
        """
        cfg = config or LSTMConfig()
        target_path = path or cfg.model_path
        if not target_path.exists():
            raise FileNotFoundError(f"LSTM model not found at {target_path}")

        # PyTorch 2.6+ defaults weights_only=True, but our checkpoint includes
        # a config object. We explicitly allow full checkpoint loading here.
        state = torch.load(target_path, map_location=cfg.device, weights_only=False)
        loaded_config: LSTMConfig = state.get("config", cfg)

        instance = cls(loaded_config)
        instance.model.load_state_dict(state["model_state_dict"])
        instance.model.to(instance.device)
        return instance


__all__ = ["LSTMConfig", "RULSequenceDataset", "LSTMRULModel", "LSTMRULTrainer"]

