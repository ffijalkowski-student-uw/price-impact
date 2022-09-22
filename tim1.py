import numpy as np
from xcorrelation import x2correlation
from sklearn.metrics import r2_score
import torch
from torch.utils.data import Dataset, DataLoader


class TIM1_Dataset(Dataset):

    def __init__(self, S: list[np.ndarray], R: list[np.ndarray], window_length: int):
        for s, r in zip(S, R):
            assert s.shape == r.shape, 'All elements of S and R must have the same shape'
            assert len(s.shape) == 1, 'All elements of S and R must be 1-dimensional'
        self.window = window_length
        self.S = S
        self.R = R

    def __len__(self):
        return sum([len(s) for s in self.S]) - len(self.S) * (self.window - 1)

    def __getitem__(self, idx):
        lengths = np.cumsum([len(s) - self.window + 1 for s in self.S])
        n = np.searchsorted(lengths, idx, side='right')
        w = self.window
        if n > 0:
            idx -= lengths[n-1]
        x = torch.from_numpy(self.S[n][idx:idx+w]).float()
        y = self.R[n][idx+w-1]
        return x, y


class TIM1:

    def __init__(
        self,
        n: int,
        method: str
    ):
        assert method in ('moments', 'sgd')
        self.n = n
        self.method = method
        self.parameters = None

    def fit(self, S: list[np.ndarray], R: list[np.ndarray]) -> None:
        for s, r in zip(S, R):
            assert s.shape == r.shape, 'All elements of S and R must have the same shape'
            assert len(s.shape) == 1, 'All elements of S and R must be 1-dimensional'
        n = self.n
        if self.method == 'moments':
            c = x2correlation(S, S, n)
            C = np.zeros((n, n), dtype=np.double)
            for m in range(n):
                C[m, :] = np.concatenate((c[1:m+1][::-1], c[:n-m]), axis=0) if m > 0 else c
            R = x2correlation(R, S, n)
            g = np.linalg.solve(C, R.reshape(-1, 1)).flatten()
            self.parameters = g
        elif self.method == 'sgd':
            dataset = TIM1_Dataset(S, R, self.n)
            dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
            epochs = (.1, .1, .1, .01, .01, .01, .001, .001, .001, .0001, .0001, .0001)
            model = torch.nn.Linear(n, 1, bias=False, dtype=torch.float32)
            criterion = torch.nn.MSELoss()
            for lr in epochs:
                optimizer = torch.optim.SGD(model.parameters(), lr=lr)
                for X, y_true in dataloader:
                    y_pred = model(X.float())
                    loss = criterion(y_pred, y_true.float().view(-1, 1))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            self.parameters = model.weight.detach().numpy().flatten()[::-1]

    def score(self, S: list[np.ndarray], R: list[np.ndarray]) -> float:
        y_true = []
        y_pred = []
        for s, r in zip(S, R):
            y_true.extend(list(r))
            y_pred.extend(list(self.predict(s)))
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return r2_score(y_true, y_pred)

    def predict(self, s: np.ndarray) -> np.ndarray:
        assert self.parameters is not None, 'The model must be fitted first'
        assert len(s.shape) == 1, 's must be 1-dimensional'
        n = self.n
        y_pred = np.zeros_like(s, dtype=float)
        for m in range(len(s)):
            if m < n:
                y_pred[m] = np.dot(self.parameters[:m+1][::-1], s[:m+1])
            else:
                y_pred[m] = np.dot(self.parameters[::-1], s[m+1-n:m+1])
        return y_pred
