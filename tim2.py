import numpy as np
from xcorrelation import x2correlation
from sklearn.metrics import r2_score
import torch
from torch.utils.data import Dataset, DataLoader


class TIM2_Dataset(Dataset):

    def __init__(self, S: list[np.ndarray], E: list[np.ndarray], R: list[np.ndarray], window_length: int):
        for s, r in zip(S, R):
            assert s.shape == r.shape, 'All elements of S and R must have the same shape'
            assert len(s.shape) == 1, 'All elements of S and R must be 1-dimensional'
        self.window = window_length
        self.S = S
        self.R = R
        self.E = E

    def __len__(self):
        return sum([len(s) for s in self.S]) - len(self.S) * (self.window - 1)

    def __getitem__(self, idx):
        lengths = np.cumsum([len(s) - self.window + 1 for s in self.S])
        n = np.searchsorted(lengths, idx, side='right')
        w = self.window
        if n > 0:
            idx -= lengths[n-1]
        e = self.E[n][idx:idx+w]
        x0 = self.S[n][idx:idx+w]
        x = torch.from_numpy(np.concatenate(
            ((x0 * e), (x0 * np.invert(e))), axis=0)
        ).float()
        y = self.R[n][idx+w-1]
        return x, y


class TIM2:

    def __init__(
        self,
        n: int,
        method: str
    ):
        assert method in ('moments', 'sgd')
        self.n = n
        self.method = method
        self.parameters = None

    def fit(self, S: list[np.ndarray], E: list[np.ndarray], R: list[np.ndarray]) -> None:
        for s, e, r in zip(S, E, R):
            assert s.shape == e.shape == r.shape, 'All elements of S and R must have the same shape'
            assert len(s.shape) == 1, 'All elements of S and R must be 1-dimensional'
        n = self.n
        if self.method == 'moments':
            S_c = [s * e for s, e in zip(S, E)]
            S_n = [s * np.invert(e) for s, e in zip(S, E)]
            R = {
                'n': x2correlation(R, S_n, n),
                'c': x2correlation(R, S_c, n)
            }
            C = {
                'cc': np.zeros((n, n), dtype=np.double),
                'nc': np.zeros((n, n), dtype=np.double),
                'nn': np.zeros((n, n), dtype=np.double),
                'cn': np.zeros((n, n), dtype=np.double)
            }
            c = {
                'cc': x2correlation(S_c, S_c, n),
                'nc': x2correlation(S_n, S_c, n),
                'nn': x2correlation(S_n, S_n, n),
                'cn': x2correlation(S_c, S_n, n)
            }
            for e1 in ('c', 'n'):
                for e2 in ('c', 'n'):
                    for m in range(n):
                        e = '{}{}'.format(e1, e2)
                        C[e][m, :] = np.concatenate((c[e][1:m + 1][::-1], c[e][:n - m]), axis=0) if m > 0 else c[e]
            g = np.linalg.solve(
                np.block([[C['cc'], C['cn']], [C['nc'], C['nn']]]),
                np.concatenate((R['c'], R['n']), axis=0).reshape(-1, 1)
            ).flatten()
            self.parameters = {
                'c': g[:n],
                'n': g[n:]
            }
        elif self.method == 'sgd':
            dataset = TIM2_Dataset(S, E, R, self.n)
            dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
            epochs = (.1, .1, .1, .01, .01, .01, .001, .001, .001, .0001, .0001, .0001)
            model = torch.nn.Linear(2*n, 1, bias=False, dtype=torch.float32)
            criterion = torch.nn.MSELoss()
            for lr in epochs:
                optimizer = torch.optim.SGD(model.parameters(), lr=lr)
                for X, y_true in dataloader:
                    y_pred = model(X.float())
                    loss = criterion(y_pred, y_true.float().view(-1, 1))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            self.parameters = model.weight.detach().numpy()[::-1]
            self.parameters = {
                'c': self.parameters.flatten()[:n][::-1],
                'n': self.parameters.flatten()[n:][::-1]
            }

    def score(self, S: list[np.ndarray], E: list[np.ndarray], R: list[np.ndarray]) -> float:
        y_true = []
        y_pred = []
        for s, e, r in zip(S, E, R):
            y_true.extend(list(r))
            y_pred.extend(list(self.predict(s, e)))
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return r2_score(y_true, y_pred)

    def predict(self, s: np.ndarray, e: np.ndarray) -> np.ndarray:
        assert self.parameters is not None, 'The model must be fitted first'
        assert len(s.shape) == len(e.shape) == 1, 's and e must be 1-dimensional'
        n = self.n
        y_pred = np.zeros_like(s, dtype=float)
        for m in range(len(s)):
            if m < n:
                y_pred[m] += np.dot(self.parameters['c'][:m + 1][::-1], s[:m + 1] * e[:m + 1])
                y_pred[m] += np.dot(self.parameters['n'][:m + 1][::-1], s[:m + 1] * np.invert(e[:m + 1]))
            else:
                y_pred[m] += np.dot(self.parameters['c'][::-1], s[m + 1 - n:m + 1] * e[m + 1 - n:m + 1])
                y_pred[m] += np.dot(self.parameters['n'][::-1], s[m + 1 - n:m + 1] * np.invert(e[m + 1 - n:m + 1]))
        return y_pred
