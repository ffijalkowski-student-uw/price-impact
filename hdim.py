import numpy as np
from xcorrelation import x2correlation, x3correlation
from sklearn.metrics import r2_score
import torch
from torch.utils.data import Dataset, DataLoader


class HDIM_Dataset(Dataset):

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
        e_end = self.E[n][idx+w-1]
        x0 = self.S[n][idx:idx+w]
        x = torch.from_numpy(np.concatenate(
            (
                e_end * (x0 * e),
                e_end * (x0 * np.invert(e)),
                (not e_end) * (x0 * e),
                (not e_end) * (x0 * np.invert(e))
            ), axis=0)
        ).float()
        y = self.R[n][idx+w-1]
        return x, y


class HDIM:

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
        not_E = [np.invert(e) for e in E]
        if self.method == 'moments':
            S_c = [s * e for s, e in zip(S, E)]
            S_n = [s * np.invert(e) for s, e in zip(S, E)]
            R_c = [r * e for r, e in zip(R, E)]
            R_n = [r * np.invert(e) for r, e in zip(R, E)]
            R_nc = x2correlation(R_c, S_n, n)
            R_cc = x2correlation(R_c, S_c, n)
            C_nnc = x3correlation(S_n, S_n, E, n)
            C_cnc = x3correlation(S_c, S_n, E, n)
            C_ncc = x3correlation(S_n, S_c, E, n)
            C_ccc = x3correlation(S_c, S_c, E, n)
            g = {}
            g_1 = np.linalg.solve(
                np.block([[C_nnc, C_cnc], [C_ncc, C_ccc]]),
                np.concatenate((R_nc, R_cc), axis=0).reshape(-1, 1)
            )
            g['nc'], g['cc'] = g_1[:n].flatten(), g_1[n:].flatten()
            g['nc'][0] = .0
            R_cn = x2correlation(R_n, S_c, n)
            R_nn = x2correlation(R_n, S_n, n)
            C_ccn = x3correlation(S_c, S_c, not_E, n)
            C_ncn = x3correlation(S_n, S_c, not_E, n)
            C_cnn = x3correlation(S_c, S_n, not_E, n)
            C_nnn = x3correlation(S_n, S_n, not_E, n)
            g_2 = np.linalg.solve(
                np.block([[C_ccn, C_ncn], [C_cnn, C_nnn]]),
                np.concatenate((R_cn, R_nn), axis=0).reshape(-1, 1)
            )
            g['cn'], g['nn'] = g_2[:n].flatten(), g_2[n:].flatten()
            g['cn'][0] = .0
            self.parameters = g
        elif self.method == 'sgd':
            dataset = HDIM_Dataset(S, E, R, self.n)
            dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
            epochs = (.1, .1, .1, .01, .01, .01, .001, .001, .001, .0001, .0001, .0001)
            model = torch.nn.Linear(4 * n, 1, bias=False, dtype=torch.float32)
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
                'cc': self.parameters.flatten()[:n][::-1],
                'nc': self.parameters.flatten()[n:2*n][::-1],
                'cn': self.parameters.flatten()[2*n:3*n][::-1],
                'nn': self.parameters.flatten()[3*n:][::-1]
            }
            self.parameters['nc'][0] = .0
            self.parameters['cn'][0] = .0

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
                y_pred[m] += np.dot(self.parameters['cn'][:m + 1][::-1], s[:m + 1] * e[:m + 1] * (not e[m]))
                y_pred[m] += np.dot(self.parameters['nn'][:m + 1][::-1], s[:m + 1] * np.invert(e[:m + 1]) * (not e[m]))
                y_pred[m] += np.dot(self.parameters['nc'][:m + 1][::-1], s[:m + 1] * np.invert(e[:m + 1]) * e[m])
                y_pred[m] += np.dot(self.parameters['cc'][:m + 1][::-1], s[:m + 1] * e[:m + 1] * e[m])
            else:
                y_pred[m] += np.dot(self.parameters['cn'][::-1], s[m + 1 - n:m + 1] * e[m + 1 - n:m + 1] * (not e[m]))
                y_pred[m] += np.dot(self.parameters['nn'][::-1], s[m + 1 - n:m + 1] * np.invert(e[m + 1 - n:m + 1]) * (not e[m]))
                y_pred[m] += np.dot(self.parameters['nc'][::-1], s[m + 1 - n:m + 1] * np.invert(e[m + 1 - n:m + 1]) * e[m])
                y_pred[m] += np.dot(self.parameters['cc'][::-1], s[m + 1 - n:m + 1] * e[m + 1 - n:m + 1] * e[m])
        return y_pred

