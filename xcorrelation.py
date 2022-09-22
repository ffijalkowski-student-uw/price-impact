import numpy as np
from scipy import fft


def x2correlation(
    F: list[np.ndarray],
    G: list[np.ndarray],
    size: int = 0
) -> np.ndarray:
    """
    Given two lists of time series F, G returns an array such that:
        xcorrelation(F, G)[l] = E[f[t] * g[t-l]]
    where f is an element of F, g is an element of G
    :param list[np.ndarray] f: time series
    :param list[np.ndarray] g: time series
    :param int size: maximal lag e.g. length of xcorrelation(f, g)
    :return np.array:
    """
    for f, g in zip(F, G):
        assert f.shape == g.shape, 'All elements of F and G must have the same shape'
        assert len(f.shape) == 1, 'All elements of F and G must be 1-dimensional'
    m = min([len(f) for f in F])
    size = min(m, size) if size > 0 else m
    c = np.zeros(size, dtype=np.double)
    for n in range(size):
        normalization = 0
        for f, g in zip(F, G):
            if n == 0:
                c[n] += np.sum(f * g)
                normalization += len(f)
            else:
                c[n] += np.sum(f[n:] * g[:-n])
                normalization += len(f[n:])
        c[n] = c[n] / normalization
    return c


def x3correlation(
    F: list[np.ndarray],
    G: list[np.ndarray],
    H: list[np.ndarray],
    size: int = 0
):
    """
    Given three lists of time series F, G, H returns an array such that:
        xcorrelation(F, G)[l, k] = E[f[t-l] * g[t-k] * h[t]]
    where f is an element of F, g is an element of G and h is an element of H
    :param list[np.ndarray] f: time series
    :param list[np.ndarray] g: time series
    :param int size: maximal lag e.g. length of xcorrelation(f, g)
    :return np.array:
    """
    for f, g, h in zip(F, G, H):
        assert f.shape == g.shape == h.shape, 'All elements of F and G must have the same shape'
        assert len(f.shape) == 1, 'All elements of F and G must be 1-dimensional'
    m = min([len(f) for f in F])
    size = min(m, size) if size > 0 else m
    c = np.zeros((size, size), dtype=np.double)
    for n in range(size):
        for m in range(size):
            normalization = 0
            if n == 0 and m == 0:
                c[n, m] += np.sum(f * g * h)
                normalization += len(f)
            else:
                if n > m:
                    c[n, m] += np.sum(f[n:] * g[m:m-n] * h[:-n])
                    normalization += len(f) - n
                elif n == m:
                    c[n, m] += np.sum(f[n:] * g[n:] * h[:-n])
                    normalization += len(f) - n
                else:
                    c[n, m] += np.sum(f[n:n-m] * g[m:] * h[:-m])
                    normalization += len(f) - m
            c[n, m] = c[n, m] / normalization
    return c
