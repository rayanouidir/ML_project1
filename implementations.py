import numpy as np



def compute_mse(y, tx, w):
    e = y - tx @ w
    return e @ e / (2 * len(y))


def compute_mse_gradient(y, tx, w):
    e = y - tx @ w
    return -tx.T @ e / len(y)


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def compute_logistic_loss(y, tx, w):
    z = tx @ w
    return np.mean(np.logaddexp(0.0, z) - y * z)


def compute_logistic_gradient(y, tx, w):
    return tx.T @ (sigmoid(tx @ w) - y) / len(y)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    w = np.array(initial_w, dtype=float)
    for _ in range(max_iters):
        w = w - gamma * compute_mse_gradient(y, tx, w)
    return w, compute_mse(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    w = np.array(initial_w, dtype=float)
    n = len(y)
    for _ in range(max_iters):
        i = np.random.randint(n)                      
        w = w - gamma * compute_mse_gradient(y[i:i + 1], tx[i:i + 1], w)
    return w, compute_mse(y, tx, w)



def least_squares(y, tx):
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w, compute_mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    n, d = tx.shape
    w = np.linalg.solve(tx.T @ tx + 2 * n * lambda_ * np.eye(d), tx.T @ y)
    return w, compute_mse(y, tx, w)



def _logistic_gd(y, tx, lambda_, initial_w, max_iters, gamma):
    w = np.array(initial_w, dtype=float)
    for _ in range(max_iters):
        grad = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad
    return w, compute_logistic_loss(y, tx, w)        


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    return _logistic_gd(y, tx, 0.0, initial_w, max_iters, gamma)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    return _logistic_gd(y, tx, lambda_, initial_w, max_iters, gamma)