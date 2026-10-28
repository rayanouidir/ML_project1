"""The 6 required ML methods of Project 1 (NumPy only) + small helpers.

Conventions (as required by the project statement):
  - every method returns (w, loss): the LAST weight vector and its loss;
  - the loss of the regularized methods does NOT include the penalty term;
  - all vectors are 1D arrays of shape (X,);
  - for the logistic methods y is in {0, 1}.
"""

import numpy as np


# ----------------------------------------------------------------------------
# Losses and gradients
# ----------------------------------------------------------------------------
def compute_mse(y, tx, w):
    """MSE loss with the 1/2 factor: 1/(2N) * sum (y - tx w)^2."""
    e = y - tx @ w
    return e @ e / (2 * len(y))


def compute_mse_gradient(y, tx, w):
    """Gradient of the MSE loss: -1/N * tx^T (y - tx w)."""
    e = y - tx @ w
    return -tx.T @ e / len(y)


def sigmoid(z):
    """Numerically stable sigmoid."""
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def compute_logistic_loss(y, tx, w):
    """Negative log-likelihood (mean), y in {0,1}: mean(log(1+e^z) - y z)."""
    z = tx @ w
    return np.mean(np.logaddexp(0.0, z) - y * z)


def compute_logistic_gradient(y, tx, w):
    """Gradient of the logistic loss: 1/N * tx^T (sigmoid(tx w) - y)."""
    return tx.T @ (sigmoid(tx @ w) - y) / len(y)


# ----------------------------------------------------------------------------
# 1-2. Linear regression with (stochastic) gradient descent
# ----------------------------------------------------------------------------
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using full-batch gradient descent."""
    w = np.array(initial_w, dtype=float)
    for _ in range(max_iters):
        w = w - gamma * compute_mse_gradient(y, tx, w)
    return w, compute_mse(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using SGD with mini-batch size 1."""
    w = np.array(initial_w, dtype=float)
    n = len(y)
    for _ in range(max_iters):
        i = np.random.randint(n)                       # one random datapoint
        w = w - gamma * compute_mse_gradient(y[i:i + 1], tx[i:i + 1], w)
    return w, compute_mse(y, tx, w)


# ----------------------------------------------------------------------------
# 3-4. Normal equations
# ----------------------------------------------------------------------------
def least_squares(y, tx):
    """Least squares via the normal equations tx^T tx w = tx^T y."""
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w, compute_mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Ridge regression via normal equations (penalty lambda * ||w||^2).
    The returned loss is the plain MSE (no penalty term)."""
    n, d = tx.shape
    w = np.linalg.solve(tx.T @ tx + 2 * n * lambda_ * np.eye(d), tx.T @ y)
    return w, compute_mse(y, tx, w)


# ----------------------------------------------------------------------------
# 5-6. Logistic regression (gradient descent)
# ----------------------------------------------------------------------------
def _logistic_gd(y, tx, lambda_, initial_w, max_iters, gamma):
    w = np.array(initial_w, dtype=float)
    for _ in range(max_iters):
        grad = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad
    return w, compute_logistic_loss(y, tx, w)         # loss without penalty


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent (y in {0,1})."""
    return _logistic_gd(y, tx, 0.0, initial_w, max_iters, gamma)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """L2-regularized logistic regression (penalty lambda * ||w||^2)."""
    return _logistic_gd(y, tx, lambda_, initial_w, max_iters, gamma)