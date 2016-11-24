
import numpy as np

def hypothesis(X,theta):
  '''
  Hypothesis
  '''
  return X.dot(theta)

def compute_cost(X, y, theta, lmbda=0):
  '''
  Compute cost function for linear regression
  '''
  m = y.size
  predictions = hypothesis(X,theta)
  # reg = lmbda * (theta[1:] **  2).sum()
  reg = lmbda * (theta **  2).sum()
  sq_errors = (predictions - y) ** 2
  constant = 1.0 / (2 * m)
  J = constant * (sq_errors.sum() + reg)
  return J

def compute_gradient(X, y, theta, lmbda=0):
  m = X.shape[0]
  h = hypothesis(X, theta)
  errors = np.dot(X.T, h-y)
  return (1.0 / m) * errors


def gradient_descent(X, y, theta, alpha, num_iters, lmbda=0):
  '''
  Performs gradient descent to learn theta 
  by taking num_items gradient steps with 
  learning rate alpha
  '''
  n = theta.size
  m = y.size

  J_history = np.zeros(shape=(num_iters, 1))

  for i in range(num_iters):
    predictions = X.dot(theta)
    errors = X.T.dot(predictions - y)
    constant = alpha * (1.0 / m)
    errors = errors * constant
    reg_constant = 1 - (alpha * (lmbda/m))
    theta = ((theta*reg_constant) - errors)
    J_history[i, 0] = compute_cost(X, y, theta, lmbda)

  return theta, J_history

