
import numpy as np

def hypothesis(X,theta):
  '''
  Hypothesis
  '''
  return X.dot(theta)

def compute_cost(X, y, theta):
  '''
  Compute cost function for linear regression
  '''
  m = y.size
  predictions = hypothesis(X,theta)
  sq_errors = (predictions - y) ** 2
  constant = 1.0 / (2 * m)
  J = constant * sq_errors.sum()
  return J

def gradient_descent(X, y, theta, alpha, num_iters):
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
    errors = (predictions - y) * X
    errors = errors.sum(axis=0)
    constant = alpha * (1.0 / m)
    errors = errors * constant
    errors = np.reshape(errors, (n, 1))
    theta = (theta - errors)
    J_history[i, 0] = compute_cost(X, y, theta)

  return theta, J_history







