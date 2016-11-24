
import numpy as np

def sigmoid(z):
  '''
  Compute sigmoid function
  '''
  return 1 / (1 + np.e ** -z)

def hypothesis(X, theta):
  return sigmoid(X.dot(theta))

def cost_function(X, y, theta, lmbda=0):
  '''
  Computes Logistical cost function, return Cost, Gradient
  '''
  m = X.shape[0]
  h = hypothesis(X, theta)
  part_1 = y * (np.log(h))
  part_2 = (1 - y) * (np.log(1-h))
  reg = lmbda/(2.0*m) * (theta[1:] **  2).sum()
  constant = -1.0/m
  combined = (part_1 + part_2).sum() + reg
  return constant * (combined)

def featureNormalize(X):
  '''
  Returns a normalized version of X where
  the mean value of each feature is 0 and the standard deviation
  is 1. This is often a good preprocessing step to do when
  working with learning algorithms.
  '''
  
  mean = np.mean(X, axis=0)
  std = np.std(X, axis=0, ddof=1)
  return (X - mean)/std

def gradient_descent(X, y, theta, alpha, num_iters, lmbda=0):
  m = y.size
  n = theta.size
  J_history = np.zeros(shape=(num_iters, 1))
  
  constant = alpha*(1.0/m) 
  for i in range(num_iters):
    h = hypothesis(X, theta)
    error = (h - y)
    # r = error * X
    rr = np.dot(X.T, error)
    # rr = constant * r.sum(axis=0)
    # rr = np.reshape(rr,(n,1))
    
    # TODO check reg
    theta = theta - rr
    reg = (lmbda/m) * theta[1:]
    theta[1:] = theta[1:] - reg

    cost = cost_function(X, y, theta)
    J_history[i,0] = cost

  return theta, J_history
  

