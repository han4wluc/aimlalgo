
import numpy as np

def sigmoid(z):
  '''
  Compute sigmoid function
  '''
  return 1 / (1 + np.e ** -z)

def hypothesis(X, theta):
  return sigmoid(X.dot(theta))

def cost_function(X, y, theta):
  '''
  Computes Logistical cost function, return Cost, Gradient
  '''
  m = X.shape[0]
  h = hypothesis(X, theta)
  part_1 = y * (np.log(h))
  part_2 = (1 - y) * (np.log(1-h))
  constant = -1.0/m
  combined = (part_1 + part_2).sum()
  return constant * combined

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

def gradient_descent(X, y, theta, alpha, num_iters):
  m = y.size
  n = theta.size
  J_history = np.zeros(shape=(num_iters, 1))
  
  constant = alpha*(1.0/m) 
  for i in range(num_iters):
    h = hypothesis(X, theta)
    # print 'h', h
    # print 'y', y
    error = (h - y)
    # print 'error', error
    r = error * X
    # print 'r', r
    rr = constant * r.sum(axis=0)
    # print 'rrsum', rr
    rr = np.reshape(rr,(n,1))
    # print 'rr', rr
    theta = theta - rr
    cost = cost_function(X, y, theta)
    J_history[i,0] = cost

  return theta, J_history
  

