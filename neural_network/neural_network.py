
import numpy as np

# X = np.array([[1,4,5,6],
#               [1,3,6,4],             
#               [1,3,6,4]
#              ])
# y = np.array([[0.9],
#               [0.8],
#               [0.7]])
# theta1 = np.array([[1,1],
#                    [10,40],
#                    [20,50],
#                    [30,60]])
# theta2 = np.array([[1],
#                    [70],
#                    [80],])
# thetas = [theta1, theta2]


def sigmoid(z):
  '''
  Compute sigmoid function
  '''
  return 1 / (1 + np.e ** -z)

def sigmoid_gradient(z):
  return sigmoid(z).dot(1-sigmoid(z))

def regularization(lmbda, m, thetas):
  reg = 0
  for theta in thetas:
    reg += (theta[1:] ** 2).sum()

  constant = lmbda / (2.0*m) 
  return constant * reg

def forward_propagation(X, thetas):
  a = X
  coeff = X[:,[0]]
  l = len(thetas)
  for i, theta in enumerate(thetas):
    a = a.dot(theta)
    a = sigmoid(a)
    if i != l-1:
      a = np.concatenate((coeff, a), axis=1)
  return a

def compute_cost(X, y, thetas, lmbda=0):
  m = X.shape[0]
  n = X.shape[1]
  a = forward_propagation(X, thetas)
  reg = regularization(lmbda, m, thetas)
  return (-1.0/m) * a.sum() + reg
