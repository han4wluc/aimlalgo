import unittest
import numpy as np

from logistic_regression.logistic_regression import sigmoid, cost_function, gradient_descent, hypothesis, featureNormalize

class TestLogisticRegression(unittest.TestCase):

  def test_sigmoid(self):
    np.testing.assert_almost_equal(sigmoid(1200000), 1)
    np.testing.assert_almost_equal(sigmoid(-250), 0)
    np.testing.assert_almost_equal(sigmoid(0), 0.5)

  def testTwo(self):
    z = np.array([4, 5, 6])
    expected = np.array([0.9820, 0.9933, 0.9975])
    np.testing.assert_almost_equal(sigmoid(z), expected, decimal=4)
    
  def testThree(self):
    z = np.array([[8, 1, 6],
                  [3, 5, 7],
                  [4, 9, 2],])
    expected = np.array([[0.9997, 0.7311, 0.9975],
                         [0.9526, 0.9933, 0.9991],
                         [0.9820, 0.9999, 0.8808],])
    np.testing.assert_almost_equal(sigmoid(z), expected, decimal=4)

class TestCostFunction(unittest.TestCase):
  def test_cost_function(self):
    X = np.array([[8, 1, 6],
                  [3, 5, 7],
                  [4, 9, 2],
                  [8, 1, 6],
                  [3, 5, 7],
                  [4, 9, 2]])
    y = np.array([[1],
                  [0],
                  [1],
                  [0],
                  [1],
                  [0]])
    theta = np.array([[0],
                      [1],
                      [0]])
    expectedJ     = 2.6067
    expectedTheta = np.array([[1.7760],
                              [2.3988],
                              [1.9464]])
    actualJ = cost_function(X, y, theta)
    # [actualJ, actualTheta] = LR.costFunction(X, y, theta)
    np.testing.assert_almost_equal(actualJ, expectedJ, decimal=4)
    # np.testing.assert_almost_equal(actualTheta, expectedTheta, decimal=4)

  def test_one(self):
    X = np.array([[1, 2, 3],
                  [1, 2, 3],
                  [1, 2, 3],
                  [2, 2, 1],
                  [3, 2, 1],
                  [15, 14, 13],
                  [15, 14, 14],
                  [13, 13, 15],
                  [12, 4, 5],
                  [13, 13, 13],])
    X = featureNormalize(X)

    X_test = X[-2:]
    X = X[:-2]
    y = np.array([[0],
                  [0],
                  [0],
                  [0],
                  [0],
                  [1],
                  [1],
                  [1]])

    theta = np.array([[0.1],
                      [0.1],
                      [0.1]])
    alpha = 0.01
    num_iters = 1000
    theta, cost_hist = gradient_descent(X, y, theta, alpha, num_iters)
    # print 'theta', theta
    # print 'cost_hist', cost_hist
    predicted =  hypothesis(X_test, theta)
    np.testing.assert_equal(predicted[0][0] < 0.5, True)
    np.testing.assert_equal(predicted[1][0] > 0.5, True)

  def test_one_vs_all(self):
    X = np.array([[1, 2, 3],
                  [1, 2, 3],
                  [1, 2, 3],
                  [2, 2, 1],
                  [3, 2, 1],
                  [45, 14, 13],
                  [45, 14, 14],
                  [43, 13, 15],
                  [43, 13, 15],
                  [43, 13, 15],
                  [101, 102, 103],
                  [101, 98, 103],
                  [101, 102, 99],
                  [101, 100, 103],
                  [1, 2, 1],
                  [43, 13, 15],
                  [102, 100, 99]])
    X = featureNormalize(X)

    X_test = X[-3:]
    X = X[:-3]
    y = np.array([[0],
                  [0],
                  [0],
                  [0],
                  [0],
                  [1],
                  [1],
                  [1],
                  [1],
                  [1],
                  [2],
                  [2],
                  [2],
                  [2]])
    m = y.size
    y1s = y == 0
    y2s = y == 1
    y3s = y == 2
    y1 = np.zeros(shape=(m,1))
    y1[y1s] = 1
    y2 = np.zeros(shape=(m,1))
    y2[y2s] = 1
    y3 = np.zeros(shape=(m,1))
    y3[y3s] = 1

    alpha = 0.01
    num_iters = 1000
    theta = np.array([[0.1],
                  [0.1],
                  [0.1]])

    theta1, _ = gradient_descent(X, y1, theta, alpha, num_iters)
    theta2, _ = gradient_descent(X, y2, theta, alpha, num_iters)
    theta3, _ = gradient_descent(X, y3, theta, alpha, num_iters)

    h1 = hypothesis(X_test, theta1)
    h2 = hypothesis(X_test, theta2)
    h3 = hypothesis(X_test, theta3)

    h = np.concatenate((h1,h2,h3),axis=1)
    res = np.argmax(h, axis=0)
    np.testing.assert_equal(res, np.array([0,1,2]))


if __name__ == '__main__':
    unittest.main()

