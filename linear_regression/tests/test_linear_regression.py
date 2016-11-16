import unittest
import numpy as np

from linear_regression.linear_regression import hypothesis, compute_cost, gradient_descent

class TestLinearRegression(unittest.TestCase):

  def test_hypothesis(self):
    X = np.array(
      [[1, 2, 3],
       [1, 3, 4],
       [1, 4, 5],
       [1, 5, 6]]
    )
    expected = np.array(
      [ [8], [11], [14], [17]]
    )
    theta = np.array([[0],[1],[2]])
    actual = hypothesis(X,theta)
    np.testing.assert_almost_equal(actual, expected)

  def test_compute_cost(self):
    X = np.array([[1, 2],
                  [1, 3],
                  [1, 4],
                  [1, 5]])
    y = np.array([[ 7.],
                  [ 6.], 
                  [ 5.], 
                  [ 4.]]);
    theta = np.array([[0.1],
                      [0.2]])
    expected = 11.9450
    actual = compute_cost(X, y, theta)
    np.testing.assert_almost_equal(actual, expected)

  def test_gradient_descent(self):
    X = np.array([[1, 5],
                  [1, 2],
                  [1, 4],
                  [1, 5]])
    y = np.array([[1],
                  [6],
                  [4],
                  [2]])
    theta = np.array([[0],[0]])
    alpha = 0.01;
    numOfIter = 1000;
    expectedTheta = np.array([[ 5.2148],
                              [-0.5733]])
    # expectedJHist[0] = 0.85426;
    [actualTheta, actualJHist] = gradient_descent(X, y, theta, alpha, numOfIter)
    # print actualJHist
    np.testing.assert_almost_equal(actualTheta, expectedTheta, decimal=4);



if __name__ == '__main__':
    unittest.main()



