
from __future__ import division
import math

def _calc_size(low, high, bucket):
  difference = high - low
  width = difference / bucket
  return width

def _get_discrete_quantity(low, high, value, n_of_parts, width):
  if value <= low:
    return 0
  if value >= high:
    return n_of_parts - 1
  discrete_value = int(math.floor(value - low) / width)
  return discrete_value

class Quantizer(object):
  """Observation Quantizer
  This class is used for quantizing the observations into discrete states
  to be used for Qtable.QAgent
  """
  def __init__(self, low, high, buckets):
    """
    Parameters
    ----------
    low: list/tuble of lowerst possible observation values
    high: list/tuble of highest possible observation values
    bucket: number of buckets to quantize the dimension into (list or tuple)
    kwargs: extra arguments passed (not needed)
    ----------
    """
    # static attributes
    self.low = low
    self.high = high
    self.buckets = buckets
    self.dim = len(low)
    self.width = map(_calc_size, low, high, buckets)

    def quantize(self, observation):
      """Quantize the observation
      """
    quantized_obs = []

  def quantize(self, observations):
    """Quantize the observation
    """

    discrete_observations = []
    for i in range(len(observations)):
      observation = observations[i]
      values = []
      for j in range(self.dim):
        values.append(
          _get_discrete_quantity(
            self.low[j],
            self.high[j],
            observation[j],
            self.buckets[j],
            self.width[j]
          )
        )
      discrete_observations.append(values)


    return discrete_observations

