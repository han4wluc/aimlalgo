
from collections import Hashable
import numpy as np

class QAgent(object):
  """Table-based Q Learning Agent
  A table-based Q learning agent. Updates table with canonical Bellman iteration method.
  Random exploration using epsilon or soft-probability strategy.
  Train the agent with QAgent.observer_and_act().
  Use the current observation and the reward from last action.
  Reset agent state with QAgent.reset(), set forget_table = True to drop the leraned table.
  """
  def __init__(
    self,
    actions=None,
    alpha=1.0,
    gamma=0.02,
    epsilon=0.02,
    explore_strategy='epsilon',
    verbose=0,
    **kwargs):
    """
    Parameters
    ----------
    actions: Legitimate actions. (list or tuple)
    alpha: Learning rate. The weight for update q value. (float, between (0,1])
    gamma: Reward discount factor. (float, between [0, 1))
    epsilon: exploration rate for epsilon-greedy exploration. (float, [0,1))
    explore_strategy: 'eplison' or 'soft_probability'
    verbose: verbosity level.
    kwargs:
    Returns
    ----------
    """
    super(QAgent, self).__init__(**kwargs)

    # static attributes
    if not actions:
      raise ValueError("Passed in None action list")

    self.ACTIONS = actions # legitimate actions
    self.ALPHA = alpha     # learning rate
    self.GAMMA = gamma     # discount factor
    self.EPSILON = epsilon # exploration probability for 'epsilon-greedy' strategy
    self.DEFAULT_QVAL = 0  # default initial value for Q table entries
    self.EXPLORE = explore_strategy
    self.verbose = verbose

    # dynamic attributes
    self.last_state = None
    self.last_action = None
    self.q_table = {}    


  def observe_and_act(self, observation, last_reward=None):
    """A single reinforcement learning step
    Pass in the current observation and reward from last action.
    First try to internalize the observation as an
    agent state, then reinforce the agent with current experience,
    finally perform action based on current experience.
    # The internalization of observation is done by the transition_() method.
    This method is expected to be materialized in the child classes.
    If not, the state will simply be the current observation.
    """
    
    state = observation

    # Improve agent given current state and last_reward
    update_result = self.reinforce_(state=state, last_reward=last_reward)

    # 
    # 
    # 

  def reinforce_(self, state, last_reward):
    """Improve agent based on current exprience (last_state, last_action, last_reward, state)
    """
    last_state = self.last_state
    last_action = self.last_action
    if last_state is None or state is None or last_reward is None:
      update_result = None
    else:
      update_result = self.update_table_(last_state, last_action, last_reward, state)
    return update_result

  def update_table_(self, last_state, last_action, reward, current_state):
    """Update Q table using Bellman iteration
    """
    best_qval = max(self.lookup_table_(current_state))
    # if not isinstance(last_state, Hashable):
    #   last_state = tuple(last_state.rave()) # passed in numpy array
    delta_q = reward + self.GAMMA * best_qval
    last_state_action = (last_state, last_action)
    if last_state_action in self.q_table:
      a = self.ALPHA * delta_q
      alpha = 1 - self.ALPHA
      qvalue = a + self.q_table[last_state_action] * alpha
    else:
      qvalue = self.DEFAULT_QVAL
    self.q_table[last_state_action] = qvalue
    return None

  def lookup_table_(self, state):
    """return the q values of all ACTIONS at a given state
    """
    # if not isinstance(state, Hashable):
      # state = tuple(state.ravel())
    qvalues = []
    for a in self.ACTIONS:
      if(state, a) in self.q_table:
        qvalue = self.q_table[(state, a)]
        qvalues.append(qvalue)
      else:
        qvalues.append(self.DEFAULT_QVAL)
    return qvalues







