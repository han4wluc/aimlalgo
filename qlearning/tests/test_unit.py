import unittest
from qlearning.Quantizer import Quantizer, _calc_size, _get_discrete_quantity
from qlearning.QAgent import QAgent

class TestQuantizer(unittest.TestCase):

  def test_constructor(self):

    quantizer = Quantizer(low=[0],high=[3],buckets=[1])
    self.assertEqual(quantizer.width, [3])

  def test_calc_size(self):
    width = _calc_size(1,4,2)
    self.assertEqual(width, 1.5)

    width = _calc_size(-2,2,4)
    self.assertEqual(width, 1)

  # def test_get_discrete_quantity(self):
  #   res = _get_discrete_quantity(low=0,high=10,value=-1,n_of_parts=2,width=2)
  #   self.assertEqual(res, 0)

  #   res = _get_discrete_quantity(low=0,high=10,value=0,n_of_parts=2,width=2)
  #   self.assertEqual(res, 0)

  #   res = _get_discrete_quantity(low=0,high=10,value=10,n_of_parts=2,width=2)
  #   self.assertEqual(res, 1)

  #   res = _get_discrete_quantity(low=0,high=10,value=11,n_of_parts=2,width=2)
  #   self.assertEqual(res, 1)

  #   res = _get_discrete_quantity(low=0,high=10,value=9.9,n_of_parts=3,width=3.33)
  #   self.assertEqual(res, 2)


  def test_quantize(self):
    quantizer = Quantizer(low=[0],high=[10],buckets=[2])

    res = quantizer.quantize([7])
    self.assertEqual(res, [1])

    quantizer = Quantizer(low=[0,20],high=[10,60],buckets=[2,4])

    res = quantizer.quantize([7,2])
    self.assertEqual(res, [1,0])



class TestQtable(unittest.TestCase):

  def test_constructor(self):

    with self.assertRaises(Exception) as context:
        qagent = QAgent()

    self.assertEqual('Passed in None action list', context.exception[0])


  def test_lool_up_table(self):

    qagent = QAgent(actions=[1,2,3])
    qvalues = qagent.lookup_table_(11)
    self.assertEqual(qvalues, [0,0,0])

    qagent.q_table = {
      ((0,0), 1): 2.2,
      ((0,0), 2): 3.2,
    }
    qvalues = qagent.lookup_table_((0,0))
    self.assertEqual(qvalues, [2.2,3.2,0])


  def test_update_table_(self):

    def mock(_):
      return [1,2]

    qagent = QAgent(actions=[1,2,3])

    # mocks
    qagent.lookup_table_ = mock
    qagent.q_table = {}

    last_state = (0,0)
    last_action = 1
    reward = 2
    current_state = (0,1)
    qagent.update_table_(last_state=last_state,last_action=last_action,reward=reward,current_state=current_state)
    self.assertEqual(qagent.q_table, {
      ((0,0),1): 0
    })

    qagent.update_table_(last_state=last_state,last_action=last_action,reward=reward,current_state=current_state)
    self.assertEqual(qagent.q_table, {
      ((0,0),1): 2.04
    })

  def test_reinforce_(self):
    qagent = QAgent(actions=[1,2,3])

    # mocks
    qagent.last_state = (0,0)
    qagent.last_action = 1
    def update_table_(_,__,___,____,):
      return 9999
    qagent.update_table_ = update_table_

    res = qagent.reinforce_(state=(0,1),last_reward=3)
    self.assertEqual(res, 9999)

    res = qagent.reinforce_(state=(0,1),last_reward=None)
    self.assertEqual(res, None)

  def test_act_(self):
    qagent = QAgent(actions=[1,2,3])

    action = qagent.act_(None)
    self.assertTrue(action in [1,2,3])


    action = qagent.act_((1,1))
    self.assertTrue(action in [1,2,3])


if __name__ == '__main__':
    unittest.main()
