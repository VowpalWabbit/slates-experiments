import pandas as pd
import numpy as np
from vowpalwabbit import pyvw

def create_slates_example(vw, shared, action_sets, outcome=None, debug=False):
  def add(container, ex_string):
    if(debug):
      print(ex_string)
    else:
      container.append(vw.example(ex_string, labelType=pyvw.pylibvw.vw.lConditionalContextualBandit))

  examples = []
  add(examples, "ccb shared |User {}".format(shared))
  counter = 0
  actions = []
  slots = []

  slot_index = 0
  slot_thresh = 0
  for action_set in action_sets:
    ids = []
    for action in action_set:
      add(actions, "ccb action |Action {}".format(action))
      ids.append(str(counter))
      counter += 1

    if(outcome is not None):
      chosen,cost,prob = outcome[slot_index]
      # Transform back to original space
      chosen += slot_thresh
      slot_str = "ccb slot {}:{}:{} {} |Slot slot_id={} constant".format(chosen, cost, prob, ",".join(ids), slot_index)
    else:
      slot_str = "ccb slot {} |Slot slot_id={} constant".format(",".join(ids), slot_index)
    add(slots, slot_str)

    slot_index += 1
    slot_thresh = counter
  examples.extend(actions)
  examples.extend(slots)
  return examples

def create_cb_example(vw, shared, actions, outcome=None, debug=False):
  def add(container, ex_string):
    if(debug):
      print(ex_string)
    else:
      container.append(vw.example(ex_string, labelType=pyvw.pylibvw.vw.lContextualBandit))

  examples = []
  add(examples, "shared |User {}".format(shared))

  if(outcome is not None):
    chosen,cost,prob = outcome
  else:
    chosen = -1

  for i, action in enumerate(actions):
    if i == chosen:
      action_str = "{}:{}:{} |Action {}".format(chosen, cost, prob, action)
    else:
      action_str = "|Action {}".format(action)
    add(examples, action_str)

  return examples

def slate_pred_conv(prediction):
  size_so_far = 0
  for action_score in prediction:
    for i, a_s in enumerate(action_score):
      a,s = a_s
      action_score[i] = (a - size_so_far, s)
    size_so_far += len(action_score)
  return prediction

def combine_float_actions(x_actions, y_actions, z_actions):
  all_string_actions = []
  all_actions = []
  for x_action in x_actions:
    for y_action in y_actions:
      for z_action in z_actions:
        all_string_actions.append("x={} y={} z={}".format(x_action, y_action, z_action))
        all_actions.append((x_action, y_action, z_action))
  return all_string_actions, all_actions

def combine_float_actions_categorical(x_actions, y_actions, z_actions):
  all_string_actions = []
  all_actions = []
  for x_action in x_actions:
    for y_action in y_actions:
      for z_action in z_actions:
        all_string_actions.append("x={},y={},z={}".format(x_action, y_action, z_action))
        all_actions.append((x_action, y_action, z_action))
  return all_string_actions, all_actions

def create_cb_example(vw, shared, actions, outcome=None, debug=False):
  def add(container, ex_string):
    if(debug):
      print(ex_string)
    else:
      container.append(vw.example(ex_string, labelType=pyvw.pylibvw.vw.lContextualBandit))

  examples = []
  add(examples, "shared |User {}".format(shared))

  if(outcome is not None):
    chosen,cost,prob = outcome
  else:
    chosen = -1

  for i, action in enumerate(actions):
    if i == chosen:
      action_str = "{}:{}:{} |Action {}".format(chosen, cost, prob, action)
    else:
      action_str = "|Action {}".format(action)
    add(examples, action_str)

  return examples

def slate_pred_conv(prediction):
  size_so_far = 0
  for action_score in prediction:
    for i, a_s in enumerate(action_score):
      a,s = a_s
      action_score[i] = (a - size_so_far, s)
    size_so_far += len(action_score)
  return prediction

def sample(pred):
    """Returns the sampled prediction, the slot chosen, and the action chosen in that slot"""
    outer_sample = np.random.choice(len(pred))
    unzipped = list(zip(*pred[outer_sample]))
    # Normalize
    unzipped[1] = [float(i)/sum(unzipped[1]) for i in unzipped[1]]
    chosen = np.random.choice(unzipped[0], p=unzipped[1])

#     print("Sampling from slot: {}".format(outer_sample))
#     print("Probs: {}".format(unzipped[1]))
#     print("Action chosen: {}".format(chosen))
    if chosen != 0:
        pred[outer_sample][0], pred[outer_sample][chosen] = pred[outer_sample][chosen], pred[outer_sample][0]
    return pred, outer_sample, chosen

# def sample_cb(pred):
#     outer_sample = np.random.choice(len(pred))
#     unzipped = list(zip(*pred[outer_sample]))
#     # Normalize
#     unzipped[1] = [float(i)/sum(unzipped[1]) for i in unzipped[1]]
#     chosen = np.random.choice(unzipped[0], p=unzipped[1])

# #     print("Sampling from slot: {}".format(outer_sample))
# #     print("Probs: {}".format(unzipped[1]))
# #     print("Action chosen: {}".format(chosen))
#     if chosen != 0:
#         pred[outer_sample][0], pred[outer_sample][chosen] = pred[outer_sample][chosen], pred[outer_sample][0]
#     return pred, outer_sample, chosen
