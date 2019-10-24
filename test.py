import os
import torch
import copy
from env import Env

import numpy as np
# Globals
Ts, rewards, Qs, best_avg_reward = [], [], [], -1e10

# Test DQN
def test(args, T, dqn, val_mem, specific_title = '', evaluate=False):
  global Ts, rewards, Qs, best_avg_reward
  env = Env(args)
  env.eval()
  Ts.append(T)
  T_rewards, T_Qs = [], []
  
  
  # Test performance over several episodes
  done = True
  for _ in range(args.evaluation_episodes):
    while True:
      if done:
        state, reward_sum, done = env.reset(), 0, False

      action = dqn.act_e_greedy(state)  # Choose an action ε-greedily
      state, reward, done = env.step(action)  # Step
      reward_sum += reward
      if args.render:
        env.render()

      if done:
        T_rewards.append(reward_sum)
        break
  env.close()

  # Test Q-values over validation memory
  for state in val_mem:  # Iterate over valid states
    T_Qs.append(dqn.evaluate_q(state))

  avg_reward, avg_Q = sum(T_rewards) / len(T_rewards), sum(T_Qs) / len(T_Qs)
  if not evaluate:
    # Append to results
    rewards.append(T_rewards)
    Qs.append(T_Qs)


    # Save model parameters if improved
    if avg_reward > best_avg_reward:
      best_avg_reward = avg_reward
    if args.model != None:
      dqn.save(args.base_folder, args.title, args)

  np.save(args.base_folder + 'results/avg_reward_{}_{}_{}.npy'.format(specific_title, T, args.add_appendix), avg_reward)
  if args.add_appendix:
    np.save(args.base_folder + 'results/app_lin_{}_{}_{}.npy'.format(specific_title, T, args.add_appendix), dqn.online_net.app_lin.weight.data)
  # Return average reward and Q-value
  return avg_reward, avg_Q

