import argparse
from datetime import datetime
from tensorboardX import SummaryWriter
import random
import torch
import os
from agent import Agent
from env import Env
from memory import ReplayMemory
from test import test
from tqdm import tqdm
import copy

parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--game', type=str, default='space_invaders', help='ATARI game')
parser.add_argument('--T-max', type=int, default=int(50e6), metavar='STEPS', help='Number of training steps (4x number of frames)')
parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH', help='Max episode length (0 to disable)')
parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')
parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω', help='Prioritised experience replay exponent (originally denoted α)')
parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β', help='Initial prioritised experience replay importance sampling weight')
parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--target-update', type=int, default=int(32e3), metavar='τ', help='Number of steps after which to update target network')
parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--lr', type=float, default=0.0000625, metavar='η', help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--learn-start', type=int, default=int(80e3), metavar='STEPS', help='Number of steps before starting training')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=10000, metavar='STEPS', help='Number of training steps between evaluations')
parser.add_argument('--evaluation-episodes', type=int, default=3, metavar='N', help='Number of evaluation episodes to average over')
parser.add_argument('--evaluation-size', type=int, default=500, metavar='N', help='Number of transitions to use for validating Q')
parser.add_argument('--log-interval', type=int, default=25000, metavar='STEPS', help='Number of training steps between logging status')
parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
parser.add_argument('--add_appendix', action='store_true', help='Add appendix')
parser.add_argument('--compress', action='store_true', help='Compress model')
parser.add_argument('--title', type=str, default='default', help='title of the run')
parser.add_argument('--appendix_type', type=str, default='', help='type of appendix to add')
parser.add_argument('--gpu', type=int, default=0, help='gpu_id')
parser.add_argument('--type_of_norm', type=str, default='L0', help='L0 or L1')
parser.add_argument('--multiplier', type=float, default=-0.01, help='multiplier')
parser.add_argument('--appendix_model', type=str, metavar='PARAMS', help='Pretrained model (state dict) for appendix')
parser.add_argument('--freeze_merge', action='store_true', help='Freeze merge layer')
parser.add_argument('--merge_type', type=str, default='addition', metavar='PARAMS', help='type of appendix')
parser.add_argument('--just_appendix', action='store_true', help='remove original network')
parser.add_argument('--base_folder', type=str, default='/storage/michaeld/', metavar='PARAMS', help='base folder for files')

from torchvision import models, transforms
# Setup
args = parser.parse_args()
print(' ' * 26 + 'Options')
for k, v in vars(args).items():
  print(' ' * 26 + k + ': ' + str(v))
random.seed(args.seed)
torch.manual_seed(random.randint(1, 10000))
if torch.cuda.is_available() and not args.disable_cuda:
  args.device = torch.device('cuda', args.gpu)
  torch.cuda.manual_seed(random.randint(1, 10000))
  torch.backends.cudnn.enabled = False  # Disable nondeterministic ops (not sure if critical but better safe than sorry)
else:
  args.device = torch.device('cpu')


# Simple ISO 8601 timestamped logger
def log(s):
  print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)


# Environment
env = Env(args)
env.train()
action_space = env.action_space()


# Agent
dqn = Agent(args, env)
# print(dqn.online_net.conv1.weight)

mem = ReplayMemory(args, args.memory_capacity)
priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)

current_time = str(datetime.now().timestamp())
log_dir = args.base_folder + 'logs/' + current_time
summary_writer = SummaryWriter(log_dir='{}/tensorboard/{}_{}/'.format(args.base_folder, args.title, datetime.now().strftime('%Y-%m-%d_%H:%M:%S')))

# Construct validation memory
val_mem = ReplayMemory(args, args.evaluation_size)
T, done = 0, True
while T < args.evaluation_size:
  if done:
    state, done = env.reset(), False
  next_state, _, done = env.step(random.randint(0, action_space - 1))
  val_mem.append(state, None, None, done)
  state = next_state
  T += 1

if args.evaluate:
  dqn.eval()  # Set DQN (online network) to evaluation mode
  avg_reward, avg_Q = test(args, 0, dqn, val_mem, args.title, evaluate=True)  # Test
  print('Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
else:
  # Training loop
  dqn.train()
  T, done = 0, True
  for T in tqdm(range(args.T_max)):
    if done:
      state, done = env.reset(), False
    
    if T % args.replay_frequency == 0:
      dqn.reset_noise()  # Draw a new set of noisy weights

    action = dqn.act(state)  # Choose an action greedily (with noisy weights)
    next_state, reward, done = env.step(action)  # Step
    if args.reward_clip > 0:
      reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
    mem.append(state, action, reward, done)  # Append transition to memory
    T += 1

    if T == args.learn_start:
      if args.add_appendix:
        dqn.add_appendix(args, args.type_of_norm)
    if T >= args.learn_start:
      mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1

      if T % args.replay_frequency == 0:
        if (args.compress and (T >= args.learn_start)):
          loss, regularization_norm, weighted_regularization_norm = dqn.learn(mem, args.add_appendix, args.compress, (T >= args.learn_start), args.type_of_norm, summary_writer, T)  # Train with n-step distributional double-Q learning
          log('T = ' + str(T) + ' / ' + str(args.T_max) + ' | reward: ' + str(reward) + ' | loss: ' + str(loss.detach().cpu().numpy().mean()) + ' | regularization_norm: ' + str(regularization_norm.detach().cpu().numpy()) + ' | weighted regularization_norm: ' + str(weighted_regularization_norm.detach().cpu().numpy()))
        else:
          loss = dqn.learn(mem, args.add_appendix, args.compress, T >= (args.learn_start), args.type_of_norm, summary_writer, T)  # Train with n-step distributional double-Q learning
          log('T = ' + str(T) + ' / ' + str(args.T_max) + ' | reward: ' + str(reward) + ' | loss: ' + str(loss.detach().cpu().numpy().mean()))

      if T % args.evaluation_interval == 0:
        dqn.eval()  # Set DQN (online network) to evaluation mode
        avg_reward, avg_Q = test(args, T, dqn, val_mem, args.title + '_compressed')  # Test
        summary_writer.add_scalar('avg_reward', avg_reward, T)
        previous_forward = dqn.online_net.forward
        dqn.online_net.forward = dqn.online_net.regular_forward
        uncompressed_avg_reward, uncompressed_avg_Q = test(args, T, dqn, val_mem, args.title + '_not_compressed')  # Test
        summary_writer.add_scalar('uncompressed_avg_reward', uncompressed_avg_reward, T)
        dqn.online_net.forward = previous_forward

        log('T = ' + str(T) + ' / ' + str(args.T_max) + ' | Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
        dqn.train()  # Set DQN (online network) back to training mode

      # Update target network
      if T % args.target_update == 0:
        dqn.update_target_net()

    state = next_state

env.close()
summary_writer.export_scalars_to_json("./{}_all_scalars.json".format(args.title))
summary_writer.close()