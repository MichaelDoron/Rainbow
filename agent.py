import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import optim
from model import DQN
import matplotlib
matplotlib.use('agg')
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import random
import numpy as np

class Agent():
  def __init__(self, args, env):
    self.action_space = env.action_space()
    self.args = args
    self.atoms = args.atoms
    self.Vmin = args.V_min
    self.Vmax = args.V_max
    self.original_loss = torch.Tensor([0.0]).to(device=args.device)
    self.original_raw_loss = torch.Tensor([0.0]).to(device=args.device)
    self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)  # Support (range) of z
    self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
    self.batch_size = args.batch_size
    self.n = args.multi_step
    self.discount = args.discount
    self.losses = []
    self.device = args.device
    self.parameters = {}
    self.zeroed_weights = []
    self.regularization_norms = []
    self.raw_losses = []
    self.regularization_weight = torch.Tensor([0.]).to(device=args.device)  
    self.regularization_norm = 1
    self.online_net = DQN(args, self.action_space).to(device=args.device)
    self.target_net = DQN(args, self.action_space).to(device=args.device)
    if (args.compress and (args.evaluate)):
      print('added compression for evaluation')
      self.target_net.add_regularization_norm(args)
      self.online_net.add_regularization_norm(args)
    if (args.add_appendix and (args.evaluate or args.compress == False)):
      print('added appendix')
      self.add_appendix(args, args.type_of_norm)
      if (args.just_appendix):
        self.online_net.forward = self.online_net.forward_appendix_just_appendix
    if args.model and os.path.isfile(args.model):
      # Always load tensors onto CPU by default, will shift to GPU if necessary
      self.online_net.load_state_dict(torch.load(args.model, map_location='cpu'), strict=False)

    self.online_net.train()

    self.update_target_net()
    self.target_net.train()

    if (args.compress and not (args.evaluate)):
      self.target_net.add_regularization_norm(args)
      self.online_net.add_regularization_norm(args)

    for param in self.target_net.parameters():
      param.requires_grad = False

    self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.lr, eps=args.adam_eps)

  # Resets noisy weights in all linear layers (of online net only)
  def reset_noise(self):
    self.online_net.reset_noise()

  # Acts based on single state (no batch)
  def act(self, state):
    with torch.no_grad():
      return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).argmax(1).item()

  # Acts with an ε-greedy policy (used for evaluation only)
  def act_e_greedy(self, state, epsilon=0.001):  # High ε can reduce evaluation scores drastically
    return random.randrange(self.action_space) if random.random() < epsilon else self.act(state)

  def learn(self, mem, add_appendix, compress, started_compressing, type_of_norm, summary_writer, T):
    # Sample transitions
    idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)

    # Calculate current state probabilities (online network noise already sampled)
    log_ps = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
    log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)

    with torch.no_grad():
      # Calculate nth next state probabilities
      pns = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
      dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
      argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
      self.target_net.reset_noise()  # Sample new target net noise
      pns = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
      pns_a = pns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

      # Compute Tz (Bellman operator T applied to z)
      Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
      Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
      # Compute L2 projection of Tz onto fixed support z
      b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
      l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
      # Fix disappearing probability mass when l = b = u (b is int)
      l[(u > 0) * (l == u)] -= 1
      u[(l < (self.atoms - 1)) * (l == u)] += 1

      # Distribute probability of Tz
      m = states.new_zeros(self.batch_size, self.atoms)
      offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
      m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
      m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)
    
    loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
    
    self.losses.append((weights * loss).mean().mean().detach().cpu().numpy())
    self.raw_losses.append(loss.mean().mean().detach().cpu().numpy())
    if self.original_loss.data == 0: self.original_loss = torch.mean(torch.Tensor(np.array(self.losses)).to(device=self.device)).to(device=self.device)
    if self.original_raw_loss.data == 0: self.original_raw_loss = torch.mean(torch.Tensor(np.array(self.raw_losses)).to(device=self.device)).to(device=self.device)
    
    if (compress and started_compressing): # Add L0 norm for sparsity
      self.set_regularization_weight(print_weights = add_appendix, type_of_norm = type_of_norm)
    self.online_net.zero_grad()

    if (compress and started_compressing) or (self.args.add_appendix and (self.args.compress == False)): # Add L0 norm for sparsity
      final_loss = (weights * loss)
      final_loss += self.get_weighted_regularization_norm(type_of_norm)
      summary_writer.add_scalar('loss', (self.losses[-1]), T)
      summary_writer.add_scalar('raw loss', (self.raw_losses[-1]), T)
      summary_writer.add_scalar('regularization loss', (self.get_regularization_norm(type_of_norm)).detach().cpu().numpy(), T)
      summary_writer.add_scalar('weighted regularization loss', (self.get_weighted_regularization_norm(type_of_norm)).detach().cpu().numpy(), T)
      summary_writer.add_scalar('regularization weight', self.regularization_weight, T)
      (final_loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
    if (add_appendix):
      summary_writer.add_scalar('Merger weight', (self.online_net.app4.sigmoid_center).detach().cpu().numpy().mean(), T)
      summary_writer.add_scalar('linear weight', (self.online_net.app_lin.weight).detach().cpu().numpy().mean(), T)
      if self.args.freeze_merge == False:
        summary_writer.add_scalar('Merger grad', (self.online_net.app4.sigmoid_center.grad.mean()).detach().cpu().numpy(), T)

      if type_of_norm == 'L0':
        self.zeroed_weights.append(self.get_regularization_percentage())
      self.regularization_norms.append(self.get_regularization_norm(type_of_norm))

    else:
      ((weights * loss)).mean().backward()  # Backpropagate importance-weighted minibatch loss
      summary_writer.add_scalar('loss', (self.losses[-1]), T)
      summary_writer.add_scalar('raw loss', (self.raw_losses[-1]), T)
    self.optimiser.step()
    if (add_appendix): # Add L0 norm for sparsity
      if self.args.merge_type == 'addition':
        self.online_net.app4.sigmoid_center.data.clamp_(min=0.0, max=1.0)
      else:
        self.online_net.app4.sigmoid_center.data.clamp_(min=-0.05, max=1.05)

    mem.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions
    if (compress and started_compressing):
      return -torch.sum(m * log_ps_a, 1), self.get_regularization_norm(type_of_norm), self.get_weighted_regularization_norm(type_of_norm)
    return loss
  def update_target_net(self):
    self.target_net.load_state_dict(self.online_net.state_dict())
  
  # Save model parameters on current device (don't move model between devices)
  def save(self, path, specific_title, args):
    torch.save(self.online_net.state_dict(), os.path.join(path, self.args.base_folder + 'models/' + specific_title + '_' + 'model.pth'))

  # Evaluates Q-value based on single state (no batch)
  def evaluate_q(self, state):
    with torch.no_grad():
      return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).max(1)[0].item()
      
  def train(self):
    self.online_net.train()

  def eval(self):
    self.online_net.eval()

  def weight_increase(self, losses, raw_losses, smoothness, max_value, regularization_weight):
    result_raw = ((self.original_raw_loss - raw_losses[-10:].mean()).clamp_(max=0) * smoothness).exp()
    result = ((self.original_loss - losses[-10:].mean()).clamp_(max=0) * smoothness).exp()
    return ((min(result, result_raw) * max_value * 10) - (9 * max_value))

  def set_regularization_weight(self, print_weights, type_of_norm):
    if len(self.losses) <= 1000: 
      if (type_of_norm == 'L0'):
        self.online_net.conv1.lamba = self.regularization_weight
        self.online_net.conv2.lamba = self.regularization_weight
        self.online_net.conv3.lamba = self.regularization_weight
      return self.regularization_weight
    self.regularization_weight += self.weight_increase(torch.Tensor(np.array(self.losses)).to(device=self.device),torch.Tensor(np.array(self.raw_losses)).to(device=self.device), smoothness = 50, max_value = 0.0001, regularization_weight = self.regularization_weight).to(device=self.device)
    self.regularization_weight.clamp_(min=0)

    if print_weights:
      print('self.original_loss {}'.format(self.original_loss.item()))
      print('self.original_raw_loss {}'.format(self.original_raw_loss.item()))
      print('Merger appendix weights: {}'.format((self.online_net.app4.sigmoid_center).detach().cpu().numpy().mean()))
      print('Non merger appendix weights: {}'.format((1 - self.online_net.app4.sigmoid_center).detach().cpu().numpy().mean()))
        
    if (type_of_norm == 'L0'):
      self.online_net.conv1.lamba = self.regularization_weight
      self.online_net.conv2.lamba = self.regularization_weight
      self.online_net.conv3.lamba = self.regularization_weight
    return self.regularization_weight

  def get_regularization_percentage(self):
    result = 0
    result += (float((self.online_net.conv1.sample_weights() == 0).sum()))
    result += (float((self.online_net.conv2.sample_weights() == 0).sum()))
    result += (float((self.online_net.conv3.sample_weights() == 0).sum()))
    result /=  (float((self.online_net.conv1.sample_weights()).numel()) + float((self.online_net.conv2.sample_weights()).numel()) + float((self.online_net.conv3.sample_weights()).numel()))
    return result    

  def get_regularization_norm(self, type_of_norm):
    if (type_of_norm == 'L0'):
      self.online_net.conv1.lamba = 1
      self.online_net.conv2.lamba = 1
      self.online_net.conv3.lamba = 1
      self.regularization_norm = - (1. / 50000) * self.online_net.conv1.regularization()
      self.regularization_norm += - (1. / 50000) * self.online_net.conv2.regularization()
      self.regularization_norm += - (1. / 50000) * self.online_net.conv3.regularization()
      self.online_net.conv1.lamba = self.regularization_weight
      self.online_net.conv2.lamba = self.regularization_weight
      self.online_net.conv3.lamba = self.regularization_weight    
    else:
      self.regularization_norm =  self.online_net.conv1.weight.abs().sum()
      self.regularization_norm += self.online_net.conv2.weight.abs().sum()
      self.regularization_norm += self.online_net.conv3.weight.abs().sum()

    return self.regularization_norm

  def get_weighted_regularization_norm(self, type_of_norm):
    if (type_of_norm == 'L0'):
      self.regularization_norm = - (1. / 50000) * self.online_net.conv1.regularization()
      self.regularization_norm += - (1. / 50000) * self.online_net.conv2.regularization()
      self.regularization_norm += - (1. / 50000) * self.online_net.conv3.regularization()
    else:
      self.regularization_norm =  self.online_net.conv1.weight.abs().sum()
      self.regularization_norm += self.online_net.conv2.weight.abs().sum()
      self.regularization_norm += self.online_net.conv3.weight.abs().sum()
    return self.regularization_weight * self.regularization_norm    

  def get_number_of_zero_elements(self):
    number_of_zero_elements = 0
    print('zero elements in conv1 {}, conv2 {}, conv3 {}'.format(
      len([p for p in self.online_net.conv1.sample_z(1, False).flatten().detach().cpu().numpy() if p == 0]),
      len([p for p in self.online_net.conv2.sample_z(1, False).flatten().detach().cpu().numpy() if p == 0]),
      len([p for p in self.online_net.conv3.sample_z(1, False).flatten().detach().cpu().numpy() if p == 0])))
    return number_of_zero_elements

  def add_appendix(self, args, type_of_norm):
    self.online_net.add_appendix(args.device, args.appendix_type, type_of_norm, args.multiplier)
    self.online_net.to(device=args.device)
    self.target_net.add_appendix(args.device, args.appendix_type, type_of_norm, args.multiplier)
    self.target_net.to(device=args.device)
    self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.lr, eps=args.adam_eps)
    if type_of_norm == 'L0':
      with torch.no_grad(): self.original_regularization_norm = self.online_net.conv1.regularization().abs()
    if (self.args.just_appendix):
      self.online_net.forward = self.online_net.forward_appendix_just_appendix
      self.target_net.forward = self.target_net.forward_appendix_just_appendix

