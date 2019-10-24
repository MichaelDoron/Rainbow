import math
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torchvision import models, transforms
from torch.nn import AdaptiveAvgPool2d
import copy
from env import Env
# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, std_init=0.4):
    super(NoisyLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.std_init = std_init
    self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
    self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
    self.bias_mu = nn.Parameter(torch.empty(out_features))
    self.bias_sigma = nn.Parameter(torch.empty(out_features))
    self.register_buffer('bias_epsilon', torch.empty(out_features))
    self.reset_parameters()
    self.reset_noise()

  def reset_parameters(self):
    mu_range = 1 / math.sqrt(self.in_features)
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

  def _scale_noise(self, size):
    x = torch.randn(size)
    return x.sign().mul_(x.abs().sqrt_())

  def reset_noise(self):
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
    self.bias_epsilon.copy_(epsilon_out)

  def forward(self, input):
    if self.training:
      return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
    else:
      return F.linear(input, self.weight_mu, self.bias_mu)

class Merger(torch.nn.Module):
    def __init__(self, in_features, device):
        super(Merger, self).__init__()
        self.in_features = in_features
        self.sigmoid_center = torch.nn.Parameter((torch.abs(torch.rand(1)) - 1))
        self.sigmoid_center.data.clamp_(min=0,max=1)
        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self, multiplier = 0.01):
        self.sigmoid_center = torch.nn.Parameter(torch.Tensor([multiplier]))
        self.sigmoid_center.data.clamp_(min=0,max=1)

    def forward(self, appendix_input, original_input):
        return (appendix_input * self.sigmoid_center) + (original_input * (1 - self.sigmoid_center))

class Merger_probability_addition(torch.nn.Module):
    def __init__(self, in_features, device):
        super(Merger_probability_addition, self).__init__()
        self.device = device
        self.in_features = in_features
        self.sigmoid_center = torch.nn.Parameter(torch.Tensor([0.01]))
        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self, multiplier = -0.01):
        self.sigmoid_center = torch.nn.Parameter(torch.Tensor([multiplier]))
        self.sigmoid_center.data.clamp_(min=-0.1,max=1.1)

    def forward(self, appendix_input, original_input):
        result = torch.zeros(size=original_input.shape).to(device=self.device)
        sigmoid = torch.sigmoid((torch.linspace(0, 1, self.in_features).to(device=self.device) - self.sigmoid_center) * 10**3).to(device=self.device)
        pivot = (sigmoid == sigmoid.max()).nonzero()[0]
        result = result + original_input * sigmoid
        result = result + appendix_input * (1 - sigmoid)
        if pivot > 0:
            result[:,pivot]  = (appendix_input[:, pivot]) * 0.5 + (original_input[:, pivot]) * 0.5
        return result

class Merger_probability(torch.nn.Module):
    def __init__(self, in_features, device):
        super(Merger_probability, self).__init__()
        self.device = device
        self.in_features = in_features
        self.sigmoid_center = torch.nn.Parameter(torch.Tensor([0.01]))
        
        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self, multiplier = 0.01):
        self.sigmoid_center = torch.nn.Parameter(torch.Tensor([multiplier]))
        self.sigmoid_center.data.clamp_(min=-0.2,max=1.2)

    def forward(self, appendix_input, original_input):
        result = torch.zeros(size=original_input.shape).to(device=self.device)
        sigmoid = torch.sigmoid((torch.linspace(0, 1, self.in_features).to(device=self.device) - self.sigmoid_center) * 10**3).to(device=self.device)
        pivot = (sigmoid == sigmoid.max()).nonzero()[0]
        result = result + original_input * sigmoid
        result = result + appendix_input * (1 - sigmoid)
        if sigmoid[pivot] > 0:
            if torch.rand(1) > 0.5:
                result[:,pivot]  = (appendix_input[:, pivot])
            else:
                result[:,pivot]  = (original_input[:, pivot])
        return result

class Merger_percentage(torch.nn.Module):
    def __init__(self, in_features, device):
        super(Merger, self).__init__()
        self.device = device
        self.in_features = in_features
        self.sigmoid_center = torch.nn.Parameter(torch.Tensor([0.1]))
        self.merge_layer = torch.arange(self.in_features).long().to(device=self.device)
        
        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self, multiplier = 0.1):
        self.sigmoid_center = torch.nn.Parameter(torch.Tensor([multiplier]))
        self.sigmoid_center.data.clamp_(min=-0.2,max=1.2)

    def forward(self, appendix_input, original_input):
        result = torch.zeros(size=original_input.shape).to(device=self.device)
        sigmoid = torch.sigmoid((torch.linspace(0, 1, self.in_features).to(device=self.device) - self.sigmoid_center) * 10**4).to(device=self.device)
        original_bits = torch.where(sigmoid == sigmoid.max(), sigmoid, torch.zeros(sigmoid.shape).to(device=self.device))
        original_bits = original_bits / original_bits.max()
        result[:,self.merge_layer] = result[:,self.merge_layer] + (original_input[:, self.merge_layer] * original_bits)
        
        appendix_bits = torch.where(sigmoid != sigmoid.max(), sigmoid, torch.zeros(sigmoid.shape).to(device=self.device))
        if appendix_bits.sum() > 0: appendix_bits = appendix_bits / appendix_bits.max()
        result[:,self.merge_layer] = result[:,self.merge_layer] + (appendix_input[:, self.merge_layer] * appendix_bits)
        return result

class DQN(nn.Module):
  def __init__(self, args, action_space):
    super().__init__()
    self.args = args
    self.atoms = args.atoms
    self.action_space = action_space
    self.device = args.device
    self.conv1 = nn.Conv2d(args.history_length, 32, 8, stride=4, padding=1)
    self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
    self.conv3 = nn.Conv2d(64, 64, 3)
    self.fc_h_v = NoisyLinear(3136, args.hidden_size, std_init=args.noisy_std)
    self.fc_h_a = NoisyLinear(3136, args.hidden_size, std_init=args.noisy_std)
    self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
    self.fc_z_a = NoisyLinear(args.hidden_size, action_space * self.atoms, std_init=args.noisy_std)

  def add_appendix(self, device, appendix_type, type_of_norm, multiplier):
    if (appendix_type == 'conv'):
        self.app1 = nn.Conv2d(self.args.history_length, 32, 8, stride=4, padding=1)
        self.app2 = nn.Conv2d(32, 64, 4, stride=2)
        self.app3 = nn.Conv2d(64, 64, 3)
        if self.args.merge_type == 'addition':
            self.app4 = Merger(3136, device)
        elif self.args.merge_type == 'probability_continuous':
            self.app4 = Merger_probability_addition(3136, device)
        elif self.args.merge_type == 'probability_dropout':
            self.app4 = Merger_probability(3136, device)

        self.forward = self.forward_appendix_conv
        for param in self.parameters(): param.requires_grad = True
        for param in self.app1.parameters(): param.requires_grad = True
        for param in self.app2.parameters(): param.requires_grad = True
        for param in self.app3.parameters(): param.requires_grad = True
        for param in self.app4.parameters(): param.requires_grad = True
    elif (appendix_type == 'same'):
        self.app1 = nn.Conv2d(self.args.history_length, 32, 8, stride=4, padding=1)
        self.app2 = nn.Conv2d(32, 64, 4, stride=2)
        self.app3 = nn.Conv2d(64, 64, 3)
        if type_of_norm == 'L0':
            self.app1.weight = torch.nn.Parameter(torch.Tensor(np.copy(self.conv1.weights.detach().cpu().numpy())))
            self.app2.weight = torch.nn.Parameter(torch.Tensor(np.copy(self.conv2.weights.detach().cpu().numpy())))
            self.app3.weight = torch.nn.Parameter(torch.Tensor(np.copy(self.conv3.weights.detach().cpu().numpy())))
        elif type_of_norm == 'L1':
            self.app1.weight = torch.nn.Parameter(torch.Tensor(np.copy(self.conv1.weight.detach().cpu().numpy())))
            self.app2.weight = torch.nn.Parameter(torch.Tensor(np.copy(self.conv2.weight.detach().cpu().numpy())))
            self.app3.weight = torch.nn.Parameter(torch.Tensor(np.copy(self.conv3.weight.detach().cpu().numpy())))

        if self.args.merge_type == 'addition':
            self.app4 = Merger(3136, device)
        elif self.args.merge_type == 'probability_continuous':
            self.app4 = Merger_probability_addition(3136, device)
        elif self.args.merge_type == 'probability_dropout':
            self.app4 = Merger_probability(3136, device)

        self.app4.reset_parameters(multiplier = multiplier)
        self.forward = self.forward_appendix_model_merge

        for param in self.parameters(): param.requires_grad = True
        for param in self.app1.parameters(): param.requires_grad = False
        for param in self.app2.parameters(): param.requires_grad = False
        for param in self.app3.parameters(): param.requires_grad = False
        for param in self.app4.parameters(): param.requires_grad = True      
        if (self.args.freeze_merge):
            for param in self.app4.parameters(): param.requires_grad = False      
        else:
            for param in self.app4.parameters(): param.requires_grad = True      

    elif (appendix_type == 'flow'):
        class args:
            rgb_max = 255.
            fp16 = False
            resume = base_folder + 'models/FlowNet2_checkpoint.pth.tar'
        self.app1 = FlowNet2(args)
        checkpoint = torch.load(args.resume)
        self.app1.load_state_dict(checkpoint['state_dict'])
        self.app2 = nn.Linear(2 * 64 * 64, 3136)
        if self.args.merge_type == 'addition':
            self.app4 = Merger(3136, device)
        elif self.args.merge_type == 'probability_continuous':
            self.app4 = Merger_probability_addition(3136, device)
        elif self.args.merge_type == 'probability_dropout':
            self.app4 = Merger_probability(3136, device)

        self.forward = self.forward_appendix_flow
        for param in self.parameters(): param.requires_grad = True
        for param in self.app1.parameters(): param.requires_grad = False
        for param in self.app2.parameters(): param.requires_grad = True        
        for param in self.app4.parameters(): param.requires_grad = True   
        if (self.args.freeze_merge):
            for param in self.app4.parameters(): param.requires_grad = False      
        else:
            for param in self.app4.parameters(): param.requires_grad = True      

    elif (appendix_type == 'resnet'):
        self.app1 = models.resnet18(pretrained=True)
        for param in self.app1.parameters(): param.requires_grad = False
        self.app1.fc = nn.Linear(self.app1.fc.in_features, 3136)
        self.app_lin = nn.Linear(3136, 3136)
        self.app_lin.weight.data = torch.eye(n=self.app_lin.weight.data.shape[0])
        for param in self.app1.fc.parameters(): param.requires_grad = True
        if self.args.merge_type == 'addition':
            self.app4 = Merger(3136, device)
        elif self.args.merge_type == 'probability_continuous':
            self.app4 = Merger_probability_addition(3136, device)
        elif self.args.merge_type == 'probability_dropout':
            self.app4 = Merger_probability(3136, device)

        self.app4.reset_parameters(multiplier = multiplier)
        self.forward = self.forward_appendix_resnet
        if (self.args.freeze_merge):
            for param in self.app4.parameters(): param.requires_grad = False      
        else:
            for param in self.app4.parameters(): param.requires_grad = True      

    elif (appendix_type == 'model'):
        new_args = copy.deepcopy(self.args)
        new_args.game = self.args.appendix_model[(self.args.appendix_model.rfind('/') + 1) : self.args.appendix_model.rfind('.pth')]
        action_space = Env(new_args).action_space()
        self.apendix_model = DQN(args = new_args, action_space = action_space).to(device=self.args.device)
        self.apendix_model.load_state_dict(torch.load(self.args.appendix_model, map_location='cpu'), strict=False)
        
        self.app1 = nn.Conv2d(self.args.history_length, 32, 8, stride=4, padding=1)
        self.app2 = nn.Conv2d(32, 64, 4, stride=2)
        self.app3 = nn.Conv2d(64, 64, 3)
        self.app_lin = nn.Linear(3136, 3136)
        self.app_lin.weight.data = torch.eye(n=self.app_lin.weight.data.shape[0])

        self.app1.weight = torch.nn.Parameter(torch.Tensor(np.copy(self.apendix_model.conv1.weight.detach().cpu().numpy())))
        self.app2.weight = torch.nn.Parameter(torch.Tensor(np.copy(self.apendix_model.conv2.weight.detach().cpu().numpy())))
        self.app3.weight = torch.nn.Parameter(torch.Tensor(np.copy(self.apendix_model.conv3.weight.detach().cpu().numpy())))

        if self.args.merge_type == 'addition':
            self.app4 = Merger(3136, device)
        elif self.args.merge_type == 'probability_continuous':
            self.app4 = Merger_probability_addition(3136, device)
        elif self.args.merge_type == 'probability_dropout':
            self.app4 = Merger_probability(3136, device)

        self.app4.reset_parameters(multiplier = multiplier)
        self.forward = self.forward_appendix_model_merge

        for param in self.parameters(): param.requires_grad = True
        for param in self.app1.parameters(): param.requires_grad = False
        for param in self.app2.parameters(): param.requires_grad = False
        for param in self.app3.parameters(): param.requires_grad = False
        for param in self.app_lin.parameters(): param.requires_grad = True
        if (self.args.freeze_merge):
            for param in self.app4.parameters(): param.requires_grad = False
        else:
            for param in self.app4.parameters(): param.requires_grad = True

  def add_regularization_norm(self, args):
      if (args.type_of_norm == 'L0'):
          conv1_weights = self.conv1.weight
          self.conv1 = L0Conv2d(args.history_length, 32, 8, stride=4, padding=1, dilation=1, groups=1, bias=True,
                     droprate_init=0.001, temperature=2./3., weight_decay=0., lamba=1., local_rep=False, device=args.device).to(args.device)
          self.conv1.weights = conv1_weights
          conv2_weights = self.conv2.weight
          self.conv2 = L0Conv2d(32, 64, 4, stride=2, padding=0, dilation=1, groups=1, bias=True,
                     droprate_init=0.001, temperature=2./3., weight_decay=0., lamba=1., local_rep=False, device=args.device).to(args.device)
          self.conv2.weights = conv2_weights
          conv3_weights = self.conv3.weight
          self.conv3 = L0Conv2d(64, 64, 3, stride=1, padding=0, dilation=1, groups=1, bias=True,
                     droprate_init=0.001, temperature=2./3., weight_decay=0., lamba=1., local_rep=False, device=args.device).to(args.device)
          self.conv3.weights = conv3_weights
          for param in self.conv1.parameters(): param.requires_grad = True
          for param in self.conv2.parameters(): param.requires_grad = True
          for param in self.conv3.parameters(): param.requires_grad = True

  def regular_forward(self, x, log=False):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = x.view(-1, 3136)
    v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    if log:  # Use log softmax for numerical stability
      q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
    else:
      q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
    return q

  def forward(self, x, log=False):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = x.view(-1, 3136)
    v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    if log:  # Use log softmax for numerical stability
      q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
    else:
      q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
    return q

  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()

  def forward_appendix_model_merge(self, x, log=False):
    app_x = x.clone()

    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = x.view(-1, 3136)

    app_x = F.relu(self.app1(app_x))
    app_x = F.relu(self.app2(app_x))
    app_x = F.relu(self.app3(app_x))
    app_x = app_x.view(-1, 3136)    
    app_x = F.relu(self.app_lin(app_x))
    x = self.app4(app_x, x)


    v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    if log:  # Use log softmax for numerical stability
      q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
    else:
      q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
    return q

  def forward_appendix_just_appendix(self, x, log=False):
    app_x = x.clone()

    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = x.view(-1, 3136)

    app_x = F.relu(self.app1(app_x))
    app_x = F.relu(self.app2(app_x))
    app_x = F.relu(self.app3(app_x))
    app_x = app_x.view(-1, 3136)    
    app_x = F.relu(self.app_lin(app_x))
    x = self.app4(app_x, torch.zeros(size=x.size()).to(device=self.device))


    v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    if log:  # Use log softmax for numerical stability
      q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
    else:
      q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
    return q

  def forward_appendix_resnet(self, x, log=False):
    app_x = x.clone()

    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = x.view(-1, 3136)

    adaptiveAvgPool2d = AdaptiveAvgPool2d((256, 256))
    img = adaptiveAvgPool2d(app_x[:, 3, :, :]).unsqueeze(1).repeat(1,3,1,1)
    app_x = F.relu(self.app1(img))
    app_x = app_x.view(-1, 3136)
    app_x = F.relu(self.app_lin(app_x))

    x = self.app4(app_x, x)

    v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    if log:  # Use log softmax for numerical stability
      q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
    else:
      q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
    return q
