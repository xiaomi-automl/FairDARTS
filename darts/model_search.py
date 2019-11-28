import math
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype

from model import NetworkCIFAR
from thop import profile

SearchControllerConf = {
    'noise_darts': {
     'noise_type': 'N',
     'noise_scheduler': 'cosine_anne',
     'base_lr': 1.0,
     'T_max': 50
     },
    'reweight': False,
    'random_search': {
      'num_identity': 2,
      'num_arch': 8,
      'flops_threshold': None
    }
}

DEBUG_CNT = 0
class NoiseIdentity(nn.Module):
  def __init__(self,  noise_type=None, scheduler='cosine_anne', **kwargs):
    super(NoiseIdentity, self).__init__()
    self.noise_type = noise_type
    if self.noise_type is not None:
      self.scheduler = scheduler
      self.base_lr = kwargs['base_lr']
      self.last_iter = 0
      self.T_max = kwargs['T_max']
      self.gamma_groups = np.linspace(1, 0, self.T_max)
      self.cnt = 0

  def forward(self, x, step=None):
    if step is None:
      self.last_iter += 1
    else:
      self.last_iter = step

    if self.training and self.noise_type is not None:
      if self.scheduler == 'cosine_anne':
        decay = self.base_lr * (1 + math.cos(math.pi * self.last_iter / self.T_max)) / 2.0 if self.last_iter <= self.T_max else 0
      elif self.scheduler == 'step':
        decay = self.gamma_groups[self.last_iter] if self.last_iter < len(self.gamma_groups) and self.last_iter <= self.T_max else 0
      else:
        raise NotImplementedError('not support scheduler {}'.format(self.scheduler))

      # add external noise (guassian noise is prefer)
      if self.noise_type == 'N':
        x = x + decay * torch.randn_like(x, requires_grad=False)
        global DEBUG_CNT
        if DEBUG_CNT % 1000 == 0:
          DEBUG_CNT = 0
          print('===== step: {}'.format(step))
          print('===== decay: {}'.format(decay))
        DEBUG_CNT += 1
      elif self.noise_type == 'U':
        aux = torch.zeros_like(x).data.random_(-3, 3)
        aux.requires_grad = False
        x = x + decay * aux
        # rand_like only generate 0-1, no much effect.
        # x = x + decay * torch.rand_like(x, requires_grad=False)

    return x

class MixedOp(nn.Module):
  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    self.use_noise_identity = True if SearchControllerConf['noise_darts'] and stride == 1 else False

    if self.use_noise_identity:
      self.noise_identity = NoiseIdentity(noise_type=SearchControllerConf['noise_darts']['noise_type'],
                    noise_scheduler=SearchControllerConf['noise_darts']['noise_scheduler'],
                    T_max=SearchControllerConf['noise_darts']['T_max'],
                    base_lr=SearchControllerConf['noise_darts']['base_lr'])

    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      if 'skip' in primitive and self.use_noise_identity:
        op = self.noise_identity
      self._ops.append(op)

  def forward(self, x, weights, epoch=None):
    return sum(w * self.noise_identity(x, epoch) if isinstance(op, NoiseIdentity) and self.training else w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()

    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride)
        self._ops.append(op)

  def forward(self, s0, s1, weights, epoch=None):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0

    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j], epoch) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)

class Network(nn.Module):
  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self._reweight = SearchControllerConf['reweight']

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )

    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]

      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input, epoch=None):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
      s0, s1 = s1, cell(s0, s1, weights, epoch)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0), -1))
    return logits

  def _loss(self, input, target, epoch):
    logits = self(input, epoch)
    return self._criterion(logits, target)

  def _initialize_alphas(self):
    """Initialize the architecture parameter: alpha
    """
    # k = 2 + 3 + 4 + 5 = 14
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    # alphas_normal: size = 14 * 8; alphas_reduce = 14 * 8
    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)

    # init the history
    self.alphas_normal_history = {}
    self.alphas_reduce_history = {}
    mm = 0
    last_id = 1
    node_id = 0
    for i in range(k):
      for j in range(num_ops):
        self.alphas_normal_history['edge: {}, op: {}'.format((node_id, mm), PRIMITIVES[j])] = []
        self.alphas_reduce_history['edge: {}, op: {}'.format((node_id, mm), PRIMITIVES[j])] = []
      if mm == last_id:
        mm = 0
        last_id += 1
        node_id += 1
      else:
        mm += 1

    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights, reweight):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()

        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]

        for j in edges:
          k_best = None
          for k in range(len(W[j])):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          if reweight:
            gene.append((PRIMITIVES[k_best], j, W[j][k_best])) # geno item: (operation, node idx, weight)
          else:
            gene.append((PRIMITIVES[k_best], j))              # geno item: (operation, node idx)
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(), self._reweight)
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(), self._reweight)

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

  def update_history(self):
    mm = 0
    last_id = 1
    node_id = 0
    weights1 = F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy()
    weights2 = F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy()

    k, num_ops = weights1.shape
    for i in range(k):
      for j in range(num_ops):
        self.alphas_normal_history['edge: {}, op: {}'.format((node_id, mm), PRIMITIVES[j])].append(float(weights1[i][j]))
        self.alphas_reduce_history['edge: {}, op: {}'.format((node_id, mm), PRIMITIVES[j])].append(float(weights2[i][j]))
      if mm == last_id:
        mm = 0
        last_id += 1
        node_id += 1
      else:
        mm += 1

  def random_generate(self):

    num_skip_connect = SearchControllerConf['random_search']['num_identity']
    num_arch = SearchControllerConf['random_search']['num_arch']
    flops_threshold = SearchControllerConf['random_search']['flops_threshold']

    """Random generate the architecture"""
    # k = 2 + 3 + 4 + 5 = 14
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    self.random_arch_list = []
    for ai in range(num_arch):
      seed = random.randint(0, 1000)
      torch.manual_seed(seed)
      while True:
        self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=False)
        self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=False)
        arch = self.genotype()
        # if the skip connect meet num_skip_connect
        op_names, indices = zip(*arch.normal)
        cnt = 0
        for name, index in zip(op_names, indices):
          if name == 'skip_connect':
            cnt += 1
        if cnt == num_skip_connect:
          # the flops threshold
          model = NetworkCIFAR(36, 10, 20, True, arch, False)
          flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32),), verbose=False)
          if flops / 1e6 >= flops_threshold:
            self.random_arch_list += [('arch_' + str(ai), arch)]
            break
          else:
            continue

    return self.random_arch_list