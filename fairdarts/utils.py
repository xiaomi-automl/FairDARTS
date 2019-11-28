import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
from auto_augment import CIFAR10Policy

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  """Compute the top1 and top5 accuracy

  """
  maxk = max(topk)
  batch_size = target.size(0)

  # Return the k largest elements of the given input tensor
  # along a given dimension -> N * k
  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def _data_transforms_cifar(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124] if args.dataset == 'cifar10' else [0.50707519, 0.48654887, 0.44091785]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768] if args.dataset == 'cifar10' else [0.26733428, 0.25643846, 0.27615049]

  normalize_transform = [
      transforms.ToTensor(),
      transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]

  random_transform = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip()]

  if args.auto_aug:
    random_transform += [CIFAR10Policy()]

  if args.cutout:
    cutout_transform = [Cutout(args.cutout_length)]
  else:
    cutout_transform = []

  train_transform = transforms.Compose(
      random_transform + normalize_transform + cutout_transform
  )

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for v in model.parameters())/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.makedirs(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.makedirs(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

def calc_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    t, h = divmod(h, 24)
    return {'day': t, 'hour': h, 'minute': m, 'second': int(s)}

def parse(weights, operation_set,
           op_threshold, parse_method, steps):
  gene = []
  if parse_method == 'darts':
    n = 2
    start = 0
    for i in range(steps): # step = 4
      end = start + n
      W = weights[start:end].copy()
      edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]
      for j in edges:
        k_best = None
        for k in range(len(W[j])):
            if k_best is None or W[j][k] > W[j][k_best]:
              k_best = k
        gene.append((operation_set[k_best], j)) # geno item : (operation, node idx)
      start = end
      n += 1
  elif 'threshold' in parse_method:
    n = 2
    start = 0
    for i in range(steps): # step = 4
      end = start + n
      W = weights[start:end].copy()
      if 'edge' in parse_method:
        edges = list(range(i + 2))
      else: # select edges using darts methods
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]

      for j in edges:
        if 'edge' in parse_method: # OP_{prob > T} AND |Edge| <= 2
          topM = sorted(enumerate(W[j]), key=lambda x: x[1])[-2:]
          for k, v in topM: # Get top M = 2 operations for one edge
            if W[j][k] >= op_threshold:
              gene.append((operation_set[k], i+2, j))
        elif 'sparse' in parse_method: # max( OP_{prob > T} ) and |Edge| <= 2
          k_best = None
          for k in range(len(W[j])):
            if k_best is None or W[j][k] > W[j][k_best]:
              k_best = k
          if W[j][k_best] >= op_threshold:
            gene.append((operation_set[k_best], i+2, j))
        else:
            raise NotImplementedError("Not support parse method: {}".format(parse_method))
      start = end
      n += 1
  return gene


from genotypes import Genotype, PRIMITIVES
def parse_genotype(alphas, steps, multiplier, path = None,
                   parse_method='threshold_sparse', op_threshold=0.85):
    alphas_normal, alphas_reduce = alphas
    gene_normal = parse(alphas_normal, PRIMITIVES, op_threshold, parse_method, steps)
    gene_reduce = parse(alphas_reduce, PRIMITIVES, op_threshold, parse_method, steps)
    concat = range(2 + steps - multiplier, steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )

    if path is not None:
        if not os.path.exists(path):
            os.makedirs(path)
        print('Architecture parsing....\n', genotype)
        save_path = os.path.join(path,parse_method + '_' + str(op_threshold) + '.txt')
        with open(save_path, "w+") as f:
            f.write(str(genotype))
            print('Save in :', save_path)

import matplotlib.pyplot as plt
import json
def save_file(recoder, size = (14, 7), path='./'):
    fig, axs = plt.subplots(*size, figsize = (36, 98))
    num_ops = size[1]
    row = 0
    col = 0
    for (k, v) in recoder.items():
        axs[row, col].set_title(k)
        axs[row, col].plot(v, 'r+')
        if col == num_ops-1:
            col = 0
            row += 1
        else:
            col += 1
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(os.path.join(path, 'output.png'), bbox_inches='tight')
    plt.tight_layout()
    print('save history weight in {}'.format(os.path.join(path, 'output.png')))
    with open(os.path.join(path, 'history_weight.json'), 'w') as outf:
        json.dump(recoder, outf)
        print('save history weight in {}'.format(os.path.join(path, 'history_weight.json')))
