import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network, SearchControllerConf
from architect import Architect

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='/home/work/dataset/cifar/', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--noise_darts', action='store_true', default=False, help='use noise_darts')
parser.add_argument('--noise_type', type=str, default='N', help='noise type (default gaussian N(0,1)')
parser.add_argument('--max_step', type=int, default=30, help='T_max')
parser.add_argument('--random_search', action='store_true', default=False, help='use single level')
parser.add_argument('--num_identity',  type=int, default=2,  help='the number of skip connect (when random search is true)')
parser.add_argument('--num_arch',  type=int, default=8,  help='the number of architecture (when random search is true)')
parser.add_argument('--flops_threshold',  type=int, default=500,  help='the flops of architecture random generate')
parser.add_argument('--reweight', action='store_true', default=False, help='reweight the two operation with edge')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP-Original', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_lr_gamma', type=float, default=0.9, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(0)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()

  """Noise Darts"""
  if args.noise_darts:
    SearchControllerConf['noise_darts']['noise_type'] = args.noise_type
    SearchControllerConf['noise_darts']['T_max'] = args.max_step
  else:
    SearchControllerConf['noise_darts'] = None

  """Random Darts"""
  if args.random_search:
    SearchControllerConf['random_search']['num_identity'] = args.num_identity
    SearchControllerConf['random_search']['num_arch'] = args.num_arch
    SearchControllerConf['random_search']['flops_threshold'] = args.flops_threshold
  else:
    SearchControllerConf['random_search'] = None

  """Reweight Darts"""
  SearchControllerConf['reweight'] = args.reweight

  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  if args.random_search:
    genotype_list = model.random_generate()
    logging.info('genotype list = %s', genotype_list)
    logging.info('generate done!')
    sys.exit(0)

  model_optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  ## single level
  arch_optimizer = torch.optim.Adam(model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.9, 0.999), weight_decay=args.arch_weight_decay)

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        model_optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)
  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()

    logging.info('genotype = %s', genotype)

    logging.info(F.softmax(model.alphas_normal, dim=-1))
    logging.info(F.softmax(model.alphas_reduce, dim=-1))
    model.update_history()

    # training and search the model
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, model_optimizer, lr, epoch)
    logging.info('train_acc %f', train_acc)

    # validation the model
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)

    utils.save(model, os.path.join(args.save, 'weights.pt'))
    utils.save_file(recoder = model.alphas_normal_history, path = os.path.join(args.save, 'normal'))
    utils.save_file(recoder = model.alphas_reduce_history, path = os.path.join(args.save, 'reduce'))


def train(train_queue, valid_queue, model, architect, criterion, model_optimizer, lr, epoch=None):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)
    input = Variable(input, requires_grad=False).cuda(non_blocking=True)
    target = Variable(target, requires_grad=False).cuda(non_blocking=True)

    # Get a random minibatch from the search queue(validation set) with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search, requires_grad=False).cuda(non_blocking=True)
    target_search = Variable(target_search, requires_grad=False).cuda(non_blocking=True)

    # bi-level
    architect.step(input, target, input_search, target_search, lr, model_optimizer, unrolled=args.unrolled, epoch=epoch)

    # single-level
    model_optimizer.zero_grad()
    ## single-level
    # arch_optimizer.zero_grad()

    logits = model(input, epoch)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)

    # Update the network parameters
    model_optimizer.step()
    ## single level
    # arch_optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d loss: %e top1: %f top5: %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      input = Variable(input, volatile=True).cuda(non_blocking=True)
      target = Variable(target, volatile=True).cuda(non_blocking=True)

      logits = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)

      if step % args.report_freq == 0:
        logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
  main()
