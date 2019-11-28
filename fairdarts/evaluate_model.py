import os
import sys
import utils as dutils
import argparse
import torch.utils
import genotypes
import torchvision.datasets as dset
from model import NetworkCIFAR as Network
from thop import profile

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='/home/work/dataset/cifar/', help='location of the data corpus')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--parse_method', type=str, default='darts', help='experiment name')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')


if __name__ == '__main__':

    args = parser.parse_args()
    args.dataset = 'cifar10'
    args.auto_aug = False
    args.cutout = False
    train_transform, valid_transform = dutils._data_transforms_cifar(args)
    valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
    valid_queue = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=False, pin_memory=True, num_workers=4)

    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    genotype = eval('genotypes.%s' % args.arch)
    print('Parsing Genotypes: {}'.format(genotype))
    model = Network(36, 10, 20, 0.4, genotype, args.parse_method)

    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32),), verbose=False)
    print('flops = %fM' % (flops / 1e6))
    print('param size = %fM' %( params / 1e6))

    model = model.cuda()

    if args.model_path and os.path.isfile(args.model_path):
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print('The Pre-Trained Model Is InValid!')
        sys.exit(-1)

    top1 = dutils.AvgrageMeter()
    top5 = dutils.AvgrageMeter()
    model.eval()
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            logits, _ = model(input)
            prec1, prec5 = dutils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                print('valid %03d %f %f' % (step, top1.avg, top5.avg))
        print("Final Mean Top1: {}, Top5: {}".format(top1.avg, top5.avg))