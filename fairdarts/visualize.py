import sys
import genotypes
from graphviz import Digraph


def plot1(genotype, filename):
  g = Digraph(
      format='png',
      edge_attr=dict(fontsize='20', fontname="times"),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
      engine='dot')
  g.body.extend(['rankdir=LR'])

  g.node("c_{k-2}", fillcolor='darkseagreen2')
  g.node("c_{k-1}", fillcolor='darkseagreen2')
  assert len(genotype) % 2 == 0
  steps = len(genotype) // 2

  for i in range(steps):
    g.node(str(i), fillcolor='lightblue')

  for i in range(steps):
    for k in [2*i, 2*i + 1]:
      op, j = genotype[k]
      if j == 0:
        u = "c_{k-2}"
      elif j == 1:
        u = "c_{k-1}"
      else:
        u = str(j-2)
      v = str(i)
      g.edge(u, v, label=op, fillcolor="gray")

  g.node("c_{k}", fillcolor='palegoldenrod')
  for i in range(steps):
    g.edge(str(i), "c_{k}", fillcolor="gray")

  g.render(filename, view=True)

def parse(genotype):
  op_names, tos, froms = zip(*genotype)
  ops = {}
  for name_i, to_i, from_i in zip(op_names, tos, froms):
    if str(to_i) in ops.keys():
      if str(from_i) in ops[str(to_i)]:
        ops[str(to_i)][str(from_i)] += [name_i]
      else:
        ops[str(to_i)][str(from_i)] = []
        ops[str(to_i)][str(from_i)] += [name_i]
    else:
      ops[str(to_i)] = {}
      ops[str(to_i)][str(from_i)] = []
      ops[str(to_i)][str(from_i)] += [name_i]
  return ops

def plot2(genotype, filename):
  g = Digraph(
      format='png',
      edge_attr=dict(fontsize='20', fontname="times"),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
      engine='dot')
  g.body.extend(['rankdir=LR'])

  g.node("c_{k-2}", fillcolor='darkseagreen2')
  g.node("c_{k-1}", fillcolor='darkseagreen2')

  ops = parse(genotype)

  for k, v in ops.items():
    g.node(str(int(k)-2), fillcolor='lightblue')

  for to_i, v in ops.items():
    for from_i, op_i in v.items():
      if from_i == '0':
        u = "c_{k-2}"
      elif from_i == '1':
        u = "c_{k-1}"
      else:
        u = str(int(from_i)-2)
      for op in op_i:
        g.edge(u, str(int(to_i)-2), label=op, fillcolor="gray")

  g.node("c_{k}", fillcolor='palegoldenrod')
  for to_i, v in ops.items():
    g.edge(str(int(to_i)-2), "c_{k}", fillcolor="gray")

  g.render(filename, view=True)

import os
from utils import create_exp_dir
if __name__ == '__main__':
  if len(sys.argv) < 2:
    print("usage:\n python {} ARCH_NAME".format(sys.argv[0]))
    sys.exit(1)
  genotype_name = sys.argv[1]
  file_path = './vis/' + genotype_name
  create_exp_dir(file_path)
  if len(sys.argv) == 3:
    parse_method = sys.argv[2]
  else:
    parse_method = None

  try:
    genotype = eval('genotypes.{}'.format(genotype_name))
  except AttributeError:
    print("{} is not specified in genotypes.py".format(genotype_name)) 
    sys.exit(1)

  if parse_method == 'threshold':
    plot2(genotype.normal, os.path.join(file_path, "normal"))
    plot2(genotype.reduce, os.path.join(file_path, "reduction"))
  elif parse_method == None or parse_method == 'darts':
    plot1(genotype.normal, os.path.join(file_path, "normal"))
    plot1(genotype.reduce, os.path.join(file_path, "reduction"))
  else:
    print('Not support parse method :{}'.format(parse_method))

