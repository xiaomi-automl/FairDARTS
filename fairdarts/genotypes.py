from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

"""
Operation sets
"""
PRIMITIVES = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

"""====== Different Archirtecture By Other Methods"""
# 608M, 3.83M
NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)
    
AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)

# https://arxiv.org/pdf/1802.03268.pdf
# 627M, 4.02M
ENASNet = Genotype(
  normal = [
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('sep_conv_5x5', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 1),
    ('avg_pool_3x3', 0),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 1), # 2
    ('sep_conv_3x3', 1),
    ('avg_pool_3x3', 1), # 3
    ('sep_conv_3x3', 1),
    ('avg_pool_3x3', 1), # 4
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 4), # 5
    ('sep_conv_3x3', 5),
    ('sep_conv_5x5', 0),
  ],
  reduce_concat = [2, 3, 4, 5, 6],
)

# from https://github.com/D-X-Y/NAS-Projects/blob/master/others/GDAS/lib/nas/genotypes.py
# 519M, 3.36M
GDAS_V1 = Genotype(
  normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 3), ('skip_connect', 0), ('sep_conv_5x5', 4), ('sep_conv_3x3', 3)],
  normal_concat=range(2, 6),
  reduce=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2)],
  reduce_concat=range(2, 6)
)

# 528M, 3.3
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

# from https://github.com/tanglang96/MDENAS/blob/master/run_darts_cifar.sh
# 599M 3.78M
MdeNAS = Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 3), ('sep_conv_3x3', 1), ('dil_conv_5x5', 3), ('max_pool_3x3', 4)], normal_concat=range(2, 6),
                  reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 3), ('skip_connect', 2), ('dil_conv_3x3', 3), ('sep_conv_5x5', 0)], reduce_concat=range(2, 6))

# 558M, 3.63M
PC_DARTS_cifar = Genotype(normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6),
                          reduce=[('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))

# from https://github.com/tanglang96/MDENAS/blob/master/run_darts_cifar.sh
# 532M, 3.43M
PDARTS = Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0),('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3',0), ('dil_conv_5x5', 4)], normal_concat=range(2, 6),
                  reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))

# from https://arxiv.org/abs/1812.09926
# 422M, 2.66
SNAS = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
                        ('skip_connect', 0), ('dil_conv_3x3', 1),
                        ('skip_connect', 1), ('skip_connect', 0), 
                        ('skip_connect',0),  ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
                reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1),
                 ('max_pool_3x3', 1), ('skip_connect', 2),
                 ('skip_connect', 2), ('max_pool_3x3', 1),
                 ('max_pool_3x3', 0), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))


"""Sparse"""														
# threshold = 0.85 for edge and weight=10 lr = 0.0025
# FLOPS: 373M Params: 2.83
FairDARTS_a = Genotype(normal=[('sep_conv_3x3', 2, 0), ('sep_conv_5x5', 2, 1), ('max_pool_3x3', 4, 0), ('sep_conv_3x3', 5, 0)], normal_concat=range(2, 6),
         reduce=[('max_pool_3x3', 2, 0), ('avg_pool_3x3', 2, 1), ('avg_pool_3x3', 3, 0), ('dil_conv_5x5', 3, 1), ('avg_pool_3x3', 4, 0), ('sep_conv_5x5', 4, 1),
                 ('skip_connect', 5, 0), ('skip_connect', 5, 1)], reduce_concat=range(2, 6))

# FLOPS: 536M Params: 3.88
FairDARTS_b = Genotype(normal=[('sep_conv_3x3', 2, 0), ('sep_conv_3x3', 2, 1), ('sep_conv_3x3', 3, 1), ('dil_conv_3x3', 4, 0), ('sep_conv_5x5', 4, 1), ('dil_conv_5x5', 5, 1)], normal_concat=range(2, 6),
reduce=[('skip_connect', 2, 0), ('dil_conv_3x3', 2, 1), ('skip_connect', 3, 0), ('dil_conv_3x3', 3, 1), ('max_pool_3x3', 4, 0), ('sep_conv_3x3', 4, 1), ('skip_connect', 5, 2), ('max_pool_3x3', 5, 0)], reduce_concat=range(2, 6))

# FLOPS: 400M Params: 2.59M
FairDARTS_c = Genotype(normal=[('max_pool_3x3', 2, 0), ('sep_conv_5x5', 2, 1), ('dil_conv_3x3', 4, 0), ('dil_conv_5x5', 4, 2), ('skip_connect', 5, 3), ('sep_conv_3x3', 5, 0)], normal_concat=range(2, 6),
                       reduce=[('dil_conv_3x3', 2, 1), ('dil_conv_5x5', 2, 0), ('dil_conv_3x3', 3, 0), ('sep_conv_3x3', 3, 1), ('sep_conv_5x5', 4, 0), ('sep_conv_5x5', 4, 3), ('sep_conv_5x5', 5, 0), ('skip_connect', 5, 1)], reduce_concat=range(2, 6))

# FLOPS: 532M Params: 3.84M
FairDARTS_d = Genotype(normal=[('sep_conv_3x3', 2, 0), ('sep_conv_5x5', 2, 1), ('dil_conv_3x3', 3, 1), ('max_pool_3x3', 3, 0), ('dil_conv_3x3', 4, 0), ('dil_conv_3x3', 4, 1), ('sep_conv_3x3', 5, 0), ('dil_conv_5x5', 5, 1)], normal_concat=range(2, 6),
reduce=[('max_pool_3x3', 2, 0), ('sep_conv_5x5', 2, 1), ('avg_pool_3x3', 3, 0), ('dil_conv_5x5', 3, 2), ('dil_conv_3x3', 4, 3), ('avg_pool_3x3', 4, 0), ('avg_pool_3x3', 5, 0), ('skip_connect', 5, 3)], reduce_concat=range(2, 6))

# FLOPS: 414M Params: 3.12M
FairDARTS_e = Genotype(normal=[('sep_conv_3x3', 2, 0), ('sep_conv_3x3', 2, 1), ('dil_conv_3x3', 4, 1), ('dil_conv_3x3', 4, 2), ('dil_conv_3x3', 5, 0), ('dil_conv_5x5', 5, 1)], normal_concat=range(2, 6),
reduce=[('max_pool_3x3', 2, 1), ('max_pool_3x3', 2, 0), ('max_pool_3x3', 3, 1), ('max_pool_3x3', 3, 0), ('sep_conv_5x5', 4, 1), ('max_pool_3x3', 4, 0), ('avg_pool_3x3', 5, 0), ('dil_conv_5x5', 5, 1)], reduce_concat=range(2, 6))

# FLOPS: 497M Params: 3.62M
FairDARTS_f = Genotype(normal=[('max_pool_3x3', 2, 0), ('sep_conv_3x3', 2, 1), ('dil_conv_3x3', 3, 1), ('sep_conv_5x5', 4, 1), ('sep_conv_3x3', 5, 0), ('sep_conv_3x3', 5, 1)], normal_concat=range(2, 6),
reduce=[('max_pool_3x3', 2, 0), ('max_pool_3x3', 2, 1), ('max_pool_3x3', 3, 0), ('dil_conv_3x3', 3, 1), ('dil_conv_3x3', 4, 2), ('max_pool_3x3', 4, 0), ('max_pool_3x3', 5, 0), ('sep_conv_3x3', 5, 1)], reduce_concat=range(2, 6))

# FLOPS: 453M Params: 3.375M
FairDARTS_g = Genotype(normal=[('sep_conv_3x3', 2, 0), ('sep_conv_3x3', 2, 1), ('skip_connect', 4, 3), ('sep_conv_5x5', 4, 1), ('dil_conv_3x3', 5, 0), ('sep_conv_3x3', 5, 1)], normal_concat=range(2, 6),
reduce=[('avg_pool_3x3', 2, 1), ('skip_connect', 2, 0), ('skip_connect', 3, 2), ('max_pool_3x3', 3, 1), ('sep_conv_5x5', 4, 3), ('max_pool_3x3', 4, 0), ('dil_conv_3x3', 5, 1), ('dil_conv_3x3', 5, 4)], reduce_concat=range(2, 6))


"""Batch size = 64"""
# FLOPS: 469M Params: 3.01M
DCO_SPARSE_BS_64 = Genotype(normal=[('sep_conv_3x3', 2, 0), ('dil_conv_3x3', 2, 1), ('sep_conv_5x5', 3, 0), ('dil_conv_3x3', 3, 1), ('max_pool_3x3', 4, 0), ('dil_conv_5x5', 5, 0)], normal_concat=range(2, 6),
                    reduce=[('skip_connect', 2, 0), ('dil_conv_5x5', 2, 1), ('sep_conv_5x5', 3, 0), ('sep_conv_5x5', 3, 2), ('sep_conv_5x5', 4, 3), ('avg_pool_3x3', 4, 0), ('dil_conv_3x3', 5, 0), ('dil_conv_5x5', 5, 1)], reduce_concat=range(2, 6))


DCO_EDGE_BS_64 = Genotype(normal=[('dil_conv_3x3', 2, 0), ('sep_conv_3x3', 2, 0), ('sep_conv_3x3', 2, 1), ('dil_conv_3x3', 2, 1), ('max_pool_3x3', 3, 0), ('sep_conv_5x5', 3, 0), ('dil_conv_3x3', 3, 1), ('max_pool_3x3', 4, 0), ('dil_conv_5x5', 5, 0)], normal_concat=range(2, 6),
         reduce=[('sep_conv_5x5', 2, 0), ('skip_connect', 2, 0), ('dil_conv_5x5', 2, 1), ('skip_connect', 2, 1), ('skip_connect', 3, 0), ('max_pool_3x3', 3, 0), ('max_pool_3x3', 3, 1), ('sep_conv_3x3', 3, 1), ('skip_connect', 3, 2), ('sep_conv_5x5', 3, 2), ('max_pool_3x3', 4, 0), ('avg_pool_3x3', 4, 0), ('dil_conv_5x5', 4, 1), ('sep_conv_3x3', 4, 1), ('sep_conv_5x5', 4, 2), ('skip_connect', 4, 2), ('skip_connect', 4, 3), ('sep_conv_5x5', 4, 3), ('max_pool_3x3', 5, 0), ('avg_pool_3x3', 5, 0), ('dil_conv_5x5', 5, 1), ('dil_conv_3x3', 5, 1), ('skip_connect', 5, 2), ('sep_conv_5x5', 5, 2), ('sep_conv_5x5', 5, 3), ('dil_conv_5x5', 5, 3), ('dil_conv_3x3', 5, 4), ('skip_connect', 5, 4)], reduce_concat=range(2, 6))

DARTS = FairDARTS_a
