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

DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5],
                    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)],
                    reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)],
                    normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])


"""----Experiment 1: """
# Weight:  Similar
DARTS_EXP1_RE1 = Genotype(normal=[('skip_connect', 0, 0.2386), ('sep_conv_3x3', 1, 0.2340),
                                  ('dil_conv_3x3', 2, 0.3028), ('sep_conv_3x3', 0, 0.2745),
                                  ('dil_conv_3x3', 3, 0.3016), ('skip_connect', 0, 0.2737),
                                  ('dil_conv_5x5', 4, 0.3499), ('dil_conv_3x3', 3, 0.2212)], normal_concat=range(2, 6),
                         reduce= [('avg_pool_3x3', 0, 0.2173), ('skip_connect', 1, 0.1925),
                                  ('skip_connect', 2, 0.2302), ('max_pool_3x3', 0, 0.1860),
                                  ('skip_connect', 2, 0.3068), ('skip_connect', 3, 0.2695),
                                  ('skip_connect', 2, 0.2690), ('avg_pool_3x3', 0, 0.2323)], reduce_concat=range(2, 6))

# Random Large Weight1 and 2 (0.9, 0.99): Similar
DARTS_EXP1_RE2 = Genotype(normal=[('skip_connect', 0, 0.9), ('sep_conv_3x3', 1, 0.1),
                                  ('dil_conv_3x3', 2, 0.1), ('sep_conv_3x3', 0, 0.9),
                                  ('dil_conv_3x3', 3, 0.1), ('skip_connect', 0, 0.9),
                                  ('dil_conv_5x5', 4, 0.9), ('dil_conv_3x3', 3, 0.1)], normal_concat=range(2, 6),
                         reduce= [('avg_pool_3x3', 0, 0.1), ('skip_connect', 1, 0.9),
                                  ('skip_connect', 2, 0.9), ('max_pool_3x3', 0, 0.1),
                                  ('skip_connect', 2, 0.9), ('skip_connect', 3, 0.1),
                                  ('skip_connect', 2, 0.9), ('avg_pool_3x3', 0, 0.1)], reduce_concat=range(2, 6))

# Follow Large weight 1: Bad Result: 94.97
DARTS_EXP1_RE3 = Genotype(normal=[('skip_connect', 0, 0.99), ('sep_conv_3x3', 1, 0.01),
                                  ('dil_conv_3x3', 2, 0.99), ('sep_conv_3x3', 0, 0.01),
                                  ('dil_conv_3x3', 3, 0.99), ('skip_connect', 0, 0.01),
                                  ('dil_conv_5x5', 4, 0.99), ('dil_conv_3x3', 3, 0.01)], normal_concat=range(2, 6),
                          reduce=[('avg_pool_3x3', 0, 0.99), ('skip_connect', 1, 0.01),
                                  ('skip_connect', 2, 0.99), ('max_pool_3x3', 0, 0.01),
                                  ('skip_connect', 2, 0.99), ('skip_connect', 3, 0.01),
                                  ('skip_connect', 2, 0.99), ('avg_pool_3x3', 0, 0.01)], reduce_concat=range(2, 6))

# Follow Large weight2:
DARTS_EXP1_RE4 = Genotype(normal=[('skip_connect', 0, 0.9), ('sep_conv_3x3', 1, 0.1),
                                  ('dil_conv_3x3', 2, 0.9), ('sep_conv_3x3', 0, 0.1),
                                  ('dil_conv_3x3', 3, 0.9), ('skip_connect', 0, 0.1),
                                  ('dil_conv_5x5', 4, 0.9), ('dil_conv_3x3', 3, 0.1)], normal_concat=range(2, 6),
                          reduce=[('avg_pool_3x3', 0, 0.9), ('skip_connect', 1, 0.1),
                                  ('skip_connect', 2, 0.9), ('max_pool_3x3', 0, 0.1),
                                  ('skip_connect', 2, 0.9), ('skip_connect', 3, 0.1),
                                  ('skip_connect', 2, 0.9), ('avg_pool_3x3', 0, 0.1)], reduce_concat=range(2, 6))


# Invert Large weight: Similar
DARTS_EXP1_RE5 = Genotype(normal=[('skip_connect', 0, 0.01), ('sep_conv_3x3', 1, 0.99),
                                  ('dil_conv_3x3', 2, 0.01), ('sep_conv_3x3', 0, 0.99),
                                  ('dil_conv_3x3', 3, 0.01), ('skip_connect', 0, 0.99),
                                  ('dil_conv_5x5', 4, 0.01), ('dil_conv_3x3', 3, 0.99)], normal_concat=range(2, 6),
                          reduce=[('avg_pool_3x3', 0, 0.01), ('skip_connect', 1, 0.99),
                                  ('skip_connect', 2, 0.01), ('max_pool_3x3', 0, 0.99),
                                  ('skip_connect', 2, 0.01), ('skip_connect', 3, 0.99),
                                  ('skip_connect', 2, 0.01), ('avg_pool_3x3', 0, 0.99)], reduce_concat=range(2, 6))

"""----Experiment 2: """

DARTS_EXP2_RE1 = Genotype(normal=[
                                ('sep_conv_3x3', 1, 0.2777), ('sep_conv_3x3', 0, 0.2562),
                                ('dil_conv_3x3', 1, 0.2619), ('dil_conv_3x3', 2, 0.2521),
                                ('skip_connect', 0, 0.3661), ('skip_connect', 2, 0.2936),
                                ('dil_conv_3x3', 4, 0.2966), ('dil_conv_5x5', 3, 0.2533)], normal_concat=range(2, 6),
                                reduce=[
                                ('max_pool_3x3', 0, 0.2234), ('sep_conv_3x3', 1, 0.1629),
                                ('max_pool_3x3', 0, 0.2241), ('skip_connect', 2, 0.2205),
                                ('skip_connect', 2, 0.2800), ('max_pool_3x3', 0, 0.2083),
                                ('skip_connect', 2, 0.2475), ('skip_connect', 4, 0.2225)
                                ], reduce_concat=range(2, 6))

# Random Large Weight1 and 2
DARTS_EXP2_RE2 = Genotype(normal=[
                                ('sep_conv_3x3', 1, 0.99), ('sep_conv_3x3', 0, 0.01),
                                ('dil_conv_3x3', 1, 0.01), ('dil_conv_3x3', 2, 0.99),
                                ('skip_connect', 0, 0.99), ('skip_connect', 2, 0.01),
                                ('dil_conv_3x3', 4, 0.01), ('dil_conv_5x5', 3, 0.99)], normal_concat=range(2, 6),
                                reduce=[
                                ('max_pool_3x3', 0, 0.99), ('sep_conv_3x3', 1, 0.01),
                                ('max_pool_3x3', 0, 0.01), ('skip_connect', 2, 0.99),
                                ('skip_connect', 2, 0.99), ('max_pool_3x3', 0, 0.01),
                                ('skip_connect', 2, 0.99), ('skip_connect', 4, 0.01)
                                ], reduce_concat=range(2, 6))

# Follow Large weight 1
DARTS_EXP2_RE3 = Genotype(normal=[
                                ('sep_conv_3x3', 1, 0.99), ('sep_conv_3x3', 0, 0.01),
                                ('dil_conv_3x3', 1, 0.99), ('dil_conv_3x3', 2, 0.01),
                                ('skip_connect', 0, 0.99), ('skip_connect', 2, 0.01),
                                ('dil_conv_3x3', 4, 0.99), ('dil_conv_5x5', 3, 0.01)], normal_concat=range(2, 6),
                                reduce=[
                                ('max_pool_3x3', 0, 0.99), ('sep_conv_3x3', 1, 0.01),
                                ('max_pool_3x3', 0, 0.99), ('skip_connect', 2, 0.01),
                                ('skip_connect', 2, 0.99), ('max_pool_3x3', 0, 0.01),
                                ('skip_connect', 2, 0.99), ('skip_connect', 4, 0.01)
                                ], reduce_concat=range(2, 6))
# Follow Large weight 2
DARTS_EXP2_RE4 = Genotype(normal=[
                                ('sep_conv_3x3', 1, 0.9), ('sep_conv_3x3', 0, 0.1),
                                ('dil_conv_3x3', 1, 0.9), ('dil_conv_3x3', 2, 0.1),
                                ('skip_connect', 0, 0.9), ('skip_connect', 2, 0.1),
                                ('dil_conv_3x3', 4, 0.9), ('dil_conv_5x5', 3, 0.1)], normal_concat=range(2, 6),
                                reduce=[
                                ('max_pool_3x3', 0, 0.9), ('sep_conv_3x3', 1, 0.1),
                                ('max_pool_3x3', 0, 0.9), ('skip_connect', 2, 0.1),
                                ('skip_connect', 2, 0.9), ('max_pool_3x3', 0, 0.1),
                                ('skip_connect', 2, 0.9), ('skip_connect', 4, 0.1)
                                ], reduce_concat=range(2, 6))

# Invert Large weight
DARTS_EXP2_RE5 = Genotype(normal=[
                                ('sep_conv_3x3', 1, 0.1), ('sep_conv_3x3', 0, 0.9),
                                ('dil_conv_3x3', 1, 0.1), ('dil_conv_3x3', 2, 0.9),
                                ('skip_connect', 0, 0.1), ('skip_connect', 2, 0.9),
                                ('dil_conv_3x3', 4, 0.1), ('dil_conv_5x5', 3, 0.9)], normal_concat=range(2, 6),
                                reduce=[
                                ('max_pool_3x3', 0, 0.1), ('sep_conv_3x3', 1, 0.9),
                                ('max_pool_3x3', 0, 0.1), ('skip_connect', 2, 0.9),
                                ('skip_connect', 2, 0.1), ('max_pool_3x3', 0, 0.9),
                                ('skip_connect', 2, 0.1), ('skip_connect', 4, 0.9)
                                ], reduce_concat=range(2, 6))

"""========== Random not control the flops ============="""

# FLOPS: 433M, Params: 3.20M
DARTS_ID2_1 = Genotype(normal=[('skip_connect', 1), ('sep_conv_5x5', 0), ('skip_connect', 1), ('dil_conv_5x5', 0), ('avg_pool_3x3', 1), ('dil_conv_3x3', 0), ('max_pool_3x3', 0), ('sep_conv_3x3', 3)], normal_concat=range(2, 6),
                        reduce=[('dil_conv_3x3', 1), ('dil_conv_5x5', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 2), ('skip_connect', 0), ('dil_conv_3x3', 1), ('avg_pool_3x3', 4), ('sep_conv_5x5', 1)], reduce_concat=range(2, 6))

# FLOPS: 476, Params: 3.48
DARTS_ID2_2 = Genotype(normal=[('skip_connect', 1), ('skip_connect', 0), ('dil_conv_3x3', 0), ('sep_conv_5x5', 2), ('dil_conv_5x5', 1), ('sep_conv_5x5', 3), ('dil_conv_3x3', 3), ('max_pool_3x3', 4)], normal_concat=range(2, 6),
                       reduce=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2), ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))

# FLOPS: 378, Params: 2.81
DARTS_ID2_3 = Genotype(normal=[('avg_pool_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 0), ('dil_conv_3x3', 2), ('sep_conv_5x5', 2), ('skip_connect', 3), ('avg_pool_3x3', 1), ('avg_pool_3x3', 4)], normal_concat=range(2, 6),
                       reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('avg_pool_3x3', 1), ('sep_conv_3x3', 2), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))

# FLOPS: 407M, Params: 3.10
DARTS_ID2_4 = Genotype(normal=[('skip_connect', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 0), ('avg_pool_3x3', 2), ('dil_conv_5x5', 0), ('dil_conv_3x3', 3), ('max_pool_3x3', 1), ('skip_connect', 4)], normal_concat=range(2, 6),
                       reduce=[('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))

# FLOPS: 442M, Params: 3.18
DARTS_ID2_5 = Genotype(normal=[('dil_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_5x5', 0), ('skip_connect', 2), ('sep_conv_5x5', 3), ('dil_conv_5x5', 2), ('avg_pool_3x3', 0), ('max_pool_3x3', 3)], normal_concat=range(2, 6),
                       reduce=[('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 2), ('max_pool_3x3', 0), ('avg_pool_3x3', 2), ('dil_conv_3x3', 0), ('dil_conv_3x3', 2)], reduce_concat=range(2, 6))

# FLOPS: 517M, Paramss: 3.76M
DARTS_ID2_6 = Genotype(normal=[('sep_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('avg_pool_3x3', 2), ('skip_connect', 0), ('sep_conv_3x3', 0), ('dil_conv_5x5', 3)], normal_concat=range(2, 6),
                        reduce=[('avg_pool_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2), ('skip_connect', 1), ('avg_pool_3x3', 2), ('skip_connect', 1), ('sep_conv_3x3', 4), ('dil_conv_3x3', 1)], reduce_concat=range(2, 6))

# FLOPS: 480M, Params: 3.46M
DARTS_ID2_7 = Genotype(normal=[('skip_connect', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('skip_connect', 4), ('sep_conv_5x5', 0)], normal_concat=range(2, 6),
                        reduce=[('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 1), ('max_pool_3x3', 3), ('dil_conv_5x5', 2), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))

"""===============Random FLOPS >= 500M"""

# FLOPS: 518M, Parmas 3.81
DARTS_ID2_FPT_1 = Genotype(normal=[('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('dil_conv_3x3', 2), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 2), ('dil_conv_5x5', 1)], normal_concat=range(2, 6),
                           reduce=[('skip_connect', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 2), ('dil_conv_5x5', 0), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))

# FLOPS: 519M, Parmas 3.72
DARTS_ID2_FPT_2 = Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('skip_connect', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 1), ('dil_conv_5x5', 4)], normal_concat=range(2, 6),
                           reduce=[('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('avg_pool_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 1), ('skip_connect', 3), ('skip_connect', 0), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))

# FLOPS: 505M, Parmas 3.64
DARTS_ID2_FPT_3 = Genotype(normal=[('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 1), ('skip_connect', 0), ('skip_connect', 2), ('max_pool_3x3', 1), ('sep_conv_5x5', 3), ('sep_conv_3x3', 2)], normal_concat=range(2, 6),
                           reduce=[('sep_conv_5x5', 0), ('skip_connect', 1), ('max_pool_3x3', 1), ('sep_conv_5x5', 2), ('dil_conv_5x5', 3), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 2)], reduce_concat=range(2, 6))

# FLOPS: 506M, Parmas 3.64
DARTS_ID2_FPT_4 = Genotype(normal=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 2), ('dil_conv_5x5', 0), ('sep_conv_5x5', 3), ('sep_conv_5x5', 0), ('skip_connect', 3), ('skip_connect', 0)], normal_concat=range(2, 6),
                           reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 3), ('sep_conv_5x5', 2), ('avg_pool_3x3', 1), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))

# FLOPS: 543M, Parmas 3.87
DARTS_ID2_FPT_5 = Genotype(normal=[('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('skip_connect', 0), ('sep_conv_5x5', 3), ('sep_conv_5x5', 0), ('avg_pool_3x3', 1), ('skip_connect', 0)], normal_concat=range(2, 6),
                           reduce=[('max_pool_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 1), ('skip_connect', 0)], reduce_concat=range(2, 6))

# FLOPS: 502M, Parmas 3.64
DARTS_ID2_FPT_6 = Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_3x3', 3), ('max_pool_3x3', 0), ('skip_connect', 2)], normal_concat=range(2, 6),
                           reduce=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 2), ('skip_connect', 0), ('dil_conv_5x5', 0), ('sep_conv_5x5', 3), ('sep_conv_3x3', 1), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))

# FLOPS: 553M, Parmas 3.9
DARTS_ID2_FPT_7 = Genotype(normal=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('skip_connect', 1), ('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('skip_connect', 4), ('sep_conv_5x5', 0)], normal_concat=range(2, 6),
                           reduce=[('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('sep_conv_5x5', 0), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 3), ('skip_connect', 0)], reduce_concat=range(2, 6))

"""================Add Noise in Identity ================="""

# Flops: 553M, Params: 3.96
# drop_path 0.05
NOISEDARTS_50_1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 2),
                                   ('sep_conv_3x3', 1), ('dil_conv_5x5', 3), ('dil_conv_5x5', 4), ('dil_conv_3x3', 3)], normal_concat=range(2, 6),
                         reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2),
                                 ('skip_connect', 1), ('max_pool_3x3', 0), ('skip_connect', 3), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))

# flops: 522m, Params: 3.78
NOISEDARTS_50_2 = Genotype(normal=[('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2), ('dil_conv_3x3', 1),
                                   ('sep_conv_3x3', 1), ('dil_conv_5x5', 3), ('dil_conv_3x3', 3), ('dil_conv_5x5', 4)], normal_concat=range(2, 6),
                           reduce=[('dil_conv_5x5', 1), ('skip_connect', 0), ('max_pool_3x3', 0), ('dil_conv_3x3', 2),
                                   ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))

# flops: 489m, Params: 3.5
NOISEDARTS_50_3 = Genotype(normal=[('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('dil_conv_5x5', 4), ('dil_conv_5x5', 1)], normal_concat=range(2, 6),
                           reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2), ('avg_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2)], reduce_concat=range(2, 6))


DARTS = DARTS_V2
