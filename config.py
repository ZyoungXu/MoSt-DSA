from functools import partial
import torch.nn as nn

from model import encoder
from model import decoder


def init_model_config(F=32, lambda_range='local', depth=[2, 2, 2, 4]):
    return {
        'embed_dims':[F, 2*F, 4*F, 8*F],
        'motion_dims':[0, 0, 0, 8*F//depth[-1]],
        'num_heads':[4],
        'mlp_ratios':[4],
        'lambda_global_or_local': lambda_range,
        'lambda_dim_k':16,
        'lambda_dim_u':1,
        'lambda_n':32,
        'lambda_r':15,
        'norm_layer':partial(nn.LayerNorm, eps=1e-6),
        'depths':depth,
    }, {
        'embed_dims':[F, 2*F, 4*F, 8*F],
        'motion_dims':[0, 0, 0, 8*F//depth[-1]],
        'depths':depth,
        'scales':[4, 8],
        'hidden_dims':[4*F],
        'c':F
    }


MODEL_CONFIG = {
    'LOGNAME': 'Hust-Model',
    'MODEL_TYPE': (encoder, decoder),
    'MODEL_ARCH': init_model_config(
        F = 32,
        lambda_range='local',
        depth = [2, 2, 2, 4]
    )
}
