from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Dropout, Sequential, Module
from helpers import get_blocks, Flatten, bottleneck_IR, bottleneck_IR_SE, l2_norm

"""
Modified Backbone implementation from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch)
"""


class Backbone(Module):
    def __init__(self, input_size, num_layers, mode='ir', drop_ratio=0.4, affine=True):
        super().__init__()
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_layer = Sequential(BatchNorm2d(512),
                           Dropout(drop_ratio),
                           Flatten(),
                           Linear(512 * 7 * 7, 512),
                           BatchNorm1d(512, affine=affine))
        self.body = nn.ModuleList([])
        for block in blocks:
            for bottleneck in block:
                self.body.append(unet_module(bottleneck.in_channel,
                                 bottleneck.depth,
                                 bottleneck.stride))
        self.pool = nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = nn.AdaptiveAvgPool2d((112, 112))

    def forward(self, x):
        x = self.pool(x)
        x = x[:, :, 35:223, 32:220]
        x = self.face_pool(x)
        
        x = self.input_layer(x)
        samples = ()
        for i, body_block in enumerate(self.body):
            x = body_block(x)
            samples = samples + (x,)
        xx = self.output_layer(x)
        return l2_norm(x), samples


def IR_50(input_size):
    """Constructs a ir-50 model."""
    model = Backbone(input_size, num_layers=50, mode='ir', drop_ratio=0.4, affine=False)
    return model


def IR_101(input_size):
    """Constructs a ir-101 model."""
    model = Backbone(input_size, num_layers=100, mode='ir', drop_ratio=0.4, affine=False)
    return model


def IR_152(input_size):
    """Constructs a ir-152 model."""
    model = Backbone(input_size, num_layers=152, mode='ir', drop_ratio=0.4, affine=False)
    return model


def IR_SE_50(input_size):
    """Constructs a ir_se-50 model."""
    model = Backbone(input_size, num_layers=50, mode='ir_se', drop_ratio=0.4, affine=False)
    return model


def IR_SE_101(input_size):
    """Constructs a ir_se-101 model."""
    model = Backbone(input_size, num_layers=100, mode='ir_se', drop_ratio=0.4, affine=False)
    return model


def IR_SE_152(input_size):
    """Constructs a ir_se-152 model."""
    model = Backbone(input_size, num_layers=152, mode='ir_se', drop_ratio=0.4, affine=False)
    return model
