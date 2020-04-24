"""
Still in development.
"""
import math
import torch
from torch import nn
from models.layers_transposed import Conv, Hourglass, SELayer, Backbone
from models.loss_model_parallel import MultiTaskLossParallel
from models.loss_model import MultiTaskLoss
from torchvision.models import densenet


class Merge(nn.Module):
    """Change the channel dimension of the input tensor"""

    def __init__(self, x_dim, y_dim, bn=False):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=bn)

    def forward(self, x):
        return self.conv(x)


class Features(nn.Module):
    """Input: feature maps produced by hourglass block
       Return: 5 different scales of feature maps, 128*128, 64*64, 32*32, 16*16, 8*8"""

    def __init__(self, inp_dim, increase=128, bn=False):
        super(Features, self).__init__()
        # Regress 5 different scales of heatmaps per stack
        self.before_regress = nn.ModuleList(
            [nn.Sequential(Conv(inp_dim + i * increase, inp_dim, 3, bn=bn, dropout=False),
                           Conv(inp_dim, inp_dim, 3, bn=bn, dropout=False),
                           # ##################### Channel Attention layer  #####################
                           SELayer(inp_dim),
                           ) for i in range(5)])

    def forward(self, fms):
        assert len(fms) == 5, "hourglass output {} tensors,but 5 scale heatmaps are supervised".format(len(fms))
        return [self.before_regress[i](fms[i]) for i in range(5)]


class PoseNet(nn.Module):
    def __init__(self, num_stages, inp_dim, oup_dim, bn=False, increase=128, init_weights=True, **kwargs):
        """
        Pack or initialize the trainable parameters of the network
        :param num_stages: number of stack
        :param inp_dim: input tensor channels fed into the hourglass block
        :param oup_dim: channels of regressed feature maps
        :param bn: use batch normalization
        :param increase: increased channels once down-sampling
        :param kwargs:
        """
        super(PoseNet, self).__init__()
        # <512x512>
        # nstack: 4
        # inp_dim: 256
        # oup_dim: 50
        # bn: True
        # increase: 128
        # init_weights: True
        self.pre = Backbone()  # > 256: It doesn't affect the results regardless of which self.pre is used
        self.hourglass = nn.ModuleList()
        self.features = nn.ModuleList()
        self.outs = nn.ModuleList()
        self.merge_features = nn.ModuleList()
        self.merge_preds = nn.ModuleList()
        for t in range(num_stages):  # 4
            self.hourglass.append(Hourglass(depth=4, nFeat=inp_dim, increase=increase, bn=bn))
            self.features.append(Features(inp_dim=inp_dim, increase=increase, bn=bn))
            # TODO: change the outs layers, Conv(inp_dim + j * increase, oup_dim, 1, relu=False, bn=False)
            self.outs.append(nn.ModuleList([Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for _ in range(5)]))

            # TODO: change the merge layers, Merge(inp_dim + j * increase, inp_dim + j * increase)
            if t < num_stages - 1:
                self.merge_features.append(nn.ModuleList([Merge(inp_dim, inp_dim + j * increase, bn=bn) for j in range(5)]))
                self.merge_preds.append(nn.ModuleList([Merge(oup_dim, inp_dim + j * increase, bn=bn) for j in range(5)]))
        self.num_stages = num_stages
        self.num_scales = 5
        if init_weights:
            self._initialize_weights()

    def forward(self, imgs):
        # Input Tensor: a batch of images within [0,1], shape=(N, H, W, C). Pre-processing was done in data generator
        x = imgs.permute(0, 3, 1, 2)  # Permute the dimensions of images to (N, C, H, W)
        x = self.pre(x)
        pred = []
        feat_caches = [[]] * self.num_scales
        # loop over stack
        for t, hg, se_block, out_blocks in \
                zip(range(self.num_stages), self.hourglass, self.features, self.outs):  # > 4
            preds_instack = []
            # -> (0:256, 1:384, 2:512, 3:640, 4:786)
            hg_feats = hg(x)  # -> 5 scales of feature maps

            if t == 0:  # cache for smaller feature maps produced by hourglass block
                feat_caches = [torch.zeros_like(hg_feats[s]) for s in range(self.num_scales)]
            else:
                hg_feats = [hg_feats[s] + feat_caches[s] for s in range(self.num_scales)]

            feats_instack = se_block(hg_feats)  # > 5 convs -> 5 scales

            for s, feats, head in zip(range(self.num_scales), feats_instack, out_blocks):  # handle 5 scales of heatmaps
                # > outs/bottlenecks: 1x1 conv layer * 5
                pred_out = head(feats)
                preds_instack.append(pred_out)

                if t != self.num_stages - 1:
                    cache = self.merge_preds[t][s](pred_out) + self.merge_features[t][s](feats)
                    if s == 0:
                        x = x + cache
                    feat_caches[s] = cache
            pred.append(preds_instack)
        # returned list shape: [nstack * [batch*128*128, batch*64*64, batch*32*32, batch*16*16, batch*8*8]]z
        return pred

    def _initialize_weights(self):
        for m in self.modules():
            # 卷积的初始化方法
            if isinstance(m, nn.Conv2d):
                # TODO: 使用正态分布进行初始化（0, 0.01) 网络权重看看
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # He kaiming 初始化, 方差为2/n. math.sqrt(2. / n) 或者直接使用现成的nn.init中的函数。在这里会梯度爆炸
                m.weight.data.normal_(0, 0.001)    # # math.sqrt(2. / n)
                # torch.nn.init.kaiming_normal_(m.weight)
                # bias都初始化为0
                if m.bias is not None:  # 当有BN层时，卷积层Con不加bias！
                    m.bias.data.zero_()
            # batchnorm使用全1初始化 bias全0
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)  # todo: 0.001?
                # m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class Network(torch.nn.Module):
    """
    Wrap the network module as well as the loss module on all GPUs to balance the computation among GPUs.
    """
    def __init__(self, opt, config, bn=False, dist=False, swa=False):
        super(Network, self).__init__()
        self.posenet = PoseNet(num_stages=opt.nstack,
                               inp_dim=opt.hourglass_inp_dim,
                               oup_dim=config.num_layers,
                               bn=bn,
                               increase=opt.increase)
        # If we use train_parallel, we implement the parallel loss . And if we use train_distributed,
        # we should use single process loss because each process on these 4 GPUs  is independent
        self.criterion = MultiTaskLoss(opt, config) if dist else MultiTaskLossParallel(opt, config)
        self.swa = swa

    def forward(self, input_all):
        # Batch will be divided and Parallel Model will call this forward on every GPU
        inp_imgs = input_all[0]
        target_tuple = input_all[1:]
        output_tuple = self.posenet(inp_imgs)

        if not self.training:  # testing mode
            loss = self.criterion(output_tuple, target_tuple)
            # output will be concatenated  along batch channel automatically after the parallel model return
            return output_tuple, loss

        else:  # training mode
            if not self.swa:
                loss = self.criterion(output_tuple, target_tuple)

                # output will be concatenated  along batch channel automatically after the parallel model return
                return loss
            else:
                return output_tuple


class NetworkEval(torch.nn.Module):
    """
    Wrap the network module as well as the loss module on all GPUs to balance the computation among GPUs.
    """
    def __init__(self, opt, config, bn=False):
        super(NetworkEval, self).__init__()
        self.posenet = PoseNet(opt.nstack, opt.hourglass_inp_dim, config.num_layers, bn=bn, init_weights=False,
                               increase=opt.increase)

    def forward(self, inp_imgs):
        # Batch will be divided and Parallel Model will call this forward on every GPU
        output_tuple = self.posenet(inp_imgs)

        if not self.training:
            # output will be concatenated  along batch channel automatically after the parallel model return
            return output_tuple
        else:
            # output will be concatenated  along batch channel automatically after the parallel model return
            raise ValueError('\nOnly eval mode is available!!')


if __name__ == '__main__':
    from time import time

    pose = PoseNet(4, 256, 54, bn=True)  # .cuda()
    for param in pose.parameters():
        if param.requires_grad:
            print('param autograd')
            break

    t0 = time()
    input = torch.rand(1, 128, 128, 3)  # .cuda()
    print(pose)
    output = pose(input)  # type: torch.Tensor

    output[0][0].sum().backward()

    t1 = time()
    print('********** Consuming Time is: {} second  **********'.format(t1 - t0))

    # #
    # import torch.onnx
    #
    # pose = PoseNet(4, 256, 34)
    # dummy_input = torch.randn(1, 512, 512, 3)
    # torch.onnx.export(pose, dummy_input, "posenet.onnx")  # netron --host=localhost
