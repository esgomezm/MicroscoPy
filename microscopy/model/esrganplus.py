# Model based on: https://github.com/ncarraz/ESRGANpluss

import numpy as np
import os

from skimage import transform
from skimage import filters
from skimage import io
from skimage.util import random_noise
from skimage.util import img_as_ubyte

from scipy.ndimage.interpolation import zoom as npzoom

from collections import OrderedDict

import math
import functools
import logging

logger = logging.getLogger("base")

from skimage.metrics import structural_similarity, peak_signal_noise_ratio

import torch
from torch import nn
from torch.nn import init
from torch.utils.data import Dataset, DataLoader

import torchvision

from pytorch_lightning.core import LightningModule

from ..datasets import PytorchDataset, ToTensor
from ..datasets import RandomHorizontalFlip, RandomVerticalFlip, RandomRotate
from ..optimizer_scheduler_utils import select_optimizer, select_lr_schedule

from matplotlib import pyplot as plt

####################
# Basic blocks
####################


def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == "relu":
        layer = nn.ReLU(inplace)
    elif act_type == "leakyrelu":
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == "prelu":
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(
            "activation layer [{:s}] is not found".format(act_type)
        )
    return layer


def norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == "batch":
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == "instance":
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError(
            "normalization layer [{:s}] is not found".format(norm_type)
        )
    return layer


def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == "reflect":
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == "replicate":
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError(
            "padding layer [{:s}] is not implemented".format(pad_type)
        )
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


class ShortcutBlock(nn.Module):
    # Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = "Identity + \n|"
        modstr = self.sub.__repr__().replace("\n", "\n|")
        tmpstr = tmpstr + modstr
        return tmpstr


def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError("sequential does not support OrderedDict input.")
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1, is_relative_detach=False):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0, dtype=torch.float).to(torch.device("cuda"))

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = (
                self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            )
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


def conv_block(
    in_nc,
    out_nc,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    bias=True,
    pad_type="zero",
    norm_type=None,
    act_type="relu",
    mode="CNA",
):
    """
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    """
    assert mode in ["CNA", "NAC", "CNAC"], "Wong conv mode [{:s}]".format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != "zero" else None
    padding = padding if pad_type == "zero" else 0

    c = nn.Conv2d(
        in_nc,
        out_nc,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        groups=groups,
    )
    a = act(act_type) if act_type else None
    if "CNA" in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == "NAC":
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


####################
# Useful blocks
####################


class ResidualDenseBlock_5C(nn.Module):
    """
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(
        self,
        nc,
        kernel_size=3,
        gc=32,
        stride=1,
        bias=True,
        pad_type="zero",
        norm_type=None,
        act_type="leakyrelu",
        mode="CNA",
        gaussian_noise=True,
    ):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.noise = GaussianNoise() if gaussian_noise else None
        self.conv1x1 = conv1x1(nc, gc)
        self.conv1 = conv_block(
            nc,
            gc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        self.conv2 = conv_block(
            nc + gc,
            gc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        self.conv3 = conv_block(
            nc + 2 * gc,
            gc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        self.conv4 = conv_block(
            nc + 3 * gc,
            gc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        if mode == "CNA":
            last_act = None
        else:
            last_act = act_type
        self.conv5 = conv_block(
            nc + 4 * gc,
            nc,
            3,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=last_act,
            mode=mode,
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x2 = x2 + self.conv1x1(x)
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x4 = x4 + x2
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return self.noise(x5.mul(0.2) + x)


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    """

    def __init__(
        self,
        nc,
        kernel_size=3,
        gc=32,
        stride=1,
        bias=True,
        pad_type="zero",
        norm_type=None,
        act_type="leakyrelu",
        mode="CNA",
    ):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(
            nc, kernel_size, gc, stride, bias, pad_type, norm_type, act_type, mode
        )
        self.RDB2 = ResidualDenseBlock_5C(
            nc, kernel_size, gc, stride, bias, pad_type, norm_type, act_type, mode
        )
        self.RDB3 = ResidualDenseBlock_5C(
            nc, kernel_size, gc, stride, bias, pad_type, norm_type, act_type, mode
        )

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(0.2) + x


####################
# Upsampler
####################


def pixelshuffle_block(
    in_nc,
    out_nc,
    upscale_factor=2,
    kernel_size=3,
    stride=1,
    bias=True,
    pad_type="zero",
    norm_type=None,
    act_type="relu",
):
    """
    Pixel shuffle layer
    (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
    Neural Network, CVPR17)
    """
    conv = conv_block(
        in_nc,
        out_nc * (upscale_factor**2),
        kernel_size,
        stride,
        bias=bias,
        pad_type=pad_type,
        norm_type=None,
        act_type=None,
    )
    pixel_shuffle = nn.PixelShuffle(upscale_factor)

    n = norm(norm_type, out_nc) if norm_type else None
    a = act(act_type) if act_type else None
    return sequential(conv, pixel_shuffle, n, a)


def upconv_blcok(
    in_nc,
    out_nc,
    upscale_factor=2,
    kernel_size=3,
    stride=1,
    bias=True,
    pad_type="zero",
    norm_type=None,
    act_type="relu",
    mode="nearest",
):
    # Up conv
    # described in https://distill.pub/2016/deconv-checkerboard/
    upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv = conv_block(
        in_nc,
        out_nc,
        kernel_size,
        stride,
        bias=bias,
        pad_type=pad_type,
        norm_type=norm_type,
        act_type=act_type,
    )
    return sequential(upsample, conv)


class RRDBNet(nn.Module):
    def __init__(
        self,
        in_nc,
        out_nc,
        nf,
        nb,
        gc=32,
        upscale=4,
        norm_type=None,
        act_type="leakyrelu",
        mode="CNA",
        upsample_mode="upconv",
    ):
        super(RRDBNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        rb_blocks = [
            RRDB(
                nf,
                kernel_size=3,
                gc=32,
                stride=1,
                bias=True,
                pad_type="zero",
                norm_type=norm_type,
                act_type=act_type,
                mode="CNA",
            )
            for _ in range(nb)
        ]
        LR_conv = conv_block(
            nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode
        )

        if upsample_mode == "upconv":
            upsample_block = upconv_blcok
        elif upsample_mode == "pixelshuffle":
            upsample_block = pixelshuffle_block
        else:
            raise NotImplementedError(
                "upsample mode [{:s}] is not found".format(upsample_mode)
            )
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [
                upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)
            ]
        HR_conv0 = conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = sequential(
            fea_conv,
            ShortcutBlock(sequential(*rb_blocks, LR_conv)),
            *upsampler,
            HR_conv0,
            HR_conv1
        )

    def forward(self, x):
        x = self.model(x)
        return x


# VGG style Discriminator with input size 128*128
class Discriminator_VGG_128(nn.Module):
    def __init__(
        self, in_nc, base_nf, norm_type="batch", act_type="leakyrelu", mode="CNA"
    ):
        super(Discriminator_VGG_128, self).__init__()
        # features
        # hxw, c
        # 128, 64
        conv0 = conv_block(
            in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, mode=mode
        )
        conv1 = conv_block(
            base_nf,
            base_nf,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        # 64, 64
        conv2 = conv_block(
            base_nf,
            base_nf * 2,
            kernel_size=3,
            stride=1,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        conv3 = conv_block(
            base_nf * 2,
            base_nf * 2,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        # 32, 128
        conv4 = conv_block(
            base_nf * 2,
            base_nf * 4,
            kernel_size=3,
            stride=1,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        conv5 = conv_block(
            base_nf * 4,
            base_nf * 4,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        # 16, 256
        conv6 = conv_block(
            base_nf * 4,
            base_nf * 8,
            kernel_size=3,
            stride=1,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        conv7 = conv_block(
            base_nf * 8,
            base_nf * 8,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        # 8, 512
        conv8 = conv_block(
            base_nf * 8,
            base_nf * 8,
            kernel_size=3,
            stride=1,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        conv9 = conv_block(
            base_nf * 8,
            base_nf * 8,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
        )
        # 4, 512
        self.features = sequential(
            conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9
        )

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# Assume input range is [0, 1]
class VGGFeatureExtractor(nn.Module):
    def __init__(
        self,
        feature_layer=34,
        use_bn=False,
        use_input_norm=True,
        device=torch.device("cpu"),
    ):
        super(VGGFeatureExtractor, self).__init__()
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)
        self.features = nn.Sequential(
            *list(model.features.children())[: (feature_layer + 1)]
        )
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


# Define GAN loss
class GANLoss(nn.Module):
    def __init__(self, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        self.loss = nn.BCEWithLogitsLoss()

    def get_target_label(self, input, target_is_real):
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("Linear") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("BatchNorm2d") != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


# Generator
def define_G(scale):
    netG = RRDBNet(
        in_nc=1,
        out_nc=1,
        nf=64,
        nb=23,
        gc=32,
        upscale=scale,
        norm_type=None,
        act_type="leakyrelu",
        mode="CNA",
        upsample_mode="upconv",
    )
    # netG = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, upscale=4,
    #               norm_type=None, act_type='leakyrelu', mode="CNA",
    #               upsample_mode='upconv')

    weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=0.1)
    netG.apply(weights_init_kaiming_)

    if torch.cuda.is_available():
        netG = nn.DataParallel(netG)
    return netG


# Discriminator
def define_D():
    netD = Discriminator_VGG_128(
        in_nc=1, base_nf=64, norm_type="batch", mode="CNA", act_type="leakyrelu"
    )
    # netD = Discriminator_VGG_128(in_nc=3, base_nf=64, norm_type="batch",
    #                             mode="CNA", act_type="leakyrelu")

    weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=1)
    netD.apply(weights_init_kaiming_)

    if torch.cuda.is_available():
        netD = nn.DataParallel(netD)
    return netD


def define_F(use_bn=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = VGGFeatureExtractor(
        feature_layer=feature_layer, use_bn=use_bn, use_input_norm=True, device=device
    )

    if torch.cuda.is_available():
        netF = nn.DataParallel(netF)
    netF.eval()  # No need to train
    return netF


###



class ESRGANplus(LightningModule):
    def __init__(
        self,
        batchsize: int = 8,
        lr_patch_size_x: int = 128,
        lr_patch_size_y: int = 128,
        scale_factor: int = 2,
        datagen_sampling_pdf: int = 1,
        learning_rate_d: float = 0.0001,
        learning_rate_g: float = 0.0001,
        n_critic_steps: int = 5,
        epochs: int = 151,
        rotation: bool = True,
        horizontal_flip: bool = True,
        vertical_flip: bool = True,
        train_hr_path: str = "",
        train_lr_path: str = "",
        train_filenames: list = [],
        val_hr_path: str = "",
        val_lr_path: str = "",
        val_filenames: list = [],
        save_basedir: str = None,
        crappifier_method: str = None,
        gen_checkpoint: str = None,
        g_optimizer: str = None,
        d_optimizer: str = None,
        g_scheduler: str = None,
        d_scheduler: str = None,
        additonal_configuration: dict = {},
        verbose: int = 0,
    ):
        super(ESRGANplus, self).__init__()

        self.verbose = verbose
        print('self.verbose: {}'.format(self.verbose))

        if self.verbose > 0:
            print('\nVerbose: Model initialized (begining)\n')

        self.save_hyperparameters()

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

        if gen_checkpoint is not None:
            checkpoint = torch.load(gen_checkpoint)
            self.generator = define_G(checkpoint["scale_factor"])
            self.generator.load_state_dict(checkpoint["model_state_dict"])
            self.best_valid_loss = checkpoint["best_valid_loss"]
        else:
            self.generator = define_G(scale_factor)
            self.best_valid_loss = float("inf")

        self.discriminator = define_D()

        # Generator pixel loss
        self.cri_pix = nn.L1Loss()
        self.l_pix_w = 0.01

        # G feature loss
        self.cri_fea = nn.L1Loss()
        self.l_fea_w = 1

        # load VGG perceptual loss
        self.netF = define_F(use_bn=False)

        # GD gan loss
        self.cri_gan = GANLoss(1.0, 0.0)
        self.l_gan_w = 0.005

        # D_update_ratio and D_init_iters are for WGAN
        self.D_update_ratio = 1
        self.D_init_iters = 0

        self.mae = nn.L1Loss()

        if train_hr_path or train_lr_path:
            self.len_data = self.train_dataloader().__len__()
            self.total_iters = epochs * self.len_data

        if self.verbose > 0:
            print(
                "Generators parameters: {}".format(
                    sum(p.numel() for p in self.generator.parameters())
                )
            )
            print(
                "Discriminators parameters: {}".format(
                    sum(p.numel() for p in self.discriminator.parameters())
                )
            )
            print(
                "self.netF parameters: {}".format(
                    sum(p.numel() for p in self.netF.parameters())
                )
            )
        
        self.validation_step_lr = []
        self.validation_step_hr = []
        self.validation_step_pred = []
        
        self.val_ssim = []
        
        if self.verbose > 0:
            os.makedirs(f"{self.hparams.save_basedir}/training_images", exist_ok=True)


        self.step_schedulers = ['CosineDecay', 'OneCycle', 'MultiStepScheduler']
        self.epoch_schedulers = ['ReduceOnPlateau']

        if self.verbose > 0:
            print('\nVerbose: Model initialized (end)\n')

    def save_model(self, filename):
        if self.hparams.save_basedir is not None:
            torch.save(
                {
                    "model_state_dict": self.generator.state_dict(),
                    "optimizer_state_dict": self.opt_g.state_dict(),
                    "scale_factor": self.hparams.scale_factor,
                    "best_valid_loss": self.best_valid_loss,
                },
                self.hparams.save_basedir + "/" + filename,
            )
        else:
            raise Exception(
                "No save_basedir was specified in the construction of the WGAN object."
            )

    def forward(self, x):
        if isinstance(x, dict):
            return self.generator(x["lr"])
        else:
            return self.generator(x)

    def training_step(self, batch, batch_idx):

        if self.verbose > 1:
            print('\nVerbose: Training step (begining)\n')

        lr, hr = batch["lr"], batch["hr"]

        # Extract the optimizers
        g_opt, d_opt = self.optimizers()

        # Extract the schedulers
        if self.hparams.g_scheduler == "Fixed" and self.hparams.d_scheduler != "Fixed":
            sched_d = self.lr_schedulers()
        elif self.hparams.g_scheduler != "Fixed" and self.hparams.d_scheduler == "Fixed":
            sched_g = self.lr_schedulers()
        elif self.hparams.g_scheduler != "Fixed" and self.hparams.d_scheduler != "Fixed":
            sched_g, sched_d = self.lr_schedulers()
        else:
            # There are no schedulers
            pass

        # The generator is updated every self.hparams.n_critic_steps
        if (batch_idx + 1) % self.hparams.n_critic_steps == 0:
            if self.verbose > 0:
                print(f'Generator updated on step {batch_idx + 1}')

            # Optimize generator
            # toggle_optimizer(): Makes sure only the gradients of the current optimizer's parameters are calculated
            #                     in the training step to prevent dangling gradients in multiple-optimizer setup.
            self.toggle_optimizer(g_opt)

            fake_hr = self.generator(lr)

            # pixel loss
            l_g_pix = self.l_pix_w * self.cri_pix(fake_hr, hr)

            # feature loss
            real_fea = self.netF(hr)
            fake_fea = self.netF(fake_hr)
            l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)

            # G gan + cls loss
            pred_g_fake = self.discriminator(fake_hr)
            # pred_d_real = self.discriminator(hr)

            l_g_gan = self.l_gan_w * self.cri_gan(pred_g_fake, True)
            # l_g_gan = self.l_gan_w * (self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
            #                          self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
            
            l_g_total = l_g_pix + l_g_fea + l_g_gan

            self.log("g_loss", l_g_total, prog_bar=True, on_epoch=True)
            self.log("g_pixel_loss", l_g_pix, prog_bar=True, on_epoch=True)
            self.log("g_features_loss", l_g_fea, prog_bar=True, on_epoch=True)
            self.log("g_adversarial_loss", l_g_gan, prog_bar=True, on_epoch=True)

            # Optimize generator
            self.manual_backward(l_g_total)
            g_opt.step()
            g_opt.zero_grad()
            self.untoggle_optimizer(g_opt)
            
            if self.hparams.g_scheduler in self.step_schedulers:
                sched_g.step()

        # The discriminator is updated every step
        if (batch_idx + 1) % 1 == 0:
            if self.verbose > 0:
                print(f'Discriminator  updated on step {batch_idx + 1}')
            # Optimize discriminator
            self.toggle_optimizer(d_opt)

            fake_hr = self.generator(lr)

            pred_d_real = self.discriminator(hr)
            pred_d_fake = self.discriminator(fake_hr)  # detach to avoid BP to G
            # l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
            # l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)
            l_d_real = self.cri_gan(pred_d_real, True)
            l_d_fake = self.cri_gan(pred_d_fake, False)

            # l_d_total = (l_d_real + l_d_fake) / 2
            l_d_total = l_d_real + l_d_fake

            self.log("d_loss", l_d_total, prog_bar=True, on_epoch=True)
            self.log("d_real", l_d_real, prog_bar=False, on_epoch=True)
            self.log("d_fake", l_d_fake, prog_bar=False, on_epoch=True)
 
            # Optimize discriminator/critic
            self.manual_backward(l_d_total)
            d_opt.step()
            d_opt.zero_grad()
            self.untoggle_optimizer(d_opt)

            if self.hparams.d_scheduler in self.step_schedulers:
                sched_d.step()

        if self.verbose > 0:
            print('\nVerbose: Training step (end)\n')

    def validation_step(self, batch, batch_idx):
        
        if self.verbose > 0:
            print('\nVerbose: validation_step (begining)\n')
            
        # Right now used for just plotting, might want to change it later
        lr, hr = batch["lr"], batch["hr"]
        generated = self(lr)

        if self.verbose > 0:
            print(f'lr.shape: {lr.shape} lr.min: {lr.min()} lr.max: {lr.max()}')
            print(f'hr.shape: {hr.shape} hr.min: {hr.min()} hr.max: {hr.max()}')
            print(f'generated.shape: {generated.shape} generated.min: {generated.min()} generated.max: {generated.max()}')

        true = hr.cpu().numpy()
        fake = generated.cpu().numpy()
        xlr = lr.cpu().numpy()

        for i in range(lr.size(0)):
            ssim = structural_similarity(
                true[i, 0, ...], fake[i, 0, ...], data_range=1.0
            )
            self.log("val_ssim", ssim)
            self.val_ssim.append(ssim)

            psnr = peak_signal_noise_ratio(
                true[i, 0, ...], fake[i, 0, ...], data_range=1.0
            )
            self.log("val_psnr", psnr)
            
        self.validation_step_lr.append(lr)
        self.validation_step_hr.append(hr)
        self.validation_step_pred.append(generated)

        if batch_idx < 3 and self.verbose > 0:
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 4, 1)
            plt.title("Input LR image")
            plt.imshow(xlr[0,0], "gray")
            plt.subplot(1, 4, 2)
            plt.title("Ground truth")
            plt.imshow(true[0,0], "gray")
            plt.subplot(1, 4, 3)
            plt.title("Prediction")
            plt.imshow(fake[0,0], "gray")
            plt.subplot(1, 4, 4)
            plt.title(f"SSIM: {ssim:.3f}")
            plt.imshow(true[0,0] - fake[0,0], "inferno")

            plt.tight_layout()
            plt.savefig(f"{self.hparams.save_basedir}/training_images/{self.current_epoch}_{batch_idx}.png")
            plt.close()

        if self.verbose > 0:
            print('\nVerbose: validation_step (end)\n')

        return lr, hr, generated

    def on_validation_epoch_end(self):

        if self.verbose > 0:
            print('\nVerbose: on_validation_epoch_end (begining)\n')

        # Right now used for just plotting, might want to change it later
        # lr, hr, generated = torch.stack(self.validation_step_outputs)
        lr = torch.cat(self.validation_step_lr, 0)
        hr = torch.cat(self.validation_step_hr, 0)
        generated = torch.cat(self.validation_step_pred, 0)
        
        if self.verbose > 0:
            print(f'lr.shape: {lr.shape} lr.min: {lr.min()} lr.max: {lr.max()} values: {lr[0,0,0,:10]}')
            print(f'hr.shape: {hr.shape} hr.min: {hr.min()} hr.max: {hr.max()} values: {hr[0,0,0,:10]}')
            print(f'generated.shape: {generated.shape} generated.min: {generated.min()} generated.max: {generated.max()} values: {generated[0,0,0,:10]}')


        l_g_total = 0

        # pixel loss
        l_g_pix = self.l_pix_w * self.cri_pix(generated, hr)
        
        # feature loss
        real_fea = self.netF(hr).detach()
        fake_fea = self.netF(generated)
        l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
        
        # G gan + cls loss
        pred_g_fake = self.discriminator(generated)
        #pred_d_real = self.discriminator(hr).detach()
        pred_d_real = self.discriminator(hr)

        l_g_gan = self.l_gan_w * (self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                                  self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
        l_g_total = l_g_pix + l_g_fea + l_g_gan

        self.log('val_g_loss', l_g_total, prog_bar=True, on_epoch=True)
        self.log('val_g_pixel_loss', l_g_pix, prog_bar=True, on_epoch=True)
        self.log('val_g_features_loss', l_g_fea, prog_bar=True, on_epoch=True)
        self.log('val_g_adversarial_loss', l_g_gan, prog_bar=True, on_epoch=True)

        if self.verbose > 0:
            print(f'g_loss: {l_g_total}')
            print(f'self.best_valid_loss: {self.best_valid_loss}')

        if l_g_total < self.best_valid_loss:
            self.best_valid_loss = l_g_total
            self.save_model("best_checkpoint.pth")

        self.validation_step_lr.clear()  # free memory
        self.validation_step_hr.clear()  # free memory
        self.validation_step_pred.clear()  # free memory


        # Extract the schedulers
        if self.hparams.g_scheduler == "Fixed" and self.hparams.d_scheduler != "Fixed":
            sched_d = self.lr_schedulers()
        elif self.hparams.g_scheduler != "Fixed" and self.hparams.d_scheduler == "Fixed":
            sched_g = self.lr_schedulers()
        elif self.hparams.g_scheduler != "Fixed" and self.hparams.d_scheduler != "Fixed":
            sched_g, sched_d = self.lr_schedulers()
        else:
            # There are no schedulers
            pass
        
        mean_val_ssim = np.array(self.val_ssim).mean()

        # Note that step should be called after validate()
        if self.hparams.d_scheduler in self.epoch_schedulers:
            sched_d.step(mean_val_ssim)
        if self.hparams.g_scheduler in self.epoch_schedulers:
            sched_g.step(mean_val_ssim)

        self.val_ssim.clear() # free memory

        if self.verbose > 0:
            print('\nVerbose: on_validation_epoch_end (end)\n')

    def on_train_end(self):
        self.save_model("last_checkpoint.pth")

    def configure_optimizers(self):
        
        if self.verbose > 0:
            print('\nVerbose: configure_optimizers (begining)\n')
            print(f'Generator optimizer: {self.hparams.g_optimizer}')
            print(f'Discriminator optimizer: {self.hparams.d_optimizer}')
            print(f'Generator scheduler: {self.hparams.g_scheduler}')
            print(f'Discriminator scheduler: {self.hparams.d_scheduler}')

        self.opt_g = select_optimizer(
            library_name="pytorch",
            optimizer_name=self.hparams.g_optimizer,
            learning_rate=self.hparams.learning_rate_g,
            check_point=self.hparams.gen_checkpoint,
            parameters=self.generator.parameters(),
            additional_configuration=self.hparams.additonal_configuration,
            verbose=self.verbose
        )

        self.opt_d = select_optimizer(
            library_name="pytorch",
            optimizer_name=self.hparams.d_optimizer,
            learning_rate=self.hparams.learning_rate_d,
            check_point=None,
            parameters=self.discriminator.parameters(),
            additional_configuration=self.hparams.additonal_configuration,
            verbose=self.verbose
        )

        sched_g = select_lr_schedule(
            library_name="pytorch",
            lr_scheduler_name=self.hparams.g_scheduler,
            data_len=self.len_data,
            num_epochs=self.hparams.epochs,
            learning_rate=self.hparams.learning_rate_g,
            monitor_loss="val_g_loss",
            name="g_lr",
            optimizer=self.opt_g,
            frequency=self.hparams.n_critic_steps,
            additional_configuration=self.hparams.additonal_configuration,
            verbose=self.verbose
        )

        sched_d = select_lr_schedule(
            library_name="pytorch",
            lr_scheduler_name=self.hparams.d_scheduler,
            data_len=self.len_data,
            num_epochs=self.hparams.epochs,
            learning_rate=self.hparams.learning_rate_d,
            monitor_loss="val_g_loss",
            name="d_lr",
            optimizer=self.opt_d,
            frequency=1,
            additional_configuration=self.hparams.additonal_configuration,
            verbose=self.verbose
        )

        if sched_g is None and sched_d is None:
            scheduler_list = []
        else:
            scheduler_list = [sched_g, sched_d]

        return [self.opt_g, self.opt_d], scheduler_list

    def train_dataloader(self):
        
        if self.verbose > 0:
            print('\nVerbose: train_dataloader (begining)\n')

        transformations = []

        if self.hparams.horizontal_flip:
            transformations.append(RandomHorizontalFlip())
        if self.hparams.vertical_flip:
            transformations.append(RandomVerticalFlip())
        if self.hparams.rotation:
            transformations.append(RandomRotate())

        transformations.append(ToTensor())

        if self.verbose > 0:
            print(f'Transformations: {transformations}')

        transf = torchvision.transforms.Compose(transformations)

        if self.hparams.val_hr_path is None:
            dataset = PytorchDataset(
                hr_data_path=self.hparams.train_hr_path,
                lr_data_path=self.hparams.train_lr_path,
                filenames=self.hparams.train_filenames,
                scale_factor=self.hparams.scale_factor,
                crappifier_name=self.hparams.crappifier_method,
                lr_patch_shape=(
                    self.hparams.lr_patch_size_x,
                    self.hparams.lr_patch_size_y,
                ),
                transformations=transf,
                datagen_sampling_pdf=self.hparams.datagen_sampling_pdf,
                val_split=0.1,
                validation=False,
                verbose=self.verbose
            )

        else:
            dataset = PytorchDataset(
                hr_data_path=self.hparams.train_hr_path,
                lr_data_path=self.hparams.train_lr_path,
                filenames=self.hparams.train_filenames,
                scale_factor=self.hparams.scale_factor,
                crappifier_name=self.hparams.crappifier_method,
                lr_patch_shape=(
                    self.hparams.lr_patch_size_x,
                    self.hparams.lr_patch_size_y,
                ),
                transformations=transf,
                datagen_sampling_pdf=self.hparams.datagen_sampling_pdf,
                verbose=self.verbose
            )

        if self.verbose > 0:

            print(f'hr_data_path: {dataset.hr_data_path}')
            print(f'lr_data_path: {dataset.lr_data_path}')
            print(f'filenames[:3]: {dataset.filenames[:3]}')
            print(f'transformations: {dataset.transformations}')
            print(f'scale_factor: {dataset.scale_factor}')
            print(f'crappifier_name: {dataset.crappifier_name}')
            print(f'lr_patch_shape: {dataset.lr_patch_shape}')
            print(f'datagen_sampling_pdf: {dataset.datagen_sampling_pdf}')
            print(f'actual_scale_factor: {dataset.actual_scale_factor}')

        if self.verbose > 0:
            print('\nVerbose: train_dataloader (end)\n')

        return DataLoader(
            dataset, batch_size=self.hparams.batchsize, shuffle=True, num_workers=0
        )

    def val_dataloader(self):
 
        if self.verbose > 0:
            print('\nVerbose: val_dataloader (begining)\n')

        transf = ToTensor()

        if self.hparams.val_hr_path is None:
            dataset = PytorchDataset(
                hr_data_path=self.hparams.train_hr_path,
                lr_data_path=self.hparams.train_lr_path,
                filenames=self.hparams.train_filenames,
                scale_factor=self.hparams.scale_factor,
                crappifier_name=self.hparams.crappifier_method,
                lr_patch_shape=(
                    self.hparams.lr_patch_size_x,
                    self.hparams.lr_patch_size_y,
                ),
                transformations=transf,
                datagen_sampling_pdf=self.hparams.datagen_sampling_pdf,
                val_split=0.1,
                validation=True,
                verbose=self.verbose
            )
        else:
            dataset = PytorchDataset(
                hr_data_path=self.hparams.val_hr_path,
                lr_data_path=self.hparams.val_lr_path,
                filenames=self.hparams.val_filenames,
                scale_factor=self.hparams.scale_factor,
                crappifier_name=self.hparams.crappifier_method,
                lr_patch_shape=(
                    self.hparams.lr_patch_size_x,
                    self.hparams.lr_patch_size_y,
                ),
                transformations=transf,
                datagen_sampling_pdf=self.hparams.datagen_sampling_pdf,
                verbose=self.verbose
            )

        if self.verbose > 0:

            print(f'hr_data_path: {dataset.hr_data_path}')
            print(f'lr_data_path: {dataset.lr_data_path}')
            print(f'filenames[:3]: {dataset.filenames[:3]}')
            print(f'transformations: {dataset.transformations}')
            print(f'scale_factor: {dataset.scale_factor}')
            print(f'crappifier_name: {dataset.crappifier_name}')
            print(f'lr_patch_shape: {dataset.lr_patch_shape}')
            print(f'datagen_sampling_pdf: {dataset.datagen_sampling_pdf}')
            print(f'actual_scale_factor: {dataset.actual_scale_factor}')

        if self.verbose > 0:
            print('\nVerbose: val_dataloader (end)\n')

        return DataLoader(
            dataset, batch_size=self.hparams.batchsize, shuffle=False, num_workers=0
        )