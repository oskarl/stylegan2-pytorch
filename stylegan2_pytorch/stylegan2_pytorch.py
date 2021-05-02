import os
import sys
import math
import fire
import json

import numpy as np
import os
import gzip, pickle
import tensorflow as tf
from imageio import imread
from scipy import linalg
import pathlib
import urllib
import warnings

import Evaluators as Evaluators
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tqdm import tqdm
from math import floor, log2
from random import random
from shutil import rmtree
from functools import partial
import multiprocessing
from contextlib import contextmanager, ExitStack

import numpy as np

import torch
from torch import nn
from torch.utils import data
from torch.optim import Adam
import torch.nn.functional as F
from torch.autograd import grad as torch_grad
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from kornia.filters import filter2D

import torchvision
from torchvision import transforms
#from stylegan2_pytorch.version import __version__
from diff_augment import DiffAugment

from vector_quantize_pytorch import VectorQuantize
from linear_attention_transformer import ImageLinearAttention

from PIL import Image
from pathlib import Path

import extragradient

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

import aim

assert torch.cuda.is_available(), 'You need to have an Nvidia GPU with CUDA installed.'

# FID
bs = 32
fid = Evaluators.FID(batch_size=bs, real_mean_cov_file='/content/drive/MyDrive/GAN/files/ffhq_128_stats_ttur.pickle', resize_to=139)#celeba_64_stats.pickle')

# constants

NUM_CORES = multiprocessing.cpu_count()
EXTS = ['jpg', 'jpeg', 'png']

# helper classes

class NanException(Exception):
    pass

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
    def update_average(self, old, new):
        if not exists(old):
            return new
        return old * self.beta + (1 - self.beta) * new

class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)

class RandomApply(nn.Module):
    def __init__(self, prob, fn, fn_else = lambda x: x):
        super().__init__()
        self.fn = fn
        self.fn_else = fn_else
        self.prob = prob
    def forward(self, x):
        fn = self.fn if random() < self.prob else self.fn_else
        return fn(x)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x

class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return self.fn(x) * self.g

class PermuteToFrom(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        out, loss = self.fn(x)
        out = out.permute(0, 3, 1, 2)
        return out, loss

class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)
    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f [None, :, None]
        return filter2D(x, f, normalized=True)

# one layer of self-attention and feedforward, for images

attn_and_ff = lambda chan: nn.Sequential(*[
    Residual(Rezero(ImageLinearAttention(chan, norm_queries = True))),
    Residual(Rezero(nn.Sequential(nn.Conv2d(chan, chan * 2, 1), leaky_relu(), nn.Conv2d(chan * 2, chan, 1))))
])

# helpers

def exists(val):
    return val is not None

@contextmanager
def null_context():
    yield

def combine_contexts(contexts):
    @contextmanager
    def multi_contexts():
        with ExitStack() as stack:
            yield [stack.enter_context(ctx()) for ctx in contexts]
    return multi_contexts

def default(value, d):
    return value if exists(value) else d

def cycle(iterable):
    while True:
        for i in iterable:
            yield i

def cast_list(el):
    return el if isinstance(el, list) else [el]

def is_empty(t):
    if isinstance(t, torch.Tensor):
        return t.nelement() == 0
    return not exists(t)

def raise_if_nan(t):
    if torch.isnan(t):
        raise NanException

def gradient_accumulate_contexts(gradient_accumulate_every, is_ddp, ddps):
    if is_ddp:
        num_no_syncs = gradient_accumulate_every - 1
        head = [combine_contexts(map(lambda ddp: ddp.no_sync, ddps))] * num_no_syncs
        tail = [null_context]
        contexts =  head + tail
    else:
        contexts = [null_context] * gradient_accumulate_every

    for context in contexts:
        with context():
            yield

def loss_backwards(fp16, loss, optimizer, loss_id, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer, loss_id) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

def gradient_penalty(images, output, weight = 10):
    batch_size = images.shape[0]
    gradients = torch_grad(outputs=output, inputs=images,
                           grad_outputs=torch.ones(output.size(), device=images.device),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.reshape(batch_size, -1)
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

def calc_pl_lengths(styles, images):
    device = images.device
    num_pixels = images.shape[2] * images.shape[3]
    pl_noise = torch.randn(images.shape, device=device) / math.sqrt(num_pixels)
    outputs = (images * pl_noise).sum()

    pl_grads = torch_grad(outputs=outputs, inputs=styles,
                          grad_outputs=torch.ones(outputs.shape, device=device),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]

    return (pl_grads ** 2).sum(dim=2).mean(dim=1).sqrt()

def noise(n, latent_dim, device):
    return torch.randn(n, latent_dim).cuda(device)

def noise_list(n, layers, latent_dim, device):
    return [(noise(n, latent_dim, device), layers)]

def mixed_list(n, layers, latent_dim, device):
    tt = int(torch.rand(()).numpy() * layers)
    return noise_list(n, tt, latent_dim, device) + noise_list(n, layers - tt, latent_dim, device)

def latent_to_w(style_vectorizer, latent_descr):
    return [(style_vectorizer(z), num_layers) for z, num_layers in latent_descr]

def image_noise(n, im_size, device):
    return torch.FloatTensor(n, im_size, im_size, 1).uniform_(0., 1.).cuda(device)

def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)

def evaluate_in_chunks(max_batch_size, model, *args):
    split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)

def styles_def_to_tensor(styles_def):
    return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)

def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool

def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res

# dataset

def convert_rgb_to_transparent(image):
    if image.mode != 'RGBA':
        return image.convert('RGBA')
    return image

def convert_transparent_to_rgb(image):
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image

class expand_greyscale(object):
    def __init__(self, transparent):
        self.transparent = transparent

    def __call__(self, tensor):
        channels = tensor.shape[0]
        num_target_channels = 4 if self.transparent else 3

        if channels == num_target_channels:
            return tensor

        alpha = None
        if channels == 1:
            color = tensor.expand(3, -1, -1)
        elif channels == 2:
            color = tensor[:1].expand(3, -1, -1)
            alpha = tensor[1:]
        else:
            raise Exception(f'image with invalid number of channels given {channels}')

        if not exists(alpha) and self.transparent:
            alpha = torch.ones(1, *tensor.shape[1:], device=tensor.device)

        return color if not self.transparent else torch.cat((color, alpha))

def resize_to_minimum_size(min_size, image):
    if max(*image.size) < min_size:
        return torchvision.transforms.functional.resize(image, min_size)
    return image

class Dataset(data.Dataset):
    def __init__(self, folder, image_size, transparent = False, aug_prob = 0.):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in EXTS for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        assert len(self.paths) > 0, f'No images were found in {folder} for training'

        convert_image_fn = convert_transparent_to_rgb if not transparent else convert_rgb_to_transparent
        num_channels = 3 if not transparent else 4

        self.transform = transforms.Compose([
            transforms.Lambda(convert_image_fn),
            transforms.Lambda(partial(resize_to_minimum_size, image_size)),
            transforms.Resize(image_size),
            RandomApply(aug_prob, transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.98, 1.02)), transforms.CenterCrop(image_size)),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale(transparent))
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# augmentations

def random_hflip(tensor, prob):
    if prob > random():
        return tensor
    return torch.flip(tensor, dims=(3,))

class AugWrapper(nn.Module):
    def __init__(self, D, image_size):
        super().__init__()
        self.D = D

    def forward(self, images, prob = 0., types = [], detach = False):
        if random() < prob:
            images = random_hflip(images, prob=0.5)
            images = DiffAugment(images, types=types)

        if detach:
            images = images.detach()

        return self.D(images)

# stylegan2 classes

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul = 1, bias = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)

class StyleVectorizer(nn.Module):
    def __init__(self, emb, depth, lr_mul = 0.1):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(emb, emb, lr_mul), leaky_relu()])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return self.net(x)

class RGBBlock(nn.Module):
    def __init__(self, latent_dim, input_channel, upsample, rgba = False):
        super().__init__()
        self.input_channel = input_channel
        self.to_style = nn.Linear(latent_dim, input_channel)

        out_filters = 3 if not rgba else 4
        self.conv = Conv2DMod(input_channel, out_filters, 1, demod=False)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=False),
            Blur()
        ) if upsample else None

    def forward(self, x, prev_rgb, istyle):
        b, c, h, w = x.shape
        style = self.to_style(istyle)
        x = self.conv(x, style)

        if exists(prev_rgb):
            x = x + prev_rgb

        if exists(self.upsample):
            x = self.upsample(x)

        return x

class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, eps = 1e-8, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        self.eps = eps
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x

class GeneratorBlock(nn.Module):
    def __init__(self, latent_dim, input_channels, filters, upsample = True, upsample_rgb = True, rgba = False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None

        self.to_style1 = nn.Linear(latent_dim, input_channels)
        self.to_noise1 = nn.Linear(1, filters)
        self.conv1 = Conv2DMod(input_channels, filters, 3)
        
        self.to_style2 = nn.Linear(latent_dim, filters)
        self.to_noise2 = nn.Linear(1, filters)
        self.conv2 = Conv2DMod(filters, filters, 3)

        self.activation = leaky_relu()
        self.to_rgb = RGBBlock(latent_dim, filters, upsample_rgb, rgba)

    def forward(self, x, prev_rgb, istyle, inoise):
        if exists(self.upsample):
            x = self.upsample(x)

        inoise = inoise[:, :x.shape[2], :x.shape[3], :]
        noise1 = self.to_noise1(inoise).permute((0, 3, 2, 1))
        noise2 = self.to_noise2(inoise).permute((0, 3, 2, 1))

        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)
        x = self.activation(x + noise1)

        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x + noise2)

        rgb = self.to_rgb(x, prev_rgb, istyle)
        return x, rgb

class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride = (2 if downsample else 1))

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            leaky_relu(),
            nn.Conv2d(filters, filters, 3, padding=1),
            leaky_relu()
        )

        self.downsample = nn.Sequential(
            Blur(),
            nn.Conv2d(filters, filters, 3, padding = 1, stride = 2)
        ) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        if exists(self.downsample):
            x = self.downsample(x)
        x = (x + res) * (1 / math.sqrt(2))
        return x

class Generator(nn.Module):
    def __init__(self, image_size, latent_dim, network_capacity = 16, transparent = False, attn_layers = [], no_const = False, fmap_max = 512):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_layers = int(log2(image_size) - 1)

        filters = [network_capacity * (2 ** (i + 1)) for i in range(self.num_layers)][::-1]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]

        in_out_pairs = zip(filters[:-1], filters[1:])
        self.no_const = no_const

        if no_const:
            self.to_initial_block = nn.ConvTranspose2d(latent_dim, init_channels, 4, 1, 0, bias=False)
        else:
            self.initial_block = nn.Parameter(torch.randn((1, init_channels, 4, 4)))

        self.initial_conv = nn.Conv2d(filters[0], filters[0], 3, padding=1)
        self.blocks = nn.ModuleList([])
        self.attns = nn.ModuleList([])

        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)
            num_layer = self.num_layers - ind

            attn_fn = attn_and_ff(in_chan) if num_layer in attn_layers else None

            self.attns.append(attn_fn)

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample = not_first,
                upsample_rgb = not_last,
                rgba = transparent
            )
            self.blocks.append(block)

    def forward(self, styles, input_noise):
        batch_size = styles.shape[0]
        image_size = self.image_size

        if self.no_const:
            avg_style = styles.mean(dim=1)[:, :, None, None]
            x = self.to_initial_block(avg_style)
        else:
            x = self.initial_block.expand(batch_size, -1, -1, -1)

        rgb = None
        styles = styles.transpose(0, 1)
        x = self.initial_conv(x)

        for style, block, attn in zip(styles, self.blocks, self.attns):
            if exists(attn):
                x = attn(x)
            x, rgb = block(x, rgb, style, input_noise)

        return rgb

class Discriminator(nn.Module):
    def __init__(self, image_size, network_capacity = 16, fq_layers = [], fq_dict_size = 256, attn_layers = [], transparent = False, fmap_max = 512):
        super().__init__()
        num_layers = int(log2(image_size) - 1)
        num_init_filters = 3 if not transparent else 4

        blocks = []
        filters = [num_init_filters] + [(network_capacity * 4) * (2 ** i) for i in range(num_layers + 1)]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        chan_in_out = list(zip(filters[:-1], filters[1:]))

        blocks = []
        attn_blocks = []
        quantize_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DiscriminatorBlock(in_chan, out_chan, downsample = is_not_last)
            blocks.append(block)

            attn_fn = attn_and_ff(out_chan) if num_layer in attn_layers else None

            attn_blocks.append(attn_fn)

            quantize_fn = PermuteToFrom(VectorQuantize(out_chan, fq_dict_size)) if num_layer in fq_layers else None
            quantize_blocks.append(quantize_fn)

        self.blocks = nn.ModuleList(blocks)
        self.attn_blocks = nn.ModuleList(attn_blocks)
        self.quantize_blocks = nn.ModuleList(quantize_blocks)

        chan_last = filters[-1]
        latent_dim = 2 * 2 * chan_last

        self.final_conv = nn.Conv2d(chan_last, chan_last, 3, padding=1)
        self.flatten = Flatten()
        self.to_logit = nn.Linear(latent_dim, 1)

    def forward(self, x):
        b, *_ = x.shape

        quantize_loss = torch.zeros(1).to(x)

        for (block, attn_block, q_block) in zip(self.blocks, self.attn_blocks, self.quantize_blocks):
            x = block(x)

            if exists(attn_block):
                x = attn_block(x)

            if exists(q_block):
                x, _, loss = q_block(x)
                quantize_loss += loss

        x = self.final_conv(x)
        x = self.flatten(x)
        x = self.to_logit(x)
        return x.squeeze(), quantize_loss

class StyleGAN2(nn.Module):
    def __init__(self, image_size, latent_dim = 512, fmap_max = 512, style_depth = 8, network_capacity = 16, transparent = False, fp16 = False, cl_reg = False, steps = 1, lr = 1e-4, ttur_mult = 2, fq_layers = [], fq_dict_size = 256, attn_layers = [], no_const = False, lr_mlp = 0.1, rank = 0, opt_scheme = 'Adam'):
        super().__init__()
        self.lr = lr
        self.steps = steps
        self.ema_updater = EMA(0.995)

        self.S = StyleVectorizer(latent_dim, style_depth, lr_mul = lr_mlp)
        self.G = Generator(image_size, latent_dim, network_capacity, transparent = transparent, attn_layers = attn_layers, no_const = no_const, fmap_max = fmap_max)
        self.D = Discriminator(image_size, network_capacity, fq_layers = fq_layers, fq_dict_size = fq_dict_size, attn_layers = attn_layers, transparent = transparent, fmap_max = fmap_max)

        self.SE = StyleVectorizer(latent_dim, style_depth, lr_mul = lr_mlp)
        self.GE = Generator(image_size, latent_dim, network_capacity, transparent = transparent, attn_layers = attn_layers, no_const = no_const)

        self.D_cl = None

        if cl_reg:
            from contrastive_learner import ContrastiveLearner
            # experimental contrastive loss discriminator regularization
            assert not transparent, 'contrastive loss regularization does not work with transparent images yet'
            self.D_cl = ContrastiveLearner(self.D, image_size, hidden_layer='flatten')

        # wrapper for augmenting all images going into the discriminator
        self.D_aug = AugWrapper(self.D, image_size)

        # turn off grad for exponential moving averages
        set_requires_grad(self.SE, False)
        set_requires_grad(self.GE, False)

        # init optimizers
        generator_params = list(self.G.parameters()) + list(self.S.parameters())
        self.opt_scheme = opt_scheme
        if opt_scheme == 'Adam':
            self.G_opt = Adam(generator_params, lr = self.lr, betas=(0.5, 0.9))
            self.D_opt = Adam(self.D.parameters(), lr = self.lr * ttur_mult, betas=(0.5, 0.9))
        elif opt_scheme == 'ExtraAdam':
            self.G_opt = extragradient.ExtraAdam(generator_params, lr = self.lr, betas=(0.5, 0.9))
            self.D_opt = extragradient.ExtraAdam(self.D.parameters(), lr = self.lr * ttur_mult, betas=(0.5, 0.9))
        elif opt_scheme == 'OptAdam':
            self.G_opt = extragradient.OptimisticAdam(generator_params, lr = self.lr, betas=(0.5, 0.9))
            self.D_opt = extragradient.OptimisticAdam(self.D.parameters(), lr = self.lr * ttur_mult, betas=(0.5, 0.9))
        print(self.G_opt)
        print(self.D_opt)

        # init weights
        self._init_weights()
        self.reset_parameter_averaging()

        self.cuda(rank)

        # startup apex mixed precision
        self.fp16 = fp16
        if fp16:
            (self.S, self.G, self.D, self.SE, self.GE), (self.G_opt, self.D_opt) = amp.initialize([self.S, self.G, self.D, self.SE, self.GE], [self.G_opt, self.D_opt], opt_level='O1', num_losses=3)

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        for block in self.G.blocks:
            nn.init.zeros_(block.to_noise1.weight)
            nn.init.zeros_(block.to_noise2.weight)
            nn.init.zeros_(block.to_noise1.bias)
            nn.init.zeros_(block.to_noise2.bias)

    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

        update_moving_average(self.SE, self.S)
        update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.SE.load_state_dict(self.S.state_dict())
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x):
        return x

class Trainer():
    def __init__(
        self,
        name = 'default',
        results_dir = 'results',
        models_dir = 'models',
        base_dir = './',
        image_size = 128,
        network_capacity = 16,
        fmap_max = 512,
        transparent = False,
        batch_size = 4,
        mixed_prob = 0.9,
        gradient_accumulate_every=1,
        lr = 2e-4,
        lr_mlp = 0.1,
        ttur_mult = 2,
        rel_disc_loss = False,
        num_workers = None,
        save_every = 1000,
        evaluate_every = 1000,
        num_image_tiles = 8,
        trunc_psi = 0.6,
        fp16 = False,
        cl_reg = False,
        no_pl_reg = False,
        fq_layers = [],
        fq_dict_size = 256,
        attn_layers = [],
        no_const = False,
        aug_prob = 0.,
        aug_types = ['translation', 'cutout'],
        top_k_training = False,
        generator_top_k_gamma = 0.99,
        generator_top_k_frac = 0.5,
        dataset_aug_prob = 0.,
        calculate_fid_every = None,
        calculate_fid_num_images = 12800,
        clear_fid_cache = False,
        is_ddp = False,
        rank = 0,
        world_size = 1,
        log = False,
        opt_scheme = 'Adam',
        *args,
        **kwargs
    ):
        print('Init trainer')
        self.GAN_params = [args, kwargs]
        self.GAN = None

        self.name = name

        base_dir = Path(base_dir)
        self.base_dir = base_dir
        self.results_dir = base_dir / results_dir
        self.models_dir = base_dir / models_dir
        self.fid_dir = base_dir / 'fid' / name
        self.config_path = self.models_dir / name / '.config.json'

        assert log2(image_size).is_integer(), 'image size must be a power of 2 (64, 128, 256, 512, 1024)'
        self.image_size = image_size
        self.network_capacity = network_capacity
        self.fmap_max = fmap_max
        self.transparent = transparent

        self.opt_scheme = opt_scheme

        self.fq_layers = cast_list(fq_layers)
        self.fq_dict_size = fq_dict_size
        self.has_fq = len(self.fq_layers) > 0

        self.attn_layers = cast_list(attn_layers)
        self.no_const = no_const

        self.aug_prob = aug_prob
        self.aug_types = aug_types

        self.lr = lr
        self.lr_mlp = lr_mlp
        self.ttur_mult = ttur_mult
        self.rel_disc_loss = rel_disc_loss
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixed_prob = mixed_prob

        self.num_image_tiles = num_image_tiles
        self.evaluate_every = evaluate_every
        self.save_every = save_every
        self.steps = 0

        self.av = None
        self.trunc_psi = trunc_psi

        self.no_pl_reg = no_pl_reg
        self.pl_mean = None

        self.gradient_accumulate_every = gradient_accumulate_every

        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex is not available for you to use mixed precision training'
        self.fp16 = fp16

        self.cl_reg = cl_reg

        self.d_loss = 0
        self.g_loss = 0
        self.q_loss = None
        self.last_gp_loss = None
        self.last_cr_loss = None
        self.last_fid = None

        self.pl_length_ma = EMA(0.99)
        self.init_folders()

        self.loader = None
        self.dataset_aug_prob = dataset_aug_prob

        self.calculate_fid_every = calculate_fid_every
        self.calculate_fid_num_images = calculate_fid_num_images
        self.clear_fid_cache = clear_fid_cache

        self.top_k_training = top_k_training
        self.generator_top_k_gamma = generator_top_k_gamma
        self.generator_top_k_frac = generator_top_k_frac

        assert not (is_ddp and cl_reg), 'Contrastive loss regularization does not work well with multi GPUs yet'
        self.is_ddp = is_ddp
        self.is_main = rank == 0
        self.rank = rank
        self.world_size = world_size

        self.logger = aim.Session(experiment=name) if log else None

    @property
    def image_extension(self):
        return 'jpg' if not self.transparent else 'png'

    @property
    def checkpoint_num(self):
        return floor(self.steps // self.save_every)

    @property
    def hparams(self):
        return {'image_size': self.image_size, 'network_capacity': self.network_capacity}
        
    def init_GAN(self):
        args, kwargs = self.GAN_params
        self.GAN = StyleGAN2(lr = self.lr, lr_mlp = self.lr_mlp, ttur_mult = self.ttur_mult, image_size = self.image_size, network_capacity = self.network_capacity, fmap_max = self.fmap_max, transparent = self.transparent, fq_layers = self.fq_layers, fq_dict_size = self.fq_dict_size, attn_layers = self.attn_layers, fp16 = self.fp16, cl_reg = self.cl_reg, no_const = self.no_const, rank = self.rank, opt_scheme = self.opt_scheme, *args, **kwargs)

        if self.is_ddp:
            ddp_kwargs = {'device_ids': [self.rank]}
            self.S_ddp = DDP(self.GAN.S, **ddp_kwargs)
            self.G_ddp = DDP(self.GAN.G, **ddp_kwargs)
            self.D_ddp = DDP(self.GAN.D, **ddp_kwargs)
            self.D_aug_ddp = DDP(self.GAN.D_aug, **ddp_kwargs)

        if exists(self.logger):
            self.logger.set_params(self.hparams)

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config()))

    def load_config(self):
        config = self.config() if not self.config_path.exists() else json.loads(self.config_path.read_text())
        self.image_size = config['image_size']
        self.network_capacity = config['network_capacity']
        self.transparent = config['transparent']
        self.fq_layers = config['fq_layers']
        self.fq_dict_size = config['fq_dict_size']
        self.fmap_max = config.pop('fmap_max', 512)
        self.attn_layers = config.pop('attn_layers', [])
        self.no_const = config.pop('no_const', False)
        self.lr_mlp = config.pop('lr_mlp', 0.1)
        del self.GAN
        self.init_GAN()

    def config(self):
        return {'image_size': self.image_size, 'network_capacity': self.network_capacity, 'lr_mlp': self.lr_mlp, 'transparent': self.transparent, 'fq_layers': self.fq_layers, 'fq_dict_size': self.fq_dict_size, 'attn_layers': self.attn_layers, 'no_const': self.no_const}

    def set_data_src(self, folder):
        self.dataset = Dataset(folder, self.image_size, transparent = self.transparent, aug_prob = self.dataset_aug_prob)
        num_workers = num_workers = default(self.num_workers, NUM_CORES if not self.is_ddp else 0)
        sampler = DistributedSampler(self.dataset, rank=self.rank, num_replicas=self.world_size, shuffle=True) if self.is_ddp else None
        dataloader = data.DataLoader(self.dataset, num_workers = num_workers, batch_size = math.ceil(self.batch_size / self.world_size), sampler = sampler, shuffle = not self.is_ddp, drop_last = True, pin_memory = True)
        self.loader = cycle(dataloader)

        # auto set augmentation prob for user if dataset is detected to be low
        num_samples = len(self.dataset)
        if not exists(self.aug_prob) and num_samples < 1e5:
            self.aug_prob = min(0.5, (1e5 - num_samples) * 3e-6)
            print(f'autosetting augmentation probability to {round(self.aug_prob * 100)}%')

    def train(self):
        assert exists(self.loader), 'You must first initialize the data source with `.set_data_src(<folder of images>)`'

        if not exists(self.GAN):
            self.init_GAN()

        self.GAN.train()
        total_disc_loss = torch.tensor(0.).cuda(self.rank)
        total_gen_loss = torch.tensor(0.).cuda(self.rank)

        batch_size = math.ceil(self.batch_size / self.world_size)

        image_size = self.GAN.G.image_size
        latent_dim = self.GAN.G.latent_dim
        num_layers = self.GAN.G.num_layers

        aug_prob   = self.aug_prob
        aug_types  = self.aug_types
        aug_kwargs = {'prob': aug_prob, 'types': aug_types}

        apply_gradient_penalty = self.steps % 4 == 0
        apply_path_penalty = False#not self.no_pl_reg and self.steps > 5000 and self.steps % 32 == 0
        apply_cl_reg_to_generated = False#self.steps > 20000

        S = self.GAN.S if not self.is_ddp else self.S_ddp
        G = self.GAN.G if not self.is_ddp else self.G_ddp
        D = self.GAN.D if not self.is_ddp else self.D_ddp
        D_aug = self.GAN.D_aug if not self.is_ddp else self.D_aug_ddp

        backwards = partial(loss_backwards, self.fp16)

        if exists(self.GAN.D_cl):
            self.GAN.D_opt.zero_grad()

            if apply_cl_reg_to_generated:
                for i in range(self.gradient_accumulate_every):
                    get_latents_fn = mixed_list if random() < self.mixed_prob else noise_list
                    style = get_latents_fn(batch_size, num_layers, latent_dim, device=self.rank)
                    noise = image_noise(batch_size, image_size, device=self.rank)

                    w_space = latent_to_w(self.GAN.S, style)
                    w_styles = styles_def_to_tensor(w_space)

                    generated_images = self.GAN.G(w_styles, noise)
                    self.GAN.D_cl(generated_images.clone().detach(), accumulate=True)

            for i in range(self.gradient_accumulate_every):
                image_batch = next(self.loader).cuda(self.rank)
                self.GAN.D_cl(image_batch, accumulate=True)

            loss = self.GAN.D_cl.calculate_loss()
            self.last_cr_loss = loss.clone().detach().item()
            backwards(loss, self.GAN.D_opt, loss_id = 0)

            self.GAN.D_opt.step()

        # train discriminator

        avg_pl_length = self.pl_mean
        self.GAN.D_opt.zero_grad()

        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps=[D_aug, S, G]):
            get_latents_fn = mixed_list if random() < self.mixed_prob else noise_list
            style = get_latents_fn(batch_size, num_layers, latent_dim, device=self.rank)
            noise = image_noise(batch_size, image_size, device=self.rank)

            w_space = latent_to_w(S, style)
            w_styles = styles_def_to_tensor(w_space)

            generated_images = G(w_styles, noise)
            fake_output, fake_q_loss = D_aug(generated_images.clone().detach(), detach = True, **aug_kwargs)

            image_batch = next(self.loader).cuda(self.rank)
            image_batch.requires_grad_()
            real_output, real_q_loss = D_aug(image_batch, **aug_kwargs)

            real_output_loss = real_output
            fake_output_loss = fake_output

            if self.rel_disc_loss:
                real_output_loss = real_output_loss - fake_output.mean()
                fake_output_loss = fake_output_loss - real_output.mean()

            divergence = (F.relu(1 + real_output_loss) + F.relu(1 - fake_output_loss)).mean()
            disc_loss = divergence

            if self.has_fq:
                quantize_loss = (fake_q_loss + real_q_loss).mean()
                self.q_loss = float(quantize_loss.detach().item())

                disc_loss = disc_loss + quantize_loss

            if apply_gradient_penalty:
                gp = gradient_penalty(image_batch, real_output)
                self.last_gp_loss = gp.clone().detach().item()
                self.track(self.last_gp_loss, 'GP')
                disc_loss = disc_loss + gp

            disc_loss = disc_loss / self.gradient_accumulate_every
            disc_loss.register_hook(raise_if_nan)
            backwards(disc_loss, self.GAN.D_opt, loss_id = 1)

            total_disc_loss += divergence.detach().item() / self.gradient_accumulate_every

        self.d_loss = float(total_disc_loss)
        self.track(self.d_loss, 'D')

        if self.steps%2 == 0 and self.opt_scheme == 'ExtraAdam':
            print("extra")
            self.GAN.D_opt.extrapolation()
        else:
            print("step")
            self.GAN.D_opt.step()

        # train generator

        self.GAN.G_opt.zero_grad()

        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps=[S, G, D_aug]):
            style = get_latents_fn(batch_size, num_layers, latent_dim, device=self.rank)
            noise = image_noise(batch_size, image_size, device=self.rank)

            w_space = latent_to_w(S, style)
            w_styles = styles_def_to_tensor(w_space)

            generated_images = G(w_styles, noise)
            fake_output, _ = D_aug(generated_images, **aug_kwargs)
            fake_output_loss = fake_output

            if self.top_k_training:
                epochs = (self.steps * batch_size * self.gradient_accumulate_every) / len(self.dataset)
                k_frac = max(self.generator_top_k_gamma ** epochs, self.generator_top_k_frac)
                k = math.ceil(batch_size * k_frac)

                if k != batch_size:
                    fake_output_loss, _ = fake_output_loss.topk(k=k, largest=False)

            loss = fake_output_loss.mean()
            gen_loss = loss

            if apply_path_penalty:
                pl_lengths = calc_pl_lengths(w_styles, generated_images)
                avg_pl_length = np.mean(pl_lengths.detach().cpu().numpy())

                if not is_empty(self.pl_mean):
                    pl_loss = ((pl_lengths - self.pl_mean) ** 2).mean()
                    if not torch.isnan(pl_loss):
                        gen_loss = gen_loss + pl_loss

            gen_loss = gen_loss / self.gradient_accumulate_every
            gen_loss.register_hook(raise_if_nan)
            backwards(gen_loss, self.GAN.G_opt, loss_id = 2)

            total_gen_loss += loss.detach().item() / self.gradient_accumulate_every

        self.g_loss = float(total_gen_loss)
        self.track(self.g_loss, 'G')

        if self.steps%2 == 0 and self.opt_scheme == 'ExtraAdam':
            self.GAN.G_opt.extrapolation()
        else:
            self.GAN.G_opt.step()

        # calculate moving averages

        if apply_path_penalty and not np.isnan(avg_pl_length):
            self.pl_mean = self.pl_length_ma.update_average(self.pl_mean, avg_pl_length)
            self.track(self.pl_mean, 'PL')

        if self.is_main and self.steps % 10 == 0:# and self.steps > 20000:
            self.GAN.EMA()

        #if self.is_main and self.steps <= 25000 and self.steps % 1000 == 2:
        #    self.GAN.reset_parameter_averaging()

        # save from NaN errors

        if any(torch.isnan(l) for l in (total_gen_loss, total_disc_loss)):
            print(f'NaN detected for generator or discriminator. Loading from checkpoint #{self.checkpoint_num}')
            self.load(self.checkpoint_num)
            raise NanException

        # periodically save results

        if self.is_main:
            if self.steps % self.save_every == 0:
                self.save(self.checkpoint_num)

            if self.steps % self.evaluate_every == 0 or (self.steps % 100 == 0 and self.steps < 2500):
                self.evaluate(floor(self.steps / self.evaluate_every))

            if exists(self.calculate_fid_every) and self.steps % self.calculate_fid_every == 0 and self.steps != 0:
                num_batches = math.ceil(self.calculate_fid_num_images / self.batch_size)
                fid_ema, fid = self.calculate_fid(num_batches)
                self.last_fid = fid

                with open(str(self.results_dir / self.name / f'fid_ema_scores.txt'), 'a') as f:
                    f.write(f'{self.steps},{fid_ema}\n')
                with open(str(self.results_dir / self.name / f'fid_scores.txt'), 'a') as f:
                    f.write(f'{self.steps},{fid}\n')

        self.steps += 1
        self.av = None

    @torch.no_grad()
    def evaluate(self, num = 0, trunc = 1.0):
        self.GAN.eval()
        ext = self.image_extension
        num_rows = self.num_image_tiles
    
        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # latents and noise

        latents = noise_list(num_rows ** 2, num_layers, latent_dim, device=self.rank)
        n = image_noise(num_rows ** 2, image_size, device=self.rank)

        # regular

        generated_images = self.generate_truncated(self.GAN.S, self.GAN.G, latents, n, trunc_psi = self.trunc_psi)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}.{ext}'), nrow=num_rows)
        
        # moving averages

        generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, n, trunc_psi = self.trunc_psi)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}-ema.{ext}'), nrow=num_rows)

        # mixing regularities

        def tile(a, dim, n_tile):
            init_dim = a.size(dim)
            repeat_idx = [1] * a.dim()
            repeat_idx[dim] = n_tile
            a = a.repeat(*(repeat_idx))
            order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda(self.rank)
            return torch.index_select(a, dim, order_index)

        nn = noise(num_rows, latent_dim, device=self.rank)
        tmp1 = tile(nn, 0, num_rows)
        tmp2 = nn.repeat(num_rows, 1)

        tt = int(num_layers / 2)
        mixed_latents = [(tmp1, tt), (tmp2, num_layers - tt)]

        generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, mixed_latents, n, trunc_psi = self.trunc_psi)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}-mr.{ext}'), nrow=num_rows)

    @torch.no_grad()
    def calculate_fid(self, num_batches):
        torch.cuda.empty_cache()

        fid_img_size = 128
        z_dim = 512

        self.GAN.eval()

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        imgs = []
        for i in range(10000//bs + 1):
            latents = noise_list(bs, num_layers, latent_dim, device=self.rank)
            noise = image_noise(bs, image_size, device=self.rank)
            generated_images = self.generate_truncated(self.GAN.S, self.GAN.G, latents, noise, trunc_psi = self.trunc_psi)
            g2 = generated_images.cpu().view(generated_images.size(0), 3, fid_img_size, fid_img_size)
            g2 = g2.permute(0, 2, 3, 1)
            imgs.append(g2)
        imgs = torch.cat(imgs).data.numpy() * 255
        imgs = fid.fd._preprocess(imgs)

        imgs_ema = []
        for i in range(10000//bs + 1):
            latents = noise_list(bs, num_layers, latent_dim, device=self.rank)
            noise = image_noise(bs, image_size, device=self.rank)
            # moving averages
            generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, noise, trunc_psi = self.trunc_psi)
            g2 = generated_images.cpu().view(generated_images.size(0), 3, fid_img_size, fid_img_size)
            g2 = g2.permute(0, 2, 3, 1)
            imgs_ema.append(g2)
        imgs_ema = torch.cat(imgs_ema).data.numpy() * 255
        imgs_ema = fid.fd._preprocess(imgs_ema)

        torch.cuda.empty_cache()

        with tf.Session() as sess:
          sess.run(tf.global_variables_initializer())
          m,s = calculate_activation_statistics(imgs, sess)
          m_ema,s_ema = calculate_activation_statistics(imgs_ema, sess)
          mu = fid.fd.real_mean
          sigma = fid.fd.real_cov
          fid_s, fid_ema = calculate_frechet_distance(m,s,mu,sigma), calculate_frechet_distance(m_ema,s_ema,mu,sigma)
        return (fid_ema, fid_s)

        '''from pytorch_fid import fid_score
        torch.cuda.empty_cache()

        real_path = self.fid_dir / 'real'
        fake_path = self.fid_dir / 'fake'

        # remove any existing files used for fid calculation and recreate directories

        if not real_path.exists() or self.clear_fid_cache:
            rmtree(real_path, ignore_errors=True)
            os.makedirs(real_path)

            for batch_num in tqdm(range(num_batches), desc='calculating FID - saving reals'):
                real_batch = next(self.loader)
                for k, image in enumerate(real_batch.unbind(0)):
                    filename = str(k + batch_num * self.batch_size)
                    torchvision.utils.save_image(image, str(real_path / f'{filename}.png'))

        # generate a bunch of fake images in results / name / fid_fake

        rmtree(fake_path, ignore_errors=True)
        os.makedirs(fake_path)

        self.GAN.eval()
        ext = self.image_extension

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        for batch_num in tqdm(range(num_batches), desc='calculating FID - saving generated'):
            # latents and noise
            latents = noise_list(self.batch_size, num_layers, latent_dim, device=self.rank)
            noise = image_noise(self.batch_size, image_size, device=self.rank)

            # moving averages
            generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, noise, trunc_psi = self.trunc_psi)

            for j, image in enumerate(generated_images.unbind(0)):
                torchvision.utils.save_image(image, str(fake_path / f'{str(j + batch_num * self.batch_size)}-ema.{ext}'))

        return fid_score.calculate_fid_given_paths([str(real_path), str(fake_path)], 256, noise.device, 2048)'''

    @torch.no_grad()
    def truncate_style(self, tensor, trunc_psi = 0.75):
        S = self.GAN.S
        batch_size = self.batch_size
        latent_dim = self.GAN.G.latent_dim

        if not exists(self.av):
            z = noise(2000, latent_dim, device=self.rank)
            samples = evaluate_in_chunks(batch_size, S, z).cpu().numpy()
            self.av = np.mean(samples, axis = 0)
            self.av = np.expand_dims(self.av, axis = 0)

        av_torch = torch.from_numpy(self.av).cuda(self.rank)
        tensor = trunc_psi * (tensor - av_torch) + av_torch
        return tensor

    @torch.no_grad()
    def truncate_style_defs(self, w, trunc_psi = 0.75):
        w_space = []
        for tensor, num_layers in w:
            tensor = self.truncate_style(tensor, trunc_psi = trunc_psi)            
            w_space.append((tensor, num_layers))
        return w_space

    @torch.no_grad()
    def generate_truncated(self, S, G, style, noi, trunc_psi = 0.75, num_image_tiles = 8):
        w = map(lambda t: (S(t[0]), t[1]), style)
        w_truncated = self.truncate_style_defs(w, trunc_psi = trunc_psi)
        w_styles = styles_def_to_tensor(w_truncated)
        generated_images = evaluate_in_chunks(self.batch_size, G, w_styles, noi)
        return generated_images.clamp_(0., 1.)

    @torch.no_grad()
    def generate_interpolation(self, num = 0, num_image_tiles = 8, trunc = 1.0, num_steps = 100, save_frames = False):
        self.GAN.eval()
        ext = self.image_extension
        num_rows = num_image_tiles

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # latents and noise

        latents_low = noise(num_rows ** 2, latent_dim, device=self.rank)
        latents_high = noise(num_rows ** 2, latent_dim, device=self.rank)
        n = image_noise(num_rows ** 2, image_size, device=self.rank)

        ratios = torch.linspace(0., 8., num_steps)

        frames = []
        for ratio in tqdm(ratios):
            interp_latents = slerp(ratio, latents_low, latents_high)
            latents = [(interp_latents, num_layers)]
            generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, n, trunc_psi = self.trunc_psi)
            images_grid = torchvision.utils.make_grid(generated_images, nrow = num_rows)
            pil_image = transforms.ToPILImage()(images_grid.cpu())
            
            if self.transparent:
                background = Image.new("RGBA", pil_image.size, (255, 255, 255))
                pil_image = Image.alpha_composite(background, pil_image)
                
            frames.append(pil_image)

        frames[0].save(str(self.results_dir / self.name / f'{str(num)}.gif'), save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)

        if save_frames:
            folder_path = (self.results_dir / self.name / f'{str(num)}')
            folder_path.mkdir(parents=True, exist_ok=True)
            for ind, frame in enumerate(frames):
                frame.save(str(folder_path / f'{str(ind)}.{ext}'))

    def print_log(self):
        data = [
            ('G', self.g_loss),
            ('D', self.d_loss),
            ('GP', self.last_gp_loss),
            ('PL', self.pl_mean),
            ('CR', self.last_cr_loss),
            ('Q', self.q_loss),
            ('FID', self.last_fid)
        ]

        data = [d for d in data if exists(d[1])]
        log = ' | '.join(map(lambda n: f'{n[0]}: {n[1]:.2f}', data))
        print(log)

    def track(self, value, name):
        if not exists(self.logger):
            return
        self.logger.track(value, name = name)

    def model_name(self, num):
        return str(self.models_dir / self.name / f'model_{num}.pt')

    def init_folders(self):
        (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)

    def clear(self):
        rmtree(str(self.models_dir / self.name), True)
        rmtree(str(self.results_dir / self.name), True)
        rmtree(str(self.fid_dir), True)
        rmtree(str(self.config_path), True)
        self.init_folders()

    def save(self, num):
        save_data = {
            'GAN': self.GAN.state_dict(),
            'version': 1#__version__
        }

        if self.GAN.fp16:
            save_data['amp'] = amp.state_dict()

        torch.save(save_data, self.model_name(num))
        self.write_config()

    def load(self, num = -1):
        self.load_config()

        name = num
        if num == -1:
            file_paths = [p for p in Path(self.models_dir / self.name).glob('model_*.pt')]
            saved_nums = sorted(map(lambda x: int(x.stem.split('_')[1]), file_paths))
            if len(saved_nums) == 0:
                return
            name = saved_nums[-1]
            print(f'continuing from previous epoch - {name}')

        self.steps = name * self.save_every

        load_data = torch.load(self.model_name(name))

        if 'version' in load_data:
            print(f"loading from version {load_data['version']}")

        try:
            self.GAN.load_state_dict(load_data['GAN'])
        except Exception as e:
            print('unable to load save model. please try downgrading the package to the version specified by the saved model')
            raise e
        if self.GAN.fp16 and 'amp' in load_data:
            amp.load_state_dict(load_data['amp'])

class ModelLoader:
    def __init__(self, *, base_dir, name = 'default', load_from = -1):
        self.model = Trainer(name = name, base_dir = base_dir)
        self.model.load(load_from)

    def noise_to_styles(self, noise, trunc_psi = None):
        noise = noise.cuda()
        w = self.model.GAN.SE(noise)
        if exists(trunc_psi):
            w = self.model.truncate_style(w)
        return w

    def styles_to_images(self, w):
        batch_size, *_ = w.shape
        num_layers = self.model.GAN.GE.num_layers
        image_size = self.model.image_size
        w_def = [(w, num_layers)]

        w_tensors = styles_def_to_tensor(w_def)
        noise = image_noise(batch_size, image_size, device = 0)

        images = self.model.GAN.GE(w_tensors, noise)
        images.clamp_(0., 1.)
        return images


# FID

#!/usr/bin/env python3
''' Calculates the Frechet Inception Distance (FID) to evalulate GANs.
The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.
When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).
The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectivly.
See --help to see further details.
'''

class InvalidFIDException(Exception):
    pass


def create_inception_graph(pth):
    """Creates a graph from saved GraphDef file."""
    # Creates graph from saved graph_def.pb.
    with tf.io.gfile.GFile( pth, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString( f.read())
        _ = tf.import_graph_def( graph_def, name='FID_Inception_Net')
#-------------------------------------------------------------------------------


# code for handling inception net derived from
#   https://github.com/openai/improved-gan/blob/master/inception_score/model.py
def _get_inception_layer(sess):
    """Prepares inception net for batched usage and returns pool_3 layer. """
    layername = 'FID_Inception_Net/pool_3:0'
    pool3 = sess.graph.get_tensor_by_name(layername)
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims is not None:
              #shape = [s.value for s in shape] TF 1.x
              shape = [s for s in shape] #TF 2.x
              new_shape = []
              for j, s in enumerate(shape):
                if s == 1 and j == 0:
                  new_shape.append(None)
                else:
                  new_shape.append(s)
              o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
    return pool3
#-------------------------------------------------------------------------------


def get_activations(images, sess, batch_size=50, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 256.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """
    inception_layer = _get_inception_layer(sess)
    n_images = images.shape[0]
    if batch_size > n_images:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = n_images
    n_batches = n_images//batch_size # drops the last batch if < batch_size
    pred_arr = np.empty((n_batches * batch_size,2048))
    for i in range(n_batches):
        if verbose:
            print("\rPropagating batch %d/%d" % (i+1, n_batches), end="", flush=True)
        start = i*batch_size
        
        if start+batch_size < n_images:
            end = start+batch_size
        else:
            end = n_images
        
        batch = images[start:end]
        pred = sess.run(inception_layer, {'FID_Inception_Net/ExpandDims:0': batch})
        pred_arr[start:end] = pred.reshape(batch.shape[0],-1)
    if verbose:
        print(" done")
    return pred_arr
#-------------------------------------------------------------------------------


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
            
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
#-------------------------------------------------------------------------------


def calculate_activation_statistics(images, sess, batch_size=50, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 255.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the available hardware.
    -- verbose     : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the incption model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the incption model.
    """
    act = get_activations(images, sess, batch_size, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma
    

#------------------
# The following methods are implemented to obtain a batched version of the activations.
# This has the advantage to reduce memory requirements, at the cost of slightly reduced efficiency.
# - Pyrestone
#------------------


def load_image_batch(files):
    """Convenience method for batch-loading images
    Params:
    -- files    : list of paths to image files. Images need to have same dimensions for all files.
    Returns:
    -- A numpy array of dimensions (num_images,hi, wi, 3) representing the image pixel values.
    """
    return np.array([imread(str(fn)).astype(np.float32) for fn in files])

def get_activations_from_files(files, sess, batch_size=50, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- files      : list of paths to image files. Images need to have same dimensions for all files.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """
    inception_layer = _get_inception_layer(sess)
    n_imgs = len(files)
    if batch_size > n_imgs:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = n_imgs
    n_batches = n_imgs//batch_size + 1
    pred_arr = np.empty((n_imgs,2048))
    for i in range(n_batches):
        if verbose:
            print("\rPropagating batch %d/%d" % (i+1, n_batches), end="", flush=True)
        start = i*batch_size
        if start+batch_size < n_imgs:
            end = start+batch_size
        else:
            end = n_imgs
        
        batch = load_image_batch(files[start:end])
        pred = sess.run(inception_layer, {'FID_Inception_Net/ExpandDims:0': batch})
        pred_arr[start:end] = pred.reshape(batch_size,-1)
        del batch #clean up memory
    if verbose:
        print(" done")
    return pred_arr
    
def calculate_activation_statistics_from_files(files, sess, batch_size=50, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- files      : list of paths to image files. Images need to have same dimensions for all files.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the available hardware.
    -- verbose     : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the incption model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the incption model.
    """
    act = get_activations_from_files(files, sess, batch_size, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma
    
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# The following functions aren't needed for calculating the FID
# they're just here to make this module work as a stand-alone script
# for calculating FID scores
#-------------------------------------------------------------------------------
def check_or_download_inception(inception_path):
    ''' Checks if the path to the inception file is valid, or downloads
        the file if it is not present. '''
    INCEPTION_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    if inception_path is None:
        inception_path = '/tmp'
    inception_path = pathlib.Path(inception_path)
    model_file = inception_path / 'classify_image_graph_def.pb'
    if not model_file.exists():
        print("Downloading Inception model")
        from urllib import request
        import tarfile
        fn, _ = request.urlretrieve(INCEPTION_URL)
        with tarfile.open(fn, mode='r') as f:
            f.extract('classify_image_graph_def.pb', str(model_file.parent))
    return str(model_file)


def _handle_path(path, sess, low_profile=False):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        path = pathlib.Path(path)
        files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        if low_profile:
            m, s = calculate_activation_statistics_from_files(files, sess)
        else:
            x = np.array([imread(str(fn)).astype(np.float32) for fn in files])
            m, s = calculate_activation_statistics(x, sess)
            del x #clean up memory
    return m, s


def calculate_fid_given_paths(paths, inception_path, low_profile=False):
    ''' Calculates the FID of two paths. '''
    inception_path = check_or_download_inception(inception_path)

    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError("Invalid path: %s" % p)

    create_inception_graph(str(inception_path))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        m1, s1 = _handle_path(paths[0], sess, low_profile=low_profile)
        m2, s2 = _handle_path(paths[1], sess, low_profile=low_profile)
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value

def check_or_download_inception(inception_path):
    ''' Checks if the path to the inception file is valid, or downloads
        the file if it is not present. '''
    INCEPTION_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    if inception_path is None:
        inception_path = '/tmp'
    inception_path = pathlib.Path(inception_path)
    model_file = inception_path / 'classify_image_graph_def.pb'
    if not model_file.exists():
        print("Downloading Inception model")
        from urllib import request
        import tarfile
        fn, _ = request.urlretrieve(INCEPTION_URL)
        with tarfile.open(fn, mode='r') as f:
            f.extract('classify_image_graph_def.pb', str(model_file.parent))
    return str(model_file)

inception_path = check_or_download_inception(None) # download inception network
create_inception_graph(inception_path)
