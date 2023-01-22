
import math
import logging
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import os
import urllib
import warnings

from functools import partial
from tqdm import tqdm

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import StdConv2dSame, DropPath, to_2tuple, trunc_normal_
from timm.models.resnet import resnet26d, resnet50d
from timm.models.resnetv2 import ResNetV2
from timm.models.registry import register_model
from torchvision import transforms

_logger = logging.getLogger(__name__)



class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


inception_unnormalize = transforms.Compose(
    [UnNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
)


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, mask=None):
        _x, attn = self.attn(self.norm1(x), mask=mask)
        x = x + self.drop_path(_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        no_patch_embed_bias=False,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False if no_patch_embed_bias else True,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        x = self.proj(x)
        return x


class PerceiverVisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        representation_size=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        add_norm_before_transformer=False,
        no_patch_embed_bias=False,
        use_video=False,
        max_frames=8,
        pretrained=False,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
            use_video (bool): Whether video will be used as an input to the module or not
            max_frames (int): Maximum frames for the video
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.add_norm_before_transformer = add_norm_before_transformer

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.patch_size = patch_size
        self.patch_dim = img_size // patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.total_output_patch = num_patches + 1
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.use_video = use_video

        if use_video:
            ## 64 is the upper bound hard-coded in temporal embedding
            self.temporal_embed = nn.Parameter(torch.zeros(1, 64, embed_dim))
            self.max_frames = max_frames

        if add_norm_before_transformer:
            self.pre_norm = norm_layer(embed_dim)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        if pretrained:
            raise NotImplementedError

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def mask_tokens(self, orig_image, feats):
        """
        Prepare masked tokens inputs/labels for masked patch prediction: 80% MASK, 10% random, 10% original.
        """
        img_unnorm = orig_image * 0.5 + 0.5
        _, _, ph, pw = self.patch_embed.proj.weight.shape
        with torch.no_grad():
            img_unnorm_patch = F.conv2d(
                img_unnorm,
                weight=torch.ones(3, 1, ph, pw).to(img_unnorm) / (ph * pw),
                bias=None,
                stride=(ph, pw),
                padding=0,
                groups=3,
            )
        labels = (
            ((img_unnorm_patch * 255).long().flatten(start_dim=2, end_dim=3))
            .permute(0, 2, 1)
            .contiguous()
        )

        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape[:-1], 0.15)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape[:-1], 0.8)).bool() & masked_indices
        )
        feats[indices_replaced] = self.mask_token.to(feats)

        return feats, labels

    def visual_embed(self, _x, max_image_len=200, mask_it=False, device=torch.device('cuda')):
        if len(_x.size()) == 5:
            use_video = True
            ori_shape = _x.size()
            video_mask = _x.reshape(ori_shape[0], ori_shape[1], -1).mean(-1)==0
            _x = _x.reshape(ori_shape[0]*ori_shape[1], ori_shape[2], ori_shape[3], ori_shape[4])
        else:
            use_video = False

        _, _, ph, pw = self.patch_embed.proj.weight.shape ## Kernel size

        # [bs, embed_dim, 14=[(img_h + 2*pad - kernel)/stride + 1], [(img_h + 2*pad - kernel)/stride + 1]]
        x = self.patch_embed(_x)
        # [bs, 1, img_h, img_w]. Why !=0 ??
        x_mask = torch.ones_like(_x.sum(dim=1) != 0).float()[:, None, :, :]
        # [bs, 1, 14, 14]. F.interpolate assumes first two channels are bs, channels
        x_mask = F.interpolate(x_mask, size=(x.shape[2], x.shape[3])).long()
        # [bs]: Stores 14
        x_h = x_mask[:, 0].sum(dim=1)[:, 0]
        # [bs]: Stores 14
        x_w = x_mask[:, 0].sum(dim=2)[:, 0]
        B, C, H, W = x.shape
        # pos_embed is a parameter with [1, num_patches+1, embed_dim]; patch_dim=img_size/patch_size
        # spatial_pos [1, embed_dim, 14, 14]
        spatial_pos = (
            self.pos_embed[:, 1:, :]
            .transpose(1, 2)
            .view(1, C, self.patch_dim, self.patch_dim)
        )
        # [bs, embed_dim, 14, 14]
        pos_embed = torch.cat(
            [
                F.pad(
                    F.interpolate(
                        spatial_pos, size=(h, w), mode="bilinear", align_corners=True,
                    ),
                    (0, W - w, 0, H - h), ## No need for this when calculations are correct.
                )
                for h, w in zip(x_h, x_w) ## x_h, x_w store h,w values of patches formed in bs
            ],
            dim=0,
        )

        # [bs, 14*14, embed_dim]
        pos_embed = pos_embed.flatten(2).transpose(1, 2)
        # [bs, 14*14, embed_dim]
        x = x.flatten(2).transpose(1, 2)
        # [bs, 14*14, 2]: Stores all the index values like (0,0), (0,1), ..., (13, 12), (13,13)
        patch_index = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(x_mask.shape[-2]), torch.arange(x_mask.shape[-1])
                ),
                dim=-1,
            )[None, None, :, :, :]
            .expand(x_mask.shape[0], x_mask.shape[1], -1, -1, -1)
            .flatten(1, 3)
        ).to(device)
        # make x_mask of shape [bs, 14*14]
        x_mask = x_mask.flatten(1)

        if mask_it:
            x, label = self.mask_tokens(_x, x)

        if (
            max_image_len < 0
            or max_image_len is None
            or not isinstance(max_image_len, int)
        ):
            # suppose aug is 800 x 1333, then, maximum effective res is 800 x 1333 (if one side gets bigger, the other will be constrained and be shrinked)
            # (800 // self.patch_size) * (1333 // self.patch_size) is the maximum number of patches that single image can get.
            # if self.patch_size = 32, 25 * 41 = 1025
            # if res is 384 x 640, 12 * 20 = 240
            eff = x_h * x_w
            max_image_len = eff.max()
        else:
            eff = x_h * x_w
            max_image_len = min(eff.max(), max_image_len)

        # [no. non-zero values, bs]
        valid_idx = x_mask.nonzero(as_tuple=False)
        # [no. zero values, bs]
        non_valid_idx = (1 - x_mask).nonzero(as_tuple=False)
        unique_rows = valid_idx[:, 0].unique() ## gives you all the valid data points in bs
        valid_row_idx = [valid_idx[valid_idx[:, 0] == u] for u in unique_rows] ## filter out
        non_valid_row_idx = [
            non_valid_idx[non_valid_idx[:, 0] == u] for u in unique_rows
        ] # can be empty but size is [non-valid, bs]

        valid_nums = [v.size(0) for v in valid_row_idx]
        non_valid_nums = [v.size(0) for v in non_valid_row_idx]
        pad_nums = [max_image_len - v for v in valid_nums]

        select = list()
        for i, (v, nv, p) in enumerate(zip(valid_nums, non_valid_nums, pad_nums)): ## iterating over batch size
            if p <= 0:
                valid_choice = torch.multinomial(torch.ones(v).float(), max_image_len) ## samples indices. for image, v is an int
                select.append(valid_row_idx[i][valid_choice])
            else:
                pad_choice = torch.multinomial(
                    torch.ones(nv).float(), p, replacement=True
                )
                select.append(
                    torch.cat(
                        [valid_row_idx[i], non_valid_row_idx[i][pad_choice]], dim=0,
                    )
                )

        # [image_patches, 2]
        select = torch.cat(select, dim=0)
        ## [select indices]
        # [bs, image_patches, embed_dim]
        x = x[select[:, 0], select[:, 1]].view(B, -1, C)
        ## [bs, image_patches]
        x_mask = x_mask[select[:, 0], select[:, 1]].view(B, -1)
        ## [bs, image_patches, 2]
        patch_index = patch_index[select[:, 0], select[:, 1]].view(B, -1, 2)
        ## [bs, image_patches, embed_dim]
        pos_embed = pos_embed[select[:, 0], select[:, 1]].view(B, -1, C)

        if mask_it:
            label = label[select[:, 0], select[:, 1]].view(B, -1, 3)

            label[x_mask == 0] = -100
            label = torch.cat(
                [torch.full((label.shape[0], 1, 3), -100).to(label), label,], dim=1,
            )

        ## [bs, 1, embed_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        ## [bs, image_patches+1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)
        ## concatenating the first pos_embed with other patch specific pos_embed. Padding it since we are using classification token
        ## [bs, image_patches+1, embed_dim]
        pos_embed = torch.cat(
            (self.pos_embed[:, 0, :][:, None, :].expand(B, -1, -1), pos_embed), dim=1
        )

        ## adding positional embedding whereas concatenating classification token
        x = x + pos_embed

        if use_video:
            x_mask = x_mask.view(ori_shape[0], ori_shape[1], x_mask.size(1))
            x_mask = torch.cat([torch.ones(ori_shape[0], ori_shape[1], 1).to(x_mask), x_mask], dim=-1)
            x_mask[video_mask]=0
            x_mask = x_mask.view(ori_shape[0], -1)

            ## make it like [video batch, t*image_patches, embed_dim]
            x = x.view(ori_shape[0], ori_shape[1] * x.size(1), x.size(-1))
            x += torch.repeat_interleave(self.temporal_embed[:, :self.max_frames], x.size(1)//ori_shape[1], dim=1)
        else:
            x_mask = torch.cat([torch.ones(x_mask.shape[0], 1).to(x_mask), x_mask], dim=1) ## to make it [bs, num_patches+1]


        ## Dropouts
        x = self.pos_drop(x)

        if self.add_norm_before_transformer:
            x = self.pre_norm(x)

#         x_mask = torch.cat([torch.ones(x_mask.shape[0], 1).to(x_mask), x_mask], dim=1)

        if mask_it:
            return x, x_mask, (patch_index, (H, W)), label, None
        else:
            return x, x_mask, (patch_index, (H, W)), None, None

    def forward(self, x, x_mask):

        for blk in self.blocks:
            x, _ = blk(x, mask=x_mask)

        x = self.norm(x)
        return x

def resize_pos_embed(posemb, posemb_new):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info("Resized position embedding: %s to %s", posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if True:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    _logger.info("Position embedding grid-size from %s to %s", gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode="bilinear")
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if "model" in state_dict:
        # For deit models
        state_dict = state_dict["model"]
    for k, v in state_dict.items():
        if "patch_embed.proj.weight" in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == "pos_embed" and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(v, model.pos_embed)
        out_dict[k] = v
    return out_dict


def _create_vision_transformer(variant, pretrained=False, distilled=False, **kwargs):
    default_cfg = default_cfgs[variant]
    default_num_classes = default_cfg["num_classes"]
    default_img_size = default_cfg["input_size"][-1]

    num_classes = kwargs.pop("num_classes", default_num_classes)
    img_size = kwargs.pop("img_size", default_img_size)
    repr_size = kwargs.pop("representation_size", None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None

    model_cls = DistilledPerceiverVisionTransformer if distilled else PerceiverVisionTransformer
    model = model_cls(
        img_size=img_size,
        num_classes=num_classes,
        representation_size=repr_size,
        **kwargs,
    )
    model.default_cfg = default_cfg

    if pretrained:
        load_pretrained(
            model,
            num_classes=num_classes,
            in_chans=kwargs.get("in_chans", 3),
            filter_fn=partial(checkpoint_filter_fn, model=model),
            strict=True,
        )
    return model


@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    """ My custom 'small' ViT model. Depth=8, heads=8= mlp_ratio=3."""
    model_kwargs = dict(
        patch_size=16,
        embed_dim=768,
        depth=8,
        num_heads=8,
        mlp_ratio=3.0,
        qkv_bias=False,
        norm_layer=nn.LayerNorm,
        **kwargs,
    )
    if pretrained:
        # NOTE my scale was wrong for original weights, leaving this here until I have better ones for this model
        model_kwargs.setdefault("qk_scale", 768 ** -0.5)
    model = _create_vision_transformer(
        "vit_small_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_base_patch32_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929). No pretrained weights.
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch32_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_base_patch16_384(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch16_384", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_base_patch32_384(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch32_384", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_large_patch16_224(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(
        "vit_large_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_large_patch32_224(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929). No pretrained weights.
    """
    model_kwargs = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(
        "vit_large_patch32_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_large_patch16_384(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(
        "vit_large_patch16_384", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_large_patch32_384(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(
        "vit_large_patch32_384", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_base_patch16_224_in21k(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        representation_size=768,
        **kwargs,
    )
    model = _create_vision_transformer(
        "vit_base_patch16_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_base_patch32_224_in21k(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=32,
        embed_dim=768,
        depth=12,
        num_heads=12,
        representation_size=768,
        **kwargs,
    )
    model = _create_vision_transformer(
        "vit_base_patch32_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_large_patch16_224_in21k(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        representation_size=1024,
        **kwargs,
    )
    model = _create_vision_transformer(
        "vit_large_patch16_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_large_patch32_224_in21k(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=32,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        representation_size=1024,
        **kwargs,
    )
    model = _create_vision_transformer(
        "vit_large_patch32_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_huge_patch14_224_in21k(pretrained=False, **kwargs):
    """ ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model_kwargs = dict(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        representation_size=1280,
        **kwargs,
    )
    model = _create_vision_transformer(
        "vit_huge_patch14_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_base_resnet50_224_in21k(pretrained=False, **kwargs):
    """ R50+ViT-B/16 hybrid model from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    # create a ResNetV2 w/o pre-activation, that uses StdConv and GroupNorm and has 3 stages, no head
    backbone = ResNetV2(
        layers=(3, 4, 9),
        num_classes=0,
        global_pool="",
        in_chans=kwargs.get("in_chans", 3),
        preact=False,
        stem_type="same",
        conv_layer=StdConv2dSame,
    )
    model_kwargs = dict(
        embed_dim=768,
        depth=12,
        num_heads=12,
        hybrid_backbone=backbone,
        representation_size=768,
        **kwargs,
    )
    model = _create_vision_transformer(
        "vit_base_resnet50_224_in21k", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_base_resnet50_384(pretrained=False, **kwargs):
    """ R50+ViT-B/16 hybrid from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    # create a ResNetV2 w/o pre-activation, that uses StdConv and GroupNorm and has 3 stages, no head
    backbone = ResNetV2(
        layers=(3, 4, 9),
        num_classes=0,
        global_pool="",
        in_chans=kwargs.get("in_chans", 3),
        preact=False,
        stem_type="same",
        conv_layer=StdConv2dSame,
    )
    model_kwargs = dict(
        embed_dim=768, depth=12, num_heads=12, hybrid_backbone=backbone, **kwargs
    )
    model = _create_vision_transformer(
        "vit_base_resnet50_384", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_small_resnet26d_224(pretrained=False, **kwargs):
    """ Custom ViT small hybrid w/ ResNet26D stride 32. No pretrained weights.
    """
    backbone = resnet26d(
        pretrained=pretrained,
        in_chans=kwargs.get("in_chans", 3),
        features_only=True,
        out_indices=[4],
    )
    model_kwargs = dict(
        embed_dim=768,
        depth=8,
        num_heads=8,
        mlp_ratio=3,
        hybrid_backbone=backbone,
        **kwargs,
    )
    model = _create_vision_transformer(
        "vit_small_resnet26d_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_small_resnet50d_s3_224(pretrained=False, **kwargs):
    """ Custom ViT small hybrid w/ ResNet50D 3-stages, stride 16. No pretrained weights.
    """
    backbone = resnet50d(
        pretrained=pretrained,
        in_chans=kwargs.get("in_chans", 3),
        features_only=True,
        out_indices=[3],
    )
    model_kwargs = dict(
        embed_dim=768,
        depth=8,
        num_heads=8,
        mlp_ratio=3,
        hybrid_backbone=backbone,
        **kwargs,
    )
    model = _create_vision_transformer(
        "vit_small_resnet50d_s3_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_base_resnet26d_224(pretrained=False, **kwargs):
    """ Custom ViT base hybrid w/ ResNet26D stride 32. No pretrained weights.
    """
    backbone = resnet26d(
        pretrained=pretrained,
        in_chans=kwargs.get("in_chans", 3),
        features_only=True,
        out_indices=[4],
    )
    model_kwargs = dict(
        embed_dim=768, depth=12, num_heads=12, hybrid_backbone=backbone, **kwargs
    )
    model = _create_vision_transformer(
        "vit_base_resnet26d_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_base_resnet50d_224(pretrained=False, **kwargs):
    """ Custom ViT base hybrid w/ ResNet50D stride 32. No pretrained weights.
    """
    backbone = resnet50d(
        pretrained=pretrained,
        in_chans=kwargs.get("in_chans", 3),
        features_only=True,
        out_indices=[4],
    )
    model_kwargs = dict(
        embed_dim=768, depth=12, num_heads=12, hybrid_backbone=backbone, **kwargs
    )
    model = _create_vision_transformer(
        "vit_base_resnet50d_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_deit_tiny_patch16_224(pretrained=False, **kwargs):
    """ DeiT-tiny model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer(
        "vit_deit_tiny_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_deit_small_patch16_224(pretrained=False, **kwargs):
    """ DeiT-small model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer(
        "vit_deit_small_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_deit_base_patch16_224(pretrained=False, **kwargs):
    """ DeiT base model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_deit_base_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_deit_base_patch16_384(pretrained=False, **kwargs):
    """ DeiT base model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_deit_base_patch16_384", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    """ DeiT-tiny distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer(
        "vit_deit_tiny_distilled_patch16_224",
        pretrained=pretrained,
        distilled=True,
        **model_kwargs,
    )
    return model


@register_model
def vit_deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    """ DeiT-small distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer(
        "vit_deit_small_distilled_patch16_224",
        pretrained=pretrained,
        distilled=True,
        **model_kwargs,
    )
    return model


@register_model
def vit_deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    """ DeiT-base distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_deit_base_distilled_patch16_224",
        pretrained=pretrained,
        distilled=True,
        **model_kwargs,
    )
    return model


@register_model
def vit_deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    """ DeiT-base distilled model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_deit_base_distilled_patch16_384",
        pretrained=pretrained,
        distilled=True,
        **model_kwargs,
    )
    return model
