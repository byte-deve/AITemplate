from mmdet.models.builder import BACKBONES
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmcv.cnn import fuse_conv_bn

import torch
import numpy as np

def build_backbone(cfg):
    return BACKBONES.build(cfg)

def build_swinT():
    bevfusion_config = dict(
        type="SwinTransformer",
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=[1, 2, 3],
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type="Pretrained", checkpoint="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth"))
    model = build_backbone(bevfusion_config)
    checkpoint = load_checkpoint(model, "/home/tonyy/yt_ws/66_nvbugs/Robotics/03_bevfusion/CUDA-BEVFusion/tool/checkpoint/bevfusion-det.pth", map_location="cpu")
    model = fuse_conv_bn(model)
    model.cuda().eval()

    inputs = np.random.normal(0, 1, (6, 3, 256, 704)).astype("float32")
    image = torch.Tensor(inputs).cuda()
    out = model(image)

    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)

if __name__ == "__main__":
    build_swinT()