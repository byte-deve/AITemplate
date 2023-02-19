#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import io
import unittest

import numpy as np
import torch
from aitemplate.compiler import compile_model
from aitemplate.compiler.base import Tensor

from aitemplate.testing import detect_target

from modeling.swin_transformer import SwinTransformer

from mmdet.models.builder import BACKBONES
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmcv.cnn import fuse_conv_bn

import torch
import numpy as np

def build_backbone(cfg):
    return BACKBONES.build(cfg)

def mark_output(y):
    if type(y) is not tuple:
        y = (y,)
    for i in range(len(y)):
        y[i]._attrs["is_output"] = True
        y[i]._attrs["name"] = "output_%d" % (i)
        y_shape = [d._attrs["values"][0] for d in y[i]._attrs["shape"]]
        print("output_{} shape: {}".format(i, y_shape))


def compile_swinT(
    model_name,
    batch_size=1,
    use_fp16_acc=True,
):
    assert model_name == "swin_tiny_patch4_window7_224"

    ait_model = SwinTransformer(
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

    ait_model.name_parameter_tensor()
    inputs_ait = Tensor(
        [batch_size, 256, 704, 3], name="input0", is_input=True
    )
    Y = ait_model(inputs_ait)
    mark_output(Y)

    target = detect_target(use_fp16_acc=use_fp16_acc)
    exe_module = compile_model(
        Y, target, "./tmp", "swin_tiny_patch4_window7_224"
    )
    return exe_module


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
    checkpoint = load_checkpoint(model, "./checkpoint/bevfusion-det.pth", map_location="cpu")
    model = fuse_conv_bn(model)
    model.cuda().eval()

    return model
    inputs = np.random.normal(0, 1, (6, 3, 256, 704)).astype("float32")
    image = torch.Tensor(inputs).cuda()
    out = model(image)

    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)

class VITVerification(unittest.TestCase):
    def test_vit(self):
        swint_pt = build_swinT().half()

        swint_ait = compile_swinT(
            model_name="swin_tiny_patch4_window7_224",
            batch_size=6,
            use_fp16_acc=True,
        )

        # prepare params
        params_pt = swint_pt.named_parameters()
        params_ait = {}
        for key, arr in params_pt:
            ait_key = key.replace(".", "_")
            if len(arr.shape) == 4:         # NCHW -> NHWC
                embed_dim, small_c, patch_h, patch_w = arr.shape
                assert small_c == 3 and patch_h == patch_w
                arr = arr.permute((0, 2, 3, 1)).contiguous()
                if detect_target().name() == "cuda":
                    conv0_w_pad = (
                        torch.zeros((embed_dim, patch_h, patch_w, 4))
                        .cuda()
                        .half()
                    )
                    conv0_w_pad[:, :, :, :3] = arr
                    arr = conv0_w_pad
            params_ait[f"{ait_key}"] = arr
        # params_ait["cls_token_mask"] = (
        #     torch.zeros((batch_size, 1, embed_dim)).cuda().half()
        # )
        # if detect_target().name() == "cuda":
        #     ait_key = "attn_cu_length"
        #     for i in range(depth):
        #         prefix = "blocks_%d" % (i)
        #         cu_len = np.cumsum([0] + [seqlen] * batch_size).astype("int32")
        #         params_ait[f"{prefix}_{ait_key}"] = torch.from_numpy(cu_len).cuda()

        # set weights
        for name, weight in params_ait.items():
            swint_ait.set_constant_with_tensor(name, weight)

        batch_size = 6
        with torch.no_grad():
            x_pt = (
                torch.rand(
                    (batch_size, 3, 256, 704),
                    dtype=torch.float16,
                    device="cuda",
                )
                * 255
            )
            x_ait = x_pt.permute(0, 2, 3, 1).contiguous()
            y_pt = swint_pt.patch_embed(x_pt)[0]
            y_ait = torch.empty_like(y_pt)
            swint_ait.run_with_tensors([x_ait], [y_ait])
            torch.testing.assert_close(y_ait, y_pt, atol=1e-1, rtol=1e-1)


if __name__ == "__main__":
    unittest.main()
