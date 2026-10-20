"""Opt-in true P8 feature path; original encoder and HeightSplat untouched."""

import torch
from torch.nn import functional as F
from models.m5b_cached_encoder import cache_rgb
from datasets.inputs import validate_tensor_inputs


def p8_features(encoder, images):
    b, n, c, h, w = images.shape
    x = (images.reshape(b * n, c, h, w) - encoder.mean) / encoder.std
    r = encoder.backbone
    x = r.maxpool(r.relu(r.bn1(r.conv1(x))))
    x = r.layer1(x)
    p2 = r.layer2(x)
    p3 = r.layer3(p2)
    p4 = r.layer4(p3)
    # Real stride8 layer2 signal, not interpolated final stride16 context.
    f = (
        encoder.lateral[0](p2)
        + F.interpolate(
            encoder.lateral[1](p3),
            size=p2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        + F.interpolate(
            encoder.lateral[2](p4),
            size=p2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    )
    f = encoder.output(f)
    return f.reshape(b, n, *f.shape[1:])


def cache_p8(net, inputs, amp=True):
    validate_tensor_inputs(inputs)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        f = p8_features(net.encoder, inputs["images"])
        b, n, c, h, w = f.shape
        ih, iw = inputs["images"].shape[-2:]
        K = inputs["intrinsics"]
        T = inputs["cam_to_ego"]
        camera = torch.stack(
            (
                K[..., 0, 0] / iw,
                K[..., 1, 1] / ih,
                K[..., 0, 2] / iw,
                K[..., 1, 2] / ih,
            ),
            -1,
        )
        ext = torch.cat((T[..., :3, :3].flatten(-2), T[..., :3, 3] / 10), -1)
        cond = net.camera_mlp(torch.cat((camera, ext), -1)).reshape(b * n, c, 1, 1)
        logits = net.depth_head(f.flatten(0, 1) + cond).reshape(b, n, 48, h, w)
        return dict(
            features=f.detach(),
            probs=logits.float().softmax(2).detach(),
            K=K,
            T=T,
            image_hw=(ih, iw),
        )
