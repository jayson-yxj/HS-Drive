"""Frozen RGB/depth cache, original Context/Splat/BEV path. No GT accepted."""

import torch


def cache_rgb(net, inputs, *, amp=True):
    from datasets.inputs import validate_tensor_inputs

    validate_tensor_inputs(inputs)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        f = net.encoder(inputs["images"])
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


def cached_forward(
    net, cache, *, probabilities=None, context=None, return_parts=False, amp=True
):
    f = cache["features"]
    b, n, c, h, w = f.shape
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        ctx = (
            net.context_head(f.flatten(0, 1)).reshape(b, n, 64, h, w)
            if context is None
            else context
        )
        height, mass = net.splat(
            ctx,
            cache["probs"] if probabilities is None else probabilities,
            cache["K"],
            cache["T"],
            cache["image_hw"],
        )
        bev = net.bev_encoder(torch.cat((height.flatten(1, 2), torch.log1p(mass)), 1))
    return dict(bev=bev, context=ctx, height=height, mass=mass) if return_parts else bev
