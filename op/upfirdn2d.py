import torch
from torch.nn import functional as F


def upfirdn2d(x, kernel, up=1, down=1, pad=(0, 0)):
    """Pure PyTorch upfirdn2d.

    Args:
      x:      Tensor [N, C, H, W].
      kernel: 2D FIR kernel tensor [kh, kw] or 1D that we make separable.
      up:     Integer upsampling factor (same in x & y).
      down:   Integer downsampling factor (same in x & y).
      pad:    Tuple (pad_x0, pad_x1); same padding used in y.

    Returns:
      Tensor [N, C, H_out, W_out] with upsample + FIR filter + downsample.
    """
    if not torch.is_tensor(kernel):
        kernel = torch.tensor(kernel, dtype=x.dtype, device=x.device)

    # Make kernel 2D if 1D is provided (separable).
    if kernel.ndim == 1:
        kernel = kernel[:, None] * kernel[None, :]

    up_x = up_y = int(up)
    down_x = down_y = int(down)
    pad_x0, pad_x1 = pad
    pad_y0, pad_y1 = pad_x0, pad_x1

    return _upfirdn2d_native(
        x, kernel, up_x, up_y, down_x, down_y,
        pad_x0, pad_x1, pad_y0, pad_y1
    )


def _upfirdn2d_native(
    x, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    """Native (no C++ extension) implementation.

    This is adapted from the StyleGAN2 + original Score-SDE implementation,
    but runs entirely in PyTorch and works on both CPU and GPU tensors.
    """
    # Input: [N, C, H, W]
    n, c, in_h, in_w = x.shape

    # Reshape to [N*C, H, W, 1] so we can treat channels as "minor" dimension
    # as in the original implementation.
    x = x.reshape(-1, in_h, in_w, 1)
    _, in_h, in_w, minor = x.shape
    kernel_h, kernel_w = kernel.shape

    # Upsample by inserting zeros.
    # x: [NC, H, W, 1] -> [NC, H, up_y, W, up_x, 1]
    x = x.view(-1, in_h, 1, in_w, 1, minor)
    if up_x > 1 or up_y > 1:
        x = F.pad(
            x,
            pad=(0, 0,          # minor dim
                 0, up_x - 1,   # width dim (W)
                 0, 0,          # in_w dim, already handled
                 0, up_y - 1),  # height dim (H)
        )
    x = x.view(-1, in_h * up_y, in_w * up_x, minor)

    # Pad spatially (can be positive or negative).
    x = F.pad(
        x,
        pad=(
            0, 0,
            max(pad_x0, 0), max(pad_x1, 0),
            max(pad_y0, 0), max(pad_y1, 0),
        ),
    )

    # Crop if padding is negative.
    y_start = max(-pad_y0, 0)
    y_end = x.shape[1] - max(-pad_y1, 0)
    x_start = max(-pad_x0, 0)
    x_end = x.shape[2] - max(-pad_x1, 0)
    x = x[:, y_start:y_end, x_start:x_end, :]

    # Now shape is [NC, H', W', minor]. Move minor to channel for conv2d.
    x = x.permute(0, 3, 1, 2)  # -> [NC, minor, H', W']

    # Prepare kernel as conv filter, flipped as usual.
    k = torch.flip(kernel, dims=[0, 1]).to(x.dtype).to(x.device)
    k = k.view(1, 1, kernel_h, kernel_w)
    k = k.repeat(minor, 1, 1, 1)  # groups = minor

    # Convolve with groups=minor.
    x = x.reshape(-1, minor, x.shape[2], x.shape[3])
    out = F.conv2d(x, k, groups=minor)

    # Reshape back to [NC, H_f, W_f, minor]
    out = out.view(
        -1,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
        minor,
    )

    # Downsample by striding.
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

    # Final shape [N, C, H_out, W_out]
    return out.view(-1, c, out_h, out_w)

# import os

# import torch
# from torch.nn import functional as F
# from torch.autograd import Function
# from torch.utils.cpp_extension import load


# module_path = os.path.dirname(__file__)
# upfirdn2d_op = load(
#     "upfirdn2d",
#     sources=[
#         os.path.join(module_path, "upfirdn2d.cpp"),
#         os.path.join(module_path, "upfirdn2d_kernel.cu"),
#     ],
# )


# class UpFirDn2dBackward(Function):
#     @staticmethod
#     def forward(
#         ctx, grad_output, kernel, grad_kernel, up, down, pad, g_pad, in_size, out_size
#     ):

#         up_x, up_y = up
#         down_x, down_y = down
#         g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1 = g_pad

#         grad_output = grad_output.reshape(-1, out_size[0], out_size[1], 1)

#         grad_input = upfirdn2d_op.upfirdn2d(
#             grad_output,
#             grad_kernel,
#             down_x,
#             down_y,
#             up_x,
#             up_y,
#             g_pad_x0,
#             g_pad_x1,
#             g_pad_y0,
#             g_pad_y1,
#         )
#         grad_input = grad_input.view(in_size[0], in_size[1], in_size[2], in_size[3])

#         ctx.save_for_backward(kernel)

#         pad_x0, pad_x1, pad_y0, pad_y1 = pad

#         ctx.up_x = up_x
#         ctx.up_y = up_y
#         ctx.down_x = down_x
#         ctx.down_y = down_y
#         ctx.pad_x0 = pad_x0
#         ctx.pad_x1 = pad_x1
#         ctx.pad_y0 = pad_y0
#         ctx.pad_y1 = pad_y1
#         ctx.in_size = in_size
#         ctx.out_size = out_size

#         return grad_input

#     @staticmethod
#     def backward(ctx, gradgrad_input):
#         kernel, = ctx.saved_tensors

#         gradgrad_input = gradgrad_input.reshape(-1, ctx.in_size[2], ctx.in_size[3], 1)

#         gradgrad_out = upfirdn2d_op.upfirdn2d(
#             gradgrad_input,
#             kernel,
#             ctx.up_x,
#             ctx.up_y,
#             ctx.down_x,
#             ctx.down_y,
#             ctx.pad_x0,
#             ctx.pad_x1,
#             ctx.pad_y0,
#             ctx.pad_y1,
#         )
#         # gradgrad_out = gradgrad_out.view(ctx.in_size[0], ctx.out_size[0], ctx.out_size[1], ctx.in_size[3])
#         gradgrad_out = gradgrad_out.view(
#             ctx.in_size[0], ctx.in_size[1], ctx.out_size[0], ctx.out_size[1]
#         )

#         return gradgrad_out, None, None, None, None, None, None, None, None


# class UpFirDn2d(Function):
#     @staticmethod
#     def forward(ctx, input, kernel, up, down, pad):
#         up_x, up_y = up
#         down_x, down_y = down
#         pad_x0, pad_x1, pad_y0, pad_y1 = pad

#         kernel_h, kernel_w = kernel.shape
#         batch, channel, in_h, in_w = input.shape
#         ctx.in_size = input.shape

#         input = input.reshape(-1, in_h, in_w, 1)

#         ctx.save_for_backward(kernel, torch.flip(kernel, [0, 1]))

#         out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
#         out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
#         ctx.out_size = (out_h, out_w)

#         ctx.up = (up_x, up_y)
#         ctx.down = (down_x, down_y)
#         ctx.pad = (pad_x0, pad_x1, pad_y0, pad_y1)

#         g_pad_x0 = kernel_w - pad_x0 - 1
#         g_pad_y0 = kernel_h - pad_y0 - 1
#         g_pad_x1 = in_w * up_x - out_w * down_x + pad_x0 - up_x + 1
#         g_pad_y1 = in_h * up_y - out_h * down_y + pad_y0 - up_y + 1

#         ctx.g_pad = (g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1)

#         out = upfirdn2d_op.upfirdn2d(
#             input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
#         )
#         # out = out.view(major, out_h, out_w, minor)
#         out = out.view(-1, channel, out_h, out_w)

#         return out

#     @staticmethod
#     def backward(ctx, grad_output):
#         kernel, grad_kernel = ctx.saved_tensors

#         grad_input = UpFirDn2dBackward.apply(
#             grad_output,
#             kernel,
#             grad_kernel,
#             ctx.up,
#             ctx.down,
#             ctx.pad,
#             ctx.g_pad,
#             ctx.in_size,
#             ctx.out_size,
#         )

#         return grad_input, None, None, None, None


# def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
#     if input.device.type == "cpu":
#         out = upfirdn2d_native(
#             input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1]
#         )

#     else:
#         out = UpFirDn2d.apply(
#             input, kernel, (up, up), (down, down), (pad[0], pad[1], pad[0], pad[1])
#         )

#     return out


# def upfirdn2d_native(
#     input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
# ):
#     _, channel, in_h, in_w = input.shape
#     input = input.reshape(-1, in_h, in_w, 1)

#     _, in_h, in_w, minor = input.shape
#     kernel_h, kernel_w = kernel.shape

#     out = input.view(-1, in_h, 1, in_w, 1, minor)
#     out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
#     out = out.view(-1, in_h * up_y, in_w * up_x, minor)

#     out = F.pad(
#         out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
#     )
#     out = out[
#         :,
#         max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
#         max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
#         :,
#     ]

#     out = out.permute(0, 3, 1, 2)
#     out = out.reshape(
#         [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
#     )
#     w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
#     out = F.conv2d(out, w)
#     out = out.reshape(
#         -1,
#         minor,
#         in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
#         in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
#     )
#     out = out.permute(0, 2, 3, 1)
#     out = out[:, ::down_y, ::down_x, :]

#     out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
#     out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

#     return out.view(-1, channel, out_h, out_w)
