import torch as pt
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(pt.ones(normalized_shape))
        self.bias = nn.Parameter(pt.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / pt.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, scale_init=None):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = LayerNorm(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, 1)
        self.gamma = None
        if scale_init:
            self.gamma = nn.Parameter(scale_init * pt.ones((1)), requires_grad=True)

    def forward(self, x):
        res = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        scale = (1 / 4) if self.gamma is None else pt.pow(self.gamma, 2)
        return res + scale * x


class ConvNeXtIsotropic(nn.Module):
    def __init__(
        self,
        in_chans=3,
        num_classes=10,
        depths=8,
        dims=64,
        layer_scale_init=None,
    ):
        super().__init__()
        self.downsample_layers = nn.Sequential(
            nn.Conv2d(in_chans, dims, kernel_size=4, stride=4),
            LayerNorm(dims),
        )
        self.stages = nn.ModuleList(
            [ConvNeXtBlock(dims, scale_init=layer_scale_init) for _ in range(depths)]
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(dims),
            nn.Linear(dims, num_classes),
        )

    def forward(self, x):
        x = self.downsample_layers(x)
        intermediate_features = [x]
        for stage in self.stages:
            x = stage(x)
            intermediate_features.append(x)
        x = self.head(x)
        return x, intermediate_features


def load_checkpoint(path: str, device: str = "cpu"):
    if not path.startswith("https"):
        return pt.load(path, map_location=device)
    return pt.hub.load_state_dict_from_url(path, map_location=device, check_hash=True)


def make_net(depth, width, layer_scale_init):
    net = ConvNeXtIsotropic(
        depths=depth,
        dims=width,
        layer_scale_init=layer_scale_init,
    )
    if pt.cuda.is_available():
        net = net.cuda()
    net = net.to(memory_format=pt.channels_last)
    return net


if __name__ == "__main__":
    from sys import argv

    assert len(argv) > 1
    path = argv[1]
    print(path)
    model: nn.Module = load_checkpoint(path)["model"]
    for name, layer in model.items():
        if "gamma" in name:
            print(name.replace(".gamma", ""), layer.item())
