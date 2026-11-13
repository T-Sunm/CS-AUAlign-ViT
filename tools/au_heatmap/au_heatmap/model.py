"""
FANAU: AU heatmap regression with "adaptation layers" (Ntinou et al., IEEE T-AFFC,
"A Transfer Learning approach to Heatmap Regression for Action Unit intensity estimation").

Architecture copied from the evaluation code shipped with WarpedGANSpace
(https://github.com/chi0tzp/WarpedGANSpace, lib/evaluation/au_detector/hourglass.py),
which loads the checkpoint `disfa_adaptation_f0.pth`. Parameter names are kept identical
so that the checkpoint loads with strict=True.

Pipeline:
  image (3x256x256) -> frozen FAN (QFAN) -> 68 landmark heatmaps + conv4 features (128x64x64)
  -> two 1x1 branches, summed -> lightweight Hourglass -> 12 AU heatmaps (12x64x64)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)


def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, lightweight=False):
        super().__init__()
        conv = conv1x1 if lightweight else conv3x3
        self.conv1 = conv(in_planes, out_planes // 2)
        self.conv2 = conv(out_planes // 2, out_planes // 4)
        self.conv3 = conv(out_planes // 4, out_planes // 4)
        self.bn1 = nn.BatchNorm2d(out_planes // 2)
        self.bn2 = nn.BatchNorm2d(out_planes // 4)
        self.bn3 = nn.BatchNorm2d(out_planes // 4)
        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.ReLU6(True),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        out1 = F.relu6(self.bn1(self.conv1(x)), True)
        out2 = F.relu6(self.bn2(self.conv2(out1)), True)
        out3 = F.relu6(self.bn3(self.conv3(out2)), True)
        out3 = torch.cat((out1, out2, out3), 1)
        if self.downsample is not None:
            residual = self.downsample(residual)
        return out3 + residual


class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features, lightweight=False):
        super().__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.lightweight = lightweight
        self._generate_network(depth)

    def _generate_network(self, level):
        self.add_module(f"b1_{level}", ConvBlock(self.features, self.features, lightweight=self.lightweight))
        self.add_module(f"b2_{level}", ConvBlock(self.features, self.features))
        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module(f"b2_plus_{level}", ConvBlock(self.features, self.features))
        self.add_module(f"b3_{level}", ConvBlock(self.features, self.features))

    def _forward(self, level, inp):
        up1 = self._modules[f"b1_{level}"](inp)
        low1 = self._modules[f"b2_{level}"](F.max_pool2d(inp, 2, stride=2))
        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = self._modules[f"b2_plus_{level}"](low1)
        low3 = self._modules[f"b3_{level}"](low2)
        return up1 + F.interpolate(low3, scale_factor=2, mode="nearest")

    def forward(self, x):
        return self._forward(self.depth, x)


class QFAN(nn.Module):
    """Single-hourglass Face Alignment Network (68 landmarks)."""

    def __init__(self, num_modules=1, num_in=3, num_features=128, num_out=68, return_features=False):
        super().__init__()
        self.num_modules = num_modules
        self.num_features = num_features
        self.num_out = num_out
        self.return_features = return_features

        self.conv1 = nn.Conv2d(num_in, num_features // 2, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(num_features // 2)
        self.conv2 = ConvBlock(num_features // 2, num_features // 2)
        self.conv3 = ConvBlock(num_features // 2, num_features)
        self.conv4 = ConvBlock(num_features, num_features)

        for i in range(num_modules):
            self.add_module(f"m{i}", HourGlass(1, 4, num_features))
            self.add_module(f"top_m_{i}", ConvBlock(num_features, num_features))
            self.add_module(f"conv_last{i}", nn.Conv2d(num_features, num_features, kernel_size=1))
            self.add_module(f"bn_end{i}", nn.BatchNorm2d(num_features))
            self.add_module(f"l{i}", nn.Conv2d(num_features, num_out, kernel_size=1))
            if i < num_modules - 1:
                self.add_module(f"bl{i}", nn.Conv2d(num_features, num_features, kernel_size=1))
                self.add_module(f"al{i}", nn.Conv2d(num_out, num_features, kernel_size=1))

    def forward(self, x):
        features = []
        x = F.relu(self.bn1(self.conv1(x)), True)
        x = F.max_pool2d(self.conv2(x), 2, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)
        if self.return_features:
            features.append(x)

        previous = x
        outputs = []
        for i in range(self.num_modules):
            ll = self._modules[f"m{i}"](previous)
            ll = self._modules[f"top_m_{i}"](ll)
            ll = F.relu(self._modules[f"bn_end{i}"](self._modules[f"conv_last{i}"](ll)), True)
            tmp_out = self._modules[f"l{i}"](ll)
            outputs.append(tmp_out)
            if i < self.num_modules - 1:
                previous = previous + self._modules[f"bl{i}"](ll) + self._modules[f"al{i}"](tmp_out)

        return (outputs, features) if self.return_features else outputs


class FANAU(nn.Module):
    """FAN backbone + adaptation layers + lightweight Hourglass -> n_points AU heatmaps."""

    def __init__(self, num_features=128, n_points=12):
        super().__init__()
        self.num_modules = 1
        self.num_features = num_features
        self.fan = QFAN(num_modules=1, return_features=True)
        self.conv1 = nn.Sequential(nn.Conv2d(68, num_features, 1, 1), nn.BatchNorm2d(num_features), nn.ReLU6())
        self.conv2 = nn.Sequential(nn.Conv2d(num_features, num_features, 1, 1), nn.BatchNorm2d(num_features), nn.ReLU6())
        self.net = HourGlass(1, 4, num_features, lightweight=True)
        self.conv_last = nn.Sequential(nn.Conv2d(num_features, num_features, 1, 1), nn.BatchNorm2d(num_features), nn.ReLU6())
        self.l = nn.Conv2d(num_features, n_points, 1, 1)

    def forward(self, x, return_landmarks=False):
        self.fan.eval()
        output, features = self.fan(x)
        lmk_heatmaps = output[-1]                       # B x 68 x 64 x 64
        h = self.conv1(lmk_heatmaps) + self.conv2(features[0])
        h = self.net(h)
        h = self.conv_last(h)
        au_heatmaps = self.l(h)                         # B x n_points x 64 x 64
        if return_landmarks:
            return au_heatmaps, lmk_heatmaps
        return au_heatmaps
