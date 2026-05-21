import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBNReLU(nn.Module):
    """Convolution followed by BatchNorm and ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class _DSConv(nn.Module):
    """Depthwise separable convolution."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class _DWConv(nn.Module):
    """Depthwise convolution."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class LinearBottleneck(nn.Module):
    """Linear bottleneck block used in MobileNetV2."""

    def __init__(self, in_channels, out_channels, t=6, stride=2):
        super().__init__()

        self.use_shortcut = stride == 1 and in_channels == out_channels

        self.block = nn.Sequential(
            _ConvBNReLU(in_channels, in_channels * t, kernel_size=1),
            _DWConv(in_channels * t, in_channels * t, stride=stride),
            nn.Conv2d(
                in_channels * t,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.block(x)

        if self.use_shortcut:
            out = x + out

        return out


class PyramidPooling(nn.Module):
    """Pyramid pooling module."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        inter_channels = in_channels // 4

        self.conv1 = _ConvBNReLU(in_channels, inter_channels, kernel_size=1)
        self.conv2 = _ConvBNReLU(in_channels, inter_channels, kernel_size=1)
        self.conv3 = _ConvBNReLU(in_channels, inter_channels, kernel_size=1)
        self.conv4 = _ConvBNReLU(in_channels, inter_channels, kernel_size=1)

        self.out = _ConvBNReLU(in_channels * 2, out_channels, kernel_size=1)

    @staticmethod
    def pool(x, size):
        return F.adaptive_avg_pool2d(x, output_size=size)

    @staticmethod
    def upsample(x, size):
        return F.interpolate(
            x,
            size=size,
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, x):
        size = x.shape[2:]

        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)

        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)

        return x


class LearningToDownsample(nn.Module):
    """Learning-to-downsample module."""

    def __init__(self, in_channels=3, dw_channels1=32, dw_channels2=48, out_channels=64):
        super().__init__()

        self.conv = _ConvBNReLU(
            in_channels,
            dw_channels1,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.dsconv1 = _DSConv(
            dw_channels1,
            dw_channels2,
            stride=2,
        )

        self.dsconv2 = _DSConv(
            dw_channels2,
            out_channels,
            stride=2,
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)

        return x


class GlobalFeatureExtractor(nn.Module):
    """Global feature extractor module."""

    def __init__(
        self,
        in_channels=64,
        block_channels=(64, 96, 128),
        out_channels=128,
        t=6,
        num_blocks=(3, 3, 3),
    ):
        super().__init__()

        self.bottleneck1 = self._make_layer(
            in_channels=in_channels,
            out_channels=block_channels[0],
            num_blocks=num_blocks[0],
            t=t,
            stride=2,
        )

        self.bottleneck2 = self._make_layer(
            in_channels=block_channels[0],
            out_channels=block_channels[1],
            num_blocks=num_blocks[1],
            t=t,
            stride=2,
        )

        self.bottleneck3 = self._make_layer(
            in_channels=block_channels[1],
            out_channels=block_channels[2],
            num_blocks=num_blocks[2],
            t=t,
            stride=1,
        )

        self.ppm = PyramidPooling(
            in_channels=block_channels[2],
            out_channels=out_channels,
        )

    @staticmethod
    def _make_layer(in_channels, out_channels, num_blocks, t=6, stride=1):
        layers = [
            LinearBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                t=t,
                stride=stride,
            )
        ]

        for _ in range(1, num_blocks):
            layers.append(
                LinearBottleneck(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    t=t,
                    stride=1,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.ppm(x)

        return x


class FeatureFusionModule(nn.Module):
    """Feature fusion module."""

    def __init__(self, higher_in_channels, lower_in_channels, out_channels):
        super().__init__()

        self.dwconv = _DWConv(
            lower_in_channels,
            out_channels,
            stride=1,
        )

        self.conv_lower_res = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.conv_higher_res = nn.Sequential(
            nn.Conv2d(
                higher_in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, higher_res_feature, lower_res_feature):
        _, _, h, w = higher_res_feature.shape

        lower_res_feature = F.interpolate(
            lower_res_feature,
            size=(h, w),
            mode="bilinear",
            align_corners=True,
        )

        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        higher_res_feature = self.conv_higher_res(higher_res_feature)

        out = higher_res_feature + lower_res_feature
        out = self.relu(out)

        return out


class Classifier(nn.Module):
    """Classifier head."""

    def __init__(self, in_channels, num_classes, stride=1):
        super().__init__()

        self.dsconv1 = _DSConv(
            in_channels,
            in_channels,
            stride=stride,
        )

        self.dsconv2 = _DSConv(
            in_channels,
            in_channels,
            stride=stride,
        )

        self.conv = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Conv2d(
                in_channels,
                num_classes,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)

        return x


class FastSCNN(nn.Module):
    """
    Standard uncompressed Fast-SCNN model.

    Architecture:
    - Learning-to-downsample module
    - Global feature extractor
    - Feature fusion module
    - Classifier head
    """

    def __init__(
        self,
        in_channels=3,
        n_classes=19,
        use_pretrained_weights=False,
        aux=False,
    ):
        super().__init__()

        if in_channels != 3:
            raise ValueError(f"Fast-SCNN requires 3 input channels, got {in_channels}.")

        if use_pretrained_weights:
            raise NotImplementedError(
                "Pretrained weights are not supported in this Fast-SCNN implementation."
            )

        self.aux = aux

        self.learning_to_downsample = LearningToDownsample(
            in_channels=in_channels,
            dw_channels1=32,
            dw_channels2=48,
            out_channels=64,
        )

        self.global_feature_extractor = GlobalFeatureExtractor(
            in_channels=64,
            block_channels=(64, 96, 128),
            out_channels=128,
            t=6,
            num_blocks=(3, 3, 3),
        )

        self.feature_fusion = FeatureFusionModule(
            higher_in_channels=64,
            lower_in_channels=128,
            out_channels=128,
        )

        self.classifier = Classifier(
            in_channels=128,
            num_classes=n_classes,
        )

        if self.aux:
            self.auxlayer = nn.Sequential(
                nn.Conv2d(
                    64,
                    32,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
                nn.Conv2d(
                    32,
                    n_classes,
                    kernel_size=1,
                ),
            )

    def forward(self, x):
        input_size = x.shape[2:]

        higher_res_features = self.learning_to_downsample(x)

        features = self.global_feature_extractor(higher_res_features)
        features = self.feature_fusion(higher_res_features, features)

        out = self.classifier(features)
        out = F.interpolate(
            out,
            size=input_size,
            mode="bilinear",
            align_corners=True,
        )

        if not self.training or not self.aux:
            return out

        auxout = self.auxlayer(higher_res_features)
        auxout = F.interpolate(
            auxout,
            size=input_size,
            mode="bilinear",
            align_corners=True,
        )

        return out, auxout


if __name__ == "__main__":
    model = FastSCNN(
        in_channels=3,
        n_classes=19,
        aux=True,
    )

    model.train()
    x = torch.randn(2, 3, 512, 1024)

    out, auxout = model(x)

    print("Training main output:", out.shape)
    print("Training auxiliary output:", auxout.shape)

    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 512, 1024)
        out = model(x)

    print("Evaluation output:", out.shape)