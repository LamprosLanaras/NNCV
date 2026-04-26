import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerModel


class PSPModule(nn.Module):
    """Pyramid pooling module used to capture multi-scale global context."""

    def __init__(self, in_channels, out_channels, pool_scales=(1, 2, 3, 6)):
        super().__init__()

        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
                for scale in pool_scales
            ]
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels + len(pool_scales) * out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.shape[2:]

        priors = [
            F.interpolate(
                stage(x),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            for stage in self.stages
        ]

        x = torch.cat([x] + priors, dim=1)
        x = self.bottleneck(x)

        return x


class UPerNetDecoder(nn.Module):
    """UPerNet decoder with PPM and FPN fusion."""

    def __init__(
        self,
        encoder_channels=(64, 128, 320, 512),
        fpn_dim=512,
        n_classes=19,
    ):
        super().__init__()

        self.ppm = PSPModule(
            in_channels=encoder_channels[3],
            out_channels=fpn_dim,
        )

        self.lat1 = nn.Conv2d(encoder_channels[0], fpn_dim, kernel_size=1)
        self.lat2 = nn.Conv2d(encoder_channels[1], fpn_dim, kernel_size=1)
        self.lat3 = nn.Conv2d(encoder_channels[2], fpn_dim, kernel_size=1)

        self.smooth1 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1)

        self.classifier = nn.Sequential(
            nn.Conv2d(
                fpn_dim * 4,
                fpn_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_dim, n_classes, kernel_size=1),
        )

    def forward(self, features):
        c1, c2, c3, c4 = features

        p4 = self.ppm(c4)

        p3 = self.lat3(c3) + F.interpolate(
            p4,
            size=c3.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        p3 = self.smooth3(p3)

        p2 = self.lat2(c2) + F.interpolate(
            p3,
            size=c2.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        p2 = self.smooth2(p2)

        p1 = self.lat1(c1) + F.interpolate(
            p2,
            size=c1.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        p1 = self.smooth1(p1)

        p2_up = F.interpolate(
            p2,
            size=p1.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        p3_up = F.interpolate(
            p3,
            size=p1.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        p4_up = F.interpolate(
            p4,
            size=p1.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat([p1, p2_up, p3_up, p4_up], dim=1)
        x = self.classifier(x)

        return x


class UperFormer(nn.Module):
    """
    UPerFormer model.

    Architecture:
    - SegFormer-B5 encoder
    - UPerNet decoder with PPM and FPN fusion
    """

    def __init__(
        self,
        in_channels=3,
        n_classes=19,
        use_pretrained_weights=False,
    ):
        super().__init__()

        if in_channels != 3:
            raise ValueError(f"SegFormer requires 3 input channels, got {in_channels}.")

        if use_pretrained_weights:
            self.encoder = SegformerModel.from_pretrained("nvidia/mit-b5")
        else:
            config = SegformerConfig(
                num_channels=in_channels,
                depths=[3, 6, 40, 3],
                hidden_sizes=[64, 128, 320, 512],
            )
            self.encoder = SegformerModel(config)

        self.decoder = UPerNetDecoder(
            encoder_channels=(64, 128, 320, 512),
            fpn_dim=512,
            n_classes=n_classes,
        )

    def forward(self, x):
        input_size = x.shape[-2:]

        outputs = self.encoder(
            pixel_values=x,
            output_hidden_states=True,
        )
        features = outputs.hidden_states

        logits = self.decoder(features)

        if logits.shape[-2:] != input_size:
            logits = F.interpolate(
                logits,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )

        return logits


if __name__ == "__main__":
    model = UperFormer(
        in_channels=3,
        n_classes=19,
        use_pretrained_weights=False,
    )

    model.eval()

    x = torch.randn(1, 3, 512, 512)

    with torch.no_grad():
        out = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", out.shape)