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


class FCNHead(nn.Module):
    """Auxiliary FCN head for deep supervision."""

    def __init__(self, in_channels, channels, n_classes=19):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(channels, n_classes, kernel_size=1),
        )

    def forward(self, x):
        return self.head(x)


class UPerNetDecoder(nn.Module):
    """UPerNet decoder with PPM, FPN fusion, and auxiliary head."""

    def __init__(
        self,
        encoder_channels=(64, 128, 320, 512),
        fpn_dim=512,
        n_classes=19,
        aux_channels=256,
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

        self.aux_head = FCNHead(
            in_channels=encoder_channels[2],
            channels=aux_channels,
            n_classes=n_classes,
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
        main_logits = self.classifier(x)

        if self.training:
            aux_logits = self.aux_head(c3)
            return main_logits, aux_logits

        return main_logits


class AuxLovaszUperFormer(nn.Module):
    """
    Auxiliary UPerFormer model.

    Architecture:
    - SegFormer-B5 encoder
    - UPerNet decoder with PPM and FPN fusion
    - Auxiliary FCN head on the third encoder stage during training

    During training:
        returns main_logits, aux_logits

    During evaluation:
        returns main_logits only
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
            aux_channels=256,
        )

    def forward(self, x):
        input_size = x.shape[-2:]

        outputs = self.encoder(
            pixel_values=x,
            output_hidden_states=True,
        )
        features = outputs.hidden_states

        if self.training:
            main_logits, aux_logits = self.decoder(features)

            main_logits = F.interpolate(
                main_logits,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )

            aux_logits = F.interpolate(
                aux_logits,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )

            return main_logits, aux_logits

        main_logits = self.decoder(features)

        if main_logits.shape[-2:] != input_size:
            main_logits = F.interpolate(
                main_logits,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )

        return main_logits


if __name__ == "__main__":
    model = AuxLovaszUperFormer(
        in_channels=3,
        n_classes=19,
        use_pretrained_weights=False,
    )

    model.train()

    x = torch.randn(2, 3, 512, 512)
    main_out, aux_out = model(x)

    print("Training input shape:", x.shape)
    print("Training main output shape:", main_out.shape)
    print("Training auxiliary output shape:", aux_out.shape)

    model.eval()

    with torch.no_grad():
        x = torch.randn(1, 3, 512, 512)
        out = model(x)

    print("Evaluation input shape:", x.shape)
    print("Evaluation output shape:", out.shape)