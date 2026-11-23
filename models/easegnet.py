import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class ConvBNReLU(nn.Module):
    """Apply convolution, batch normalization, and ReLU6 activation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding.
        dilation: Convolution dilation.

    Returns:
        A feature tensor with out_channels channels.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        """Transform the input feature tensor.

        Args:
            x: Input feature tensor.

        Returns:
            Convolutionally transformed feature tensor.
        """
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    """Apply lightweight depthwise and pointwise convolutions.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride used by the depthwise convolution.

    Returns:
        A feature tensor with out_channels channels.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        """Apply depthwise-separable feature extraction.

        Args:
            x: Input feature tensor.

        Returns:
            Output feature tensor.
        """
        return self.block(x)


class MobileNetV2Encoder(nn.Module):
    """Extract multi-resolution features using pretrained MobileNetV2.

    Args:
        pretrained: Whether to load ImageNet-pretrained weights.

    Returns:
        Five feature maps at progressively lower spatial resolutions.
    """

    def __init__(self, pretrained=True):
        super().__init__()

        weights = (
            MobileNet_V2_Weights.DEFAULT
            if pretrained
            else None
        )

        backbone = mobilenet_v2(weights=weights)
        self.features = backbone.features

    def forward(self, x):
        """Extract intermediate MobileNetV2 feature maps.

        Args:
            x: Input RGB image tensor.

        Returns:
            Tuple containing feature maps at strides 2, 4, 8, 16, and 32.
        """
        features = []

        for index, layer in enumerate(self.features):
            x = layer(x)

            if index in (1, 3, 6, 13, 17):
                features.append(x)

        return tuple(features)


class AtrousContext(nn.Module):
    """Combine feature information using multiple atrous dilation rates.

    Args:
        in_channels: Number of input feature channels.
        out_channels: Number of output feature channels.

    Returns:
        A fused multi-scale feature tensor.
    """

    def __init__(self, in_channels=320, out_channels=96):
        super().__init__()

        branch_channels = 32

        self.branch1 = DepthwiseSeparableConv(
            in_channels,
            branch_channels
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=in_channels,
                bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(
                in_channels,
                branch_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU6(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=4,
                dilation=4,
                groups=in_channels,
                bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(
                in_channels,
                branch_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU6(inplace=True)
        )

        self.fusion = ConvBNReLU(
            branch_channels * 3,
            out_channels,
            kernel_size=1,
            padding=0
        )

    def forward(self, x):
        """Fuse features extracted at different dilation rates.

        Args:
            x: Deep MobileNetV2 feature tensor.

        Returns:
            Fused multi-scale context features.
        """
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)

        combined = torch.cat(
            (branch1, branch2, branch3),
            dim=1
        )

        return self.fusion(combined)


class AtrousDecoder(nn.Module):
    """Decode multi-resolution features into semantic segmentation logits.

    Args:
        num_classes: Number of segmentation classes.

    Returns:
        Full-resolution segmentation logits.
    """

    def __init__(self, num_classes=11):
        super().__init__()

        self.context = AtrousContext(320, 96)

        self.decode4 = DepthwiseSeparableConv(96 + 96, 64)
        self.decode3 = DepthwiseSeparableConv(64 + 32, 48)
        self.decode2 = DepthwiseSeparableConv(48 + 24, 32)
        self.decode1 = DepthwiseSeparableConv(32 + 16, 24)

        self.classifier = nn.Conv2d(
            24,
            num_classes,
            kernel_size=1
        )

    def forward(self, features, output_size):
        """Fuse encoder features and produce segmentation logits.

        Args:
            features: Five feature maps from MobileNetV2.
            output_size: Required output height and width.

        Returns:
            Segmentation logits with num_classes channels.
        """
        x1, x2, x3, x4, x5 = features

        x = self.context(x5)

        x = F.interpolate(
            x,
            size=x4.shape[2:],
            mode="bilinear",
            align_corners=False
        )
        x = self.decode4(torch.cat((x, x4), dim=1))

        x = F.interpolate(
            x,
            size=x3.shape[2:],
            mode="bilinear",
            align_corners=False
        )
        x = self.decode3(torch.cat((x, x3), dim=1))

        x = F.interpolate(
            x,
            size=x2.shape[2:],
            mode="bilinear",
            align_corners=False
        )
        x = self.decode2(torch.cat((x, x2), dim=1))

        x = F.interpolate(
            x,
            size=x1.shape[2:],
            mode="bilinear",
            align_corners=False
        )
        x = self.decode1(torch.cat((x, x1), dim=1))

        x = self.classifier(x)

        return F.interpolate(
            x,
            size=output_size,
            mode="bilinear",
            align_corners=False
        )


class EASegNet(nn.Module):
    """Build Efficient Atrous Segmentation Network for CamVid.

    Args:
        num_classes: Number of target segmentation classes.
        pretrained: Whether to initialize MobileNetV2 with ImageNet weights.

    Returns:
        Full-resolution per-class segmentation logits.
    """

    def __init__(self, num_classes=11, pretrained=True):
        super().__init__()

        self.encoder = MobileNetV2Encoder(pretrained=pretrained)
        self.decoder = AtrousDecoder(num_classes=num_classes)

    def forward(self, x):
        """Generate semantic segmentation logits for input images.

        Args:
            x: Input image tensor of shape [B, 3, H, W].

        Returns:
            Logits tensor of shape [B, num_classes, H, W].
        """
        output_size = x.shape[2:]
        features = self.encoder(x)

        return self.decoder(features, output_size)


if __name__ == "__main__":
    model = EASegNet(num_classes=11, pretrained=True)

    dummy_input = torch.randn(2, 3, 360, 480)
    output = model(dummy_input)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    print("Total parameters:", f"{total_parameters:,}")
    print("Trainable parameters:", f"{trainable_parameters:,}")