"""ResNet18 U-Net with explicit raw-channel initialization and weight provenance."""
from pathlib import Path

from . import ROOT
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18

from filament.common import read_json, sha256
from .runtime import atomic_json


def prepare_weights():
    directory = ROOT / "checkpoints/pretrained"
    directory.mkdir(parents=True, exist_ok=True)
    weights = ResNet18_Weights.IMAGENET1K_V1
    path = directory / Path(weights.url).name
    if not path.exists():
        torch.hub.download_url_to_file(weights.url, str(path), hash_prefix="f37072fd", progress=False)
    digest = sha256(path)
    if not digest.startswith("f37072fd"):
        raise ValueError("Official weight hash prefix mismatch")
    provenance = {"architecture": "resnet18", "enum": "ResNet18_Weights.IMAGENET1K_V1", "source_url": weights.url,
                  "sha256": digest, "bytes": path.stat().st_size, "relative_path": str(path.relative_to(ROOT)),
                  "pretraining": "ImageNet-1K classification; no competition masks or external filament labels",
                  "documentation": "https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.resnet18.html",
                  "implementation_license": "BSD-3-Clause: https://github.com/pytorch/vision/blob/v0.23.0/LICENSE",
                  "weight_terms_note": "torchvision distinguishes implementation license from training-data/weight usage terms; retain upstream documentation.",
                  "input_adaptation": "sum pretrained RGB conv1 kernels into raw channel; zero extra channels; fixed (x-0.5)/0.25 normalization"}
    metadata = directory / "resnet18-provenance.json"
    if metadata.exists() and read_json(metadata) != provenance:
        raise ValueError("Weight provenance changed")
    atomic_json(metadata, provenance)
    return path, provenance


class ConvBlock(nn.Sequential):
    def __init__(self, incoming, outgoing):
        super().__init__(nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False), nn.GroupNorm(8, outgoing), nn.ReLU(),
                         nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False), nn.GroupNorm(8, outgoing), nn.ReLU())


class ResNetUNet(nn.Module):
    def __init__(self, channels=1, weights_path=None):
        super().__init__()
        if channels < 1:
            raise ValueError("At least one raw channel is required")
        self.encoder = resnet18(weights=None)
        if weights_path is not None:
            self.encoder.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        rgb_kernel = self.encoder.conv1.weight.detach().clone()
        # Construct the complete single-channel model first, preserving E001 RNG order.
        self.encoder.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.encoder.conv1.weight.zero_()
            self.encoder.conv1.weight[:, :1].copy_(rgb_kernel.sum(dim=1, keepdim=True))
        self.encoder.fc = nn.Identity()
        self.dec4 = ConvBlock(512 + 256, 256)
        self.dec3 = ConvBlock(256 + 128, 128)
        self.dec2 = ConvBlock(128 + 64, 64)
        self.dec1 = ConvBlock(64 + 64, 32)
        self.dec0 = ConvBlock(33, 16)
        self.head = nn.Conv2d(16, 1, 1)
        if channels > 1:
            encoder_input = nn.Conv2d(channels, 64, 7, stride=2, padding=3, bias=False)
            decoder_input = nn.Conv2d(32 + channels, 16, 3, padding=1, bias=False)
            with torch.no_grad():
                encoder_input.weight.zero_()
                encoder_input.weight[:, :1].copy_(self.encoder.conv1.weight)
                decoder_input.weight.zero_()
                decoder_input.weight[:, :33].copy_(self.dec0[0].weight)
            self.encoder.conv1, self.dec0[0] = encoder_input, decoder_input

    def forward(self, image):
        raw = (image - 0.5) / 0.25
        e = self.encoder
        one = e.relu(e.bn1(e.conv1(raw)))
        two = e.layer1(e.maxpool(one))
        three = e.layer2(two)
        four = e.layer3(three)
        five = e.layer4(four)
        value = five
        for block, skip in ((self.dec4, four), (self.dec3, three), (self.dec2, two), (self.dec1, one), (self.dec0, raw)):
            value = block(torch.cat([F.interpolate(value, size=skip.shape[-2:], mode="nearest"), skip], dim=1))
        return self.head(value)


def segmentation_loss(logits, target):
    logits, target = logits.float(), target.float()
    probability = logits.sigmoid()
    intersection = (probability * target).sum(dim=(1, 2, 3))
    dice = (2 * intersection + 1) / (probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return bce + (1 - dice).mean()


def binary_dice(logits, target):
    predicted = (logits.float().sigmoid() >= 0.5).float()
    intersection = (predicted * target).sum(dim=(1, 2, 3))
    return (2 * intersection + 1) / (predicted.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1)
