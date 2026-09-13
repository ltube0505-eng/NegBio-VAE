"""Hierarchical negative-binomial VAE.

This module implements the multi-scale bottom-up/top-down skeleton described
in ``HNB-VAE.md``.  It intentionally lives next to, rather than inside,
``GenericVAE`` so existing one-layer checkpoints and training commands keep
their original parameter names and forward signature.
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from distribution import NegBinomial


_LOG2 = math.log(2.0)


def _positive(raw: torch.Tensor, minimum: float = 1e-3,
              maximum: float = 100.0) -> torch.Tensor:
    """Positive transform whose neutral raw value (zero) maps to one."""
    return (F.softplus(raw) / _LOG2).clamp(minimum, maximum)


def _zero_init(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class BottomUpResidualBlock(nn.Module):
    """Residual convolutional block used by the bottom-up encoder."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            SqueezeExcite(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TopDownLatentBlock(nn.Module):
    """Depthwise-separable block that injects a latent into TD features."""

    def __init__(self, latent_channels: int, td_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_channels, td_channels, 1),
            nn.BatchNorm2d(td_channels),
            nn.SiLU(),
            nn.Conv2d(
                td_channels, td_channels, 5, padding=2, groups=td_channels
            ),
            nn.BatchNorm2d(td_channels),
            nn.SiLU(),
            nn.Conv2d(td_channels, td_channels, 1),
            nn.BatchNorm2d(td_channels),
            SqueezeExcite(td_channels),
        )

    def forward(self, zeta: torch.Tensor) -> torch.Tensor:
        return self.net(zeta)


class PriorNet(nn.Module):
    def __init__(self, td_channels: int, latent_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(td_channels),
            nn.SiLU(),
            nn.Conv2d(td_channels, 2 * latent_channels, 1),
        )

    def forward(self, d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_alpha, raw_beta = self.net(d).chunk(2, dim=1)
        return _positive(raw_alpha), _positive(raw_beta)


class PosteriorNet(nn.Module):
    def __init__(self, td_channels: int, latent_channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.BatchNorm2d(2 * td_channels),
            nn.SiLU(),
            nn.Conv2d(2 * td_channels, td_channels, 3, padding=1),
            nn.BatchNorm2d(td_channels),
            nn.SiLU(),
        )
        self.output = nn.Conv2d(td_channels, 2 * latent_channels, 1)
        # q=p at initialization: both normalized-softplus multipliers are 1.
        _zero_init(self.output)

    def forward(self, e: torch.Tensor,
                d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_delta_alpha, raw_delta_beta = self.output(
            self.body(torch.cat([e, d], dim=1))
        ).chunk(2, dim=1)
        return _positive(raw_delta_alpha), _positive(raw_delta_beta)


class MonteCarloNBRelaxation(nn.Module):
    """Original relaxed-NB sampler with a Monte Carlo discrete-space KL."""

    def __init__(self, reparam_type: str, tau: float, max_count: int,
                 num_samples: int):
        super().__init__()
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        self.tau = float(tau)
        self.num_samples = int(num_samples)
        self.dist = NegBinomial(
            reparam_type=reparam_type,
            tau=tau,
            max_count=max_count,
            num_samples=num_samples,
        )

    def set_temperature(self, tau: float) -> None:
        if tau <= 0:
            raise ValueError("temperature must be positive")
        self.tau = float(tau)
        if self.dist.reparam_type == "gumbel":
            self.dist.strategy.tau = self.tau

    def _sample(self, alpha, beta, hard=False):
        return self.dist.rsample(alpha.log(), beta.log(), self.tau, hard=hard)

    def sample_and_kl(self, alpha_p, beta_p, alpha_q, beta_q):
        hard = not torch.is_grad_enabled()
        zeta = self._sample(alpha_q, beta_q, hard=hard)
        samples = [zeta]
        samples.extend(
            self._sample(alpha_q, beta_q, hard=hard)
            for _ in range(self.num_samples - 1)
        )
        samples = torch.stack(samples, dim=0)
        log_q = NegBinomial._log_prob(
            samples, alpha_q.log(), beta_q.log()
        )
        log_p = NegBinomial._log_prob(
            samples, alpha_p.log(), beta_p.log()
        )
        return zeta, (log_q - log_p).mean(dim=0)

    def sample_prior(self, alpha_p, beta_p):
        hard = not torch.is_grad_enabled()
        return self._sample(alpha_p, beta_p, hard=hard)

    @staticmethod
    def posterior_mean(alpha_q, beta_q):
        return alpha_q / beta_q


def build_relaxation(kl_type: str, reparam_type: str, tau: float,
                     max_count: int, num_samples: int) -> nn.Module:
    if kl_type != "mc":
        raise ValueError(
            "HNB changes both negative-binomial parameters at every group, "
            "so it requires kl='mc'. The one-layer analytical objective is "
            "only valid when dispersion is shared."
        )
    return MonteCarloNBRelaxation(
        reparam_type, tau, max_count, num_samples
    )


class HNBLayer(nn.Module):
    """One spatial latent group in the hierarchical model."""

    def __init__(self, td_channels: int, latent_channels: int,
                 relaxation: nn.Module, unconditional_prior: bool = False):
        super().__init__()
        self.latent_channels = latent_channels
        self.unconditional_prior = unconditional_prior
        self.relaxation = relaxation
        if unconditional_prior:
            self.prior_raw = nn.Parameter(
                torch.zeros(1, 2 * latent_channels, 1, 1)
            )
            self.prior_net = None
        else:
            self.prior_raw = None
            self.prior_net = PriorNet(td_channels, latent_channels)
        self.posterior_net = PosteriorNet(td_channels, latent_channels)
        self.td_block = TopDownLatentBlock(latent_channels, td_channels)

    def prior_parameters(self, d):
        if self.prior_raw is None:
            return self.prior_net(d)
        raw_alpha, raw_beta = self.prior_raw.chunk(2, dim=1)
        shape = (d.size(0), -1, d.size(2), d.size(3))
        return (
            _positive(raw_alpha).expand(shape),
            _positive(raw_beta).expand(shape),
        )

    def forward(self, d: torch.Tensor, e: Optional[torch.Tensor] = None):
        alpha_p, beta_p = self.prior_parameters(d)
        if e is None:
            zeta = self.relaxation.sample_prior(alpha_p, beta_p)
            return d + self.td_block(zeta), zeta

        delta_alpha, delta_beta = self.posterior_net(e, d)
        alpha_q = (alpha_p * delta_alpha).clamp(1e-3, 100.0)
        beta_q = (beta_p / delta_beta).clamp(1e-3, 100.0)
        zeta, kl = self.relaxation.sample_and_kl(
            alpha_p, beta_p, alpha_q, beta_q
        )
        d = d + self.td_block(zeta)
        params = {
            "alpha_p": alpha_p,
            "beta_p": beta_p,
            "alpha_q": alpha_q,
            "beta_q": beta_q,
        }
        return d, zeta, kl, params


@dataclass
class HNBOutput:
    reconstruction: torch.Tensor
    latent: torch.Tensor
    representation: torch.Tensor
    kl_diag: torch.Tensor
    kl_per_group: torch.Tensor
    total_kl: torch.Tensor
    latents: List[torch.Tensor]
    parameters: List[Dict[str, torch.Tensor]]


class HNBVAE(nn.Module):
    """Multi-scale hierarchical negative-binomial VAE."""

    DATASET_SHAPES = {
        "MNIST": (1, 28),
        "fmnist": (1, 28),
        "Omniglot": (1, 28),
        "CIFAR16": (3, 16),
        "CIFAR10": (3, 32),
        "SVHN": (3, 32),
        "CelebA": (3, 64),
        "CelebA64": (3, 64),
        "CelebAHQ": (3, 128),
        "FFHQ": (3, 256),
    }

    def __init__(
        self,
        input_channels: int,
        image_size: int,
        groups_per_scale: Sequence[int],
        channels_per_scale: Sequence[int],
        latent_channels_per_scale: Sequence[int],
        kl_type: str = "mc",
        reparam_type: str = "gamma",
        tau: float = 0.1,
        max_count: int = 15,
        num_samples: int = 5,
        spatial_sizes: Optional[Sequence[int]] = None,
        free_bits: float = 0.0,
        kl_balance: str = "uniform",
        output_activation: str = "sigmoid",
    ):
        super().__init__()
        self.input_channels = int(input_channels)
        self.image_size = int(image_size)
        self.groups_per_scale = [int(v) for v in groups_per_scale]
        self.channels_per_scale = [int(v) for v in channels_per_scale]
        self.latent_channels_per_scale = [
            int(v) for v in latent_channels_per_scale
        ]
        self.num_scales = len(self.groups_per_scale)
        if self.num_scales < 1:
            raise ValueError("HNB requires at least one scale")
        if not (
            len(self.channels_per_scale)
            == len(self.latent_channels_per_scale)
            == self.num_scales
        ):
            raise ValueError(
                "groups, TD channels, and latent channels need one value "
                "per scale"
            )
        if any(v < 1 for v in self.groups_per_scale):
            raise ValueError("Every HNB scale must contain at least one group")
        if any(v < 1 for v in self.channels_per_scale
               + self.latent_channels_per_scale):
            raise ValueError("HNB channel counts must be positive")

        if spatial_sizes is None:
            finest = max(2, self.image_size // 2)
            spatial_sizes = [
                max(1, finest // (2 ** (self.num_scales - 1 - s)))
                for s in range(self.num_scales)
            ]
        self.spatial_sizes = [int(v) for v in spatial_sizes]
        if len(self.spatial_sizes) != self.num_scales:
            raise ValueError("spatial_sizes needs one value per scale")
        if any(v < 1 for v in self.spatial_sizes):
            raise ValueError("spatial_sizes must be positive")
        if any(a >= b for a, b in zip(
                self.spatial_sizes, self.spatial_sizes[1:])):
            raise ValueError("spatial_sizes must increase from top to bottom")

        self.kl_type = kl_type
        self.reparam_type = reparam_type
        self.tau = float(tau)
        self.free_bits = float(free_bits)
        if self.free_bits < 0:
            raise ValueError("free_bits cannot be negative")
        if kl_balance not in {"uniform", "scale"}:
            raise ValueError("kl_balance must be 'uniform' or 'scale'")
        self.kl_balance = kl_balance

        fine_channels = self.channels_per_scale[-1]
        self.input_stem = nn.Sequential(
            nn.Conv2d(input_channels, fine_channels, 3, padding=1),
            nn.SiLU(),
            BottomUpResidualBlock(fine_channels),
        )

        self.bu_transitions = nn.ModuleList()
        for s in range(self.num_scales - 1):
            self.bu_transitions.append(nn.Sequential(
                nn.Conv2d(
                    self.channels_per_scale[s + 1],
                    self.channels_per_scale[s],
                    3,
                    padding=1,
                ),
                nn.SiLU(),
            ))

        self.bu_group_blocks = nn.ModuleList()
        self.td_layers = nn.ModuleList()
        group_scales = []
        group_index = 0
        for s, n_groups in enumerate(self.groups_per_scale):
            bu_at_scale = nn.ModuleList(
                BottomUpResidualBlock(self.channels_per_scale[s])
                for _ in range(n_groups)
            )
            self.bu_group_blocks.append(bu_at_scale)
            for _ in range(n_groups):
                relaxation = build_relaxation(
                    kl_type=kl_type,
                    reparam_type=reparam_type,
                    tau=tau,
                    max_count=max_count,
                    num_samples=num_samples,
                )
                self.td_layers.append(HNBLayer(
                    self.channels_per_scale[s],
                    self.latent_channels_per_scale[s],
                    relaxation,
                    unconditional_prior=(group_index == 0),
                ))
                group_scales.append(s)
                group_index += 1
        self.group_scales = group_scales
        self.num_groups = len(group_scales)

        self.td_transitions = nn.ModuleList()
        for s in range(self.num_scales - 1):
            self.td_transitions.append(nn.Sequential(
                nn.Conv2d(
                    self.channels_per_scale[s],
                    self.channels_per_scale[s + 1],
                    1,
                ),
                nn.SiLU(),
            ))

        top_ch = self.channels_per_scale[0]
        top_size = self.spatial_sizes[0]
        self.h_init = nn.Parameter(torch.zeros(1, top_ch, top_size, top_size))
        final_ch = self.channels_per_scale[-1]
        final_activation = (
            nn.Tanh() if output_activation == "tanh" else nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            BottomUpResidualBlock(final_ch),
            nn.BatchNorm2d(final_ch),
            nn.SiLU(),
            nn.Conv2d(final_ch, input_channels, 3, padding=1),
            final_activation,
        )

        base_weights = []
        for s, n_groups in enumerate(self.groups_per_scale):
            value = 1.0 if kl_balance == "uniform" else 1.0 / n_groups
            base_weights.extend([value] * n_groups)
        self.register_buffer(
            "base_group_weights", torch.tensor(base_weights), persistent=False
        )

    @classmethod
    def from_config(cls, cfg: dict) -> "HNBVAE":
        model_cfg = cfg["model"]
        dataset = cfg["dataset"]["name"]
        if dataset not in cls.DATASET_SHAPES:
            raise ValueError(f"HNB has no image shape registered for {dataset}")
        input_channels, image_size = cls.DATASET_SHAPES[dataset]
        groups = model_cfg.get("groups_per_scale", [4, 4])
        channels = model_cfg.get("channels_per_scale", [64, 32])
        latent_channels = model_cfg.get("latent_channels_per_scale", [16, 16])
        if isinstance(latent_channels, int):
            latent_channels = [latent_channels] * len(groups)
        output_activation = "tanh" if dataset in {
            "SVHN", "CIFAR10", "CIFAR16", "CelebA", "CelebA64",
            "CelebAHQ", "FFHQ"
        } else "sigmoid"
        return cls(
            input_channels=input_channels,
            image_size=image_size,
            groups_per_scale=groups,
            channels_per_scale=channels,
            latent_channels_per_scale=latent_channels,
            kl_type=model_cfg.get("kl", "mc"),
            reparam_type=model_cfg.get("reparam_type", "gamma"),
            tau=model_cfg.get("tau", 0.1),
            max_count=model_cfg.get("max_count", 15),
            num_samples=model_cfg.get("num_samples", 5),
            spatial_sizes=model_cfg.get("spatial_sizes"),
            free_bits=model_cfg.get("free_bits", 0.0),
            kl_balance=model_cfg.get("kl_balance", "uniform"),
            output_activation=output_activation,
        )

    def _bottom_up(self, x: torch.Tensor) -> List[torch.Tensor]:
        e = self.input_stem(x)
        features: List[Optional[torch.Tensor]] = [None] * self.num_groups
        scale_offsets = []
        offset = 0
        for n_groups in self.groups_per_scale:
            scale_offsets.append(offset)
            offset += n_groups

        for s in range(self.num_scales - 1, -1, -1):
            target = self.spatial_sizes[s]
            if e.shape[-2:] != (target, target):
                e = F.adaptive_avg_pool2d(e, (target, target))
            if s < self.num_scales - 1:
                e = self.bu_transitions[s](e)
            for g in range(self.groups_per_scale[s] - 1, -1, -1):
                e = self.bu_group_blocks[s][g](e)
                features[scale_offsets[s] + g] = e
        return features  # type: ignore[return-value]

    def _decode(self, d: torch.Tensor) -> torch.Tensor:
        if d.shape[-2:] != (self.image_size, self.image_size):
            d = F.interpolate(
                d, size=(self.image_size, self.image_size), mode="nearest"
            )
        return self.decoder(d)

    @staticmethod
    def mse_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (y - x).square().flatten(1).sum(dim=1).mean()

    def set_temperature(self, tau: float) -> None:
        self.tau = float(tau)
        for layer in self.td_layers:
            setter = getattr(layer.relaxation, "set_temperature", None)
            if setter is not None:
                setter(tau)

    def training_kl(self, kl_per_group: torch.Tensor,
                    warmup_progress: Optional[float] = None) -> torch.Tensor:
        """Return per-sample balanced/free-bits/ladder-warmed KL."""
        effective = kl_per_group
        if self.free_bits > 0:
            effective = effective.clamp_min(self.free_bits)
        weights = self.base_group_weights.to(effective)
        if warmup_progress is not None:
            progress = max(0.0, min(1.0, float(warmup_progress)))
            order = torch.arange(
                self.num_groups, device=effective.device, dtype=effective.dtype
            )
            ladder = (progress * self.num_groups - order).clamp(0.0, 1.0)
            weights = weights * ladder
        return (effective * weights.unsqueeze(0)).sum(dim=1)

    def forward(self, x: torch.Tensor) -> HNBOutput:
        if x.ndim != 4:
            raise ValueError("HNB expects image tensors shaped [B,C,H,W]")
        if x.size(1) != self.input_channels:
            raise ValueError(
                f"HNB expected {self.input_channels} input channels, got {x.size(1)}"
            )
        e_list = self._bottom_up(x)
        d = self.h_init.expand(x.size(0), -1, -1, -1)

        latents = []
        parameters = []
        kl_channels = []
        kl_groups = []
        representations = []
        layer_idx = 0
        for s, n_groups in enumerate(self.groups_per_scale):
            if s > 0:
                d = F.interpolate(
                    d,
                    size=(self.spatial_sizes[s], self.spatial_sizes[s]),
                    mode="nearest",
                )
                d = self.td_transitions[s - 1](d)
            for _ in range(n_groups):
                d, zeta, kl, params = self.td_layers[layer_idx](
                    d, e_list[layer_idx]
                )
                latents.append(zeta)
                parameters.append(params)
                kl_channels.append(kl.sum(dim=(2, 3)))
                kl_groups.append(kl.flatten(1).sum(dim=1))
                mean_q = self.td_layers[layer_idx].relaxation.posterior_mean(
                    params["alpha_q"], params["beta_q"]
                )
                representations.append(mean_q.mean(dim=(2, 3)))
                layer_idx += 1

        kl_diag = torch.cat(kl_channels, dim=1)
        kl_per_group = torch.stack(kl_groups, dim=1)
        total_kl = kl_per_group.sum(dim=1)
        latent = torch.cat(
            [z.mean(dim=(2, 3)) for z in latents], dim=1
        )
        representation = torch.cat(representations, dim=1)
        return HNBOutput(
            reconstruction=self._decode(d),
            latent=latent,
            representation=representation,
            kl_diag=kl_diag,
            kl_per_group=kl_per_group,
            total_kl=total_kl,
            latents=latents,
            parameters=parameters,
        )

    def sample_prior(self, num_samples: int):
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        d = self.h_init.expand(num_samples, -1, -1, -1)
        latents = []
        layer_idx = 0
        for s, n_groups in enumerate(self.groups_per_scale):
            if s > 0:
                d = F.interpolate(
                    d,
                    size=(self.spatial_sizes[s], self.spatial_sizes[s]),
                    mode="nearest",
                )
                d = self.td_transitions[s - 1](d)
            for _ in range(n_groups):
                d, zeta = self.td_layers[layer_idx](d, e=None)
                latents.append(zeta)
                layer_idx += 1
        pooled = torch.cat(
            [z.mean(dim=(2, 3)) for z in latents], dim=1
        )
        return pooled, self._decode(d)
