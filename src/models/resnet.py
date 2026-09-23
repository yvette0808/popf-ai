"""ResNet model construction for the modern POPF experiments."""

from __future__ import annotations

from typing import Any, Iterable

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


RESNET18_MAJOR_MODULES = (
    "conv1",
    "bn1",
    "layer1",
    "layer2",
    "layer3",
    "layer4",
    "fc",
)


def configure_resnet18_trainability(
    model: nn.Module,
    trainable_modules: Iterable[str],
) -> dict[str, Any]:
    """Freeze all parameters, then enable the named top-level modules."""

    requested = tuple(dict.fromkeys(trainable_modules))
    unknown = sorted(set(requested).difference(RESNET18_MAJOR_MODULES))
    if unknown:
        raise ValueError(f"Unknown ResNet-18 trainable modules: {unknown}")
    if "fc" not in requested:
        raise ValueError("The replacement classification head `fc` must be trainable")

    for parameter in model.parameters():
        parameter.requires_grad = False
    for module_name in requested:
        module = getattr(model, module_name)
        for parameter in module.parameters():
            parameter.requires_grad = True

    model._popf_trainable_module_names = requested  # type: ignore[attr-defined]
    model._popf_frozen_module_names = tuple(  # type: ignore[attr-defined]
        name for name in RESNET18_MAJOR_MODULES if name not in requested
    )
    return resnet18_trainability_audit(model)


def resnet18_trainability_audit(model: nn.Module) -> dict[str, Any]:
    """Return auditable parameter counts for the major ResNet-18 modules."""

    module_audit: dict[str, dict[str, Any]] = {}
    for module_name in RESNET18_MAJOR_MODULES:
        module = getattr(model, module_name)
        parameters = list(module.parameters())
        module_audit[module_name] = {
            "trainable": bool(parameters) and all(
                parameter.requires_grad for parameter in parameters
            ),
            "parameter_count": sum(parameter.numel() for parameter in parameters),
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in parameters
                if parameter.requires_grad
            ),
        }
    total = total_parameter_count(model)
    trainable = trainable_parameter_count(model)
    return {
        "modules": module_audit,
        "trainable_modules": list(
            getattr(model, "_popf_trainable_module_names", ())
        ),
        "frozen_modules": list(getattr(model, "_popf_frozen_module_names", ())),
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
    }


def build_resnet18(
    num_classes: int,
    *,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    trainable_modules: Iterable[str] | None = None,
) -> nn.Module:
    """Build a ResNet-18 with a replacement classification head.

    This is a 2026 Reconstruction model. The pretrained weights are the
    torchvision ImageNet weights and are not part of the 2019 Original.
    """

    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, num_classes)

    if trainable_modules is not None:
        configure_resnet18_trainability(model, trainable_modules)
    elif freeze_backbone:
        configure_resnet18_trainability(model, ("fc",))
    else:
        for parameter in model.parameters():
            parameter.requires_grad = True
        model._popf_trainable_module_names = RESNET18_MAJOR_MODULES  # type: ignore[attr-defined]
        model._popf_frozen_module_names = ()  # type: ignore[attr-defined]
    return model


def trainable_parameter_count(model: nn.Module) -> int:
    """Return the number of parameters that will be updated by the optimizer."""

    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def total_parameter_count(model: nn.Module) -> int:
    """Return the total number of model parameters."""

    return sum(parameter.numel() for parameter in model.parameters())
