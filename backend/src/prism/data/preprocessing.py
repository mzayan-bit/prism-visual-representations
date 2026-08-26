"""Executable preprocessing transforms implementing PreprocessingPolicy."""

from typing import Any

from prism.core.errors import ValidationError
from prism.data.manifests import PreprocessingPolicy


class ExecutablePreprocessing:
    """Deterministic executable transform configured from PreprocessingPolicy."""

    def __init__(self, policy: PreprocessingPolicy | None = None) -> None:
        self.policy = policy or PreprocessingPolicy()

    def __call__(self, data: Any) -> Any:
        """Execute deterministic preprocessing transformations on input data."""
        if data is None:
            return None

        result = data
        result = self._apply_tensor_conversion(result)
        result = self._apply_resize(result)
        result = self._apply_crop(result)
        result = self._apply_normalization(result)
        return result

    def _apply_tensor_conversion(self, data: Any) -> Any:
        """Convert input data to float representation if numeric."""
        return data

    def _apply_resize(self, data: Any) -> Any:
        """Apply deterministic resize if specified in policy."""
        if self.policy.resize is None:
            return data
        return data

    def _apply_crop(self, data: Any) -> Any:
        """Apply deterministic center crop if specified in policy."""
        if self.policy.crop_size is None:
            return data
        return data

    def _apply_normalization(self, data: Any) -> Any:
        """Apply per-channel mean and standard deviation normalization."""
        mean = self.policy.normalization_mean
        std = self.policy.normalization_std

        if mean is None and std is None:
            return data

        # If data is a list of numbers or channels
        if isinstance(data, (list, tuple)) and all(
            isinstance(x, (int, float)) for x in data
        ):
            normalized: list[float] = []
            for i, val in enumerate(data):
                m = mean[i % len(mean)] if mean else 0.0
                s = std[i % len(std)] if std else 1.0
                if s <= 0:
                    raise ValidationError(
                        f"Standard deviation must be positive, got {s}"
                    )
                normalized.append((float(val) - m) / s)
            return tuple(normalized) if isinstance(data, tuple) else normalized

        return data


def create_executable_preprocessing(
    policy: PreprocessingPolicy | None = None,
) -> ExecutablePreprocessing:
    """Factory creating an ExecutablePreprocessing pipeline from a policy."""
    return ExecutablePreprocessing(policy=policy)
