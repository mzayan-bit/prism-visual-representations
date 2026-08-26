"""Experiment execution harness preparing reproducible execution environments."""

import platform
import sys
from pathlib import Path
from typing import Any

from prism.core.enums import RunStatus
from prism.core.errors import (
    LifecycleError,
    ValidationError,
)
from prism.core.identifiers import generate_run_id
from prism.core.metadata import CodeRevisionMetadata, EnvironmentMetadata
from prism.experiments.context import PreparedExecution
from prism.experiments.definitions import ExperimentDefinition
from prism.experiments.environment import capture_environment
from prism.experiments.hardware import probe_hardware
from prism.experiments.provenance import inspect_git_provenance
from prism.experiments.runs import ExperimentRun
from prism.experiments.seeding import initialize_seeds


class ExperimentExecutionHarness:
    """Harness that prepares and audits a deterministic experiment runtime context.

    Validates experiment definitions, probes the physical host and hardware backends,
    initializes multi-backend pseudo-random seeds, inspects git version control state,
    and binds the resulting snapshot to an ExperimentRun before workload execution.

    NOTE: The harness intentionally STOPS before workload execution (no training loops,
    dataloaders, or tensor allocations).
    """

    def prepare(
        self,
        experiment: ExperimentDefinition,
        run: ExperimentRun | None = None,
        strict_reproducibility: bool = False,
        git_cwd: Path | str | None = None,
        git_runner: Any | None = None,
        torch_module: Any | None = None,
        numpy_module: Any | None = None,
    ) -> tuple[ExperimentRun, PreparedExecution]:
        """Validate an experiment definition and prepare its execution environment.

        Args:
            experiment: Immutable ExperimentDefinition to be executed.
            run: Optional pre-created ExperimentRun in PLANNED state.
                 If None, a new planned run is generated automatically.
            strict_reproducibility: If True, raise ReproducibilityError if any
                                    requested deterministic mode cannot be applied.
            git_cwd: Optional repository working directory for git provenance.
            git_runner: Optional custom git runner for testing and mocking.
            torch_module: Optional torch module or mock for testing.
            numpy_module: Optional numpy module or mock for testing.

        Returns:
            Tuple of (prepared_run, prepared_execution_context).

        Raises:
            ValidationError: If experiment definition or run linkages are invalid.
            LifecycleError: If a provided run is not in PLANNED state.
            ReproducibilityError: If strict_reproducibility is True and a requested
                                  deterministic mode cannot be satisfied.
        """
        all_warnings: list[str] = []

        # 1. Compute deterministic configuration fingerprint
        fingerprint = experiment.compute_fingerprint()

        # 2. Initialize or validate ExperimentRun lifecycle state
        if run is None:
            active_run = ExperimentRun(
                run_id=generate_run_id(),
                experiment_id=experiment.experiment_id,
                status=RunStatus.PLANNED,
                configuration_fingerprint=fingerprint,
                reproducibility=experiment.reproducibility,
            )
        else:
            if run.experiment_id != experiment.experiment_id:
                raise ValidationError(
                    f"Run experiment_id '{run.experiment_id}' does not match "
                    f"definition experiment_id '{experiment.experiment_id}'"
                )
            if run.status != RunStatus.PLANNED:
                raise LifecycleError(
                    f"Cannot prepare run '{run.run_id}' with status "
                    f"'{run.status.value}'. Runs must be in PLANNED state."
                )
            active_run = run
            active_run.configuration_fingerprint = fingerprint
            active_run.reproducibility = experiment.reproducibility

        # 3. Inspect Git version control provenance
        if experiment.reproducibility.capture_code_revision:
            git_prov = inspect_git_provenance(
                cwd=git_cwd,
                runner=git_runner,
            )
            code_revision = git_prov.to_code_revision_metadata()
            all_warnings.extend(git_prov.warnings)
        else:
            code_revision = CodeRevisionMetadata()

        # 4. Probe Hardware and compute capabilities
        hardware = probe_hardware(torch_module=torch_module)

        # 5. Capture Environment snapshot
        if experiment.reproducibility.capture_environment:
            environment = capture_environment(hardware_override=hardware)
        else:
            environment = EnvironmentMetadata(
                python_version=sys.version.split()[0],
                python_implementation=platform.python_implementation(),
                os=f"{platform.system()} {platform.release()}".strip(),
                platform=platform.platform(),
                hardware="Unrecorded",
                hardware_info=hardware,
            )

        # 6. Initialize Multi-Backend Pseudo-Random Seeds
        seeding_result = initialize_seeds(
            seed=experiment.reproducibility.seed,
            deterministic=experiment.reproducibility.deterministic,
            strict=strict_reproducibility,
            torch_module=torch_module,
            numpy_module=numpy_module,
        )
        all_warnings.extend(seeding_result.warnings)

        # 7. Bind captured provenance and environment to the ExperimentRun
        active_run.code_revision = code_revision
        active_run.environment = environment

        # 8. Construct the immutable PreparedExecution runtime context
        prepared_execution = PreparedExecution(
            experiment_id=experiment.experiment_id,
            run_id=active_run.run_id,
            configuration_fingerprint=fingerprint,
            reproducibility=experiment.reproducibility,
            seeding_result=seeding_result,
            environment=environment,
            hardware=hardware,
            code_revision=code_revision,
            warnings=all_warnings,
        )

        return active_run, prepared_execution
