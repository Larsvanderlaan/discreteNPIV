from ._version import __version__
from .api import estimate_average_functional, fit_dual_nuisance, fit_structural_nuisance
from .grouping import group_means, leave_one_out_group_means, make_stratified_folds
from .reproduction import PaperExperimentSummary, run_small_paper_experiment, summarize_legacy_archive
from .results import (
    ExperimentEncodingResult,
    DualFitResult,
    FunctionalEstimate,
    LongTermEffectResult,
    LongTermMeanResult,
    NPIVInferenceResult,
    RegularizationChoice,
    StructuralFitResult,
)
from .simulation import generate_synthetic_data
from .surrogates import (
    encode_experiment_arms,
    estimate_long_term_effect_from_surrogates,
    estimate_long_term_mean_from_surrogates,
)

__all__ = [
    "__version__",
    "ExperimentEncodingResult",
    "DualFitResult",
    "FunctionalEstimate",
    "LongTermEffectResult",
    "LongTermMeanResult",
    "NPIVInferenceResult",
    "PaperExperimentSummary",
    "RegularizationChoice",
    "StructuralFitResult",
    "encode_experiment_arms",
    "estimate_average_functional",
    "estimate_long_term_effect_from_surrogates",
    "estimate_long_term_mean_from_surrogates",
    "fit_dual_nuisance",
    "fit_structural_nuisance",
    "generate_synthetic_data",
    "group_means",
    "leave_one_out_group_means",
    "make_stratified_folds",
    "run_small_paper_experiment",
    "summarize_legacy_archive",
]
