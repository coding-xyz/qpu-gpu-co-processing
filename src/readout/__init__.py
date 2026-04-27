from ..common import pick_device, to_tensor, make_loader
from ..common import save_json, maybe_write_json, ensure_dir

from .models import LDA, MatchedFilter, TinyCNN, AmortizedBayesNet, TinyTransformer
from .bayes_core import bayes_init
from .bayes_em import bayes_em_fit
from .hmm_gaussian import fit_hmm_templates_fixed, hmm_classify
from .path_signature_features import signature2_features, SignatureLogReg

from .train_readout import train_simple, eval_bayes, eval_nn, eval_signature
from .data_utils import make_loader, load_experiment_data, split_train_test
from .experiments import (
    compute_psd_rfft,
    peak_snr_and_width_from_psd,
    find_demod_freq,
    demod_iq,
    metrics_from_peaks_gennorm,
    scores_from_psd_using_median_floor,
    best_threshold_from_scores,
    predict_fidelities_from_spectrum,
    train_test_fidelity_from_scores,
)
