"""eeg_preproc — a parameterised MNE EEG preprocessing pipeline (BCI IV 2a demo).

Public API:
    preprocess  — run the full pipeline (re-reference / notch / bandpass / ICA)
    load_raw    — load a BCI IV 2a GDF and canonicalise channels
    classify_accuracy — 4-class CSP + LDA accuracy via stratified 5-fold CV
    csp_snr     — CSP class-discriminative SNR proxy (dB)
"""

from .preprocess import (load_raw, get_cue_events, process, apply_reference,
                         apply_notch, apply_bandpass, apply_ica,
                         build_epochs, _drop_expert_rejected,
                         classify_accuracy, csp_snr, MultiCSP, CANON, EOG_CH)

__all__ = [
    'load_raw', 'get_cue_events', 'process', 'apply_reference', 'apply_notch',
    'apply_bandpass', 'apply_ica', 'build_epochs', '_drop_expert_rejected',
    'classify_accuracy', 'csp_snr', 'MultiCSP', 'CANON', 'EOG_CH',
]
