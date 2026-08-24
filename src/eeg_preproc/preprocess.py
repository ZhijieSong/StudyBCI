"""BCI Competition IV Dataset 2a (subject A01) - MNE preprocessing pipeline.

Provides a parameterised pipeline (re-reference / notch / bandpass / ICA) that
turns a raw GDF file into clean epochs, plus a CSP+LDA classifier that reports
classification accuracy, plus a CSP-based discriminative SNR proxy.

Author: 实验员 (MNE preprocessing do-it-right demo)
"""
import numpy as np
from collections import Counter

import mne
from mne.preprocessing import ICA
from mne.preprocessing import compute_current_source_density

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from mne.decoding import CSP

import scipy.linalg

# Canonical BCI IV 2a montage order (22 EEG channels). Verified against the
# labelled channels stored in the GDF: Fz@0, C3@7, Cz@9, C4@11, Pz@19.
CANON = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
         'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz',
         'P2', 'POz']
EOG_CH = ['EOG-left', 'EOG-central', 'EOG-right']

# Event codes (decimal) from desc_2a.pdf, Table 2.
CUE_EVENTS = {1: 769, 2: 770, 3: 771, 4: 772}   # class -> raw GDF event code
CUE_EVENT_IDS = {str(v): k for k, v in CUE_EVENTS.items()}


def load_raw(gdf_path):
    """Load a BCI IV 2a GDF file, rename the 22 EEG channels to the canonical
    montage and attach the standard 10-20 positions (needed for Laplacian /
    bipolar referencing and for the montage figure)."""
    raw = mne.io.read_raw_gdf(gdf_path, preload=True)
    raw.rename_channels({raw.ch_names[i]: CANON[i] for i in range(22)})
    # mark the 3 monopolar EOG channels so that picks='eeg' selects only EEG
    raw.set_channel_types({c: 'eog' for c in EOG_CH})
    raw.set_montage(mne.channels.make_standard_montage('standard_1020'),
                    on_missing='ignore')
    return raw


def get_cue_events(raw):
    """Return events for the 4 motor-imagery cues with event ids = class 1..4,
    and the positions of the cue events belonging to expert-rejected trials
    (GDF event 1023 sits on the trial-start marker; the cue is 2 s = 500 samples
    later, per desc_2a.pdf 'cue at t=2 s')."""
    events, ev_id = mne.events_from_annotations(raw, event_id=CUE_EVENT_IDS,
                                                verbose=False)
    rej_events, _ = mne.events_from_annotations(
        raw, event_id={'1023': 1}, verbose=False)
    rejected_cue_positions = rej_events[:, 0] + 500 if len(rej_events) else \
        np.array([], int)
    return events, rejected_cue_positions


def apply_reference(raw, reference='CAR'):
    """Apply a re-referencing scheme. 'mastoid' keeps the recording reference
    (left mastoid, i.e. no offline re-reference)."""
    picks_eeg = mne.pick_types(raw.info, eeg=True)
    if reference == 'CAR':
        raw.set_eeg_reference('average', projection=False)
    elif reference == 'mastoid':
        pass  # as recorded
    elif reference == 'laplacian':
        # Spherical-spline surface Laplacian (Perrin/Cohen/Kayser-Tenke).
        # Output channels are re-labelled 'csd' with units V/m^2; the rest of the
        # pipeline selects channels by name (CANON), so the type does not matter.
        raw = compute_current_source_density(raw, sphere='auto')
    else:
        raise ValueError(f'unknown reference: {reference}')
    return raw


def apply_notch(raw, notch_freq):
    if notch_freq:
        raw.notch_filter(freqs=[notch_freq], picks=CANON,
                         notch_widths=1.0, verbose=False)
    return raw


def apply_bandpass(raw, bandpass):
    if bandpass:
        raw.filter(l_freq=bandpass[0], h_freq=bandpass[1], picks=CANON,
                   fir_design='firwin', verbose=False)
    return raw


def apply_ica(raw, eog_channels=EOG_CH, n_components=None, threshold=3.0,
              random_state=42):
    """Fit ICA on the EEG channels and remove components that correlate with
    the EOG channels (blink / eye-movement artifacts). Returns (raw, n_excluded)."""
    ica = ICA(n_components=n_components, method='fastica',
              random_state=random_state, max_iter=500)
    ica.fit(raw, picks=CANON)
    eog_bads = ica.find_bads_eog(raw, ch_name=eog_channels,
                                 threshold=threshold)[0]
    ica.exclude = sorted(set(eog_bads))
    raw = ica.apply(raw)
    return raw, len(ica.exclude)


def build_epochs(raw, events, tmin=-1.5, tmax=4.5, baseline=(-1.5, 0)):
    """Epoch the (already re-referenced/filtered/ICA'd) raw around cue events,
    retaining only the 22 EEG channels and baseline-correcting."""
    epochs = mne.Epochs(raw, events, event_id={str(k): k for k in (1, 2, 3, 4)},
                        tmin=tmin, tmax=tmax, baseline=baseline,
                        picks=CANON, preload=True, on_missing='ignore',
                        reject=None)
    return epochs


def process(raw, events, reference='CAR', notch=None, bandpass=(8, 30),
            ica=False, ica_eog_channels=EOG_CH, ica_threshold=2.5, highpass=1.0):
    """Run the full preprocessing pipeline and return (epochs, n_ica_excluded).

    Recommended order (documented in the article):
        1. re-reference
        2. notch (optional)
        3. high-pass (drift removal; also the pre-ICA band)
        4. ICA on the wideband data (EOG-correlated components removed)
        5. task band-pass (optional; *none* keeps the wideband [highpass,100 Hz])
        6. epoch
    ICA is applied *before* the narrow band-pass precisely because blink / eye
    movement artefact is low-frequency and would already be removed by a mu-band
    filter, leaving no EOG signal for the ICA to find.
    """
    raw = raw.copy()
    raw = apply_reference(raw, reference)
    raw = apply_notch(raw, notch)
    raw.filter(l_freq=highpass, h_freq=None, picks=CANON,
               fir_design='firwin', verbose=False)          # high-pass (drift)
    n_excl = 0
    if ica:
        raw, n_excl = apply_ica(raw, eog_channels=ica_eog_channels,
                                threshold=ica_threshold)
    if bandpass:
        raw = apply_bandpass(raw, bandpass)
    epochs = build_epochs(raw, events)
    return epochs, n_excl


def _drop_expert_rejected(epochs, rejected_cue_positions, tol_samples=50):
    """Drop epochs whose cue onset is within tol_samples of an expert-rejected
    trial's cue position, matching the competition's 'artifact-free trials' rule."""
    if len(rejected_cue_positions) == 0:
        return epochs
    rec = epochs.events[:, 0]
    drop = np.zeros(len(rec), bool)
    for rt in rejected_cue_positions:
        drop |= np.abs(rec - rt) <= tol_samples
    keep = ~drop
    return epochs[keep]


def classify_accuracy(epochs, tmin=0.5, tmax=4.0, n_per_class=4,
                      n_splits=5, random_state=42):
    """4-class CSP + LDA classification accuracy via stratified k-fold CV.

    Uses MultiCSP (robust one-vs-rest CSP with shrunken covariances) followed
    by an LDA classifier. Filters are fit inside each CV fold (no leakage).
    """
    X = epochs.copy().crop(tmin=tmin, tmax=tmax).get_data()
    y = epochs.events[:, 2]
    clf = make_pipeline(MultiCSP(n_per_class=n_per_class), LinearDiscriminantAnalysis())
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    return float(scores.mean()), float(scores.std())


class MultiCSP(BaseEstimator, TransformerMixin):
    """Robust one-vs-rest Common Spatial Patterns (multiclass) with shrunken
    covariance matrices. Each class's filters maximise that class's spatial
    variance relative to the pooled variance of the other classes; log-variance
    features are concatenated across classes. Fit on train folds only."""

    def __init__(self, n_per_class=4, alpha=0.01):
        self.n_per_class = n_per_class
        self.alpha = alpha

    def _shrink(self, cov):
        d = cov.shape[0]
        floor = self.alpha * np.trace(cov) / d
        return cov + floor * np.eye(d)

    def _spatial_cov(self, X, mask):
        Xc = X[mask]
        covs = np.einsum('ect,edt->ecd', Xc, Xc) / Xc.shape[-1]
        return covs.mean(axis=0)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.filters_ = {}
        for c in self.classes_:
            S_c = self._spatial_cov(X, y == c)
            S_rest = self._spatial_cov(X, y != c)
            S_c = self._shrink(S_c)
            S_rest = self._shrink(S_rest)
            lam, W = scipy.linalg.eigh(S_c, S_rest)
            order = np.argsort(lam)[::-1][:self.n_per_class]
            self.filters_[c] = W[:, order]
        return self

    def transform(self, X):
        feats = []
        for c in self.classes_:
            W = self.filters_[c]
            proj = np.einsum('kc,ect->ekt', W.T, X)      # (n_ep, k, t)
            var = proj.var(axis=-1)                      # (n_ep, k)
            feats.append(np.log(var + 1e-12))
        return np.hstack(feats)


def csp_snr(epochs, tmin=0.5, tmax=4.0, alpha=0.01):
    """CSP-class-discriminative SNR proxy (dB).

    For each class the one-vs-rest generalised eigenvalue problem
        S_c w = lam * S_rest w
    is solved on the regularised spatial covariances of the imagery window; the
    largest eigenvalue lam_max is the strongest class-vs-rest power ratio.
    Reported value = 10*log10(mean(lam_max over classes)). Larger = more
    class-discriminative power = better effective signal-to-noise.
    """
    X = epochs.copy().crop(tmin=tmin, tmax=tmax).get_data()   # (n_ep, ch, t)
    y = epochs.events[:, 2]
    classes = np.unique(y)

    def spatial_cov(mask):
        Xc = X[mask]
        covs = np.einsum('ect,edt->ecd', Xc, Xc) / Xc.shape[-1]
        return covs.mean(axis=0)

    def shrink(cov):
        d = cov.shape[0]
        return cov + (alpha * np.trace(cov) / d) * np.eye(d)

    lam_max = []
    for c in classes:
        S_c = spatial_cov(y == c)
        S_rest = spatial_cov(y != c)
        S_c = shrink(S_c)
        S_rest = shrink(S_rest)
        lam = np.real(scipy.linalg.eigvalsh(S_c, S_rest))
        lam = lam[np.isfinite(lam) & (lam > 1e-12)]
        lam_max.append(lam.max())
    return float(10 * np.log10(np.mean(lam_max)))
