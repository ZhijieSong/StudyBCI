"""Run the section-7 comparison grid for BCI IV 2a subject A01.

For each configuration we report 4-class CSP+LDA accuracy (stratified 5-fold CV)
and the CSP class-discriminative SNR proxy (dB). Every config is run on the same
expert-artifact-free trial set; the only thing that changes between rows is the
preprocessing choice under test (reference / notch / bandpass / ICA).

Usage:  python run_grid.py
"""
import sys
import os
import logging
import time

logging.getLogger('mne').setLevel(logging.ERROR)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np
import pandas as pd

from eeg_preproc.preprocess import (load_raw, get_cue_events, process,
                                    _drop_expert_rejected, classify_accuracy,
                                    csp_snr)

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'A01T.gdf')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')

# Configurations: (label, reference, notch, bandpass, ica)
CONFIGS = [
    ('mastoid | wideband | no-notch | no-ICA',
     'mastoid', None, None, False),
    ('CAR | wideband | no-notch | no-ICA',
     'CAR', None, None, False),
    ('CAR | wideband | notch | no-ICA',
     'CAR', 50, None, False),
    ('CAR | wideband | notch | ICA',
     'CAR', 50, None, True),
    ('CAR | band8-30 | no-notch | ICA   [recommended]',
     'CAR', None, (8, 30), True),
    ('CAR | band8-30 | no-notch | no-ICA',
     'CAR', None, (8, 30), False),
    ('mastoid | band8-30 | no-notch | ICA',
     'mastoid', None, (8, 30), True),
    ('laplacian | band8-30 | no-notch | ICA',
     'laplacian', None, (8, 30), True),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    raw = load_raw(DATA)
    events, rej_pos = get_cue_events(raw)

    rows = []
    t0 = time.time()
    for label, ref, notch, band, ica in CONFIGS:
        t1 = time.time()
        ep, n_excl = process(raw, events, reference=ref, notch=notch,
                             bandpass=band, ica=ica)
        ep = _drop_expert_rejected(ep, rej_pos)
        n_epochs = len(ep.events)
        acc, acc_std = classify_accuracy(ep)
        snr = csp_snr(ep)
        d = {
            'config': label,
            'reference': ref,
            'bandpass': f'{band}' if band else 'wideband(1-100Hz)',
            'notch': 'no' if not notch else f'{notch}Hz',
            'ica': 'yes' if ica else 'no',
            'ica_excluded': n_excl,
            'n_epochs': n_epochs,
            'accuracy': round(acc, 4),
            'accuracy_std': round(acc_std, 4),
            'snr_db': round(snr, 3),
        }
        rows.append(d)
        print(f"[{time.time()-t1:5.1f}s] {label:55s} acc={acc:.3f}±{acc_std:.3f} "
              f"snr={snr:.2f}dB n={n_epochs} ica_ex={n_excl}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'comparison_table.csv'), index=False)
    print('---')
    print(df.to_string(index=False))
    print(f"\nTotal time {time.time()-t0:.1f}s. Table -> {os.path.join(OUT, 'comparison_table.csv')}")


if __name__ == '__main__':
    main()
