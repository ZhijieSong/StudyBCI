"""Generate the figures for the section-7 comparison.

Produces:
  results/spectrum_preprocessing.png  - PSD of raw vs after CAR / notch / bandpass
  results/montage_1020.png            - the 22-channel 10-20 montage
  results/comparison_results.png      - accuracy + CSP-SNR bar charts
  results/noise_analysis.txt          - numeric 50 Hz line-noise / band-power summary
"""
import os
import sys
import logging

logging.getLogger('mne').setLevel(logging.ERROR)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from eeg_preproc.preprocess import load_raw, CANON

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
DATA = os.path.join(ROOT, 'data', 'A01T.gdf')
OUT = os.path.join(ROOT, 'results')
os.makedirs(OUT, exist_ok=True)


def puv(raw):
    """Band-averaged power spectrum in uV^2/Hz."""
    psd = raw.compute_psd(method='welch', fmin=1, fmax=120, n_fft=4096,
                          verbose=False)
    return psd.freqs, psd.get_data().mean(axis=0) * 1e12


def band_p(fre, P, f0, f1):
    m = (fre >= f0) & (fre < f1)
    return P[m].mean()


def main():
    raw = load_raw(DATA)

    # ---------------- spectra ----------------
    raw_m = raw.copy().pick(CANON)                                   # as recorded (mastoid)
    raw_car = raw.copy(); raw_car.set_eeg_reference('average', projection=False)
    raw_car = raw_car.pick(CANON)
    raw_car_notch = raw.copy(); raw_car_notch.set_eeg_reference('average', projection=False)
    raw_car_notch.notch_filter(freqs=[50], picks=CANON, notch_widths=1.0, verbose=False)
    raw_car_notch = raw_car_notch.pick(CANON)
    raw_bp = raw.copy(); raw_bp.set_eeg_reference('average', projection=False)
    raw_bp.filter(l_freq=8, h_freq=30, picks=CANON, fir_design='firwin', verbose=False)
    raw_bp = raw_bp.pick(CANON)

    sigs = [('raw (mastoid ref)', raw_m),
            ('CAR', raw_car),
            ('CAR + 50Hz notch', raw_car_notch),
            ('CAR + bandpass 8-30Hz', raw_bp)]
    fre = None
    freqs = []
    series = []
    for name, r in sigs:
        f, P = puv(r)
        freqs.append(f); series.append((name, P))

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, P in series:
        ax.semilogy(freqs[0], P, lw=1.4, label=name)
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Power (µV²/Hz, log)')
    ax.set_title('A01T PSD: effect of re-reference / notch / bandpass')
    ax.set_xlim(0, 120); ax.set_ylim(1e-5, 5e1); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'spectrum_preprocessing.png'), dpi=150)
    plt.close(fig)

    # numeric 50 Hz analysis on the raw, as-recorded spectrum
    f, P = freqs[0], series[0][1]
    c = band_p(f, P, 49.5, 50.5)
    nb = (band_p(f, P, 47, 49) + band_p(f, P, 51, 53)) / 2
    lines = []
    lines.append('=== A01T noise analysis (band-averaged power, uV^2/Hz) ===')
    lines.append(f'as-recorded (mastoid ref, 0.5-100Hz + 50Hz notch already on):')
    lines.append(f'  50Hz peak (49.5-50.5): {c:.4f}')
    lines.append(f'  local neighbours  (47-49 & 51-53 avg): {nb:.4f}')
    lines.append(f'  50Hz prominence ratio (peak/neighbour): {c/nb:.3f}  (≈1 => no line-noise peak present)')
    for nm, P in series:
        lines.append(f'{nm:24s}: mu8-13={band_p(f,P,8,13):.3f}  beta15-30={band_p(f,P,15,30):.3f}  '
                     f'45-55={band_p(f,P,45,55):.3f}  EMG55-95={band_p(f,P,55,95):.3f}')
    with open(os.path.join(OUT, 'noise_analysis.txt'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('spectrum + noise analysis written')

    # ---------------- montage ----------------
    fig = raw.copy().pick(CANON).plot_sensors(show_names=True, show=False,
                                              sphere='auto', block=False)
    fig.savefig(os.path.join(OUT, 'montage_1020.png'), dpi=150)
    plt.close(fig)
    print('montage written')

    # ---------------- results bar chart ----------------
    df = pd.read_csv(os.path.join(OUT, 'comparison_table.csv'))
    codes = [f'C{i+1}' for i in range(len(df))]
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.5))
    xs = np.arange(len(df))
    axs[0].bar(xs, df['accuracy'] * 100, color='#3b7dd8')
    axs[0].errorbar(xs, df['accuracy'] * 100, yerr=df['accuracy_std'] * 100,
                    fmt='none', ecolor='k', lw=1)
    axs[0].set_ylabel('4-class accuracy (%)'); axs[0].set_ylim(0, 100)
    axs[0].set_xticks(xs); axs[0].set_xticklabels(codes, fontsize=9)
    axs[0].set_title('Classification accuracy (CSP+LDA, 5-fold CV, A01T)')
    axs[0].grid(axis='y', alpha=.3)
    for x, v in zip(xs, df['accuracy'] * 100):
        axs[0].text(x, v + 2, f'{v:.0f}', ha='center', fontsize=7)

    axs[1].bar(xs, df['snr_db'], color='#d87f3b')
    axs[1].set_ylabel('CSP discriminative SNR (dB)'); axs[1].set_ylim(0, 2.5)
    axs[1].set_xticks(xs); axs[1].set_xticklabels(codes, fontsize=9)
    axs[1].set_title('CSP class-discriminative SNR')
    axs[1].grid(axis='y', alpha=.3)
    for x, v in zip(xs, df['snr_db']):
        axs[1].text(x, v + 0.04, f'{v:.2f}', ha='center', fontsize=7)

    legend = '\n'.join(f'{c}: {cfg}' for c, cfg in zip(codes, df['config']))
    fig.text(0.5, -0.02, legend, ha='center', va='top', fontsize=6.5, family='monospace')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'comparison_results.png'), dpi=150,
                                    bbox_inches='tight')
    plt.close(fig)
    print('results chart written')


if __name__ == '__main__':
    main()
