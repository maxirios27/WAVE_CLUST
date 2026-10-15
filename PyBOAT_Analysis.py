# ==============================================================================
# DEFINITIVE BATCH SCRIPT V5.6 - INTEGRATED GRAPHICAL LOGIC V2 (COI GUARANTEED)
# ==============================================================================

import glob
import os
import re
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Direct and safe import of WAnalyzer to avoid module AttributeError
import pyboat
try:
    from pyboat import WAnalyzer
except ImportError:
    from pyboat.api import WAnalyzer

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. EXPERIMENTAL MODELS AND PATHS CONFIGURATION
# ==============================================================================
MODELO_ACTUAL = 'NEURONAS_CORTICALES'  # Options: 'NEURONAS_CORTICALES', 'MIO_M1', 'NEURONAS', 'MGCS_CHICKEN'

CONFIG_MODELOS = {
    'Cortical_Neurons': {
        'DT': 1.0,                       # Frame sampling interval (1.0 s)
        'PERIODS': np.linspace(6, 80, 150),
        'T_STIM_START': 45.0,            # Stimulus onset time (45.0 s)
        'T_CUTOFF': 100.0,
        'W_ENVELOPE': 30.0,
        'INPUT_DIR': r'path/to/your/input_directory',   # <-- Replace with your input directory
        'OUTPUT_DIR': r'path/to/your/output_directory', # <-- Replace with your output directory
    },
    'MIO_M1': {
        'DT': 1.0,
        'PERIODS': np.linspace(6, 80, 150),
        'T_STIM_START': 45.0,
        'T_CUTOFF': 100.0,
        'W_ENVELOPE': 30.0,
        'INPUT_DIR': r'path/to/your/input_directory',
        'OUTPUT_DIR': r'path/to/your/output_directory',
    },
    'Retinal_Neurons': {
        'DT': 2.0,
        'PERIODS': np.linspace(8, 80, 150),
        'T_STIM_START': 60.0,
        'T_CUTOFF': 100.0,
        'W_ENVELOPE': 30.0,
        'INPUT_DIR': r'path/to/your/input_directory',
        'OUTPUT_DIR': r'path/to/your/output_directory',
    },
    'MGCS_CHICKEN': {
        'DT': 2.0,
        'PERIODS': np.linspace(8, 60, 100),
        'T_STIM_START': 45.0,
        'T_CUTOFF': 75.0,
        'W_ENVELOPE': 25.0,
        'INPUT_DIR': r'path/to/your/input_directory',
        'OUTPUT_DIR': r'path/to/your/output_directory',
    },
}

PARAMS = CONFIG_MODELOS[MODELO_ACTUAL]
DT = PARAMS['DT']
PERIODS = PARAMS['PERIODS']
T_STIM_START = PARAMS['T_STIM_START']
T_CUTOFF = PARAMS['T_CUTOFF']
W_ENVELOPE = PARAMS['W_ENVELOPE']
INPUT_DIR = PARAMS['INPUT_DIR']
OUTPUT_DIR = PARAMS['OUTPUT_DIR']

MAX_POWER_SCALE = 25.0
POWER_THRESH = 3.0
SHOW_COI = True
MAX_DELTA_F_LIMIT = 10.0


# ==============================================================================
# ROBUST FILE LOADING HELPER FUNCTION (SUPPORTS CSV DISGUISED AS .XLSX)
# ==============================================================================
def cargar_dataframe(filepath: str) -> pd.DataFrame:
    """Attempts to load as Excel; if it fails due to format mismatch, attempts as CSV."""
    if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        try:
            return pd.read_excel(filepath)
        except Exception:
            return pd.read_csv(filepath)
    else:
        try:
            return pd.read_csv(filepath)
        except Exception:
            return pd.read_excel(filepath)


# ==============================================================================
# 2. MAIN PROCESSING FUNCTION
# ==============================================================================
def procesar_archivo_cwt_v5(filepath: str, output_folder: str):
    file_stem = os.path.splitext(os.path.basename(filepath))[0]
    output_csv_path = os.path.join(output_folder, f'df_wavelet_{file_stem}.csv')
    output_qc_path = os.path.join(output_folder, f'qc_summary_{file_stem}.csv')

    img_dir = os.path.join(output_folder, f'plots_{file_stem}')
    os.makedirs(img_dir, exist_ok=True)

    print(f'\n--> Processing file: {os.path.basename(filepath)}', flush=True)

    # Smart file loading
    df_raw = cargar_dataframe(filepath)

    # 1. Strict column filtering (valid ROIs)
    valid_cols = []
    cols_to_ignore = [
        'time',
        'seg',
        'index',
        'unnamed',
        'frame',
        'slice',
        'label',
        'x',
        'y',
    ]
    for col in df_raw.columns:
        col_str = str(col).strip()
        if not col_str or col_str.lower().startswith('unnamed'):
            continue
        if any(k in col_str.lower() for k in cols_to_ignore) and not re.search(
            r'\d+', col_str
        ):
            continue
        if df_raw[col].dtype in [np.float64, np.int64, float, int]:
            valid_cols.append(col)

    print(f'   Detected {len(valid_cols)} valid cells/ROIs.', flush=True)

    time_vector = np.arange(len(df_raw)) * DT
    time_rounded = np.round(time_vector, 3)
    master_df = pd.DataFrame({'time': time_rounded})

    n_basal = int(T_STIM_START / DT)
    qc_records = []

    for idx, col in enumerate(valid_cols):
        col_name = str(col).strip()

        status = 'KEPT'
        f0 = np.nan
        max_val = np.nan

        try:
            signal = df_raw[col].values.astype(float)

            # 2. NaN and inactivity check
            if np.isnan(signal).all():
                qc_records.append({
                    'cell_id': col_name,
                    'f0_baseline': np.nan,
                    'max_raw_signal': np.nan,
                    'status': 'EXCLUDED',
                    'exclusion_reason': 'All_NaN_Data',
                })
                continue

            if np.isnan(signal).any():
                signal = pd.Series(signal).interpolate().bfill().ffill().values

            if np.std(signal) < 1e-6:
                qc_records.append({
                    'cell_id': col_name,
                    'f0_baseline': np.median(signal),
                    'max_raw_signal': np.max(signal),
                    'status': 'EXCLUDED',
                    'exclusion_reason': 'Flat_Signal_Std_Zero',
                })
                continue

            max_val = np.max(signal)

            # 3. Baseline F0 estimation and normalization
            basal_window = signal[:n_basal] if n_basal > 0 else signal
            f0 = (
                np.median(basal_window) if len(basal_window) > 0 else np.median(signal)
            )

            if f0 > 0.001:
                f_over_f0 = signal / f0
                delta_f_f0 = (signal - f0) / f0
            else:
                f_over_f0 = signal + 1.0
                delta_f_f0 = signal

            delta_f_f0_clean = np.clip(delta_f_f0, -0.99, MAX_DELTA_F_LIMIT)

            # Safe instance creation per ROI
            w = WAnalyzer(
                periods=PERIODS, dt=DT, time_unit_label='s', p_max=MAX_POWER_SCALE
            )

            # 4. Spectrum computation with V2 syntax
            with np.errstate(divide='ignore', invalid='ignore'):
                w.compute_spectrum(
                    delta_f_f0_clean,
                    T_c=T_CUTOFF,
                    window_size=W_ENVELOPE,
                    do_plot=True,
                    draw_coi=SHOW_COI,
                )

            # 5. Ridge extraction and overlay
            ridge = None
            try:
                ridge = w.get_maxRidge(power_thresh=POWER_THRESH)
                if ridge is not None and not ridge.empty:
                    if hasattr(w, 'draw_Ridge'):
                        w.draw_Ridge()
            except Exception:
                pass

            # 6. Save spectrogram
            plt.savefig(
                os.path.join(img_dir, f'{col_name}_spectrogram.png'),
                dpi=100,
                bbox_inches='tight',
            )
            plt.close('all')

            # 7. Master DataFrame assignment and COI masking
            master_df[f'{col_name}_raw'] = signal
            master_df[f'{col_name}_norm'] = f_over_f0
            master_df[f'{col_name}_delta_f_f0'] = delta_f_f0_clean
            
            master_df[f'{col_name}_detrended'] = getattr(w, 'ana_signal', delta_f_f0_clean)

            if ridge is not None and not ridge.empty:
                ridge_temp = ridge.copy()
                ridge_temp['time'] = np.round(ridge_temp['time'].values, 3)

                if hasattr(w, 'coi') and w.coi is not None:
                    coi_array = np.asarray(w.coi)[: len(ridge_temp)]
                    invalid_coi = ridge_temp['periods'].values > coi_array
                    
                    ridge_temp.loc[invalid_coi, ['periods', 'phase', 'power', 'amplitude']] = np.nan

                aligned_ridge = pd.merge(
                    master_df[['time']], ridge_temp, on='time', how='left'
                )

                master_df[f'{col_name}_periods'] = aligned_ridge['periods'].values
                master_df[f'{col_name}_phase'] = aligned_ridge['phase'].values
                master_df[f'{col_name}_power'] = aligned_ridge['power'].fillna(np.nan).values
                master_df[f'{col_name}_amplitude'] = aligned_ridge['amplitude'].fillna(np.nan).values
            else:
                master_df[f'{col_name}_periods'] = np.nan
                master_df[f'{col_name}_phase'] = np.nan
                master_df[f'{col_name}_power'] = np.nan
                master_df[f'{col_name}_amplitude'] = np.nan

            qc_records.append({
                'cell_id': col_name,
                'f0_baseline': f0,
                'max_raw_signal': max_val,
                'status': 'KEPT',
                'exclusion_reason': 'None',
            })

        except Exception as e:
            plt.close('all')
            print(f'\n   ❌ ERROR in ROI {col_name}: {type(e).__name__}: {e}', flush=True)
            qc_records.append({
                'cell_id': col_name,
                'f0_baseline': f0,
                'max_raw_signal': max_val,
                'status': 'EXCLUDED',
                'exclusion_reason': f'Processing_Error_{type(e).__name__}: {str(e)}',
            })

    # 8. Export
    master_df.to_csv(output_csv_path, index=False)
    df_qc = pd.DataFrame(qc_records)
    df_qc.to_csv(output_qc_path, index=False)

    n_kept = len(df_qc[df_qc['status'] == 'KEPT'])
    n_excl = len(df_qc[df_qc['status'] == 'EXCLUDED'])
    total = len(valid_cols)
    pct = (n_excl / total * 100) if total > 0 else 0.0
    print(
        f'\n   [OK] Saved {file_stem}: {n_kept}/{total} cells accepted,'
        f' {n_excl} excluded ({pct:.1f}%).',
        flush=True,
    )


# ==============================================================================
# 3. BATCH EXECUTION
# ==============================================================================
if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = glob.glob(
        os.path.join(INPUT_DIR, '**', '*.csv'), recursive=True
    ) + glob.glob(os.path.join(INPUT_DIR, '**', '*.xlsx'), recursive=True)

    files = [
        f
        for f in files
        if not os.path.basename(f).startswith(
            ('df_wavelet_', 'df_integrated_', 'qc_summary_')
        )
    ]

    print(
        f'Starting PyBOAT V5.6 Batch ({MODELO_ACTUAL}) on {len(files)}'
        ' file(s)...',
        flush=True,
    )
    for f in files:
        procesar_archivo_cwt_v5(f, OUTPUT_DIR)

    print('\n¡BATCH PROCESS COMPLETED SUCCESSFULLY!', flush=True)
