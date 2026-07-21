# SCRIPT: Pyboat Fluorescence Analysis
# ---------------------------------------------------
# Description: Automates fluorescence data processing via wavelets for ROI clustering.
# Data Sources: Requires 'Results.csv' and 'Overlay_Elements.csv' exported from ImageJ/Fiji.
# Workflow:
# 1. Normalization: Applies method (F/F0) using pre-stimulus mean.
# 2. Analysis: Extracts spectral features (ridges) via pyboat.
# 3. Integration: Maps temporal features with spatial coordinates.
# 4. Export: Saves features for clustering to the 'Topology' directory.
# Instructions:
# 1. Install dependencies: pandas, pyboat, numpy, matplotlib.
# 2. Update 'BASE_PATH' below with your data directory.
# 3. Place 'Results.csv' and 'Overlay_Elements.csv' in 'BASE_PATH'.
# ---------------------------------------------------


import pandas as pd
import pyboat
import numpy as np
import os
import matplotlib.pyplot as plt
import re


# --- PATH CONFIGURATION ---
BASE_PATH='/path/to/your/data'
RESULTS_FILE = os.path.join(BASE_PATH, 'Results.csv')
OVERLAY_FILE = os.path.join(BASE_PATH, 'Overlay_Elements.csv')
OUTPUT_DIR=os.path.join(BASE_PATH, 'Topology')
IMG_DIR = os.path.join(OUTPUT_DIR, 'quality_control_readouts')


# --- WAVELET PARAMETERS (Consistent with pyboat) ---
DT = 2.0 
PERIODS = np.linspace(8, 180, 200)
MAX_POWER_SCALE = 25.0
T_CUTOFF = 120.0
W_ENVELOPE = 60.0
POWER_THRESH = 0.5
T_STIM_START=60.0# Stimulus start [5]


os.makedirs(IMG_DIR, exist_ok=True)


# 1. DATA LOADING
print(f"Loading files from: {BASE_PATH}")
df_raw = pd.read_csv(RESULTS_FILE)
df_spatial = pd.read_csv(OVERLAY_FILE)


w = pyboat.WAnalyzer(periods=PERIODS, dt=DT, time_unit_label="s", p_max=MAX_POWER_SCALE)
time_vector = np.arange(len(df_raw)) * DT
master_df = pd.DataFrame({'time': time_vector})
spatial_metadata = []


print(f"Processing {len(df_raw.columns)-1} ROIs and merging topology...")


# 2. PROCESSING AND NORMALIZATION F/F0
for col in df_raw.columns:
# ROBUST FILTER: Ignore columns without numbers or metadata
   match = re.search(r'\d+', str(col))
   if not match or any(k in col.lower() for k in ['time', 'unnamed', 'index']):
       continue
  
   try:
       roi_num = int(match.group())
roi_id=roi_num-1# Sync base 0 with ImageJ Index [2]
      
# Link Coordinates
       roi_info = df_spatial[df_spatial['Index'] == roi_id]
       if roi_info.empty: continue
       x_c, y_c = roi_info['X'].values, roi_info['Y'].values


# Normalization F/F0 (Methodology Q1 [4])
       signal = df_raw[col].values
       n_basal = int(T_STIM_START / DT)
       f0 = np.mean(signal[:n_basal]) if n_basal > 0 else np.mean(signal)
       norm_signal = signal / f0


# Wavelet Analysis
       w.compute_spectrum(norm_signal, T_c=T_CUTOFF, window_size=W_ENVELOPE)
       ridge = w.get_maxRidge(power_thresh=POWER_THRESH)
      
# Consolidate Features for Colab [8]
       master_df[col] = w.ana_signal
      
       if ridge is not None and not ridge.empty:
           ridge_features = ridge.drop(columns=['time']).copy()
           ridge_features.columns = [f"{col}_{c}" for c in ridge_features.columns]
           master_df = pd.concat([master_df, ridge_features], axis=1)
          
           spatial_metadata.append({
               'ROI_Name': col, 'ROI_Index': roi_id, 'X': x_c, 'Y': y_c
           })
      
# Visual Quality Control and MEMORY RELEASE [3, 9]
       if roi_num % 20 == 0:
           w.plot_readout(draw_coi=True)
           plt.savefig(os.path.join(IMG_DIR, f"{col}_readout.png"), dpi=100)
      
plt.close('all') # <--- FINALLY SOLVES RUNTIMEWARNING


   except Exception as e:
print(f"\nError in {col}: {e}")
       plt.close('all')


# 3. FINAL EXPORT
master_df.to_csv(os.path.join(OUTPUT_DIR, 'df_combined_all_features.csv'), index=False)
df_spatial_final = pd.DataFrame(spatial_metadata)
df_spatial_final.to_csv(os.path.join(OUTPUT_DIR, 'spatial_mapping_metadata.csv'), index=False)


print(f"\nPROCESS COMPLETED!")
print(f"Files ready for the clustering model in Colab.")
