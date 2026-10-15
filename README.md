# WAVE_CLUST: Wavelet-based Analysis for Visualizing and Evaluating Clusters

**WAVE_CLUST** is a Python-based toolkit designed to analyze, visualize, and evaluate time-series or multi-frequency clusters using continuous wavelet transforms (CWT). Developed to decode complex, non-stationary biological signals (such as intracellular calcium $\text{Ca}^{2+}$ dynamics in retinal neurons and Müller glial cells), it seamlessly integrates spectral decomposition, Cone of Influence (COI) masking, max-ridge extraction, and unsupervised hierarchical clustering.

---

## 🌟 Key Features

* **Wavelet-Based Feature Extraction:** Extracts dynamic frequency, phase, and instantaneous power information across non-stationary time-series data using signal processing engines like `PyBOAT`. Features rigorous filtering via the Cone of Influence (COI) to discard edge artifacts.
* **Scientific Preprocessing & Normalization:** Automated pipeline for baseline estimation ($\Delta F/F_0$), adaptive signal detrending, and ROI-by-ROI quality control (QC) designed for analyzing dynamic cellular responses to photic or chemical stimuli.
* **Quantitative Cluster Evaluation:** Mathematical determination of the optimal partition size ($k$) by simultaneously evaluating cluster separation and cohesion metrics (*Silhouette Score*, *Davies-Bouldin Index*, and *Calinski-Harabasz Index*).
* **Multidimensional Visualization:**
  * Hierarchical clustering dendrograms with configurable cut-off thresholds.
  * Cluster-averaged time-frequency power spectro-temporal heatmaps.
  * Reduced 2D embeddings via **UMAP** (*Uniform Manifold Approximation and Projection*) to reveal underlying functional landscapes.
* **Interactive Jupyter Workflows:** Includes ready-to-run interactive notebooks (`WAVE_Clust.ipynb` and `WAVE-Clust_global.ipynb`) for rapid prototyping in local environments or Google Colab.

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone [https://github.com/maxirios27/WAVE_CLUST.git](https://github.com/maxirios27/WAVE_CLUST.git)
cd WAVE_CLUST

2. Install Dependencies
Bash
pip install -r requirements.txt
(Ensure pyboat, scikit-learn, scipy, pandas, matplotlib, seaborn, and umap-learn are installed).

3. Run the Notebooks
Open WAVE_Clust.ipynb or WAVE-Clust_global.ipynb in Jupyter Notebook, JupyterLab, or Google Colab to execute the step-by-step pipeline on your custom time-series datasets.

📖 Citation
If you use WAVE_CLUST in your research, publications, or software workflows, please cite the original peer-reviewed paper:

Rios, M. N., Marchese, N. A., & Guido, M. E. (2026). Unveiling Glial Heterogeneity in the Inner Retina: Non-Visual Photic Spectrum Responses and Calcium Dynamics in Müller Glial Cells vs. Neurons. International Journal of Molecular Sciences, 27(19), 8879. https://doi.org/10.3390/ijms27198879
