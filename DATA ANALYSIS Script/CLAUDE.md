# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a biomechanics data analysis pipeline for cyclogram accelerometer data from gait analysis studies. The project processes multi-plane (XY, XZ, YZ, 3D) accelerometer cyclogram metrics to identify gait impairment phenotypes in patients with lower extremity fractures through cluster analysis and statistical testing.

**Research Context**: Injured vs. contralateral limb comparison using delta/rho effect features to discover recovery phenotypes and bilateral asymmetries.

## Core Architecture

### Data Flow Pipeline

```
Raw Data (Data/)
  → Load & Parse (mean±std strings)
  → Effect Feature Engineering (delta/rho transforms)
  → Preprocessing (KNN imputation, robust scaling)
  → Clustering (GMM/KMeans/Hierarchical with consensus)
  → Statistical Testing (normality, bilateral, group comparisons)
  → Visualization & Export (plots, Excel, CSV)
```

### Three-Tier Analysis Scripts

1. **`cyclogram_publication_analysis.py`** - Comprehensive statistical analysis pipeline
   - Loads clustered data from `Data/Clustered_Data/`
   - Performs normality tests, bilateral comparisons, group tests, correlations
   - Generates publication-ready outputs in `data_output/RESULTS/`

2. **`injured_vs_contralateral_cluster_analysis.py`** - Phenotype discovery engine
   - Builds delta/rho effect features from injured vs. contralateral limbs
   - Multi-algorithm clustering (GMM, KMeans, Hierarchical)
   - Consensus clustering with bootstrap validation
   - Outputs to `data_output/RESULTS/CLUSTERS/`

3. **`cluster_analysis_complete.py`** - Extended cluster analysis and validation
   - Extends `injured_vs_contralateral_cluster_analysis.py`
   - Generates cluster characterization figures (radar, UMAP, dendrograms)
   - Cluster centroid statistical tests and stability metrics

### Feature Engineering Strategy

**Effect Features**: Transform left/right measurements to injured vs. control comparisons
- **Delta (Δ)**: Difference for bounded metrics (compactness, symmetry indices)
  - `delta_X = injured_X - control_X`
- **Rho (ρ)**: Log-ratio for unbounded positive metrics (area, perimeter, curvature)
  - `rho_X = log(injured_X / control_X)`
- **Circular**: Angular difference for phase metrics
  - `delta_angle = arctan2(sin(θ_injured - θ_control), cos(θ_injured - θ_control))`
- **Bilateral**: Already aggregated metrics (no left/right split)

**Feature Families** (from `FEATURE_FAMILIES` in `injured_vs_contralateral_cluster_analysis.py:60-70`):
- `sway_envelope`: area, perimeter, compactness, closure_error
- `smoothness`: trajectory_smoothness
- `curvature`: mean_curvature, curvature_std, curv_phase_* metrics
- `timing_sync`: relative_phase, MARP, coupling_angle_variability
- `asymmetry`: bilateral_symmetry_index, bilateral_mirror_correlation

## Directory Structure

```
Data/
  ├── ACC_XY.xlsx, ACC_XZ.xlsx, ACC_YZ.xlsx, ACC_3D.xlsx  # Raw input
  ├── patient_list.xlsx                                    # Patient metadata
  └── Clustered_Data/
      ├── ACC_*_CLUSTERED.xlsx                            # Pre-clustered versions
      └── CLUSTERED_DATA_SUMMARY.md                       # Cluster assignment docs

data_output/                                               # All outputs
  ├── RESULTS/
  │   ├── CLUSTERS/                                       # Cluster analysis outputs
  │   │   ├── figs/                                       # Cluster visualizations
  │   │   ├── logs/                                       # Audit trails
  │   │   ├── plane_memberships.csv
  │   │   ├── consensus_membership.csv
  │   │   └── cluster_centroid_tests.xlsx
  │   └── figs/                                           # Statistical plots
  └── visualizations/                                      # Publication figures

Backup/                                                    # Checkpointed versions
```

## Data Format Conventions

### Excel File Structure
Each `ACC_*.xlsx` file contains two sheets:
- **Cyclogram Sheet**: Sway envelope and trajectory metrics
- **Gait Sheet**: Temporal-spatial gait parameters

### Column Patterns
- `left_*` / `right_*`: Lateralized measurements
- `bilateral_*`: Pre-aggregated bilateral metrics
- `delta_*`: Difference features (injured - control)
- `rho_*`: Log-ratio features (log(injured/control))
- `*_Cluster`: Cluster assignment columns (0-N or "OUTLIER")

### Metadata Columns (Exclude from Analysis)
- Dataset
- Patient Name
- Patient ID
- Fracture Site (Lt/Rt)
- Fracture Name

## Common Development Tasks

### Running Full Analysis Pipeline

```bash
# Complete publication analysis (requires clustered data)
python3 cyclogram_publication_analysis.py

# Run clustering from raw data
python3 injured_vs_contralateral_cluster_analysis.py

# Generate cluster characterization figures
python3 cluster_analysis_complete.py

# Generate missing visualizations
python3 generate_missing_figures_simplified.py
```

### Data Processing Parameters

**Imputation Thresholds** (from `cyclogram_publication_analysis.py:51-53`):
- `IMPUTE_THRESHOLD = 0.05`: ≤5% missingness → KNN imputation
- `EXCLUDE_THRESHOLD = 0.10`: >10% missingness → exclude from tests

**Clustering Parameters** (from `injured_vs_contralateral_cluster_analysis.py:75-79`):
- `K_RANGE = range(2, 7)`: Test 2-6 clusters
- `BOOTSTRAP_ITERATIONS = 100`: Stability validation
- `JACCARD_THRESHOLD = 0.75`: Consensus agreement
- `SPEARMAN_THRESHOLD = 0.90`: Feature redundancy removal

### Mean±Std Parsing

All scripts use `parse_mean_std_string()` to handle "mean ± std" formatted cells:
```python
# Extracts mean value from "123.45 ± 12.34" → 123.45
# Also handles plain numeric values
```

## Critical Implementation Details

### Multi-Sheet Excel Processing
When loading Excel files, **always process both sheets**:
```python
xls = pd.ExcelFile(filepath)
cyclogram_sheet = pd.read_excel(xls, sheet_name=0)  # First sheet
gait_sheet = pd.read_excel(xls, sheet_name=1)       # Second sheet
```

### Circular Feature Handling
Phase metrics (angles) require circular statistics - never use standard arithmetic:
```python
# Convert to radians, compute circular difference
delta_angle = np.rad2deg(np.arctan2(
    np.sin(θ_injured - θ_control),
    np.cos(θ_injured - θ_control)
))
```

### FDR Correction
All p-values undergo FDR correction using Benjamini-Hochberg:
```python
from statsmodels.stats.multitest import multipletests
reject, pvals_corrected, _, _ = multipletests(p_values, method='fdr_bh')
```

### Audit Logging
All analysis scripts maintain timestamped audit trails:
```python
AUDIT_LOG = []

def log_action(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {message}"
    AUDIT_LOG.append(entry)
    print(entry)
```

## Output Expectations

### Cluster Analysis Outputs
- **plane_memberships.csv**: Per-plane cluster assignments (XY/XZ/YZ/3D)
- **consensus_membership.csv**: Cross-plane consensus clusters
- **cluster_centroid_tests.xlsx**: H₀: mean(delta/rho) = 0 tests per cluster
- **stability_metrics.csv**: Bootstrap Jaccard indices, silhouette scores

### Statistical Test Outputs
- **normality_tests.csv**: Shapiro-Wilk + D'Agostino-Pearson per metric
- **bilateral_tests.csv**: Paired t-test/Wilcoxon for left vs. right
- **group_comparison_tests.csv**: ANOVA/Kruskal-Wallis by fracture type/cluster
- **correlations.csv**: Cross-plane vs. 3D metric correlations

### Visualization Outputs
All figures saved as both `.png` (300 DPI) and `.pdf`:
- Cluster scatter plots (outliers color-coded, fracture site labeled)
- Radar plots (cluster phenotype characterization)
- UMAP/t-SNE embeddings
- Dendrograms (hierarchical clustering)

## Python Environment

**Python Version**: 3.12.3

**Core Dependencies**:
```
numpy==2.3.3
pandas==2.3.3
scipy==1.16.2
matplotlib==3.10.7
seaborn==0.13.2
scikit-learn (not version-specified, check with pip3 list)
statsmodels==0.14.5
openpyxl==3.1.5
umap-learn==0.5.9.post2
```

**Optional**:
- `plotly`: For Sankey diagrams (gracefully skipped if missing)

## Fracture Site Color Coding

**Standard Convention** (from `request.txt:20`):
- **Lt (Left)**: Red
- **Rt (Right)**: Blue
- **Outliers**: Different color (typically gray or black markers)

## Known Data Characteristics

From `request.txt:8-11`:
- **Total patients**: 143 in patient list
- **Total directories**: 234 (multi-visit patients)
- **Excel data rows**: 225
- **Multi-visit patients**: 60 (42%)

## Code Style Patterns

- **Warnings suppression**: `warnings.filterwarnings('ignore')` at script start
- **Path handling**: Use `pathlib.Path` exclusively
- **Error tolerance**: Graceful degradation with log warnings (e.g., missing files)
- **DataFrame operations**: Avoid chained assignments; use `.copy()` explicitly
- **Plotting**: Seaborn for statistical plots, Matplotlib for custom figures
- **File I/O**: Always use context managers for Excel operations

## Script Interdependencies

```
injured_vs_contralateral_cluster_analysis.py
  ↓ (generates clustered data)
cluster_analysis_complete.py (extends above)
  ↓ (imports functions from above)
cyclogram_publication_analysis.py (independent, uses clustered outputs)
  ↓ (loads clustered Excel files)
generate_missing_figures_simplified.py (visualization only)
```

**Import Pattern**: `cluster_analysis_complete.py` imports functions directly:
```python
from injured_vs_contralateral_cluster_analysis import (
    load_data, build_effect_features, preprocess_features,
    perform_clustering, select_best_model, consensus_clustering
)
```

---

## Detailed Methodological Plans and Formulas

### 1. Data Loading and Harmonization

**Location**: `cyclogram_publication_analysis.py:91-167` and `injured_vs_contralateral_cluster_analysis.py:100-180`

#### 1.1 Mean±Std String Parsing
```python
def parse_mean_std_string(value):
    """
    Parse 'mean ± std' format cells
    Input: "123.45 ± 12.34" or numeric value
    Output: 123.45 (mean only)
    """
    if '±' in str(value):
        mean_str = str(value).split('±')[0].strip()
        return float(mean_str)
    else:
        return float(value)
```

#### 1.2 Phase Metric Standardization
```python
# Normalize phase metrics to [0-100%]
if max_value ≤ 1.0:
    phase_value_normalized = phase_value × 100
```

#### 1.3 Symmetry Index Standardization
```python
# Normalize symmetry indices to [0-1]
if max_value > 1.0:
    symmetry_index_normalized = symmetry_index / 100
```

---

### 2. Effect Feature Engineering

**Location**: `injured_vs_contralateral_cluster_analysis.py:169-302`

#### 2.1 Delta Features (Bounded Metrics)
**Used for**: compactness, trajectory_smoothness, symmetry indices, mirror correlation, circular correlation

```
Δ_X = X_injured - X_contralateral
```

**Example**:
```python
delta_compactness = injured_compactness - control_compactness
```

**Interpretation**: Positive values indicate injured limb has higher compactness; negative indicates lower.

#### 2.2 Rho Features (Log-Ratio for Unbounded Metrics)
**Used for**: area, perimeter, curvature_std, mean_curvature, MARP, phase variability, RMS differences

```
ρ_X = log(X_injured / X_contralateral)
```

**Properties**:
- ρ > 0: Injured limb higher than contralateral
- ρ < 0: Injured limb lower than contralateral
- ρ = 0: Perfect symmetry
- Symmetric around zero (ρ(a/b) = -ρ(b/a))

**Example**:
```python
if injured_area > 0 and control_area > 0:
    rho_area = np.log(injured_area / control_area)
```

#### 2.3 Circular Features (Angular Metrics)
**Used for**: curv_phase_peak_phase_%, orientation_angle

```
θ_injured_rad = θ_injured × π/180
θ_control_rad = θ_control × π/180

Δθ = arctan2(sin(θ_injured - θ_control), cos(θ_injured - θ_control)) × 180/π
```

**Properties**:
- Handles angle wraparound (e.g., 359° and 1° are 2° apart)
- Result in range [-180°, 180°]

**Example**:
```python
injured_rad = np.deg2rad(injured_phase)
control_rad = np.deg2rad(control_phase)

delta_angle = np.rad2deg(np.arctan2(
    np.sin(injured_rad - control_rad),
    np.cos(injured_rad - control_rad)
))
```

#### 2.4 Bilateral Features
**Used for**: bilateral_symmetry_index, bilateral_mirror_correlation, bilateral_rms_trajectory_diff_*

```
No transformation needed - already aggregated across limbs
```

---

### 3. Quality Control and Preprocessing

**Location**: `cyclogram_publication_analysis.py:172-222` and `injured_vs_contralateral_cluster_analysis.py:307-445`

#### 3.1 Missingness Strategy
```
If missing_fraction = 0:
    → Keep as-is

If 0 < missing_fraction ≤ 0.05 (5%):
    → Median imputation (if skewness |s| > 1)
    → Mean imputation (if |s| ≤ 1)

If 0.05 < missing_fraction ≤ 0.15 (15%):
    → KNN imputation (k=5 neighbors)

If missing_fraction > 0.15 (15%):
    → Exclude from analysis
```

**Skewness Formula**:
```
s = E[(X - μ)³] / σ³
```

#### 3.2 KNN Imputation
```python
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X_features)
```

**Algorithm**: For each missing value, find 5 nearest complete observations using Euclidean distance on available features, then average their values.

#### 3.3 Winsorization (Outlier Capping)
```
X_winsorized = clip(X, q₀.₀₁, q₀.₉₉)

Where:
  q₀.₀₁ = 1st percentile
  q₀.₉₉ = 99th percentile
```

**Purpose**: Cap extreme outliers without removing observations entirely.

#### 3.4 Robust Scaling
```
X_scaled = (X - median(X)) / IQR(X)

Where:
  IQR = Q₃ - Q₁
  Q₁ = 25th percentile
  Q₃ = 75th percentile
```

**Applied per plane separately** to account for plane-specific distributions.

```python
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_features)
```

#### 3.5 Collinearity Removal
```
For each feature pair (i, j):
    ρ_Spearman = Spearman(X_i, X_j)

    If |ρ_Spearman| > 0.90:
        Drop feature with higher mean absolute correlation
```

**Spearman Correlation**:
```
ρ = 1 - (6 Σd²) / (n(n²-1))

Where:
  d = rank(X_i) - rank(X_j)
  n = number of observations
```

---

### 4. Clustering Algorithms

**Location**: `injured_vs_contralateral_cluster_analysis.py:449-610`

#### 4.1 Gaussian Mixture Model (GMM)
```python
gmm = GaussianMixture(
    n_components=k,
    covariance_type='full',  # Full covariance matrix
    random_state=42
)
labels = gmm.fit_predict(X)
```

**Model**:
```
P(x) = Σ(k=1 to K) π_k · N(x | μ_k, Σ_k)

Where:
  π_k = mixing coefficient (cluster weight)
  μ_k = cluster centroid
  Σ_k = full covariance matrix
```

**Selection Criteria**:
- BIC (Bayesian Information Criterion): `BIC = -2·log(L) + p·log(n)`
- AIC (Akaike Information Criterion): `AIC = -2·log(L) + 2p`
- Lower is better

#### 4.2 K-Means Clustering
```python
kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=20  # 20 random initializations
)
labels = kmeans.fit_predict(X)
```

**Algorithm**:
```
1. Initialize k centroids randomly
2. Assign each point to nearest centroid:
   c_i = argmin_k ||x_i - μ_k||²
3. Update centroids:
   μ_k = mean(x_i where c_i = k)
4. Repeat 2-3 until convergence
```

#### 4.3 Hierarchical Clustering (Ward Linkage)
```python
linkage_matrix = linkage(X, method='ward')
labels = fcluster(linkage_matrix, k, criterion='maxclust') - 1
```

**Ward's Method Distance**:
```
d(A, B) = √(2·n_A·n_B / (n_A + n_B)) · ||μ_A - μ_B||²

Where:
  n_A, n_B = cluster sizes
  μ_A, μ_B = cluster centroids
```

**Minimizes within-cluster variance** at each merge step.

---

### 5. Clustering Validation Metrics

**Location**: `injured_vs_contralateral_cluster_analysis.py:461-467`

#### 5.1 Silhouette Score
```
s_i = (b_i - a_i) / max(a_i, b_i)

Where:
  a_i = mean distance to points in same cluster
  b_i = mean distance to points in nearest other cluster

Silhouette = mean(s_i) ∈ [-1, 1]

Interpretation:
  s > 0.7: Strong clustering
  s > 0.5: Reasonable clustering
  s < 0.25: Weak/arbitrary clustering
```

#### 5.2 Davies-Bouldin Index
```
DBI = (1/k) Σ(i=1 to k) max(j≠i) [(σ_i + σ_j) / d(c_i, c_j)]

Where:
  σ_i = average distance of points in cluster i to centroid
  d(c_i, c_j) = distance between centroids

Lower is better (minimum = 0)
```

#### 5.3 Calinski-Harabasz Index
```
CH = [SS_B / (k-1)] / [SS_W / (n-k)]

Where:
  SS_B = between-cluster sum of squares
  SS_W = within-cluster sum of squares
  k = number of clusters
  n = number of observations

Higher is better
```

#### 5.4 Bootstrap Stability (Jaccard Index)
```python
def bootstrap_stability(X, labels, n_iterations=100):
    for i in range(n_iterations):
        # Bootstrap resample
        idx = random_sample_with_replacement(n)
        X_boot = X[idx]

        # Re-cluster
        labels_boot = cluster(X_boot)

        # Compute Adjusted Rand Index as Jaccard proxy
        ARI_i = adjusted_rand_score(labels[idx], labels_boot)
        jaccard_scores.append(max(0, ARI_i))

    return mean(jaccard_scores)
```

**Adjusted Rand Index**:
```
ARI = (RI - E[RI]) / (max(RI) - E[RI])

Where:
  RI = Rand Index = (TP + TN) / (TP + FP + TN + FN)

RI ∈ [-1, 1], with 1 = perfect agreement
```

---

### 6. Model Selection Strategy

**Location**: `injured_vs_contralateral_cluster_analysis.py:614-658`

#### 6.1 Composite Scoring
```
For each model:
    If Bootstrap_Jaccard > 0.75 AND Silhouette > 0:
        Composite_Score = Silhouette - (DBI/max_DBI) + Bootstrap_Jaccard

Best_Model = argmax(Composite_Score)
```

**Rationale**:
- Silhouette: Quality of cluster separation (maximize)
- DBI: Normalized cluster compactness (minimize)
- Bootstrap_Jaccard: Stability across resampling (maximize)

#### 6.2 Fallback Strategy
```
If no models with Jaccard > 0.75:
    Select model with highest Silhouette score

If no models with Silhouette > 0:
    Log warning: "No valid clustering found for this plane"
```

---

### 7. Consensus Clustering

**Location**: `injured_vs_contralateral_cluster_analysis.py:663-749`

#### 7.1 Co-Association Matrix
```
For each patient pair (i, j):
    coassoc[i, j] = (# planes where i and j in same cluster) / (# planes)

coassoc ∈ [0, 1]
```

**Algorithm**:
```python
co_assoc = zeros(n_patients, n_patients)

for plane in ['XY', 'XZ', 'YZ', '3D']:
    labels_plane = best_model[plane].labels
    patients_plane = get_patients(plane)

    for i in range(len(patients_plane)):
        for j in range(len(patients_plane)):
            if labels_plane[i] == labels_plane[j]:
                co_assoc[patient_idx[i], patient_idx[j]] += 1

co_assoc /= n_planes  # Normalize
```

#### 7.2 Consensus Hierarchical Clustering
```
# Convert similarity to distance
distance_matrix = 1 - co_assoc

# Symmetrize
distance_matrix = (distance_matrix + distance_matrix.T) / 2

# Hierarchical clustering on consensus distances
linkage_matrix = linkage(distance_matrix, method='average')

# Select K with best silhouette
for k in range(2, 7):
    consensus_labels = fcluster(linkage_matrix, k, criterion='maxclust')
    silhouette_k = silhouette_score(distance_matrix, consensus_labels, metric='precomputed')

best_k = argmax(silhouette_k)
```

**Average Linkage**:
```
d(A, B) = (1/(|A|·|B|)) Σ(i∈A) Σ(j∈B) d(i, j)
```

---

### 8. Statistical Testing Framework

#### 8.1 Normality Tests

**Location**: `cyclogram_publication_analysis.py:305-351`

##### Shapiro-Wilk Test
```
H₀: Data comes from normal distribution

W = (Σ a_i · x_(i))² / Σ(x_i - x̄)²

Where:
  x_(i) = i-th order statistic (sorted values)
  a_i = coefficients from expected order statistics

p-value calculated from W statistic
Reject H₀ if p < 0.05 (after FDR correction)
```

```python
stat, p_value = shapiro(values)
```

##### FDR Correction (Benjamini-Hochberg)
```
1. Sort p-values: p_(1) ≤ p_(2) ≤ ... ≤ p_(m)
2. Find largest i where: p_(i) ≤ (i/m) · α
3. Reject H₀ for all p_(j) where j ≤ i

α = 0.05 (family-wise error rate)
```

```python
reject, pvals_corrected, _, _ = multipletests(p_values, method='fdr_bh')
```

#### 8.2 Bilateral Comparisons (Left vs Right)

**Location**: `cyclogram_publication_analysis.py:356-429`

##### Paired t-test (If Normal)
```
H₀: μ_diff = 0

t = (x̄_diff - 0) / (s_diff / √n)

Where:
  x̄_diff = mean(Left - Right)
  s_diff = std(Left - Right)
  n = number of pairs

df = n - 1
p-value from t-distribution
```

```python
stat, p_value = ttest_rel(left_vals, right_vals)
```

##### Cohen's d Effect Size
```
d = x̄_diff / s_diff

Interpretation:
  |d| < 0.2: Negligible
  |d| < 0.5: Small
  |d| < 0.8: Medium
  |d| ≥ 0.8: Large
```

##### Wilcoxon Signed-Rank Test (If Non-Normal)
```
H₀: Median difference = 0

1. Compute differences: d_i = Left_i - Right_i
2. Rank absolute differences: rank(|d_i|)
3. Assign signs: signed_rank_i = sign(d_i) · rank(|d_i|)
4. W = sum of positive signed ranks

p-value from Wilcoxon distribution
```

```python
stat, p_value = wilcoxon(left_vals, right_vals)
```

##### Effect Size r
```
r = Z / √N

Where:
  Z = normalized test statistic
  N = number of pairs

Interpretation:
  |r| < 0.1: Negligible
  |r| < 0.3: Small
  |r| < 0.5: Medium
  |r| ≥ 0.5: Large
```

#### 8.3 Group Comparisons

**Location**: `cyclogram_publication_analysis.py:450-532`

##### One-Way ANOVA (If Normal and Homoscedastic)
```
H₀: μ₁ = μ₂ = ... = μ_k

F = MS_between / MS_within

Where:
  MS_between = SS_between / (k-1)
  MS_within = SS_within / (N-k)

  SS_between = Σ n_i(x̄_i - x̄)²
  SS_within = Σ Σ(x_ij - x̄_i)²

df₁ = k-1, df₂ = N-k
p-value from F-distribution
```

```python
stat, p_value = f_oneway(*groups)
```

##### Levene's Test (Homogeneity of Variance)
```
H₀: σ₁² = σ₂² = ... = σ_k²

Used to check ANOVA assumptions
If p < 0.05: unequal variances → use Kruskal-Wallis
```

##### Kruskal-Wallis Test (If Non-Normal)
```
H₀: All groups have same median

H = (12 / (N(N+1))) Σ (R_i² / n_i) - 3(N+1)

Where:
  R_i = sum of ranks for group i
  n_i = sample size of group i
  N = total sample size

p-value from χ² distribution with df=k-1
```

```python
stat, p_value = kruskal(*groups)
```

##### Eta-Squared Effect Size
```
η² = SS_between / SS_total

Interpretation:
  η² < 0.01: Negligible
  η² < 0.06: Small
  η² < 0.14: Medium
  η² ≥ 0.14: Large
```

#### 8.4 Cluster Centroid Tests

**Location**: `cluster_analysis_complete.py:71-150`

##### One-Sample t-test (H₀: mean = 0)
```
For each delta/rho feature in cluster:
    H₀: μ_feature = 0 (no injury effect)
    H₁: μ_feature ≠ 0 (injury effect present)

t = (x̄ - 0) / (s / √n)

df = n - 1
p-value from t-distribution (two-tailed)
```

```python
stat, p_value = ttest_1samp(cluster_values, popmean=0)
```

##### Wilcoxon Signed-Rank (H₀: median = 0)
```
For non-normal distributions:
    H₀: median_feature = 0

Similar to bilateral Wilcoxon but testing against zero
```

```python
stat, p_value = wilcoxon(cluster_values)
```

##### Effect Size
```
For t-test:
    Cohen's d = x̄ / s

For Wilcoxon:
    r = Z / √N
```

---

### 9. Correlation Analysis

**Location**: `cyclogram_publication_analysis.py:629-703`

#### 9.1 Pearson Correlation
```
r = Σ(x_i - x̄)(y_i - ȳ) / √[Σ(x_i - x̄)² · Σ(y_i - ȳ)²]

r ∈ [-1, 1]

Assumptions:
  - Linear relationship
  - Bivariate normal distribution
  - No major outliers
```

```python
pearson_r, pearson_p = pearsonr(x_plane, y_3D)
```

#### 9.2 Spearman Correlation
```
ρ = Pearson_correlation(rank(X), rank(Y))

ρ ∈ [-1, 1]

Advantages:
  - Robust to outliers
  - Captures monotonic (not just linear) relationships
  - No normality assumption
```

```python
spearman_r, spearman_p = spearmanr(x_plane, y_3D)
```

**Interpretation**:
```
|r| or |ρ| < 0.3: Weak
|r| or |ρ| < 0.7: Moderate
|r| or |ρ| ≥ 0.7: Strong
```

---

### 10. Phenotype Interpretation Framework

**Location**: `cluster_analysis_complete.py:181-268`

#### 10.1 Feature Signature Patterns

**Sway-Dominant Instability**:
```
Criteria:
  - High: delta_area > 0, delta_perimeter > 0, rho_area > 0
  - Low: delta_compactness < 0, delta_trajectory_smoothness < 0

Interpretation: ↑ CoM excursion, ↓ compactness, ↓ smoothness
Biomechanical deficit: Mediolateral control impairment
```

**Propulsion/Impact Dyscontrol**:
```
Criteria:
  - High: rho_curvature_std > 0, delta_curv_phase_entropy > 0
  - Low: delta_curv_phase_circular_corr < 0, delta_mean_relative_phase < 0

Interpretation: ↑ curvature variability, ↑ entropy, phase lag
Biomechanical deficit: Irregular push-off and landing patterns
```

**Rotational Compensation**:
```
Criteria:
  - High: eccentricity > threshold, orientation_angle bias
  - Low: delta_trajectory_smoothness < 0, phase_coherence < 0

Interpretation: ↑ eccentricity, orientation bias, ↓ phase coherence
Biomechanical deficit: Trunk rotation to offload injured limb
```

**Near-Recovered**:
```
Criteria:
  - Mean absolute effect size < 0.3 across all features
  - Symmetry indices near zero
  - High smoothness, low entropy

Interpretation: Functional restoration, near-normal gait patterns
```

#### 10.2 Significance Aggregation
```
For each cluster:
    sig_features = features where p_FDR < 0.05

    sway_score = count(sig_features ∩ sway_features)
    smoothness_score = count(sig_features ∩ smoothness_features)
    curvature_score = count(sig_features ∩ curvature_features)
    timing_score = count(sig_features ∩ timing_features)

    dominant_domain = argmax(domain_scores)
```

---

### 11. Descriptive Statistics

**Location**: `cyclogram_publication_analysis.py:227-300`

#### 11.1 Summary Statistics per Group
```
For each metric by (Plane, Fracture_Name, Fracture_Site, Cluster):
    N = count
    Mean = Σx_i / n
    SD = √[Σ(x_i - x̄)² / (n-1)]
    Median = 50th percentile
    Q1 = 25th percentile
    Q3 = 75th percentile
    IQR = Q3 - Q1
    Min = minimum value
    Max = maximum value
```

#### 11.2 Confidence Intervals
```
95% CI for mean:
    CI = x̄ ± t_(α/2, df) · (s / √n)

    Where t_(0.025, df) from t-distribution

Bootstrap percentile CI:
    1. Resample with replacement B times
    2. Compute statistic for each bootstrap sample
    3. CI = [percentile(2.5), percentile(97.5)]
```

---

### 12. Visualization Methods

#### 12.1 Radar Plots (Cluster Phenotype Characterization)
```
For each cluster:
    Select top 8 features by |effect_size|

    For feature_i at angle θ_i = 2πi/8:
        r_i = normalized_effect_size_i

    Plot polar coordinates (θ_i, r_i) and connect
```

#### 12.2 UMAP Dimensionality Reduction
```python
import umap
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=42
)
embedding = reducer.fit_transform(X_features)
```

**UMAP preserves** local neighborhood structure better than PCA or t-SNE for cluster visualization.

#### 12.3 Dendrogram (Hierarchical Clustering Visualization)
```python
from scipy.cluster.hierarchy import dendrogram
dendrogram(linkage_matrix,
           truncate_mode='lastp',
           p=20,  # Show last 20 merges
           orientation='top')
```

---

### 13. Output File Formats

#### 13.1 CSV Outputs
```
Structure:
  - Header row with column names
  - One row per observation/test result
  - Numeric precision: 6 decimal places
  - Missing values: empty cells (not NaN or NULL)
```

#### 13.2 Excel Outputs
```python
with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='Results', index=False)

    # Auto-adjust column widths
    worksheet = writer.sheets['Results']
    for column in worksheet.columns:
        max_length = max(len(str(cell.value)) for cell in column)
        worksheet.column_dimensions[column[0].column_letter].width = max_length + 2
```

#### 13.3 Figure Outputs
```python
# Save both PNG (raster) and PDF (vector)
fig.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{filename}.pdf', bbox_inches='tight')
```

**DPI Settings**:
- Figures: 300 DPI (publication quality)
- Display: 100 DPI (screen viewing)

---

### 14. Computational Complexity Notes

**Clustering Complexity**:
- GMM: O(n·k·d²·iterations) where d = dimensions
- K-Means: O(n·k·d·iterations)
- Hierarchical: O(n²·d) for linkage, O(n²) for distance matrix

**Bootstrap Stability**: O(B·n·k·d) where B = bootstrap iterations

**Consensus Clustering**: O(P·n²) where P = number of planes

**Memory Requirements**:
- Co-association matrix: O(n²) for n patients
- Feature matrix: O(n·d) for d features
- Recommend: 8GB RAM minimum for n > 200 patients

---

## Related Pipeline: Insole Gait Analysis (`insole-analysis.py`)

**Location**: `/home/shivam/Desktop/Human_Pose/PROJECT CYCLOGRAM/CYCLOGRAM-PROCESSING Script/insole-analysis.py`

This is a **separate but related** pipeline for processing smart insole pressure and IMU data to generate gait cyclograms. While the DATA ANALYSIS Script processes **existing cyclogram metrics from Excel files**, this CYCLOGRAM-PROCESSING Script **generates cyclograms from raw sensor data**.

### Pipeline Architecture

```
Raw Insole CSV (Temp-csv/)
  → Load & Calibrate (FFT adaptive filtering)
  → Gait Event Detection (heel-strike, toe-off)
  → 8-Phase Gait Segmentation (IC, LR, MSt, TSt, PSw, ISw, MSw, TSw)
  → Cyclogram Generation (2D: ACC_XY/XZ/YZ, 3D: ACC_XYZ)
  → Morphological Mean Cyclogram (MMC) computation
  → Advanced Metrics Calculation (Levels 1-3)
  → Visualization & Export (PNG + JSON metadata)
```

### Key Components

#### 1. Configuration (`InsoleConfig`)
```python
sampling_rate: 100.0 Hz
filter_cutoff: 20.0 Hz  # Butterworth lowpass
filter_order: 4

# Gait cycle constraints
min_cycle_duration: 0.8s
max_cycle_duration: 3.0s  # Increased for elderly/pathological gait

# Phase detection thresholds
pressure_threshold: 0.5  # Normalized
gyro_swing_threshold: 50.0 deg/s

# Validation (relaxed for pathological gait)
stance_swing_ratio_min: 0.8  # Allows asymmetric patterns
stance_swing_ratio_max: 3.0
bilateral_tolerance: 0.20  # 20% duration difference allowed

# GPU acceleration (CuPy)
use_gpu: True  # 30-40× speedup for MMC computation
```

#### 2. Data Structures

**GaitPhase** - Single gait phase (1-8):
- IC (Initial Contact, #1): Double support
- LR (Loading Response, #2): Double support
- MSt (Mid Stance, #3): Single support
- TSt (Terminal Stance, #4): Single support
- PSw (Pre-Swing, #5): Double support
- ISw (Initial Swing, #6): Swing
- MSw (Mid Swing, #7): Swing
- TSw (Terminal Swing, #8): Swing

**Support Type Classification**:
- Double Support: Both feet on ground (~20% of cycle)
- Single Support: One foot on ground (~40% of cycle)
- Swing: Foot off ground (~40% of cycle)

**CyclogramData** - Single gait cycle cyclogram:
```python
x_signal: np.ndarray  # e.g., ACC_X
y_signal: np.ndarray  # e.g., ACC_Y
z_signal: Optional[np.ndarray]  # For 3D cyclograms
phase_indices: List[int]  # Phase boundaries
phase_labels: List[str]  # For phase-colored visualization
```

**MorphologicalMeanCyclogram (MMC)** - Robust mean trajectory:
```python
median_trajectory: np.ndarray  # (101, 2) or (101, 3)
variance_envelope_lower/upper: np.ndarray
shape_dispersion_index: float
alignment_quality: float  # DTW-based
n_loops: int
median_area: float
```

#### 3. Data Loading (`Data_handling`)

**Multi-Version CSV Support**:
- Version 22: 2 header rows
- Version 23: 3 header rows
- Version 207: 4 header rows (with summary statistics)

**Auto-Detection Algorithm**:
```python
# Search first 20 lines for actual header
# Must contain: timestamp + pressure sensors + accelerometers
# Skip summary rows (GCT/Stance/Swing metrics only)
```

**Expected Columns**:
- `timestamp`: Milliseconds → converted to seconds
- `L_value1-4`, `R_value1-4`: Pressure sensors
  - value1, value3: Midfoot
  - value2: Forefoot
  - value4: Hindfoot (heel)
- `L_ACC_X/Y/Z`, `R_ACC_X/Y/Z`: Accelerometers
- `L_GYRO_X/Y/Z`, `R_GYRO_X/Y/Z`: Gyroscopes

**Filtering**:
```python
# Butterworth 4th-order lowpass
cutoff = 20 Hz (gait dynamics)
filtered = filtfilt(b, a, signal)  # Zero-phase filtering
```

#### 4. Gait Event Detection (`Gait_event_detection`)

**Precision Event Detection** (anatomically grounded):

##### Heel Strike Detection
```python
sensor: hindfoot (value4)
threshold: 10% of max pressure
signature: Sharp rise in hindfoot pressure

confidence = peak_pressure / max_pressure
```

##### Mid-Stance Detection
```python
sensors: midfoot (value1 + value3)
condition: midfoot_active AND forefoot_off
signature: Weight transfer to midfoot region
```

##### Toe-Off Detection
```python
sensor: forefoot (value2)
condition: Forefoot final release
signature: Pressure below threshold + gyro spike
```

#### 5. 8-Phase Gait Segmentation (`Gait_sub_event_detection`)

**Dynamic Phase Detection**:
```python
# Stance phases (1-5)
IC: Heel strike instant
LR: 0-10% of stance (weight acceptance)
MSt: 10-30% of stance (midfoot loading)
TSt: 30-50% of stance (heel off preparation)
PSw: 50-60% of stance (forefoot push-off)

# Swing phases (6-8)
ISw: 60-73% of cycle (foot clearance)
MSw: 73-87% of cycle (leg advancement)
TSw: 87-100% of cycle (deceleration)
```

**Perry's Gait Model Compliance**:
- Based on anatomical reference (Jacquelin Perry, 1992)
- Temporal percentages validated on 200+ subjects
- Support type classification for biomechanical accuracy

#### 6. Cyclogram Generation (`Cyclogram_preparation`)

**2D Cyclograms**:
```
ACC_XY: Mediolateral vs. Anteroposterior acceleration
ACC_XZ: Mediolateral vs. Vertical acceleration
ACC_YZ: Anteroposterior vs. Vertical acceleration
```

**3D Cyclogram**:
```
ACC_XYZ: Full 3D trajectory in acceleration space
```

**Resampling**:
```python
# Standardize to 101 points per gait cycle
target_length = 101
resampled = interp1d(old_indices, signal, kind='linear')
```

#### 7. Advanced Cyclogram Metrics (`AdvancedCyclogramMetrics`)

**Level 1: Geometric/Morphology Metrics**

##### Compactness Ratio
```
C = 4πA / P²

Where:
  A = area (shoelace formula)
  P = perimeter

C ∈ [0, 1], 1 = perfect circle
```

##### Aspect Ratio & Eccentricity (PCA-based ellipse fit)
```
# Covariance eigenvalues = axis lengths
eigenvalues, eigenvectors = eig(cov(x, y))

a = major_axis = √(λ₁) × 2
b = minor_axis = √(λ₂) × 2
aspect_ratio = a / b

eccentricity = √(1 - b²/a²)
e ∈ [0, 1], 0 = circle, 1 = line
```

##### Orientation Angle
```
θ = arctan2(eigenvector₁_y, eigenvector₁_x)
```

##### Curvature
```
κ = (x'y'' - y'x'') / (x'² + y'²)^(3/2)

Where:
  x', y' = first derivatives (gradient)
  x'', y'' = second derivatives
```

##### Trajectory Smoothness
```
smoothness = 1 / (1 + std(κ))

Higher smoothness = better motor control
```

**Level 2: Temporal-Coupling Metrics**

##### Continuous Relative Phase (CRP)
```python
# Hilbert transform for phase extraction
phase_x = angle(hilbert(ACC_X))
phase_y = angle(hilbert(ACC_Y))

CRP = angle(exp(1j × (phase_x - phase_y)))  # Wrap to [-π, π]
```

##### Mean Absolute Relative Phase (MARP)
```
MARP = mean(|CRP|)

Interpretation:
  MARP < π/4: Strong in-phase coupling
  MARP > 3π/4: Anti-phase coordination
```

##### Coupling Angle Variability
```
coupling_angle = arctan2(∇y, ∇x)
CAV = std(coupling_angle)

Low CAV = stable coordination
```

##### Phase Shift
```
phase_shift = mean(phase_x - phase_y)

Indicates timing lag between signals
```

**Level 3: Bilateral Symmetry Metrics**

##### Symmetry Index (SI)
```
SI = [(Area_L - Area_R) / mean(Area_L, Area_R)] × 100

SI = 0: Perfect symmetry
|SI| > 10: Asymmetric gait
```

##### Mirror Correlation
```
# Mirror right cyclogram horizontally
right_x_mirrored = -right_x

mirror_corr = corrcoef(left_x, right_x_mirrored)[0, 1]

High correlation = bilateral similarity
```

##### RMS Trajectory Difference
```
RMS_x = √(mean((left_x - right_x)²))
RMS_y = √(mean((left_y - right_y)²))
RMS_total = √(RMS_x² + RMS_y²)

Functional asymmetry measure
```

#### 8. Morphological Mean Cyclogram (`MorphologicalMeanCyclogram`)

**Robust Median Computation** (replaces naive averaging):

##### Algorithm Steps
```
1. Find median reference loop:
   - Compute all pairwise DTW distances
   - Select loop with minimum total distance

2. Center all loops to centroid:
   - centroid = (mean(x), mean(y))
   - centered_loop = loop - centroid

3. Rescale to median area:
   - areas = [compute_area(loop) for loop in loops]
   - median_area = median(|areas|)
   - scale_factor = √(median_area / current_area)
   - rescaled = centered_loop × scale_factor

4. Compute median shape:
   - Stack aligned loops: (n_loops, 101, 2)
   - median_trajectory = median(stacked_loops, axis=0)

5. Variance envelope:
   - std_trajectory = std(stacked_loops, axis=0)
   - envelope_lower = median - std
   - envelope_upper = median + std

6. Shape dispersion index:
   - var_x = var(loops[:, :, 0], axis=0)
   - var_y = var(loops[:, :, 1], axis=0)
   - SDI = (mean(var_x) + mean(var_y)) / median_area
```

##### DTW Distance (Dynamic Time Warping)
```python
# Using fastdtw library
A = stack([loop1.x, loop1.y], axis=1)
B = stack([loop2.x, loop2.y], axis=1)

distance, path = fastdtw(A, B, dist=euclidean_norm)
normalized_distance = distance / max(len(A), len(B))
```

##### GPU Acceleration (CuPy)
```python
# 30-40× speedup for large datasets
if GPU_AVAILABLE and use_gpu:
    import cupy as cp
    loops_gpu = cp.array(loops)
    median_gpu = cp.median(loops_gpu, axis=0)
    median = cp.asnumpy(median_gpu)
```

#### 9. Visualization & Export

**7 Plot Sets Generated**:
1. **Multi-Cycle Gait**: Individual cycles overlaid
2. **Stride Cyclograms**: Per-stride ACC_XY/XZ/YZ
3. **MMC Cyclograms**: Morphological mean with variance envelope
4. **2D Gait Cyclograms**: Aggregate 2D planes
5. **3D Cyclograms**: Full 3D trajectory visualization
6. **3D Gait**: 3D aggregate cyclogram
7. **Gait Event Timeline**: Pressure sensors + phase annotations

**Dual Output Format**:
- `.png`: 300 DPI publication-quality raster
- `.json`: Complete metadata with all metrics

**JSON Metadata Structure**:
```json
{
  "file_info": {
    "csv_source": "filename.csv",
    "subject_name": "extracted_from_filename",
    "processing_timestamp": "ISO-8601"
  },
  "gait_summary": {
    "left_cycles": N,
    "right_cycles": N,
    "total_duration": seconds
  },
  "stride_events": [
    {
      "leg": "left/right",
      "stride_id": 1,
      "start_time": seconds,
      "duration": seconds,
      "phases": [
        {
          "phase_name": "IC",
          "phase_number": 1,
          "support_type": "double_support",
          "start_time": seconds,
          "duration": seconds
        }
      ],
      "cyclograms": [
        {
          "type": "ACC_X_vs_ACC_Y",
          "area": float,
          "perimeter": float,
          "closure_error": float
        }
      ]
    }
  ],
  "cyclogram_metrics": {
    "geometric": {...},
    "temporal": {...},
    "bilateral": {...}
  },
  "mmc_metrics": {
    "n_loops": int,
    "median_area": float,
    "shape_dispersion_index": float,
    "alignment_quality": float
  }
}
```

#### 10. CSV Exports

**precision_gait_events.csv**:
```
leg, event_type, event_start_time, event_end_time, duration_ms,
sensor_source, frame_start, frame_end, confidence
```

**detailed_gait_phases_{left/right}.csv**:
```
leg, cycle_id, cycle_start_time, cycle_duration,
phase_number, phase_name, support_type,
phase_start_time, phase_end_time, phase_duration,
phase_start_idx, phase_end_idx, percentage_of_cycle
```

#### 11. Batch Processing

```python
INPUT_DIR = Path("insole-sample/Temp-csv")
OUTPUT_DIR = Path("insole-output/DEMC")

# Auto-discover all CSV files
CSV_FILES = list(INPUT_DIR.glob("*.csv"))

# Process all files sequentially
for csv_file in CSV_FILES:
    pipeline.process_file(csv_file)
```

### Integration with DATA ANALYSIS Script

The insole-analysis.py output can feed into the DATA ANALYSIS Script:

1. **Insole Processing** generates per-stride cyclogram metrics
2. **Manual aggregation** creates Excel files with left/right columns
3. **DATA ANALYSIS Script** performs statistical clustering and analysis

**Column Mapping**:
```
insole-analysis.py metrics → Excel columns
─────────────────────────────────────────
area                      → left_area, right_area
perimeter                 → left_perimeter, right_perimeter
compactness_ratio         → left_compactness, right_compactness
mean_curvature            → left_mean_curvature, right_mean_curvature
trajectory_smoothness     → left_trajectory_smoothness, right_trajectory_smoothness
mean_relative_phase       → left_mean_relative_phase, right_mean_relative_phase
symmetry_index            → bilateral_symmetry_index
mirror_correlation        → bilateral_mirror_correlation
```

### Key Differences from DATA ANALYSIS Script

| Aspect | insole-analysis.py | DATA ANALYSIS Script |
|--------|-------------------|---------------------|
| **Input** | Raw CSV sensor data | Pre-computed Excel metrics |
| **Purpose** | Generate cyclograms | Analyze cyclogram phenotypes |
| **Output** | PNG plots + JSON metadata | Statistical tests + clusters |
| **Processing** | Single-subject, multi-cycle | Multi-subject, aggregated |
| **Metrics** | Real-time computation | Batch statistical analysis |
| **GPU** | CuPy acceleration for MMC | CPU-only (scikit-learn) |

### Performance Notes

**Computational Complexity**:
- Gait event detection: O(n) where n = number of samples
- 8-phase segmentation: O(m) where m = number of detected events
- MMC computation (CPU): O(L·N²) where L = loop length, N = number of cycles
- MMC computation (GPU): O(L·N²/P) where P = parallelization factor (~30-40)

**Typical Processing Times** (100Hz, 60s recording):
- Data loading & filtering: ~0.5s
- Gait event detection: ~1s
- Cyclogram generation: ~2s
- MMC computation (CPU): ~15s for 20 cycles
- MMC computation (GPU): ~0.5s for 20 cycles
- Visualization: ~5s
- **Total**: ~8s (GPU) or ~23s (CPU)

**Memory Requirements**:
- Raw data: ~10MB per 60s recording
- Filtered signals: ~5MB
- Cyclograms (all cycles): ~2MB
- Peak usage: ~50MB per file
- GPU: Requires 2GB+ VRAM for CuPy
