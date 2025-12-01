# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Comprehensive gait analysis research system for biomechanical assessment using **cyclograms** (angle-angle phase diagrams) from dual data sources: MediaPipe pose estimation and smart insole pressure/IMU sensors. The project encompasses both raw data processing (cyclogram generation) and statistical analysis (phenotype discovery).

**Core Workflows**:
1. **Cyclogram Generation**: Process raw sensor data → Generate cyclograms with gait metrics → Export visualizations + metadata
2. **Statistical Analysis**: Load cyclogram metrics → Cluster analysis → Phenotype discovery → Publication-ready statistical tests

**Research Focus**: Lower extremity fracture recovery - comparing injured vs. contralateral limb gait patterns to identify recovery phenotypes and bilateral asymmetries.

## Repository Structure

```
PROJECT CYCLOGRAM/
├── CYCLOGRAM-PROCESSING Script/   # Raw data → cyclograms (Level 1)
│   ├── Pose-Analysis.py           # MediaPipe pose cyclograms
│   ├── insole-analysis.py         # Smart insole cyclograms
│   ├── gpu_acceleration.py        # CuPy GPU acceleration for MMC
│   └── Pose-Checkpoint.py         # Backup version
├── DATA ANALYSIS Script/          # Cyclogram metrics → statistics (Level 2)
│   ├── cyclogram_publication_analysis.py        # Main statistical pipeline
│   ├── injured_vs_contralateral_cluster_analysis.py  # Phenotype discovery
│   ├── cluster_analysis_complete.py             # Extended cluster validation
│   ├── generate_missing_figures_simplified.py   # Visualization generator
│   └── Data/                      # Excel input files + clustered outputs
├── claudedocs/                    # Technical documentation (Level 3)
│   ├── CLAUDE.md                  # Pose analysis comprehensive guide
│   ├── cyclogram_system_design.md # Architecture specification
│   ├── MMC_METHODOLOGY.md         # Morphological Mean Cyclogram
│   └── [other technical docs]
├── Dataset/                       # Patient data and outputs
│   ├── PATIENT INFORMATION/       # Patient metadata and SPPB scores
│   ├── CYCLOGRAM-PROCESSED/       # Processed cyclogram outputs
│   └── PLOTS/                     # Generated visualizations
└── .venv/                         # Python virtual environment
```

## Python Environment

**Version**: Python 3.12
**Virtual Environment**: `.venv/` (activate before running scripts)

```bash
# Activate environment (Linux/macOS)
source .venv/bin/activate

# Verify activation
which python3  # Should point to .venv/bin/python3
```

**Key Dependencies**:
- `numpy==2.3.3` - Numerical computing
- `pandas==2.3.3` - Data manipulation
- `scipy==1.16.2` - Scientific computing, signal processing
- `matplotlib==3.10.7` - Visualization
- `scikit-learn` - Machine learning, clustering
- `statsmodels==0.14.5` - Statistical tests
- `openpyxl==3.1.5` - Excel I/O
- `cupy` (optional) - GPU acceleration for MMC computation (30-40× speedup)
- `fastdtw` (optional) - Dynamic Time Warping acceleration

**Install dependencies** (if needed):
```bash
pip3 install numpy pandas scipy matplotlib scikit-learn statsmodels openpyxl
pip3 install fastdtw  # Optional but recommended for DTW speedup
pip3 install cupy-cuda12x  # Optional GPU acceleration (requires NVIDIA GPU + CUDA)
```

## Common Commands

### Cyclogram Generation (CYCLOGRAM-PROCESSING Script/)

**Pose-based analysis** (MediaPipe joint angles):
```bash
# Activate environment
source .venv/bin/activate

# Basic usage - all subjects (publication mode: 10×8" @ 300 DPI)
python3 "CYCLOGRAM-PROCESSING Script/Pose-Analysis.py"

# Single subject (specify exact folder name from Dataset/)
python3 "CYCLOGRAM-PROCESSING Script/Pose-Analysis.py" \
  --subject-name "Openpose_조정자_1917321_20240117_1"

# Adaptive mode for content-aware sizing (smaller file sizes)
python3 "CYCLOGRAM-PROCESSING Script/Pose-Analysis.py" --adaptive-mode

# Custom parameters with locked axis limits for direct comparison
python3 "CYCLOGRAM-PROCESSING Script/Pose-Analysis.py" \
  --smooth-window 15 \
  --smooth-threshold 7.0 \
  --lock-axis-limits

# Disable angle enhancement (if raw data quality is excellent)
python3 "CYCLOGRAM-PROCESSING Script/Pose-Analysis.py" --no-enhance-angles

# Custom input/output directories
python3 "CYCLOGRAM-PROCESSING Script/Pose-Analysis.py" \
  --input-dir "/path/to/data" \
  --output-dir "/path/to/results"
```

**Insole pressure analysis** (smart insole IMU + pressure):
```bash
# Single subject
python3 "CYCLOGRAM-PROCESSING Script/insole-analysis.py" \
  --input insole-sample/Temp-csv/10MWT.csv \
  --output insole-output/DEMC/10MWT

# Batch processing (all CSVs in insole-sample/Temp-csv/)
python3 "CYCLOGRAM-PROCESSING Script/insole-analysis.py" --batch

# Custom sampling rate and filter
python3 "CYCLOGRAM-PROCESSING Script/insole-analysis.py" \
  --input data.csv \
  --sampling-rate 100 \
  --filter-cutoff 20.0
```

### Statistical Analysis (DATA ANALYSIS Script/)

```bash
# Navigate to analysis directory
cd "DATA ANALYSIS Script"

# Complete publication analysis (requires clustered data in Data/Clustered_Data/)
python3 cyclogram_publication_analysis.py

# Run clustering from raw data (generates clustered Excel files)
python3 injured_vs_contralateral_cluster_analysis.py

# Extended cluster characterization with radar plots, UMAP, dendrograms
python3 cluster_analysis_complete.py

# Generate missing publication figures
python3 generate_missing_figures_simplified.py
```

## Architecture Overview

### Dual-Pipeline Design

**Pipeline 1: Cyclogram Generation** (`CYCLOGRAM-PROCESSING Script/`)
```
Raw Sensor Data (pose keypoints OR insole CSV)
  → Load & Calibrate
  → Gait Event Detection (heel strike, toe off)
  → Cycle Segmentation (HS→HS full gait cycle)
  → Quality Control (coverage, gaps, stability gates)
  → Angle Enhancement (PCHIP interpolation, geometric recalc)
  → Cyclogram Generation (2D/3D angle-angle diagrams)
  → Metrics Computation (area, perimeter, curvature, symmetry)
  → Morphological Mean Cyclogram (MMC) - robust median averaging
  → Export (PNG @ 300 DPI + JSON metadata)
```

**Pipeline 2: Statistical Analysis** (`DATA ANALYSIS Script/`)
```
Excel Cyclogram Metrics (left_*, right_* columns)
  → Parse mean±std strings
  → Effect Feature Engineering (delta/rho transforms)
  → Preprocessing (KNN imputation, robust scaling)
  → Multi-Algorithm Clustering (GMM, KMeans, Hierarchical)
  → Bootstrap Validation (Jaccard stability)
  → Consensus Clustering (cross-plane agreement)
  → Statistical Testing (normality, bilateral, group comparisons)
  → Phenotype Interpretation (radar plots, UMAP)
  → Export (Excel tests + publication figures)
```

### Key Architectural Decisions

**Cyclogram Generation**:
- **HS→HS cycles** (not HS→TO stance): Produces naturally closed loops for valid area/hysteresis metrics
- **Index-based L-R pairing** (L₁↔R₁, not mid-time): Phase-true comparison prevents spurious asymmetry
- **PCHIP interpolation** (not cubic splines): Shape-preserving, no overshoots in biomechanical data
- **No forced closure**: Loops close naturally; closure error is a quality metric
- **Adaptive quality gates**: Thresholds relax for poor-quality data based on MAD assessment

**Statistical Analysis**:
- **Effect features** (delta/rho): Transform left/right to injured vs. control comparisons
  - `delta_X = injured_X - control_X` (bounded metrics)
  - `rho_X = log(injured_X / control_X)` (unbounded positive metrics)
- **Robust statistics**: MAD-based preprocessing, winsorization for outliers
- **FDR correction**: Benjamini-Hochberg for all p-values
- **Multi-plane consensus**: Hierarchical clustering on co-association matrix

## Critical Data Structures

### Pose Analysis (Pose-Analysis.py)

**StrideWindow** (line 732): Represents gait cycle (HS→HS)
- Timing, duration, frame indices, quality metrics (coverage, gaps, stability)

**CyclogramLoop** (line 747): Single angle-angle trajectory
- Proximal/distal angles resampled to 101 points (0-100% gait cycle)
- Closure error, NaN%, quality flags

**CyclogramMetrics** (line 874): Paired L-R comparison
- Area, Procrustes, RMSE, DTW, orientation, hysteresis

**AnalysisConfig** (line 905): Central configuration
- Processing: smoothing, FPS, resampling
- Segmentation: stride duration constraints
- Quality gates: coverage, gaps, stability, sanity thresholds

**DataQualityMetrics** (line 1095): MAD-based quality assessment
- Coverage, gap analysis, temporal consistency, signal quality
- Composite quality score (0-100) for adaptive threshold calibration

**PlotConfig** (line 964): Publication mode configuration
- Sizing modes: 'publication' (default 10×8" @ 300 DPI), 'adaptive', 'fixed'
- Adaptive thresholds calibrated from data using MAD

### Insole Analysis (insole-analysis.py)

**InsoleConfig** (line 76): Central configuration
- Signal processing: sampling rate, filter settings
- Gait constraints: min/max cycle duration (relaxed for elderly/pathological)
- Validation: stance/swing ratio bounds, bilateral tolerance

**GaitPhase** (line 109): Single gait phase (1-8)
- IC, LR, MSt, TSt, PSw, ISw, MSw, TSw
- Support type: double_support, single_support, swing

**CyclogramData** (line 121): IMU cyclogram
- Sensor signals (ACC_X, ACC_Y, ACC_Z, GYRO_*)
- Phase boundaries for colored visualization

**MorphologicalMeanCyclogram**: Robust median trajectory
- DTW-based median reference selection
- Variance envelope (±SD) for variability visualization
- Shape Dispersion Index (SDI) for consistency quantification

### Statistical Analysis Data Structures

**Effect Features** (injured_vs_contralateral_cluster_analysis.py:169-302):
- `delta_*`: Difference for bounded metrics (compactness, symmetry)
- `rho_*`: Log-ratio for unbounded metrics (area, curvature)
- `circular_*`: Angular difference for phase metrics (handles wraparound)

**Feature Families** (line 60-70):
- `sway_envelope`: area, perimeter, compactness, closure_error
- `smoothness`: trajectory_smoothness
- `curvature`: mean_curvature, curvature_std, curv_phase_*
- `timing_sync`: relative_phase, MARP, coupling_angle_variability
- `asymmetry`: bilateral_symmetry_index, bilateral_mirror_correlation

## Critical Functions

### Pose Analysis

**enhance_angles_from_keypoints()** (line 238): Multi-tier angle recovery
- Tier 1: PCHIP interpolation for small gaps
- Tier 2: Geometric recalculation from MediaPipe landmarks (hip=23/24, knee=25/26, ankle=27/28)
- Tier 3: Temporal smoothing with jump limiting
- 8× improvement in valid pairs for incomplete data

**calibrate_filter_parameters()** (line 1641): FFT + MAD adaptive calibration
- FFT analysis: Dominant frequency detection per angle signal
- MAD variability: Noise measurement using Median Absolute Deviation
- Adaptive window: `period/4 × fps` constrained to [5, 31]
- Polynomial order: 2 for MAD > 15°, else 3

**auto_calibrate_config()** (line 1722): Data-adaptive parameters
- Returns: `Tuple[AnalysisConfig, DataQualityMetrics]`
- Quality-driven gates: Relaxes thresholds for poor data (score < 50: 50% coverage)
- MAD-based smoothing: `2.5 × MAD × 1.4826`

**pair_strides()** (line 1849): Phase-true L-R matching
- Cycles: Index-based pairing (L_k ↔ R_k) after aligning starts
- Validates temporal overlap (≥30% default)

**compute_morphological_mean()** (line 2175): MMC computation
- DTW-based median reference selection
- Centroid centering and area rescaling
- Median trajectory with variance envelope
- See `claudedocs/MMC_METHODOLOGY.md` for complete details

### Insole Analysis

**detect_heel_strikes_adaptive()** (line 883): Multi-loop optimization
- 5 iterations with dynamic threshold adjustment
- Contact period detection with clean transition validation
- Returns heel strikes with confidence scores (~88% typical)

**detect_gait_sub_phases()** (line ~1100): 8-phase segmentation
- Dynamic detection: IC, LR, MSt, TSt, PSw, ISw, MSw, TSw
- Phase duration validation against biomechanical constraints (Perry's model)

**build_subplot_grid()** (line ~2301): Automated figure layout
- Auto-detects grid dimensions (2×3, 2×2, 1×3, 1×2)
- Returns matplotlib Figure and axes array

### Statistical Analysis

**parse_mean_std_string()** (cyclogram_publication_analysis.py:91): Extract mean from "mean ± std"
- Handles both formatted strings and plain numeric values

**build_effect_features()** (injured_vs_contralateral_cluster_analysis.py:169): Transform to injured vs. control
- Delta for bounded, rho for unbounded, circular for angles

**perform_clustering()** (line 449): Multi-algorithm clustering
- GMM (BIC/AIC selection), KMeans (20 inits), Hierarchical (Ward linkage)
- Bootstrap stability (100 iterations, Jaccard index)

**consensus_clustering()** (line 663): Cross-plane agreement
- Co-association matrix from plane-specific clusters
- Hierarchical clustering on consensus distances
- Silhouette-based K selection

## Output Files

### Pose Analysis Outputs
```
Output/
├── main_output.csv                    # Aggregate metrics across all subjects
└── Openpose_SUBJECT_ID/
    ├── CK_hip_knee_AllStrides.png     # Hip-knee cyclograms
    ├── CK_knee_ankle_AllStrides.png   # Knee-ankle cyclograms
    ├── CK_hip_ankle_AllStrides.png    # Hip-ankle cyclograms
    ├── LR_Similarity_Summary.png      # Quality-colored symmetry visualization
    ├── cyclogram_stride_metrics.csv   # Per-stride metrics (area, RMSE, DTW, etc.)
    └── cyclogram_session_summary.csv  # Session-level statistics
```

### Insole Analysis Outputs
```
insole-output/SUBJECT_ID/
├── plots/
│   ├── gait_phases/          # Gait event timelines
│   ├── stride_cyclograms/    # Per-stride cyclograms (first 3 cycles)
│   ├── gait_cyclograms/      # Aggregated multi-cycle overlays with MMC
│   └── symmetry/             # Bilateral comparison plots
├── json/                     # Metadata companions for all PNGs
├── precision_gait_events.csv         # High-precision event timing
├── detailed_gait_phases_left.csv     # 8-phase annotations
└── detailed_gait_phases_right.csv
```

### Statistical Analysis Outputs
```
DATA ANALYSIS Script/data_output/RESULTS/
├── CLUSTERS/
│   ├── figs/                         # Cluster visualizations (radar, UMAP, dendrograms)
│   ├── logs/                         # Audit trails
│   ├── plane_memberships.csv         # Per-plane cluster assignments
│   ├── consensus_membership.csv      # Cross-plane consensus
│   └── cluster_centroid_tests.xlsx   # H₀: mean(delta/rho) = 0 tests
├── figs/                             # Statistical plots
│   ├── normality_tests.csv           # Shapiro-Wilk + D'Agostino-Pearson
│   ├── bilateral_tests.csv           # Paired t-test/Wilcoxon (L vs R)
│   ├── group_comparison_tests.csv    # ANOVA/Kruskal-Wallis
│   └── correlations.csv              # Cross-plane vs 3D correlations
└── visualizations/                   # Publication-ready figures
```

## Biomechanical Concepts

**Cyclogram**: Angle-angle diagram showing joint coordination
- X-axis: Proximal joint (e.g., hip)
- Y-axis: Distal joint (e.g., knee)
- Trajectory: Joint coordination during gait cycle
- Naturally closed loop when properly segmented

**Full Gait Cycle (HS→HS)**: Complete locomotion pattern
- Stance phase (~60%): Foot on ground
- Swing phase (~40%): Foot in air
- Used for all cyclogram analysis (not HS→TO stance)

**Quality Gates** (Pose-Analysis.py:562):
1. **Coverage**: ≥70% non-NaN frames per joint
2. **Gaps**: Max contiguous NaN ≤30 frames
3. **Stability**: Pelvis vertical motion std ≤15 pixels
4. **Sanity**: Max frame-to-frame angle change ≤45°

**Morphological Mean Cyclogram (MMC)**:
- Replaces naive mean with phase-aligned morphological median
- Pipeline: Filter valid → Find median ref (DTW) → Center → Rescale → Median shape
- Shape Dispersion Index (SDI): < 0.1 = consistent, 0.1-0.3 = normal, > 0.3 = high variability
- All "mean cyclogram" operations use MMC, never `np.mean()`

## Publication Mode (Default for Pose Analysis)

**Enabled by default** - generates uniform publication-grade plots:
- **Uniform dimensions**: All plots 10×8 inches
- **High resolution**: 300 DPI for print quality
- **Consistent styling**: Same fonts, grid, padding
- **Locked axis limits** (optional): Consistent scaling for direct comparison

```bash
# Publication mode (default)
python3 "CYCLOGRAM-PROCESSING Script/Pose-Analysis.py"

# Publication mode with locked axis limits
python3 "CYCLOGRAM-PROCESSING Script/Pose-Analysis.py" --lock-axis-limits

# Adaptive mode (opt-in for smaller file sizes during development)
python3 "CYCLOGRAM-PROCESSING Script/Pose-Analysis.py" --adaptive-mode
```

## GPU Acceleration (Optional)

**CuPy acceleration for MMC computation** (30-40× speedup):
```bash
# Check if GPU is available
python3 -c "from CYCLOGRAM-PROCESSING\ Script.gpu_acceleration import is_gpu_available; print(is_gpu_available())"

# Install CuPy (requires NVIDIA GPU + CUDA)
pip3 install cupy-cuda12x  # For CUDA 12.x
```

**Performance comparison** (20 gait cycles):
- CPU: ~15 seconds
- GPU: ~0.5 seconds

## Common Development Tasks

### Modify Joint Pairs (Pose Analysis)
```python
# In AnalysisConfig (line ~923)
joint_pairs = [
    ("hip", "knee"),     # Thigh coordination
    ("knee", "ankle"),   # Shank coordination
    ("hip", "ankle")     # Full leg coordination
]
```

### Adjust Quality Thresholds
```python
# In AnalysisConfig (line ~948)
min_coverage_pct = 70.0      # Tighten: 80.0, Relax: 60.0
max_gap_frames = 30          # Tighten: 20, Relax: 50
max_angle_jump = 45.0        # Tighten: 30.0, Relax: 60.0
```

### Add New Cyclogram Metrics
1. Define calculation function in "METRICS CALCULATION" section (line ~1960)
2. Add field to `CyclogramMetrics` dataclass (line ~874)
3. Compute in `compute_all_metrics()` (line ~2175)
4. Export in `write_stride_metrics()` (line ~2613)

### Modify Clustering Parameters
```python
# In injured_vs_contralateral_cluster_analysis.py (lines 75-79)
K_RANGE = range(2, 7)              # Test 2-6 clusters
BOOTSTRAP_ITERATIONS = 100         # Stability validation
JACCARD_THRESHOLD = 0.75           # Consensus agreement
SPEARMAN_THRESHOLD = 0.90          # Feature redundancy removal
```

## Critical Warnings

**DO NOT**:
- Force loop closure by setting `last_point = first_point`
- Use stance windows (HS→TO) for area/hysteresis metrics (only for full cycles)
- Pair by mid-time proximity for cycles (use index-based L_k ↔ R_k)
- Apply cubic splines to biomechanical angles (use PCHIP)
- Skip quality gates without understanding impact on metric validity
- Use standard mean for cyclogram averaging (use MMC via `compute_morphological_mean()`)

**MediaPipe Landmark Indices** (Pose-Analysis.py:283):
- Left: hip=23, knee=25, ankle=27, foot=31
- Right: hip=24, knee=26, ankle=28, foot=32
- Pelvis: average of landmarks 23 and 24

**Insole Analysis Warnings**:
- Always use adaptive step detection (line 883), not simple threshold-based
- Gait cyclogram layouts must be 2×3 (Left row 0, Right row 1)
- Use set tracking for gait event legend to avoid duplicate labels
- Phase color scheme: Stance (blue gradient), Swing (red gradient)

## Debugging Tips

**Check Data Quality** (Pose Analysis):
```python
# Quality score printed during analysis
# Score ≥ 75: Excellent (standard gates)
# Score 50-75: Good (relaxed gates)
# Score < 50: Poor (very relaxed gates, consider --enhance-angles)
```

**Pipeline Report** (automatically printed):
- Shows stage counts: events → cycles → QC-passed → paired
- Rejection reasons indicate which quality gate failed
- Pairing efficiency reveals temporal alignment issues

**Common Issues - Pose Analysis**:
- Few pairs: Check event detection, adjust cycle duration constraints
- High rejection rate: Relax quality gates or use angle enhancement
- Poor closure: Likely event detection error or non-cycle segmentation
- NaN in metrics: Empty loops or failed interpolation

**Common Issues - Insole Analysis**:
- Few heel strikes: Check pressure sensor quality, adjust adaptive thresholds
- Subplot figures missing: Verify cyclograms generated, check sensor data columns
- Legend overcrowding: Ensure set tracking for unique phase labels
- Wrong subplot layout: Verify 2×3 grid for gait cyclograms
- "No data" subplots: Check sensor pair exists, verify gait cycles detected

## Performance Notes

**Pose Analysis**:
- Runtime: ~45-75 sec/subject with enhancement, ~30-60 sec without
- Memory: ~200-500 MB/subject
- Bottleneck: DTW computation O(N²) - use `fastdtw` for speedup
- Batch: Sequential processing

**Insole Analysis**:
- Runtime: ~8 sec/subject (GPU), ~23 sec (CPU)
- Memory: ~50 MB/file
- Bottleneck: MMC computation (GPU acceleration recommended)
- Batch: Sequential with `--batch` flag

**Statistical Analysis**:
- Runtime: ~2-5 min for full pipeline (143 patients, 4 planes)
- Memory: ~1 GB peak
- Bottleneck: Bootstrap clustering (100 iterations)

## Key Documentation Files

- `claudedocs/CLAUDE.md` - Comprehensive pose analysis guide
- `claudedocs/cyclogram_system_design.md` - Architecture specification
- `claudedocs/MMC_METHODOLOGY.md` - Morphological Mean Cyclogram methodology
- `claudedocs/PUBLICATION_MODE_IMPLEMENTATION.md` - Publication mode details
- `claudedocs/GPU_ACCELERATION_GUIDE.md` - CuPy GPU setup
- `DATA ANALYSIS Script/CLAUDE.md` - Statistical analysis comprehensive guide
- `claudedocs/insole_subplot_visualization_system.md` - Subplot architecture
