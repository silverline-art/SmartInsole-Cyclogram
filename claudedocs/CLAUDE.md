# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Human gait analysis system using **cyclograms** (angle-angle phase plots) to assess left-right symmetry from **both pose and pressure data**. Analyzes MediaPipe pose estimation and smart insole pressure data to compute biomechanical metrics for clinical gait assessment.

**Core concept**: Cyclograms visualize joint coordination by plotting one joint angle against another throughout the gait cycle (heel strike → heel strike). Left-right asymmetry indicates potential gait pathology.

**Dual analysis approach**:
- **Pose-based cyclograms**: MediaPipe joint angles (hip-knee-ankle coordination)
- **Pressure-based cyclograms**: Smart insole force distribution patterns

## Repository Information

**GitHub**: https://github.com/silverline-art/Step-Cyclogram

**Branch structure**:
- `main`: Stable production branch for unified pose and insole cyclogram analysis
- Feature branches: Use `feature/*` naming convention

## Development Environment

**Python virtual environment**: `.venv/`
```bash
# Activate environment
source .venv/bin/activate

# Run pose-based cyclogram analysis
python3 Code-Script/Pose-Analysis.py --enhance-angles

# Run insole pressure analysis
python3 Code-Script/insole-analysis.py

# Or from Code-Script directory
cd Code-Script
python3 Pose-Analysis.py --enhance-angles
python3 insole-analysis.py
```

**Key dependencies**: numpy, pandas, matplotlib, scipy, sklearn, fastdtw (optional)

**Analysis scripts**:
- `Pose-Analysis.py`: MediaPipe pose-based gait cyclogram analysis
- `insole-analysis.py`: Smart insole pressure cyclogram analysis
- `Pose-Checkpoint.py`: Checkpoint/backup version of pose analysis

## Data Architecture

**Pose data input** (`Sample-Data/`):
```
Sample-Data/
└── Openpose_SUBJECT_ID_DATE_TRIAL/
    ├── Raw_Angles.csv          # Joint angles (may have NaN gaps)
    ├── Angle_Events.csv        # Heel strike/toe-off events
    ├── Clean_keypoints.csv     # MediaPipe landmarks (for enhancement)
    └── calibrated_keypoints.csv
```

**Insole data input** (`insole-sample/`):
```
insole-sample/
└── 10MWT.csv                   # Smart insole pressure data
```

**Output structure**:
```
Output/                         # Pose analysis outputs
├── main_output.csv
└── Openpose_SUBJECT_ID/
    ├── CK_hip_knee_AllStrides.png
    ├── CK_knee_ankle_AllStrides.png
    ├── CK_hip_ankle_AllStrides.png
    ├── LR_Similarity_Summary.png
    ├── cyclogram_stride_metrics.csv
    └── cyclogram_session_summary.csv

insole-output/                  # Insole analysis outputs
└── [pressure cyclogram visualizations and metrics]
```

## Core Pipeline (Pose-Analysis.py)

**Full pipeline flow**:
1. **Load data**: Raw_Angles.csv + Angle_Events.csv
2. **Enhancement** (optional `--enhance-angles`):
   - Tier 1: PCHIP interpolation for small gaps
   - Tier 2: Geometric recalculation from keypoints
   - Tier 3: Temporal smoothing with jump limiting
3. **Auto-calibration**: Adapt parameters to data characteristics
4. **Cycle segmentation**: HS→HS (full gait cycle, not stance)
5. **Quality control**: Multi-gate validation (coverage, gaps, stability, sanity)
6. **Pairing**: Index-based L↔R pairing with phase overlap validation
7. **Metrics**: DTW, Procrustes, RMSE, area, orientation, hysteresis
8. **Visualization**: Side-by-side cyclograms with asymmetry indicators
9. **Export**: CSV metrics + PNG plots

**Critical architectural decisions**:
- **HS→HS cycles** (not HS→TO stance): Produces naturally closed loops for valid area/hysteresis metrics
- **Index-based pairing** (L₁↔R₁, not mid-time proximity): Phase-true comparison prevents spurious asymmetry
- **PCHIP interpolation** (not cubic splines): Shape-preserving, no overshoots in biomechanical data
- **No forced closure**: Loops close naturally; closure error is a quality metric
- **Adaptive quality gates**: Thresholds relax for poor-quality data, skip for imputed data

## Running Analysis

**Pose-based cyclogram analysis**:
```bash
# Basic usage (all subjects) - publication mode by default (10×8" @ 300 DPI)
python3 Code-Script/Pose-Analysis.py

# Single subject (publication mode by default)
python3 Code-Script/Pose-Analysis.py \
  --subject-name "Openpose_조정자_1917321_20240117_1"

# Use adaptive mode for content-aware sizing
python3 Code-Script/Pose-Analysis.py --adaptive-mode

# Custom parameters with locked axis limits
python3 Code-Script/Pose-Analysis.py \
  --smooth-window 15 \
  --smooth-threshold 7.0 \
  --lock-axis-limits \
  --input-dir "/path/to/data" \
  --output-dir "/path/to/results"

# Disable angle enhancement if needed
python3 Code-Script/Pose-Analysis.py --no-enhance-angles
```

**Insole pressure analysis**:
```bash
# Single subject analysis
python3 Code-Script/insole-analysis.py --input insole-sample/10MWT.csv --output insole-output/10MWT

# Batch processing (all CSVs in insole-sample/)
python3 Code-Script/insole-analysis.py --batch

# Custom sampling rate and filter settings
python3 Code-Script/insole-analysis.py --input data.csv --sampling-rate 100 --filter-cutoff 20.0
```

**Insole analysis outputs**:
- **Organized subplot figures**: 7 types of multi-panel visualizations
  - Plot Set 1: Gyroscopic Stride Cyclograms (2×3 grid) ✓
  - Plot Set 2: Accelerometer Stride Cyclograms (2×3 grid) ✓
  - Plot Set 3: 3D Stride Cyclograms (2×2 grid) - Placeholder
  - Plot Set 4: Gyroscopic Gait Cyclograms (2×3 grid) ✓ **Multi-cycle overlay with mean + ±SD**
  - Plot Set 5: Accelerometer Gait Cyclograms (2×3 grid) ✓ **Multi-cycle overlay with mean + ±SD**
  - Plot Set 6: 3D Gait Cyclograms (1×2 grid) - Placeholder
  - Plot Set 7: Gait Event Timeline (1×2 grid) ✓ **Clean legend with unique phase labels**
- **Gait-level enhancements** (Sets 4 & 5):
  - Layout: 2×3 grid (Left leg row 0, Right leg row 1 × X-Y/X-Z/Y-Z columns)
  - All individual gait cycles overlaid (semi-transparent alpha=0.2)
  - Morphological mean cyclogram (MMC) computed via median-based robust averaging
  - Shaded ±SD envelope showing cycle-to-cycle variability
  - Bilateral left-right comparison for direct symmetry assessment
  - Visualizes intra-subject consistency and bilateral symmetry
- **Step detection**: Adaptive multi-loop optimization (5 iterations)
  - Dynamic threshold and prominence calculation
  - Contact period detection with clean transition validation
  - Typical results: 18 heel strikes per leg with ~88% confidence
- **PNG+JSON pairs**: Every visualization has metadata companion
- **Directory structure**: Categorized into `plots/` and `json/` with subcategories:
  - `gait_phases/`: Gait event timelines
  - `stride_cyclograms/`: Stride-level cyclograms
  - `gait_cyclograms/`: Gait-level (aggregated) cyclograms
  - `mean_cyclograms/`: Morphological mean cyclograms
  - `symmetry/`: Bilateral symmetry analyses
- **Individual cyclograms**: First 3 cycles per sensor pair with phase segmentation
- **Summary files**: Gait cycle metrics, symmetry analysis, validation results

See `claudedocs/insole_subplot_visualization_system.md` for complete documentation on the subplot visualization system.

**Angle enhancement** (pose analysis):
- **Enabled by default** - automatic angle recalculation from keypoints
- Handles NaN gaps using PCHIP interpolation and geometric recalculation
- Provides 8x improvement in valid pairs (1-2 → 8+) for incomplete data
- Use `--no-enhance-angles` to disable if raw data quality is already excellent
- Auto-enhancement still applies separately for poor quality data (controlled by `--no-auto-enhance`)

## Key Classes and Data Structures

**StrideWindow** (Pose-Analysis.py:732): Represents a gait cycle (HS→HS) or stance (HS→TO)
- Stores timing, duration, frame indices, window type
- Contains quality metrics (coverage, gaps, stability)

**CyclogramLoop** (Pose-Analysis.py:747): A single angle-angle trajectory
- Proximal/distal angles resampled to 101 points (0-100% gait cycle)
- Tracks closure error, NaN%, quality flags
- Used for all metric calculations

**CyclogramMetrics** (Pose-Analysis.py:874): Paired L-R comparison metrics
- Area metrics: signed area, normalized area, delta area %
- Shape metrics: Procrustes, RMSE, DTW
- Orientation: PCA major axis angle
- Hysteresis: CW/CCW direction

**AnalysisConfig** (Pose-Analysis.py:905): Central configuration
- Processing: smoothing, FPS, resampling
- Segmentation: stride duration constraints
- Quality gates: coverage, gaps, stability, sanity thresholds
- Enhancement: PCHIP confidence, pairing strategy

**PipelineReport** (Pose-Analysis.py:782): Stage-by-stage tracking
- Events → cycles → QC-passed → paired
- Rejection reasons (coverage, gaps, stability, sanity)
- Efficiency metrics (QC pass rate, pairing efficiency)

**PlotConfig** (Pose-Analysis.py:964): Adaptive visualization configuration with publication mode
- **Sizing modes**: 'adaptive' (default), 'fixed', 'publication'
  - Adaptive: Content-aware sizing, screen DPI (150), optimal file sizes
  - Publication: Uniform 10×8" @ 300 DPI, locked axis limits for consistent appearance
- **Adaptive quality thresholds**: Calibrates from data using MAD
  - `calibrate_quality_thresholds()`: Sets excellent/good/acceptable thresholds based on score distribution
  - Excellent: median + 0.5×MAD, Good: median, Acceptable: median - 0.5×MAD
- **Dynamic color assignment**: `get_quality_color()` maps scores to colors using adaptive thresholds
- **Methods**: `get_dpi()`, `calculate_cyclogram_figsize()`, `calculate_similarity_figsize()`
- Figure sizing, color schemes, font sizes, DPI presets
- Quality-based visual encoding (green → yellow → red gradient)

**standard_plot_setup()** (Pose-Analysis.py:1212): Utility function for standardized figures
- Creates uniform figures with consistent styling across all visualizations
- Automatically applies publication mode settings when enabled
- Configures grid, fonts, axis limits, background, and tight layout
- Returns (fig, axes) tuple ready for plotting

**DataQualityMetrics** (Pose-Analysis.py:1095): Comprehensive MAD-based quality assessment
- **Coverage metrics**: Overall, per-joint, per-leg coverage percentages
- **Gap analysis**: Max gap, mean/median gap sizes, gap count
- **Temporal consistency**: Sampling rate median/MAD (robust to outliers)
- **Signal quality**: SNR estimates, jump severity (max/median), jump count
- **Angle variability**: MAD for each joint (replaces standard deviation)
- **Composite quality score**: 0-100 scale combining all metrics
- **Key methods**:
  - `compute_mad()`: Static method for Median Absolute Deviation calculation
  - `compute()`: Generates complete quality report from angles DataFrame
  - `print_report()`: Formatted console output with visual indicators

### Insole Data Structures (insole-analysis.py)

**InsoleConfig** (line 64): Central configuration
- Signal processing: sampling rate, filter cutoff, filter order
- Gait constraints: min/max cycle duration, min stance/swing duration
- Phase detection: pressure thresholds, gyro swing threshold
- Validation: stance/swing ratio bounds, bilateral tolerance
- Visualization: plot DPI, cyclogram resolution

**GaitPhase** (line 94): Single gait phase annotation
- Phase info: name (IC, LR, MSt, etc.), leg, phase number
- Timing: start/end time, start/end index, duration

**GaitCycle** (line 107): Complete gait cycle
- Cycle info: leg, cycle_id, timing
- Contains list of GaitPhase objects
- Stance/swing durations and ratio

**CyclogramData** (line 121): IMU-based cyclogram
- Sensor signals: x, y, z (optional) arrays
- Linked GaitCycle object
- Phase boundaries: indices and labels for coloring
- 2D/3D flag

**GaitEventRecord** (line 136): High-precision gait event
- Event type: heel_strike, mid_stance, toe_off
- Timing: start, end, duration (milliseconds)
- Sensor source and frame indices
- Confidence score (0.0-1.0)

## Critical Functions

### Pose Analysis (Pose-Analysis.py)

**enhance_angles_from_keypoints()** (line 238): Multi-tier angle recovery
- PCHIP interpolation → geometric recalculation → temporal smoothing
- Uses MediaPipe landmarks (hip=23/24, knee=25/26, ankle=27/28, foot=31/32)
- Hip uses atan2 for full 360° range (prevents wrapping issues)

**calibrate_filter_parameters()** (line 1641): FFT + MAD based adaptive filter calibration
- **FFT analysis**: Computes dominant frequency for each angle signal
- **MAD variability**: Measures signal noise using Median Absolute Deviation
- **Adaptive window sizing**: `window = period/4 × fps`, constrained to [5, 31]
- **Polynomial order selection**: poly_order = 2 for high noise (MAD > 15°), else 3
- **Returns**: `(window_length, poly_order)` optimized for signal characteristics
- **Fallback logic**: Uses MAD thresholds when FFT fails or frequencies invalid

**build_cycle_windows()** (line 1538): Extract HS→HS cycles with QC
- Pairs consecutive heel strikes (same leg)
- Applies quality gates if enabled
- Returns valid windows + rejection statistics

**pair_strides()** (line 1849): Phase-true L-R matching
- **Cycles**: Index-based pairing (L_k ↔ R_k) after aligning starts
- **Stance**: Mid-time proximity (fallback)
- Validates temporal overlap (default ≥30%)

**compute_all_metrics()** (line 2175): Complete metric suite
- Area/hysteresis only for closed cycles (is_closed=True)
- DTW captures timing asymmetry (uses fastdtw if available)
- Normalized area = area / (π·σₓ·σᵧ) for scale-free comparison

**auto_calibrate_config()** (line 1722): Enhanced data-adaptive parameters with MAD
- **Returns**: `Tuple[AnalysisConfig, DataQualityMetrics]` (quality metrics + calibrated config)
- **Data quality assessment**: Computes comprehensive quality score (0-100) using MAD
- **Adaptive filter calibration**: FFT + MAD analysis determines optimal Savitzky-Golay parameters
- **Quality-driven thresholds**: Relaxes gates for poor-quality data (score < 50: 50% coverage, score < 75: 60% coverage)
- **MAD-based smoothing**: Threshold = 2.5 × MAD × 1.4826 (robust to outliers)
- Cycle duration from percentiles (10th-90th)
- Pairing tolerance from coefficient of variation

**compute_morphological_mean()** (line 2175): MMC computation
- DTW-based median reference selection
- Centroid centering and area rescaling
- Median trajectory with variance envelope
- Returns MorphologicalMeanCyclogram or None

### Insole Analysis (insole-analysis.py)

**detect_heel_strikes_adaptive()** (line 883): Adaptive step detection
- Multi-loop optimization (5 iterations)
- Dynamic threshold: `base + loop * 0.1 * std(pressure)`
- Contact period detection with clean transition validation
- Returns heel strikes with confidence scores

**detect_gait_sub_phases()** (line ~1100): 8-phase segmentation
- Dynamic detection: IC, LR, MSt, TSt, PSw, ISw, MSw, TSw
- Phase duration validation against biomechanical constraints
- Returns GaitPhase objects with timing and indices

**build_subplot_grid()** (line ~2301): Automated figure layout
- Auto-detects grid dimensions from analysis type
- Returns matplotlib Figure and axes array
- Handles 2×3, 2×2, 1×3, 1×2 layouts

**create_and_populate_subplot_figure()** (line ~2789): Complete workflow
- Creates grid → populates subplots → generates metadata
- Returns figure, metadata, base_name for saving
- Automatic categorization and directory routing

**_generate_subplot_figures()** (line ~3131): Pipeline integration
- Organizes cyclograms by sensor type and leg
- Maps sensor pairs to subplot labels
- Generates and saves all subplot figures
- Reports generation status

## Biomechanical Concepts

**Cyclogram**: Angle-angle diagram showing joint coordination
- X-axis: Proximal joint (e.g., hip)
- Y-axis: Distal joint (e.g., knee)
- Trajectory shows how joints coordinate during gait cycle

**Full gait cycle** (HS→HS): Complete locomotion pattern
- Stance phase (~60%): foot on ground
- Swing phase (~40%): foot in air
- Naturally closed loop when properly segmented

**Quality gates** (Pose-Analysis.py:562):
1. **Coverage**: ≥70% non-NaN frames per joint
2. **Gaps**: Max contiguous NaN ≤30 frames
3. **Stability**: Pelvis vertical motion std ≤15 pixels
4. **Sanity**: Max frame-to-frame angle change ≤45°

**Symmetry metrics**:
- **Procrustes**: Shape similarity after optimal alignment
- **RMSE**: Amplitude-normalized pointwise difference
- **DTW**: Timing/phase mismatch (critical for cadence asymmetry)
- **Area**: Loop size (indicates range of motion)
- **Orientation**: PCA major axis (indicates tilt/bias)

**Morphological Mean Cyclogram (MMC)** (Pose-Analysis.py:2175):
- Replaces naive mean averaging with phase-aligned morphological median
- **Pipeline**: Filter valid loops → Find median reference (DTW) → Center to centroid → Rescale to median area → Compute median shape → Variance envelope
- **Shape Dispersion Index (SDI)**: Quantifies stride-to-stride variability (< 0.1 = consistent, 0.1-0.3 = normal, > 0.3 = high variability)
- **Benefits**: Robust to outliers, preserves gait shape, visualizes variability via ±SD envelope
- **Use**: All "mean cyclogram" references should use MMC, not `np.mean()`
- See `claudedocs/MMC_METHODOLOGY.md` for complete details

## Common Development Tasks

**Modify joint pairs**:
```python
# In AnalysisConfig (line ~923)
joint_pairs = [
    ("hip", "knee"),     # Thigh coordination
    ("knee", "ankle"),   # Shank coordination
    ("hip", "ankle")     # Full leg coordination
]
```

**Adjust quality thresholds**:
```python
# In AnalysisConfig (line ~948)
min_coverage_pct = 70.0      # Tighten: 80.0, Relax: 60.0
max_gap_frames = 30          # Tighten: 20, Relax: 50
max_angle_jump = 45.0        # Tighten: 30.0, Relax: 60.0
```

**Change metric weights**:
```python
# In AnalysisConfig (line ~930)
similarity_weights = {
    "area": 0.30,           # Loop size
    "procrustes": 0.30,     # Shape similarity
    "rmse": 0.30,           # Amplitude difference
    "orientation": 0.10     # Tilt/orientation
}
```

**Add new metrics**:
1. Define calculation function in "METRICS CALCULATION" section (line ~1960)
2. Add field to `CyclogramMetrics` dataclass (line ~874)
3. Compute in `compute_all_metrics()` (line ~2175)
4. Export in `write_stride_metrics()` (line ~2613)

## Design Documents

**Documentation location**: `claudedocs/`
- `insole_subplot_visualization_system.md`: Complete subplot visualization architecture
- `MMC_METHODOLOGY.md`: Morphological Mean Cyclogram implementation details
- `angle_calculation_design.md`: Joint angle calculation methodology
- `cyclogram_system_design.md`: Cyclogram generation system architecture
- `ROOT_CAUSE_ANALYSIS.md`: Major bug investigations and resolutions
- `CRITICAL_IMPROVEMENTS_ANALYSIS.md`: Performance and quality improvements
- `IMPLEMENTATION_STATUS.md`: Feature implementation status tracking
- `POSE_ANALYSIS_IMPROVEMENT_PLAN.md`: Comprehensive improvement plan for data handling and visualization
- `POSE_DETECTION_ACTUAL_ROOT_CAUSE.md`: Root cause analysis for quality gate rejection issues
- `POSE_ANALYSIS_DETECTION_FAILURE_ROOT_CAUSE.md`: Data type mismatch investigation

**Important context** (`request update.txt`):
- Contains biomechanical rationale for cycle-based analysis
- Explains why HS→HS (not HS→TO) is critical
- Details proper pairing strategies (index vs mid-time)
- Justifies PCHIP over cubic interpolation

**Recent fixes** (`TROUBLESHOOTING_SUMMARY.md`):
- Legend overcrowding in gait events timeline (line 2656-2671)
- Cyclogram subplot layout from 1×3 to 2×3 (lines 2773-2917)
- Adaptive step detection algorithm (lines 883-969)

## Recent Improvements (2025-11-04)

### Robust Statistics with MAD (Median Absolute Deviation)

**Motivation**: Replace standard deviation with MAD for outlier-resistant data quality assessment and calibration.

**Implementation**:
1. **DataQualityMetrics class** (line 1095):
   - `compute_mad()`: Static method for MAD calculation
   - MAD-based temporal consistency: `sampling_rate_mad` instead of std
   - MAD-based variability: `angle_variability_mad` for each joint
   - Robust quality scoring using MAD-derived thresholds

2. **PlotConfig adaptive thresholds** (line 1052):
   - `calibrate_quality_thresholds()`: Uses MAD to set excellent/good/acceptable bounds
   - Formula: `excellent = median + 0.5×MAD×1.4826`, `good = median`, `acceptable = median - 0.5×MAD×1.4826`
   - Scale factor 1.4826 matches std dev for normal distributions

3. **Adaptive filter calibration** (line 1641):
   - `calibrate_filter_parameters()`: FFT + MAD analysis
   - FFT determines dominant signal frequency
   - MAD measures noise level for polynomial order selection
   - Window sizing: `period/4 × fps`, polynomial: 2 for MAD > 15°, else 3

4. **Enhanced auto_calibrate_config()** (line 1722):
   - Now returns `Tuple[AnalysisConfig, DataQualityMetrics]`
   - Computes comprehensive quality metrics with MAD
   - Adaptive quality gates based on quality score:
     - Score < 50: coverage≥50%, max_gap=50
     - Score < 75: coverage≥60%, max_gap=40
     - Score ≥ 75: coverage≥70%, max_gap=30 (standard)
   - MAD-based smoothing threshold: `2.5 × MAD × 1.4826`

**Benefits**:
- Robust to outliers in biomechanical data
- Adaptive thresholds prevent spurious rejections
- Data-driven calibration improves quality control
- Better handling of noisy or incomplete data

**Usage**:
```bash
# Enhanced auto-calibration now automatic
python3 Pose-Analysis.py --enhance-angles

# Quality report shows MAD-based metrics
# Output includes:
# - Temporal consistency: median ± MAD (not mean ± std)
# - Angle variability: MAD per joint
# - Adaptive filter: window/poly based on FFT + MAD
```

**Validation**: Tested with sample data showing 89.4/100 quality score with adaptive filter window=15, poly_order=2 based on MAD=15.61°.

### Publication Mode for Standardized Visualizations

**Motivation**: Generate uniform, publication-grade plots with consistent dimensions, DPI, fonts, and axis limits across all visualizations.

**Problem Solved**: Previous versions generated plots with inconsistent sizes (varying aspect ratios), DPI settings, and margins, making them unsuitable for publications and reports.

**Implementation**:

1. **PlotConfig publication mode** (line 964):
   ```python
   # Publication mode settings (uniform dimensions for all plots)
   publication_figsize: Tuple[float, float] = (10, 8)  # Uniform size for publications
   publication_dpi: int = 300  # High-resolution for print
   lock_axis_limits: bool = False  # Enforce consistent axis ranges
   sizing_mode: str = 'adaptive'  # 'fixed', 'adaptive', 'publication'
   ```

2. **standard_plot_setup() utility function** (line 1212):
   - Creates standardized figures with uniform styling
   - Automatically applies publication settings when enabled
   - Configures grid, fonts, axis limits, and layout
   ```python
   fig, axes = standard_plot_setup(plot_cfg, "Gait Cyclograms", nrows=2, ncols=3)
   ```

3. **Adaptive sizing modes**:
   - **'adaptive'** (default): Content-aware sizing with logarithmic scaling, screen DPI (150)
   - **'fixed'**: Legacy fixed sizes
   - **'publication'**: Uniform dimensions (10×8"), high DPI (300), locked axis limits

4. **Global plot configuration** (main function, line 4165):
   - Created from CLI arguments
   - Passed to all analysis functions
   - Ensures consistency across all visualizations

**Benefits**:
- **Uniform dimensions**: All plots have identical size (10×8 inches)
- **High resolution**: 300 DPI for print-quality output
- **Consistent styling**: Same fonts, grid, padding across all plots
- **Locked axis limits**: Optional consistent scaling for easy comparison
- **Professional appearance**: Publication-ready outputs without manual editing
- **95% file size reduction**: Adaptive mode (153-239 KB vs 8 MB for fixed 16×8" @ 300 DPI)

**Usage**:

```bash
# Publication mode (DEFAULT): uniform 10×8" @ 300 DPI
python3 Pose-Analysis.py

# Publication mode with locked axis limits (0-100%, -10-160°)
python3 Pose-Analysis.py --lock-axis-limits

# Adaptive mode for content-aware sizing @ 150 DPI (if needed)
python3 Pose-Analysis.py --adaptive-mode

# Single subject (publication mode by default)
python3 Pose-Analysis.py \
  --subject-name "Openpose_조정자_1917321_20240117_1" \
  --lock-axis-limits
```

**When to use each mode**:
- **publication** (DEFAULT): Manuscripts, reports, presentations - uniform appearance and professional outputs
- **adaptive** (opt-in with `--adaptive-mode`): Development, exploration, varied content - optimal file sizes
- **lock-axis-limits**: Direct visual comparison, consistent scaling across subjects (use with publication mode)

**Plot generation flow**:
```
CLI args → GLOBAL_PLOT_CONFIG → analyze_subject() → plot_overlayed_cyclograms()
                                                   → plot_similarity_summary()
```

**Technical details**:
- `get_dpi()`: Returns 300 in publication mode, otherwise uses preset mapping (web=96, screen=150, print=300)
- `calculate_cyclogram_figsize()`: Returns publication_figsize in publication mode, otherwise adaptive/fixed
- `calculate_similarity_figsize()`: Same behavior for similarity plots
- `standard_plot_setup()`: Utility for creating figures with automatic styling and layout

**Validation**: Generates uniform plots across all subjects with identical dimensions and styling, suitable for direct inclusion in publications.

## Critical Warnings

**DO NOT**:
- Force loop closure by setting `last_point = first_point` (line ~1828 shows proper approach)
- Use stance windows (HS→TO) for area/hysteresis metrics (only for cycles)
- Pair by mid-time proximity for cycles (use index-based, line ~1870)
- Apply cubic splines to biomechanical angles (use PCHIP, line ~1816)
- Skip quality gates without understanding impact on metric validity

**MediaPipe landmark indices** (Pose-Analysis.py:283):
- Left: hip=23, knee=25, ankle=27, foot=31
- Right: hip=24, knee=26, ankle=28, foot=32
- Pelvis: average of landmarks 23 and 24

**Insole analysis warnings** (insole-analysis.py):
- Always use adaptive step detection (line 883), not simple threshold-based
- Gait cyclogram layouts must be 2×3 (Left row 0, Right row 1), not 1×3 overlay
- Use set tracking for gait event legend to avoid 400+ duplicate labels (line 2656)
- Ensure metadata flags `_has_gyro` and `_has_acc` are set for conditional plot generation (line 3568)
- Phase color scheme: Stance (blue gradient), Swing (red gradient)

## Debugging Tips

**Check data quality**:
```python
# Data quality score printed during analysis
# 1.0 = perfect, 0.5 = moderate, 0.0 = poor
# Score < 0.5 → consider --enhance-angles
```

**Pipeline report** (printed automatically):
- Shows stage-by-stage counts: events → cycles → QC-passed → paired
- Rejection reasons indicate which quality gate failed
- Pairing efficiency reveals temporal alignment issues

**Common issues (Pose analysis)**:
- **Few pairs**: Check event detection quality, adjust cycle duration constraints
- **High rejection rate**: Relax quality gates or use `--enhance-angles`
- **Poor closure**: Likely event detection error or non-cycle segmentation
- **NaN in metrics**: Check for empty loops or failed interpolation

**Common issues (Insole analysis)**:
- **Few heel strikes detected**: Check pressure sensor data quality, adjust adaptive thresholds
- **Subplot figures not generated**: Verify cyclograms generated successfully, check sensor data columns
- **Legend overcrowding**: Ensure set tracking implemented for unique phase labels only
- **Wrong subplot layout**: Verify 2×3 grid for gait cyclograms (not 1×3)
- **"No data" subplots**: Check sensor pair exists in input, verify gait cycles detected
- **Phase colors incorrect**: Review phase_indices/phase_labels alignment in cyclogram

## Performance Notes

**Pose Analysis**:
- **Runtime**: ~45-75 seconds per subject with enhancement (~30-60 without)
- **Memory**: ~200-500MB per subject
- **Bottlenecks**: DTW computation (O(N²), use fastdtw for speedup)
- **Batch processing**: Sequential (not parallel)

**Insole Analysis**:
- **Runtime**: ~30-60 seconds per subject (including subplot generation)
- **Memory**: ~200-500MB per subject (50-100MB per subplot figure at 300 DPI)
- **Bottlenecks**: Adaptive step detection (5 iterations), subplot figure generation (~5-10 sec/figure)
- **Optimization**: Close figures after saving with `plt.close(fig)`, reduce DPI to 150 for prototyping
- **Batch processing**: Sequential (use `--batch` flag for multiple CSVs)
