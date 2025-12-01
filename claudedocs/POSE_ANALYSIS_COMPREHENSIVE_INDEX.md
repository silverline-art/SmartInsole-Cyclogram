# Pose-Analysis.py - Comprehensive Technical Index

**Script**: `CYCLOGRAM-PROCESSING Script/Pose-Analysis.py`
**Version**: Updated November 4, 2025
**Lines**: 4065
**Purpose**: MediaPipe pose-based gait cyclogram analysis with bilateral L-R symmetry assessment

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Data Structures](#data-structures)
4. [Function Reference](#function-reference)
5. [Processing Workflows](#processing-workflows)
6. [Configuration Guide](#configuration-guide)
7. [Recent Improvements (Nov 4, 2025)](#recent-improvements)
8. [Usage Examples](#usage-examples)

---

## Overview

### Purpose

Analyzes MediaPipe pose estimation data to generate cyclograms (angle-angle phase plots) for gait symmetry assessment. Processes joint angle time series data to extract heel-strike-to-heel-strike (HS→HS) gait cycles and computes left-right comparison metrics.

### Key Features

- **Multi-tier angle enhancement** (PCHIP + geometric recalculation + smoothing)
- **Adaptive quality gates** (MAD-based robust statistics)
- **Auto-enhancement** (triggers automatically for poor quality data)
- **Adaptive visualization** (content-aware plot sizing, screen-optimized DPI)
- **Comprehensive metrics** (DTW, Procrustes, RMSE, area, orientation, hysteresis)
- **FFT + MAD calibration** (data-driven filter parameter optimization)

### Recent Updates (November 4, 2025)

1. **MAD-Based Robust Statistics** - Replaced standard deviation with Median Absolute Deviation
2. **Adaptive Plot Sizing** - Content-aware figure dimensions, 95% file size reduction
3. **Auto-Enhancement** - Intelligent quality detection with automatic imputation
4. **FFT+MAD Filter Calibration** - Frequency-domain + robust variability analysis

---

## Architecture

### Data Flow Pipeline

```
Input Files
├─ Raw_Angles.csv (joint angles, may have NaN gaps)
├─ Angle_Events.csv (heel strike, toe-off events)
└─ Clean_keypoints.csv (MediaPipe landmarks, for enhancement)
    │
    ├─> [1] LOAD & AUTO-ENHANCE CHECK
    │   ├─ Load CSVs
    │   ├─ Quick quality check (coverage %)
    │   └─ Auto-enable enhancement if coverage < 70%
    │
    ├─> [2] ENHANCEMENT (if enabled)
    │   ├─ Tier 1: PCHIP interpolation (small gaps)
    │   ├─ Tier 2: Geometric recalculation (from keypoints)
    │   └─ Tier 3: Temporal smoothing (jump limiting)
    │
    ├─> [3] COLUMN DETECTION
    │   └─ Regex patterns: hip/knee/ankle + flex/dorsi + L/R + deg
    │
    ├─> [4] TIMEBASE NORMALIZATION
    │   └─ Align angles_df and events_df time indices
    │
    ├─> [5] AUTO-CALIBRATION (NEW: FFT+MAD)
    │   ├─ Data quality assessment (MAD-based)
    │   ├─ FFT + MAD filter parameter calibration
    │   ├─ Adaptive quality gate thresholds
    │   ├─ MAD-based smoothing threshold
    │   └─ Stride duration + pairing tolerance calibration
    │
    ├─> [6] PREPROCESSING
    │   └─ Savitzky-Golay smoothing (adaptive window+poly)
    │
    ├─> [7] CYCLE SEGMENTATION
    │   ├─ HS→HS cycle extraction (both legs)
    │   └─ Quality gate filtering (coverage, gaps, stability, sanity)
    │
    ├─> [8] CYCLOGRAM EXTRACTION
    │   ├─ Resample to 101 points (0-100% gait cycle)
    │   └─ Create CyclogramLoop objects per joint pair
    │
    ├─> [9] L-R PAIRING
    │   ├─ Index-based pairing (L₁↔R₁, phase-true)
    │   └─ Validate temporal overlap (≥30%)
    │
    ├─> [10] METRICS COMPUTATION
    │   ├─ Shape: Procrustes, RMSE, DTW
    │   ├─ Area: Signed area, normalized area, delta %
    │   ├─ Orientation: PCA major axis angle
    │   └─ Hysteresis: CW/CCW direction
    │
    ├─> [11] VISUALIZATION (NEW: Adaptive sizing)
    │   ├─ Cyclogram overlays (L+R side-by-side, adaptive DPI)
    │   └─ Similarity summary (bar chart)
    │
    └─> [12] EXPORT
        ├─ cyclogram_stride_metrics.csv (per-stride metrics)
        └─ cyclogram_session_summary.csv (per-pair summary)
```

### Quality Gate System

**Stage 1: Cycle Extraction** → **Stage 2: Quality Gates** → **Stage 3: Valid Cycles**

Quality gates (line 562-688):
1. **Coverage**: ≥70% non-NaN frames per joint (adaptive: 50% for poor data)
2. **Gaps**: Max contiguous NaN ≤30 frames (adaptive: 50 for poor data)
3. **Stability**: Pelvis vertical std ≤15 pixels
4. **Sanity**: Max frame-to-frame angle change ≤45°

Adaptive thresholds based on DataQualityMetrics score (line 1812-1843).

---

## Data Structures

### 1. WindowQualityMetrics (Line 690-729)

**Purpose**: Quality assessment results for a single gait cycle window

**Fields**:
```python
coverage_pct: Dict[str, float]     # % non-NaN per angle (e.g., {'hip': 98.5, 'knee': 95.0})
max_gap_frames: int                # Longest contiguous NaN gap
pelvis_stability: Optional[float]  # Pelvis_y std dev (pixels), None if no keypoints
max_angle_jumps: Dict[str, float]  # Max frame-to-frame change per angle
passes_coverage: bool              # Coverage ≥ threshold
passes_gaps: bool                  # Max gap ≤ threshold
passes_stability: bool             # Pelvis std ≤ threshold (or True if N/A)
passes_sanity: bool                # All angle jumps ≤ threshold
overall_pass: bool                 # All gates passed
```

**Usage**: Returned by `compute_window_quality()` (line 562)

---

### 2. StrideWindow (Line 731-744)

**Purpose**: Represents a single gait cycle (HS→HS) or stance phase (HS→TO)

**Fields**:
```python
leg: str                           # 'L' or 'R'
start_time: float                  # Start timestamp (seconds)
end_time: float                    # End timestamp (seconds)
start_idx: int                     # Start frame index
end_idx: int                       # End frame index
duration: float                    # end_time - start_time
window_type: str                   # 'cycle' (HS→HS) or 'stance' (HS→TO)
quality: Optional[WindowQualityMetrics]  # Quality assessment results
```

**Usage**: Extracted by `build_cycle_windows()` (line 1993)

---

### 3. CyclogramLoop (Line 746-770)

**Purpose**: A single angle-angle cyclogram trajectory (one joint pair, one stride)

**Fields**:
```python
joint_pair: Tuple[str, str]        # e.g., ('hip', 'knee')
leg: str                           # 'L' or 'R'
stride_id: int                     # Stride number (0-indexed)
proximal: np.ndarray               # Proximal joint angle (101 points, 0-100% gait)
distal: np.ndarray                 # Distal joint angle (101 points)
time_pct: np.ndarray               # Normalized time 0-100%
is_closed: bool                    # Loop closure check (start ≈ end)
closure_error: float               # Euclidean distance between first and last points
nan_pct: float                     # % NaN points (0-100)
```

**Computed Properties**:
- `is_closed`: `True` if `closure_error < 10` (degrees)
- Used for area/hysteresis metrics (only valid for closed loops)

**Usage**: Created by `extract_cyclogram()` (line 2279)

---

### 4. PairedStrides (Line 772-779)

**Purpose**: Paired left-right stride windows for bilateral comparison

**Fields**:
```python
left: StrideWindow                 # Left leg stride
right: StrideWindow                # Right leg stride
time_overlap_pct: float            # Temporal overlap (0-100%)
```

**Usage**: Created by `pair_strides()` (line 2389)

---

### 5. PipelineReport (Line 781-871)

**Purpose**: Stage-by-stage tracking of cycle extraction and quality control

**Fields**:
```python
# Event counts
events_L: int                      # Left heel strikes + toe offs
events_R: int

# Cycle extraction
cycles_L: int                      # Left HS→HS cycles extracted
cycles_R: int

# Quality control results
qc_passed_L: int                   # Left cycles passing quality gates
qc_passed_R: int

# Pairing results
paired_L: int                      # Left cycles successfully paired
paired_R: int

# Rejection tracking
rejection_reasons: Dict[str, Dict[str, int]]  # {leg: {reason: count}}
# Example: {'L': {'coverage': 3, 'gaps': 1}, 'R': {'coverage': 5}}
```

**Computed Metrics**:
```python
def qc_pass_rate_L() -> float     # qc_passed_L / cycles_L * 100
def qc_pass_rate_R() -> float
def pairing_efficiency_L() -> float  # paired_L / qc_passed_L * 100
def pairing_efficiency_R() -> float
def overall_efficiency_L() -> float  # paired_L / events_L * 100
def overall_efficiency_R() -> float
```

**Usage**: Accumulates statistics throughout `analyze_subject()` (line 3576)

---

### 6. CyclogramMetrics (Line 873-902)

**Purpose**: Bilateral L-R comparison metrics for a paired stride

**Fields**:
```python
joint_pair: Tuple[str, str]        # e.g., ('hip', 'knee')
stride_id_L: int
stride_id_R: int

# Area metrics (only for closed loops)
area_L: float                      # Signed area (deg²)
area_R: float
delta_area_pct: float              # (area_L - area_R) / mean * 100
normalized_area_L: float           # area / (π·σₓ·σᵧ)
normalized_area_R: float

# Shape similarity metrics
rmse: float                        # Root mean squared error (amplitude-normalized)
procrustes_distance: float         # Shape similarity after optimal alignment
dtw_distance: float                # Dynamic time warping (timing asymmetry)

# Orientation metrics
orientation_L: float               # PCA major axis angle (degrees)
orientation_R: float
delta_orientation: float           # abs(orientation_L - orientation_R)

# Hysteresis
hysteresis_L: str                  # 'CW' or 'CCW'
hysteresis_R: str
hysteresis_mismatch: bool          # True if L != R

# Composite similarity score
similarity_score: float            # Weighted combo (0-100), higher = more symmetric
```

**Usage**: Computed by `compute_all_metrics()` (line 2901)

---

### 7. AnalysisConfig (Line 904-960)

**Purpose**: Central configuration for all processing parameters

**Key Fields** (selected):
```python
# Timing
fps: Optional[float] = None        # Auto-detect from timestamp

# Smoothing (Savitzky-Golay filter)
smooth_window: int = 11            # Window size (must be odd)
smooth_poly: int = 2               # Polynomial order
smooth_threshold: float = 5.0      # Jump detection threshold (degrees)

# Stride constraints
min_stride_duration: float = 0.3   # Min cycle duration (seconds)
max_stride_duration: float = 3.0   # Max cycle duration

# Pairing
pairing_tolerance: float = 0.15    # Max duration ratio difference (15%)
use_index_pairing: bool = True     # Index-based (Lₖ↔Rₖ) vs mid-time proximity
min_phase_overlap: float = 0.3     # Min temporal overlap (30%)

# Quality gates
min_coverage_pct: float = 70.0     # Min % non-NaN frames per joint
max_gap_frames: int = 30           # Max contiguous NaN gap
max_pelvis_std: float = 15.0       # Max pelvis vertical motion (pixels)
max_angle_jump: float = 45.0       # Max frame-to-frame angle change (degrees)

# Enhancement (NEW - Nov 4)
enhance_angles: bool = False       # Enable keypoint-based recalculation
auto_enhance: bool = True          # Auto-enable for poor quality data
auto_enhance_threshold: float = 70.0  # Coverage % threshold for auto-enhancement

# Visualization (DEPRECATED)
plot_dpi: int = 300                # Deprecated - PlotConfig handles DPI now

# Joint pairs
joint_pairs: List[Tuple[str, str]] = [
    ("hip", "knee"),
    ("knee", "ankle"),
    ("hip", "ankle")
]

# Similarity weights
similarity_weights: Dict[str, float] = {
    "area": 0.30,
    "procrustes": 0.30,
    "rmse": 0.30,
    "orientation": 0.10
}
```

**Usage**: Created in `main()` (line 3995), passed to all processing functions

---

### 8. PlotConfig (Line 964-1182)

**Purpose**: Adaptive visualization configuration (NEW - Nov 4, 2025)

**Key Fields**:
```python
# Sizing strategy (NEW)
sizing_mode: str = 'adaptive'      # 'fixed' or 'adaptive'
dpi_preset: str = 'screen'         # 'web' (96), 'screen' (150), 'print' (300)

# Base sizes for adaptive scaling (NEW)
base_cyclogram_figsize: Tuple[float, float] = (8, 6)   # Reference size
base_similarity_figsize: Tuple[float, float] = (10, 6)

# Size constraints (NEW)
min_figsize: Tuple[float, float] = (6, 4)
max_figsize: Tuple[float, float] = (20, 15)

# Legacy fixed sizes (used only when sizing_mode='fixed')
cyclogram_figsize: Tuple[float, float] = (16, 8)
similarity_figsize: Tuple[float, float] = (12, 7)
dpi: int = 150                     # Default screen quality (was 300)

# Font hierarchy
title_fontsize: int = 16
subtitle_fontsize: int = 14
label_fontsize: int = 13
tick_fontsize: int = 11

# Colors
left_color: str = '#1f77b4'        # Blue
right_color: str = '#d62728'       # Red
excellent_color: str = '#2ca02c'   # Green
good_color: str = '#ff7f0e'        # Orange
warning_color: str = '#d62728'     # Red

# Line styling
mean_linewidth: float = 3.5
individual_linewidth: float = 1.0

# Quality thresholds (adaptive - calibrated via MAD)
excellent_threshold: float = 90.0
good_threshold: float = 75.0
acceptable_threshold: float        # Set by calibrate_quality_thresholds()
```

**Key Methods** (NEW):
```python
def get_dpi() -> int:
    """Get DPI from preset (96/150/300) or direct value."""

def calculate_cyclogram_figsize(num_cycles: int) -> Tuple[float, float]:
    """Adaptive sizing based on number of cycles (logarithmic scaling)."""

def calculate_similarity_figsize(num_metrics: int) -> Tuple[float, float]:
    """Adaptive sizing based on number of metrics."""

def calibrate_quality_thresholds(quality_scores: List[float]) -> None:
    """Set thresholds using MAD: excellent = median + 0.5×MAD×1.4826."""

def get_quality_color(score: float) -> str:
    """Map score to color using adaptive thresholds."""
```

**Usage**: Created in plot functions (line 3844, 3853)

---

### 9. DataQualityMetrics (Line 1185-1384)

**Purpose**: Comprehensive MAD-based data quality assessment (NEW - Nov 4, 2025)

**Fields**:
```python
# Coverage metrics
overall_coverage_pct: float        # Overall % non-NaN (all angles)
per_joint_coverage: Dict[str, float]  # e.g., {'hip_flex_L_deg': 98.5}
per_leg_coverage: Dict[str, float]    # e.g., {'L': 95.0, 'R': 92.0}

# Gap analysis
max_gap_frames: int                # Largest contiguous NaN gap
mean_gap_size: float
median_gap_size: float             # Robust center
gap_count: int

# Temporal consistency (MAD-based - ROBUST)
sampling_rate_median: float        # Median frame interval (seconds)
sampling_rate_mad: float           # MAD of frame intervals (robust spread)
temporal_completeness: float       # % frames with expected timing

# Signal quality
snr_estimates: Dict[str, float]    # Signal-to-noise ratio per angle (dB)
jump_severity_max: float           # Max frame-to-frame change (degrees)
jump_severity_median: float        # Median jump severity (robust)
jump_count: int                    # Number of jumps exceeding threshold

# Angle variability (MAD - ROBUST)
angle_variability_mad: Dict[str, float]  # MAD per angle (replaces std)

# Composite quality score
quality_score: float               # 0-100, weighted composite
```

**Key Methods**:
```python
@staticmethod
def compute_mad(data: np.ndarray) -> float:
    """Compute Median Absolute Deviation: median(|x - median(x)|)."""

@staticmethod
def compute(angles_df, col_map, config) -> 'DataQualityMetrics':
    """Generate complete quality report from raw angle data."""

def print_report(self) -> None:
    """Formatted console output with visual indicators."""
```

**Quality Score Calculation** (line 1346-1366):
```python
score = (
    coverage_weight * min(overall_coverage_pct, 100) +
    gap_weight * (100 - min(max_gap_frames / max_gap_threshold * 100, 100)) +
    temporal_weight * min(temporal_completeness, 100) +
    snr_weight * min(mean_snr / 30 * 100, 100)
)
```

**Usage**: Called in `auto_calibrate_config()` (line 1836)

---

## Function Reference

### Enhancement Functions

#### `enhance_angles_from_keypoints()` (Line 238-457)

**Purpose**: Multi-tier angle recovery for missing/poor quality data

**Tiers**:
1. **PCHIP Interpolation** (line 269-316): Fill small NaN gaps with shape-preserving interpolation
2. **Geometric Recalculation** (line 318-395): Recalculate angles from MediaPipe keypoints
3. **Temporal Smoothing** (line 397-442): Smooth with jump limiting

**Parameters**:
```python
subject_dir: Path          # Subject directory (for Clean_keypoints.csv)
angles_df: pd.DataFrame    # Raw angles DataFrame
config: AnalysisConfig     # Configuration
```

**Returns**: Enhanced `angles_df` with filled gaps and recalculated angles

**Enhancement Statistics Printed**:
```
L leg: gap_fill=359 (hip=114, knee=114, ankle=131), recalc=120 (hip=40, knee=40, ankle=40)
R leg: gap_fill=489 (hip=154, knee=154, ankle=181), recalc=120 (hip=40, knee=40, ankle=40)
```

---

### Calibration Functions (NEW - Nov 4, 2025)

#### `calibrate_filter_parameters()` (Line 1731-1810)

**Purpose**: FFT + MAD based adaptive Savitzky-Golay filter calibration

**Algorithm**:
1. Compute MAD for each angle signal (robust variability measure)
2. Run FFT to find dominant frequency
3. Calculate window: `window = period/4 × fps` (capped 5-31, must be odd)
4. Select polynomial order: 2 for high noise (MAD > 15°), else 3

**Parameters**:
```python
angles_df: pd.DataFrame    # Angle data
angle_cols: List[str]      # Angle column names
fps: float                 # Sampling rate
```

**Returns**: `(window_length, poly_order)` optimized for signal characteristics

**Example Output**:
```
⚙️ Calibrated filter (FFT + MAD): window=15, poly_order=2
   Signal variability (MAD median): 15.61°
```

---

#### `auto_calibrate_config()` (Line 1812-1939)

**Purpose**: Comprehensive data-driven parameter calibration with MAD

**Enhanced Features** (Nov 4):
- Data quality assessment using `DataQualityMetrics.compute()`
- Adaptive quality gates based on quality score:
  - Score < 50: coverage≥50%, max_gap=50
  - Score < 75: coverage≥60%, max_gap=40
  - Score ≥ 75: coverage≥70%, max_gap=30 (standard)
- MAD-based smoothing threshold: `2.5 × MAD × 1.4826`
- FFT + MAD filter calibration

**Parameters**:
```python
angles_df: pd.DataFrame
events_df: pd.DataFrame
col_map: Dict[str, Dict[str, str]]
base_config: AnalysisConfig
min_cycles: int = 5
```

**Returns**: `Tuple[AnalysisConfig, DataQualityMetrics]`

**Calibrates**:
1. Data quality metrics (NEW - MAD-based)
2. Filter parameters (NEW - FFT + MAD)
3. Quality gate thresholds (NEW - adaptive to data quality)
4. Smoothing threshold (enhanced with MAD)
5. Stride duration constraints
6. Pairing tolerance

---

### Quality Assessment Functions

#### `compute_window_quality()` (Line 562-688)

**Purpose**: Multi-gate quality assessment for a single gait cycle

**Quality Gates**:
1. **Coverage**: `≥ min_coverage_pct` (default 70%)
2. **Gaps**: `max_gap ≤ max_gap_frames` (default 30)
3. **Stability**: `pelvis_std ≤ max_pelvis_std` (default 15 pixels)
4. **Sanity**: `max_angle_jump ≤ max_angle_jump` (default 45°)

**Parameters**:
```python
angles_df: pd.DataFrame
keypoints_df: Optional[pd.DataFrame]
window: StrideWindow
angle_cols: List[str]
config: AnalysisConfig
```

**Returns**: `WindowQualityMetrics` with pass/fail results

---

### Cycle Extraction Functions

#### `build_cycle_windows()` (Line 1993-2181)

**Purpose**: Extract HS→HS gait cycles with quality control

**Process**:
1. Load events for specified leg
2. Pair consecutive heel strikes (HS_i → HS_i+1)
3. Validate duration constraints
4. Apply quality gates if enabled
5. Track rejection reasons

**Enhanced Error Reporting** (Nov 4, line 2119-2174):
```python
if all cycles rejected:
    print rejection reasons (coverage, gaps, stability, sanity)
    suggest --enhance-angles flag
```

**Parameters**:
```python
events_df: pd.DataFrame
leg: str                           # 'L' or 'R'
angles_df: pd.DataFrame
keypoints_df: Optional[pd.DataFrame]
config: AnalysisConfig
min_duration: float
max_duration: float
apply_quality_gates: bool = True
```

**Returns**: `Tuple[List[StrideWindow], Dict[str, int]]`
- Windows: Valid HS→HS cycles
- Rejections: `{'rejected_coverage': 3, 'rejected_gaps': 1, ...}`

---

### Metrics Computation Functions

#### `compute_all_metrics()` (Line 2901-3019)

**Purpose**: Complete bilateral L-R comparison metric suite

**Metrics Computed**:
1. **Area** (if loops closed):
   - Signed area (shoelace formula)
   - Normalized area (scale-free)
   - Delta area %
2. **Shape**:
   - RMSE (amplitude-normalized)
   - Procrustes distance (optimal alignment)
   - DTW distance (timing asymmetry)
3. **Orientation**:
   - PCA major axis angles
   - Delta orientation
4. **Hysteresis**:
   - CW/CCW direction (signed area)
   - Mismatch detection
5. **Similarity Score**:
   - Weighted composite (0-100)

**Parameters**:
```python
loop_L: CyclogramLoop
loop_R: CyclogramLoop
weights: Dict[str, float]
```

**Returns**: `CyclogramMetrics` with all bilateral comparison values

---

### Visualization Functions (NEW - Nov 4, 2025)

#### `plot_overlayed_cyclograms()` (Line 3023-3349)

**Purpose**: Side-by-side L+R cyclogram dashboard with adaptive sizing

**Features**:
- Adaptive figure size based on `num_cycles` (logarithmic scaling)
- Screen-optimized DPI (150 instead of 300)
- Individual strides + mean cyclogram
- Asymmetry indicators
- Statistics boxes

**Adaptive Sizing** (NEW - line 3047-3049):
```python
num_cycles = max(len(loops_L), len(loops_R))
figsize = plot_config.calculate_cyclogram_figsize(num_cycles=num_cycles)
dpi = plot_config.get_dpi()  # Returns 150 for 'screen' preset
```

**Parameters**:
```python
loops_L: List[CyclogramLoop]
loops_R: List[CyclogramLoop]
metrics: List[CyclogramMetrics]
joint_pair: Tuple[str, str]
output_path: str
plot_config: Optional[PlotConfig]
```

**Output**: PNG file (e.g., `CK_hip_knee_AllStrides.png`)

**File Size**: ~150-240 KB (was 5-10 MB with old 300 DPI)

---

#### `plot_similarity_summary()` (Line 3352-3426)

**Purpose**: Bar chart of L-R similarity scores per joint pair

**Adaptive Sizing** (NEW - line 3369-3371):
```python
num_metrics = len(session_summary)
figsize = plot_config.calculate_similarity_figsize(num_metrics=num_metrics)
dpi = plot_config.get_dpi()
```

**Parameters**:
```python
session_summary: pd.DataFrame
output_path: str
plot_config: Optional[PlotConfig]
```

**Output**: PNG file (`LR_Similarity_Summary.png`)

**File Size**: ~48 KB (was ~4 MB)

---

## Processing Workflows

### Main Pipeline (analyze_subject)

**Location**: Line 3576-3900

**Step-by-Step**:

```
1. LOAD DATA (line 3595-3611)
   ├─ Read Raw_Angles.csv
   ├─ Read Angle_Events.csv
   └─ Print: " Loaded data: {frames} frames, {events} events"

2. AUTO-ENHANCEMENT CHECK (line 3612-3641) [NEW - Nov 4]
   ├─ IF auto_enhance=True AND enhance_angles=False:
   │   ├─ Quick coverage check
   │   ├─ IF coverage < auto_enhance_threshold (70%):
   │   │   ├─ Print warning banner
   │   │   └─ SET enhance_angles = True
   │   └─ ELSE: Skip enhancement
   └─ Continue

3. ENHANCEMENT (line 3643-3645)
   ├─ IF enhance_angles=True:
   │   └─ Call enhance_angles_from_keypoints()
   │       ├─ Tier 1: PCHIP interpolation
   │       ├─ Tier 2: Geometric recalculation
   │       └─ Tier 3: Temporal smoothing
   └─ Print enhancement statistics

4. COLUMN DETECTION (line 3647-3653)
   └─ Call detect_columns(angles_df)
       └─ Regex: hip/knee/ankle + flex/dorsi + L/R + deg

5. TIMEBASE NORMALIZATION (line 3655-3661)
   └─ Align angles_df and events_df time indices

6. AUTO-CALIBRATION (line 3663-3668) [ENHANCED - Nov 4]
   └─ Call auto_calibrate_config()
       ├─ Compute DataQualityMetrics (MAD-based)
       ├─ Calibrate filter (FFT + MAD)
       ├─ Adaptive quality gates (based on quality score)
       ├─ MAD-based smoothing threshold
       ├─ Stride duration constraints
       └─ Pairing tolerance

7. PREPROCESSING (line 3670-3680)
   └─ Savitzky-Golay smoothing (adaptive window+poly)

8. CYCLE SEGMENTATION (line 3682-3716)
   ├─ Load keypoints (for pelvis stability check)
   ├─ Call build_cycle_windows(leg='L')
   ├─ Call build_cycle_windows(leg='R')
   └─ Print: " · Segmented cycles (HS→HS): {L} left, {R} right"

9. QUALITY CHECK (line 3718-3726)
   ├─ IF insufficient cycles (< 2 per leg):
   │   └─ Print warning and return None
   └─ ELSE: Continue

10. CYCLOGRAM EXTRACTION (line 3728-3760)
    ├─ FOR each joint pair:
    │   ├─ FOR each L cycle:
    │   │   └─ extract_cyclogram() → CyclogramLoop
    │   └─ FOR each R cycle:
    │       └─ extract_cyclogram() → CyclogramLoop
    └─ Print: " Extracted cyclograms: {L} left, {R} right"

11. QUALITY FILTERING (line 3762-3775)
    ├─ Filter loops with high NaN% or unclosed
    └─ Print: "· Extracted cyclograms (after QC): {L} left, {R} right"

12. L-R PAIRING (line 3777-3793)
    ├─ Call pair_strides() (index-based: L₁↔R₁)
    ├─ Validate temporal overlap (≥30%)
    └─ Print: " Paired cycles: {pairs} pairs"

13. PIPELINE SUMMARY (line 3795-3806)
    └─ Print quality control summary table

14. METRICS COMPUTATION (line 3808-3856)
    ├─ FOR each paired stride:
    │   ├─ Get CyclogramLoop_L and CyclogramLoop_R
    │   └─ compute_all_metrics() → CyclogramMetrics
    └─ Print: " Calculated metrics: {count} comparisons"

15. VISUALIZATION (line 3858-3882) [ENHANCED - Nov 4]
    ├─ FOR each joint pair:
    │   ├─ Create PlotConfig() (adaptive sizing, screen DPI)
    │   └─ plot_overlayed_cyclograms() → PNG
    └─ IF metrics exist:
        ├─ Create PlotConfig()
        └─ plot_similarity_summary() → PNG

16. EXPORT (line 3884-3897)
    ├─ write_stride_metrics() → cyclogram_stride_metrics.csv
    ├─ write_session_summary() → cyclogram_session_summary.csv
    └─ Print: " Exported metrics"

17. COMPLETION (line 3899)
    └─ Print: " Analysis complete for {subject_name}"
```

---

### Enhancement Workflow (Multi-Tier)

**Location**: Line 238-457

**Tier 1: PCHIP Interpolation** (Line 269-316)

```
FOR each angle column:
    ├─ Identify NaN segments
    ├─ FOR each gap:
    │   ├─ IF gap_size ≤ 10 frames:
    │   │   ├─ Extract surrounding valid data
    │   │   ├─ Fit PCHIP (shape-preserving cubic Hermite)
    │   │   └─ Fill gap with interpolated values
    │   └─ ELSE: Skip (too large for interpolation)
    └─ Track: gap_fill_count per angle
```

**Tier 2: Geometric Recalculation** (Line 318-395)

```
Load Clean_keypoints.csv

FOR each angle type (hip, knee, ankle):
    ├─ FOR each leg (L, R):
    │   ├─ Extract landmark coordinates (hip=23/24, knee=25/26, ankle=27/28, foot=31/32)
    │   ├─ FOR each frame where angle is NaN:
    │   │   ├─ IF landmarks valid:
    │   │   │   ├─ calculate_hip_flexion(hip_y, knee_y, pelvis_y)
    │   │   │   │   OR calculate_knee_flexion(hip_y, hip_x, knee_y, knee_x, ankle_y, ankle_x)
    │   │   │   │   OR calculate_ankle_dorsiflexion(knee_y, knee_x, ankle_y, ankle_x, foot_y, foot_x)
    │   │   │   └─ Fill NaN with calculated angle
    │   │   └─ ELSE: Skip frame
    │   └─ Track: recalc_count per angle
    └─ Print statistics
```

**Tier 3: Temporal Smoothing** (Line 397-442)

```
FOR each angle column:
    ├─ Identify remaining NaN segments
    ├─ FOR each gap:
    │   ├─ IF gap_size ≤ 3 frames AND has valid neighbors:
    │   │   ├─ Linear interpolation between neighbors
    │   │   └─ Limit max change to smooth_threshold (15°)
    │   └─ ELSE: Leave as NaN
    └─ Final smoothing pass (Savitzky-Golay, window=5)
```

**Final Report**:
```
✓ Enhancement complete - Final data quality:
    L coverage: hip=100.0%, knee=100.0%, ankle=100.0%
    L max jumps: hip=10.8°, knee=15.0°, ankle=15.0°
    R coverage: hip=100.0%, knee=100.0%, ankle=100.0%
    R max jumps: hip=5.7°, knee=15.0°, ankle=15.0°
```

---

### Quality Gate Workflow

**Location**: Line 2110-2176 (in build_cycle_windows)

```
FOR each extracted cycle:
    │
    ├─> [1] COVERAGE GATE
    │   ├─ Compute % non-NaN frames per angle
    │   ├─ IF any angle < min_coverage_pct:
    │   │   ├─ REJECT cycle
    │   │   └─ Track: rejection_counts['rejected_coverage'] += 1
    │   └─ ELSE: PASS
    │
    ├─> [2] GAP GATE
    │   ├─ Find longest contiguous NaN gap per angle
    │   ├─ IF any gap > max_gap_frames:
    │   │   ├─ REJECT cycle
    │   │   └─ Track: rejection_counts['rejected_gaps'] += 1
    │   └─ ELSE: PASS
    │
    ├─> [3] STABILITY GATE
    │   ├─ IF keypoints_df exists:
    │   │   ├─ Compute pelvis_y std dev
    │   │   ├─ IF pelvis_std > max_pelvis_std:
    │   │   │   ├─ REJECT cycle
    │   │   │   └─ Track: rejection_counts['rejected_stability'] += 1
    │   │   └─ ELSE: PASS
    │   └─ ELSE: PASS (N/A)
    │
    ├─> [4] SANITY GATE
    │   ├─ Compute max frame-to-frame angle change per angle
    │   ├─ IF any jump > max_angle_jump:
    │   │   ├─ REJECT cycle
    │   │   └─ Track: rejection_counts['rejected_sanity'] += 1
    │   └─ ELSE: PASS
    │
    └─> ALL GATES PASSED → Add to valid_windows
```

**Enhanced Error Reporting** (if ALL cycles rejected):
```
================================================================================
❌ QUALITY GATE FAILURE (L leg): All 8 cycles rejected
================================================================================
Primary issue:
  • Insufficient coverage (<50.0%) in 8 cycles

💡 SOLUTION: Use --enhance-angles flag to fill gaps and recalculate missing data
   python3 Pose-Analysis.py --input-dir <path> --enhance-angles
================================================================================
```

---

## Configuration Guide

### AnalysisConfig Parameters

**Timing**:
```python
fps: Optional[float] = None        # Auto-detect from timestamp column
```

**Smoothing** (Savitzky-Golay filter):
```python
smooth_window: int = 11            # Window size (must be odd, 5-31 range)
smooth_poly: int = 2               # Polynomial order (2 or 3)
smooth_threshold: float = 5.0      # Jump detection threshold (degrees)
```
*Calibrated automatically by `calibrate_filter_parameters()` using FFT + MAD*

**Stride/Cycle Duration Constraints**:
```python
min_stride_duration: float = 0.3   # Min HS→HS cycle duration (seconds)
max_stride_duration: float = 3.0   # Max HS→HS cycle duration
```
*Calibrated automatically using 10th-90th percentiles of actual durations*

**Pairing**:
```python
pairing_tolerance: float = 0.15    # Max duration ratio difference (15%)
use_index_pairing: bool = True     # Index-based (Lₖ↔Rₖ) vs mid-time proximity
min_phase_overlap: float = 0.3     # Min temporal overlap (30%)
```
*Pairing tolerance calibrated from coefficient of variation*

**Quality Gates**:
```python
min_coverage_pct: float = 70.0     # Min % non-NaN frames per joint
max_gap_frames: int = 30           # Max contiguous NaN gap (frames)
max_pelvis_std: float = 15.0       # Max pelvis vertical motion (pixels)
max_angle_jump: float = 45.0       # Max frame-to-frame angle change (degrees)
```
*Adaptive thresholds based on DataQualityMetrics score:*
- Score < 50: `min_coverage_pct = 50.0`, `max_gap_frames = 50`
- Score < 75: `min_coverage_pct = 60.0`, `max_gap_frames = 40`
- Score ≥ 75: Standard thresholds (70%, 30 frames)

**Enhancement** (NEW - Nov 4):
```python
enhance_angles: bool = False       # Enable keypoint-based angle recalculation
auto_enhance: bool = True          # Auto-enable enhancement for poor quality data
auto_enhance_threshold: float = 70.0  # Coverage % threshold
```

**Joint Pairs**:
```python
joint_pairs: List[Tuple[str, str]] = [
    ("hip", "knee"),      # Thigh coordination
    ("knee", "ankle"),    # Shank coordination
    ("hip", "ankle")      # Full leg coordination
]
```

**Similarity Weights**:
```python
similarity_weights: Dict[str, float] = {
    "area": 0.30,         # Loop size contribution
    "procrustes": 0.30,   # Shape similarity contribution
    "rmse": 0.30,         # Amplitude difference contribution
    "orientation": 0.10   # Tilt/orientation contribution
}
```

---

### PlotConfig Parameters (NEW - Nov 4)

**Sizing Mode**:
```python
sizing_mode: str = 'adaptive'      # 'adaptive' or 'fixed'
```
- `'adaptive'`: Content-aware sizing (recommended)
- `'fixed'`: Use legacy hardcoded sizes

**DPI Preset**:
```python
dpi_preset: str = 'screen'         # 'web', 'screen', or 'print'
```
- `'web'`: 96 DPI (smallest files, web graphics)
- `'screen'`: 150 DPI (balanced quality/size, default)
- `'print'`: 300 DPI (publication quality, largest files)

**Base Sizes** (for adaptive mode):
```python
base_cyclogram_figsize: Tuple[float, float] = (8, 6)   # Single subplot reference
base_similarity_figsize: Tuple[float, float] = (10, 6)
```

**Size Constraints**:
```python
min_figsize: Tuple[float, float] = (6, 4)
max_figsize: Tuple[float, float] = (20, 15)
```

**Legacy Fixed Sizes** (used when `sizing_mode='fixed'`):
```python
cyclogram_figsize: Tuple[float, float] = (16, 8)
similarity_figsize: Tuple[float, float] = (12, 7)
dpi: int = 150                     # Direct DPI override
```

**Example Results**:
- Adaptive mode, screen DPI: 1464×1097 pixels (~150 KB files)
- Fixed mode, print DPI: 4800×2400 pixels (~8 MB files)

---

## Recent Improvements

### November 4, 2025 Updates

#### 1. MAD-Based Robust Statistics

**Motivation**: Standard deviation is sensitive to outliers; MAD (Median Absolute Deviation) provides robust variability estimation.

**Implementation**:

**DataQualityMetrics class** (line 1185):
- New `compute_mad()` static method
- MAD-based temporal consistency: `sampling_rate_mad`
- MAD-based variability: `angle_variability_mad` per joint
- Robust quality scoring using MAD thresholds

**PlotConfig adaptive thresholds** (line 1052):
- `calibrate_quality_thresholds()`: Uses MAD to set excellent/good/acceptable bounds
- Formula: `excellent = median + 0.5×MAD×1.4826`
- Scale factor 1.4826 matches std dev for normal distributions

**Filter calibration** (line 1731):
- FFT determines dominant frequency
- MAD measures noise level
- Window: `period/4 × fps`, Polynomial: 2 for MAD > 15°, else 3

**Auto-calibration** (line 1812):
- Returns `Tuple[AnalysisConfig, DataQualityMetrics]`
- Adaptive quality gates based on quality score
- MAD-based smoothing threshold: `2.5 × MAD × 1.4826`

**Benefits**:
- Robust to outliers in biomechanical data
- Adaptive thresholds prevent spurious rejections
- Better handling of noisy or incomplete data

---

#### 2. Adaptive Plot Sizing

**Motivation**: Fixed 16×8" @ 300 DPI created 4800×2400 pixel images (~8 MB files) regardless of content.

**Implementation**:

**PlotConfig enhancements** (line 964):
```python
sizing_mode: str = 'adaptive'      # Content-aware sizing
dpi_preset: str = 'screen'         # 150 DPI default (was 300)
base_cyclogram_figsize = (8, 6)    # Reference size
```

**Adaptive methods** (line 1108-1180):
```python
def calculate_cyclogram_figsize(num_cycles: int) -> Tuple[float, float]:
    """Logarithmic scaling based on number of cycles."""
    base_w, base_h = self.base_cyclogram_figsize
    width = base_w * 2.0  # Two subplots (L+R)
    density_factor = min(1.3, 1.0 + np.log1p(num_cycles) / 10)
    height = base_h * density_factor
    return (np.clip(width, 6, 20), np.clip(height, 4, 15))
```

**Plot generation updated** (line 3844, 3853):
```python
# OLD: plot_cfg = PlotConfig(dpi=config.plot_dpi)  # 300 DPI
# NEW: plot_cfg = PlotConfig()  # Adaptive sizing, 150 DPI
```

**Results**:
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Cyclogram DPI | 300 | 150 | 50% reduction |
| Cyclogram Size | 4800×2400px | 1464×1097px | 85% fewer pixels |
| File Size | 5-10 MB | 153-239 KB | **95% smaller** |
| Similarity File | 2-5 MB | 48 KB | **97% smaller** |

**Benefits**:
- Perfect screen quality maintained
- Faster loading and sharing
- Adapts to data density
- Flexible DPI presets (web/screen/print)

---

#### 3. Auto-Enhancement

**Motivation**: Users had to manually add `--enhance-angles` even when script detected poor quality data (37.2/100 score).

**Implementation**:

**New AnalysisConfig fields** (line 946-947):
```python
auto_enhance: bool = True          # Auto-enable enhancement for poor data
auto_enhance_threshold: float = 70.0  # Coverage % threshold
```

**Auto-enhancement check** (line 3612-3641):
```python
if config.auto_enhance and not config.enhance_angles:
    # Quick coverage check
    overall_coverage = (valid_values / total_values) * 100

    if overall_coverage < config.auto_enhance_threshold:
        print banner: "⚠️ LOW DATA QUALITY DETECTED"
        print: "Coverage: {coverage}% (threshold: {threshold}%)"
        print: "Auto-enabling angle enhancement..."
        config.enhance_angles = True
```

**New CLI flag** (line 3983):
```python
--no-auto-enhance    # Disable automatic enhancement
```

**Behavior**:
- Default: Auto-enhances if coverage < 70%
- User can disable: `--no-auto-enhance`
- User can force: `--enhance-angles`

**Results**:
- Coverage: 58.6% → 100.0%
- Quality score: 37.2/100 → 89.4/100
- Cycles extracted: 0 → 17
- No user intervention required

---

## Usage Examples

### Basic Usage (Smart Defaults)

```bash
# Analyze all subjects with auto-enhancement (RECOMMENDED)
python3 Pose-Analysis.py --input-dir "/path/to/data"

# Single subject
python3 Pose-Analysis.py --subject-name "Openpose_subject_001"
```

**What happens**:
1. Auto-detects poor quality data (coverage < 70%)
2. Auto-enables enhancement (PCHIP + recalculation + smoothing)
3. Adaptive plot sizing (screen DPI, content-aware dimensions)
4. Generates cyclograms + metrics

---

### Disable Auto-Enhancement (For Testing)

```bash
# Force error on poor quality data
python3 Pose-Analysis.py --no-auto-enhance
```

**Result**: Shows quality gate failures with helpful error messages

---

### Explicit Enhancement (Always Enhance)

```bash
# Force enhancement even for good quality data
python3 Pose-Analysis.py --enhance-angles
```

---

### Print Quality Output

```bash
# High-resolution plots for publications
python3 Pose-Analysis.py --enhance-angles --dpi-preset print
```

**Note**: `--dpi-preset` flag would need to be added to CLI parser. Currently controlled via PlotConfig defaults.

---

### Custom Parameters

```bash
# Custom smoothing and constraints
python3 Pose-Analysis.py \
  --enhance-angles \
  --smooth-window 15 \
  --smooth-threshold 7.0 \
  --input-dir "/home/user/data" \
  --output-dir "/home/user/results"
```

---

### Batch Processing

```bash
# Process all subjects in directory
python3 Pose-Analysis.py --input-dir "/path/with/multiple/subjects"
```

**Output structure**:
```
Output/
├── Openpose_subject_001/
│   ├── CK_hip_knee_AllStrides.png
│   ├── CK_knee_ankle_AllStrides.png
│   ├── CK_hip_ankle_AllStrides.png
│   ├── LR_Similarity_Summary.png
│   ├── cyclogram_stride_metrics.csv
│   └── cyclogram_session_summary.csv
└── Openpose_subject_002/
    └── ... (same structure)
```

---

## Quick Reference

### Line Number Index

| Component | Line | Description |
|-----------|------|-------------|
| **Angle Calculation** | 54-120 | Knee, hip, ankle flexion formulas |
| **Signal Processing** | 122-237 | Smoothing, gap filling |
| **Enhancement** | 238-457 | 3-tier angle recovery |
| **Data Structures** | 690-1184 | 8 @dataclass definitions |
| **DataQualityMetrics** | 1185-1384 | MAD-based quality assessment |
| **Column Detection** | 1385-1436 | Regex pattern matching |
| **Preprocessing** | 1437-1509 | Timebase, smoothing |
| **Filter Calibration** | 1731-1810 | FFT + MAD (NEW) |
| **Auto-Calibration** | 1812-1939 | Enhanced with MAD (NEW) |
| **Cycle Extraction** | 1993-2181 | HS→HS segmentation + QC |
| **Cyclogram Generation** | 2183-2388 | Loop extraction |
| **Pairing** | 2389-2502 | Index-based L-R matching |
| **Metrics** | 2503-2900 | Area, RMSE, Procrustes, DTW |
| **Visualization** | 3023-3426 | Adaptive plots (NEW) |
| **Export** | 3427-3575 | CSV serialization |
| **Main Pipeline** | 3576-3900 | analyze_subject() |
| **CLI** | 3960-4065 | argparse + batch |

---

### Key Functions

| Function | Line | Purpose |
|----------|------|---------|
| `enhance_angles_from_keypoints` | 238 | 3-tier enhancement |
| `detect_columns` | 1385 | Regex column detection |
| `normalize_timebase` | 1437 | Time alignment |
| `calibrate_filter_parameters` | 1731 | FFT + MAD filter (NEW) |
| `auto_calibrate_config` | 1812 | Enhanced calibration (NEW) |
| `build_cycle_windows` | 1993 | HS→HS extraction + QC |
| `extract_cyclogram` | 2279 | Loop generation |
| `pair_strides` | 2389 | Index-based pairing |
| `compute_all_metrics` | 2901 | Bilateral metrics |
| `plot_overlayed_cyclograms` | 3023 | Adaptive visualization (NEW) |
| `plot_similarity_summary` | 3352 | Bar chart |
| `analyze_subject` | 3576 | Main pipeline |

---

### Configuration Defaults

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `auto_enhance` | `True` | Auto-enable for poor data (NEW) |
| `auto_enhance_threshold` | `70.0` | Coverage % trigger (NEW) |
| `smooth_window` | `11` | Savitzky-Golay window |
| `smooth_poly` | `2` | Polynomial order |
| `min_coverage_pct` | `70.0` | Quality gate (adaptive) |
| `max_gap_frames` | `30` | Quality gate (adaptive) |
| `use_index_pairing` | `True` | Phase-true pairing |
| `sizing_mode` | `'adaptive'` | Content-aware plots (NEW) |
| `dpi_preset` | `'screen'` | 150 DPI default (NEW) |

---

**End of Comprehensive Index**
