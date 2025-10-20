# Cyclogram Analysis System - Comprehensive Design Specification

## Executive Summary

This document specifies a production-ready system for generating stride-wise cyclograms with quantitative left-right (LR) asymmetry analysis. The system processes gait analysis data to produce joint angle phase plots (angle-angle diagrams) per stride, overlays them for comparative visualization, and computes geometric similarity metrics between left and right limbs.

**Key Features:**
- Stride-wise cyclogram generation from joint angle time series
- Left-right stride pairing and asymmetry quantification
- Multiple geometric metrics (area, RMSE, Procrustes distance, orientation)
- Composite similarity scoring (0-100 scale)
- Multi-panel visualization with statistical overlays
- Robust handling of edge cases and missing data

---

## 1. System Architecture

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CYCLOGRAM SYSTEM                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │   Data       │───▶│  Processing  │───▶│ Computation  │    │
│  │   Loader     │    │   Pipeline   │    │   Engine     │    │
│  └──────────────┘    └──────────────┘    └──────────────┘    │
│         │                    │                    │            │
│         ▼                    ▼                    ▼            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │  Validation  │    │   Stride     │    │   Metrics    │    │
│  │   Module     │    │  Segmenter   │    │  Calculator  │    │
│  └──────────────┘    └──────────────┘    └──────────────┘    │
│                                                   │            │
│                                                   ▼            │
│                           ┌──────────────────────────┐        │
│                           │  Visualization Engine    │        │
│                           └──────────────────────────┘        │
│                                     │                          │
│                                     ▼                          │
│                           ┌──────────────────────────┐        │
│                           │   Output Serializer      │        │
│                           └──────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Module Dependency Graph

```
DataLoader
    ├──▶ ColumnDetector
    ├──▶ TimebaseNormalizer
    └──▶ DataValidator
         └──▶ PreProcessor (smoothing, unwrap)
              └──▶ StrideSegmenter
                   └──▶ CyclogramExtractor
                        ├──▶ StridePairer
                        └──▶ MetricsCalculator
                             ├──▶ AreaCalculator
                             ├──▶ RMSECalculator
                             ├──▶ ProcrustesCalculator
                             ├──▶ OrientationCalculator
                             └──▶ SimilarityScorer
                                  └──▶ VisualizationEngine
                                       └──▶ OutputSerializer
```

---

## 2. Data Model

### 2.1 Input Data Structures

#### 2.1.1 Raw Angles CSV
```python
RawAnglesDataFrame:
    columns:
        - frame: int                    # Frame index (0-based)
        - timestamp: float              # Time in seconds
        - {joint}_{side}_deg: float     # Joint angles in degrees
          where joint ∈ {hip_flex, knee_flex, ankle_dorsi}
          where side ∈ {L, R}

    example_columns:
        ["frame", "timestamp", "hip_flex_L_deg", "hip_flex_R_deg",
         "knee_flex_L_deg", "knee_flex_R_deg",
         "ankle_dorsi_L_deg", "ankle_dorsi_R_deg"]
```

#### 2.1.2 Angle Events CSV
```python
EventsDataFrame:
    columns:
        - frame: int                    # Frame index
        - timestamp: float              # Time in seconds
        - side: str                     # "L" or "R"
        - event_type: str               # "heel_strikes" or "toe_offs"

    constraints:
        - Events must be sorted by timestamp
        - Each stride requires both heel_strike and toe_off
        - Events alternate per leg (heel_strike → toe_off → heel_strike)
```

#### 2.1.3 Stride Stats CSV (Optional Enrichment)
```python
StrideStatsDataFrame:
    columns:
        - side: str
        - stride_number: int
        - start_frame: int
        - end_frame: int
        - duration_s: float
        - speed_ms: float (optional)
```

### 2.2 Internal Data Structures

#### 2.2.1 Stride Window
```python
@dataclass
class StrideWindow:
    """Represents a single stride window from heel strike to toe-off."""
    leg: Literal["L", "R"]
    stride_id: int                      # Sequential ID per leg
    start_time: float                   # Heel strike time (s)
    end_time: float                     # Toe-off time (s)
    duration: float                     # end_time - start_time (s)
    start_frame: int
    end_frame: int
    speed: Optional[float] = None       # Walking speed if available
```

#### 2.2.2 Cyclogram Loop
```python
@dataclass
class CyclogramLoop:
    """Represents a single cyclogram trajectory."""
    leg: Literal["L", "R"]
    stride_id: int
    joint_pair: Tuple[str, str]        # e.g., ("hip", "knee")
    proximal: np.ndarray               # Shape: (101,) - proximal joint angles
    distal: np.ndarray                 # Shape: (101,) - distal joint angles
    time_normalized: np.ndarray        # Shape: (101,) - normalized time 0-100%
    duration: float                    # Stride duration (s)
    speed: Optional[float] = None

    @property
    def points(self) -> np.ndarray:
        """Returns (101, 2) array of [proximal, distal] coordinates."""
        return np.column_stack([self.proximal, self.distal])
```

#### 2.2.3 Paired Strides
```python
@dataclass
class PairedStrides:
    """Represents matched left-right stride pair."""
    left_stride: StrideWindow
    right_stride: StrideWindow
    time_difference: float             # |mid_time_L - mid_time_R| (s)
    temporal_overlap: float            # Overlap percentage
```

#### 2.2.4 Cyclogram Metrics
```python
@dataclass
class CyclogramMetrics:
    """Quantitative comparison metrics for a paired cyclogram."""
    joint_pair: Tuple[str, str]
    left_stride_id: int
    right_stride_id: int

    # Geometric metrics
    area_left: float                   # Signed polygon area (deg²)
    area_right: float
    delta_area_percent: float          # 200*(AL-AR)/(|AL|+|AR|)

    rmse: float                        # Root mean square error after z-norm
    procrustes_distance: float         # Shape similarity (0=identical)

    orientation_left: float            # PCA major axis angle (deg)
    orientation_right: float
    delta_orientation: float           # |θL - θR| (deg)

    hysteresis_left: str              # "CW" or "CCW"
    hysteresis_right: str
    hysteresis_mismatch: bool          # True if directions differ

    # Composite score
    similarity_score: float            # 0-100 (100=perfect symmetry)
```

### 2.3 Output Data Structures

#### 2.3.1 Stride-Level Metrics CSV
```csv
stride_id_L,stride_id_R,joint_pair,area_L,area_R,delta_area_pct,
rmse,procrustes,delta_orient,similarity_score
1,1,Hip-Knee,245.3,238.7,2.7,0.15,0.08,3.2,92.5
1,1,Knee-Ankle,189.4,195.1,-3.0,0.22,0.12,5.1,88.3
...
```

#### 2.3.2 Session-Level Summary CSV
```csv
joint_pair,n_pairs,delta_area_mean,delta_area_std,procrustes_mean,
procrustes_std,similarity_mean,similarity_std
Hip-Knee,15,2.1,1.8,0.09,0.04,91.2,3.5
Knee-Ankle,15,-1.5,2.3,0.14,0.06,87.8,4.2
Hip-Ankle,15,3.2,2.9,0.11,0.05,89.5,3.9
```

---

## 3. Component Specifications

### 3.1 Data Loader Module

#### 3.1.1 ColumnDetector
**Purpose:** Auto-detect joint angle columns using regex patterns.

**Algorithm:**
```python
def detect_columns(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """
    Returns: {
        "L": {"hip": "hip_flex_L_deg", "knee": "knee_flex_L_deg", ...},
        "R": {"hip": "hip_flex_R_deg", ...}
    }
    """
    patterns = {
        "hip": r"(?i)(L|left).*(hip|flex).*(?!knee|ankle)",
        "knee": r"(?i)(L|left).*knee.*flex",
        "ankle": r"(?i)(L|left).*ankle.*(dorsi|flex)"
    }
    # Apply patterns for both L and R
    # Raise ValueError if required columns missing
```

**Validation:**
- All 6 joint columns must be present (hip/knee/ankle × L/R)
- Columns must contain numeric data
- No duplicate mappings allowed

#### 3.1.2 TimebaseNormalizer
**Purpose:** Ensure consistent time-based indexing.

**Algorithm:**
```python
def normalize_timebase(angles_df: pd.DataFrame,
                       events_df: pd.DataFrame,
                       fps: Optional[float] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Priority: Use 'timestamp' or 'time_s' if present
    Fallback: Derive from 'frame' column using FPS
    """
    if "timestamp" in angles_df.columns:
        time_col = "timestamp"
    elif "time_s" in angles_df.columns:
        time_col = "time_s"
    else:
        if fps is None:
            raise ValueError("FPS required when timestamp column missing")
        angles_df["timestamp"] = angles_df["frame"] / fps
        events_df["timestamp"] = events_df["frame"] / fps
```

#### 3.1.3 PreProcessor
**Purpose:** Smooth and unwrap joint angles.

**Algorithm:**
```python
def preprocess_angles(df: pd.DataFrame,
                     angle_cols: List[str],
                     smooth_window: int = 11,
                     poly_order: int = 2,
                     smooth_threshold: float = 5.0) -> pd.DataFrame:
    """
    1. Check angle variability (std dev)
    2. Apply smoothing only if std > threshold
    3. Unwrap angles to avoid 0/360 discontinuities
    """
    for col in angle_cols:
        angles = df[col].values

        # Conditional smoothing
        if np.std(angles) > smooth_threshold:
            from scipy.signal import savgol_filter
            angles = savgol_filter(angles, smooth_window, poly_order)

        # Unwrap discontinuities
        angles = np.unwrap(np.deg2rad(angles))
        angles = np.rad2deg(angles)

        df[col] = angles

    return df
```

**Parameters:**
- `smooth_window`: 11 (must be odd)
- `poly_order`: 2
- `smooth_threshold`: 5.0 degrees (only smooth if std > threshold)

---

### 3.2 Stride Segmentation Module

#### 3.2.1 StrideSegmenter
**Purpose:** Extract stride windows from gait events.

**Algorithm:**
```python
def build_stride_windows(events_df: pd.DataFrame,
                        leg: str,
                        min_duration: float = 0.3,
                        max_duration: float = 3.0) -> List[StrideWindow]:
    """
    For each leg:
    1. Filter events for this leg
    2. Sort by timestamp
    3. Pair heel_strikes with subsequent toe_offs
    4. Validate duration constraints
    """
    leg_events = events_df[events_df["side"] == leg].sort_values("timestamp")

    strikes = leg_events[leg_events["event_type"] == "heel_strikes"]
    toe_offs = leg_events[leg_events["event_type"] == "toe_offs"]

    windows = []
    for i, strike_row in strikes.iterrows():
        # Find next toe_off after this strike
        subsequent_toeoffs = toe_offs[toe_offs["timestamp"] > strike_row["timestamp"]]
        if subsequent_toeoffs.empty:
            continue  # Incomplete stride

        toeoff_row = subsequent_toeoffs.iloc[0]
        duration = toeoff_row["timestamp"] - strike_row["timestamp"]

        # Validate duration
        if not (min_duration <= duration <= max_duration):
            continue  # Outlier stride

        windows.append(StrideWindow(
            leg=leg,
            stride_id=len(windows) + 1,
            start_time=strike_row["timestamp"],
            end_time=toeoff_row["timestamp"],
            duration=duration,
            start_frame=strike_row["frame"],
            end_frame=toeoff_row["frame"]
        ))

    return windows
```

**Validation:**
- Minimum 2 strides per leg required
- Maximum stride count: unlimited
- Duration constraints: 0.3s ≤ duration ≤ 3.0s

---

### 3.3 Cyclogram Extraction Module

#### 3.3.1 CyclogramExtractor
**Purpose:** Generate normalized cyclogram loops for each stride.

**Algorithm:**
```python
def extract_cyclogram(angles_df: pd.DataFrame,
                     window: StrideWindow,
                     joint_pair: Tuple[str, str],
                     col_map: Dict[str, str],
                     n_points: int = 101) -> CyclogramLoop:
    """
    1. Extract time window from angles dataframe
    2. Get proximal and distal joint angles
    3. Resample to n_points using normalized time
    """
    # Time window selection
    mask = (angles_df["timestamp"] >= window.start_time) & \
           (angles_df["timestamp"] <= window.end_time)
    stride_data = angles_df[mask].copy()

    if len(stride_data) < 10:
        raise ValueError(f"Insufficient data points in stride {window.stride_id}")

    # Extract joint angles
    proximal_col = col_map[window.leg][joint_pair[0]]
    distal_col = col_map[window.leg][joint_pair[1]]

    proximal = stride_data[proximal_col].values
    distal = stride_data[distal_col].values

    # Normalize time to 0-100%
    time_actual = stride_data["timestamp"].values
    time_norm = (time_actual - time_actual[0]) / (time_actual[-1] - time_actual[0]) * 100

    # Resample to 101 points (0%, 1%, ..., 100%)
    from scipy.interpolate import interp1d
    time_target = np.linspace(0, 100, n_points)

    interp_proximal = interp1d(time_norm, proximal, kind='cubic',
                               fill_value='extrapolate')
    interp_distal = interp1d(time_norm, distal, kind='cubic',
                            fill_value='extrapolate')

    proximal_resampled = interp_proximal(time_target)
    distal_resampled = interp_distal(time_target)

    return CyclogramLoop(
        leg=window.leg,
        stride_id=window.stride_id,
        joint_pair=joint_pair,
        proximal=proximal_resampled,
        distal=distal_resampled,
        time_normalized=time_target,
        duration=window.duration,
        speed=window.speed
    )
```

**Key Details:**
- Resampling: Cubic spline interpolation
- Target points: 101 (for 0-100% inclusive)
- Edge handling: Extrapolation for edge cases

#### 3.3.2 Joint Pair Definitions
```python
JOINT_PAIRS = [
    ("hip", "knee"),      # Hip flexion vs Knee flexion
    ("knee", "ankle"),    # Knee flexion vs Ankle dorsiflexion
    ("hip", "ankle")      # Hip flexion vs Ankle dorsiflexion
]
```

---

### 3.4 Stride Pairing Module

#### 3.4.1 StridePairer
**Purpose:** Match left and right strides occurring at similar times.

**Algorithm:**
```python
def pair_strides(left_windows: List[StrideWindow],
                right_windows: List[StrideWindow],
                tolerance_ratio: float = 0.15) -> List[PairedStrides]:
    """
    For each left stride:
    1. Calculate mid-time: (start + end) / 2
    2. Find right stride with closest mid-time
    3. Accept pair if time_diff < tolerance * left_duration
    """
    pairs = []
    used_right = set()

    for left_win in left_windows:
        left_mid = (left_win.start_time + left_win.end_time) / 2
        tolerance = tolerance_ratio * left_win.duration

        best_right = None
        min_diff = float('inf')

        for right_win in right_windows:
            if right_win.stride_id in used_right:
                continue

            right_mid = (right_win.start_time + right_win.end_time) / 2
            diff = abs(left_mid - right_mid)

            if diff < tolerance and diff < min_diff:
                best_right = right_win
                min_diff = diff

        if best_right:
            pairs.append(PairedStrides(
                left_stride=left_win,
                right_stride=best_right,
                time_difference=min_diff,
                temporal_overlap=calculate_overlap(left_win, best_right)
            ))
            used_right.add(best_right.stride_id)

    return pairs
```

**Parameters:**
- `tolerance_ratio`: 0.15 (15% of left stride duration)

**Handling Unpaired Strides:**
- Keep for per-leg visualizations
- Exclude from LR comparison metrics
- Log count of unpaired strides per leg

---

### 3.5 Metrics Calculation Module

#### 3.5.1 AreaCalculator
**Purpose:** Compute signed polygon area using shoelace formula.

**Algorithm:**
```python
def calculate_loop_area(loop: CyclogramLoop) -> float:
    """
    Shoelace formula: A = 0.5 * Σ(x_i * y_{i+1} - x_{i+1} * y_i)
    """
    x = loop.proximal
    y = loop.distal

    # Close the loop
    x_closed = np.append(x, x[0])
    y_closed = np.append(y, y[0])

    area = 0.5 * np.sum(x_closed[:-1] * y_closed[1:] -
                        x_closed[1:] * y_closed[:-1])

    return area  # Signed area (CW negative, CCW positive)

def calculate_delta_area_percent(area_L: float, area_R: float) -> float:
    """
    Normalized area difference: 200 * (AL - AR) / (|AL| + |AR|)
    Range: [-200, 200] where 0 = perfect symmetry
    """
    denominator = abs(area_L) + abs(area_R)
    if denominator < 1e-6:  # Avoid division by zero
        return 0.0
    return 200.0 * (area_L - area_R) / denominator
```

#### 3.5.2 RMSECalculator
**Purpose:** Compute shape distance after normalization.

**Algorithm:**
```python
def calculate_rmse(loop_L: CyclogramLoop,
                  loop_R: CyclogramLoop) -> float:
    """
    1. Z-score normalize each loop independently
    2. Compute Euclidean distance at each time point
    3. Return RMSE
    """
    # Normalize proximal and distal separately
    def zscore_normalize(arr):
        return (arr - np.mean(arr)) / (np.std(arr) + 1e-8)

    L_norm = np.column_stack([
        zscore_normalize(loop_L.proximal),
        zscore_normalize(loop_L.distal)
    ])

    R_norm = np.column_stack([
        zscore_normalize(loop_R.proximal),
        zscore_normalize(loop_R.distal)
    ])

    # Point-wise Euclidean distance
    distances = np.linalg.norm(L_norm - R_norm, axis=1)
    rmse = np.sqrt(np.mean(distances ** 2))

    return rmse
```

#### 3.5.3 ProcrustesCalculator
**Purpose:** Compute optimal rigid alignment distance.

**Algorithm:**
```python
def procrustes_distance(loop_L: CyclogramLoop,
                       loop_R: CyclogramLoop) -> float:
    """
    Procrustes analysis: optimal translation + scaling + rotation
    Returns residual distance after alignment.
    """
    from scipy.spatial import procrustes

    X = loop_L.points  # (101, 2)
    Y = loop_R.points  # (101, 2)

    # Procrustes returns (standardized_X, standardized_Y, disparity)
    _, _, disparity = procrustes(X, Y)

    return disparity
```

**Note:** Uses SciPy's `procrustes` which:
1. Translates both shapes to origin (center of mass)
2. Scales to unit variance
3. Rotates for minimum squared distance
4. Returns disparity (sum of squared distances)

#### 3.5.4 OrientationCalculator
**Purpose:** Extract major axis orientation via PCA.

**Algorithm:**
```python
def calculate_orientation(loop: CyclogramLoop) -> float:
    """
    PCA on loop points, return angle of first principal component.
    """
    from sklearn.decomposition import PCA

    points = loop.points  # (101, 2)

    # Center the data
    centered = points - np.mean(points, axis=0)

    # PCA
    pca = PCA(n_components=1)
    pca.fit(centered)

    # Principal component direction
    pc1 = pca.components_[0]

    # Angle in degrees
    angle = np.arctan2(pc1[1], pc1[0]) * 180 / np.pi

    return angle  # Range: [-180, 180]

def calculate_delta_orientation(angle_L: float, angle_R: float) -> float:
    """
    Minimum angular difference (handles wrap-around).
    """
    diff = abs(angle_L - angle_R)
    if diff > 180:
        diff = 360 - diff
    return diff
```

#### 3.5.5 HysteresisDetector
**Purpose:** Determine loop traversal direction.

**Algorithm:**
```python
def detect_hysteresis(loop: CyclogramLoop) -> str:
    """
    Use signed area to determine direction.
    Positive area = CCW, Negative area = CW
    """
    area = calculate_loop_area(loop)
    return "CCW" if area > 0 else "CW"
```

#### 3.5.6 SimilarityScorer
**Purpose:** Compute composite LR similarity score (0-100).

**Algorithm:**
```python
def calculate_similarity_score(metrics: CyclogramMetrics,
                              weight_area: float = 0.30,
                              weight_procrustes: float = 0.30,
                              weight_rmse: float = 0.30,
                              weight_orientation: float = 0.10) -> float:
    """
    Composite score from normalized metrics.
    100 = perfect symmetry, 0 = maximum asymmetry

    Steps:
    1. Normalize each metric to [0, 1] (0=bad, 1=good)
    2. Weight and combine
    3. Scale to 0-100
    """
    # Normalize delta_area_percent: |ΔA%| → [0, 1]
    # Assume 0% = perfect (1.0), 50% = poor (0.0)
    area_norm = max(0, 1 - abs(metrics.delta_area_percent) / 50.0)

    # Normalize Procrustes: 0 = perfect (1.0), >0.5 = poor (0.0)
    procrustes_norm = max(0, 1 - metrics.procrustes_distance / 0.5)

    # Normalize RMSE: 0 = perfect (1.0), >1.0 = poor (0.0)
    rmse_norm = max(0, 1 - metrics.rmse / 1.0)

    # Normalize orientation: 0° = perfect (1.0), 30° = poor (0.0)
    orient_norm = max(0, 1 - metrics.delta_orientation / 30.0)

    # Weighted combination
    score = (weight_area * area_norm +
             weight_procrustes * procrustes_norm +
             weight_rmse * rmse_norm +
             weight_orientation * orient_norm) * 100

    return np.clip(score, 0, 100)
```

**Configurable Weights:**
```python
DEFAULT_WEIGHTS = {
    "area": 0.30,
    "procrustes": 0.30,
    "rmse": 0.30,
    "orientation": 0.10
}
```

---

### 3.6 Visualization Module

#### 3.6.1 OverlayPlotter
**Purpose:** Generate multi-stride overlay plots with statistics.

**Specification:**
```python
def plot_overlayed_cyclograms(loops_L: List[CyclogramLoop],
                              loops_R: List[CyclogramLoop],
                              metrics: List[CyclogramMetrics],
                              joint_pair: Tuple[str, str],
                              output_path: str) -> None:
    """
    Creates figure with:
    - All left strides (blue, alpha=0.25)
    - All right strides (red, alpha=0.25)
    - Mean left loop (blue, linewidth=2.5)
    - Mean right loop (red, linewidth=2.5)
    - Statistics text box
    """
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

    # Plot individual strides
    for loop in loops_L:
        ax.plot(loop.proximal, loop.distal,
                color='blue', alpha=0.25, linewidth=0.5)

    for loop in loops_R:
        ax.plot(loop.proximal, loop.distal,
                color='red', alpha=0.25, linewidth=0.5)

    # Calculate and plot means
    mean_L = np.mean([loop.points for loop in loops_L], axis=0)
    mean_R = np.mean([loop.points for loop in loops_R], axis=0)

    ax.plot(mean_L[:, 0], mean_L[:, 1],
            color='blue', linewidth=2.5, label='Left Mean')
    ax.plot(mean_R[:, 0], mean_R[:, 1],
            color='red', linewidth=2.5, label='Right Mean')

    # Statistics text box
    stats_text = format_statistics(metrics, len(loops_L), len(loops_R))
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Formatting
    ax.set_xlabel(f"{joint_pair[0].capitalize()} Angle (deg)", fontsize=12)
    ax.set_ylabel(f"{joint_pair[1].capitalize()} Angle (deg)", fontsize=12)
    ax.set_title(f"Cyclogram: {joint_pair[0].capitalize()}-{joint_pair[1].capitalize()}",
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def format_statistics(metrics: List[CyclogramMetrics],
                     n_left: int, n_right: int) -> str:
    """Format statistics text box content."""
    delta_area_values = [m.delta_area_percent for m in metrics]
    procrustes_values = [m.procrustes_distance for m in metrics]
    similarity_values = [m.similarity_score for m in metrics]

    return f"""n(Left): {n_left}
n(Right): {n_right}
ΔArea%: {np.mean(delta_area_values):.1f}±{np.std(delta_area_values):.1f}
Procrustes: {np.mean(procrustes_values):.3f}
Similarity: {np.mean(similarity_values):.1f}±{np.std(similarity_values):.1f}"""
```

#### 3.6.2 Per-Leg Plotter (Optional)
**Purpose:** Generate separate left-only and right-only views.

**Algorithm:**
```python
def plot_single_leg_cyclograms(loops: List[CyclogramLoop],
                               leg: str,
                               joint_pair: Tuple[str, str],
                               output_path: str,
                               axes_limits: Tuple[float, float, float, float]) -> None:
    """
    Same as overlay plot but single leg.
    Uses shared axes limits for fair comparison.
    """
    color = 'blue' if leg == 'L' else 'red'

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

    for loop in loops:
        ax.plot(loop.proximal, loop.distal,
                color=color, alpha=0.3, linewidth=0.5)

    mean_loop = np.mean([loop.points for loop in loops], axis=0)
    ax.plot(mean_loop[:, 0], mean_loop[:, 1],
            color=color, linewidth=2.5, label=f'{leg} Mean')

    ax.set_xlim(axes_limits[0], axes_limits[1])
    ax.set_ylim(axes_limits[2], axes_limits[3])

    # ... formatting ...

    plt.savefig(output_path)
    plt.close()
```

#### 3.6.3 Similarity Dashboard
**Purpose:** Bar chart of per-pair similarity scores.

**Algorithm:**
```python
def plot_similarity_summary(session_summary: pd.DataFrame,
                           output_path: str) -> None:
    """
    Bar chart with error bars showing:
    - X-axis: Joint pairs
    - Y-axis: Similarity score (0-100)
    - Error bars: ±1 SD
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    x = np.arange(len(session_summary))
    means = session_summary['similarity_mean'].values
    stds = session_summary['similarity_std'].values

    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=['#1f77b4', '#ff7f0e', '#2ca02c'],
                  alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(session_summary['joint_pair'].values, fontsize=12)
    ax.set_ylabel('LR Similarity Score', fontsize=12)
    ax.set_ylim(0, 100)
    ax.axhline(90, color='green', linestyle='--', alpha=0.5, label='Excellent (>90)')
    ax.axhline(75, color='orange', linestyle='--', alpha=0.5, label='Good (>75)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.title('Left-Right Similarity by Joint Pair', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
```

#### 3.6.4 Distribution Plots (Optional)
**Purpose:** Violin/box plots for metric distributions.

**Algorithm:**
```python
def plot_metric_distributions(stride_metrics: pd.DataFrame,
                              output_path: str) -> None:
    """
    Create 2x2 subplot grid:
    1. Delta Area % distribution
    2. Procrustes distance distribution
    3. RMSE distribution
    4. Orientation difference distribution
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)

    for pair in stride_metrics['joint_pair'].unique():
        pair_data = stride_metrics[stride_metrics['joint_pair'] == pair]

        # Violin plots for each metric
        axes[0, 0].violinplot(pair_data['delta_area_pct'], ...)
        axes[0, 1].violinplot(pair_data['procrustes'], ...)
        axes[1, 0].violinplot(pair_data['rmse'], ...)
        axes[1, 1].violinplot(pair_data['delta_orient'], ...)

    # ... formatting ...

    plt.savefig(output_path)
    plt.close()
```

---

### 3.7 Output Serialization Module

#### 3.7.1 MetricsWriter
**Purpose:** Export metrics to CSV files.

**Algorithm:**
```python
def write_stride_metrics(metrics: List[CyclogramMetrics],
                        output_path: str) -> None:
    """
    Write per-stride metrics CSV.
    """
    rows = []
    for m in metrics:
        rows.append({
            'stride_id_L': m.left_stride_id,
            'stride_id_R': m.right_stride_id,
            'joint_pair': f"{m.joint_pair[0]}-{m.joint_pair[1]}",
            'area_L': m.area_left,
            'area_R': m.area_right,
            'delta_area_pct': m.delta_area_percent,
            'rmse': m.rmse,
            'procrustes': m.procrustes_distance,
            'delta_orient': m.delta_orientation,
            'hysteresis_L': m.hysteresis_left,
            'hysteresis_R': m.hysteresis_right,
            'hysteresis_mismatch': m.hysteresis_mismatch,
            'similarity_score': m.similarity_score
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, float_format='%.4f')

def write_session_summary(metrics: List[CyclogramMetrics],
                         output_path: str) -> None:
    """
    Aggregate metrics by joint pair.
    """
    df = pd.DataFrame([{
        'stride_id_L': m.left_stride_id,
        'stride_id_R': m.right_stride_id,
        'joint_pair': f"{m.joint_pair[0]}-{m.joint_pair[1]}",
        'delta_area_pct': m.delta_area_percent,
        'procrustes': m.procrustes_distance,
        'similarity_score': m.similarity_score
    } for m in metrics])

    summary = df.groupby('joint_pair').agg({
        'stride_id_L': 'count',  # n_pairs
        'delta_area_pct': ['mean', 'std'],
        'procrustes': ['mean', 'std'],
        'similarity_score': ['mean', 'std']
    }).round(4)

    summary.columns = ['n_pairs', 'delta_area_mean', 'delta_area_std',
                      'procrustes_mean', 'procrustes_std',
                      'similarity_mean', 'similarity_std']

    summary.to_csv(output_path)
```

---

## 4. Execution Pipeline

### 4.1 Main Workflow

```python
def run_cyclogram_analysis(config: CyclogramConfig) -> None:
    """
    Main execution pipeline.
    """
    # 1. Load and validate data
    angles_df = load_csv(config.raw_angles_path)
    events_df = load_csv(config.events_path)

    col_map = detect_columns(angles_df)
    angles_df, events_df = normalize_timebase(angles_df, events_df, config.fps)

    # 2. Preprocess angles
    angle_cols = [col for leg in col_map.values() for col in leg.values()]
    angles_df = preprocess_angles(angles_df, angle_cols,
                                  config.smooth_window,
                                  config.smooth_poly,
                                  config.smooth_threshold)

    # 3. Segment strides
    left_windows = build_stride_windows(events_df, "L",
                                       config.min_stride_duration,
                                       config.max_stride_duration)
    right_windows = build_stride_windows(events_df, "R",
                                        config.min_stride_duration,
                                        config.max_stride_duration)

    # 4. Extract cyclograms
    all_loops_L = []
    all_loops_R = []

    for pair in config.joint_pairs:
        for window in left_windows:
            loop = extract_cyclogram(angles_df, window, pair, col_map)
            all_loops_L.append(loop)

        for window in right_windows:
            loop = extract_cyclogram(angles_df, window, pair, col_map)
            all_loops_R.append(loop)

    # 5. Pair strides
    paired_strides = pair_strides(left_windows, right_windows,
                                  config.pairing_tolerance)

    # 6. Calculate metrics
    all_metrics = []
    for pair_obj in paired_strides:
        for joint_pair in config.joint_pairs:
            # Get corresponding loops
            loop_L = [l for l in all_loops_L
                     if l.stride_id == pair_obj.left_stride.stride_id
                     and l.joint_pair == joint_pair][0]
            loop_R = [l for l in all_loops_R
                     if l.stride_id == pair_obj.right_stride.stride_id
                     and l.joint_pair == joint_pair][0]

            # Compute metrics
            metrics = compute_all_metrics(loop_L, loop_R, joint_pair)
            all_metrics.append(metrics)

    # 7. Visualize
    for joint_pair in config.joint_pairs:
        pair_loops_L = [l for l in all_loops_L if l.joint_pair == joint_pair]
        pair_loops_R = [l for l in all_loops_R if l.joint_pair == joint_pair]
        pair_metrics = [m for m in all_metrics if m.joint_pair == joint_pair]

        # Overlay plot
        plot_overlayed_cyclograms(
            pair_loops_L, pair_loops_R, pair_metrics, joint_pair,
            config.output_dir / f"CK_{joint_pair[0]}-{joint_pair[1]}_AllStrides.png"
        )

        # Per-leg plots (optional)
        if config.generate_per_leg_plots:
            plot_single_leg_cyclograms(pair_loops_L, "L", joint_pair, ...)
            plot_single_leg_cyclograms(pair_loops_R, "R", joint_pair, ...)

    # Summary dashboard
    plot_similarity_summary(all_metrics, config.output_dir / "LR_Similarity_Summary.png")

    # 8. Export metrics
    write_stride_metrics(all_metrics, config.output_dir / "cyclogram_stride_metrics.csv")
    write_session_summary(all_metrics, config.output_dir / "cyclogram_session_summary.csv")
```

### 4.2 Configuration Dataclass

```python
@dataclass
class CyclogramConfig:
    """Centralized configuration for cyclogram analysis."""

    # Input paths
    raw_angles_path: Path
    events_path: Path
    stride_stats_path: Optional[Path] = None

    # Output paths
    output_dir: Path = Path("plots/cyclograms")
    export_dir: Path = Path("exports")

    # Processing parameters
    fps: Optional[float] = None
    smooth_window: int = 11
    smooth_poly: int = 2
    smooth_threshold: float = 5.0

    # Stride segmentation
    min_stride_duration: float = 0.3
    max_stride_duration: float = 3.0

    # Stride pairing
    pairing_tolerance: float = 0.15

    # Cyclogram extraction
    n_resample_points: int = 101
    joint_pairs: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("hip", "knee"),
        ("knee", "ankle"),
        ("hip", "ankle")
    ])

    # Metrics weights
    similarity_weights: Dict[str, float] = field(default_factory=lambda: {
        "area": 0.30,
        "procrustes": 0.30,
        "rmse": 0.30,
        "orientation": 0.10
    })

    # Visualization
    generate_per_leg_plots: bool = False
    generate_distribution_plots: bool = True
    plot_dpi: int = 300

    # Outlier handling
    outlier_percentile: float = 1.0  # Remove top/bottom 1% by area
```

---

## 5. Edge Cases & Robustness

### 5.1 Data Quality Issues

| Issue | Detection | Handling |
|-------|-----------|----------|
| Missing events | Check stride completion | Drop incomplete strides, log warning |
| Jittery angles | Compute std dev | Apply smoothing only if std > threshold |
| Zero/360 jumps | Check for discontinuities | Unwrap angles |
| Outlier strides | Duration/area thresholds | Exclude from analysis |
| Sparse data | Check points per stride | Require minimum 10 points |

### 5.2 Algorithm Robustness

#### 5.2.1 Division by Zero Protection
```python
# Area calculation
if abs(area_L) + abs(area_R) < 1e-6:
    delta_area_pct = 0.0

# Normalization
std_dev = np.std(data) + 1e-8  # Prevent division by zero
```

#### 5.2.2 Interpolation Edge Cases
```python
# Use extrapolation for edge points
interp_fn = interp1d(x, y, kind='cubic',
                     bounds_error=False,
                     fill_value='extrapolate')
```

#### 5.2.3 Empty Stride Handling
```python
if len(left_windows) < 2 or len(right_windows) < 2:
    raise ValueError("Insufficient strides for analysis (minimum 2 per leg)")
```

### 5.3 Column Name Variations

**Supported Patterns:**
```python
HIP_PATTERNS = [
    r"(?i)(L|left|lt).*(hip|flex|flexion).*(?!knee|ankle)",
    r"(?i)hip.*(L|left|lt)",
]

KNEE_PATTERNS = [
    r"(?i)(L|left|lt).*knee.*flex",
    r"(?i)knee.*(L|left|lt)",
]

ANKLE_PATTERNS = [
    r"(?i)(L|left|lt).*ankle.*(dorsi|flex)",
    r"(?i)ankle.*(L|left|lt)",
]
```

**Fallback:**
```python
if not all_columns_detected:
    raise ValueError(f"Could not auto-detect columns. Please ensure columns match patterns or provide explicit mapping.")
```

---

## 6. Performance Considerations

### 6.1 Computational Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Data loading | O(n) | n = number of frames |
| Smoothing | O(n) | Per angle column |
| Stride segmentation | O(e) | e = number of events |
| Cyclogram extraction | O(s × p) | s = strides, p = resample points |
| Stride pairing | O(s²) | Worst case, typically O(s) |
| Metrics calculation | O(s × j × p) | j = joint pairs |
| Visualization | O(s × j) | Matplotlib bottleneck |

**Expected Runtime:**
- Small dataset (500 frames, 10 strides): <1 second
- Medium dataset (5000 frames, 50 strides): ~5 seconds
- Large dataset (50000 frames, 500 strides): ~60 seconds

### 6.2 Memory Optimization

**Lazy Loading:**
```python
# Process one subject at a time for batch analysis
for subject_dir in subject_directories:
    run_cyclogram_analysis(config)
    gc.collect()  # Force garbage collection
```

**Chunked Processing:**
```python
# For very long recordings, process in time windows
time_windows = generate_time_windows(total_duration, window_size=60.0)
for window in time_windows:
    process_window(angles_df, events_df, window)
```

---

## 7. Testing Strategy

### 7.1 Unit Tests

```python
def test_area_calculation():
    """Test shoelace formula with known shapes."""
    # Square: 10×10 → area = 100
    loop = create_square_loop(side_length=10)
    assert abs(calculate_loop_area(loop) - 100) < 0.1

def test_stride_pairing():
    """Test temporal pairing logic."""
    left = [StrideWindow("L", 1, 0.0, 0.5, 0.5)]
    right = [StrideWindow("R", 1, 0.1, 0.6, 0.5)]
    pairs = pair_strides(left, right, tolerance_ratio=0.2)
    assert len(pairs) == 1
    assert pairs[0].time_difference == 0.05

def test_similarity_score_bounds():
    """Ensure score stays in [0, 100]."""
    metrics = CyclogramMetrics(...)
    score = calculate_similarity_score(metrics)
    assert 0 <= score <= 100
```

### 7.2 Integration Tests

```python
def test_end_to_end_pipeline():
    """Test full pipeline with synthetic data."""
    # Generate synthetic gait data
    angles_df, events_df = generate_synthetic_gait(n_strides=10)

    config = CyclogramConfig(
        raw_angles_path="test_angles.csv",
        events_path="test_events.csv",
        output_dir=Path("test_output")
    )

    run_cyclogram_analysis(config)

    # Verify outputs exist
    assert (config.output_dir / "CK_hip-knee_AllStrides.png").exists()
    assert (config.export_dir / "cyclogram_stride_metrics.csv").exists()
```

### 7.3 Validation Tests

```python
def test_known_asymmetry():
    """Test with artificially asymmetric data."""
    # Create left loop (normal) and right loop (20% smaller area)
    loop_L = create_loop(area=100)
    loop_R = create_loop(area=80)

    metrics = compute_all_metrics(loop_L, loop_R, ("hip", "knee"))

    # Expect ~22% area difference: 200*(100-80)/(100+80)
    assert abs(metrics.delta_area_percent - 22.2) < 1.0

    # Similarity score should reflect asymmetry
    assert metrics.similarity_score < 95
```

---

## 8. Acceptance Criteria

### 8.1 Functional Requirements

✅ **FR1:** System processes Raw_Angles.csv and Angle_Events.csv without errors
✅ **FR2:** Auto-detects column names with regex patterns
✅ **FR3:** Segments strides per leg using heel strike/toe-off events
✅ **FR4:** Generates normalized 101-point cyclograms per stride
✅ **FR5:** Pairs left-right strides by temporal proximity
✅ **FR6:** Computes all specified metrics (area, RMSE, Procrustes, orientation)
✅ **FR7:** Calculates composite similarity score (0-100)
✅ **FR8:** Generates overlayed cyclogram plots (Left=blue, Right=red)
✅ **FR9:** Exports stride-level and session-level CSV metrics
✅ **FR10:** Handles missing data and outliers gracefully

### 8.2 Quality Requirements

✅ **QR1:** Loops are closed and smooth (no jagged edges)
✅ **QR2:** Mean loops are visibly centered in overlays
✅ **QR3:** Left vs Right overlays share axes for fair comparison
✅ **QR4:** Statistics text box displays all required metrics
✅ **QR5:** Metrics stable across repeated runs (±1% variance)
✅ **QR6:** Removing outlier strides doesn't flip conclusions
✅ **QR7:** Code is modular and maintainable (single responsibility)
✅ **QR8:** Configuration is centralized and adjustable

### 8.3 Visual Requirements

✅ **VR1:** Left strides always rendered in blue (RGB: 31, 119, 180)
✅ **VR2:** Right strides always rendered in red (RGB: 255, 127, 14)
✅ **VR3:** Individual stride alpha = 0.25, mean loop linewidth = 2.5
✅ **VR4:** Equal aspect ratio for all cyclogram plots
✅ **VR5:** Grid enabled with 30% alpha
✅ **VR6:** Axes labels include units (deg)
✅ **VR7:** Title clearly identifies joint pair
✅ **VR8:** Statistics box positioned top-left, non-overlapping

---

## 9. Future Enhancements

### 9.1 Short-Term (Next Version)

1. **Swing Phase Cyclograms:** Add toe_off → heel_strike analysis
2. **Speed Correlation:** Analyze metric variation with walking speed
3. **Interactive Plots:** Bokeh/Plotly for hover tooltips
4. **Batch Processing:** Multi-subject pipeline with summary reports

### 9.2 Medium-Term

1. **Machine Learning:** Classify gait abnormalities from cyclogram features
2. **Temporal Analysis:** Stride-to-stride variability metrics
3. **3D Cyclograms:** Three-joint phase space visualization
4. **Real-time Processing:** Streaming data support

### 9.3 Long-Term

1. **Clinical Integration:** FHIR-compatible output formats
2. **Normative Database:** Age/sex-matched reference comparisons
3. **Mobile App:** On-device analysis for wearable sensors
4. **AI Diagnostics:** Deep learning for pathology detection

---

## 10. References

### 10.1 Theoretical Background

1. **Cyclogram Theory:**
   - Goswami, A. (1998). "A new gait parameterization technique by means of cyclogram moments." *Gait & Posture*, 8(1), 15-36.
   - Hershler, C., & Milner, M. (1980). "Angle–angle diagrams in the assessment of locomotion." *American Journal of Physical Medicine*, 59(3), 109-125.

2. **Procrustes Analysis:**
   - Gower, J. C., & Dijksterhuis, G. B. (2004). *Procrustes Problems*. Oxford University Press.
   - Kendall, D. G. (1989). "A survey of the statistical theory of shape." *Statistical Science*, 4(2), 87-99.

3. **Gait Analysis:**
   - Perry, J., & Burnfield, J. M. (2010). *Gait Analysis: Normal and Pathological Function*. SLACK Incorporated.

### 10.2 Software Libraries

- **NumPy:** Array operations and numerical computing
- **Pandas:** Data manipulation and CSV I/O
- **SciPy:** Signal processing (Savitzky-Golay), Procrustes analysis
- **Matplotlib:** Static visualization
- **scikit-learn:** PCA for orientation analysis

---

## Appendix A: Example Output Files

### A.1 cyclogram_stride_metrics.csv (excerpt)
```csv
stride_id_L,stride_id_R,joint_pair,area_L,area_R,delta_area_pct,rmse,procrustes,delta_orient,hysteresis_L,hysteresis_R,hysteresis_mismatch,similarity_score
1,1,hip-knee,245.32,238.74,2.73,0.15,0.08,3.24,CCW,CCW,False,92.5
1,1,knee-ankle,189.43,195.12,-3.01,0.22,0.12,5.13,CW,CW,False,88.3
1,1,hip-ankle,412.87,419.53,-1.61,0.18,0.09,2.87,CCW,CCW,False,91.7
2,2,hip-knee,242.18,243.91,-0.71,0.11,0.06,2.15,CCW,CCW,False,95.2
...
```

### A.2 cyclogram_session_summary.csv
```csv
joint_pair,n_pairs,delta_area_mean,delta_area_std,procrustes_mean,procrustes_std,similarity_mean,similarity_std
hip-knee,15,2.1,1.8,0.09,0.04,91.2,3.5
knee-ankle,15,-1.5,2.3,0.14,0.06,87.8,4.2
hip-ankle,15,3.2,2.9,0.11,0.05,89.5,3.9
```

---

## Appendix B: Configuration Template

```python
# config.py
from pathlib import Path
from cyclogram_config import CyclogramConfig

config = CyclogramConfig(
    # Input data
    raw_angles_path=Path("DATA-Front View/Subject_001/Raw_Angles.csv"),
    events_path=Path("DATA-Front View/Subject_001/Angle_Events.csv"),
    stride_stats_path=Path("DATA-Front View/Subject_001/Angle_Stride_Stats.csv"),

    # Output directories
    output_dir=Path("plots/cyclograms"),
    export_dir=Path("exports"),

    # Processing parameters
    fps=30.0,  # Only needed if timestamp column missing
    smooth_window=11,
    smooth_poly=2,
    smooth_threshold=5.0,

    # Stride constraints
    min_stride_duration=0.3,
    max_stride_duration=3.0,

    # Pairing tolerance
    pairing_tolerance=0.15,

    # Cyclogram settings
    n_resample_points=101,
    joint_pairs=[
        ("hip", "knee"),
        ("knee", "ankle"),
        ("hip", "ankle")
    ],

    # Similarity scoring weights
    similarity_weights={
        "area": 0.30,
        "procrustes": 0.30,
        "rmse": 0.30,
        "orientation": 0.10
    },

    # Visualization options
    generate_per_leg_plots=False,
    generate_distribution_plots=True,
    plot_dpi=300,

    # Outlier handling
    outlier_percentile=1.0
)
```

---

## Document Control

**Version:** 1.0
**Date:** 2025-10-10
**Status:** Final Design Specification
**Author:** Claude (Anthropic)
**Reviewed By:** Pending

**Revision History:**
- v1.0 (2025-10-10): Initial comprehensive design specification

**Approval:**
- [ ] Technical Review
- [ ] Stakeholder Approval
- [ ] Ready for Implementation
