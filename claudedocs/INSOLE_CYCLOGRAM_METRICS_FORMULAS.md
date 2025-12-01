# Insole Cyclogram Metrics: Formulas and Methods

**Complete documentation for MARP, aggregate mean relative phase, aggregate coupling angle variability, and curve phase variability index difference calculations in insole-analysis.py**

---

## Table of Contents

1. [Overview](#overview)
2. [MARP (Mean Absolute Relative Phase)](#marp-mean-absolute-relative-phase)
3. [Mean Relative Phase](#mean-relative-phase)
4. [Coupling Angle Variability](#coupling-angle-variability)
5. [Curvature Phase Variability Index Difference](#curvature-phase-variability-index-difference)
6. [Aggregate Metrics](#aggregate-metrics)
7. [Implementation Location](#implementation-location)

---

## Overview

The insole-analysis.py script computes advanced cyclogram metrics across three hierarchical levels:

- **Level 1 (Geometric)**: Shape and morphology metrics
- **Level 2 (Temporal-Coupling)**: Phase coordination metrics (includes MARP)
- **Level 3 (Bilateral Symmetry)**: Left-right comparison metrics

This document focuses on Level 2 temporal metrics and bilateral symmetry metrics related to phase coupling and curvature variability.

---

## MARP (Mean Absolute Relative Phase)

### Definition
MARP quantifies the average magnitude of phase difference between two oscillating signals (e.g., gyroscope X vs Y axes), regardless of direction. It measures **coordination stability** in gait patterns.

### Formula
```
MARP = mean(|CRP|)
```

where:
- `CRP` = Continuous Relative Phase (instantaneous phase difference array)
- `| |` = absolute value
- `mean()` = arithmetic mean

### Mathematical Expansion
```
CRP = φ_x(t) - φ_y(t)
CRP_wrapped = angle(exp(i · CRP))    # Wrap to [-π, π]
MARP = (1/N) · Σ|CRP_wrapped[i]|     # i = 0 to N-1
```

### Computational Steps (insole-analysis.py:316-341)

**Step 1: Compute phase angles using Hilbert transform**
```python
def _compute_phase_angle(signal: np.ndarray) -> np.ndarray:
    """
    Compute phase angle using Hilbert transform.

    Args:
        signal: 1D time series data (e.g., gyro_x)

    Returns:
        Phase angle array in radians
    """
    from scipy.signal import hilbert

    # Remove DC component (mean centering)
    analytic_signal = hilbert(signal - np.mean(signal))

    # Extract instantaneous phase
    phase = np.angle(analytic_signal)  # Returns values in [-π, π]

    return phase
```

**Step 2: Calculate Continuous Relative Phase (CRP)**
```python
# Compute phase for both signals
phase_x = _compute_phase_angle(self.x)  # e.g., gyro_x
phase_y = _compute_phase_angle(self.y)  # e.g., gyro_y

# Calculate phase difference
crp = phase_x - phase_y

# Wrap to [-π, π] to handle phase wrapping
crp = np.angle(np.exp(1j * crp))
```

**Step 3: Calculate MARP**
```python
# Mean Absolute Relative Phase
marp = np.mean(np.abs(crp))
```

### Interpretation
- **Range**: 0 to π radians (0° to 180°)
- **Low MARP (< 0.5 rad ≈ 30°)**: Strong in-phase or anti-phase coordination
- **Moderate MARP (0.5-1.5 rad)**: Typical gait coordination
- **High MARP (> 1.5 rad ≈ 85°)**: Poor coordination, unstable gait

### Clinical Significance
- **Gait stability**: Lower MARP indicates more consistent joint coordination
- **Asymmetry detection**: Compare left vs right MARP to identify unilateral instability
- **Rehabilitation tracking**: MARP should decrease as gait improves

---

## Mean Relative Phase

### Definition
The **directional average** of phase difference between two signals. Unlike MARP (which uses absolute values), this preserves the **leading/lagging relationship**.

### Formula
```
Mean Relative Phase = mean(CRP)
```

where `CRP` is the wrapped continuous relative phase (see MARP section).

### Computational Implementation (insole-analysis.py:340)
```python
metrics['mean_relative_phase'] = np.mean(crp)
```

### Interpretation
- **Range**: -π to +π radians (-180° to +180°)
- **Positive values**: Signal X leads signal Y (phase advance)
- **Negative values**: Signal X lags signal Y (phase delay)
- **Near zero**: In-phase coordination
- **Near ±π**: Anti-phase coordination

### Example
```
If gyro_x leads gyro_y by consistent 45°:
mean_relative_phase ≈ +0.785 rad (π/4)

If signals are perfectly in-phase:
mean_relative_phase ≈ 0.0 rad
```

### Clinical Use
- **Phase shift detection**: Identifies temporal lag between joint movements
- **Bilateral comparison**: Left vs right phase shift indicates asymmetric timing
- **Complements MARP**: Use together to understand both magnitude AND direction of coupling

---

## Coupling Angle Variability

### Definition
Measures the **stability of the directional relationship** between two cyclogram signals (X and Y trajectories). It quantifies how much the **orientation** of the velocity vector varies throughout the gait cycle.

### Formula
```
Coupling Angle Variability = std(θ_coupling)
```

where:
```
θ_coupling(t) = arctan2(dY/dt, dX/dt)
```

### Mathematical Details
```
θ_coupling = arctan2(∇Y, ∇X)

where:
  ∇X = gradient(X)  # Discrete derivative (velocity in X direction)
  ∇Y = gradient(Y)  # Discrete derivative (velocity in Y direction)
```

### Computational Implementation (insole-analysis.py:343-345)
```python
# Compute coupling angle at each time point
coupling_angle = np.arctan2(np.gradient(self.y), np.gradient(self.x))

# Variability is standard deviation
metrics['coupling_angle_variability'] = np.std(coupling_angle)
```

### Interpretation
- **Range**: 0 to ~1.8 radians (0° to ~100°)
- **Low variability (< 0.3 rad)**: Highly consistent directional coupling
- **Moderate variability (0.3-0.8 rad)**: Normal gait coordination
- **High variability (> 0.8 rad)**: Unstable, erratic coordination patterns

### Physical Meaning
The coupling angle represents the **instantaneous direction** of the cyclogram trajectory:
- **0°**: Moving purely in +X direction
- **90°**: Moving purely in +Y direction
- **Variability**: How much this direction fluctuates (stability measure)

### Clinical Significance
- **Motor control**: Lower variability indicates better neuromuscular control
- **Pathology detection**: High variability may indicate:
  - Neurological impairment
  - Compensation strategies
  - Pain-related gait modifications
- **Rehabilitation**: Should decrease with recovery

---

## Curvature Phase Variability Index Difference

### Context
This metric is part of **Section F: Curvature Phase-Variation Metrics**, which analyzes how curvature changes throughout the gait cycle phase.

### Definition
The **bilateral asymmetry** in curvature consistency between left and right legs.

### Formula
```
curv_phase_variability_index_diff = |VI_left - VI_right|
```

where **Variability Index (VI)** for each leg is:
```
VI = std(|κ|) / mean(|κ|)
```

and **κ (curvature)** is:
```
κ(t) = (X'Y'' - Y'X'') / (X'² + Y'²)^(3/2)
```

### Computational Pipeline

**Step 1: Compute curvature for each leg** (insole-analysis.py:290-312)
```python
def _compute_curvature(self) -> np.ndarray:
    """
    Compute trajectory curvature κ = (x'y'' - y'x'') / (x'² + y'²)^(3/2).

    Returns:
        Array of curvature values
    """
    # First derivatives (velocities)
    dx = np.gradient(self.x)
    dy = np.gradient(self.y)

    # Second derivatives (accelerations)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # Curvature formula
    numerator = dx * ddy - dy * ddx
    denominator = (dx**2 + dy**2)**(3/2)

    # Avoid division by zero
    curvature = np.where(denominator > 1e-10, numerator / denominator, 0)

    return curvature
```

**Step 2: Calculate Variability Index for each leg**
```python
# From geometric metrics computation (insole-analysis.py:236-238)
curvature = self._compute_curvature()
mean_curvature = np.mean(np.abs(curvature))
curvature_std = np.std(curvature)

# Variability Index (coefficient of variation)
variability_index = curvature_std / mean_curvature if mean_curvature > 0 else 0
```

**Step 3: Compute bilateral difference**
```python
# This is computed during bilateral symmetry analysis
# (see claudedocs/SECTION_F_METRICS_REFERENCE.md lines 151-158)

curv_phase_variability_index_diff = abs(VI_left - VI_right)
```

### Interpretation
- **Range**: 0 to 2+ (absolute difference in coefficients of variation)
- **< 0.1**: Symmetric curvature variability (excellent)
- **0.1-0.3**: Mild asymmetry (normal variation)
- **> 0.3**: One leg significantly more variable (pathological)

### Physical Meaning
- **Curvature**: How sharply the cyclogram trajectory bends
- **Variability Index**: How consistent that curvature is throughout the cycle
- **Difference**: Asymmetry in gait smoothness/predictability between legs

### Clinical Applications
- **Post-injury assessment**: Injured leg often shows higher VI (compensatory variability)
- **Fall risk**: High bilateral difference indicates asymmetric motor control
- **Rehabilitation tracking**: Should decrease toward zero with recovery

### Related Metrics (Section F)
See `claudedocs/SECTION_F_METRICS_REFERENCE.md` for complete documentation on:
- `curv_phase_rms_diff`: Magnitude asymmetry
- `curv_phase_circular_corr`: Pattern similarity
- `curv_phase_peak_phase_diff_%`: Timing asymmetry
- Phase-binned curvature profiles (10 bins per leg)

---

## Aggregate Metrics

### Definition
**Aggregate metrics** are summary statistics computed across multiple gait cycles, grouped by cyclogram type and leg. They provide population-level insights from individual cycle measurements.

### Computation Location
File: `insole-analysis.py:4691-4709`

### Implementation
```python
# Group by cyclogram type and leg
aggregate = df_metrics.groupby(['cyclogram_type', 'leg']).agg({
    'area': ['mean', 'std'],
    'compactness_ratio': ['mean', 'std'],
    'aspect_ratio': ['mean', 'std'],
    'eccentricity': ['mean', 'std'],
    'mean_curvature': ['mean', 'std'],
    'trajectory_smoothness': ['mean', 'std'],
    'mean_relative_phase': ['mean', 'std'],          # ← AGGREGATE
    'marp': ['mean', 'std'],                         # ← AGGREGATE
    'coupling_angle_variability': ['mean', 'std']    # ← AGGREGATE
}).round(4)

# Save to CSV
aggregate_path = output_dir / 'cyclogram_metrics_aggregate.csv'
aggregate.to_csv(aggregate_path)
```

### Output Structure
**File**: `cyclogram_metrics_aggregate.csv`

**Row index**: Multi-level (cyclogram_type, leg)
- Example: `(gyro_X_vs_Y, left)`, `(gyro_X_vs_Y, right)`, `(acc_X_vs_Y, left)`, etc.

**Columns**: Hierarchical (metric, statistic)
```
| cyclogram_type | leg   | mean_relative_phase |      | marp   |      | coupling_angle_variability |      |
|                |       | mean       | std   | mean   | std  | mean              | std   |
|----------------|-------|------------|-------|--------|------|-------------------|-------|
| gyro_X_vs_Y    | left  | 0.1234     | 0.0567| 0.8901 | 0.123| 0.4567            | 0.089 |
| gyro_X_vs_Y    | right | 0.1456     | 0.0623| 0.9123 | 0.145| 0.4789            | 0.098 |
| acc_X_vs_Y     | left  | 0.2341     | 0.0789| 1.0234 | 0.178| 0.5678            | 0.112 |
...
```

### Aggregate Statistics Explained

**For each metric, two statistics are computed:**

1. **Mean (agg mean)**: Average value across all gait cycles
   ```
   agg_mean_relative_phase_left = mean(mean_relative_phase[cycle_1, cycle_2, ..., cycle_N])
   ```
   - Represents **typical** value for that leg/cyclogram type
   - Smooths out cycle-to-cycle variations

2. **Standard Deviation (agg std)**: Variability across cycles
   ```
   agg_std_relative_phase_left = std(mean_relative_phase[cycle_1, cycle_2, ..., cycle_N])
   ```
   - Measures **consistency** of gait cycles
   - High std indicates variable gait (instability, adaptation, pathology)

### Example Interpretation
```csv
cyclogram_type,leg,mean_relative_phase_mean,mean_relative_phase_std,marp_mean,marp_std
gyro_X_vs_Y,left,0.1234,0.0567,0.8901,0.1234
gyro_X_vs_Y,right,0.1456,0.0623,0.9123,0.1456
```

**Analysis**:
- Left leg has average phase lag of 0.1234 rad (±0.0567 rad across cycles)
- Right leg has average phase lag of 0.1456 rad (±0.0623 rad across cycles)
- Right leg shows slightly higher MARP (0.9123 vs 0.8901) → slightly poorer coordination
- Both legs show moderate cycle-to-cycle variability in MARP (~0.12-0.15 std)

### Clinical Use of Aggregate Metrics
1. **Population norms**: Establish reference ranges from healthy subjects
2. **Patient comparison**: Compare individual patient's aggregate values to norms
3. **Longitudinal tracking**: Monitor how aggregate metrics change over rehabilitation
4. **Bilateral asymmetry**: Compare left vs right aggregate means for asymmetry detection
5. **Gait stability**: Low aggregate std indicates consistent, stable gait

---

## Implementation Location

### Class: `AdvancedCyclogramMetrics`
File: `insole-analysis.py:173-437`

### Key Methods

**Temporal Metrics Computation**
```python
def compute_temporal_metrics(self) -> Dict:
    """
    Compute Level 2 temporal-coupling metrics.

    Location: insole-analysis.py:316-351

    Returns:
        Dictionary containing:
        - continuous_relative_phase: CRP array
        - mean_relative_phase: Average phase difference
        - marp: Mean absolute relative phase
        - coupling_angle_variability: Coordination stability
        - deviation_phase: Phase difference std
        - phase_shift: Mean timing lag
    """
```

**Helper: Phase Angle Calculation**
```python
def _compute_phase_angle(self, signal: np.ndarray) -> np.ndarray:
    """
    Compute phase angle using Hilbert transform.

    Location: insole-analysis.py:353-366

    Uses scipy.signal.hilbert for analytic signal representation.
    """
```

**Geometric Metrics (includes curvature)**
```python
def compute_geometric_metrics(self) -> Dict:
    """
    Compute Level 1 geometric/morphology metrics.

    Location: insole-analysis.py:200-241

    Returns:
        - mean_curvature: Movement control fineness
        - curvature_std: Curvature variability
        - trajectory_smoothness: 1/(1 + curvature_std)
    """
```

**Curvature Calculation**
```python
def _compute_curvature(self) -> np.ndarray:
    """
    Compute trajectory curvature κ = (x'y'' - y'x'') / (x'² + y'²)^(3/2).

    Location: insole-analysis.py:290-312

    Returns signed curvature array.
    """
```

### Aggregation Pipeline
File: `insole-analysis.py:4648-4709`

**Function**: `_save_advanced_cyclogram_metrics()`

**Pipeline steps**:
1. Compute metrics for all individual gait cycles (loops over cyclograms)
2. Store in DataFrame with columns: `cycle_id`, `cyclogram_type`, `leg`, metrics...
3. Group by `(cyclogram_type, leg)`
4. Apply aggregation: `mean` and `std` for each metric
5. Save individual metrics to `cyclogram_advanced_metrics.csv`
6. Save aggregate statistics to `cyclogram_metrics_aggregate.csv`

---

## Summary Table

| Metric | Formula | Location (line) | Interpretation |
|--------|---------|-----------------|----------------|
| **MARP** | `mean(abs(CRP))` | 341 | Average coordination magnitude (0-π rad) |
| **Mean Relative Phase** | `mean(CRP)` | 340 | Directional phase relationship (-π to +π rad) |
| **Coupling Angle Variability** | `std(arctan2(∇Y, ∇X))` | 344-345 | Direction stability (0-2 rad) |
| **Curvature VI** | `std(κ) / mean(abs(κ))` | 236-238 | Curvature consistency (coefficient of variation) |
| **VI Difference** | `abs(VI_left - VI_right)` | Section F | Bilateral curvature asymmetry |
| **Aggregate Mean** | `mean(metric[all_cycles])` | 4694-4704 | Typical value across cycles |
| **Aggregate Std** | `std(metric[all_cycles])` | 4694-4704 | Cycle-to-cycle variability |

---

## References

### Code Files
- **insole-analysis.py**: Main analysis script
  - Lines 316-351: Temporal metrics computation
  - Lines 200-241: Geometric metrics (curvature)
  - Lines 290-312: Curvature calculation helper
  - Lines 4691-4709: Aggregation pipeline

### Documentation
- **SECTION_F_METRICS_REFERENCE.md**: Curvature phase-variation metrics
- **SECTION_F_CURVATURE_IMPLEMENTATION.md**: Section F implementation details
- **insole_subplot_visualization_system.md**: Visualization architecture

### Scientific Background
- **Continuous Relative Phase (CRP)**: Kelso, J.A.S. (1995). *Dynamic Patterns*
- **Phase coupling analysis**: Hamill, J. et al. (1999). "A dynamical systems approach to lower extremity running injuries"
- **Curvature analysis**: Zeni, J.A. et al. (2008). "Two simple methods for determining gait events during treadmill and overground walking"

---

## Usage Example

### Running Analysis with Metric Output
```bash
# Run insole analysis
python3 Code-Script/insole-analysis.py \
  --input insole-sample/10MWT.csv \
  --output insole-output/10MWT_analysis

# Output files generated:
# - cyclogram_advanced_metrics.csv        (per-cycle metrics)
# - cyclogram_metrics_aggregate.csv       (aggregate statistics)
# - cyclogram_bilateral_symmetry.csv      (L-R comparisons including VI diff)
```

### Reading Aggregate Metrics
```python
import pandas as pd

# Load aggregate statistics
agg = pd.read_csv('insole-output/10MWT_analysis/cyclogram_metrics_aggregate.csv',
                  header=[0, 1], index_col=[0, 1])

# Access aggregate MARP for left gyro_X_vs_Y cyclogram
marp_mean = agg.loc[('gyro_X_vs_Y', 'left'), ('marp', 'mean')]
marp_std = agg.loc[('gyro_X_vs_Y', 'left'), ('marp', 'std')]

print(f"Left leg MARP: {marp_mean:.4f} ± {marp_std:.4f} rad")

# Compare left vs right
left_marp = agg.loc[('gyro_X_vs_Y', 'left'), ('marp', 'mean')]
right_marp = agg.loc[('gyro_X_vs_Y', 'right'), ('marp', 'mean')]
asymmetry = abs(left_marp - right_marp)

print(f"MARP asymmetry: {asymmetry:.4f} rad")
```

---

## Version History

- **v1.0** (2025-10-27): Initial comprehensive documentation of MARP, mean relative phase, coupling angle variability, curvature phase variability index difference, and aggregate metrics
