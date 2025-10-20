# Advanced Cyclogram Metrics Implementation Plan

**Status**: Design Complete - Ready for Implementation
**Priority**: Level 1 (Geometric) → Level 2 (Temporal) → Level 3 (Symmetry) → Level 4 (Advanced)
**Target**: Publication-quality gait analysis metrics

---

## Implementation Architecture

### Phase 1: Core Metrics Class (IMMEDIATE)

Create `AdvancedCyclogramMetrics` class with modular computation methods:

```python
class AdvancedCyclogramMetrics:
    """
    Comprehensive cyclogram analysis metrics for publication-quality gait research.

    Implements:
    - Level 1: Geometric/Morphology metrics
    - Level 2: Temporal-Coupling parameters
    - Level 3: Symmetry/Bilateral coordination
    - Level 4: Dynamic features (optional)
    - Level 5: 3D metrics (optional)
    - Level 6: Advanced research metrics (optional)
    """

    def __init__(self, cyclogram: CyclogramData):
        self.cyclogram = cyclogram
        self.x = cyclogram.x_signal
        self.y = cyclogram.y_signal
        self.z = cyclogram.z_signal if cyclogram.is_3d else None

    # Level 1: Geometric Metrics
    def compute_geometric_metrics(self) -> Dict:
        """Compactness, aspect ratio, eccentricity, orientation, curvature"""
        pass

    # Level 2: Temporal Metrics
    def compute_temporal_metrics(self) -> Dict:
        """Phase shift, coupling angle variability, CRP, MARP"""
        pass

    # Level 3: Symmetry Metrics
    def compute_symmetry_metrics(self, other_cyclogram) -> Dict:
        """Symmetry index, overlap, mirror correlation"""
        pass
```

---

## Level 1: Geometric/Morphology Metrics (PRIORITY 1)

### Metrics to Implement

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Compactness Ratio** | `4π × Area / Perimeter²` | Loop circularity (1.0 = perfect circle) |
| **Aspect Ratio** | `Major axis / Minor axis` | Dominant coupling direction |
| **Eccentricity** | `sqrt(1 - (b²/a²))` | Shape elongation (0 = circle, 1 = line) |
| **Orientation Angle** | `arctan(eigenvector)` | Phase relationship angle |
| **Mean Curvature** | `mean(abs(κ))` where `κ = (x'y'' - y'x'') / (x'² + y'²)^(3/2)` | Movement control fineness |
| **Trajectory Smoothness** | `1 / (1 + std(κ))` | Coordination smoothness |

### Implementation Code

```python
def compute_geometric_metrics(self) -> Dict:
    """Compute Level 1 geometric/morphology metrics."""
    metrics = {}

    # Already have: area, perimeter from existing code
    area = self._compute_area()
    perimeter = self._compute_perimeter()

    # 1. Compactness Ratio
    metrics['compactness_ratio'] = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

    # 2. Fit ellipse for aspect ratio and eccentricity
    ellipse_params = self._fit_ellipse()
    if ellipse_params:
        a, b, theta = ellipse_params  # major, minor, orientation
        metrics['aspect_ratio'] = a / b if b > 0 else np.inf
        metrics['eccentricity'] = np.sqrt(1 - (b**2 / a**2)) if a > 0 else 0
        metrics['orientation_angle'] = np.degrees(theta)

    # 3. Mean Curvature
    curvature = self._compute_curvature()
    metrics['mean_curvature'] = np.mean(np.abs(curvature))
    metrics['curvature_std'] = np.std(curvature)
    metrics['trajectory_smoothness'] = 1 / (1 + metrics['curvature_std'])

    return metrics

def _fit_ellipse(self) -> Optional[Tuple[float, float, float]]:
    """Fit ellipse using PCA or least squares."""
    # Center data
    x_centered = self.x - np.mean(self.x)
    y_centered = self.y - np.mean(self.y)

    # Covariance matrix
    coords = np.vstack([x_centered, y_centered])
    cov = np.cov(coords)

    # Eigenvalues = axes lengths
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    a = np.sqrt(eigenvalues[0]) * 2  # major axis
    b = np.sqrt(eigenvalues[1]) * 2  # minor axis
    theta = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

    return (a, b, theta)

def _compute_curvature(self) -> np.ndarray:
    """Compute trajectory curvature κ = (x'y'' - y'x'') / (x'² + y'²)^(3/2)"""
    # First derivatives
    dx = np.gradient(self.x)
    dy = np.gradient(self.y)

    # Second derivatives
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # Curvature
    numerator = dx * ddy - dy * ddx
    denominator = (dx**2 + dy**2)**(3/2)

    # Avoid division by zero
    curvature = np.where(denominator > 1e-10, numerator / denominator, 0)

    return curvature
```

---

## Level 2: Temporal-Coupling Parameters (PRIORITY 2)

### Metrics to Implement

| Metric | Description | Clinical Value |
|--------|-------------|----------------|
| **Phase Shift (Δφ)** | Mean angular lag between signals | Timing coordination |
| **Coupling Angle Variability (CAV)** | SD of phase angle | Coordination stability |
| **Continuous Relative Phase (CRP)** | Instantaneous phase difference | Dynamical coupling |
| **Mean Absolute Relative Phase (MARP)** | Average abs(φ₁ - φ₂) | Overall synchronization |

### Implementation Code

```python
def compute_temporal_metrics(self) -> Dict:
    """Compute Level 2 temporal-coupling metrics."""
    metrics = {}

    # 1. Phase angles using Hilbert transform
    phase_x = self._compute_phase_angle(self.x)
    phase_y = self._compute_phase_angle(self.y)

    # 2. Continuous Relative Phase
    crp = phase_x - phase_y
    crp = np.angle(np.exp(1j * crp))  # Wrap to [-π, π]

    metrics['continuous_relative_phase'] = crp
    metrics['mean_relative_phase'] = np.mean(crp)
    metrics['marp'] = np.mean(np.abs(crp))

    # 3. Coupling Angle Variability
    coupling_angle = np.arctan2(np.gradient(self.y), np.gradient(self.x))
    metrics['coupling_angle_variability'] = np.std(coupling_angle)
    metrics['deviation_phase'] = np.std(crp)

    # 4. Phase Shift
    metrics['phase_shift'] = np.mean(phase_x - phase_y)

    return metrics

def _compute_phase_angle(self, signal: np.ndarray) -> np.ndarray:
    """Compute phase angle using Hilbert transform."""
    from scipy.signal import hilbert
    analytic_signal = hilbert(signal - np.mean(signal))
    phase = np.angle(analytic_signal)
    return phase
```

---

## Level 3: Symmetry/Bilateral Coordination (PRIORITY 3)

### Metrics to Implement

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Symmetry Index (SI)** | `(A_L - A_R) / ((A_L + A_R)/2) × 100` | Inter-limb asymmetry % |
| **Loop Overlap Index (LOI)** | Overlapping area / Total area | Bilateral similarity |
| **Mirror Correlation (rₘ)** | `corr(left, mirror(right))` | Coupled control accuracy |
| **RMS Trajectory Difference** | RMS difference at homologous phases | Functional asymmetry |

### Implementation Code

```python
def compute_bilateral_symmetry(left_cyclogram, right_cyclogram) -> Dict:
    """Compute Level 3 bilateral symmetry metrics."""
    metrics = {}

    # 1. Symmetry Index
    area_left = compute_area(left_cyclogram.x, left_cyclogram.y)
    area_right = compute_area(right_cyclogram.x, right_cyclogram.y)
    metrics['symmetry_index'] = ((area_left - area_right) /
                                 ((area_left + area_right) / 2) * 100)

    # 2. Mirror Correlation
    # Mirror right cyclogram horizontally
    right_mirrored_x = -right_cyclogram.x

    # Resample to same length
    left_resampled = resample_cyclogram(left_cyclogram, 101)
    right_resampled = resample_cyclogram(right_cyclogram, 101)

    # Compute correlation
    metrics['mirror_correlation'] = np.corrcoef(
        left_resampled.x, -right_resampled.x
    )[0, 1]

    # 3. RMS Trajectory Difference
    rms_x = np.sqrt(np.mean((left_resampled.x - right_resampled.x)**2))
    rms_y = np.sqrt(np.mean((left_resampled.y - right_resampled.y)**2))
    metrics['rms_trajectory_diff'] = np.sqrt(rms_x**2 + rms_y**2)

    return metrics
```

---

## Integration Plan

### Step 1: Add Advanced Metrics to Existing Pipeline

Modify `_compute_cyclogram_metrics()` in `InsoleVisualizer`:

```python
def _compute_cyclogram_metrics(self, cyclogram: CyclogramData) -> Dict:
    """Enhanced with advanced metrics."""
    # Existing basic metrics
    basic_metrics = {
        'area': ...,
        'perimeter': ...,
        'closure_error': ...
    }

    # NEW: Advanced metrics
    advanced_computer = AdvancedCyclogramMetrics(cyclogram)
    geometric_metrics = advanced_computer.compute_geometric_metrics()
    temporal_metrics = advanced_computer.compute_temporal_metrics()

    # Combine all metrics
    return {
        **basic_metrics,
        **geometric_metrics,
        **temporal_metrics
    }
```

### Step 2: Add Summary Panels to Plots

Modify cyclogram plotting functions to include metric summary panels:

```python
def _plot_gyro_gait_subplots_with_summary(self, axes, data_dict):
    """Enhanced with statistical summary insets."""
    # Original cyclogram plotting...

    # NEW: Add summary panel for each subplot
    for subplot_idx, ax in enumerate(axes.flat):
        # Compute aggregate metrics across all cycles
        all_metrics = [self._compute_cyclogram_metrics(cyc)
                      for cyc in cyclograms_for_this_subplot]

        # Create summary text box
        summary_text = self._format_metrics_summary(all_metrics)

        # Add text box to subplot
        ax.text(0.02, 0.98, summary_text,
               transform=ax.transAxes,
               fontsize=6,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
```

### Step 3: Summary Text Formatting

```python
def _format_metrics_summary(self, metrics_list: List[Dict]) -> str:
    """Format metrics for display on plot."""
    # Aggregate across cycles
    compactness = [m['compactness_ratio'] for m in metrics_list]
    smoothness = [m['trajectory_smoothness'] for m in metrics_list]
    eccentricity = [m['eccentricity'] for m in metrics_list]

    summary = (
        f"Compactness: {np.mean(compactness):.3f} ± {np.std(compactness):.3f}\n"
        f"Smoothness: {np.mean(smoothness):.3f} ± {np.std(smoothness):.3f}\n"
        f"Eccentricity: {np.mean(eccentricity):.3f} ± {np.std(eccentricity):.3f}\n"
        f"n={len(metrics_list)} cycles"
    )

    return summary
```

---

## CSV Export Enhancement

Add advanced metrics to CSV output:

```python
# In _save_summary() method
summary_df = pd.DataFrame([
    {
        'cycle_id': cycle.cycle_id,
        'leg': cycle.leg,
        # Basic metrics
        'area': metrics['area'],
        'perimeter': metrics['perimeter'],
        # NEW: Advanced geometric metrics
        'compactness_ratio': metrics['compactness_ratio'],
        'aspect_ratio': metrics['aspect_ratio'],
        'eccentricity': metrics['eccentricity'],
        'orientation_angle': metrics['orientation_angle'],
        'mean_curvature': metrics['mean_curvature'],
        'trajectory_smoothness': metrics['trajectory_smoothness'],
        # NEW: Temporal metrics
        'mean_relative_phase': metrics['mean_relative_phase'],
        'coupling_angle_variability': metrics['coupling_angle_variability'],
        'phase_shift': metrics['phase_shift'],
    }
    for cycle, metrics in zip(all_cycles, all_metrics)
])
```

---

## Testing Strategy

1. **Unit Tests**: Test each metric computation independently
2. **Integration Tests**: Verify metrics pipeline with sample data
3. **Validation**: Compare against literature values for known gait patterns
4. **Performance**: Ensure <5% runtime increase

---

## Documentation Additions

### For CLAUDE.md

Add section:

```markdown
## Advanced Cyclogram Metrics

**Level 1 - Geometric/Morphology**:
- Compactness Ratio: Loop circularity (1.0 = perfect circle)
- Aspect Ratio: Major/minor axis ratio
- Eccentricity: Shape elongation (0 = circle, 1 = line)
- Orientation Angle: Phase relationship
- Mean Curvature: Movement control fineness
- Trajectory Smoothness: Coordination smoothness

**Level 2 - Temporal/Coupling**:
- Continuous Relative Phase (CRP): Dynamical coupling
- Mean Absolute Relative Phase (MARP): Synchronization
- Coupling Angle Variability (CAV): Coordination stability
- Phase Shift: Timing lag between signals

**Level 3 - Symmetry/Bilateral**:
- Symmetry Index (SI): Inter-limb asymmetry percentage
- Mirror Correlation: Bilateral similarity
- RMS Trajectory Difference: Functional asymmetry

All metrics exported to `cyclogram_advanced_metrics.csv`
```

---

## Future Enhancements (Level 4-6)

**Level 4 - Dynamic Features** (Future):
- Loop drift rate, cycle-to-cycle variability
- Temporal regularity index
- Loop frequency spectrum

**Level 5 - 3D Metrics** (Future):
- 3D volume, planarity index
- PCA energy distribution

**Level 6 - Research Metrics** (Future):
- Lyapunov exponents
- Recurrence quantification
- Approximate entropy
- Multiscale entropy

---

## Implementation Checklist

- [ ] Create `AdvancedCyclogramMetrics` class
- [ ] Implement Level 1 geometric metrics
- [ ] Implement Level 2 temporal metrics
- [ ] Implement Level 3 symmetry metrics
- [ ] Integrate into existing pipeline
- [ ] Add summary panels to plots
- [ ] Enhance CSV export
- [ ] Update documentation
- [ ] Write unit tests
- [ ] Validate with known data
- [ ] Performance optimization

---

## References

**Key Papers**:
1. Goswami, A. (1998). "A new gait parameterization technique" - Cyclogram fundamentals
2. Hsu, W. L., et al. (2003). "Control and estimation of posture during quiet stance" - Phase coupling metrics
3. Plotnik, M., et al. (2007). "Bilateral coordination of walking" - Symmetry indices
4. Lamb, P. F., & Stöckl, M. (2014). "Continuous Relative Phase" - CRP methodology

**Clinical Applications**:
- Parkinson's disease gait assessment
- Post-stroke rehabilitation monitoring
- Prosthetic gait optimization
- Sports biomechanics analysis

---

**Status**: Ready for implementation
**Estimated Effort**: 4-6 hours for Level 1-3
**Expected Impact**: Publication-quality metrics suite
