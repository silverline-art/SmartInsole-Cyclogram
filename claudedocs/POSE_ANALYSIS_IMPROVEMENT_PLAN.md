# Pose-Analysis.py Comprehensive Improvement Plan

**Date**: 2025-11-04
**Objective**: Systematize visualizations, enhance data handling, and improve calibration

---

## Current State Analysis

### ✅ Strengths
1. **PlotConfig class exists** (lines 963-1051) - centralized styling
2. **Auto-calibration implemented** (line 1403) - calibrates smoothing, duration, tolerance
3. **MMC computation** - morphological mean cyclogram for robust averaging
4. **Quality gates** - comprehensive validation system

### ❌ Issues Identified

#### 1. Visualization Problems
- **Limited plot types**: Only 2 functions (overlayed cyclograms, similarity bar)
- **No standardized templates**: Each plot manually configures matplotlib
- **Missing visualizations**:
  - Individual stride comparison grids
  - Quality dashboard with multiple metrics
  - Temporal analysis plots
  - Data quality visualization
  - Calibration diagnostic plots
- **Inconsistent styling**: Some hardcoded values bypass PlotConfig
- **No plot metadata**: Plots don't include analysis parameters, timestamps

#### 2. Data Handling Issues
- **Static quality thresholds**: excellent_threshold=90.0, good_threshold=75.0 (not adaptive)
- **Limited filter analysis**: Only Savitzky-Golay filter, no frequency analysis
- **No data quality metrics**: Missing SNR, completeness scores, temporal consistency
- **Insufficient preprocessing diagnostics**: No before/after comparison

#### 3. Calibration Limitations
- **Limited parameter coverage**: Only smoothing, duration, tolerance calibrated
- **No outlier detection calibration**: Uses fixed IQR factor=1.5
- **No quality gate calibration**: Coverage/gap/stability thresholds hardcoded
- **No adaptive filtering**: Filter cutoff frequency not calibrated
- **Missing robustness**: Single calibration run, no validation

---

## Improvement Design

### Phase 1: Enhanced Visualization Framework

#### 1.1 Standardized Plot Templates

**Create PlotTemplate enum and factory**:
```python
from enum import Enum

class PlotTemplate(Enum):
    """Standardized plot templates for consistent visualizations."""
    CYCLOGRAM_OVERLAY = "cyclogram_overlay"      # Side-by-side L-R comparison
    CYCLOGRAM_GRID = "cyclogram_grid"            # 3x3 grid for all strides
    SIMILARITY_BAR = "similarity_bar"            # Bar chart with error bars
    QUALITY_DASHBOARD = "quality_dashboard"      # 2x2 metrics dashboard
    TEMPORAL_ANALYSIS = "temporal_analysis"      # Time-series plots
    CALIBRATION_DIAGNOSTIC = "calibration_diag"  # Calibration validation
    DATA_QUALITY = "data_quality"                # Coverage/gap visualization

class PlotFactory:
    """Factory for creating standardized plots."""

    @staticmethod
    def create_figure(template: PlotTemplate, config: PlotConfig) -> Tuple[Figure, Axes]:
        """Create matplotlib figure with template-specific layout."""
        if template == PlotTemplate.CYCLOGRAM_OVERLAY:
            fig, axes = plt.subplots(1, 2, figsize=config.cyclogram_figsize, dpi=config.dpi)
        elif template == PlotTemplate.CYCLOGRAM_GRID:
            fig, axes = plt.subplots(3, 3, figsize=(18, 18), dpi=config.dpi)
        elif template == PlotTemplate.QUALITY_DASHBOARD:
            fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=config.dpi)
        # ... etc

        # Apply consistent styling
        fig.patch.set_facecolor(config.background_color)
        return fig, axes

    @staticmethod
    def add_metadata_footer(fig: Figure, metadata: dict, config: PlotConfig):
        """Add standardized metadata footer to plot."""
        metadata_text = (
            f"Generated: {metadata.get('timestamp', 'N/A')} | "
            f"Subject: {metadata.get('subject', 'N/A')} | "
            f"FPS: {metadata.get('fps', 'N/A')} | "
            f"Smoothing: {metadata.get('smooth_window', 'N/A')}-pt Savitzky-Golay"
        )
        fig.text(0.5, 0.01, metadata_text,
                ha='center', fontsize=config.annotation_fontsize-1,
                color='gray', style='italic')
```

#### 1.2 Enhanced PlotConfig

**Add adaptive color palettes and extended configuration**:
```python
@dataclass
class PlotConfig:
    """Enhanced centralized plotting configuration."""

    # ... existing fields ...

    # NEW: Extended color schemes
    quality_colormap: str = 'RdYlGn'  # Red-Yellow-Green for quality metrics
    diverging_colormap: str = 'RdBu_r'  # For asymmetry visualization
    sequential_colormap: str = 'viridis'  # For continuous metrics

    # NEW: Quality level thresholds (adaptive, set via calibration)
    excellent_threshold: float = 90.0  # Will be calibrated
    good_threshold: float = 75.0       # Will be calibrated
    acceptable_threshold: float = 60.0  # Will be calibrated

    # NEW: Grid specifications
    grid_linewidth: float = 0.5
    major_grid_alpha: float = 0.3
    minor_grid_alpha: float = 0.1

    # NEW: Title/annotation templates
    title_template: str = "{joint_pair} Cyclogram - {subject}"
    subtitle_template: str = "Symmetry: {similarity:.1f}% | Cycles: L={n_left}, R={n_right}"

    # NEW: Export settings
    export_formats: List[str] = field(default_factory=lambda: ['png', 'svg'])
    save_metadata_json: bool = True

    def get_quality_color(self, score: float) -> str:
        """Get color based on quality score using adaptive thresholds."""
        if score >= self.excellent_threshold:
            return self.excellent_color
        elif score >= self.good_threshold:
            return self.good_color
        elif score >= self.acceptable_threshold:
            return self.warning_color
        else:
            return '#8B0000'  # Dark red for poor quality

    def calibrate_quality_thresholds(self, quality_scores: List[float]):
        """Calibrate quality thresholds based on data distribution."""
        if len(quality_scores) < 5:
            return  # Keep defaults

        scores_sorted = np.sort(quality_scores)
        # Use percentiles: top 25% = excellent, 25-50% = good, 50-75% = acceptable
        self.excellent_threshold = np.percentile(scores_sorted, 75)
        self.good_threshold = np.percentile(scores_sorted, 50)
        self.acceptable_threshold = np.percentile(scores_sorted, 25)

        print(f"  ⚙️  Calibrated quality thresholds: "
              f"excellent≥{self.excellent_threshold:.1f}, "
              f"good≥{self.good_threshold:.1f}, "
              f"acceptable≥{self.acceptable_threshold:.1f}")
```

#### 1.3 New Visualization Functions

**1.3.1 Quality Dashboard**:
```python
def plot_quality_dashboard(left_cycles: List[StrideWindow],
                          right_cycles: List[StrideWindow],
                          metrics: List[CyclogramMetrics],
                          output_path: str,
                          config: PlotConfig) -> None:
    """
    Create comprehensive quality dashboard with 4 panels:
    - Top-left: Coverage heatmap per cycle
    - Top-right: Gap distribution per cycle
    - Bottom-left: Symmetry scores over time
    - Bottom-right: Quality metrics radar chart
    """
    fig, axes = PlotFactory.create_figure(PlotTemplate.QUALITY_DASHBOARD, config)
    ax_coverage, ax_gaps, ax_symmetry, ax_radar = axes.flatten()

    # Panel 1: Coverage heatmap
    # ... implementation

    # Panel 2: Gap distribution
    # ... implementation

    # Panel 3: Symmetry temporal evolution
    # ... implementation

    # Panel 4: Multi-metric radar
    # ... implementation

    PlotFactory.add_metadata_footer(fig, {...}, config)
    plt.savefig(output_path, dpi=config.dpi, bbox_inches=config.bbox_inches)
    plt.close()
```

**1.3.2 Individual Stride Grid**:
```python
def plot_stride_comparison_grid(loops_L: List[CyclogramLoop],
                                loops_R: List[CyclogramLoop],
                                joint_pair: Tuple[str, str],
                                output_path: str,
                                config: PlotConfig,
                                max_strides: int = 9) -> None:
    """
    Create 3x3 grid showing first 9 stride pairs for detailed inspection.
    Each subplot shows L-R overlay for one stride pair.
    """
    fig, axes = PlotFactory.create_figure(PlotTemplate.CYCLOGRAM_GRID, config)

    n_pairs = min(len(loops_L), len(loops_R), max_strides)

    for idx in range(n_pairs):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        # Plot left and right for this stride
        ax.plot(loops_L[idx].proximal, loops_L[idx].distal,
               color=config.left_color, linewidth=2, label='Left')
        ax.plot(loops_R[idx].proximal, loops_R[idx].distal,
               color=config.right_color, linewidth=2, label='Right')

        ax.set_title(f'Stride Pair {idx+1}', fontsize=config.subtitle_fontsize)
        ax.grid(True, alpha=config.grid_alpha)
        ax.legend(fontsize=config.legend_fontsize-2)

    # Hide unused subplots
    for idx in range(n_pairs, 9):
        row = idx // 3
        col = idx % 3
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=config.dpi)
    plt.close()
```

**1.3.3 Temporal Analysis**:
```python
def plot_temporal_analysis(loops_L: List[CyclogramLoop],
                          loops_R: List[CyclogramLoop],
                          metrics: List[CyclogramMetrics],
                          output_path: str,
                          config: PlotConfig) -> None:
    """
    Time-series visualization showing metric evolution across gait cycles.
    4 subplots: Area, RMSE, DTW, Procrustes over stride number.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=config.dpi)

    # Extract metric time series
    stride_numbers = [m.stride_id_L for m in metrics]
    areas_L = [m.area_L for m in metrics]
    areas_R = [m.area_R for m in metrics]
    rmse_values = [m.rmse for m in metrics]
    dtw_values = [m.dtw_distance for m in metrics]

    # Plot 1: Area comparison
    ax = axes[0, 0]
    ax.plot(stride_numbers, areas_L, 'o-', color=config.left_color, label='Left')
    ax.plot(stride_numbers, areas_R, 's-', color=config.right_color, label='Right')
    ax.set_title('Loop Area Evolution', fontsize=config.title_fontsize)
    ax.set_xlabel('Stride Number', fontsize=config.label_fontsize)
    ax.set_ylabel('Area (deg²)', fontsize=config.label_fontsize)
    ax.legend()
    ax.grid(True, alpha=config.grid_alpha)

    # Plot 2: RMSE
    # ... similar pattern

    plt.tight_layout()
    plt.savefig(output_path, dpi=config.dpi)
    plt.close()
```

---

### Phase 2: Advanced Data Handling

#### 2.1 Data Quality Metrics

**Add comprehensive quality assessment**:
```python
@dataclass
class DataQualityMetrics:
    """Comprehensive data quality assessment."""

    # Coverage metrics
    overall_coverage_pct: float
    per_joint_coverage: Dict[str, float]
    per_leg_coverage: Dict[str, float]

    # Gap analysis
    max_gap_frames: int
    mean_gap_size: float
    gap_count: int

    # Temporal consistency
    sampling_rate_std: float  # Variability in frame timing
    temporal_completeness: float  # % of expected frames present

    # Signal quality
    snr_estimates: Dict[str, float]  # Per joint SNR
    jump_severity_max: float  # Largest frame-to-frame jump
    jump_count: int  # Number of jumps > threshold

    # Overall score
    quality_score: float  # 0-100 composite score

    @staticmethod
    def compute(angles_df: pd.DataFrame,
                col_map: Dict[str, Dict[str, str]],
                config: AnalysisConfig) -> 'DataQualityMetrics':
        """Compute comprehensive quality metrics from raw angle data."""

        angle_cols = [col for leg in col_map.values() for col in leg.values()]

        # 1. Coverage analysis
        overall_coverage = (angles_df[angle_cols].notna().sum().sum() /
                           (len(angles_df) * len(angle_cols))) * 100

        per_joint_coverage = {}
        per_leg_coverage = {}
        for leg, joints in col_map.items():
            leg_coverage = []
            for joint, col in joints.items():
                coverage = (angles_df[col].notna().sum() / len(angles_df)) * 100
                per_joint_coverage[f"{leg}_{joint}"] = coverage
                leg_coverage.append(coverage)
            per_leg_coverage[leg] = np.mean(leg_coverage)

        # 2. Gap analysis
        max_gap = 0
        gap_sizes = []
        gap_count_total = 0

        for col in angle_cols:
            is_nan = angles_df[col].isna()
            # Find contiguous NaN runs
            nan_runs = is_nan.ne(is_nan.shift()).cumsum()[is_nan]
            if not nan_runs.empty:
                run_lengths = nan_runs.value_counts()
                if not run_lengths.empty:
                    max_gap = max(max_gap, run_lengths.max())
                    gap_sizes.extend(run_lengths.values)
                    gap_count_total += len(run_lengths)

        mean_gap = np.mean(gap_sizes) if gap_sizes else 0

        # 3. Temporal consistency
        if 'timestamp' in angles_df.columns:
            time_diffs = angles_df['timestamp'].diff().dropna()
            sampling_rate_std = time_diffs.std()
            expected_frames = (angles_df['timestamp'].max() - angles_df['timestamp'].min()) / time_diffs.median()
            temporal_completeness = (len(angles_df) / expected_frames) * 100 if expected_frames > 0 else 100
        else:
            sampling_rate_std = 0.0
            temporal_completeness = 100.0

        # 4. Signal quality (SNR estimate using variance ratio)
        snr_estimates = {}
        for col in angle_cols:
            valid_data = angles_df[col].dropna()
            if len(valid_data) > 10:
                # Rough SNR: signal variance / noise variance
                # Noise estimated from high-frequency components
                signal_var = valid_data.var()
                noise_var = valid_data.diff().dropna().var() / 2  # Diff amplifies noise
                snr = 10 * np.log10(signal_var / noise_var) if noise_var > 0 else 50.0
                snr_estimates[col] = max(0, min(50, snr))  # Clip to 0-50 dB
            else:
                snr_estimates[col] = 0.0

        # 5. Jump detection
        jump_severity_max = 0
        jump_count_total = 0
        for col in angle_cols:
            diffs = angles_df[col].diff().abs().dropna()
            jumps = diffs[diffs > config.max_angle_jump]
            if not jumps.empty:
                jump_severity_max = max(jump_severity_max, jumps.max())
                jump_count_total += len(jumps)

        # 6. Composite quality score
        # Weighted combination: 40% coverage, 30% gaps, 20% SNR, 10% jumps
        coverage_score = overall_coverage
        gap_score = max(0, 100 - (max_gap / config.max_gap_frames) * 100)
        snr_score = np.mean(list(snr_estimates.values())) * 2  # Scale to 0-100
        jump_score = max(0, 100 - (jump_count_total / len(angle_cols)) * 10)

        quality_score = (
            0.4 * coverage_score +
            0.3 * gap_score +
            0.2 * snr_score +
            0.1 * jump_score
        )

        return DataQualityMetrics(
            overall_coverage_pct=overall_coverage,
            per_joint_coverage=per_joint_coverage,
            per_leg_coverage=per_leg_coverage,
            max_gap_frames=max_gap,
            mean_gap_size=mean_gap,
            gap_count=gap_count_total,
            sampling_rate_std=sampling_rate_std,
            temporal_completeness=temporal_completeness,
            snr_estimates=snr_estimates,
            jump_severity_max=jump_severity_max,
            jump_count=jump_count_total,
            quality_score=quality_score
        )

    def print_report(self):
        """Print formatted quality report."""
        print(f"\n{'='*80}")
        print(f"DATA QUALITY REPORT")
        print(f"{'='*80}")
        print(f"Overall Quality Score: {self.quality_score:.1f}/100")
        print(f"\nCoverage:")
        print(f"  Overall: {self.overall_coverage_pct:.1f}%")
        for leg, cov in self.per_leg_coverage.items():
            print(f"  {leg} leg: {cov:.1f}%")
        print(f"\nGaps:")
        print(f"  Max gap: {self.max_gap_frames} frames")
        print(f"  Mean gap: {self.mean_gap_size:.1f} frames")
        print(f"  Total gaps: {self.gap_count}")
        print(f"\nSignal Quality (SNR):")
        for joint, snr in self.snr_estimates.items():
            print(f"  {joint}: {snr:.1f} dB")
        print(f"\nTemporal Consistency:")
        print(f"  Sampling rate std: {self.sampling_rate_std:.4f}s")
        print(f"  Completeness: {self.temporal_completeness:.1f}%")
        print(f"\nJumps:")
        print(f"  Max severity: {self.jump_severity_max:.1f}°")
        print(f"  Total count: {self.jump_count}")
        print(f"{'='*80}\n")
```

#### 2.2 Adaptive Filtering

**Add frequency analysis and adaptive filter design**:
```python
def calibrate_filter_parameters(angles_df: pd.DataFrame,
                                angle_cols: List[str],
                                fps: float) -> Tuple[int, int, float]:
    """
    Calibrate Savitzky-Golay filter parameters based on signal characteristics.

    Returns:
        (window_length, poly_order, cutoff_frequency)
    """
    # Analyze signal frequency content
    signal_frequencies = []

    for col in angle_cols:
        valid_data = angles_df[col].dropna()
        if len(valid_data) < 50:
            continue

        # Compute FFT
        fft_vals = np.fft.rfft(valid_data - valid_data.mean())
        fft_freq = np.fft.rfftfreq(len(valid_data), 1/fps)
        power = np.abs(fft_vals)**2

        # Find dominant frequency
        dominant_freq = fft_freq[np.argmax(power[1:])+1]  # Skip DC component
        signal_frequencies.append(dominant_freq)

    # Adaptive window sizing
    if signal_frequencies:
        median_freq = np.median(signal_frequencies)
        # Window should be ~1/4 of signal period
        period = 1 / median_freq if median_freq > 0 else 1.0
        window_frames = int((period / 4) * fps)
        # Ensure odd window
        window_length = window_frames if window_frames % 2 == 1 else window_frames + 1
        # Clip to reasonable range
        window_length = np.clip(window_length, 5, 31)

        # Cutoff frequency: Nyquist / 4 as conservative default
        cutoff_frequency = (fps / 2) / 4
    else:
        # Fallback defaults
        window_length = 11
        cutoff_frequency = fps / 8

    # Polynomial order: smaller for noisy data
    poly_order = 2 if window_length < 15 else 3

    print(f"  ⚙️  Calibrated filter: window={window_length}, poly_order={poly_order}, "
          f"cutoff={cutoff_frequency:.2f}Hz")

    return window_length, poly_order, cutoff_frequency
```

---

### Phase 3: Enhanced Auto-Calibration

**Expand auto_calibrate_config with additional parameters**:
```python
def auto_calibrate_config_enhanced(angles_df: pd.DataFrame,
                                  events_df: pd.DataFrame,
                                  col_map: Dict[str, Dict[str, str]],
                                  base_config: AnalysisConfig,
                                  min_cycles: int = 5) -> Tuple[AnalysisConfig, DataQualityMetrics]:
    """
    Enhanced auto-calibration with comprehensive parameter optimization.

    Calibrates:
    1. Data quality metrics (NEW)
    2. Filter parameters (NEW)
    3. Quality gate thresholds (NEW)
    4. Smoothing threshold
    5. Stride duration constraints
    6. Pairing tolerance
    7. Outlier detection parameters (NEW)
    """
    print(f"\n{'='*80}")
    print(f"⚙️  ENHANCED AUTO-CALIBRATION (minimum {min_cycles} cycles required)")
    print(f"{'='*80}")

    # 1. Compute comprehensive data quality metrics
    quality_metrics = DataQualityMetrics.compute(angles_df, col_map, base_config)
    quality_metrics.print_report()

    # 2. Calibrate filter parameters based on frequency analysis
    angle_cols = [col for leg in col_map.values() for col in leg.values()]
    window_length, poly_order, cutoff_freq = calibrate_filter_parameters(
        angles_df, angle_cols, base_config.fps)

    # 3. Calibrate quality gate thresholds based on data quality
    # Relax thresholds for poor-quality data
    if quality_metrics.quality_score < 50:
        min_coverage_pct = 50.0  # Relaxed from 70%
        max_gap_frames = 50      # Relaxed from 30
        print(f"  ⚙️  Relaxed quality gates due to low data quality ({quality_metrics.quality_score:.1f})")
    elif quality_metrics.quality_score < 75:
        min_coverage_pct = 60.0
        max_gap_frames = 40
    else:
        min_coverage_pct = 70.0
        max_gap_frames = 30

    # 4. Calibrate smoothing threshold (existing logic)
    smooth_threshold = calibrate_smoothing_threshold(angles_df, angle_cols, min_cycles)

    # 5. Calibrate cycle duration constraints (existing logic)
    min_dur_L, max_dur_L = calibrate_cycle_constraints(events_df, "L", min_cycles)
    min_dur_R, max_dur_R = calibrate_cycle_constraints(events_df, "R", min_cycles)
    min_stride_duration = max(min_dur_L, min_dur_R)
    max_stride_duration = min(max_dur_L, max_dur_R)

    # 6. Calibrate pairing tolerance (existing logic)
    # ... existing code ...

    # 7. Calibrate outlier detection parameters
    # Use IQR factor based on data variability
    angle_variability = angles_df[angle_cols].std().mean()
    if angle_variability > 15:  # High variability
        iqr_factor = 2.0  # More permissive
    elif angle_variability > 10:
        iqr_factor = 1.5  # Standard
    else:
        iqr_factor = 1.2  # More strict for low-noise data

    print(f"  ⚙️  Calibrated outlier detection: IQR factor={iqr_factor:.1f}")

    print(f"{'='*80}\n")

    # Create enhanced config
    calibrated_config = AnalysisConfig(
        fps=base_config.fps,
        smooth_window=window_length,  # CALIBRATED
        smooth_poly=poly_order,  # CALIBRATED
        smooth_threshold=smooth_threshold,  # CALIBRATED
        min_stride_duration=min_stride_duration,  # CALIBRATED
        max_stride_duration=max_stride_duration,  # CALIBRATED
        pairing_tolerance=pairing_tolerance,  # CALIBRATED
        min_coverage_pct=min_coverage_pct,  # CALIBRATED
        max_gap_frames=max_gap_frames,  # CALIBRATED
        outlier_iqr_factor=iqr_factor,  # CALIBRATED (NEW)
        n_resample_points=base_config.n_resample_points,
        joint_pairs=base_config.joint_pairs,
        similarity_weights=base_config.similarity_weights,
        plot_dpi=base_config.dpi,
        generate_per_leg_plots=base_config.generate_per_leg_plots,
        enhance_angles=base_config.enhance_angles
    )

    return calibrated_config, quality_metrics
```

---

## Implementation Priority

### 🔴 PHASE 1: Critical Improvements (Implement First)

1. **Enhanced PlotConfig with adaptive thresholds** (1-2 hours)
   - Add `calibrate_quality_thresholds()` method
   - Extend color schemes and templates
   - Location: Lines 963-1051

2. **DataQualityMetrics class** (2-3 hours)
   - Comprehensive quality assessment
   - SNR estimation, gap analysis, temporal consistency
   - Add after PlotConfig definition (~line 1052)

3. **Enhanced auto-calibration** (2-3 hours)
   - Integrate DataQualityMetrics
   - Adaptive quality gate thresholds
   - Frequency-based filter calibration
   - Update existing `auto_calibrate_config()` at line 1403

### 🟡 PHASE 2: Enhanced Visualizations (Implement Second)

4. **PlotFactory and templates** (3-4 hours)
   - Standardized figure creation
   - Metadata footer utility
   - Add after PlotConfig (~line 1052)

5. **Quality dashboard visualization** (2-3 hours)
   - 4-panel comprehensive quality view
   - Add as new function after existing plot functions (~line 2960)

6. **Stride comparison grid** (1-2 hours)
   - 3x3 individual stride visualization
   - Add after quality dashboard

### 🟢 PHASE 3: Additional Features (Optional)

7. **Temporal analysis plots** (2 hours)
   - Metric evolution over strides
8. **Calibration diagnostic plots** (1-2 hours)
   - Visualize calibration decisions
9. **Data quality visualization** (1-2 hours)
   - Coverage heatmaps, gap distribution

---

## Validation Strategy

### Test Cases

**Test 1: Poor Quality Data** (Current issue case)
```bash
python3 Pose-Analysis.py \
  --input-dir "/home/shivam/Desktop/Human_Pose/temp/front" \
  --subject-name "Openpose_김알렉세이_1750778_20231119_1"

Expected:
- DataQualityMetrics shows <50% coverage
- Auto-calibration relaxes thresholds
- Quality dashboard generated with warnings
```

**Test 2: Good Quality Data**
```bash
# Use high-quality reference data
Expected:
- DataQualityMetrics shows >85% coverage
- Standard thresholds maintained
- All visualizations generated successfully
```

**Test 3: Calibration Validation**
```bash
# Compare before/after calibration
Expected:
- Filter parameters adapt to signal frequency
- Quality thresholds match data distribution
- More cycles pass quality gates with adaptive thresholds
```

---

## Summary

This improvement plan addresses all identified issues:

✅ **Visualization**: Standardized templates, consistent styling, new plot types
✅ **Data Handling**: Comprehensive quality metrics, SNR estimation, adaptive filtering
✅ **Calibration**: Expanded parameter coverage, data-adaptive thresholds, robustness

**Key Innovations**:
1. **DataQualityMetrics** - comprehensive quality assessment
2. **Adaptive thresholding** - quality gates adjust to data characteristics
3. **Frequency-based filtering** - FFT analysis informs filter design
4. **PlotFactory** - standardized visualization pipeline

**Estimated Implementation Time**: 12-18 hours total

**Immediate Next Steps**:
1. Implement DataQualityMetrics class
2. Enhance auto-calibration with adaptive thresholds
3. Add quality dashboard visualization
4. Validate with test data
