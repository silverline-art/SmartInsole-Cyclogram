# Section F: Curvature Phase-Variation Metrics Reference

**Quick reference guide for interpreting Section F metrics in Excel output**

## Column Organization

### Per-Leg Metrics (10 columns per leg, 20 total)

#### Left Leg
```
left_curv_phase_peak_phase_%        # Phase where max curvature occurs (0-100%)
left_curv_phase_peak_value          # Maximum curvature value
left_curv_phase_variability_index   # Curvature variability (CV)
left_curv_phase_entropy             # Shannon entropy of curvature distribution
left_curv_phase_rms                 # RMS curvature
left_curv_p00                       # Mean ± std at 0-10% phase
left_curv_p10                       # Mean ± std at 10-20% phase
left_curv_p20                       # Mean ± std at 20-30% phase
left_curv_p30                       # Mean ± std at 30-40% phase
left_curv_p40                       # Mean ± std at 40-50% phase
left_curv_p50                       # Mean ± std at 50-60% phase
left_curv_p60                       # Mean ± std at 60-70% phase
left_curv_p70                       # Mean ± std at 70-80% phase
left_curv_p80                       # Mean ± std at 80-90% phase
left_curv_p90                       # Mean ± std at 90-100% phase
```

#### Right Leg
```
right_curv_phase_peak_phase_%       # Same metrics as left leg
right_curv_phase_peak_value
right_curv_phase_variability_index
right_curv_phase_entropy
right_curv_phase_rms
right_curv_p00 through right_curv_p90  # 10 phase bins
```

### Bilateral Symmetry Metrics (4 columns)

```
curv_phase_rms_diff                      # |RMS_left - RMS_right|
curv_phase_circular_corr                 # Pearson correlation L vs R
curv_phase_peak_phase_diff_%             # |Peak_phase_left - Peak_phase_right|
curv_phase_variability_index_diff        # |VI_left - VI_right|
```

## Metric Interpretations

### 1. Peak Phase Metrics

#### `{leg}_curv_phase_peak_phase_%`
- **Range**: 0-100%
- **Interpretation**:
  - 0-30%: Peak curvature in early stance/loading
  - 30-60%: Peak curvature in mid-stance/push-off
  - 60-100%: Peak curvature in swing phase
- **Clinical significance**: Timing of maximum gait complexity
- **Normal expectation**: ~50% (mid-cycle)

#### `{leg}_curv_phase_peak_value`
- **Range**: >0 (typically 0.01-0.5)
- **Interpretation**:
  - Higher values = sharper turns in cyclogram
  - Lower values = smoother, more circular patterns
- **Clinical significance**: Maximum trajectory complexity
- **Normal expectation**: Relatively stable across cycles

### 2. Statistical Metrics

#### `{leg}_curv_phase_variability_index`
- **Formula**: std(|κ|) / mean(|κ|)
- **Range**: 0-2+ (coefficient of variation)
- **Interpretation**:
  - <0.3: Low variability (consistent gait)
  - 0.3-0.6: Moderate variability
  - >0.6: High variability (unstable gait)
- **Clinical significance**: Gait consistency measure
- **Normal expectation**: <0.5 for healthy gait

#### `{leg}_curv_phase_entropy`
- **Formula**: -Σ p_i * log₂(p_i)
- **Range**: 0-3+ bits
- **Interpretation**:
  - Low entropy (<1.5): Predictable curvature pattern
  - High entropy (>2.5): Unpredictable, varied pattern
- **Clinical significance**: Gait complexity/predictability
- **Normal expectation**: 1.5-2.5 for healthy gait

#### `{leg}_curv_phase_rms`
- **Formula**: √(mean(κ²))
- **Range**: >0 (typically 0.01-0.3)
- **Interpretation**:
  - Higher RMS = more overall curvature
  - Lower RMS = straighter trajectory
- **Clinical significance**: Overall trajectory curvature magnitude
- **Normal expectation**: Stable across healthy individuals

### 3. Phase-Binned Curvature

#### `{leg}_curv_p{XX}` (format: `mean ± std`)
- **Bins**: 10 bins covering 0-100% of gait cycle
- **Interpretation per phase**:

| Phase Bin | Gait Phase | Expected Pattern |
|-----------|------------|------------------|
| p00 (0-10%) | Initial contact | Low-moderate curvature |
| p10 (10-20%) | Loading response | Increasing curvature |
| p20 (20-30%) | Mid-stance entry | Moderate curvature |
| p30 (30-40%) | Mid-stance | Variable by individual |
| p40 (40-50%) | Terminal stance | May peak here |
| p50 (50-60%) | Pre-swing | High curvature (transition) |
| p60 (60-70%) | Initial swing | Moderate curvature |
| p70 (70-80%) | Mid-swing | Lower curvature |
| p80 (80-90%) | Terminal swing | Moderate curvature |
| p90 (90-100%) | Cycle completion | Decreasing to initial |

- **Clinical significance**:
  - Identify phase-specific abnormalities
  - Compare L/R patterns by phase
  - Track rehabilitation progress phase-by-phase

### 4. Bilateral Symmetry Metrics

#### `curv_phase_rms_diff`
- **Range**: 0-1+ (absolute difference)
- **Interpretation**:
  - <0.05: Excellent symmetry
  - 0.05-0.15: Good symmetry
  - >0.15: Asymmetric gait
- **Clinical significance**: Overall curvature asymmetry
- **Pathology indicator**: Limb length discrepancy, weakness, pain avoidance

#### `curv_phase_circular_corr`
- **Range**: -1 to +1
- **Interpretation**:
  - >0.8: Highly similar L/R patterns
  - 0.5-0.8: Moderately similar
  - <0.5: Dissimilar patterns
- **Clinical significance**: Pattern similarity (not just magnitude)
- **Pathology indicator**: Neurological asymmetry, compensation patterns

#### `curv_phase_peak_phase_diff_%`
- **Range**: 0-100% (absolute difference)
- **Interpretation**:
  - <10%: Synchronized peak timing
  - 10-25%: Mild phase shift
  - >25%: Significant phase asynchrony
- **Clinical significance**: Temporal coordination
- **Pathology indicator**: Coordination deficits, neural timing issues

#### `curv_phase_variability_index_diff`
- **Range**: 0-2+ (absolute difference)
- **Interpretation**:
  - <0.1: Symmetric variability
  - 0.1-0.3: Mild asymmetry
  - >0.3: One leg significantly more variable
- **Clinical significance**: Stability asymmetry
- **Pathology indicator**: Unilateral weakness, confidence issues

## Clinical Use Cases

### 1. Post-Fracture Assessment
**Key metrics to monitor**:
- `curv_phase_rms_diff` - Overall recovery symmetry
- `{affected}_curv_phase_variability_index` - Stability of affected limb
- Phase bins p40-p60 - Push-off quality

**Expected progression**:
- Early: High VI, high RMS diff
- Mid: Decreasing VI, improving correlation
- Late: Normal range (<0.15 RMS diff, >0.7 corr)

### 2. Gait Asymmetry Diagnosis
**Workflow**:
1. Check `curv_phase_rms_diff` - Overall asymmetry magnitude
2. Check `curv_phase_circular_corr` - Pattern similarity
3. If asymmetric, examine phase bins to localize issue
4. Check `curv_phase_peak_phase_diff_%` - Timing coordination

**Red flags**:
- RMS diff >0.2 AND correlation <0.5
- Peak phase diff >30%
- VI diff >0.4

### 3. Rehabilitation Progress Tracking
**Baseline → Follow-up comparisons**:
- Entropy should decrease (more predictable)
- Variability index should decrease (more stable)
- Bilateral correlation should increase (more symmetric)
- Phase-binned patterns should normalize

**Quantitative targets**:
- VI: Reduce by 20-30% over 6 weeks
- RMS diff: Reduce by 40-50% over 6 weeks
- Correlation: Improve from <0.6 to >0.7

### 4. Fall Risk Assessment
**High-risk indicators**:
- Variability index >0.7 (inconsistent gait)
- Entropy >3.0 (unpredictable patterns)
- Low correlation <0.4 (asymmetric strategies)
- Large std in p80-p90 bins (swing phase instability)

## Data Quality Indicators

### Valid Data
- All curvature values >0 (absolute values)
- RMS > mean curvature (mathematical property)
- Entropy between 0-log₂(n_bins) ≈ 3.3
- Correlation between -1 and +1

### Suspicious Data
- Entropy = 0 (single curvature value, check data)
- RMS diff = 0 with correlation ≠ 1 (computational error)
- All phase bins identical (insufficient data)
- Peak phase always at 0% or 100% (boundary artifact)

## Excel Analysis Tips

### Conditional Formatting
```
Highlight asymmetry:
- RMS diff >0.15 → Red
- Correlation <0.5 → Yellow
- VI diff >0.3 → Orange

Highlight extremes:
- VI >0.7 → Red (unstable)
- Entropy >2.5 → Yellow (complex)
```

### Comparison Columns
```
Add calculated columns:
= (left_curv_phase_rms + right_curv_phase_rms) / 2    # Average RMS
= ABS(left_curv_phase_entropy - right_curv_phase_entropy)  # Entropy asymmetry
```

### Visualization
```
Recommended plots:
1. Scatter: VI_left vs VI_right (diagonal = symmetric)
2. Line: Phase bins p00-p90 for L vs R
3. Bar: RMS diff by patient (identify outliers)
4. Histogram: Correlation distribution across cohort
```

## Troubleshooting

### Empty Metrics
**Cause**: No `continuous_relative_phase` data in CSV
**Solution**: Verify cyclogram processing completed successfully

### All NaN
**Cause**: Phase array parsing failed
**Solution**: Check CSV format, ensure arrays are space-separated

### Unrealistic Values
**Cause**: Very short cycles (<3 samples)
**Solution**: Check cycle detection parameters, minimum cycle length

### High Variability
**Cause**: True gait instability OR noisy data
**Solution**: Compare with raw cyclograms visually, verify smoothing

## References

### Column Naming Convention
```
{leg}_{section}_{metric}_{statistic}

Examples:
left_curv_phase_peak_phase_%         # leg_section_metric
left_curv_p20                        # leg_section_bin (implicit mean±std)
curv_phase_rms_diff                  # section_metric_statistic (bilateral)
```

### Abbreviations
- `curv`: Curvature
- `p{XX}`: Phase bin starting at XX%
- `VI`: Variability Index
- `RMS`: Root Mean Square
- `diff`: Absolute difference (bilateral)
- `corr`: Correlation coefficient

## Summary

**Section F provides 58 new columns** capturing:
- **Temporal dynamics**: When curvature peaks occur
- **Statistical properties**: How variable and predictable
- **Spatial distribution**: Phase-by-phase curvature profile
- **Bilateral coordination**: L/R symmetry across all dimensions

**Clinical value**: Quantifies gait complexity, stability, and symmetry in ways that complement traditional cyclogram area/shape metrics.
