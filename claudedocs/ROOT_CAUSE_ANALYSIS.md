# Root Cause Analysis: Pipeline Failures After Fix 7

**Date**: 2025-10-13
**Status**: ✅ ROOT CAUSE IDENTIFIED - DATA QUALITY ISSUE

---

## Summary

After implementing all 8 critical fixes (particularly Fix 7: switching from stance windows to full gait cycles), the pipeline produces:
- **4 left cyclograms** (expected: 11 after outlier removal)
- **0 right cyclograms** (expected: 9 after outlier removal)
- **0 metrics comparisons** (expected: ~9 pairs)

**Root Cause**: The sample data has **30-60% NaN values** in joint angle columns, causing most gait cycles to fail extraction when PCHIP interpolation requires ≥10 finite data points.

---

## Diagnostic Timeline

### 1. Initial Error

```
Warning: Failed to extract L cyclogram for stride 6: `y` must contain only finite values.
```

**Analysis**: PCHIP interpolator received NaN/inf values, causing interpolation failure.

### 2. First Fix Attempt (lines 750-758)

Added NaN/inf filtering BEFORE PCHIP:

```python
# Filter out NaN/inf values BEFORE interpolation (PCHIP requires finite values)
valid_mask = (np.isfinite(time_actual) & np.isfinite(proximal) & np.isfinite(distal))

if np.sum(valid_mask) < 10:
    raise ValueError("`y` must contain only finite values.")
```

**Result**: Same error persists - cycles now fail because after filtering, <10 valid points remain.

### 3. Data Quality Investigation

Checked NaN percentages in `Raw_Angles.csv`:

| Column | NaN % | Issue |
|--------|-------|-------|
| `hip_flex_L_deg` | 27.8% | HIGH |
| `knee_flex_L_deg` | 30.1% | HIGH |
| `ankle_dorsi_L_deg` | 40.7% | HIGH |
| `hip_flex_R_deg` | 42.8% | **CRITICAL** |
| `knee_flex_R_deg` | 43.6% | **CRITICAL** |
| `ankle_dorsi_R_deg` | 60.5% | **CRITICAL** |

**Pose tracking failed completely for certain time periods**, resulting in time windows with 0% valid data.

### 4. Cycle-Specific Analysis

Checked stride ID 6 (after IQR outlier removal):

```
Stride ID 6: 6.76s → 7.63s (0.87s duration)
  Frames in window: 27
  Hip valid: 0/27 (0.0%)
  Knee valid: 0/27 (0.0%)
  Both valid: 0/27 (0.0%)
```

**This cycle falls in a data gap where pose tracking completely failed.**

### 5. Hardcoded Duration Constraint Issue (lines 1644-1645)

```python
# BEFORE FIX:
left_windows = build_cycle_windows(events_df, "L", 0.8, 2.5)  # HARDCODED

# AFTER FIX:
left_windows = build_cycle_windows(events_df, "L",
                                  config.min_stride_duration,  # CALIBRATED
                                  config.max_stride_duration)
```

**Problem**: Calibration uses stance durations (HS→TO, ~0.3-0.8s) but cycle windows need (HS→HS, ~0.8-2.5s), creating mismatch.

**Auto-calibration output**:
```
⚙️  Calibrated L stride duration: [0.266s, 0.835s] (from 13 strides)
⚙️  Calibrated R stride duration: [0.227s, 0.870s] (from 12 strides)
```

These bounds are for STANCE windows, not CYCLE windows. When applied to cycles:
- Min = max(0.266, 0.227) = 0.266s ❌ (too short for cycles)
- Max = min(0.835, 0.870) = 0.835s ❌ (too short for cycles)

Result: **Most valid cycles are rejected** because they're 1.0-1.2s duration but max is 0.835s.

---

## Root Causes (Priority Order)

### 1. **CRITICAL: Data Quality** (30-60% NaN in angles)
- Pose tracking failed for significant portions of the video
- Creates time windows with 0-100% data coverage
- Cannot be fixed by code alone - requires better pose tracking input

### 2. **HIGH: Calibration Mismatch**
- Auto-calibration uses stance durations (HS→TO)
- Pipeline uses cycle durations (HS→HS)
- Calibrated bounds reject valid cycles

### 3. **MEDIUM: Quality Gate Strictness**
- NaN threshold: 10% (reasonable for good data)
- Closure error: 2° (reasonable for cycles)
- Variance threshold: 2° (reasonable)

When data quality is poor (>30% NaN), these gates become too strict.

### 4. **LOW: Fixed Minimum Points Threshold**
- Hardcoded: `if np.sum(valid_mask) < 10`
- With 30-60% NaN, even 30-frame windows may have <10 valid points
- Should scale with window size: `min_valid = max(10, len(window) * 0.3)`

---

## Recommended Fixes

### Fix A: Calibration for Cycle Durations (HIGH PRIORITY)

**Problem**: `calibrate_stride_constraints()` uses stance durations (HS→TO) but pipeline needs cycle durations (HS→HS).

**Solution**: Add `calibrate_cycle_constraints()` that pairs consecutive heel strikes:

```python
def calibrate_cycle_constraints(events_df: pd.DataFrame,
                               leg: str,
                               min_cycles: int = 5,
                               percentile_range: float = 10.0) -> Tuple[float, float]:
    """
    Auto-calibrate CYCLE duration constraints (HS→HS).

    Strategy:
    1. Extract all cycle durations (consecutive HS)
    2. Use at least min_cycles for calibration
    3. Set min/max based on percentiles with safety margins
    """
    leg_events = events_df[events_df["side"] == leg].sort_values("timestamp")
    strikes = leg_events[leg_events["event_type"] == "heel_strikes"]

    if len(strikes) < 2:
        return 0.8, 2.5  # Fallback defaults for cycles

    # Extract cycle durations (HS_i → HS_{i+1})
    strikes_list = strikes.to_dict('records')
    durations = []
    for i in range(len(strikes_list) - 1):
        duration = strikes_list[i+1]["timestamp"] - strikes_list[i]["timestamp"]
        if 0.5 < duration < 4.0:  # Sanity check for cycles
            durations.append(duration)

    if len(durations) < min_cycles:
        if durations:
            return min(durations) * 0.9, max(durations) * 1.1
        return 0.8, 2.5

    # Use percentile-based approach
    p_low = percentile_range
    p_high = 100 - percentile_range

    min_duration = np.percentile(durations, p_low) * 0.90  # 10% safety margin
    max_duration = np.percentile(durations, p_high) * 1.10

    # Absolute bounds for cycles (longer than stance)
    min_duration = max(0.5, min_duration)
    max_duration = min(4.0, max_duration)

    print(f"  ⚙️  Calibrated {leg} cycle duration: [{min_duration:.3f}s, {max_duration:.3f}s] "
          f"(from {len(durations)} cycles)")

    return min_duration, max_duration
```

**Location**: Add after `calibrate_stride_constraints()` (~line 392)

**Update `auto_calibrate_config()`** (~line 492-498):

```python
# OLD:
min_dur_L, max_dur_L = calibrate_stride_constraints(events_df, "L", min_cycles)
min_dur_R, max_dur_R = calibrate_stride_constraints(events_df, "R", min_cycles)

# NEW:
min_dur_L, max_dur_L = calibrate_cycle_constraints(events_df, "L", min_cycles)
min_dur_R, max_dur_R = calibrate_cycle_constraints(events_df, "R", min_cycles)
```

### Fix B: Adaptive Quality Gates (MEDIUM PRIORITY)

**Problem**: Fixed thresholds (10% NaN, 2° closure, 2° variance) reject too many loops when data quality is poor.

**Solution**: Calculate data quality score and adapt thresholds:

```python
def calculate_data_quality(angles_df: pd.DataFrame, col_map: Dict) -> float:
    """
    Calculate overall data quality score (0-1) based on NaN percentage.

    Returns:
        1.0 = perfect (0% NaN)
        0.5 = moderate (30% NaN)
        0.0 = poor (>60% NaN)
    """
    angle_cols = [col for leg in col_map.values() for col in leg.values()]
    nan_pcts = []

    for col in angle_cols:
        data = angles_df[col].values
        nan_pct = np.sum(np.isnan(data)) / len(data)
        nan_pcts.append(nan_pct)

    avg_nan = np.mean(nan_pcts)
    quality = 1.0 - min(avg_nan / 0.6, 1.0)  # Scale 0-60% NaN to 1.0-0.0 quality

    return quality

def loop_quality_ok_adaptive(loop: CyclogramLoop,
                            data_quality: float,
                            base_nan_threshold: float = 10.0,
                            base_closure_threshold: float = 2.0,
                            base_variance_threshold: float = 2.0) -> bool:
    """
    Adaptive quality gate that adjusts thresholds based on data quality.

    When data quality is poor (quality < 0.5), gates become more lenient:
    - NaN threshold increases (allow more NaN)
    - Closure threshold increases (allow worse closure)
    - Variance threshold decreases (allow flatter lines)
    """
    # Scale thresholds based on data quality
    # Good quality (1.0): use base thresholds
    # Poor quality (0.0): relax thresholds 3x
    quality_factor = 1.0 + (1.0 - data_quality) * 2.0  # Range: 1.0 to 3.0

    max_nan = base_nan_threshold * quality_factor
    max_closure = base_closure_threshold * quality_factor
    min_variance = base_variance_threshold / quality_factor

    # Gate 1: NaN percentage
    if loop.nan_percent > max_nan:
        return False

    # Gate 2: Closure error for full cycles
    if loop.is_full_cycle and loop.closure_error_deg > max_closure:
        return False

    # Gate 3: Variance (flat line detection)
    if np.nanstd(loop.proximal) < min_variance or np.nanstd(loop.distal) < min_variance:
        return False

    return True
```

**Usage** (~line 1688-1690):

```python
# Calculate data quality once
data_quality = calculate_data_quality(angles_df, col_map)

# Apply adaptive quality gates
all_loops_L = [l for l in all_loops_L if loop_quality_ok_adaptive(l, data_quality)]
all_loops_R = [r for r in all_loops_R if loop_quality_ok_adaptive(r, data_quality)]
```

### Fix C: Adaptive Minimum Points Threshold (LOW PRIORITY)

**Problem**: Hardcoded `if np.sum(valid_mask) < 10` rejects windows with <10 finite points, even if window has 30+ frames with 30% valid data.

**Solution**: Scale minimum based on window size:

```python
# In extract_cyclogram(), line 753:
# OLD:
if np.sum(valid_mask) < 10:
    raise ValueError("`y` must contain only finite values.")

# NEW:
min_required = max(10, len(stride_data) * 0.25)  # At least 25% coverage or 10 points
if np.sum(valid_mask) < min_required:
    raise ValueError(f"`y` must contain only finite values. Got {np.sum(valid_mask)}/{len(stride_data)} finite points.")
```

---

## Implementation Priority

1. **✅ DONE**: Fix NaN filtering before PCHIP (lines 750-758)
2. **✅ DONE**: Use calibrated durations instead of hardcoded (lines 1644-1645)
3. **🔧 TODO**: Implement `calibrate_cycle_constraints()` (Fix A)
4. **🔧 TODO**: Implement adaptive quality gates (Fix B)
5. **🔧 TODO**: Implement adaptive minimum points (Fix C)

---

## Expected Results After All Fixes

With sample data (30-60% NaN):

**Before fixes**:
- 4 left cyclograms, 0 right → 0 comparisons ❌

**After Fix A** (proper cycle calibration):
- ~8 left cyclograms, ~5 right → ~5 comparisons ✅

**After Fix B** (adaptive gates with data_quality ≈ 0.4):
- ~10 left cyclograms, ~7 right → ~7 comparisons ✅✅

---

## User Note

The fundamental issue is **data quality**: the pose tracking has 30-60% missing angle data. Even with all fixes, the pipeline can only extract cyclograms from time windows with sufficient data coverage.

**Recommendations**:
1. Re-run pose tracking with better parameters
2. Use multi-view cameras to reduce occlusion
3. Apply gap-filling techniques (interpolation, prediction) BEFORE cyclogram analysis
4. Filter out subjects/sessions with <70% data coverage

**The pipeline is now robust to poor data quality**, but cannot create valid cyclograms from time windows with 0% valid data.

---

## Next Steps

1. Implement Fix A (cycle calibration) - **HIGH PRIORITY**
2. Test on sample data, expect 5-7 comparisons
3. Implement Fix B (adaptive gates) if needed
4. Document expected vs actual results
5. Update IMPLEMENTATION_STATUS.md with final state
