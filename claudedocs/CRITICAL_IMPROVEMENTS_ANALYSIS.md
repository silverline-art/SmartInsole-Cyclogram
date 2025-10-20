# Critical Cyclogram Analysis Improvements

**Date**: 2025-10-13
**Analysis**: Comprehensive clinical biomechanics review
**Status**: Partial implementation (core windowing fixed, remaining items documented)

---

## Executive Summary

The current cyclogram analysis has **THREE FUNDAMENTAL FLAWS** that invalidate clinical metrics:

1. **Stance-only windowing** (HS→TO, ~60% cycle) incorrectly called "strides"
2. **Phase-agnostic pairing** (temporal proximity) mismatches L-R comparisons
3. **Forced loop closure** manufactures invalid geometric metrics

### What's Been Fixed ✅

- ✅ Added `build_cycle_windows()` for full HS→HS gait cycles
- ✅ Updated `StrideWindow` dataclass with `window_type` field
- ✅ Updated `CyclogramLoop` with `is_full_cycle`, `closure_error_deg`, `nan_percent` tracking
- ✅ Added PCHIP interpolator import with graceful fallback
- ✅ Added fastdtw import with graceful fallback for DTW metrics

### What Needs Implementation 🔧

The remaining fixes are **interdependent** and must be done together:

1. Remove forced closure in `extract_cyclogram()`
2. Implement PCHIP interpolation (replace cubic)
3. Fix `pair_strides()` to use cycle indices
4. Add DTW distance metric
5. Add normalized area metric
6. Add quality gates before metrics computation
7. Update main analysis to use cycle windows

---

## Problem 1: Stance-Only Windowing (PARTIALLY FIXED)

### Current Implementation ❌
```python
def build_stride_windows(events_df, leg, min_duration, max_duration):
    # Pairs HS → TO (stance phase only, ~60% of gait cycle)
    for _, strike_row in strikes.iterrows():
        subsequent_toeoffs = toe_offs[toe_offs["timestamp"] > strike_row["timestamp"]]
        toeoff_row = subsequent_toeoffs.iloc[0]
        duration = toeoff_row["timestamp"] - strike_row["timestamp"]  # STANCE ONLY
```

**Impact**:
- Missing swing phase (TO→HS) = incomplete motion trajectory
- Forced closure on incomplete data creates artificial area/orientation
- Clinically incorrect to call this a "stride" (biomechanics terminology)

### Solution Implemented ✅
```python
def build_cycle_windows(events_df, leg, min_duration=0.8, max_duration=2.5):
    """
    Extract FULL GAIT CYCLE windows (HS→HS) for proper cyclogram analysis.
    Pairs consecutive heel strikes of the SAME leg (ipsilateral).
    """
    strikes = leg_events[leg_events["event_type"] == "heel_strikes"]

    for i in range(len(strikes_list) - 1):
        strike_start = strikes_list[i]
        strike_end = strikes_list[i + 1]  # NEXT ipsilateral HS
        duration = strike_end["timestamp"] - strike_start["timestamp"]  # FULL CYCLE
```

**Result**: Full gait cycle (stance + swing) = naturally closed loops

---

## Problem 2: Phase-Agnostic Pairing (NOT YET FIXED)

### Current Implementation ❌
```python
def pair_strides(left_windows, right_windows, tolerance_ratio=0.15):
    for left_win in left_windows:
        left_mid = (left_win.start_time + left_win.end_time) / 2  # Mid-point time

        for right_win in right_windows:
            right_mid = (right_win.start_time + right_win.end_time) / 2
            diff = abs(left_mid - right_mid)  # TEMPORAL PROXIMITY

            if diff < tolerance and diff < min_diff:
                best_right = right_win  # Pair by time proximity
```

**Why This Fails**:
- Gait is anti-phasic: when L foot strikes, R foot is ~50% through its cycle (mid-swing)
- Pairing L stride 3 with R stride 3 by time could actually be L3 vs R4 or R2
- Comparisons become apples-to-oranges (different phases)

### Required Fix 🔧
```python
def pair_cycles_by_index(left_windows, right_windows,
                        min_temporal_overlap=0.7) -> List[PairedStrides]:
    """
    Pair cycles by INDEX (L_k ↔ R_k) after validating temporal overlap.

    Strategy:
    1. Sort both lists by start_time
    2. Pair by index: L[i] ↔ R[i]
    3. Validate temporal overlap ≥70% of average cycle duration
    4. Reject pairs with excessive phase error
    """
    pairs = []
    min_len = min(len(left_windows), len(right_windows))

    for i in range(min_len):
        left_win = left_windows[i]
        right_win = right_windows[i]

        # Calculate temporal overlap
        overlap_start = max(left_win.start_time, right_win.start_time)
        overlap_end = min(left_win.end_time, right_win.end_time)
        overlap = max(0, overlap_end - overlap_start)
        avg_duration = (left_win.duration + right_win.duration) / 2
        overlap_pct = (overlap / avg_duration) if avg_duration > 0 else 0

        # Reject if insufficient temporal overlap
        if overlap_pct < min_temporal_overlap:
            continue

        # Calculate phase error (contralateral HS should be ~50% of cycle)
        # For proper gait: R_HS should occur near 50% of L cycle time
        time_diff = abs(left_win.start_time - right_win.start_time)
        phase_error = abs((time_diff / avg_duration) - 0.5) * 100  # % error from 50%

        # Reject if phase error > 15%
        if phase_error > 15:
            continue

        pairs.append(PairedStrides(
            left_stride=left_win,
            right_stride=right_win,
            time_difference=time_diff,
            temporal_overlap=overlap_pct
        ))

    return pairs
```

**Result**: Phase-true L-R comparisons at matching cycle indices

---

## Problem 3: Forced Loop Closure (NOT YET FIXED)

### Current Implementation ❌
```python
def extract_cyclogram(angles_df, window, joint_pair, ...):
    # Lines 693-704
    interp_proximal = interp1d(time_norm, proximal, kind='cubic',
                               fill_value='extrapolate', bounds_error=False)  # EXTRAPOLATE!
    interp_distal = interp1d(time_norm, distal, kind='cubic',
                            fill_value='extrapolate', bounds_error=False)

    proximal_resampled = interp_proximal(time_target)
    distal_resampled = interp_distal(time_target)

    # Ensure loop closure by setting last point equal to first point
    proximal_resampled[-1] = proximal_resampled[0]  # FORCED CLOSURE!
    distal_resampled[-1] = distal_resampled[0]
```

**Multiple Issues**:
1. **Forced closure**: Setting point[100] = point[0] when window might be stance-only
2. **Cubic + extrapolate**: Causes overshoot and artificial lobes beyond data range
3. **No closure error tracking**: Can't detect phase drift or incomplete cycles

### Required Fix 🔧
```python
def extract_cyclogram(angles_df, window, joint_pair, col_map, n_points=101, ...):
    """
    Generate cyclogram with proper closure error tracking and PCHIP interpolation.
    """
    # ... [data extraction code] ...

    # Resample using PCHIP (shape-preserving, no overshoot)
    from scipy.interpolate import PchipInterpolator

    time_target = np.linspace(0, 100, n_points)

    # PCHIP interpolation (NO extrapolation)
    interp_proximal = PchipInterpolator(time_norm, proximal)
    interp_distal = PchipInterpolator(time_norm, distal)

    proximal_resampled = interp_proximal(time_target)
    distal_resampled = interp_distal(time_target)

    # Calculate closure error (Euclidean distance between first and last point)
    closure_error_deg = np.sqrt(
        (proximal_resampled[-1] - proximal_resampled[0])**2 +
        (distal_resampled[-1] - distal_resampled[0])**2
    )

    # Calculate NaN percentage after resampling
    nan_count = np.sum(np.isnan(proximal_resampled)) + np.sum(np.isnan(distal_resampled))
    nan_percent = (nan_count / (2 * n_points)) * 100

    # DO NOT FORCE CLOSURE - let the data speak
    # For full cycles (HS→HS), closure should be natural (<2 degrees)
    # For stance-only, loop is intentionally open

    return CyclogramLoop(
        leg=window.leg,
        stride_id=window.stride_id,
        joint_pair=joint_pair,
        proximal=proximal_resampled,
        distal=distal_resampled,
        time_normalized=time_target,
        duration=window.duration,
        is_full_cycle=(window.window_type == "cycle"),
        closure_error_deg=closure_error_deg,
        nan_percent=nan_percent,
        speed=window.speed
    )
```

**Result**: Natural closure for full cycles, proper error tracking, no artificial geometry

---

## Additional Critical Fixes Needed

### 4. Add DTW Distance Metric 🔧
```python
def calculate_dtw_distance(loop_L: CyclogramLoop, loop_R: CyclogramLoop) -> float:
    """
    Compute Dynamic Time Warping distance to detect phase shifts.

    DTW captures temporal misalignment that Procrustes/RMSE miss.
    """
    if not HAS_FASTDTW:
        return np.nan  # Graceful degradation

    L_points = loop_L.points
    R_points = loop_R.points

    try:
        distance, _ = fastdtw(L_points, R_points, dist=spatial_distance.euclidean)
        return distance / len(L_points)  # Normalize by length
    except:
        return np.nan
```

### 5. Add Normalized Area Metric 🔧
```python
def calculate_normalized_area(loop: CyclogramLoop) -> float:
    """
    Compute scale-independent area normalized by loop variance.

    Normalized area = raw_area / (σ_proximal * σ_distal)
    Makes cross-subject/session comparisons valid.
    """
    raw_area = calculate_loop_area(loop)

    sigma_prox = np.std(loop.proximal)
    sigma_dist = np.std(loop.distal)

    if sigma_prox < 1e-6 or sigma_dist < 1e-6:
        return 0.0  # Degenerate loop

    return raw_area / (sigma_prox * sigma_dist)
```

### 6. Add Loop Quality Gates 🔧
```python
def validate_loop_quality(loop: CyclogramLoop) -> Tuple[bool, str]:
    """
    Quality gate: reject loops before metrics computation.

    Returns:
        (is_valid, reason_if_invalid)
    """
    # Gate 1: NaN percentage
    if loop.nan_percent > 5.0:
        return False, f"Excessive NaN: {loop.nan_percent:.1f}%"

    # Gate 2: Variance (flat line detection)
    prox_var = np.var(loop.proximal)
    dist_var = np.var(loop.distal)
    if prox_var < 1.0 or dist_var < 1.0:
        return False, f"Low variance: prox={prox_var:.2f}, dist={dist_var:.2f}"

    # Gate 3: Derivative spikes (noise detection)
    prox_diff = np.abs(np.diff(loop.proximal))
    dist_diff = np.abs(np.diff(loop.distal))
    if np.max(prox_diff) > 50 or np.max(dist_diff) > 50:
        return False, f"Derivative spike detected"

    # Gate 4: Closure error for full cycles
    if loop.is_full_cycle and loop.closure_error_deg > 5.0:
        return False, f"Poor closure: {loop.closure_error_deg:.2f}°"

    # Gate 5: Angle range (human biomechanics bounds)
    prox_range = np.max(loop.proximal) - np.min(loop.proximal)
    dist_range = np.max(loop.distal) - np.min(loop.distal)
    if prox_range > 180 or dist_range > 180:
        return False, f"Unrealistic angle range"

    return True, ""
```

### 7. Update Main Analysis Pipeline 🔧
```python
def analyze_subject(subject_dir, output_dir, config):
    # ... [data loading code] ...

    # USE CYCLE WINDOWS (not stance windows)
    left_cycles = build_cycle_windows(events_df, "L",
                                     config.min_stride_duration,
                                     config.max_stride_duration)
    right_cycles = build_cycle_windows(events_df, "R",
                                      config.min_stride_duration,
                                      config.max_stride_duration)

    print(f" Segmented cycles: {len(left_cycles)} left, {len(right_cycles)} right")

    # Extract cyclograms from FULL CYCLES
    all_loops_L = []
    all_loops_R = []

    for pair in config.joint_pairs:
        for window in left_cycles:
            try:
                loop = extract_cyclogram(angles_df, window, pair, col_map, ...)

                # APPLY QUALITY GATE
                is_valid, reason = validate_loop_quality(loop)
                if is_valid:
                    all_loops_L.append(loop)
                else:
                    print(f"  ⚠️  Rejected L{window.stride_id}: {reason}")
            except Exception as e:
                print(f"  ❌ Failed L{window.stride_id}: {e}")

    # USE CYCLE-INDEX PAIRING
    paired_cycles = pair_cycles_by_index(left_cycles, right_cycles,
                                        min_temporal_overlap=0.7)

    print(f" Paired cycles: {len(paired_cycles)} pairs")

    # ... [rest of analysis] ...
```

---

## Stride vs Cyclogram Discrepancy Explained

**Question**: "Why is total number of strides different than total number of cyclograms detected?"

**Answer**:

```
Strides detected = N heel strikes (events)
Cyclograms generated = (N strides) × (M joint pairs) × (extraction success rate)

Discrepancy due to:
1. **Insufficient data points**: Stride window has <10 angle measurements
2. **NaN values**: Missing/corrupted angle data during certain strides
3. **Quality gate failures**: Stride fails variance, derivative, or closure checks
4. **Extraction errors**: Interpolation fails, data range issues

With stance-only windowing (HS→TO):
- Short stance at high cadence → few frames → extraction failure

With full cycle windowing (HS→HS):
- Longer window → more frames → higher extraction success rate
- **This discrepancy will DECREASE after fixing to use cycle windows**
```

---

## Implementation Priority

**CRITICAL (Must do together - interdependent)**:
1. Replace forced closure with natural closure + error tracking
2. Replace cubic with PCHIP interpolation
3. Fix pairing to use cycle indices
4. Update main pipeline to use `build_cycle_windows()`
5. Add quality gates before metrics

**HIGH (Metrics validity)**:
6. Add DTW distance metric
7. Add normalized area metric
8. Update CyclogramMetrics dataclass to include new metrics

**MEDIUM (Robustness)**:
9. Add frontal plane angle detection + mirroring
10. Add event naming tolerance (HS/TO/heel_strike/toe_off variants)
11. Add QC overlay panels to plots

---

## Testing Checklist

After implementing all fixes, verify:

- [ ] Full cycles (HS→HS) produce closure error <2°
- [ ] L-R pairing uses cycle indices, temporal overlap ≥70%
- [ ] No forced closure artifacts in plots
- [ ] Normalized area allows cross-subject comparison
- [ ] DTW detects phase shifts that Procrustes misses
- [ ] Quality gates reject <5% of loops
- [ ] Stride count ≈ cyclogram count (within 10%)

---

## Conclusion

The current implementation has **clinically invalid** metrics due to:
1. Stance-only windowing (incomplete motion)
2. Phase-agnostic pairing (wrong comparisons)
3. Forced closure (manufactured geometry)

**Status**: Core windowing fixed, remaining fixes documented above. All fixes are interdependent and should be implemented together to maintain consistency.

**Recommended Action**: Complete the remaining 7 critical fixes before using this code for clinical analysis or publication.
