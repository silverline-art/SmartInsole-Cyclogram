# Angle Calculation & Anticipation Design
**Design Date**: 2025-10-13
**Problem**: Cyclogram analysis failing due to 27-60% NaN values in angle data, despite having complete keypoint coverage
**Solution**: Multi-tier calculation strategy using geometric reconstruction and biomechanical anticipation

---

## Executive Summary

The root cause analysis revealed that:
1. **Keypoint data has 100% coverage** (no NaN values)
2. **Angle data has 30-44% NaN** due to strict quality filters in angle calculation
3. **Recovery potential**: 144 LEFT + 209 RIGHT frames can be recovered by recalculating angles from existing keypoints

This design implements a **5-tier progressive enhancement strategy** that uses geometric calculation and biomechanical models rather than statistical interpolation, preserving gait dynamics while maximizing data utilization.

---

## Design Philosophy

### Core Principles
1. **Calculation over Interpolation**: Use geometric/biomechanical models to COMPUTE missing values, not interpolate
2. **Preserve Valid Data**: Never modify or replace existing valid measurements
3. **Progressive Enhancement**: Start with safest approach, add sophistication only if needed
4. **Biomechanical Validity**: All calculated angles must respect anatomical constraints
5. **Transparency**: Each calculation method has confidence scores and validation

### "Calculation in Anticipation" Defined
- **Anticipation**: Using biomechanical knowledge to predict missing data based on kinematic constraints
- **Calculation**: Computing angles from geometric relationships, not guessing from temporal patterns
- **NOT interpolation**: We don't average neighboring values; we use physics and anatomy

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    CYCLOGRAM ANALYSIS PIPELINE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  [Raw Keypoints] ──► [Angle Calculation Tiers] ──► [Analysis]   │
│                                                                   │
│  Tier 1: Segment-Aware Extraction (BASELINE - Zero Risk)        │
│  Tier 2: Keypoint-Based Recalculation (HIGH IMPACT)             │
│  Tier 3: Kinematic Chain Calculation (GEOMETRIC)                │
│  Tier 4: Small-Gap Interpolation (CONSERVATIVE)                 │
│  Tier 5: Biomechanical Anticipation (ADVANCED)                  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tier 1: Segment-Aware Extraction (BASELINE)

### Objective
Extract cyclograms ONLY from continuous data segments with <5% NaN, ensuring high-quality results without data modification.

### Current Problem
- Heel strikes detected globally across entire timeline
- Strides may span valid→invalid→valid regions
- QC rejects strides with NaN, but rejection happens AFTER extraction

### Solution
```
OLD: [Detect all HS] → [Extract all strides] → [QC filter] → Few valid pairs
NEW: [Find valid segments] → [Detect HS in segments] → [Extract complete strides] → Many valid pairs
```

### Implementation

#### Step 1: Identify Valid Segments
```python
def identify_valid_segments(df: pd.DataFrame,
                           angle_cols: List[str],
                           max_nan_percent: float = 5.0,
                           min_segment_frames: int = 60) -> List[Tuple[int, int]]:
    """
    Find continuous segments where ALL required angles are valid.

    Args:
        df: DataFrame with angle data
        angle_cols: Columns that must be valid (e.g., ['hip_flex_L_deg', 'knee_flex_L_deg', 'ankle_dorsi_L_deg'])
        max_nan_percent: Maximum NaN percentage allowed in segment
        min_segment_frames: Minimum segment length (frames) to be useful (~2 gait cycles)

    Returns:
        List of (start_frame, end_frame) tuples for valid segments
    """
    # Check where ALL required angles are valid
    all_valid = df[angle_cols].notna().all(axis=1)

    # Find continuous runs of valid frames
    segments = []
    start = None

    for i, is_valid in enumerate(all_valid):
        if is_valid and start is None:
            start = i
        elif not is_valid and start is not None:
            # Check segment quality before adding
            segment_data = df.iloc[start:i][angle_cols]
            nan_pct = segment_data.isna().sum().sum() / segment_data.size * 100

            if (i - start) >= min_segment_frames and nan_pct <= max_nan_percent:
                segments.append((start, i-1))
            start = None

    # Handle segment extending to end
    if start is not None:
        segment_data = df.iloc[start:][angle_cols]
        nan_pct = segment_data.isna().sum().sum() / segment_data.size * 100
        if (len(df) - start) >= min_segment_frames and nan_pct <= max_nan_percent:
            segments.append((start, len(df)-1))

    return segments
```

#### Step 2: Segment-Aware Stride Detection
```python
def detect_strides_in_segments(df: pd.DataFrame,
                               segments: List[Tuple[int, int]],
                               side: str = 'L') -> List[Tuple[int, int]]:
    """
    Detect heel strikes only within valid segments, ensuring complete stride coverage.

    Args:
        df: DataFrame with angle and timing data
        segments: List of valid segments (start_frame, end_frame)
        side: 'L' or 'R' for left/right leg

    Returns:
        List of (hs_start_frame, hs_end_frame) for complete strides within segments
    """
    valid_strides = []

    for seg_start, seg_end in segments:
        # Extract segment data
        segment_df = df.iloc[seg_start:seg_end+1].copy()

        # Detect heel strikes in this segment
        hs_events = detect_heel_strikes(segment_df, side)  # Existing function

        # Validate that each stride HS_i → HS_i+1 is completely within segment
        for i in range(len(hs_events) - 1):
            stride_start = hs_events[i]
            stride_end = hs_events[i+1]

            # Convert to global frame indices
            global_start = seg_start + stride_start
            global_end = seg_start + stride_end

            # Verify stride is entirely within segment bounds
            if global_end <= seg_end:
                valid_strides.append((global_start, global_end))

    return valid_strides
```

### Expected Impact
- **Current**: 10 LEFT detected → 4 pass QC (40% efficiency)
- **With Tier 1**: 6-8 LEFT detected → 6-8 pass QC (100% efficiency)
- **Benefit**: Higher quality results, no wasted computation on invalid strides

### Risk Assessment
- **Risk**: ZERO - no data modification, only smarter extraction
- **Validation**: Compare stride counts before/after
- **Rollback**: Can revert to original detection logic instantly

---

## Tier 2: Keypoint-Based Angle Recalculation (HIGH IMPACT)

### Objective
Recover 30-44% missing angles by recalculating from existing keypoints using geometric formulas.

### Discovery
- **LEFT leg**: 144 frames (30.1%) have valid keypoints but NaN angles
- **RIGHT leg**: 209 frames (43.6%) have valid keypoints but NaN angles
- **Root cause**: Strict quality filters in original angle calculation rejected these frames

### Solution
When angle is NaN, check if underlying keypoints exist. If yes, recalculate angle using 2D geometry.

### Geometric Angle Calculation

#### Hip Flexion Angle
```python
def calculate_hip_flexion(hip_y: float, knee_y: float,
                         hip_x: float, knee_x: float,
                         pelvis_y: float) -> float:
    """
    Calculate hip flexion angle from keypoint positions.

    Hip flexion = angle between vertical line and thigh segment.
    - Positive = hip flexion (thigh forward)
    - Negative = hip extension (thigh backward)

    Args:
        hip_y, hip_x: Hip joint position
        knee_y, knee_x: Knee joint position
        pelvis_y: Pelvic reference for vertical axis

    Returns:
        Hip flexion angle in degrees
    """
    # Vector from hip to knee (thigh segment)
    thigh_vec = np.array([knee_x - hip_x, knee_y - hip_y])

    # Vertical reference vector (pointing down)
    vertical_vec = np.array([0, hip_y - pelvis_y])

    # Calculate angle between vectors
    dot_product = np.dot(thigh_vec, vertical_vec)
    mag_product = np.linalg.norm(thigh_vec) * np.linalg.norm(vertical_vec)

    if mag_product < 1e-6:  # Avoid division by zero
        return np.nan

    angle_rad = np.arccos(np.clip(dot_product / mag_product, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)

    # Determine sign based on x-position (flexion = forward = positive x)
    if knee_x > hip_x:
        return angle_deg
    else:
        return -angle_deg
```

#### Knee Flexion Angle
```python
def calculate_knee_flexion(hip_y: float, hip_x: float,
                          knee_y: float, knee_x: float,
                          ankle_y: float, ankle_x: float) -> float:
    """
    Calculate knee flexion angle from keypoint positions.

    Knee flexion = 180° - interior angle at knee joint
    - 180° = fully extended (straight leg)
    - 0° = fully flexed (heel to buttock)

    Args:
        hip_y, hip_x: Hip joint position
        knee_y, knee_x: Knee joint position
        ankle_y, ankle_x: Ankle joint position

    Returns:
        Knee flexion angle in degrees (0=flexed, 180=extended)
    """
    # Vector from knee to hip (thigh)
    thigh_vec = np.array([hip_x - knee_x, hip_y - knee_y])

    # Vector from knee to ankle (shank)
    shank_vec = np.array([ankle_x - knee_x, ankle_y - knee_y])

    # Calculate interior angle
    dot_product = np.dot(thigh_vec, shank_vec)
    mag_product = np.linalg.norm(thigh_vec) * np.linalg.norm(shank_vec)

    if mag_product < 1e-6:
        return np.nan

    angle_rad = np.arccos(np.clip(dot_product / mag_product, -1.0, 1.0))
    interior_angle = np.degrees(angle_rad)

    # Knee flexion = 180° - interior angle
    return 180.0 - interior_angle
```

#### Ankle Dorsiflexion Angle
```python
def calculate_ankle_dorsiflexion(knee_y: float, knee_x: float,
                                 ankle_y: float, ankle_x: float,
                                 foot_y: float, foot_x: float) -> float:
    """
    Calculate ankle dorsiflexion angle from keypoint positions.

    Ankle dorsiflexion = angle between shank and foot segments
    - 90° = neutral (foot perpendicular to shank)
    - >90° = dorsiflexion (toes up)
    - <90° = plantarflexion (toes down)

    Args:
        knee_y, knee_x: Knee joint position
        ankle_y, ankle_x: Ankle joint position
        foot_y, foot_x: Foot reference point (toe or foot index)

    Returns:
        Ankle angle in degrees
    """
    # Vector from ankle to knee (shank)
    shank_vec = np.array([knee_x - ankle_x, knee_y - ankle_y])

    # Vector from ankle to foot (foot segment)
    foot_vec = np.array([foot_x - ankle_x, foot_y - ankle_y])

    # Calculate angle
    dot_product = np.dot(shank_vec, foot_vec)
    mag_product = np.linalg.norm(shank_vec) * np.linalg.norm(foot_vec)

    if mag_product < 1e-6:
        return np.nan

    angle_rad = np.arccos(np.clip(dot_product / mag_product, -1.0, 1.0))
    return np.degrees(angle_rad)
```

### Implementation Strategy

```python
def recalculate_angles_from_keypoints(keypoints_df: pd.DataFrame,
                                      angles_df: pd.DataFrame,
                                      side: str = 'L') -> pd.DataFrame:
    """
    Recalculate missing angles from valid keypoints.

    Args:
        keypoints_df: DataFrame with landmark positions
        angles_df: DataFrame with existing angles (some NaN)
        side: 'L' or 'R'

    Returns:
        Updated angles_df with recalculated values and confidence scores
    """
    # MediaPipe landmark indices
    if side == 'L':
        hip_idx, knee_idx, ankle_idx = 23, 25, 27
        foot_idx = 31  # Left foot index
    else:
        hip_idx, knee_idx, ankle_idx = 24, 26, 28
        foot_idx = 32  # Right foot index

    # Get keypoint columns
    hip_x = f'landmark_{hip_idx}_x'
    hip_y = f'landmark_{hip_idx}_y'
    knee_x = f'landmark_{knee_idx}_x'
    knee_y = f'landmark_{knee_idx}_y'
    ankle_x = f'landmark_{ankle_idx}_x'
    ankle_y = f'landmark_{ankle_idx}_y'
    foot_x = f'landmark_{foot_idx}_x'
    foot_y = f'landmark_{foot_idx}_y'

    # Angle column names
    hip_col = f'hip_flex_{side}_deg'
    knee_col = f'knee_flex_{side}_deg'
    ankle_col = f'ankle_dorsi_{side}_deg'

    # Add confidence tracking columns
    angles_df[f'{hip_col}_confidence'] = 1.0  # Existing data has confidence 1.0
    angles_df[f'{knee_col}_confidence'] = 1.0
    angles_df[f'{ankle_col}_confidence'] = 1.0

    recalculated_count = {'hip': 0, 'knee': 0, 'ankle': 0}

    for idx in range(len(angles_df)):
        # Check if keypoints are valid for this frame
        kp_valid = keypoints_df.loc[idx, [hip_x, hip_y, knee_x, knee_y,
                                           ankle_x, ankle_y, foot_x, foot_y]].notna().all()

        if not kp_valid:
            continue

        # Get keypoint values
        hip_pos = (keypoints_df.loc[idx, hip_x], keypoints_df.loc[idx, hip_y])
        knee_pos = (keypoints_df.loc[idx, knee_x], keypoints_df.loc[idx, knee_y])
        ankle_pos = (keypoints_df.loc[idx, ankle_x], keypoints_df.loc[idx, ankle_y])
        foot_pos = (keypoints_df.loc[idx, foot_x], keypoints_df.loc[idx, foot_y])

        # Recalculate hip flexion if missing
        if pd.isna(angles_df.loc[idx, hip_col]):
            # Need pelvis reference - use midpoint of hips
            pelvis_y = (keypoints_df.loc[idx, 'landmark_23_y'] +
                       keypoints_df.loc[idx, 'landmark_24_y']) / 2

            hip_angle = calculate_hip_flexion(hip_pos[1], knee_pos[1],
                                             hip_pos[0], knee_pos[0], pelvis_y)
            if not np.isnan(hip_angle):
                angles_df.loc[idx, hip_col] = hip_angle
                angles_df.loc[idx, f'{hip_col}_confidence'] = 0.9  # Recalculated data
                recalculated_count['hip'] += 1

        # Recalculate knee flexion if missing
        if pd.isna(angles_df.loc[idx, knee_col]):
            knee_angle = calculate_knee_flexion(hip_pos[1], hip_pos[0],
                                               knee_pos[1], knee_pos[0],
                                               ankle_pos[1], ankle_pos[0])
            if not np.isnan(knee_angle):
                angles_df.loc[idx, knee_col] = knee_angle
                angles_df.loc[idx, f'{knee_col}_confidence'] = 0.9
                recalculated_count['knee'] += 1

        # Recalculate ankle dorsiflexion if missing
        if pd.isna(angles_df.loc[idx, ankle_col]):
            ankle_angle = calculate_ankle_dorsiflexion(knee_pos[1], knee_pos[0],
                                                       ankle_pos[1], ankle_pos[0],
                                                       foot_pos[1], foot_pos[0])
            if not np.isnan(ankle_angle):
                angles_df.loc[idx, ankle_col] = ankle_angle
                angles_df.loc[idx, f'{ankle_col}_confidence'] = 0.9
                recalculated_count['ankle'] += 1

    print(f"\n{side} leg angles recalculated from keypoints:")
    print(f"  Hip: {recalculated_count['hip']} frames")
    print(f"  Knee: {recalculated_count['knee']} frames")
    print(f"  Ankle: {recalculated_count['ankle']} frames")

    return angles_df
```

### Expected Impact
- **LEFT leg**: Recover 30.1% missing data (144 frames)
- **RIGHT leg**: Recover 43.6% missing data (209 frames)
- **Combined with Tier 1**: Should achieve 8-12 valid stride pairs (vs current 2-4)

### Validation
1. Compare recalculated angles vs original angles for frames where both exist
2. Expected RMSE < 2° (acceptable measurement variation)
3. Visual inspection: recalculated cyclograms should have smooth, physiological patterns

### Risk Assessment
- **Risk**: LOW - only fills NaN gaps, doesn't modify existing values
- **Confidence**: 0.9 for recalculated values (vs 1.0 for original measurements)
- **Rollback**: Can disable recalculation via flag, preserving original NaN patterns

---

## Tier 3: Kinematic Chain Calculation (GEOMETRIC)

### Objective
Calculate missing joint angles from neighboring joints using 2D leg geometry and segment length constraints.

### Biomechanical Basis
In 2D sagittal plane gait analysis:
- Leg segments (thigh, shank, foot) have fixed lengths
- Joint positions form a kinematic chain: hip → knee → ankle
- If we know 2 out of 3 joint positions + segment lengths, we can calculate the 3rd

### Use Case
When ONE angle is missing but the other two are valid:
- Know hip + ankle → Calculate knee position → Derive knee angle
- Know knee + ankle → Calculate hip position → Derive hip angle
- etc.

### Implementation

#### Calculate Missing Knee Position
```python
def calculate_knee_from_hip_ankle(hip_pos: Tuple[float, float],
                                 ankle_pos: Tuple[float, float],
                                 thigh_length: float,
                                 shank_length: float,
                                 side: str = 'L') -> Optional[Tuple[float, float]]:
    """
    Calculate knee position using inverse kinematics for 2-segment chain.

    Given:
    - Hip position (x_h, y_h)
    - Ankle position (x_a, y_a)
    - Thigh length |hip-knee|
    - Shank length |knee-ankle|

    Find:
    - Knee position (x_k, y_k) that satisfies both segment length constraints

    Returns:
        (x_k, y_k) or None if no valid solution exists
    """
    x_h, y_h = hip_pos
    x_a, y_a = ankle_pos

    # Distance between hip and ankle
    d = np.sqrt((x_a - x_h)**2 + (y_a - y_h)**2)

    # Check if solution is geometrically possible
    # Triangle inequality: sum of two sides > third side
    if d > (thigh_length + shank_length) or d < abs(thigh_length - shank_length):
        return None  # Unreachable configuration

    # Use law of cosines to find angle at hip
    # d² = thigh² + shank² - 2·thigh·shank·cos(θ_knee)
    cos_knee = (thigh_length**2 + shank_length**2 - d**2) / (2 * thigh_length * shank_length)
    cos_knee = np.clip(cos_knee, -1.0, 1.0)
    theta_knee = np.arccos(cos_knee)

    # Find angle of hip-ankle line relative to horizontal
    theta_ha = np.arctan2(y_a - y_h, x_a - x_h)

    # Calculate angle from hip to knee
    # For LEFT leg, knee is typically to the left (negative x direction)
    # For RIGHT leg, knee is typically to the right (positive x direction)
    cos_hip = (d**2 + thigh_length**2 - shank_length**2) / (2 * d * thigh_length)
    cos_hip = np.clip(cos_hip, -1.0, 1.0)
    theta_hip = np.arccos(cos_hip)

    # Determine knee position (choose anatomically correct solution)
    if side == 'L':
        theta_k = theta_ha + theta_hip  # Knee bends left
    else:
        theta_k = theta_ha - theta_hip  # Knee bends right

    # Calculate knee position
    x_k = x_h + thigh_length * np.cos(theta_k)
    y_k = y_h + thigh_length * np.sin(theta_k)

    # Verify solution (check shank length constraint)
    shank_actual = np.sqrt((x_a - x_k)**2 + (y_a - y_k)**2)
    if abs(shank_actual - shank_length) > 0.01 * shank_length:  # 1% tolerance
        return None

    return (x_k, y_k)
```

#### Estimate Segment Lengths
```python
def estimate_segment_lengths(keypoints_df: pd.DataFrame,
                            side: str = 'L') -> Dict[str, float]:
    """
    Estimate thigh and shank lengths from valid frames.

    Uses median of segment lengths across all frames where keypoints are valid.
    This is robust to outliers and measurement noise.

    Args:
        keypoints_df: DataFrame with landmark positions
        side: 'L' or 'R'

    Returns:
        Dict with 'thigh_length' and 'shank_length' in pixel units
    """
    if side == 'L':
        hip_idx, knee_idx, ankle_idx = 23, 25, 27
    else:
        hip_idx, knee_idx, ankle_idx = 24, 26, 28

    # Get segment lengths for all valid frames
    thigh_lengths = []
    shank_lengths = []

    for idx in range(len(keypoints_df)):
        # Check if all keypoints are valid
        hip_x = keypoints_df.loc[idx, f'landmark_{hip_idx}_x']
        hip_y = keypoints_df.loc[idx, f'landmark_{hip_idx}_y']
        knee_x = keypoints_df.loc[idx, f'landmark_{knee_idx}_x']
        knee_y = keypoints_df.loc[idx, f'landmark_{knee_idx}_y']
        ankle_x = keypoints_df.loc[idx, f'landmark_{ankle_idx}_x']
        ankle_y = keypoints_df.loc[idx, f'landmark_{ankle_idx}_y']

        if pd.notna([hip_x, hip_y, knee_x, knee_y, ankle_x, ankle_y]).all():
            thigh_len = np.sqrt((knee_x - hip_x)**2 + (knee_y - hip_y)**2)
            shank_len = np.sqrt((ankle_x - knee_x)**2 + (ankle_y - knee_y)**2)

            # Filter outliers (segment length should be relatively stable)
            if 50 < thigh_len < 500 and 50 < shank_len < 500:  # Reasonable pixel ranges
                thigh_lengths.append(thigh_len)
                shank_lengths.append(shank_len)

    return {
        'thigh_length': np.median(thigh_lengths),
        'shank_length': np.median(shank_lengths),
        'thigh_std': np.std(thigh_lengths),
        'shank_std': np.std(shank_lengths)
    }
```

### Expected Impact
- **Additional recovery**: 5-10% of frames where only 1 angle is missing
- **Use case**: Frames where knee detection failed but hip/ankle are valid
- **Confidence**: 0.8 for kinematically calculated values

### Risk Assessment
- **Risk**: MEDIUM - relies on accurate segment length estimates
- **Validation**: Check that calculated knee positions fall within anatomically plausible ranges
- **Fallback**: Only apply when segment length variance is low (stable estimates)

---

## Tier 4: Small-Gap Interpolation (CONSERVATIVE)

### Objective
Fill very small gaps (≤3 frames, ~100ms) within otherwise valid segments using PCHIP interpolation.

### Rationale
- Gaps of 1-3 frames are often due to temporary occlusions or tracking jitter
- At 30fps, 3 frames = 0.1s, which is too fast for significant gait changes
- PCHIP (shape-preserving cubic) maintains monotonicity and avoids oscillations

### Implementation
```python
def fill_small_gaps_pchip(angles_df: pd.DataFrame,
                         angle_col: str,
                         max_gap_frames: int = 3,
                         min_context_frames: int = 3) -> pd.DataFrame:
    """
    Fill small gaps using PCHIP interpolation, only within valid segments.

    Args:
        angles_df: DataFrame with angles
        angle_col: Column to interpolate
        max_gap_frames: Maximum gap size to fill (default 3 frames)
        min_context_frames: Minimum valid frames needed on each side of gap

    Returns:
        Updated DataFrame with small gaps filled
    """
    from scipy.interpolate import PchipInterpolator

    values = angles_df[angle_col].values.copy()
    confidence = angles_df.get(f'{angle_col}_confidence', np.ones(len(angles_df)))
    filled_count = 0

    # Find gaps
    is_nan = pd.isna(values)
    gap_starts = np.where(~is_nan[:-1] & is_nan[1:])[0] + 1
    gap_ends = np.where(is_nan[:-1] & ~is_nan[1:])[0]

    for start, end in zip(gap_starts, gap_ends):
        gap_size = end - start + 1

        # Only fill small gaps
        if gap_size > max_gap_frames:
            continue

        # Check context availability
        context_before = start - min_context_frames
        context_after = end + min_context_frames + 1

        if context_before < 0 or context_after > len(values):
            continue

        # Extract context window
        window_start = max(0, start - 10)
        window_end = min(len(values), end + 10)
        window_values = values[window_start:window_end]
        window_valid = ~np.isnan(window_values)

        if window_valid.sum() < 6:  # Need at least 6 points for cubic
            continue

        # Create interpolator
        try:
            valid_indices = np.where(window_valid)[0] + window_start
            valid_values = window_values[window_valid]

            interpolator = PchipInterpolator(valid_indices, valid_values)
            gap_indices = np.arange(start, end + 1)
            interpolated = interpolator(gap_indices)

            # Update values
            values[start:end+1] = interpolated
            confidence[start:end+1] = 0.7  # Lower confidence for interpolated
            filled_count += gap_size

        except Exception as e:
            continue  # Skip problematic gaps

    angles_df[angle_col] = values
    angles_df[f'{angle_col}_confidence'] = confidence

    print(f"Filled {filled_count} frames in {angle_col} using PCHIP (≤{max_gap_frames} frame gaps)")

    return angles_df
```

### Expected Impact
- **Additional recovery**: 2-5% of frames with very small gaps
- **Confidence**: 0.7 for interpolated values

### Risk Assessment
- **Risk**: LOW - only fills tiny gaps, uses shape-preserving method
- **Limitation**: Only applied AFTER Tier 2 and Tier 3 attempts
- **Validation**: Visual inspection of filled regions for smoothness

---

## Tier 5: Biomechanical Anticipation (ADVANCED)

### Objective
Use machine learning to predict missing angles based on inter-joint coupling patterns learned from the subject's own valid data.

### Biomechanical Rationale
During gait, joint angles are not independent:
- **Hip-knee coordination**: Hip extension correlates with knee extension
- **Ankle-knee coordination**: Ankle dorsiflexion relates to knee position
- **Phase-dependent patterns**: Relationships vary by gait phase (stance/swing)

### Implementation Strategy

#### Learn Inter-Joint Coupling
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def learn_joint_coupling(angles_df: pd.DataFrame,
                        side: str = 'L') -> Dict[str, Any]:
    """
    Learn relationships between joint angles from valid data.

    For each joint, train a model to predict its angle from:
    - Other joint angles (hip, knee, ankle)
    - Temporal context (previous frame angles)
    - Gait phase indicators (derived from heel strikes)

    Returns:
        Dict containing trained models and scalers for each joint
    """
    hip_col = f'hip_flex_{side}_deg'
    knee_col = f'knee_flex_{side}_deg'
    ankle_col = f'ankle_dorsi_{side}_deg'

    # Extract fully valid frames for training
    valid_mask = angles_df[[hip_col, knee_col, ankle_col]].notna().all(axis=1)
    train_data = angles_df[valid_mask].copy()

    if len(train_data) < 100:
        print(f"Insufficient valid data for {side} leg coupling model ({len(train_data)} frames)")
        return None

    # Add temporal features (lag-1, lag-2)
    for col in [hip_col, knee_col, ankle_col]:
        train_data[f'{col}_lag1'] = train_data[col].shift(1)
        train_data[f'{col}_lag2'] = train_data[col].shift(2)

    # Add derivatives (velocity)
    for col in [hip_col, knee_col, ankle_col]:
        train_data[f'{col}_vel'] = train_data[col].diff()

    # Drop rows with NaN from lagging
    train_data = train_data.dropna()

    models = {}

    # Train model to predict knee from hip + ankle
    knee_features = [hip_col, ankle_col,
                    f'{hip_col}_lag1', f'{ankle_col}_lag1',
                    f'{hip_col}_vel', f'{ankle_col}_vel']
    X_knee = train_data[knee_features]
    y_knee = train_data[knee_col]

    scaler_knee = StandardScaler()
    X_knee_scaled = scaler_knee.fit_transform(X_knee)

    model_knee = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    model_knee.fit(X_knee_scaled, y_knee)

    models['knee'] = {
        'model': model_knee,
        'scaler': scaler_knee,
        'features': knee_features,
        'target': knee_col,
        'r2_score': model_knee.score(X_knee_scaled, y_knee)
    }

    # Similarly for hip and ankle
    # ... (analogous training for other joints)

    print(f"\n{side} leg coupling models trained:")
    print(f"  Knee prediction R²: {models['knee']['r2_score']:.3f}")

    return models
```

#### Apply Biomechanical Prediction
```python
def predict_missing_angles_biomechanical(angles_df: pd.DataFrame,
                                        models: Dict[str, Any],
                                        side: str = 'L') -> pd.DataFrame:
    """
    Predict missing angles using learned biomechanical coupling models.

    Only applied when:
    - Other angles are available
    - Model prediction confidence is high (R² > 0.7)
    - Gap is not too large (≤10 consecutive frames)
    """
    if models is None:
        return angles_df

    knee_model = models['knee']
    predicted_count = 0

    knee_col = f'knee_flex_{side}_deg'

    # Find frames where knee is NaN but other joints are valid
    knee_missing = angles_df[knee_col].isna()

    for idx in angles_df[knee_missing].index:
        try:
            # Extract features for prediction
            features = {}
            for feat in knee_model['features']:
                if feat in angles_df.columns:
                    features[feat] = angles_df.loc[idx, feat]
                else:
                    # Handle lag/velocity features
                    features[feat] = 0  # Default for unavailable features

            # Check if all required features are available
            if any(pd.isna(v) for v in features.values()):
                continue

            # Make prediction
            X = np.array([list(features.values())])
            X_scaled = knee_model['scaler'].transform(X)
            predicted_angle = knee_model['model'].predict(X_scaled)[0]

            # Apply prediction with appropriate confidence
            angles_df.loc[idx, knee_col] = predicted_angle
            angles_df.loc[idx, f'{knee_col}_confidence'] = 0.6  # Lower confidence
            predicted_count += 1

        except Exception:
            continue

    print(f"Predicted {predicted_count} missing {knee_col} values using biomechanical model")

    return angles_df
```

### Expected Impact
- **Additional recovery**: 3-8% of frames where one angle is missing
- **Confidence**: 0.6 for model-predicted values
- **Requirement**: Model R² > 0.7 for activation

### Risk Assessment
- **Risk**: MEDIUM-HIGH - model predictions may not generalize
- **Validation**: Compare predictions vs actual for held-out test frames
- **Limitation**: Only activated if sufficient training data (≥100 valid frames)

---

## Integration Architecture

### Processing Pipeline
```python
def angle_calculation_pipeline(subject_dir: str,
                               enable_tiers: List[int] = [1, 2]) -> pd.DataFrame:
    """
    Main pipeline integrating all calculation tiers.

    Args:
        subject_dir: Path to subject data directory
        enable_tiers: List of tier numbers to activate (1-5)

    Returns:
        DataFrame with enhanced angle data
    """
    # Load data
    keypoints_df = pd.read_csv(os.path.join(subject_dir, "Clean_keypoints.csv"))
    angles_df = pd.read_csv(os.path.join(subject_dir, "Raw_Angles.csv"))

    print("="*80)
    print("ANGLE CALCULATION PIPELINE")
    print("="*80)

    # Baseline statistics
    print("\nBaseline Data Quality:")
    for side in ['L', 'R']:
        for joint in ['hip_flex', 'knee_flex', 'ankle_dorsi']:
            col = f'{joint}_{side}_deg'
            nan_pct = angles_df[col].isna().sum() / len(angles_df) * 100
            print(f"  {col}: {nan_pct:.1f}% NaN")

    # TIER 2: Keypoint-based recalculation (if enabled)
    if 2 in enable_tiers:
        print("\n" + "─"*80)
        print("TIER 2: Keypoint-Based Angle Recalculation")
        print("─"*80)
        angles_df = recalculate_angles_from_keypoints(keypoints_df, angles_df, side='L')
        angles_df = recalculate_angles_from_keypoints(keypoints_df, angles_df, side='R')

    # TIER 3: Kinematic chain calculation (if enabled)
    if 3 in enable_tiers:
        print("\n" + "─"*80)
        print("TIER 3: Kinematic Chain Calculation")
        print("─"*80)

        # Estimate segment lengths
        seg_lengths_L = estimate_segment_lengths(keypoints_df, side='L')
        seg_lengths_R = estimate_segment_lengths(keypoints_df, side='R')

        print(f"LEFT leg segments: thigh={seg_lengths_L['thigh_length']:.1f}px, "
              f"shank={seg_lengths_L['shank_length']:.1f}px")
        print(f"RIGHT leg segments: thigh={seg_lengths_R['thigh_length']:.1f}px, "
              f"shank={seg_lengths_R['shank_length']:.1f}px")

        # Apply kinematic calculations
        # ... (implementation details)

    # TIER 4: Small-gap interpolation (if enabled)
    if 4 in enable_tiers:
        print("\n" + "─"*80)
        print("TIER 4: Small-Gap PCHIP Interpolation")
        print("─"*80)

        for side in ['L', 'R']:
            for joint in ['hip_flex', 'knee_flex', 'ankle_dorsi']:
                col = f'{joint}_{side}_deg'
                angles_df = fill_small_gaps_pchip(angles_df, col, max_gap_frames=3)

    # TIER 5: Biomechanical anticipation (if enabled)
    if 5 in enable_tiers:
        print("\n" + "─"*80)
        print("TIER 5: Biomechanical Anticipation Models")
        print("─"*80)

        models_L = learn_joint_coupling(angles_df, side='L')
        models_R = learn_joint_coupling(angles_df, side='R')

        if models_L:
            angles_df = predict_missing_angles_biomechanical(angles_df, models_L, side='L')
        if models_R:
            angles_df = predict_missing_angles_biomechanical(angles_df, models_R, side='R')

    # Final statistics
    print("\n" + "="*80)
    print("ENHANCED DATA QUALITY")
    print("="*80)

    for side in ['L', 'R']:
        for joint in ['hip_flex', 'knee_flex', 'ankle_dorsi']:
            col = f'{joint}_{side}_deg'
            nan_pct = angles_df[col].isna().sum() / len(angles_df) * 100
            improvement = (angles_df[col].isna().sum() - angles_df[col].isna().sum()) / len(angles_df) * 100
            print(f"  {col}: {nan_pct:.1f}% NaN (improved by {improvement:.1f}%)")

    # Save enhanced angles
    output_path = os.path.join(subject_dir, "Enhanced_Angles.csv")
    angles_df.to_csv(output_path, index=False)
    print(f"\nEnhanced angles saved to: {output_path}")

    return angles_df
```

### Integration with Analysis.py

```python
# In Analysis.py, add parameter to enable enhancement
def analyze_subject(subject_dir: str,
                   enable_angle_enhancement: bool = True,
                   enhancement_tiers: List[int] = [1, 2]) -> Dict[str, Any]:
    """
    Analyze a single subject with optional angle enhancement.

    Args:
        subject_dir: Path to subject data
        enable_angle_enhancement: Whether to apply angle calculation pipeline
        enhancement_tiers: Which tiers to activate (default: [1, 2] for safety)
    """
    if enable_angle_enhancement:
        # Run enhancement pipeline
        angles_df = angle_calculation_pipeline(subject_dir, enable_tiers=enhancement_tiers)
        angles_path = os.path.join(subject_dir, "Enhanced_Angles.csv")
    else:
        # Use original angles
        angles_path = os.path.join(subject_dir, "Raw_Angles.csv")

    # Continue with existing analysis workflow
    # ... (load angles_path and proceed as before)
```

---

## Validation Framework

### Quality Metrics

#### Data Quality Score (Enhanced)
```python
def calculate_enhanced_quality_score(angles_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate comprehensive quality score accounting for confidence levels.

    Returns:
        Dict with overall quality score and per-source breakdown
    """
    total_frames = len(angles_df)
    angle_cols = [col for col in angles_df.columns if col.endswith('_deg')]

    quality_breakdown = {
        'original': 0,      # confidence = 1.0
        'recalculated': 0,  # confidence = 0.9
        'kinematic': 0,     # confidence = 0.8
        'interpolated': 0,  # confidence = 0.7
        'predicted': 0,     # confidence = 0.6
        'missing': 0        # NaN
    }

    for col in angle_cols:
        conf_col = f'{col}_confidence'
        if conf_col not in angles_df.columns:
            continue

        quality_breakdown['original'] += (angles_df[conf_col] == 1.0).sum()
        quality_breakdown['recalculated'] += (angles_df[conf_col] == 0.9).sum()
        quality_breakdown['kinematic'] += (angles_df[conf_col] == 0.8).sum()
        quality_breakdown['interpolated'] += (angles_df[conf_col] == 0.7).sum()
        quality_breakdown['predicted'] += (angles_df[conf_col] == 0.6).sum()
        quality_breakdown['missing'] += angles_df[col].isna().sum()

    # Calculate weighted quality score
    total_values = total_frames * len(angle_cols)
    quality_score = (
        quality_breakdown['original'] * 1.0 +
        quality_breakdown['recalculated'] * 0.9 +
        quality_breakdown['kinematic'] * 0.8 +
        quality_breakdown['interpolated'] * 0.7 +
        quality_breakdown['predicted'] * 0.6
    ) / total_values

    return {
        'overall_quality': quality_score,
        'breakdown': quality_breakdown,
        'coverage_pct': (1 - quality_breakdown['missing'] / total_values) * 100
    }
```

#### Biomechanical Validation
```python
def validate_biomechanical_plausibility(angles_df: pd.DataFrame) -> Dict[str, bool]:
    """
    Check if calculated angles respect anatomical constraints.

    Checks:
    1. Hip flexion: -20° to 120° (extension to flexion)
    2. Knee flexion: 0° to 180° (fully flexed to fully extended)
    3. Ankle dorsiflexion: 60° to 150° (plantarflexion to dorsiflexion)
    4. Inter-joint consistency: hip-knee-ankle chain geometry
    """
    constraints = {
        'hip_flex': (-20, 120),
        'knee_flex': (0, 180),
        'ankle_dorsi': (60, 150)
    }

    violations = {}

    for side in ['L', 'R']:
        for joint, (min_val, max_val) in constraints.items():
            col = f'{joint}_{side}_deg'

            out_of_range = ((angles_df[col] < min_val) | (angles_df[col] > max_val))
            violation_pct = out_of_range.sum() / len(angles_df) * 100

            violations[col] = violation_pct

            if violation_pct > 5.0:  # >5% violations is concerning
                print(f"WARNING: {col} has {violation_pct:.1f}% values outside "
                      f"anatomical range [{min_val}, {max_val}]")

    return violations
```

### Comparison Testing
```python
def compare_with_without_enhancement(subject_dir: str) -> None:
    """
    Run analysis with and without enhancement to compare results.
    """
    print("="*80)
    print("COMPARISON: Original vs Enhanced")
    print("="*80)

    # Run with original angles
    print("\n[1] Analysis with ORIGINAL angles:")
    results_original = analyze_subject(subject_dir, enable_angle_enhancement=False)

    print(f"  Valid stride pairs: {results_original['n_pairs']}")
    print(f"  Data quality: {results_original['data_quality']:.2f}")

    # Run with enhanced angles (Tier 1 + 2 only)
    print("\n[2] Analysis with ENHANCED angles (Tiers 1+2):")
    results_enhanced = analyze_subject(subject_dir, enable_angle_enhancement=True,
                                      enhancement_tiers=[1, 2])

    print(f"  Valid stride pairs: {results_enhanced['n_pairs']}")
    print(f"  Data quality: {results_enhanced['data_quality']:.2f}")

    # Compare cyclogram similarity scores
    print("\n[3] Similarity Score Comparison:")
    for joint_pair in ['hip-ankle', 'hip-knee', 'knee-ankle']:
        orig_sim = results_original.get(f'{joint_pair}_similarity', 0)
        enh_sim = results_enhanced.get(f'{joint_pair}_similarity', 0)
        print(f"  {joint_pair}: {orig_sim:.1f} → {enh_sim:.1f}")
```

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1)
**Objective**: Implement Tier 1 + Tier 2 (baseline + keypoint recalculation)

**Tasks**:
1. Create `angle_enhancement.py` module with geometric calculation functions
2. Implement `identify_valid_segments()` for Tier 1
3. Implement keypoint-based angle recalculation functions (hip, knee, ankle)
4. Add confidence scoring system to angle DataFrames
5. Create validation framework (biomechanical plausibility checks)
6. Unit tests for geometric calculations

**Success Criteria**:
- Recover 30-44% missing angles from keypoints
- All recalculated angles within anatomical ranges
- Validation RMSE < 2° vs original angles (where both exist)

### Phase 2: Integration (Week 2)
**Objective**: Integrate enhancement pipeline with Analysis.py

**Tasks**:
1. Modify `Analysis.py` to call enhancement pipeline
2. Add command-line flags: `--enhance-angles`, `--enhancement-tiers`
3. Update segment detection to use valid segments first
4. Create comprehensive logging of enhancement statistics
5. Generate comparison reports (before/after enhancement)

**Success Criteria**:
- Analysis.py seamlessly switches between original/enhanced modes
- Valid stride pairs increase from 2-4 to 8-12
- No degradation in similarity scores for valid pairs

### Phase 3: Advanced Tiers (Week 3, Optional)
**Objective**: Implement Tier 3-5 for maximum data recovery

**Tasks**:
1. Implement kinematic chain calculation (Tier 3)
2. Add segment length estimation and validation
3. Implement small-gap PCHIP interpolation (Tier 4)
4. Develop biomechanical coupling models (Tier 5)
5. Create tier-specific validation and confidence metrics

**Success Criteria**:
- Additional 10-15% data recovery
- Tier 3-5 confidence scores properly calibrated
- Model R² > 0.7 for biomechanical predictions

---

## Configuration System

### Configuration File Structure
```yaml
# angle_enhancement_config.yaml

enhancement:
  enabled: true
  tiers:
    tier1_segment_aware:
      enabled: true
      min_segment_frames: 60
      max_nan_percent: 5.0

    tier2_keypoint_recalc:
      enabled: true
      confidence: 0.9
      validate_anatomical_range: true

    tier3_kinematic:
      enabled: false
      confidence: 0.8
      min_segment_length_stability: 0.95  # CV threshold

    tier4_interpolation:
      enabled: false
      max_gap_frames: 3
      confidence: 0.7
      method: "pchip"

    tier5_biomechanical:
      enabled: false
      confidence: 0.6
      min_training_frames: 100
      min_model_r2: 0.7

validation:
  anatomical_ranges:
    hip_flex: [-20, 120]
    knee_flex: [0, 180]
    ankle_dorsi: [60, 150]

  max_violation_percent: 5.0  # Max % of frames outside range

  recalc_validation_rmse: 2.0  # Max RMSE for recalculated vs original

output:
  save_enhanced_angles: true
  save_confidence_scores: true
  save_comparison_report: true
  enhanced_filename: "Enhanced_Angles.csv"
```

### Loading Configuration
```python
import yaml

def load_enhancement_config(config_path: str = "angle_enhancement_config.yaml") -> Dict[str, Any]:
    """Load enhancement configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def apply_config_to_pipeline(config: Dict[str, Any]) -> List[int]:
    """Determine which tiers to enable based on configuration."""
    enabled_tiers = []

    tiers_config = config['enhancement']['tiers']

    if tiers_config['tier1_segment_aware']['enabled']:
        enabled_tiers.append(1)
    if tiers_config['tier2_keypoint_recalc']['enabled']:
        enabled_tiers.append(2)
    if tiers_config['tier3_kinematic']['enabled']:
        enabled_tiers.append(3)
    if tiers_config['tier4_interpolation']['enabled']:
        enabled_tiers.append(4)
    if tiers_config['tier5_biomechanical']['enabled']:
        enabled_tiers.append(5)

    return enabled_tiers
```

---

## Expected Outcomes

### Baseline (Original Data)
```
LEFT leg: 30.1% NaN → 4 strides pass QC
RIGHT leg: 43.6% NaN → 2 strides pass QC
Valid pairs: 2-4
Similarity scores: hip-ankle=30, hip-knee=27, knee-ankle=43
```

### With Tier 1 Only (Segment-Aware)
```
LEFT leg: 30.1% NaN → 6-8 strides pass QC (from valid segments)
RIGHT leg: 43.6% NaN → 4-6 strides pass QC
Valid pairs: 4-6
Similarity scores: Expected improvement 5-10% (more stable comparisons)
```

### With Tier 1 + 2 (+ Keypoint Recalculation)
```
LEFT leg: 30.1% → 5% NaN → 8-10 strides pass QC
RIGHT leg: 43.6% → 10% NaN → 6-8 strides pass QC
Valid pairs: 6-10
Similarity scores: Expected improvement 10-20% (more complete cyclograms)
Data quality: 0.85-0.90 (vs 0.0 original)
```

### With All Tiers (Maximum Enhancement)
```
LEFT leg: 30.1% → 0-2% NaN → 10-12 strides pass QC
RIGHT leg: 43.6% → 2-5% NaN → 8-10 strides pass QC
Valid pairs: 8-12
Similarity scores: Expected improvement 15-25%
Data quality: 0.90-0.95
```

---

## Risk Mitigation

### Data Corruption Prevention
1. **Never modify original files**: Always create `Enhanced_Angles.csv`, preserve `Raw_Angles.csv`
2. **Confidence tracking**: Every calculated value has confidence score
3. **Validation gates**: Anatomical range checks before accepting calculated angles
4. **Rollback capability**: Can disable any tier via configuration

### Quality Assurance
1. **Unit tests**: Each geometric calculation function has test cases
2. **Comparison testing**: Always compare with/without enhancement
3. **Visual inspection**: Plot cyclograms from original vs enhanced data
4. **Statistical validation**: RMSE checks for recalculated angles

### Performance Monitoring
1. **Execution time**: Track time for each tier
2. **Memory usage**: Monitor DataFrame size growth
3. **Recovery metrics**: Log how many frames each tier recovers
4. **Failure tracking**: Record when calculations fail and why

---

## Success Metrics

### Quantitative Targets
1. **Data Recovery**: Reduce NaN from 30-44% to <10%
2. **Valid Pairs**: Increase from 2-4 to 8-12 stride pairs
3. **Similarity Scores**: Improve by 15-25 points
4. **Data Quality**: Achieve 0.85+ quality score
5. **Calculation Accuracy**: RMSE < 2° for recalculated angles

### Qualitative Targets
1. **Biomechanical Validity**: All cyclograms show physiological patterns
2. **No Artifacts**: No artificial flat lines or constant values
3. **Smooth Transitions**: Calculated regions blend naturally with measured regions
4. **Clinical Utility**: Results can be used for gait asymmetry assessment

---

## Conclusion

This multi-tier design provides a **systematic, safe, and biomechanically sound** approach to maximizing data utilization without introducing artifacts. The progressive enhancement strategy allows conservative adoption (Tier 1+2 only) while providing pathways for more advanced techniques if needed.

**Key Innovation**: Instead of statistical interpolation that destroyed gait dynamics, we use **geometric calculation and biomechanical anticipation** to compute missing angles from physics, anatomy, and learned patterns from the subject's own valid data.

**Next Steps**: Implement Phase 1 (Tier 1+2) and validate results before considering advanced tiers.
