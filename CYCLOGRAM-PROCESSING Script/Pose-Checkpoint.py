#!/usr/bin/env python3
"""
Cyclogram Analysis System - Stride-wise Joint Angle Phase Plots with Left-Right Comparison
This script performs comprehensive gait analysis by generating cyclograms (angle-angle diagrams)
for each stride and computing quantitative left-right asymmetry metrics.
Author: Claude (Anthropic)
Date: 2025-10-10/
Version: 1.0
"""

from __future__ import annotations  # Enable forward references for type hints

# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
# CONFIGURATION - Modify these paths as needed
# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

INPUT_DIR = "/home/shivam/Desktop/Human_Pose/temp/front"
OUTPUT_DIR = "/home/shivam/Desktop/Human_Pose/temp/cyclogram"

# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
# IMPORTS
# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

import os
import re
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator  # use PCHIP only (shape-preserving)
from math import isfinite
from scipy.spatial import procrustes, distance as spatial_distance
from sklearn.decomposition import PCA

# Try to import fastdtw, fallback to scipy if not available
try:
    from fastdtw import fastdtw
    HAS_FASTDTW = True
except ImportError:
    HAS_FASTDTW = False
    print("Warning: fastdtw not installed. Install with 'pip install fastdtw' for DTW metrics.")

warnings.filterwarnings('ignore')

# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
# ANGLE ENHANCEMENT FUNCTIONS
# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

def calculate_knee_flexion(hip_y: float, hip_x: float,
                          knee_y: float, knee_x: float,
                          ankle_y: float, ankle_x: float) -> float:
    """Calculate knee flexion angle from keypoint positions."""
    thigh_vec = np.array([hip_x - knee_x, hip_y - knee_y])
    shank_vec = np.array([ankle_x - knee_x, ankle_y - knee_y])

    dot_product = np.dot(thigh_vec, shank_vec)
    mag_product = np.linalg.norm(thigh_vec) * np.linalg.norm(shank_vec)

    if mag_product < 1e-6:
        return np.nan

    angle_rad = np.arccos(np.clip(dot_product / mag_product, -1.0, 1.0))
    interior_angle = np.degrees(angle_rad)

    return 180.0 - interior_angle


def calculate_hip_flexion(hip_y: float, knee_y: float,
                         hip_x: float, knee_x: float,
                         pelvis_y: float) -> float:
    """
    Calculate hip flexion angle from keypoint positions.

    Uses atan2 for full 360° range to avoid coordinate wrapping issues.
    Angle relative to vertical (pelvis reference).
    """
    thigh_vec = np.array([knee_x - hip_x, knee_y - hip_y])

    # Check for degenerate case
    if np.linalg.norm(thigh_vec) < 1e-6:
        return np.nan

    # Use atan2 for full angular range
    angle_rad = np.arctan2(thigh_vec[1], thigh_vec[0])
    angle_deg = np.degrees(angle_rad)

    # Convert to anatomical hip flexion (0° = vertical, positive = flexion)
    # Add 90° to shift from horizontal reference to vertical
    hip_flexion = 90.0 - angle_deg

    # Normalize to 0-180° range
    if hip_flexion < 0:
        hip_flexion += 360.0
    if hip_flexion > 180:
        hip_flexion = 360.0 - hip_flexion

    return hip_flexion


def calculate_ankle_dorsiflexion(knee_y: float, knee_x: float,
                                 ankle_y: float, ankle_x: float,
                                 foot_y: float, foot_x: float) -> float:
    """Calculate ankle dorsiflexion angle from keypoint positions."""
    shank_vec = np.array([knee_x - ankle_x, knee_y - ankle_y])
    foot_vec = np.array([foot_x - ankle_x, foot_y - ankle_y])

    dot_product = np.dot(shank_vec, foot_vec)
    mag_product = np.linalg.norm(shank_vec) * np.linalg.norm(foot_vec)

    if mag_product < 1e-6:
        return np.nan

    angle_rad = np.arccos(np.clip(dot_product / mag_product, -1.0, 1.0))
    return np.degrees(angle_rad)


def smooth_angle_series(angles: np.ndarray,
                        confidence: np.ndarray,
                        window: int = 5,
                        max_jump: float = 15.0) -> np.ndarray:
    """
    Apply temporal smoothing to angle series with discontinuity detection.

    Args:
        angles: Angle values (may contain NaN)
        confidence: Confidence scores (1.0=original, 0.9=recalculated)
        window: Smoothing window size
        max_jump: Maximum allowed frame-to-frame change (degrees)

    Returns:
        Smoothed angle array
    """
    smoothed = angles.copy()
    valid_mask = ~np.isnan(angles)

    if valid_mask.sum() < window:
        return smoothed

    # Apply Savitzky-Golay filter to valid regions
    try:
        # Find continuous valid segments
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) < window:
            return smoothed

        # Identify segment boundaries (gaps > 1 frame)
        segment_breaks = np.where(np.diff(valid_indices) > 1)[0] + 1
        segments = np.split(valid_indices, segment_breaks)

        for segment in segments:
            if len(segment) >= window:
                # Apply smoothing to this segment
                segment_angles = angles[segment]
                segment_conf = confidence[segment]

                # Use weighted smoothing (trust original data more)
                weights = segment_conf

                # Apply Savitzky-Golay filter
                # Window must be odd and >= polyorder + 2
                actual_window = min(window, len(segment))
                if actual_window % 2 == 0:
                    actual_window -= 1
                actual_window = max(3, actual_window)  # Minimum 3

                smoothed_segment = savgol_filter(segment_angles,
                                                actual_window,
                                                polyorder=2)

                # Blend with original using confidence weights
                smoothed[segment] = (weights * segment_angles +
                                    (1 - weights * 0.3) * smoothed_segment)

        # Post-smoothing: clip large jumps
        for i in range(1, len(smoothed)):
            if valid_mask[i] and valid_mask[i-1]:
                jump = abs(smoothed[i] - smoothed[i-1])
                if jump > max_jump:
                    # Limit jump size
                    direction = np.sign(smoothed[i] - smoothed[i-1])
                    smoothed[i] = smoothed[i-1] + direction * max_jump

    except Exception as e:
        print(f"    Warning: Smoothing failed: {e}")
        return angles

    return smoothed


def fill_small_gaps(angles: np.ndarray,
                    max_gap_size: int = 3) -> np.ndarray:
    """
    Fill small gaps using linear interpolation.

    Args:
        angles: Angle array with NaN gaps
        max_gap_size: Maximum gap size to fill (frames)

    Returns:
        Angles with small gaps filled
    """
    filled = angles.copy()
    nan_mask = np.isnan(filled)

    if not nan_mask.any():
        return filled

    # Find NaN runs
    nan_indices = np.where(nan_mask)[0]

    if len(nan_indices) == 0:
        return filled

    # Identify gap boundaries
    gap_starts = np.where(np.diff(np.concatenate([[False], nan_mask])) == 1)[0]
    gap_ends = np.where(np.diff(np.concatenate([nan_mask, [False]])) == -1)[0]

    for start, end in zip(gap_starts, gap_ends):
        gap_size = end - start

        if gap_size <= max_gap_size:
            # Check if we have valid boundaries
            if start > 0 and end < len(filled):
                # Linear interpolation
                filled[start:end] = np.linspace(filled[start-1],
                                               filled[end],
                                               gap_size + 2)[1:-1]

    return filled


def enhance_angles_from_keypoints(subject_dir: Path,
                                  angles_df: pd.DataFrame,
                                  config: Optional['AnalysisConfig'] = None,
                                  confidence_level: float = 0.9,
                                  smooth_window: int = 5,
                                  max_jump_deg: float = 15.0,
                                  fill_small_gaps_flag: bool = True) -> pd.DataFrame:
    """
    Enhanced angle recalculation with temporal smoothing and intelligent gap filling.

    Multi-tier strategy:
    1. Fill gaps with PCHIP interpolation (shape-preserving, no overshoots)
    2. Recalculate missing angles from keypoint geometry
    3. Apply temporal smoothing with confidence weighting
    4. Validate temporal coherence (reject large jumps)

    Args:
        subject_dir: Path to subject directory containing Clean_keypoints.csv
        angles_df: DataFrame with angle data (may have NaN values)
        confidence_level: Confidence score for recalculated angles (default 0.9)
        smooth_window: Temporal smoothing window size (default 5)
        max_jump_deg: Maximum allowed frame-to-frame change (default 15.0)
        fill_small_gaps_flag: Enable small gap interpolation (default True)

    Returns:
        Enhanced angles DataFrame with improved temporal coherence
    """
    keypoints_path = subject_dir / "Clean_keypoints.csv"

    if not keypoints_path.exists():
        print(f"  Warning: Clean_keypoints.csv not found, skipping enhancement")
        return angles_df

    try:
        keypoints_df = pd.read_csv(keypoints_path)
    except Exception as e:
        print(f"  Warning: Could not load keypoints: {e}")
        return angles_df

    print(f"  Enhancing angles from keypoints (multi-tier strategy)...")

    # Process each leg
    for side in ['L', 'R']:
        # MediaPipe landmark indices
        if side == 'L':
            hip_idx, knee_idx, ankle_idx, foot_idx = 23, 25, 27, 31
        else:
            hip_idx, knee_idx, ankle_idx, foot_idx = 24, 26, 28, 32

        # Get keypoint columns
        kp_cols = {
            'hip_x': f'landmark_{hip_idx}_x',
            'hip_y': f'landmark_{hip_idx}_y',
            'knee_x': f'landmark_{knee_idx}_x',
            'knee_y': f'landmark_{knee_idx}_y',
            'ankle_x': f'landmark_{ankle_idx}_x',
            'ankle_y': f'landmark_{ankle_idx}_y',
            'foot_x': f'landmark_{foot_idx}_x',
            'foot_y': f'landmark_{foot_idx}_y'
        }

        # Check if all required columns exist
        if not all(col in keypoints_df.columns for col in kp_cols.values()):
            continue

        # Angle column names
        hip_col = f'hip_flex_{side}_deg'
        knee_col = f'knee_flex_{side}_deg'
        ankle_col = f'ankle_dorsi_{side}_deg'

        # Add confidence tracking
        for col in [hip_col, knee_col, ankle_col]:
            conf_col = f'{col}_confidence'
            if conf_col not in angles_df.columns:
                angles_df[conf_col] = angles_df[col].notna().astype(float)

        # TIER 1: Fill gaps with PCHIP interpolation (shape-preserving)
        gap_filled_count = {'hip': 0, 'knee': 0, 'ankle': 0}

        # Determine interpolation confidence from config
        pchip_conf = config.pchip_confidence if config and config.use_pchip else 0.85
        use_pchip = config.use_pchip if config else True

        if fill_small_gaps_flag and use_pchip:
            for angle_col, angle_name in [(hip_col, 'hip'),
                                          (knee_col, 'knee'),
                                          (ankle_col, 'ankle')]:
                conf_col = f'{angle_col}_confidence'
                original_nans = angles_df[angle_col].isna().sum()

                # Use PCHIP for shape-preserving interpolation
                interpolated_angles, updated_conf = interpolate_angles_pchip(
                    angles_df[angle_col].values,
                    angles_df[conf_col].values,
                    target_confidence=pchip_conf
                )

                angles_df[angle_col] = interpolated_angles
                angles_df[conf_col] = updated_conf

                new_nans = angles_df[angle_col].isna().sum()
                gap_filled_count[angle_name] = original_nans - new_nans
        elif fill_small_gaps_flag:
            # Fallback to linear interpolation for small gaps
            for angle_col, angle_name in [(hip_col, 'hip'),
                                          (knee_col, 'knee'),
                                          (ankle_col, 'ankle')]:
                original_nans = angles_df[angle_col].isna().sum()
                angles_df[angle_col] = fill_small_gaps(angles_df[angle_col].values, max_gap_size=3)
                new_nans = angles_df[angle_col].isna().sum()
                gap_filled_count[angle_name] = original_nans - new_nans

                # Mark interpolated values with slightly lower confidence
                newly_filled_mask = angles_df[angle_col].notna() & (angles_df[f'{angle_col}_confidence'] == 0)
                angles_df.loc[newly_filled_mask, f'{angle_col}_confidence'] = 0.95

        # TIER 2: Geometric recalculation for remaining gaps
        recalc_count = {'hip': 0, 'knee': 0, 'ankle': 0}

        for idx in range(len(angles_df)):
            # Check if keypoints are valid
            kp_valid = keypoints_df.loc[idx, list(kp_cols.values())].notna().all()

            if not kp_valid:
                continue

            # Get keypoint values
            hip_pos = (keypoints_df.loc[idx, kp_cols['hip_x']],
                      keypoints_df.loc[idx, kp_cols['hip_y']])
            knee_pos = (keypoints_df.loc[idx, kp_cols['knee_x']],
                       keypoints_df.loc[idx, kp_cols['knee_y']])
            ankle_pos = (keypoints_df.loc[idx, kp_cols['ankle_x']],
                        keypoints_df.loc[idx, kp_cols['ankle_y']])
            foot_pos = (keypoints_df.loc[idx, kp_cols['foot_x']],
                       keypoints_df.loc[idx, kp_cols['foot_y']])

            # Recalculate hip if missing
            if pd.isna(angles_df.loc[idx, hip_col]):
                pelvis_y = (keypoints_df.loc[idx, 'landmark_23_y'] +
                           keypoints_df.loc[idx, 'landmark_24_y']) / 2

                hip_angle = calculate_hip_flexion(hip_pos[1], knee_pos[1],
                                                 hip_pos[0], knee_pos[0], pelvis_y)

                if not np.isnan(hip_angle):
                    angles_df.loc[idx, hip_col] = hip_angle
                    angles_df.loc[idx, f'{hip_col}_confidence'] = confidence_level
                    recalc_count['hip'] += 1

            # Recalculate knee if missing
            if pd.isna(angles_df.loc[idx, knee_col]):
                knee_angle = calculate_knee_flexion(hip_pos[1], hip_pos[0],
                                                   knee_pos[1], knee_pos[0],
                                                   ankle_pos[1], ankle_pos[0])

                if not np.isnan(knee_angle):
                    angles_df.loc[idx, knee_col] = knee_angle
                    angles_df.loc[idx, f'{knee_col}_confidence'] = confidence_level
                    recalc_count['knee'] += 1

            # Recalculate ankle if missing
            if pd.isna(angles_df.loc[idx, ankle_col]):
                ankle_angle = calculate_ankle_dorsiflexion(knee_pos[1], knee_pos[0],
                                                          ankle_pos[1], ankle_pos[0],
                                                          foot_pos[1], foot_pos[0])

                if not np.isnan(ankle_angle):
                    angles_df.loc[idx, ankle_col] = ankle_angle
                    angles_df.loc[idx, f'{ankle_col}_confidence'] = confidence_level
                    recalc_count['ankle'] += 1

        # TIER 3: Apply temporal smoothing to each angle series
        for angle_col in [hip_col, knee_col, ankle_col]:
            conf_col = f'{angle_col}_confidence'
            angles_df[angle_col] = smooth_angle_series(
                angles_df[angle_col].values,
                angles_df[conf_col].values,
                window=smooth_window,
                max_jump=max_jump_deg
            )

        # Report results
        total_filled = sum(gap_filled_count.values())
        total_recalc = sum(recalc_count.values())

        if total_filled + total_recalc > 0:
            print(f"    {side} leg: gap_fill={total_filled} "
                  f"(hip={gap_filled_count['hip']}, knee={gap_filled_count['knee']}, "
                  f"ankle={gap_filled_count['ankle']}), "
                  f"recalc={total_recalc} "
                  f"(hip={recalc_count['hip']}, knee={recalc_count['knee']}, "
                  f"ankle={recalc_count['ankle']})")

    # Final quality diagnostics
    print(f"  ✓ Enhancement complete - Final data quality:")
    for side in ['L', 'R']:
        hip_col = f'hip_flex_{side}_deg'
        knee_col = f'knee_flex_{side}_deg'
        ankle_col = f'ankle_dorsi_{side}_deg'

        if hip_col in angles_df.columns:
            hip_cov = (1 - angles_df[hip_col].isna().sum() / len(angles_df)) * 100
            knee_cov = (1 - angles_df[knee_col].isna().sum() / len(angles_df)) * 100
            ankle_cov = (1 - angles_df[ankle_col].isna().sum() / len(angles_df)) * 100

            # Check for large jumps
            hip_jumps = np.abs(np.diff(angles_df[hip_col].dropna())).max() if angles_df[hip_col].notna().sum() > 1 else 0
            knee_jumps = np.abs(np.diff(angles_df[knee_col].dropna())).max() if angles_df[knee_col].notna().sum() > 1 else 0
            ankle_jumps = np.abs(np.diff(angles_df[ankle_col].dropna())).max() if angles_df[ankle_col].notna().sum() > 1 else 0

            print(f"    {side} coverage: hip={hip_cov:.1f}%, knee={knee_cov:.1f}%, ankle={ankle_cov:.1f}%")
            print(f"    {side} max jumps: hip={hip_jumps:.1f}°, knee={knee_jumps:.1f}°, ankle={ankle_jumps:.1f}°")

    return angles_df


# ==============================================================================
# GAP DETECTION AND QUALITY UTILITIES
# ==============================================================================

def find_gap_segments(series: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find all contiguous NaN segments in a series.

    Args:
        series: 1D array with potential NaN values

    Returns:
        List of (start_idx, end_idx) tuples for each NaN segment
    """
    is_nan = np.isnan(series)
    gaps = []

    in_gap = False
    gap_start = 0

    for i, nan_val in enumerate(is_nan):
        if nan_val and not in_gap:
            # Start of new gap
            gap_start = i
            in_gap = True
        elif not nan_val and in_gap:
            # End of gap
            gaps.append((gap_start, i - 1))
            in_gap = False

    # Handle gap extending to end
    if in_gap:
        gaps.append((gap_start, len(series) - 1))

    return gaps


def compute_longest_gap(series: np.ndarray) -> int:
    """
    Compute length of longest contiguous NaN segment.

    Args:
        series: 1D array with potential NaN values

    Returns:
        Length in frames of longest gap (0 if no gaps)
    """
    gaps = find_gap_segments(series)

    if not gaps:
        return 0

    return max(end - start + 1 for start, end in gaps)


def interpolate_angles_pchip(angles: np.ndarray,
                              confidence: np.ndarray,
                              target_confidence: float = 0.85) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate missing angles using PCHIP (Piecewise Cubic Hermite Interpolating Polynomial).

    PCHIP preserves monotonicity and doesn't overshoot like cubic splines,
    making it ideal for biomechanical angle data.

    Args:
        angles: Array with NaN values to interpolate
        confidence: Existing confidence scores (1.0 = original data)
        target_confidence: Confidence to assign to interpolated values

    Returns:
        Tuple of (interpolated_angles, updated_confidence)
    """
    from scipy.interpolate import PchipInterpolator

    interpolated = angles.copy()
    conf_updated = confidence.copy()

    valid_mask = ~np.isnan(angles)

    if valid_mask.sum() < 2:
        # Need at least 2 points for interpolation
        return interpolated, conf_updated

    valid_indices = np.where(valid_mask)[0]
    valid_values = angles[valid_mask]

    # Create PCHIP interpolator
    pchip = PchipInterpolator(valid_indices, valid_values, extrapolate=False)

    # Interpolate missing values
    nan_mask = np.isnan(angles)
    nan_indices = np.where(nan_mask)[0]

    if len(nan_indices) > 0:
        # Only interpolate within valid range
        min_valid = valid_indices[0]
        max_valid = valid_indices[-1]

        interpolatable = (nan_indices >= min_valid) & (nan_indices <= max_valid)
        indices_to_fill = nan_indices[interpolatable]

        if len(indices_to_fill) > 0:
            interpolated[indices_to_fill] = pchip(indices_to_fill)
            conf_updated[indices_to_fill] = target_confidence

    return interpolated, conf_updated


def compute_window_quality(angles_df: pd.DataFrame,
                           window_start: int,
                           window_end: int,
                           leg: str,
                           config: 'AnalysisConfig',
                           keypoints_df: Optional[pd.DataFrame] = None) -> 'WindowQualityMetrics':
    """
    Compute quality metrics for a stride window and apply quality gates.

    Args:
        angles_df: DataFrame with angle columns
        window_start: Start frame index
        window_end: End frame index (inclusive)
        leg: 'L' or 'R'
        config: Analysis configuration with quality thresholds
        keypoints_df: Optional keypoints DataFrame for pelvis stability check

    Returns:
        WindowQualityMetrics with all quality assessments
    """
    # Extract window data
    window_data = angles_df.iloc[window_start:window_end+1]
    window_len = len(window_data)

    # Column names
    hip_col = f'hip_flex_{leg}_deg'
    knee_col = f'knee_flex_{leg}_deg'
    ankle_col = f'ankle_dorsi_{leg}_deg'

    # 1. COVERAGE METRICS
    hip_coverage = (window_data[hip_col].notna().sum() / window_len) * 100
    knee_coverage = (window_data[knee_col].notna().sum() / window_len) * 100
    ankle_coverage = (window_data[ankle_col].notna().sum() / window_len) * 100

    # 2. GAP ANALYSIS
    hip_longest_gap = compute_longest_gap(window_data[hip_col].values)
    knee_longest_gap = compute_longest_gap(window_data[knee_col].values)
    ankle_longest_gap = compute_longest_gap(window_data[ankle_col].values)

    # 3. PELVIS STABILITY
    pelvis_stability = 0.0
    if keypoints_df is not None:
        pelvis_col = f'pelvis_y_{leg}' if f'pelvis_y_{leg}' in keypoints_df.columns else 'pelvis_y'
        if pelvis_col in keypoints_df.columns:
            pelvis_window = keypoints_df[pelvis_col].iloc[window_start:window_end+1]
            pelvis_stability = pelvis_window.std() if pelvis_window.notna().sum() > 1 else 0.0
        else:
            pelvis_stability = 0.0  # Assume stable if no data

    # 4. TEMPORAL SANITY (max frame-to-frame jumps)
    def max_jump(series):
        valid_vals = series.dropna()
        if len(valid_vals) < 2:
            return 0.0
        return np.abs(np.diff(valid_vals)).max()

    hip_max_jump = max_jump(window_data[hip_col])
    knee_max_jump = max_jump(window_data[knee_col])
    ankle_max_jump = max_jump(window_data[ankle_col])

    # 5. QUALITY GATES
    passes_coverage = (hip_coverage >= config.min_coverage_pct and
                       knee_coverage >= config.min_coverage_pct and
                       ankle_coverage >= config.min_coverage_pct)

    passes_gap = (hip_longest_gap <= config.max_gap_frames and
                  knee_longest_gap <= config.max_gap_frames and
                  ankle_longest_gap <= config.max_gap_frames)

    passes_stability = pelvis_stability <= config.max_pelvis_std

    passes_sanity = (hip_max_jump <= config.max_angle_jump and
                     knee_max_jump <= config.max_angle_jump and
                     ankle_max_jump <= config.max_angle_jump)

    passes_all = passes_coverage and passes_gap and passes_stability and passes_sanity

    # 6. COMPOSITE QUALITY SCORE (0-1)
    # Weight: coverage=0.35, gaps=0.25, stability=0.20, sanity=0.20
    coverage_score = (hip_coverage + knee_coverage + ankle_coverage) / 300.0
    gap_score = 1.0 - min(1.0, (hip_longest_gap + knee_longest_gap + ankle_longest_gap) / (3 * config.max_gap_frames))
    stability_score = 1.0 - min(1.0, pelvis_stability / config.max_pelvis_std)
    sanity_score = 1.0 - min(1.0, (hip_max_jump + knee_max_jump + ankle_max_jump) / (3 * config.max_angle_jump))

    quality_score = (0.35 * coverage_score +
                     0.25 * gap_score +
                     0.20 * stability_score +
                     0.20 * sanity_score)

    # 7. REJECTION REASON
    rejection_reason = None
    if not passes_all:
        reasons = []
        if not passes_coverage:
            reasons.append(f"Low coverage: hip={hip_coverage:.1f}%, knee={knee_coverage:.1f}%, ankle={ankle_coverage:.1f}%")
        if not passes_gap:
            reasons.append(f"Long gaps: hip={hip_longest_gap}f, knee={knee_longest_gap}f, ankle={ankle_longest_gap}f")
        if not passes_stability:
            reasons.append(f"Unstable pelvis: std={pelvis_stability:.1f}px")
        if not passes_sanity:
            reasons.append(f"Large jumps: hip={hip_max_jump:.1f}°, knee={knee_max_jump:.1f}°, ankle={ankle_max_jump:.1f}°")
        rejection_reason = "; ".join(reasons)

    return WindowQualityMetrics(
        hip_coverage=hip_coverage,
        knee_coverage=knee_coverage,
        ankle_coverage=ankle_coverage,
        hip_longest_gap=hip_longest_gap,
        knee_longest_gap=knee_longest_gap,
        ankle_longest_gap=ankle_longest_gap,
        pelvis_stability=pelvis_stability,
        hip_max_jump=hip_max_jump,
        knee_max_jump=knee_max_jump,
        ankle_max_jump=ankle_max_jump,
        quality_score=quality_score,
        passes_coverage_gate=passes_coverage,
        passes_gap_gate=passes_gap,
        passes_stability_gate=passes_stability,
        passes_sanity_gate=passes_sanity,
        passes_all_gates=passes_all,
        rejection_reason=rejection_reason
    )


# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
# DATA STRUCTURES
# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

@dataclass
class WindowQualityMetrics:
    """
    Per-window quality metrics for stride detection.
    Tracks data completeness and reliability for each detection window.
    """

    # Coverage metrics (% of non-NaN frames)
    hip_coverage: float
    knee_coverage: float
    ankle_coverage: float

    # Gap analysis (longest contiguous NaN segment in frames)
    hip_longest_gap: int
    knee_longest_gap: int
    ankle_longest_gap: int

    # Pelvis stability (std dev of pelvis_y in pixels)
    pelvis_stability: float

    # Temporal sanity (max frame-to-frame angle change in degrees)
    hip_max_jump: float
    knee_max_jump: float
    ankle_max_jump: float

    # Composite quality score (0-1, higher = better)
    quality_score: float

    # Pass/fail flags
    passes_coverage_gate: bool  # All joints >= threshold
    passes_gap_gate: bool       # All gaps <= max_gap_frames
    passes_stability_gate: bool # Pelvis stable
    passes_sanity_gate: bool    # No impossible jumps

    # Overall pass (all gates must pass)
    passes_all_gates: bool

    # Rejection reason if failed
    rejection_reason: Optional[str] = None


@dataclass
class StrideWindow:
    """Represents a single gait window (stance or full cycle)."""
    leg: Literal["L", "R"]
    stride_id: int
    start_time: float
    end_time: float
    duration: float
    start_frame: int
    end_frame: int
    window_type: Literal["stance", "cycle"] = "cycle"  # stance=HS→TO, cycle=HS→HS
    speed: Optional[float] = None
    quality_metrics: Optional['WindowQualityMetrics'] = None  # Per-window quality tracking


@dataclass
class CyclogramLoop:
    """Represents a single cyclogram trajectory."""
    leg: Literal["L", "R"]
    stride_id: int
    joint_pair: Tuple[str, str]
    proximal: np.ndarray
    distal: np.ndarray
    time_normalized: np.ndarray
    duration: float
    is_full_cycle: bool = True  # True=HS→HS cycle, False=HS→TO stance
    closure_error_deg: float = 0.0  # Euclidean distance between first and last point
    nan_percent: float = 0.0  # Percentage of NaN values after resampling
    speed: Optional[float] = None

    @property
    def points(self) -> np.ndarray:
        """Returns (N, 2) array of [proximal, distal] coordinates."""
        return np.column_stack([self.proximal, self.distal])

    @property
    def is_closed(self) -> bool:
        """Check if loop is properly closed (closure error < 2 degrees)."""
        return self.is_full_cycle and self.closure_error_deg < 2.0


@dataclass
class PairedStrides:
    """Represents matched left-right stride pair."""
    left_stride: StrideWindow
    right_stride: StrideWindow
    time_difference: float
    temporal_overlap: float


@dataclass
class PipelineReport:
    """
    Stage-by-stage count tracking for quality control pipeline.
    Tracks progression: events → cycles → QC-passed → paired
    """
    # Per-leg counts
    events_detected_L: int
    events_detected_R: int

    cycles_extracted_L: int
    cycles_extracted_R: int

    cycles_qc_passed_L: int
    cycles_qc_passed_R: int

    cycles_paired: int  # Successfully paired L-R cycles

    # Rejection tracking
    cycles_rejected_coverage_L: int = 0
    cycles_rejected_coverage_R: int = 0

    cycles_rejected_gaps_L: int = 0
    cycles_rejected_gaps_R: int = 0

    cycles_rejected_stability_L: int = 0
    cycles_rejected_stability_R: int = 0

    cycles_rejected_sanity_L: int = 0
    cycles_rejected_sanity_R: int = 0

    cycles_unpaired_L: int = 0  # Passed QC but no matching R
    cycles_unpaired_R: int = 0  # Passed QC but no matching L

    # Efficiency metrics (%)
    @property
    def qc_pass_rate_L(self) -> float:
        """QC pass rate for left leg (%): (QC-passed / extracted) × 100"""
        return (self.cycles_qc_passed_L / self.cycles_extracted_L * 100) if self.cycles_extracted_L > 0 else 0.0

    @property
    def qc_pass_rate_R(self) -> float:
        """QC pass rate for right leg (%): (QC-passed / extracted) × 100"""
        return (self.cycles_qc_passed_R / self.cycles_extracted_R * 100) if self.cycles_extracted_R > 0 else 0.0

    @property
    def pairing_efficiency_L(self) -> float:
        """Pairing efficiency for left leg (%): (paired / QC-passed) × 100"""
        return (self.cycles_paired / self.cycles_qc_passed_L * 100) if self.cycles_qc_passed_L > 0 else 0.0

    @property
    def pairing_efficiency_R(self) -> float:
        """Pairing efficiency for right leg (%): (paired / QC-passed) × 100"""
        return (self.cycles_paired / self.cycles_qc_passed_R * 100) if self.cycles_qc_passed_R > 0 else 0.0

    @property
    def overall_efficiency_L(self) -> float:
        """Overall efficiency for left leg (%): (paired / events) × 100"""
        return (self.cycles_paired / self.events_detected_L * 100) if self.events_detected_L > 0 else 0.0

    @property
    def overall_efficiency_R(self) -> float:
        """Overall efficiency for right leg (%): (paired / events) × 100"""
        return (self.cycles_paired / self.events_detected_R * 100) if self.events_detected_R > 0 else 0.0

    def print_summary(self):
        """Print human-readable pipeline summary"""
        print("\n" + "="*70)
        print("QUALITY CONTROL PIPELINE SUMMARY")
        print("="*70)
        print(f"\n{'Stage':<30} {'Left':>10} {'Right':>10}")
        print("-" * 70)
        print(f"{'Events detected':<30} {self.events_detected_L:>10} {self.events_detected_R:>10}")
        print(f"{'Cycles extracted':<30} {self.cycles_extracted_L:>10} {self.cycles_extracted_R:>10}")
        print(f"{'Cycles passed QC':<30} {self.cycles_qc_passed_L:>10} {self.cycles_qc_passed_R:>10}")
        print(f"{'Cycles paired':<30} {self.cycles_paired:>10} {self.cycles_paired:>10}")

        print("\n" + "-" * 70)
        print(f"{'QC pass rate':<30} {self.qc_pass_rate_L:>9.1f}% {self.qc_pass_rate_R:>9.1f}%")
        print(f"{'Pairing efficiency':<30} {self.pairing_efficiency_L:>9.1f}% {self.pairing_efficiency_R:>9.1f}%")
        print(f"{'Overall efficiency':<30} {self.overall_efficiency_L:>9.1f}% {self.overall_efficiency_R:>9.1f}%")

        print("\n" + "-" * 70)
        print("REJECTION REASONS:")
        print(f"  Coverage gate:   L={self.cycles_rejected_coverage_L}, R={self.cycles_rejected_coverage_R}")
        print(f"  Gap gate:        L={self.cycles_rejected_gaps_L}, R={self.cycles_rejected_gaps_R}")
        print(f"  Stability gate:  L={self.cycles_rejected_stability_L}, R={self.cycles_rejected_stability_R}")
        print(f"  Sanity gate:     L={self.cycles_rejected_sanity_L}, R={self.cycles_rejected_sanity_R}")
        print(f"  Unpaired:        L={self.cycles_unpaired_L}, R={self.cycles_unpaired_R}")
        print("="*70 + "\n")


@dataclass
class CyclogramMetrics:
    """Quantitative comparison metrics for a paired cyclogram."""
    joint_pair: Tuple[str, str]
    left_stride_id: int
    right_stride_id: int

    # Geometric metrics
    area_left: float
    area_right: float
    delta_area_percent: float

    rmse: float
    procrustes_distance: float
    dtw: float  # Dynamic Time Warping distance

    orientation_left: float
    orientation_right: float
    delta_orientation: float

    hysteresis_left: str
    hysteresis_right: str
    hysteresis_mismatch: bool

    normalized_area_left: float  # Scale-free normalized area
    normalized_area_right: float

    # Composite score
    similarity_score: float


@dataclass
class AnalysisConfig:
    """Configuration for cyclogram analysis."""

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
    plot_dpi: int = 300
    generate_per_leg_plots: bool = False

    # Outlier handling
    outlier_percentile: float = 1.0

    # Angle enhancement
    enhance_angles: bool = False  # Enable keypoint-based angle recalculation

    # Quality gate thresholds
    min_coverage_pct: float = 70.0          # Minimum % non-NaN frames per joint
    max_gap_frames: int = 30                # Maximum contiguous NaN gap (frames)
    max_pelvis_std: float = 15.0            # Maximum pelvis_y std dev (pixels)
    max_angle_jump: float = 45.0            # Maximum frame-to-frame angle change (degrees)

    # Interpolation settings
    use_pchip: bool = True                  # Use PCHIP instead of cubic
    pchip_confidence: float = 0.85          # Confidence for PCHIP-interpolated values

    # Pairing strategy
    use_index_pairing: bool = True          # Use index-based (Lₖ ↔ Rₖ) instead of mid-time
    min_phase_overlap: float = 0.3          # Minimum temporal overlap for valid pairs


@dataclass
class PlotConfig:
    """
    Centralized plotting configuration for consistent visualization styling.

    All dimensions, fonts, colors, and styling parameters in one place
    to ensure stable, reproducible, and professional-looking plots.
    """

    # Figure dimensions (width, height in inches)
    cyclogram_figsize: Tuple[float, float] = (16, 8)
    similarity_figsize: Tuple[float, float] = (12, 7)

    # Resolution
    dpi: int = 300

    # Font sizes (hierarchy)
    title_fontsize: int = 16
    subtitle_fontsize: int = 14
    label_fontsize: int = 13
    tick_fontsize: int = 11
    legend_fontsize: int = 11
    annotation_fontsize: int = 9
    stats_fontsize: int = 10

    # Font weights
    title_fontweight: str = 'bold'
    label_fontweight: str = 'bold'

    # Colors - Left leg
    left_color: str = '#1f77b4'  # Blue
    left_color_light: str = '#aec7e8'
    left_mean_color: str = '#0055AA'

    # Colors - Right leg
    right_color: str = '#d62728'  # Red
    right_color_light: str = '#ff9896'
    right_mean_color: str = '#AA0000'

    # Colors - Neutral/UI
    background_color: str = '#F5F5F5'
    grid_color: str = 'gray'
    grid_alpha: float = 0.3

    # Colors - Quality indicators
    excellent_color: str = '#2ca02c'  # Green
    good_color: str = '#ff7f0e'  # Orange
    warning_color: str = '#d62728'  # Red

    # Line styling
    mean_linewidth: float = 3.5
    individual_linewidth: float = 1.0
    threshold_linewidth: float = 2.0
    grid_linestyle: str = '--'

    # Marker styling
    mean_marker: str = 'o'
    mean_markersize: float = 4
    mean_markevery: int = 20

    # Alpha (transparency)
    individual_alpha: float = 0.3
    threshold_alpha: float = 0.5
    bar_alpha: float = 0.7

    # Box styling
    stats_box_pad: float = 0.5
    stats_box_alpha: float = 0.9
    annotation_box_pad: float = 0.3
    annotation_box_alpha: float = 0.7

    # Threshold values
    excellent_threshold: float = 90.0
    good_threshold: float = 75.0

    # Error bar styling
    capsize: float = 5.0

    # Layout
    tight_layout: bool = True
    bbox_inches: str = 'tight'

    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.dpi > 0, "DPI must be positive"
        assert all(s > 0 for s in self.cyclogram_figsize), "Figure sizes must be positive"
        assert all(s > 0 for s in self.similarity_figsize), "Figure sizes must be positive"
        assert 0 <= self.grid_alpha <= 1, "Alpha must be between 0 and 1"
        assert 0 <= self.bar_alpha <= 1, "Alpha must be between 0 and 1"


# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
# DATA LOADING & PREPROCESSING
# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

def detect_columns(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """
    Auto-detect joint angle columns using regex patterns.

    Returns:
        Dict mapping leg -> joint -> column_name
        e.g., {"L": {"hip": "hip_flex_L_deg", ...}, "R": {...}}
    """
    patterns = {
        "hip": [
            r"(?i)hip.*flex.*(L|left|lt).*deg",
            r"(?i)(L|left|lt).*hip.*flex.*deg",
        ],
        "knee": [
            r"(?i)knee.*flex.*(L|left|lt).*deg",
            r"(?i)(L|left|lt).*knee.*flex.*deg",
        ],
        "ankle": [
            r"(?i)ankle.*(dorsi|flex).*(L|left|lt).*deg",
            r"(?i)(L|left|lt).*ankle.*(dorsi|flex).*deg",
        ]
    }

    col_map = {"L": {}, "R": {}}

    for joint, pattern_list in patterns.items():
        for leg in ["L", "R"]:
            found = False
            for pattern in pattern_list:
                # Adjust pattern for right leg
                if leg == "R":
                    pattern = pattern.replace("(L|left|lt)", "(R|right|rt)")

                for col in df.columns:
                    if re.search(pattern, col):
                        col_map[leg][joint] = col
                        found = True
                        break
                if found:
                    break

    # Validate all required columns found
    required_joints = ["hip", "knee", "ankle"]
    for leg in ["L", "R"]:
        for joint in required_joints:
            if joint not in col_map[leg]:
                raise ValueError(f"Could not detect {joint} column for {leg} leg. "
                               f"Available columns: {df.columns.tolist()}")

    return col_map


def normalize_timebase(angles_df: pd.DataFrame,
                       events_df: pd.DataFrame,
                       fps: Optional[float] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ensure consistent time-based indexing.

    Priority: Use 'timestamp' or 'time_s' if present.
    Fallback: Derive from 'frame' column using FPS.
    """
    # Check for existing time column
    time_col = None
    for col_name in ["timestamp", "time_s"]:
        if col_name in angles_df.columns:
            time_col = col_name
            break

    if time_col:
        # Rename to standard 'timestamp'
        if time_col != "timestamp":
            angles_df = angles_df.rename(columns={time_col: "timestamp"})
            events_df = events_df.rename(columns={time_col: "timestamp"})
    else:
        # Derive from frame
        if fps is None:
            raise ValueError("FPS required when timestamp column is missing")

        if "frame" not in angles_df.columns:
            raise ValueError("Neither 'timestamp' nor 'frame' column found")

        angles_df["timestamp"] = angles_df["frame"] / fps
        events_df["timestamp"] = events_df["frame"] / fps

    return angles_df, events_df


def preprocess_angles(df: pd.DataFrame,
                     angle_cols: List[str],
                     smooth_window: int = 11,
                     poly_order: int = 2,
                     smooth_threshold: float = 5.0) -> pd.DataFrame:
    """
    Smooth and unwrap joint angles.

    Steps:
    1. Check angle variability (std dev)
    2. Apply smoothing only if std > threshold
    3. Unwrap angles to avoid 0/360 discontinuities
    """
    df = df.copy()

    for col in angle_cols:
        angles = df[col].values

        # Conditional smoothing
        if np.std(angles) > smooth_threshold:
            # Ensure window size is valid
            if len(angles) >= smooth_window:
                angles = savgol_filter(angles, smooth_window, poly_order)

        # Unwrap discontinuities
        angles_rad = np.deg2rad(angles)
        angles_rad = np.unwrap(angles_rad)
        angles = np.rad2deg(angles_rad)

        df[col] = angles

    return df


# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
# ADAPTIVE CALIBRATION - Self-tuning parameters based on data characteristics
# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

def calibrate_smoothing_threshold(angles_df: pd.DataFrame,
                                  angle_cols: List[str],
                                  min_cycles: int = 5) -> float:
    """
    Auto-calibrate smoothing threshold based on angle variability.

    Strategy: Use median std dev across all angle columns.
    Apply smoothing only if std > threshold.
    """
    std_devs = []

    for col in angle_cols:
        angles = angles_df[col].values
        # Use nanstd to handle NaN values
        if len(angles) > 0:
            std_val = np.nanstd(angles)
            # Only include if not NaN and not too small
            if not np.isnan(std_val) and std_val > 0.1:
                std_devs.append(std_val)

    if not std_devs:
        return 5.0  # Fallback default

    # Use 75th percentile of std devs as threshold
    # This ensures we smooth high-variability signals while preserving low-variance ones
    threshold = np.percentile(std_devs, 75)

    # Clamp between reasonable bounds
    threshold = np.clip(threshold, 2.0, 15.0)

    print(f"  ⚙️  Calibrated smoothing threshold: {threshold:.2f}° (based on {len(std_devs)} angles)")

    return threshold


def calibrate_stride_constraints(events_df: pd.DataFrame,
                                 leg: str,
                                 min_cycles: int = 5,
                                 percentile_range: float = 10.0) -> Tuple[float, float]:
    """
    Auto-calibrate stride duration constraints based on initial stride analysis.

    Strategy:
    1. Extract all stride durations without constraints
    2. Use at least min_cycles strides for calibration
    3. Set min/max based on percentiles with safety margins
    """
    leg_events = events_df[events_df["side"] == leg].sort_values("timestamp")

    strikes = leg_events[leg_events["event_type"] == "heel_strikes"]
    toe_offs = leg_events[leg_events["event_type"] == "toe_offs"]

    if strikes.empty or toe_offs.empty:
        return 0.3, 3.0  # Fallback defaults

    # Extract all stride durations
    durations = []
    for _, strike_row in strikes.iterrows():
        subsequent_toeoffs = toe_offs[toe_offs["timestamp"] > strike_row["timestamp"]]
        if not subsequent_toeoffs.empty:
            toeoff_row = subsequent_toeoffs.iloc[0]
            duration = toeoff_row["timestamp"] - strike_row["timestamp"]
            if 0.1 < duration < 5.0:  # Sanity check
                durations.append(duration)

    if len(durations) < min_cycles:
        print(f"  ⚠️  Only {len(durations)} strides for {leg} leg calibration (min {min_cycles} recommended)")
        if durations:
            min_dur = min(durations) * 0.8
            max_dur = max(durations) * 1.2
            return min_dur, max_dur
        return 0.3, 3.0  # Fallback

    # Use percentile-based approach with safety margins
    p_low = percentile_range
    p_high = 100 - percentile_range

    min_duration = np.percentile(durations, p_low) * 0.85  # 15% safety margin below
    max_duration = np.percentile(durations, p_high) * 1.15  # 15% safety margin above

    # Absolute bounds for sanity
    min_duration = max(0.2, min_duration)
    max_duration = min(5.0, max_duration)

    print(f"  ⚙️  Calibrated {leg} stride duration: [{min_duration:.3f}s, {max_duration:.3f}s] "
          f"(from {len(durations)} strides)")

    return min_duration, max_duration

def calibrate_cycle_constraints(events_df: pd.DataFrame,
                                leg: str,
                                min_cycles: int = 5,
                                percentile_range: float = 10.0) -> Tuple[float, float]:
    """
    Auto-calibrate CYCLE duration constraints (HS→HS) for full gait cycles.
    
    Critical difference from calibrate_stride_constraints:
    - Uses consecutive heel strikes (HS→HS) instead of heel strike to toe-off (HS→TO)
    - Returns durations for FULL GAIT CYCLES (~1.0-1.5s) not stance phase (~0.3-0.8s)
    
    Strategy:
    1. Extract all cycle durations (consecutive HS of same leg)
    2. Use at least min_cycles for calibration
    3. Set min/max based on percentiles with safety margins
    """
    leg_events = events_df[events_df["side"] == leg].sort_values("timestamp")
    
    strikes = leg_events[leg_events["event_type"] == "heel_strikes"]
    
    if len(strikes) < 2:
        return 0.8, 2.5  # Fallback defaults for full cycles
    
    # Extract cycle durations (HS_i → HS_{i+1})
    strikes_list = strikes.to_dict('records')
    durations = []
    for i in range(len(strikes_list) - 1):
        duration = strikes_list[i+1]["timestamp"] - strikes_list[i]["timestamp"]
        if 0.5 < duration < 4.0:  # Sanity check for cycles (longer than stance)
            durations.append(duration)
    
    if len(durations) < min_cycles:
        print(f"  ⚠️  Only {len(durations)} cycles for {leg} leg calibration (min {min_cycles} recommended)")
        if durations:
            min_dur = min(durations) * 0.9
            max_dur = max(durations) * 1.1
            return min_dur, max_dur
        return 0.8, 2.5  # Fallback
    
    # Use percentile-based approach with safety margins
    p_low = percentile_range
    p_high = 100 - percentile_range
    
    min_duration = np.percentile(durations, p_low) * 0.90  # 10% safety margin below
    max_duration = np.percentile(durations, p_high) * 1.10  # 10% safety margin above
    
    # Absolute bounds for cycles (longer than stance, typical walking: 0.8-2.5s)
    min_duration = max(0.5, min_duration)
    max_duration = min(4.0, max_duration)
    
    print(f"  ⚙️  Calibrated {leg} cycle duration: [{min_duration:.3f}s, {max_duration:.3f}s] "
          f"(from {len(durations)} cycles)")
    
    return min_duration, max_duration



def calibrate_pairing_tolerance(left_durations: List[float],
                                right_durations: List[float],
                                min_cycles: int = 5) -> float:
    """
    Auto-calibrate left-right pairing tolerance based on stride timing variability.

    Strategy:
    1. Compute coefficient of variation (CV) for each leg
    2. Use max CV to set tolerance
    3. Higher variability = larger tolerance needed
    """
    if len(left_durations) < min_cycles or len(right_durations) < min_cycles:
        return 0.15  # Fallback default (15%)

    # Coefficient of variation = std / mean
    cv_left = np.std(left_durations) / np.mean(left_durations) if np.mean(left_durations) > 0 else 0
    cv_right = np.std(right_durations) / np.mean(right_durations) if np.mean(right_durations) > 0 else 0

    max_cv = max(cv_left, cv_right)

    # Tolerance scales with variability
    # Low variability (CV < 0.1): tight tolerance (10%)
    # High variability (CV > 0.3): loose tolerance (25%)
    tolerance = np.clip(0.10 + (max_cv * 0.5), 0.08, 0.30)

    print(f"  ⚙️  Calibrated pairing tolerance: {tolerance*100:.1f}% "
          f"(CV: L={cv_left:.3f}, R={cv_right:.3f})")

    return tolerance


def detect_and_remove_outliers(windows: List[StrideWindow],
                               leg: str,
                               method: str = "iqr",
                               iqr_factor: float = 1.5) -> List[StrideWindow]:
    """
    Adaptive outlier detection and removal based on stride duration distribution.

    Methods:
    - iqr: Interquartile range (robust to outliers)
    - zscore: Z-score based (assumes normal distribution)
    """
    if len(windows) < 5:
        return windows  # Too few data points for outlier detection

    durations = [w.duration for w in windows]

    if method == "iqr":
        Q1 = np.percentile(durations, 25)
        Q3 = np.percentile(durations, 75)
        IQR = Q3 - Q1

        lower_bound = Q1 - iqr_factor * IQR
        upper_bound = Q3 + iqr_factor * IQR

        filtered_windows = [w for w in windows if lower_bound <= w.duration <= upper_bound]

    elif method == "zscore":
        mean_dur = np.mean(durations)
        std_dur = np.std(durations)

        # Outliers are >3 standard deviations from mean
        filtered_windows = [w for w in windows
                          if abs(w.duration - mean_dur) <= 3 * std_dur]
    else:
        filtered_windows = windows

    n_removed = len(windows) - len(filtered_windows)
    if n_removed > 0:
        print(f"  🔧 Removed {n_removed} outlier stride(s) for {leg} leg "
              f"(method: {method}, {len(filtered_windows)} remaining)")

    return filtered_windows


def auto_calibrate_config(angles_df: pd.DataFrame,
                         events_df: pd.DataFrame,
                         col_map: Dict[str, Dict[str, str]],
                         base_config: AnalysisConfig,
                         min_cycles: int = 5) -> AnalysisConfig:
    """
    Automatically calibrate all configuration parameters based on data characteristics.

    This function analyzes the data and adjusts:
    1. Smoothing threshold (based on angle variability)
    2. Stride duration constraints (based on actual stride distribution)
    3. Pairing tolerance (based on stride timing variability)

    Requires at least min_cycles strides per leg for robust calibration.
    """
    print(f"\n{'─'*80}")
    print(f"⚙️  AUTO-CALIBRATION (minimum {min_cycles} cycles required)")
    print(f"{'─'*80}")

    # 1. Calibrate smoothing threshold
    angle_cols = [col for leg in col_map.values() for col in leg.values()]
    smooth_threshold = calibrate_smoothing_threshold(angles_df, angle_cols, min_cycles)

    # 2. Calibrate CYCLE duration constraints (HS→HS) for each leg
    min_dur_L, max_dur_L = calibrate_cycle_constraints(events_df, "L", min_cycles)
    min_dur_R, max_dur_R = calibrate_cycle_constraints(events_df, "R", min_cycles)

    # Use conservative bounds (most restrictive)
    min_stride_duration = max(min_dur_L, min_dur_R)
    max_stride_duration = min(max_dur_L, max_dur_R)

    # 3. Extract stride durations for tolerance calibration
    # Do a quick stride extraction with relaxed constraints
    temp_events_L = events_df[events_df["side"] == "L"].sort_values("timestamp")
    temp_events_R = events_df[events_df["side"] == "R"].sort_values("timestamp")

    strikes_L = temp_events_L[temp_events_L["event_type"] == "heel_strikes"]
    toe_offs_L = temp_events_L[temp_events_L["event_type"] == "toe_offs"]
    strikes_R = temp_events_R[temp_events_R["event_type"] == "heel_strikes"]
    toe_offs_R = temp_events_R[temp_events_R["event_type"] == "toe_offs"]

    dur_L = []
    for _, strike in strikes_L.iterrows():
        toeoff = toe_offs_L[toe_offs_L["timestamp"] > strike["timestamp"]]
        if not toeoff.empty:
            dur_L.append(toeoff.iloc[0]["timestamp"] - strike["timestamp"])

    dur_R = []
    for _, strike in strikes_R.iterrows():
        toeoff = toe_offs_R[toe_offs_R["timestamp"] > strike["timestamp"]]
        if not toeoff.empty:
            dur_R.append(toeoff.iloc[0]["timestamp"] - strike["timestamp"])

    # Calibrate pairing tolerance
    pairing_tolerance = calibrate_pairing_tolerance(dur_L, dur_R, min_cycles)

    print(f"{'─'*80}\n")

    # Create updated config
    calibrated_config = AnalysisConfig(
        fps=base_config.fps,
        smooth_window=base_config.smooth_window,
        smooth_poly=base_config.smooth_poly,
        smooth_threshold=smooth_threshold,  # CALIBRATED
        min_stride_duration=min_stride_duration,  # CALIBRATED
        max_stride_duration=max_stride_duration,  # CALIBRATED
        pairing_tolerance=pairing_tolerance,  # CALIBRATED
        n_resample_points=base_config.n_resample_points,
        joint_pairs=base_config.joint_pairs,
        similarity_weights=base_config.similarity_weights,
        plot_dpi=base_config.plot_dpi,
        generate_per_leg_plots=base_config.generate_per_leg_plots,
        outlier_percentile=base_config.outlier_percentile
    )

    return calibrated_config


# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
# STRIDE SEGMENTATION
# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

def build_stride_windows(events_df: pd.DataFrame,
                        leg: str,
                        min_duration: float = 0.3,
                        max_duration: float = 3.0) -> List[StrideWindow]:
    """
    Extract stride windows from gait events.

    For each leg:
    1. Filter events for this leg
    2. Sort by timestamp
    3. Pair heel_strikes with subsequent toe_offs
    4. Validate duration constraints
    """
    leg_events = events_df[events_df["side"] == leg].sort_values("timestamp")

    # Extract strikes and toe-offs
    strikes = leg_events[leg_events["event_type"] == "heel_strikes"]
    toe_offs = leg_events[leg_events["event_type"] == "toe_offs"]

    if strikes.empty or toe_offs.empty:
        print(f"Warning: No valid events found for {leg} leg")
        return []

    windows = []

    for _, strike_row in strikes.iterrows():
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
            start_frame=int(strike_row["frame"]),
            end_frame=int(toeoff_row["frame"]),
            window_type="stance"  # HS→TO stance phase
        ))

    return windows


def build_cycle_windows(events_df: pd.DataFrame,
                       leg: str,
                       angles_df: Optional[pd.DataFrame] = None,
                       keypoints_df: Optional[pd.DataFrame] = None,
                       config: Optional['AnalysisConfig'] = None,
                       min_duration: float = 0.8,
                       max_duration: float = 2.5,
                       apply_quality_gates: bool = True) -> Tuple[List[StrideWindow], Dict[str, int]]:
    """
    Extract FULL GAIT CYCLE windows (HS→HS) with quality control.

    Critical difference from build_stride_windows:
    - Pairs consecutive heel strikes of the SAME leg (ipsilateral)
    - Creates complete gait cycle: stance + swing phases
    - Naturally closed loops when extracting angle trajectories
    - Applies per-window quality gates if enabled

    For each leg:
    1. Filter heel strike events for this leg
    2. Sort by timestamp
    3. Pair consecutive heel strikes: HS(i) → HS(i+1)
    4. Validate duration constraints (typical: 0.8-2.5s for walking)
    5. Compute quality metrics and apply gates (if enabled)

    Returns:
        Tuple of (valid_windows, rejection_counts)
        - valid_windows: List of StrideWindow that passed QC
        - rejection_counts: Dict with rejection statistics
    """
    leg_events = events_df[events_df["side"] == leg].sort_values("timestamp")

    # Extract only heel strikes for full cycles
    strikes = leg_events[leg_events["event_type"] == "heel_strikes"].copy()

    if len(strikes) < 2:
        print(f"Warning: Need at least 2 heel strikes for {leg} leg cycles")
        return [], {}

    # Initialize rejection tracking
    rejection_counts = {
        'total_extracted': 0,
        'rejected_coverage': 0,
        'rejected_gaps': 0,
        'rejected_stability': 0,
        'rejected_sanity': 0,
        'passed_qc': 0
    }

    windows = []
    strikes_list = strikes.to_dict('records')

    # Pair consecutive heel strikes (HS_i → HS_{i+1})
    for i in range(len(strikes_list) - 1):
        strike_start = strikes_list[i]
        strike_end = strikes_list[i + 1]

        duration = strike_end["timestamp"] - strike_start["timestamp"]

        # Validate duration (full gait cycle is longer than stance)
        if not (min_duration <= duration <= max_duration):
            continue  # Outlier cycle

        rejection_counts['total_extracted'] += 1

        # Create window
        window = StrideWindow(
            leg=leg,
            stride_id=len(windows) + 1,
            start_time=strike_start["timestamp"],
            end_time=strike_end["timestamp"],
            duration=duration,
            start_frame=int(strike_start["frame"]),
            end_frame=int(strike_end["frame"]),
            window_type="cycle"  # HS→HS full gait cycle
        )

        # Apply quality gates if enabled and data is available
        if apply_quality_gates and angles_df is not None and config is not None:
            quality_metrics = compute_window_quality(
                angles_df=angles_df,
                window_start=window.start_frame,
                window_end=window.end_frame,
                leg=leg,
                config=config,
                keypoints_df=keypoints_df
            )

            window.quality_metrics = quality_metrics

            # Check if window passes all gates
            if quality_metrics.passes_all_gates:
                rejection_counts['passed_qc'] += 1
                windows.append(window)
            else:
                # Track rejection reasons
                if not quality_metrics.passes_coverage_gate:
                    rejection_counts['rejected_coverage'] += 1
                if not quality_metrics.passes_gap_gate:
                    rejection_counts['rejected_gaps'] += 1
                if not quality_metrics.passes_stability_gate:
                    rejection_counts['rejected_stability'] += 1
                if not quality_metrics.passes_sanity_gate:
                    rejection_counts['rejected_sanity'] += 1
        else:
            # No quality gates - accept all windows
            rejection_counts['passed_qc'] += 1
            windows.append(window)

    return windows, rejection_counts


# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
# CYCLOGRAM EXTRACTION
# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

def predict_cyclogram_from_trends(angles_df: pd.DataFrame,
                                  window: StrideWindow,
                                  joint_pair: Tuple[str, str],
                                  col_map: Dict[str, Dict[str, str]],
                                  valid_cyclograms: List['CyclogramLoop'],
                                  n_points: int = 101) -> Optional['CyclogramLoop']:
    """
    Predict cyclogram using trends from valid neighboring cyclograms.

    Strategy:
    1. Use mean cyclogram from valid strides as baseline
    2. Apply temporal interpolation based on stride timing
    3. Scale based on stride duration
    """
    if not valid_cyclograms or len(valid_cyclograms) < 2:
        return None

    # Calculate mean cyclogram from valid strides
    mean_proximal = np.mean([c.proximal for c in valid_cyclograms], axis=0)
    mean_distal = np.mean([c.distal for c in valid_cyclograms], axis=0)

    # Apply duration-based scaling
    mean_duration = np.mean([c.duration for c in valid_cyclograms])
    duration_ratio = window.duration / mean_duration if mean_duration > 0 else 1.0

    # Scale amplitude by duration ratio (longer strides tend to have larger range)
    proximal_scaled = mean_proximal * np.clip(duration_ratio, 0.5, 1.5)
    distal_scaled = mean_distal * np.clip(duration_ratio, 0.5, 1.5)

    # Do NOT force loop closure here; report closure error downstream
    time_target = np.linspace(0, 100, n_points)

    return CyclogramLoop(
        leg=window.leg,
        stride_id=window.stride_id,
        joint_pair=joint_pair,
        proximal=proximal_scaled,
        distal=distal_scaled,
        time_normalized=time_target,
        duration=window.duration,
        speed=window.speed
    )


def interpolate_gaps(time: np.ndarray,
                     angles: np.ndarray,
                     fallback_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggressively fill ALL gaps to ensure 100% stride coverage.

    Strategy (hierarchical fallback):
    1. If ≥3 valid points: use linear interpolation (unconditional)
    2. If 1-2 valid points: fill with mean of available points
    3. If 0 valid points: use fallback_value

    Returns:
        Tuple of (time, angles) with NO NaN values - guaranteed 100% coverage
    """
    nan_mask = np.isnan(angles)
    valid_idx = ~nan_mask
    n_valid = np.sum(valid_idx)

    # Strategy 1: Linear interpolation (if ≥3 valid points)
    if n_valid >= 3:
        time_valid = time[valid_idx]
        angles_valid = angles[valid_idx]
        # Use linear interpolation with extrapolation for edges
        angles_filled = np.interp(time, time_valid, angles_valid)
        return time, angles_filled

    # Strategy 2: Fill with mean (1-2 valid points)
    elif n_valid >= 1:
        valid_vals = angles[valid_idx]
        mean_val = np.mean(valid_vals)
        # Fill all NaN with mean of available points
        angles_filled = np.full_like(angles, mean_val, dtype=float)
        angles_filled[valid_idx] = angles[valid_idx]
        return time, angles_filled

    # Strategy 3: Use fallback value (0 valid points)
    else:
        # All NaN - use fallback value
        angles_filled = np.full_like(angles, fallback_value, dtype=float)
        return time, angles_filled


def extract_cyclogram(angles_df: pd.DataFrame,
                     window: StrideWindow,
                     joint_pair: Tuple[str, str],
                     col_map: Dict[str, Dict[str, str]],
                     n_points: int = 101,
                     valid_cyclograms: Optional[List['CyclogramLoop']] = None) -> CyclogramLoop:
    """
    Generate normalized cyclogram loop for a stride.

    Steps:
    1. Extract time window from angles dataframe
    2. Get proximal and distal joint angles
    3. Apply adaptive interpolation to fill small gaps (<30% NaN, gaps <20% duration)
    4. Resample to n_points using normalized time
    5. If insufficient data, use predictive interpolation from valid strides
    """
    # Time window selection
    mask = (angles_df["timestamp"] >= window.start_time) & \
           (angles_df["timestamp"] <= window.end_time)
    stride_data = angles_df[mask].copy()

    # Check for insufficient data - use predictive approach
    if len(stride_data) < 10:
        if valid_cyclograms and len(valid_cyclograms) >= 2:
            print(f"  📊 Predicting cyclogram for stride {window.stride_id} using trends from {len(valid_cyclograms)} valid strides")
            predicted = predict_cyclogram_from_trends(angles_df, window, joint_pair,
                                                     col_map, valid_cyclograms, n_points)
            if predicted:
                return predicted
        raise ValueError(f"Insufficient data points in stride {window.stride_id}")

    # Extract joint angles
    proximal_col = col_map[window.leg][joint_pair[0]]
    distal_col = col_map[window.leg][joint_pair[1]]

    proximal = stride_data[proximal_col].values
    distal = stride_data[distal_col].values

    # Normalize time to 0–100% GC
    time_actual = stride_data["timestamp"].to_numpy(dtype=float)

    # Apply aggressive interpolation to ensure 100% coverage (NO NaN values)
    time_actual, proximal = interpolate_gaps(time_actual, proximal, fallback_value=0.0)
    time_actual, distal = interpolate_gaps(time_actual, distal, fallback_value=0.0)

    # Verify no NaN values remain (should be guaranteed by aggressive interpolation)
    if np.isnan(proximal).any() or np.isnan(distal).any():
        # This should never happen, but failsafe
        proximal = np.nan_to_num(proximal, nan=0.0)
        distal = np.nan_to_num(distal, nan=0.0)

    # Verify all values are finite
    valid_mask = (np.isfinite(time_actual) & np.isfinite(proximal) & np.isfinite(distal))
    time_actual = time_actual[valid_mask]
    proximal = proximal[valid_mask]
    distal = distal[valid_mask]

    # Minimum data requirement (should always pass with aggressive interpolation)
    if len(time_actual) < 3:
        raise ValueError(f"Insufficient valid time points after interpolation: {len(time_actual)}")

    # Prevent division by zero / duplicate stamps
    denom = (time_actual[-1] - time_actual[0])
    if not isfinite(denom) or denom <= 0:
        raise ValueError("Non-increasing timestamps within window")
    time_norm = (time_actual - time_actual[0]) / denom * 100.0
    # Drop any duplicates in time (PCHIP requires strictly increasing x)
    uniq_idx = np.flatnonzero(np.r_[True, np.diff(time_norm) > 0])
    time_norm = time_norm[uniq_idx]
    proximal = proximal[uniq_idx]
    distal = distal[uniq_idx]

    # Resample to 0..100 with PCHIP, no extrapolation
    time_target = np.linspace(0.0, 100.0, n_points)
    # Clamp targets to observed range to avoid extrapolation artifacts
    tmin, tmax = float(time_norm.min()), float(time_norm.max())
    mask = (time_target >= tmin) & (time_target <= tmax)
    prox_interp = PchipInterpolator(time_norm, proximal)
    dist_interp = PchipInterpolator(time_norm, distal)
    proximal_resampled = np.full_like(time_target, np.nan, dtype=float)
    distal_resampled   = np.full_like(time_target, np.nan, dtype=float)
    proximal_resampled[mask] = prox_interp(time_target[mask])
    distal_resampled[mask]   = dist_interp(time_target[mask])

    # Loop quality metrics
    nan_percent = (np.isnan(proximal_resampled) | np.isnan(distal_resampled)).mean() * 100.0
    # Natural closure error: distance between first and last valid points (only meaningful for HS→HS)
    first = np.array([proximal_resampled[mask][0],  distal_resampled[mask][0]]) if mask.any() else np.array([np.nan, np.nan])
    last  = np.array([proximal_resampled[mask][-1], distal_resampled[mask][-1]]) if mask.any() else np.array([np.nan, np.nan])
    closure_error = float(np.linalg.norm(last - first)) if window.window_type == "cycle" else np.nan

    return CyclogramLoop(
        leg=window.leg,
        stride_id=window.stride_id,
        joint_pair=joint_pair,
        proximal=proximal_resampled,
        distal=distal_resampled,
        time_normalized=time_target,
        duration=window.duration,
        is_full_cycle=(window.window_type == "cycle"),
        closure_error_deg=(0.0 if not isfinite(closure_error) else closure_error),
        nan_percent=float(nan_percent),
        speed=window.speed
    )


# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
# STRIDE PAIRING
# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

def pair_strides(left_windows: List[StrideWindow],
                 right_windows: List[StrideWindow],
                 config: Optional['AnalysisConfig'] = None,
                 tolerance_ratio: float = 0.15) -> Tuple[List[PairedStrides], Dict[str, int]]:
    """
    Match left and right strides/cycles with quality-aware pairing.

    For CYCLES (HS→HS): Pair by INDEX (L_k ↔ R_k) with phase overlap validation
    For STANCE (HS→TO): Fall back to mid-time proximity pairing

    Returns:
        Tuple of (pairs, unpaired_counts)
        - pairs: List of successfully paired strides
        - unpaired_counts: {'unpaired_L': int, 'unpaired_R': int}
    """
    pairs: List[PairedStrides] = []
    used_right = set()

    # Get min overlap threshold from config
    min_phase_overlap = config.min_phase_overlap if config else 0.3

    # If these are full cycles, pair by index (L_k ↔ R_k) after aligning starts
    if left_windows and right_windows and \
       left_windows[0].window_type == "cycle" and right_windows[0].window_type == "cycle":
        L = sorted(left_windows, key=lambda w: w.start_time)
        R = sorted(right_windows, key=lambda w: w.start_time)

        # Trim the earlier-leading side so starts are aligned
        while L and R and abs(L[0].start_time - R[0].start_time) > tolerance_ratio * max(L[0].duration, R[0].duration):
            if L[0].start_time < R[0].start_time:
                L.pop(0)
            else:
                R.pop(0)

        n = min(len(L), len(R))
        valid_pairs = []

        for i in range(n):
            l, r = L[i], R[i]

            # Calculate temporal overlap
            overlap_start = max(l.start_time, r.start_time)
            overlap_end   = min(l.end_time,   r.end_time)
            overlap = max(0.0, overlap_end - overlap_start)

            # Compute overlap ratio (fraction of cycle duration)
            max_duration = max(l.duration, r.duration)
            overlap_ratio = overlap / max_duration if max_duration > 0 else 0.0

            # Apply phase overlap validation
            if overlap_ratio >= min_phase_overlap:
                overlap_pct = (overlap / ((l.duration + r.duration) / 2.0)) * 100.0
                valid_pairs.append(PairedStrides(
                    l, r,
                    time_difference=abs(l.start_time - r.start_time),
                    temporal_overlap=overlap_pct
                ))

        # Count unpaired cycles
        unpaired_counts = {
            'unpaired_L': len(L) - len(valid_pairs),
            'unpaired_R': len(R) - len(valid_pairs)
        }

        return valid_pairs, unpaired_counts

    # Otherwise (stance windows), keep mid-time proximity pairing
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
            # Calculate temporal overlap
            overlap_start = max(left_win.start_time, best_right.start_time)
            overlap_end = min(left_win.end_time, best_right.end_time)
            overlap = max(0, overlap_end - overlap_start)
            total = (left_win.duration + best_right.duration) / 2
            overlap_pct = (overlap / total) * 100 if total > 0 else 0

            pairs.append(PairedStrides(
                left_stride=left_win,
                right_stride=best_right,
                time_difference=min_diff,
                temporal_overlap=overlap_pct
            ))
            used_right.add(best_right.stride_id)

    # Count unpaired for stance windows
    unpaired_counts = {
        'unpaired_L': len(left_windows) - len(pairs),
        'unpaired_R': len(right_windows) - len(used_right)
    }

    return pairs, unpaired_counts


# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
# METRICS CALCULATION
# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

def calculate_loop_area(loop: CyclogramLoop) -> float:
    """
    Compute signed polygon area using shoelace formula.

    A = 0.5 * �(x_i * y_{i+1} - x_{i+1} * y_i)
    """
    x = loop.proximal
    y = loop.distal

    # Close the loop
    x_closed = np.append(x, x[0])
    y_closed = np.append(y, y[0])

    area = 0.5 * np.sum(x_closed[:-1] * y_closed[1:] - x_closed[1:] * y_closed[:-1])

    return area


def calculate_normalized_area(loop: CyclogramLoop) -> float:
    """
    Signed area normalized by 1-σ ellipse area (π σx σy).
    Scale-free and comparable across sessions/subjects.
    """
    x, y = loop.proximal, loop.distal
    sx, sy = float(np.nanstd(x)), float(np.nanstd(y))
    base = np.pi * max(sx, 1e-6) * max(sy, 1e-6)
    return calculate_loop_area(loop) / base


def calculate_delta_area_percent(area_L: float, area_R: float) -> float:
    """
    Normalized area difference: 200 * (AL - AR) / (|AL| + |AR|)
    Range: [-200, 200] where 0 = perfect symmetry
    """
    denominator = abs(area_L) + abs(area_R)
    if denominator < 1e-6:
        return 0.0
    return 200.0 * (area_L - area_R) / denominator


def calculate_rmse(loop_L: CyclogramLoop, loop_R: CyclogramLoop) -> float:
    """
    Compute shape distance after z-score normalization.

    Steps:
    1. Z-score normalize each loop independently
    2. Compute Euclidean distance at each time point
    3. Return RMSE
    """
    def zscore_normalize(arr):
        std = np.std(arr)
        if std < 1e-8:
            return arr - np.mean(arr)
        return (arr - np.mean(arr)) / std

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


def procrustes_distance(loop_L: CyclogramLoop, loop_R: CyclogramLoop) -> float:
    """
    Compute optimal rigid alignment distance using Procrustes analysis.

    Returns disparity after optimal translation, scaling, and rotation.
    """
    X = loop_L.points
    Y = loop_R.points

    try:
        _, _, disparity = procrustes(X, Y)
        return disparity
    except:
        # Fallback if procrustes fails
        return np.linalg.norm(X - Y) / len(X)


def dtw_distance(loop_L: CyclogramLoop, loop_R: CyclogramLoop) -> float:
    """
    Dynamic Time Warping distance between 2D trajectories over %GC.

    Captures phase warps that Procrustes/RMSE miss - critical for timing asymmetry.
    """
    A = loop_L.points.astype(float)
    B = loop_R.points.astype(float)

    # Replace NaNs with linear interpolation fallback
    if np.isnan(A).any() or np.isnan(B).any():
        def _fill(arr):
            m = np.isnan(arr).any(axis=1)
            if m.all():
                return np.zeros_like(arr)
            idx = np.arange(len(arr))
            for d in (0, 1):
                good = ~np.isnan(arr[:, d])
                if good.any():
                    arr[:, d] = np.interp(idx, idx[good], arr[good, d])
            return arr
        A = _fill(A.copy())
        B = _fill(B.copy())

    if HAS_FASTDTW:
        dist, _ = fastdtw(A, B, dist=lambda x, y: float(np.linalg.norm(x - y)))
        # Normalize by length to make it comparable across sessions
        return float(dist) / max(len(A), len(B))

    # O(N²) DP fallback (fine for 101 points)
    N, M = len(A), len(B)
    D = np.full((N+1, M+1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, N+1):
        for j in range(1, M+1):
            cost = float(np.linalg.norm(A[i-1] - B[j-1]))
            D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    return float(D[N, M]) / max(N, M)


def calculate_orientation(loop: CyclogramLoop) -> float:
    """
    Extract major axis orientation via PCA.

    Returns angle of first principal component in degrees.
    """
    points = loop.points

    # Check for NaN values
    if np.isnan(points).any():
        # Remove rows with NaN
        valid_mask = ~np.isnan(points).any(axis=1)
        points = points[valid_mask]

    if len(points) < 3:
        # Not enough points for PCA
        return 0.0

    # Center the data
    centered = points - np.mean(points, axis=0)

    # PCA
    pca = PCA(n_components=1)
    pca.fit(centered)

    # Principal component direction
    pc1 = pca.components_[0]

    # Angle in degrees
    angle = np.arctan2(pc1[1], pc1[0]) * 180 / np.pi

    return angle


def calculate_delta_orientation(angle_L: float, angle_R: float) -> float:
    """
    Minimum angular difference (handles wrap-around).
    """
    diff = abs(angle_L - angle_R)
    if diff > 180:
        diff = 360 - diff
    return diff


def detect_hysteresis(loop: CyclogramLoop) -> str:
    """
    Determine loop traversal direction.

    Positive area = CCW, Negative area = CW
    """
    area = calculate_loop_area(loop)
    return "CCW" if area > 0 else "CW"


def calculate_similarity_score(metrics: CyclogramMetrics,
                              weights: Dict[str, float]) -> float:
    """
    Compute composite LR similarity score (0-100).

    100 = perfect symmetry, 0 = maximum asymmetry
    """
    # Normalize delta_area_percent: |�A%| � [0, 1]
    # Assume 0% = perfect (1.0), 50% = poor (0.0)
    area_norm = max(0, 1 - abs(metrics.delta_area_percent) / 50.0)

    # Normalize Procrustes: 0 = perfect (1.0), >0.5 = poor (0.0)
    procrustes_norm = max(0, 1 - metrics.procrustes_distance / 0.5)

    # Normalize RMSE: 0 = perfect (1.0), >1.0 = poor (0.0)
    rmse_norm = max(0, 1 - metrics.rmse / 1.0)

    # Normalize orientation: 0� = perfect (1.0), 30� = poor (0.0)
    orient_norm = max(0, 1 - metrics.delta_orientation / 30.0)

    # Weighted combination
    score = (weights["area"] * area_norm +
             weights["procrustes"] * procrustes_norm +
             weights["rmse"] * rmse_norm +
             weights["orientation"] * orient_norm) * 100

    return np.clip(score, 0, 100)


def compute_all_metrics(loop_L: CyclogramLoop,
                       loop_R: CyclogramLoop,
                       weights: Dict[str, float]) -> CyclogramMetrics:
    """
    Compute all metrics for a paired cyclogram.

    Area/hysteresis only computed for closed full cycles (HS→HS).
    Shape metrics (RMSE, Procrustes, DTW) computed for all loops.
    """
    # Area metrics — only if loops are closed full cycles
    if loop_L.is_closed and loop_R.is_closed:
        area_L = calculate_loop_area(loop_L)
        area_R = calculate_loop_area(loop_R)
        delta_area_pct = calculate_delta_area_percent(area_L, area_R)
        narea_L = calculate_normalized_area(loop_L)
        narea_R = calculate_normalized_area(loop_R)
        hyst_L = detect_hysteresis(loop_L)
        hyst_R = detect_hysteresis(loop_R)
        hyst_mismatch = (hyst_L != hyst_R)
    else:
        area_L = area_R = delta_area_pct = 0.0
        narea_L = narea_R = 0.0
        hyst_L = hyst_R = "NA"
        hyst_mismatch = False

    # Shape metrics (always computed)
    rmse = calculate_rmse(loop_L, loop_R)
    procrustes_dist = procrustes_distance(loop_L, loop_R)
    dtw = dtw_distance(loop_L, loop_R)

    # Orientation metrics
    orient_L = calculate_orientation(loop_L)
    orient_R = calculate_orientation(loop_R)
    delta_orient = calculate_delta_orientation(orient_L, orient_R)

    # Create metrics object
    metrics = CyclogramMetrics(
        joint_pair=loop_L.joint_pair,
        left_stride_id=loop_L.stride_id,
        right_stride_id=loop_R.stride_id,
        area_left=area_L,
        area_right=area_R,
        delta_area_percent=delta_area_pct,
        rmse=rmse,
        procrustes_distance=procrustes_dist,
        dtw=dtw,
        orientation_left=orient_L,
        orientation_right=orient_R,
        delta_orientation=delta_orient,
        hysteresis_left=hyst_L,
        hysteresis_right=hyst_R,
        hysteresis_mismatch=hyst_mismatch,
        normalized_area_left=narea_L,
        normalized_area_right=narea_R,
        similarity_score=0.0  # Calculate next
    )

    # Calculate similarity score
    metrics.similarity_score = calculate_similarity_score(metrics, weights)

    return metrics


# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
# VISUALIZATION
# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

def format_statistics(metrics: List[CyclogramMetrics],
                     n_left: int, n_right: int) -> str:
    """Format statistics text box content."""
    if not metrics:
        return f"n(Left): {n_left}\nn(Right): {n_right}\nNo paired metrics"

    delta_area_values = [m.delta_area_percent for m in metrics]
    procrustes_values = [m.procrustes_distance for m in metrics]
    similarity_values = [m.similarity_score for m in metrics]

    return f"""n(Left): {n_left}
n(Right): {n_right}
�Area%: {np.mean(delta_area_values):.1f}�{np.std(delta_area_values):.1f}
Procrustes: {np.mean(procrustes_values):.3f}
Similarity: {np.mean(similarity_values):.1f}�{np.std(similarity_values):.1f}"""


def plot_overlayed_cyclograms(loops_L: List[CyclogramLoop],
                              loops_R: List[CyclogramLoop],
                              metrics: List[CyclogramMetrics],
                              joint_pair: Tuple[str, str],
                              output_path: str,
                              plot_config: Optional[PlotConfig] = None) -> None:
    """
    Create dashboard-style cyclogram plot with separate left/right subplots.

    Uses centralized PlotConfig for consistent styling across all plots.

    Features:
    - Side-by-side subplots for LEFT and RIGHT legs
    - Individual strides with stride ID annotations
    - Mean loop highlighted
    - Asymmetry indicators showing problematic side
    - Statistics boxes for each side
    """
    # Use default config if none provided
    if plot_config is None:
        plot_config = PlotConfig()

    # Validate configuration
    plot_config.validate()

    # Create figure with 2 subplots
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2,
        figsize=plot_config.cyclogram_figsize,
        dpi=plot_config.dpi
    )

    # Set figure background
    fig.patch.set_facecolor(plot_config.background_color)

    # Joint names for labels
    joint_names = {
        "hip": "Hip",
        "knee": "Knee",
        "ankle": "Ankle"
    }

    proximal_name = joint_names.get(joint_pair[0], joint_pair[0].capitalize())
    distal_name = joint_names.get(joint_pair[1], joint_pair[1].capitalize())

    # Determine asymmetry level from metrics
    asymmetry_status = "UNKNOWN"
    problematic_side = None
    similarity_score = 50.0  # Default

    if metrics:
        similarity_values = [m.similarity_score for m in metrics if not np.isnan(m.similarity_score)]
        if similarity_values:
            similarity_score = np.mean(similarity_values)
            delta_area_values = [m.delta_area_percent for m in metrics if not np.isnan(m.delta_area_percent)]

            # Determine asymmetry level
            if similarity_score >= 80:
                asymmetry_status = "GOOD SYMMETRY"
            elif similarity_score >= 60:
                asymmetry_status = "MILD ASYMMETRY"
            else:
                asymmetry_status = "SIGNIFICANT ASYMMETRY"

            # Determine problematic side based on area difference
            if delta_area_values:
                mean_delta = np.mean(delta_area_values)
                if abs(mean_delta) > 20:  # >20% difference
                    if mean_delta > 0:
                        problematic_side = "RIGHT"  # Left > Right, so Right is smaller
                    else:
                        problematic_side = "LEFT"

    # Color coding based on asymmetry
    if problematic_side == "LEFT":
        left_edgecolor = plot_config.warning_color  # Highlight problematic side
        left_linewidth = plot_config.threshold_linewidth * 2
    else:
        left_edgecolor = plot_config.left_color
        left_linewidth = plot_config.threshold_linewidth

    if problematic_side == "RIGHT":
        right_edgecolor = plot_config.warning_color  # Highlight problematic side
        right_linewidth = plot_config.threshold_linewidth * 2
    else:
        right_edgecolor = plot_config.right_color
        right_linewidth = plot_config.threshold_linewidth

    # ========== LEFT SUBPLOT ==========
    ax_left.set_facecolor('white')

    # Plot individual left strides with annotations
    for i, loop in enumerate(loops_L):
        ax_left.plot(loop.proximal, loop.distal,
                    color=plot_config.left_color,
                    alpha=plot_config.individual_alpha,
                    linewidth=plot_config.individual_linewidth,
                    zorder=1)

        # Add stride ID annotation at start point (only if not NaN)
        if not (np.isnan(loop.proximal[0]) or np.isnan(loop.distal[0])):
            ax_left.text(loop.proximal[0], loop.distal[0], f'L{loop.stride_id}',
                        fontsize=plot_config.annotation_fontsize,
                        color=plot_config.left_color,
                        fontweight='bold',
                        bbox=dict(boxstyle=f'round,pad={plot_config.annotation_box_pad}',
                                 facecolor='white',
                                 edgecolor=plot_config.left_color,
                                 alpha=plot_config.annotation_box_alpha))

    # Calculate and plot mean left loop
    if loops_L:
        mean_L = np.mean([loop.points for loop in loops_L], axis=0)
        ax_left.plot(mean_L[:, 0], mean_L[:, 1],
                    color=plot_config.left_mean_color, linewidth=plot_config.mean_linewidth, label='Mean', zorder=10,
                    marker=plot_config.mean_marker, markevery=plot_config.mean_markevery, markersize=plot_config.mean_markersize)

        # Add mean label at midpoint (only if not NaN)
        mid_idx = len(mean_L) // 2
        if not (np.isnan(mean_L[mid_idx, 0]) or np.isnan(mean_L[mid_idx, 1])):
            ax_left.text(mean_L[mid_idx, 0], mean_L[mid_idx, 1], 'MEAN',
                        fontsize=plot_config.stats_fontsize, color=plot_config.left_mean_color, fontweight='bold',
                        bbox=dict(boxstyle=f'round,pad={plot_config.annotation_box_pad}', facecolor='yellow',
                                 edgecolor=plot_config.left_mean_color, alpha=0.9))

    # Statistics box for left
    if loops_L:
        area_values = [calculate_loop_area(loop) for loop in loops_L]
        area_values = [a for a in area_values if not np.isnan(a)]  # Filter NaN

        if area_values:
            stats_left = f"LEFT LEG\n{'─'*20}\n"
            stats_left += f"Strides: {len(loops_L)}\n"
            stats_left += f"Area: {np.mean(area_values):.1f}±{np.std(area_values):.1f} deg²\n"

            # Add metrics for left side (from paired comparisons)
            left_metrics = [m for m in metrics if m.left_stride_id in [l.stride_id for l in loops_L]]
            if left_metrics:
                procrustes_vals = [m.procrustes_distance for m in left_metrics if not np.isnan(m.procrustes_distance)]
                similarity_vals = [m.similarity_score for m in left_metrics if not np.isnan(m.similarity_score)]
                delta_area_vals = [m.delta_area_percent for m in left_metrics if not np.isnan(m.delta_area_percent)]

                if procrustes_vals:
                    stats_left += f"Procrustes: {np.mean(procrustes_vals):.3f}±{np.std(procrustes_vals):.3f}\n"
                if similarity_vals:
                    stats_left += f"Similarity: {np.mean(similarity_vals):.1f}±{np.std(similarity_vals):.1f}\n"
                if delta_area_vals:
                    stats_left += f"ΔArea%: {np.mean(delta_area_vals):.1f}±{np.std(delta_area_vals):.1f}\n"

            if problematic_side == "LEFT":
                stats_left += f"\n⚠️ PROBLEMATIC"

            ax_left.text(0.02, 0.98, stats_left, transform=ax_left.transAxes,
                        fontsize=plot_config.stats_fontsize, verticalalignment='top',
                        bbox=dict(boxstyle=f'round,pad={plot_config.stats_box_pad}', facecolor='#FFE6E6' if problematic_side == "LEFT" else '#E6F2FF',
                                 edgecolor=left_edgecolor, linewidth=left_linewidth, alpha=0.9),
                        fontfamily='monospace')

    # Formatting left subplot
    ax_left.set_xlabel(f"{proximal_name} Angle (°)", fontsize=plot_config.label_fontsize, fontweight='bold')
    ax_left.set_ylabel(f"{distal_name} Angle (°)", fontsize=plot_config.label_fontsize, fontweight='bold')
    ax_left.set_title(f"LEFT - {proximal_name}-{distal_name}",
                     fontsize=plot_config.subtitle_fontsize, fontweight='bold', color=plot_config.left_color, pad=15)
    ax_left.grid(True, alpha=plot_config.individual_alpha, linestyle='--')
    ax_left.set_aspect('equal', adjustable='box')
    ax_left.legend(loc='lower right', fontsize=plot_config.stats_fontsize, framealpha=0.9)

    # ========== RIGHT SUBPLOT ==========
    ax_right.set_facecolor('white')

    # Plot individual right strides with annotations
    for i, loop in enumerate(loops_R):
        ax_right.plot(loop.proximal, loop.distal,
                     color=plot_config.right_color, alpha=plot_config.individual_alpha, linewidth=plot_config.individual_linewidth, zorder=1)

        # Add stride ID annotation at start point (only if not NaN)
        if not (np.isnan(loop.proximal[0]) or np.isnan(loop.distal[0])):
            ax_right.text(loop.proximal[0], loop.distal[0], f'R{loop.stride_id}',
                         fontsize=plot_config.annotation_fontsize, color=plot_config.right_color, fontweight='bold',
                         bbox=dict(boxstyle=f'round,pad={plot_config.annotation_box_pad}', facecolor='white',
                                  edgecolor=plot_config.right_color, alpha=0.7))

    # Calculate and plot mean right loop
    if loops_R:
        mean_R = np.mean([loop.points for loop in loops_R], axis=0)
        ax_right.plot(mean_R[:, 0], mean_R[:, 1],
                     color=plot_config.right_mean_color, linewidth=plot_config.mean_linewidth, label='Mean', zorder=10,
                     marker=plot_config.mean_marker, markevery=plot_config.mean_markevery, markersize=plot_config.mean_markersize)

        # Add mean label at midpoint (only if not NaN)
        mid_idx = len(mean_R) // 2
        if not (np.isnan(mean_R[mid_idx, 0]) or np.isnan(mean_R[mid_idx, 1])):
            ax_right.text(mean_R[mid_idx, 0], mean_R[mid_idx, 1], 'MEAN',
                         fontsize=plot_config.stats_fontsize, color=plot_config.right_mean_color, fontweight='bold',
                         bbox=dict(boxstyle=f'round,pad={plot_config.annotation_box_pad}', facecolor='yellow',
                                  edgecolor=plot_config.right_mean_color, alpha=0.9))

    # Statistics box for right
    if loops_R:
        area_values = [calculate_loop_area(loop) for loop in loops_R]
        area_values = [a for a in area_values if not np.isnan(a)]  # Filter NaN

        if area_values:
            stats_right = f"RIGHT LEG\n{'─'*20}\n"
            stats_right += f"Strides: {len(loops_R)}\n"
            stats_right += f"Area: {np.mean(area_values):.1f}±{np.std(area_values):.1f} deg²\n"

            # Add metrics for right side (from paired comparisons)
            right_metrics = [m for m in metrics if m.right_stride_id in [r.stride_id for r in loops_R]]
            if right_metrics:
                procrustes_vals = [m.procrustes_distance for m in right_metrics if not np.isnan(m.procrustes_distance)]
                similarity_vals = [m.similarity_score for m in right_metrics if not np.isnan(m.similarity_score)]
                delta_area_vals = [m.delta_area_percent for m in right_metrics if not np.isnan(m.delta_area_percent)]

                if procrustes_vals:
                    stats_right += f"Procrustes: {np.mean(procrustes_vals):.3f}±{np.std(procrustes_vals):.3f}\n"
                if similarity_vals:
                    stats_right += f"Similarity: {np.mean(similarity_vals):.1f}±{np.std(similarity_vals):.1f}\n"
                if delta_area_vals:
                    stats_right += f"ΔArea%: {np.mean(delta_area_vals):.1f}±{np.std(delta_area_vals):.1f}\n"

            if problematic_side == "RIGHT":
                stats_right += f"\n⚠️ PROBLEMATIC"

            ax_right.text(0.02, 0.98, stats_right, transform=ax_right.transAxes,
                         fontsize=plot_config.stats_fontsize, verticalalignment='top',
                         bbox=dict(boxstyle=f'round,pad={plot_config.stats_box_pad}', facecolor='#FFE6E6' if problematic_side == "RIGHT" else '#E6F2FF',
                                  edgecolor=right_edgecolor, linewidth=right_linewidth, alpha=0.9),
                         fontfamily='monospace')

    # Formatting right subplot
    ax_right.set_xlabel(f"{proximal_name} Angle (°)", fontsize=plot_config.label_fontsize, fontweight='bold')
    ax_right.set_ylabel(f"{distal_name} Angle (°)", fontsize=plot_config.label_fontsize, fontweight='bold')
    ax_right.set_title(f"RIGHT - {proximal_name}-{distal_name}",
                      fontsize=plot_config.subtitle_fontsize, fontweight='bold', color=plot_config.right_color, pad=15)
    ax_right.grid(True, alpha=plot_config.individual_alpha, linestyle='--')
    ax_right.set_aspect('equal', adjustable='box')
    ax_right.legend(loc='lower right', fontsize=plot_config.stats_fontsize, framealpha=0.9)

    # Add overall title with asymmetry status
    title_color = 'green' if similarity_score >= 80 else ('orange' if similarity_score >= 60 else 'red')
    fig.suptitle(f"Cyclogram Dashboard: {proximal_name}-{distal_name} | {asymmetry_status}",
                fontsize=16, fontweight='bold', y=0.98, color=title_color)

    # Add comparison metrics box at bottom
    if metrics:
        delta_area_values = [m.delta_area_percent for m in metrics if not np.isnan(m.delta_area_percent)]
        similarity_values = [m.similarity_score for m in metrics if not np.isnan(m.similarity_score)]
        procrustes_values = [m.procrustes_distance for m in metrics if not np.isnan(m.procrustes_distance)]

        comparison_text = f"L-R COMPARISON METRICS\n"
        comparison_text += f"Paired Strides: {len(metrics)}  |  "
        if delta_area_values:
            comparison_text += f"ΔArea: {np.mean(delta_area_values):.1f}%  |  "
        if similarity_values:
            comparison_text += f"Similarity: {np.mean(similarity_values):.1f}  |  "
        if procrustes_values:
            comparison_text += f"Procrustes: {np.mean(procrustes_values):.3f}"

        if problematic_side:
            comparison_text += f"\n⚠️ {problematic_side} SIDE SHOWS ASYMMETRY"

        box_color = '#FFE6E6' if problematic_side else '#FFFACD'
        fig.text(0.5, 0.02, comparison_text, ha='center', fontsize=11,
                bbox=dict(boxstyle=f'round,pad={plot_config.stats_box_pad}', facecolor=box_color,
                         edgecolor='black', linewidth=2, alpha=0.9),
                fontfamily='monospace', fontweight='bold')

    if plot_config.tight_layout:
        plt.tight_layout(rect=[0, 0.06, 1, 0.96])

    plt.savefig(output_path,
               dpi=plot_config.dpi,
               bbox_inches=plot_config.bbox_inches,
               facecolor=plot_config.background_color)
    plt.close()


def plot_similarity_summary(session_summary: pd.DataFrame,
                           output_path: str,
                           plot_config: Optional[PlotConfig] = None) -> None:
    """
    Create bar chart of per-pair similarity scores.

    Uses centralized PlotConfig for consistent styling.
    """
    if session_summary.empty:
        return

    # Use default config if none provided
    if plot_config is None:
        plot_config = PlotConfig()

    # Validate configuration
    plot_config.validate()

    fig, ax = plt.subplots(
        figsize=plot_config.similarity_figsize,
        dpi=plot_config.dpi
    )

    x = np.arange(len(session_summary))
    means = session_summary['similarity_mean'].values
    stds = session_summary['similarity_std'].values

    bars = ax.bar(x, means, yerr=stds,
                  capsize=plot_config.capsize,
                  color=[plot_config.left_color,
                         plot_config.good_color,
                         plot_config.excellent_color],
                  alpha=plot_config.bar_alpha)

    ax.set_xticks(x)
    ax.set_xticklabels(session_summary['joint_pair'].values,
                      fontsize=plot_config.label_fontsize)
    ax.set_ylabel('LR Similarity Score',
                  fontsize=plot_config.label_fontsize,
                  fontweight=plot_config.label_fontweight)
    ax.set_ylim(0, 100)

    # Threshold lines
    ax.axhline(plot_config.excellent_threshold,
              color=plot_config.excellent_color,
              linestyle=plot_config.grid_linestyle,
              alpha=plot_config.threshold_alpha,
              label=f'Excellent (>{plot_config.excellent_threshold:.0f})')
    ax.axhline(plot_config.good_threshold,
              color=plot_config.good_color,
              linestyle=plot_config.grid_linestyle,
              alpha=plot_config.threshold_alpha,
              label=f'Good (>{plot_config.good_threshold:.0f})')

    ax.legend(fontsize=plot_config.legend_fontsize)
    ax.grid(axis='y',
           alpha=plot_config.grid_alpha,
           linestyle=plot_config.grid_linestyle)

    plt.title('Left-Right Similarity by Joint Pair',
             fontsize=plot_config.title_fontsize,
             fontweight=plot_config.title_fontweight)

    if plot_config.tight_layout:
        plt.tight_layout()

    plt.savefig(output_path,
               dpi=plot_config.dpi,
               bbox_inches=plot_config.bbox_inches)
    plt.close()


# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
# OUTPUT SERIALIZATION
# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

def write_stride_metrics(metrics: List[CyclogramMetrics],
                        output_path: str) -> pd.DataFrame:
    """
    Write per-stride metrics CSV and return DataFrame.
    """
    if not metrics:
        return pd.DataFrame()

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
            'dtw': m.dtw,
            'delta_orient': m.delta_orientation,
            'hysteresis_L': m.hysteresis_left,
            'hysteresis_R': m.hysteresis_right,
            'hysteresis_mismatch': m.hysteresis_mismatch,
            'narea_L': m.normalized_area_left,
            'narea_R': m.normalized_area_right,
            'similarity_score': m.similarity_score
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, float_format='%.4f')
    return df


def write_session_summary(metrics: List[CyclogramMetrics],
                         output_path: str) -> pd.DataFrame:
    """
    Aggregate metrics by joint pair and return DataFrame.
    """
    if not metrics:
        return pd.DataFrame()

    df = pd.DataFrame([{
        'joint_pair': f"{m.joint_pair[0]}-{m.joint_pair[1]}",
        'delta_area_pct': m.delta_area_percent,
        'procrustes': m.procrustes_distance,
        'similarity_score': m.similarity_score
    } for m in metrics])

    summary = df.groupby('joint_pair').agg({
        'delta_area_pct': ['count', 'mean', 'std'],
        'procrustes': ['mean', 'std'],
        'similarity_score': ['mean', 'std']
    }).round(4)

    summary.columns = ['n_pairs', 'delta_area_mean', 'delta_area_std',
                      'procrustes_mean', 'procrustes_std',
                      'similarity_mean', 'similarity_std']

    summary = summary.reset_index()
    summary.to_csv(output_path, index=False)
    return summary


# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
# MAIN ANALYSIS PIPELINE
# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

def calculate_data_quality(angles_df: pd.DataFrame, col_map: Dict[str, Dict[str, str]]) -> float:
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


def loop_quality_ok(loop: CyclogramLoop,
                    max_nan_percent: float = 10.0,
                    max_closure_error_deg: float = 2.0,
                    min_sigma_deg: float = 2.0,
                    data_quality: Optional[float] = None) -> bool:
    """
    Adaptive quality gate that adjusts thresholds based on data quality.

    When data quality is poor (quality < 0.5), gates become more lenient:
    - NaN threshold increases (allow more NaN)
    - Closure threshold increases (allow worse closure)
    - Variance threshold decreases (allow flatter lines)

    When data quality is perfect (>0.95), SKIP ALL QC gates since the data
    was already validated by the imputation pipeline.

    Returns:
        True if loop passes quality checks, False otherwise
    """
    # Skip QC entirely for imputed data (perfect quality)
    if data_quality is not None and data_quality > 0.95:
        return True  # Trust imputed data quality validation

    # Scale thresholds based on data quality if provided
    if data_quality is not None:
        # Good quality (1.0): use base thresholds
        # Poor quality (0.0): relax thresholds 3x
        quality_factor = 1.0 + (1.0 - data_quality) * 2.0  # Range: 1.0 to 3.0

        max_nan = max_nan_percent * quality_factor
        max_closure = max_closure_error_deg * quality_factor
        min_variance = min_sigma_deg / quality_factor
    else:
        max_nan = max_nan_percent
        max_closure = max_closure_error_deg
        min_variance = min_sigma_deg

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


def analyze_subject(subject_dir: Path,
                   output_dir: Path,
                   config: AnalysisConfig) -> Dict:
    """
    Run complete cyclogram analysis for a single subject.

    Returns:
        Dict with analysis results and metadata
    """
    subject_name = subject_dir.name
    print(f"\n{'='*80}")
    print(f"Processing: {subject_name}")
    print(f"{'='*80}")

    # Create subject output directory
    subject_output = output_dir / subject_name
    subject_output.mkdir(parents=True, exist_ok=True)

    # Load data
    try:
        raw_angles_path = subject_dir / "Raw_Angles.csv"
        events_path = subject_dir / "Angle_Events.csv"

        if not raw_angles_path.exists():
            print(f"Error: Raw_Angles.csv not found in {subject_dir}")
            return None

        if not events_path.exists():
            print(f"Error: Angle_Events.csv not found in {subject_dir}")
            return None

        angles_df = pd.read_csv(raw_angles_path)
        events_df = pd.read_csv(events_path)

        print(f" Loaded data: {len(angles_df)} frames, {len(events_df)} events")


        # Enhancement: Recalculate missing angles from keypoints if requested
        if config.enhance_angles:
            angles_df = enhance_angles_from_keypoints(subject_dir, angles_df, config=config)

    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    # Detect columns
    try:
        col_map = detect_columns(angles_df)
        print(f" Detected columns: {col_map}")
    except Exception as e:
        print(f"Error detecting columns: {e}")
        return None

    # Normalize timebase
    try:
        angles_df, events_df = normalize_timebase(angles_df, events_df, config.fps)
        print(f" Normalized timebase")
    except Exception as e:
        print(f"Error normalizing timebase: {e}")
        return None

    # AUTO-CALIBRATION: Adapt parameters based on data characteristics
    try:
        config = auto_calibrate_config(angles_df, events_df, col_map, config, min_cycles=5)
    except Exception as e:
        print(f"Warning: Auto-calibration failed, using default parameters: {e}")

    # Preprocess angles
    try:
        angle_cols = [col for leg in col_map.values() for col in leg.values()]
        angles_df = preprocess_angles(angles_df, angle_cols,
                                      config.smooth_window,
                                      config.smooth_poly,
                                      config.smooth_threshold)
        print(f" Preprocessed angles")
    except Exception as e:
        print(f"Error preprocessing: {e}")
        return None

    # Segment cycles (HS→HS for phase-true cyclograms)
    try:
        # Load keypoints for pelvis stability check (if available)
        keypoints_df = None
        keypoints_path = subject_dir / "Clean_keypoints.csv"
        if keypoints_path.exists():
            try:
                keypoints_df = pd.read_csv(keypoints_path)
            except:
                pass  # Use None if loading fails

        left_windows, left_rejections = build_cycle_windows(
            events_df, "L",
            angles_df=angles_df,
            keypoints_df=keypoints_df,
            config=config,
            min_duration=config.min_stride_duration,
            max_duration=config.max_stride_duration,
            apply_quality_gates=True
        )
        right_windows, right_rejections = build_cycle_windows(
            events_df, "R",
            angles_df=angles_df,
            keypoints_df=keypoints_df,
            config=config,
            min_duration=config.min_stride_duration,
            max_duration=config.max_stride_duration,
            apply_quality_gates=True
        )

        print(f" · Segmented cycles (HS→HS): {len(left_windows)} left, {len(right_windows)} right")

        # ADAPTIVE OUTLIER REMOVAL
        if len(left_windows) >= 5:
            left_windows = detect_and_remove_outliers(left_windows, "L", method="iqr", iqr_factor=1.5)
        if len(right_windows) >= 5:
            right_windows = detect_and_remove_outliers(right_windows, "R", method="iqr", iqr_factor=1.5)

        if len(left_windows) < 2 or len(right_windows) < 2:
            print(f"Warning: Insufficient cycles (need at least 2 per leg)")
            return None

    except Exception as e:
        print(f"Error segmenting strides: {e}")
        return None

    # Extract cyclograms
    try:
        all_loops_L = []
        all_loops_R = []

        for pair in config.joint_pairs:
            for window in left_windows:
                try:
                    loop = extract_cyclogram(angles_df, window, pair, col_map,
                                           config.n_resample_points)
                    all_loops_L.append(loop)
                except Exception as e:
                    print(f"Warning: Failed to extract L cyclogram for stride {window.stride_id}: {e}")

            for window in right_windows:
                try:
                    loop = extract_cyclogram(angles_df, window, pair, col_map,
                                           config.n_resample_points)
                    all_loops_R.append(loop)
                except Exception as e:
                    print(f"Warning: Failed to extract R cyclogram for stride {window.stride_id}: {e}")

        print(f" Extracted cyclograms: {len(all_loops_L)} left, {len(all_loops_R)} right")

        # Calculate data quality for adaptive quality gates
        data_quality = calculate_data_quality(angles_df, col_map)
        print(f" Data quality score: {data_quality:.2f} (0.0=poor, 1.0=perfect)")

        # Apply quality gates
        all_loops_L = [l for l in all_loops_L if loop_quality_ok(l, data_quality=data_quality)]
        all_loops_R = [r for r in all_loops_R if loop_quality_ok(r, data_quality=data_quality)]
        print(f"· Extracted cyclograms (after QC): {len(all_loops_L)} left, {len(all_loops_R)} right")

    except Exception as e:
        print(f"Error extracting cyclograms: {e}")
        return None

    # Pair strides
    try:
        paired_strides, pairing_counts = pair_strides(
            left_windows, right_windows,
            config=config,
            tolerance_ratio=config.pairing_tolerance
        )
        print(f" Paired cycles: {len(paired_strides)} pairs")

        # Create Pipeline Report
        pipeline_report = PipelineReport(
            events_detected_L=len([e for e in events_df.to_dict('records') if e['side'] == 'L']),
            events_detected_R=len([e for e in events_df.to_dict('records') if e['side'] == 'R']),
            cycles_extracted_L=left_rejections['total_extracted'],
            cycles_extracted_R=right_rejections['total_extracted'],
            cycles_qc_passed_L=left_rejections['passed_qc'],
            cycles_qc_passed_R=right_rejections['passed_qc'],
            cycles_paired=len(paired_strides),
            cycles_rejected_coverage_L=left_rejections['rejected_coverage'],
            cycles_rejected_coverage_R=right_rejections['rejected_coverage'],
            cycles_rejected_gaps_L=left_rejections['rejected_gaps'],
            cycles_rejected_gaps_R=right_rejections['rejected_gaps'],
            cycles_rejected_stability_L=left_rejections['rejected_stability'],
            cycles_rejected_stability_R=right_rejections['rejected_stability'],
            cycles_rejected_sanity_L=left_rejections['rejected_sanity'],
            cycles_rejected_sanity_R=right_rejections['rejected_sanity'],
            cycles_unpaired_L=pairing_counts['unpaired_L'],
            cycles_unpaired_R=pairing_counts['unpaired_R']
        )

        # Print pipeline summary
        pipeline_report.print_summary()

    except Exception as e:
        print(f"Error pairing strides: {e}")
        return None

    # Calculate metrics
    try:
        all_metrics = []

        for pair_obj in paired_strides:
            for joint_pair in config.joint_pairs:
                # Find corresponding loops
                loop_L_list = [l for l in all_loops_L
                              if l.stride_id == pair_obj.left_stride.stride_id
                              and l.joint_pair == joint_pair]
                loop_R_list = [l for l in all_loops_R
                              if l.stride_id == pair_obj.right_stride.stride_id
                              and l.joint_pair == joint_pair]

                if not loop_L_list or not loop_R_list:
                    continue

                loop_L = loop_L_list[0]
                loop_R = loop_R_list[0]

                # Compute metrics
                metrics = compute_all_metrics(loop_L, loop_R, config.similarity_weights)
                all_metrics.append(metrics)

        print(f" Calculated metrics: {len(all_metrics)} comparisons")

    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None

    # Visualize
    try:
        for joint_pair in config.joint_pairs:
            pair_name = f"{joint_pair[0]}-{joint_pair[1]}"

            pair_loops_L = [l for l in all_loops_L if l.joint_pair == joint_pair]
            pair_loops_R = [l for l in all_loops_R if l.joint_pair == joint_pair]
            pair_metrics = [m for m in all_metrics if m.joint_pair == joint_pair]

            # Overlay plot
            output_path = subject_output / f"CK_{pair_name.replace('-', '_')}_AllStrides.png"
            # Create PlotConfig with DPI from config
            plot_cfg = PlotConfig(dpi=config.plot_dpi)
            plot_overlayed_cyclograms(pair_loops_L, pair_loops_R, pair_metrics,
                                     joint_pair, str(output_path), plot_cfg)

        # Summary dashboard
        if all_metrics:
            summary_path = subject_output / "LR_Similarity_Summary.png"
            summary_df = write_session_summary(all_metrics,
                                              str(subject_output / "cyclogram_session_summary.csv"))
            plot_cfg = PlotConfig(dpi=config.plot_dpi)
            plot_similarity_summary(summary_df, str(summary_path), plot_cfg)

        print(f" Generated visualizations")

    except Exception as e:
        print(f"Error generating visualizations: {e}")

    # Export metrics
    try:
        stride_metrics_df = write_stride_metrics(all_metrics,
                                                str(subject_output / "cyclogram_stride_metrics.csv"))
        session_summary_df = write_session_summary(all_metrics,
                                                   str(subject_output / "cyclogram_session_summary.csv"))

        print(f" Exported metrics")

    except Exception as e:
        print(f"Error exporting metrics: {e}")
        return None

    # Parse subject info from directory name
    # Expected format: Openpose_NAME_ID_DATE_TRIAL
    parts = subject_name.split('_')
    if len(parts) >= 4:
        subject_info = {
            'subject_name': subject_name,
            'patient_name': parts[1],
            'patient_id': parts[2],
            'date': parts[3],
            'trial': parts[4] if len(parts) > 4 else 'unknown'
        }
    else:
        subject_info = {
            'subject_name': subject_name,
            'patient_name': 'unknown',
            'patient_id': 'unknown',
            'date': 'unknown',
            'trial': 'unknown'
        }

    # Add summary statistics
    if not session_summary_df.empty:
        for _, row in session_summary_df.iterrows():
            subject_info[f"{row['joint_pair']}_similarity"] = row['similarity_mean']
            subject_info[f"{row['joint_pair']}_n_pairs"] = row['n_pairs']

    subject_info['n_strides_L'] = len(left_windows)
    subject_info['n_strides_R'] = len(right_windows)
    subject_info['n_pairs'] = len(paired_strides)

    print(f" Analysis complete for {subject_name}")

    return subject_info


def run_batch_analysis(input_dir: str, output_dir: str, config: AnalysisConfig) -> None:
    """
    Run analysis on all subjects in input directory.

    Creates:
    - Individual subject output folders with plots and CSVs
    - main_output.csv with aggregate results from all subjects
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*80}")
    print(f"# CYCLOGRAM BATCH ANALYSIS")
    print(f"{'#'*80}")
    print(f"\nInput Directory: {input_path}")
    print(f"Output Directory: {output_path}")

    # Find all subject directories (folders starting with "Openpose_")
    subject_dirs = [d for d in input_path.iterdir()
                   if d.is_dir() and d.name.startswith("Openpose_")]

    if not subject_dirs:
        print(f"\nError: No subject directories found in {input_path}")
        print(f"Expected directories starting with 'Openpose_'")
        return

    print(f"\nFound {len(subject_dirs)} subject(s) to process")

    # Process each subject
    all_results = []

    for i, subject_dir in enumerate(subject_dirs, 1):
        print(f"\n[{i}/{len(subject_dirs)}] Processing: {subject_dir.name}")

        try:
            result = analyze_subject(subject_dir, output_path, config)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"Error processing {subject_dir.name}: {e}")
            continue

    # Write aggregate results
    if all_results:
        main_output = pd.DataFrame(all_results)
        main_output_path = output_path / "main_output.csv"
        main_output.to_csv(main_output_path, index=False)

        print(f"\n{'='*80}")
        print(f"BATCH ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"\nProcessed: {len(all_results)}/{len(subject_dirs)} subjects")
        print(f"Aggregate results saved to: {main_output_path}")
        print(f"\nOutput structure:")
        print(f"  {output_path}/")
        print(f"   main_output.csv (aggregate results)")
        print(f"   Openpose_SUBJECT_1/")
        print(f"      CK_hip_knee_AllStrides.png")
        print(f"      CK_knee_ankle_AllStrides.png")
        print(f"      CK_hip_ankle_AllStrides.png")
        print(f"      LR_Similarity_Summary.png")
        print(f"      cyclogram_stride_metrics.csv")
        print(f"      cyclogram_session_summary.csv")
        print(f"   Openpose_SUBJECT_2/")
        print(f"   ...")
    else:
        print(f"\nWarning: No subjects processed successfully")


# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
# MAIN ENTRY POINT
# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Cyclogram Analysis System - Gait analysis with left-right comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run analysis on all subjects
  python3 Analysis.py

  # Run on specific subject with angle enhancement
  python3 Analysis.py --subject-name "Openpose_subject_001" --enhance-angles

  # Run with custom parameters
  python3 Analysis.py --enhance-angles --smooth-window 15
        """
    )

    parser.add_argument('--subject-name', type=str, default=None,
                       help='Specific subject to analyze (default: all subjects)')
    parser.add_argument('--enhance-angles', action='store_true',
                       help='Enable keypoint-based angle recalculation for missing data')
    parser.add_argument('--smooth-window', type=int, default=11,
                       help='Savitzky-Golay filter window size (default: 11)')
    parser.add_argument('--smooth-threshold', type=float, default=5.0,
                       help='Smoothing threshold in degrees (default: 5.0)')
    parser.add_argument('--input-dir', type=str, default=INPUT_DIR,
                       help=f'Input directory (default: {INPUT_DIR})')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                       help=f'Output directory (default: {OUTPUT_DIR})')

    args = parser.parse_args()

    # Create configuration
    config = AnalysisConfig(
        fps=None,  # Auto-detect from timestamp column
        smooth_window=args.smooth_window,
        smooth_poly=2,
        smooth_threshold=args.smooth_threshold,
        min_stride_duration=0.3,
        max_stride_duration=3.0,
        pairing_tolerance=0.15,
        n_resample_points=101,
        joint_pairs=[
            ("hip", "knee"),
            ("knee", "ankle"),
            ("hip", "ankle")
        ],
        similarity_weights={
            "area": 0.30,
            "procrustes": 0.30,
            "rmse": 0.30,
            "orientation": 0.10
        },
        plot_dpi=300,
        generate_per_leg_plots=False,
        outlier_percentile=1.0,
        enhance_angles=args.enhance_angles
    )

    # Run batch analysis
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    
    if args.subject_name:
        # Analyze single subject
        subject_dir = input_path / args.subject_name
        if not subject_dir.exists():
            print(f"Error: Subject directory not found: {subject_dir}")
        else:
            print(f"Analyzing single subject: {args.subject_name}")
            analyze_subject(subject_dir, output_path, config)
    else:
        # Run batch analysis
        run_batch_analysis(input_path, output_path, config)

