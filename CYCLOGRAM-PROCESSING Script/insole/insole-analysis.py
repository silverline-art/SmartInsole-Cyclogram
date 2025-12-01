#!/usr/bin/env python3
"""
Smart Insole Gait Analysis Module - Complete Merged Version

Anatomically grounded gait phase detection and cyclogram generation from
synchronized pressure sensor and IMU data with comprehensive enhancements.

Architecture:
- Data_handling: Load, calibrate, and filter with FFT adaptive filtering
- Gait_event_detection: Heel-strike and toe-off detection
- Gait_sub_event_detection: Dynamic 8-phase segmentation
- Cyclogram_preparation: 2D/3D cyclogram generation with metrics
- Morphological_mean_cyclogram: Phase-aligned median loop computation
- Plot_Preparation: Dual PNG+JSON output with complete metadata

Author: Gait Analysis Pipeline (Merged)
Date: 2025-10-20
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from scipy.signal import butter, filtfilt, find_peaks
from scipy.interpolate import interp1d
from scipy.spatial import procrustes
import json
import argparse
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Check for optional fastdtw
try:
    from fastdtw import fastdtw
    HAS_FASTDTW = True
except ImportError:
    HAS_FASTDTW = False


# ============================================================================
# GLOBAL BATCH PROCESSING CONFIGURATION
# ============================================================================

INPUT_DIR = Path("/home/shivam/Desktop/Human_Pose/PROJECT CYCLOGRAM/insole-sample/Temp-csv 1")
OUTPUT_DIR = Path("/home/shivam/Desktop/Human_Pose/PROJECT CYCLOGRAM/insole-output/RDW")

# Create directories if they don't exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Find all CSV files for batch processing
CSV_FILES = list(INPUT_DIR.glob("*.csv"))


# ============================================================================
# CONFIGURATION AND DATA STRUCTURES
# ============================================================================

@dataclass
class InsoleConfig:
    """Configuration for insole gait analysis."""

    # Signal processing
    sampling_rate: float = 100.0  # Hz
    filter_cutoff: float = 20.0    # Hz (lowpass for gait dynamics)
    filter_order: int = 4

    # Gait cycle constraints (seconds)
    min_cycle_duration: float = 0.8   # Minimum gait cycle duration
    max_cycle_duration: float = 3.0   # Maximum gait cycle duration (increased for elderly/pathological)
    min_stance_duration: float = 0.5  # Minimum stance phase duration
    min_swing_duration: float = 0.3   # Minimum swing phase duration

    # Phase detection thresholds
    pressure_threshold: float = 0.5    # Normalized pressure for contact detection
    pressure_derivative_threshold: float = 0.1  # For IC detection
    gyro_swing_threshold: float = 50.0  # deg/s for swing detection

    # Validation thresholds (relaxed for real-world data including elderly/pathological gait)
    stance_swing_ratio_min: float = 0.8  # Relaxed from 1.2 (allows for asymmetric/pathological gait)
    stance_swing_ratio_max: float = 3.0  # Relaxed from 2.0 (allows for slower gait patterns)
    bilateral_tolerance: float = 0.20    # 20% duration difference allowed (increased from 15%)

    # Visualization
    plot_dpi: int = 300
    cyclogram_resolution: int = 101  # Points per gait cycle


@dataclass
class GaitPhase:
    """Single gait phase annotation with support type classification."""
    phase_name: str          # IC, LR, MSt, TSt, PSw, ISw, MSw, TSw
    leg: str                 # 'left' or 'right'
    start_time: float        # seconds
    end_time: float          # seconds
    start_idx: int           # frame index
    end_idx: int             # frame index
    duration: float          # seconds
    phase_number: int        # 1-8
    support_type: str = None # 'double_support', 'single_support', 'swing' (added for biomechanical accuracy)

    @staticmethod
    def classify_support_type(phase_number: int) -> str:
        """
        Classify gait phase by support type for biomechanical accuracy.

        Support type classification based on Perry's gait model:
        - Double Support: Both feet on ground (IC, LR, PSw) ~20% of cycle
        - Single Support: Only one foot on ground (MSt, TSt) ~40% of cycle
        - Swing: Foot off ground (ISw, MSw, TSw) ~40% of cycle

        Args:
            phase_number: Phase number 1-8

        Returns:
            'double_support', 'single_support', or 'swing'
        """
        if phase_number in [1, 2, 5]:  # IC, LR, PSw
            return 'double_support'
        elif phase_number in [3, 4]:  # MSt, TSt
            return 'single_support'
        else:  # 6, 7, 8: ISw, MSw, TSw
            return 'swing'


@dataclass
class GaitCycle:
    """Complete gait cycle with all phases."""
    leg: str
    cycle_id: int
    start_time: float
    end_time: float
    duration: float
    phases: List[GaitPhase]
    stance_duration: float
    swing_duration: float
    stance_swing_ratio: float


@dataclass
class CyclogramData:
    """IMU-based cyclogram for one gait cycle."""
    cycle: GaitCycle
    x_signal: np.ndarray      # e.g., ACC_X
    y_signal: np.ndarray      # e.g., ACC_Y
    z_signal: Optional[np.ndarray] = None  # For 3D cyclograms
    x_label: str = "ACC_X"
    y_label: str = "ACC_Y"
    z_label: Optional[str] = None
    phase_indices: List[int] = field(default_factory=list)  # Phase boundary indices
    phase_labels: List[str] = field(default_factory=list)   # Phase names for coloring
    is_3d: bool = False


class AdvancedCyclogramMetrics:
    """
    Comprehensive cyclogram analysis metrics for publication-quality gait research.

    Implements:
    - Level 1: Geometric/Morphology metrics
    - Level 2: Temporal-Coupling parameters
    - Level 3: Symmetry/Bilateral coordination

    Reference: ADVANCED_CYCLOGRAM_METRICS_PLAN.md
    """

    def __init__(self, cyclogram: CyclogramData):
        """
        Initialize with cyclogram data.

        Args:
            cyclogram: CyclogramData instance with x_signal, y_signal, z_signal (optional)
        """
        self.cyclogram = cyclogram
        self.x = cyclogram.x_signal
        self.y = cyclogram.y_signal
        self.z = cyclogram.z_signal if cyclogram.is_3d else None

    # ========== LEVEL 1: GEOMETRIC/MORPHOLOGY METRICS ==========

    def compute_geometric_metrics(self) -> Dict:
        """
        Compute Level 1 geometric/morphology metrics.

        Returns:
            Dictionary containing:
            - compactness_ratio: Loop circularity (1.0 = perfect circle)
            - aspect_ratio: Major/minor axis ratio
            - eccentricity: Shape elongation (0 = circle, 1 = line)
            - orientation_angle: Phase relationship angle (degrees)
            - mean_curvature: Movement control fineness
            - curvature_std: Curvature variability
            - trajectory_smoothness: Coordination smoothness (0-1)
        """
        metrics = {}

        # Compute basic geometric properties
        area = self._compute_area()
        perimeter = self._compute_perimeter()

        # 1. Compactness Ratio (4πA/P²)
        metrics['compactness_ratio'] = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

        # 2. Fit ellipse for aspect ratio and eccentricity
        ellipse_params = self._fit_ellipse()
        if ellipse_params:
            a, b, theta = ellipse_params  # major, minor, orientation
            metrics['aspect_ratio'] = a / b if b > 0 else np.inf
            metrics['eccentricity'] = np.sqrt(1 - (b**2 / a**2)) if a > 0 else 0
            metrics['orientation_angle'] = np.degrees(theta)
        else:
            metrics['aspect_ratio'] = np.nan
            metrics['eccentricity'] = np.nan
            metrics['orientation_angle'] = np.nan

        # 3. Mean Curvature and Trajectory Smoothness
        curvature = self._compute_curvature()
        metrics['mean_curvature'] = np.mean(np.abs(curvature))
        metrics['curvature_std'] = np.std(curvature)
        metrics['trajectory_smoothness'] = 1 / (1 + metrics['curvature_std'])

        return metrics

    def _compute_area(self) -> float:
        """Compute signed area using shoelace formula."""
        if len(self.x) < 3:
            return 0.0
        # Shoelace formula
        area = 0.5 * np.abs(np.dot(self.x, np.roll(self.y, -1)) - np.dot(self.y, np.roll(self.x, -1)))
        return area

    def _compute_perimeter(self) -> float:
        """Compute perimeter as sum of segment lengths."""
        if len(self.x) < 2:
            return 0.0
        dx = np.diff(self.x)
        dy = np.diff(self.y)
        perimeter = np.sum(np.sqrt(dx**2 + dy**2))
        return perimeter

    def _fit_ellipse(self) -> Optional[Tuple[float, float, float]]:
        """
        Fit ellipse using PCA.

        Returns:
            Tuple of (major_axis, minor_axis, orientation_angle) or None if fitting fails
        """
        try:
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
        except:
            return None

    def _compute_curvature(self) -> np.ndarray:
        """
        Compute trajectory curvature κ = (x'y'' - y'x'') / (x'² + y'²)^(3/2).

        Returns:
            Array of curvature values
        """
        # First derivatives
        dx = np.gradient(self.x)
        dy = np.gradient(self.y)

        # Second derivatives
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        # Curvature formula
        numerator = dx * ddy - dy * ddx
        denominator = (dx**2 + dy**2)**(3/2)

        # Avoid division by zero
        curvature = np.where(denominator > 1e-10, numerator / denominator, 0)

        return curvature

    # ========== LEVEL 2: TEMPORAL-COUPLING METRICS ==========

    def compute_temporal_metrics(self) -> Dict:
        """
        Compute Level 2 temporal-coupling metrics.

        Returns:
            Dictionary containing:
            - continuous_relative_phase: Instantaneous phase difference array
            - mean_relative_phase: Average phase difference
            - marp: Mean absolute relative phase
            - coupling_angle_variability: Coordination stability
            - deviation_phase: Phase difference standard deviation
            - phase_shift: Mean timing lag
        """
        metrics = {}

        # 1. Phase angles using Hilbert transform
        phase_x = self._compute_phase_angle(self.x)
        phase_y = self._compute_phase_angle(self.y)

        # 2. Continuous Relative Phase (CRP)
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
        """
        Compute phase angle using Hilbert transform.

        Args:
            signal: 1D time series data

        Returns:
            Phase angle array
        """
        from scipy.signal import hilbert
        analytic_signal = hilbert(signal - np.mean(signal))
        phase = np.angle(analytic_signal)
        return phase

    # ========== LEVEL 3: SYMMETRY/BILATERAL COORDINATION ==========

    @staticmethod
    def compute_bilateral_symmetry(left_cyclogram: 'CyclogramData',
                                   right_cyclogram: 'CyclogramData') -> Dict:
        """
        Compute Level 3 bilateral symmetry metrics between left and right cyclograms.

        Args:
            left_cyclogram: Left leg cyclogram
            right_cyclogram: Right leg cyclogram

        Returns:
            Dictionary containing:
            - symmetry_index: Inter-limb asymmetry percentage
            - mirror_correlation: Bilateral similarity
            - rms_trajectory_diff: Functional asymmetry
            - rms_trajectory_diff_x: X-axis trajectory difference
            - rms_trajectory_diff_y: Y-axis trajectory difference
        """
        metrics = {}

        # Compute areas
        left_metrics_obj = AdvancedCyclogramMetrics(left_cyclogram)
        right_metrics_obj = AdvancedCyclogramMetrics(right_cyclogram)

        area_left = left_metrics_obj._compute_area()
        area_right = right_metrics_obj._compute_area()

        # 1. Symmetry Index (SI)
        if (area_left + area_right) > 0:
            metrics['symmetry_index'] = ((area_left - area_right) /
                                        ((area_left + area_right) / 2) * 100)
        else:
            metrics['symmetry_index'] = 0.0

        # 2. Mirror Correlation and RMS Trajectory Difference
        # Resample both to same length for comparison
        target_len = 101
        left_x_resampled = AdvancedCyclogramMetrics._resample_signal(left_cyclogram.x_signal, target_len)
        left_y_resampled = AdvancedCyclogramMetrics._resample_signal(left_cyclogram.y_signal, target_len)
        right_x_resampled = AdvancedCyclogramMetrics._resample_signal(right_cyclogram.x_signal, target_len)
        right_y_resampled = AdvancedCyclogramMetrics._resample_signal(right_cyclogram.y_signal, target_len)

        # Mirror right cyclogram horizontally for correlation
        right_x_mirrored = -right_x_resampled

        # Compute mirror correlation
        if len(left_x_resampled) > 1 and len(right_x_mirrored) > 1:
            try:
                corr_matrix = np.corrcoef(left_x_resampled, right_x_mirrored)
                metrics['mirror_correlation'] = corr_matrix[0, 1]
            except:
                metrics['mirror_correlation'] = np.nan
        else:
            metrics['mirror_correlation'] = np.nan

        # 3. RMS Trajectory Difference
        rms_x = np.sqrt(np.mean((left_x_resampled - right_x_resampled)**2))
        rms_y = np.sqrt(np.mean((left_y_resampled - right_y_resampled)**2))
        metrics['rms_trajectory_diff_x'] = rms_x
        metrics['rms_trajectory_diff_y'] = rms_y
        metrics['rms_trajectory_diff'] = np.sqrt(rms_x**2 + rms_y**2)

        return metrics

    @staticmethod
    def _resample_signal(signal: np.ndarray, target_length: int) -> np.ndarray:
        """
        Resample signal to target length using linear interpolation.

        Args:
            signal: Input signal array
            target_length: Desired output length

        Returns:
            Resampled signal
        """
        if len(signal) < 2:
            return np.full(target_length, signal[0] if len(signal) == 1 else 0.0)

        old_indices = np.linspace(0, 1, len(signal))
        new_indices = np.linspace(0, 1, target_length)
        resampled = np.interp(new_indices, old_indices, signal)
        return resampled


@dataclass
class GaitEventRecord:
    """High-precision gait event with sensor zone information."""
    event_type: str          # 'heel_strike', 'mid_stance', 'toe_off'
    leg: str                 # 'left' or 'right'
    event_start: float       # timestamp in seconds
    event_end: float         # timestamp in seconds
    duration: float          # milliseconds
    sensor_source: str       # e.g., 'L_value4', 'R_value2'
    frame_start: int         # frame index
    frame_end: int           # frame index
    confidence: float        # 0.0-1.0


@dataclass
class CalibrationParameters:
    """Self-calibration parameters for adaptive signal processing."""
    # Pressure sensor calibration
    pressure_baseline_left: Dict[str, float] = field(default_factory=dict)
    pressure_baseline_right: Dict[str, float] = field(default_factory=dict)
    pressure_scale_left: Dict[str, float] = field(default_factory=dict)
    pressure_scale_right: Dict[str, float] = field(default_factory=dict)

    # IMU sensor calibration
    acc_mean_left: Dict[str, float] = field(default_factory=dict)
    acc_mean_right: Dict[str, float] = field(default_factory=dict)
    acc_std_left: Dict[str, float] = field(default_factory=dict)
    acc_std_right: Dict[str, float] = field(default_factory=dict)
    gyro_mean_left: Dict[str, float] = field(default_factory=dict)
    gyro_mean_right: Dict[str, float] = field(default_factory=dict)
    gyro_std_left: Dict[str, float] = field(default_factory=dict)
    gyro_std_right: Dict[str, float] = field(default_factory=dict)

    # Adaptive filter parameters
    dominant_frequency_left: float = 1.0
    dominant_frequency_right: float = 1.0
    adaptive_cutoff_left: float = 20.0
    adaptive_cutoff_right: float = 20.0


@dataclass
class ValidationMetrics:
    """Anatomical plausibility validation results."""
    left_right_alternation: bool
    stance_swing_ratio_valid: bool
    bilateral_symmetry_valid: bool
    phase_sequence_valid: bool
    duration_constraints_valid: bool
    overall_valid: bool
    issues: List[str] = field(default_factory=list)


@dataclass
class MorphologicalMeanCyclogram:
    """
    Morphological Mean Cyclogram (MMC) - phase-aligned median trajectory.

    Replaces naive mean averaging with robust median computation after:
    1. DTW-based median reference selection
    2. Centroid centering (position-free comparison)
    3. Area normalization (scale-free comparison)
    4. Median shape computation (robust to outliers)
    """
    leg: str
    sensor_pair: Tuple[str, str]
    median_trajectory: np.ndarray  # (101, 2) or (101, 3) for 3D
    variance_envelope_lower: np.ndarray
    variance_envelope_upper: np.ndarray
    shape_dispersion_index: float
    confidence_ellipse_params: Dict[str, float]
    n_loops: int
    alignment_quality: float
    median_area: float

    @property
    def median_x(self) -> np.ndarray:
        """X-axis signal (101 points)."""
        return self.median_trajectory[:, 0]

    @property
    def median_y(self) -> np.ndarray:
        """Y-axis signal (101 points)."""
        return self.median_trajectory[:, 1]

    @property
    def median_z(self) -> Optional[np.ndarray]:
        """Z-axis signal (101 points) for 3D cyclograms."""
        if self.median_trajectory.shape[1] > 2:
            return self.median_trajectory[:, 2]
        return None


# ============================================================================
# DATA HANDLING CLASS
# ============================================================================

class Data_handling:
    """
    Data loading, calibration, and adaptive filtering.

    Consolidates:
    - InsoleLoader: CSV loading
    - SignalProcessor: Filtering and feature extraction
    - CalibrationParameters: Auto-calibration with FFT adaptive filtering
    """

    def __init__(self, config: InsoleConfig):
        self.config = config
        self.calibration = CalibrationParameters()
        self.is_calibrated = False
        self._design_filter()

    def _design_filter(self, cutoff_freq: Optional[float] = None):
        """Design Butterworth lowpass filter."""
        nyquist = self.config.sampling_rate / 2.0
        cutoff = cutoff_freq if cutoff_freq is not None else self.config.filter_cutoff
        cutoff_normalized = cutoff / nyquist
        self.b, self.a = butter(self.config.filter_order, cutoff_normalized, btype='low')

    def load_data(self, csv_path: Path) -> pd.DataFrame:
        """
        Load insole CSV data with automatic header detection.

        Supports multiple CSV formats:
        - Version 22: 2 header rows (version + column names)
        - Version 23: 3 header rows (version + extra row + column names)
        - Version 207: 4 header rows (version + 2 summary rows + column names)

        Expected columns:
        - timestamp: Time in milliseconds
        - L_value1-4, R_value1-4: Pressure sensors
        - L_ACC_X/Y/Z, R_ACC_X/Y/Z: Accelerometers
        - L_GYRO_X/Y/Z, R_GYRO_X/Y/Z: Gyroscopes
        """
        print(f"Loading insole data from {csv_path}")

        # Dynamic header detection - search for the row containing sensor columns
        # This approach is more robust than hardcoded version numbers
        # CRITICAL: Must require ALL mandatory columns to avoid v207 summary rows
        header_row = None
        max_search_lines = 20  # Search first 20 lines for header

        with open(csv_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line_num >= max_search_lines:
                    break

                # Check if this line contains ALL key sensor column identifiers
                # Version 207 has summary rows before the actual header with GCT/Stance/Swing metrics
                # We need timestamp + pressure sensors + accelerometers to confirm it's the real header
                line_lower = line.lower()
                has_timestamp = 'timestamp' in line_lower
                has_pressure = ('l_value1' in line_lower or 'r_value1' in line_lower)
                has_acc = ('l_acc_x' in line_lower or 'r_acc_x' in line_lower)

                # Skip rows containing only gait metrics (v207 summary rows)
                is_summary_row = ('gct' in line_lower or 'stance' in line_lower or 'swing' in line_lower) and not has_timestamp

                if has_timestamp and has_pressure and has_acc and not is_summary_row:
                    header_row = line_num
                    print(f"  Auto-detected header row at line {line_num + 1}")
                    break

        if header_row is None:
            # Fallback to version-based detection if auto-detection fails
            print(f"  Warning: Could not auto-detect header row, falling back to version detection")
            with open(csv_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                version_line = f.readline().strip()

            if version_line.isdigit():
                version = int(version_line)
                if version == 207:
                    header_row = 4
                elif version == 22:
                    header_row = 2
                elif version == 23:
                    header_row = 3
                else:
                    header_row = 3
                print(f"  Detected file version: {version}, using header row {header_row + 1}")
            else:
                header_row = 3
                print(f"  Using default header row {header_row + 1}")

        # Load CSV with detected header row
        df = pd.read_csv(csv_path, skiprows=header_row, encoding='utf-8')

        # Convert timestamp from ms to seconds
        if 'timestamp' in df.columns:
            df['timestamp'] = df['timestamp'] / 1000.0

            print(f"Loaded {len(df)} samples ({df['timestamp'].iloc[-1]:.1f} seconds)")
        else:
            print(f"Loaded {len(df)} samples (no timestamp column found)")

        # Verify required columns (pressure and acc are required, gyro is optional)
        required_cols = self._get_required_columns()
        optional_cols = self._get_optional_columns()

        missing_required = [col for col in required_cols if col not in df.columns]
        missing_optional = [col for col in optional_cols if col not in df.columns]

        if missing_required:
            raise ValueError(f"Missing required columns: {missing_required}")

        # Track if GYRO data is real or placeholder
        self.has_real_gyro = True

        if missing_optional:
            print(f"  Note: Optional GYRO columns not found (analysis will use ACC data only)")
            # Add placeholder columns filled with zeros
            for col in missing_optional:
                df[col] = 0.0
            self.has_real_gyro = False
        else:
            # Check if GYRO columns exist but are all zeros (placeholder data)
            gyro_data = df[optional_cols].abs().sum().sum()
            if gyro_data == 0:
                print(f"  Note: GYRO columns found but contain no data (all zeros)")
                self.has_real_gyro = False

        return df

    def _get_required_columns(self) -> List[str]:
        """Define required columns for analysis (pressure + accelerometer)."""
        pressure_cols = [f'L_value{i}' for i in range(1, 5)] + \
                       [f'R_value{i}' for i in range(1, 5)]

        acc_cols = [f'{side}_ACC_{axis}'
                   for side in ['L', 'R']
                   for axis in ['X', 'Y', 'Z']]

        return ['timestamp'] + pressure_cols + acc_cols

    def _get_optional_columns(self) -> List[str]:
        """Define optional columns (gyroscope)."""
        gyro_cols = [f'{side}_GYRO_{axis}'
                    for side in ['L', 'R']
                    for axis in ['X', 'Y', 'Z']]

        return gyro_cols

    def detect_column_mapping(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Create column mapping for left/right sensors."""
        mapping = {
            'left_pressure': [f'L_value{i}' for i in range(1, 5)],
            'right_pressure': [f'R_value{i}' for i in range(1, 5)],
            'left_acc': ['L_ACC_X', 'L_ACC_Y', 'L_ACC_Z'],
            'right_acc': ['R_ACC_X', 'R_ACC_Y', 'R_ACC_Z'],
            'left_gyro': ['L_GYRO_X', 'L_GYRO_Y', 'L_GYRO_Z'],
            'right_gyro': ['R_GYRO_X', 'R_GYRO_Y', 'R_GYRO_Z']
        }
        return mapping

    def calibrate_signals(self, df: pd.DataFrame, col_map: Dict) -> pd.DataFrame:
        """
        Self-calibrating baseline correction and normalization.

        Pressure Sensors:
        - Baseline offset: 2nd percentile (rest-state bias)
        - Scale factor: 98th percentile - baseline
        - Normalize to [0-1] range

        IMU Sensors:
        - Bias removal: subtract mean
        - Standardization: divide by std
        """
        print("  Performing self-calibration...")

        result = df.copy()

        # Calibrate pressure sensors
        for side in ['left', 'right']:
            side_key = f'{side}_pressure'
            pressure_cols = col_map[side_key]

            for col in pressure_cols:
                signal = df[col].values

                # Estimate baseline (2nd percentile for rest-state)
                baseline = np.percentile(signal, 2)

                # Determine scale factor (98th percentile - baseline)
                p98 = np.percentile(signal, 98)
                scale_factor = p98 - baseline + 1e-6

                # Apply zero-offset correction and normalize to [0-1]
                calibrated = (signal - baseline) / scale_factor
                calibrated = np.clip(calibrated, 0, 1)  # Ensure [0-1] range

                result[col] = calibrated

                # Store calibration parameters
                if side == 'left':
                    self.calibration.pressure_baseline_left[col] = baseline
                    self.calibration.pressure_scale_left[col] = scale_factor
                else:
                    self.calibration.pressure_baseline_right[col] = baseline
                    self.calibration.pressure_scale_right[col] = scale_factor

        # Calibrate IMU sensors (ACC + GYRO)
        for side in ['left', 'right']:
            # Accelerometer
            acc_cols = col_map[f'{side}_acc']
            for col in acc_cols:
                signal = df[col].values

                # Compute mean and std for bias removal and standardization
                mean_val = np.mean(signal)
                std_val = np.std(signal) + 1e-6

                # Apply bias removal and standardization
                calibrated = (signal - mean_val) / std_val

                result[col] = calibrated

                # Store parameters
                if side == 'left':
                    self.calibration.acc_mean_left[col] = mean_val
                    self.calibration.acc_std_left[col] = std_val
                else:
                    self.calibration.acc_mean_right[col] = mean_val
                    self.calibration.acc_std_right[col] = std_val

            # Gyroscope
            gyro_cols = col_map[f'{side}_gyro']
            for col in gyro_cols:
                signal = df[col].values

                mean_val = np.mean(signal)
                std_val = np.std(signal) + 1e-6

                calibrated = (signal - mean_val) / std_val

                result[col] = calibrated

                if side == 'left':
                    self.calibration.gyro_mean_left[col] = mean_val
                    self.calibration.gyro_std_left[col] = std_val
                else:
                    self.calibration.gyro_mean_right[col] = mean_val
                    self.calibration.gyro_std_right[col] = std_val

        # Analyze dominant gait frequency and adapt filter
        self._analyze_and_adapt_filter(result, col_map)

        self.is_calibrated = True
        print(f"  Calibration complete (Left freq: {self.calibration.dominant_frequency_left:.2f} Hz, "
              f"Right freq: {self.calibration.dominant_frequency_right:.2f} Hz)")

        return result

    def _analyze_and_adapt_filter(self, df: pd.DataFrame, col_map: Dict):
        """
        Analyze dominant gait frequency using FFT and adapt filter cutoff.

        Args:
            df: Calibrated dataframe
            col_map: Column mapping
        """
        for side in ['left', 'right']:
            # Use total pressure for spectral analysis
            pressure_cols = col_map[f'{side}_pressure']
            total_pressure = df[pressure_cols].sum(axis=1).values

            # Perform FFT
            dominant_freq = self._analyze_signal_spectrum(total_pressure)

            # Adaptive cutoff: 3× dominant gait frequency, bounded [5-25 Hz]
            adaptive_cutoff = np.clip(3.0 * dominant_freq, 5.0, 25.0)

            # Store parameters
            if side == 'left':
                self.calibration.dominant_frequency_left = dominant_freq
                self.calibration.adaptive_cutoff_left = adaptive_cutoff
            else:
                self.calibration.dominant_frequency_right = dominant_freq
                self.calibration.adaptive_cutoff_right = adaptive_cutoff

    def _analyze_signal_spectrum(self, signal: np.ndarray) -> float:
        """
        Analyze signal spectrum to find dominant gait frequency.

        Args:
            signal: Time-domain signal

        Returns:
            Dominant frequency in Hz
        """
        # Perform FFT
        fft_vals = np.fft.rfft(signal)
        fft_freq = np.fft.rfftfreq(len(signal), 1.0 / self.config.sampling_rate)

        # Compute power spectrum
        power = np.abs(fft_vals) ** 2

        # Find dominant frequency in gait range (0.5-3.0 Hz typical)
        gait_mask = (fft_freq >= 0.5) & (fft_freq <= 3.0)
        if np.any(gait_mask):
            gait_power = power[gait_mask]
            gait_freqs = fft_freq[gait_mask]

            # Find peak frequency
            peak_idx = np.argmax(gait_power)
            dominant_freq = gait_freqs[peak_idx]
        else:
            # Fallback to 1 Hz if no clear peak
            dominant_freq = 1.0

        return dominant_freq

    def filter_signal(self, signal: np.ndarray, adaptive: bool = False,
                     leg: Optional[str] = None) -> np.ndarray:
        """
        Apply lowpass filter with optional adaptive mode.

        Args:
            signal: Input signal
            adaptive: Use adaptive cutoff based on calibration
            leg: Leg identifier ('left' or 'right') for adaptive mode

        Returns:
            Filtered signal
        """
        if adaptive and leg and self.is_calibrated:
            # Use adaptive cutoff frequency
            cutoff = (self.calibration.adaptive_cutoff_left if leg == 'left'
                     else self.calibration.adaptive_cutoff_right)

            # Re-design filter with adaptive cutoff
            nyquist = self.config.sampling_rate / 2.0
            cutoff_normalized = cutoff / nyquist
            b, a = butter(self.config.filter_order, cutoff_normalized, btype='low')

            return filtfilt(b, a, signal)
        else:
            # Use standard filter
            return filtfilt(self.b, self.a, signal)

    def compute_pressure_features(self, df: pd.DataFrame, col_map: Dict) -> pd.DataFrame:
        """
        Compute pressure-based features with adaptive filtering.

        Features:
        - total_pressure: Sum of all sensors (adaptively filtered)
        - pressure_derivative: Rate of change for IC/TO detection
        - normalized_pressure: Z-scored for adaptive thresholding
        """
        result = df.copy()

        for side in ['left', 'right']:
            pressure_cols = col_map[f'{side}_pressure']

            # Total pressure
            total = df[pressure_cols].sum(axis=1).values

            # Apply adaptive filtering if calibrated
            if self.is_calibrated:
                total_filtered = self.filter_signal(total, adaptive=True, leg=side)
            else:
                total_filtered = self.filter_signal(total)

            result[f'{side}_pressure_total'] = total_filtered

            # Derivative for IC detection
            derivative = np.gradient(total_filtered)
            result[f'{side}_pressure_deriv'] = derivative

            # Normalized pressure
            normalized = (total_filtered - total_filtered.mean()) / (total_filtered.std() + 1e-6)
            result[f'{side}_pressure_norm'] = normalized

        return result

    def compute_imu_features(self, df: pd.DataFrame, col_map: Dict) -> pd.DataFrame:
        """
        Compute IMU-based features for phase detection with adaptive filtering.

        Features:
        - acc_magnitude: sqrt(x^2 + y^2 + z^2) (adaptively filtered)
        - gyro_magnitude: Angular velocity magnitude (adaptively filtered)
        - acc_filtered: Lowpass filtered accelerations
        - gyro_filtered: Lowpass filtered gyroscope
        """
        result = df.copy()

        for side in ['left', 'right']:
            # Accelerometer magnitude
            acc_cols = col_map[f'{side}_acc']
            acc_x, acc_y, acc_z = [df[col].values for col in acc_cols]

            acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

            # Apply adaptive filtering if calibrated
            if self.is_calibrated:
                result[f'{side}_acc_mag'] = self.filter_signal(acc_mag, adaptive=True, leg=side)
            else:
                result[f'{side}_acc_mag'] = self.filter_signal(acc_mag)

            # Filter individual axes
            for col in acc_cols:
                if self.is_calibrated:
                    result[f'{col}_filt'] = self.filter_signal(df[col].values, adaptive=True, leg=side)
                else:
                    result[f'{col}_filt'] = self.filter_signal(df[col].values)

            # Gyroscope magnitude
            gyro_cols = col_map[f'{side}_gyro']
            gyro_x, gyro_y, gyro_z = [df[col].values for col in gyro_cols]

            gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)

            # Apply adaptive filtering if calibrated
            if self.is_calibrated:
                result[f'{side}_gyro_mag'] = self.filter_signal(gyro_mag, adaptive=True, leg=side)
            else:
                result[f'{side}_gyro_mag'] = self.filter_signal(gyro_mag)

            # Filter individual axes
            for col in gyro_cols:
                if self.is_calibrated:
                    result[f'{col}_filt'] = self.filter_signal(df[col].values, adaptive=True, leg=side)
                else:
                    result[f'{col}_filt'] = self.filter_signal(df[col].values)

        return result


# ============================================================================
# PRESSURE SENSOR ZONE DETECTOR
# ============================================================================

class PressureSensorZoneDetector:
    """
    High-precision gait event detection using anatomical sensor zone mapping.

    Sensor Zone Mapping:
    - Hindfoot: Sensor 4 (heel region)
    - Midfoot: Sensors 1 & 3 (arch region)
    - Forefoot: Sensor 2 (toe region)

    Multi-sensor Logic:
    - Heel Strike: Hindfoot (sensor 4) first activation
    - Mid-Stance: Midfoot (sensors 1 & 3) active, forefoot off
    - Toe-Off: Forefoot (sensor 2) final release
    """

    def __init__(self, config: InsoleConfig):
        self.config = config
        # Calibrated activation threshold (% of max pressure)
        self.activation_threshold = 0.1  # 10% of signal max
        # Minimum duration for valid event (milliseconds)
        self.min_event_duration = 20.0  # 20ms

    def detect_heel_strike_events(self, df: pd.DataFrame, leg: str) -> List[GaitEventRecord]:
        """
        Detect heel strike events from hindfoot sensor (sensor 4).

        Biomechanical signature: Sharp rise in sensor 4 pressure.
        """
        leg_prefix = 'L' if leg == 'left' else 'R'
        sensor_col = f'{leg_prefix}_value4'  # Hindfoot sensor

        pressure = df[sensor_col].values
        timestamps = df['timestamp'].values

        # Adaptive threshold based on signal statistics
        threshold = self.activation_threshold * np.max(pressure)

        events = []
        in_contact = False
        contact_start_idx = 0

        for i in range(1, len(pressure)):
            if not in_contact and pressure[i] > threshold and pressure[i] > pressure[i-1]:
                # Contact initiated
                in_contact = True
                contact_start_idx = i
            elif in_contact and pressure[i] < threshold:
                # Contact released
                contact_end_idx = i
                duration_ms = (timestamps[contact_end_idx] - timestamps[contact_start_idx]) * 1000

                if duration_ms >= self.min_event_duration:
                    # Compute confidence based on pressure magnitude
                    peak_pressure = np.max(pressure[contact_start_idx:contact_end_idx])
                    confidence = min(1.0, peak_pressure / (np.max(pressure) + 1e-6))

                    event = GaitEventRecord(
                        event_type='heel_strike',
                        leg=leg,
                        event_start=timestamps[contact_start_idx],
                        event_end=timestamps[contact_end_idx],
                        duration=duration_ms,
                        sensor_source=sensor_col,
                        frame_start=contact_start_idx,
                        frame_end=contact_end_idx,
                        confidence=confidence
                    )
                    events.append(event)

                in_contact = False

        return events

    def detect_mid_stance_events(self, df: pd.DataFrame, leg: str) -> List[GaitEventRecord]:
        """
        Detect mid-stance events: midfoot (sensors 1 & 3) active, forefoot off.

        Biomechanical signature: Weight transfer to midfoot region.
        """
        leg_prefix = 'L' if leg == 'left' else 'R'
        midfoot_1 = df[f'{leg_prefix}_value1'].values
        midfoot_3 = df[f'{leg_prefix}_value3'].values
        forefoot = df[f'{leg_prefix}_value2'].values
        timestamps = df['timestamp'].values

        # Combined midfoot pressure
        midfoot_total = midfoot_1 + midfoot_3

        # Adaptive thresholds
        midfoot_threshold = self.activation_threshold * np.max(midfoot_total)
        forefoot_threshold = self.activation_threshold * np.max(forefoot) * 0.5  # Lower threshold for "off"

        events = []
        in_midstance = False
        stance_start_idx = 0

        for i in range(1, len(midfoot_total)):
            # Mid-stance condition: midfoot active AND forefoot off
            midfoot_active = midfoot_total[i] > midfoot_threshold
            forefoot_off = forefoot[i] < forefoot_threshold

            if not in_midstance and midfoot_active and forefoot_off:
                # Mid-stance initiated
                in_midstance = True
                stance_start_idx = i
            elif in_midstance and (not midfoot_active or not forefoot_off):
                # Mid-stance ended
                stance_end_idx = i
                duration_ms = (timestamps[stance_end_idx] - timestamps[stance_start_idx]) * 1000

                if duration_ms >= self.min_event_duration:
                    peak_midfoot = np.max(midfoot_total[stance_start_idx:stance_end_idx])
                    confidence = min(1.0, peak_midfoot / (np.max(midfoot_total) + 1e-6))

                    event = GaitEventRecord(
                        event_type='mid_stance',
                        leg=leg,
                        event_start=timestamps[stance_start_idx],
                        event_end=timestamps[stance_end_idx],
                        duration=duration_ms,
                        sensor_source=f'{leg_prefix}_value1,{leg_prefix}_value3',
                        frame_start=stance_start_idx,
                        frame_end=stance_end_idx,
                        confidence=confidence
                    )
                    events.append(event)

                in_midstance = False

        return events

    def detect_toe_off_events(self, df: pd.DataFrame, leg: str) -> List[GaitEventRecord]:
        """
        Detect toe-off events from forefoot sensor (sensor 2).

        Biomechanical signature: Final release of forefoot pressure.
        """
        leg_prefix = 'L' if leg == 'left' else 'R'
        sensor_col = f'{leg_prefix}_value2'  # Forefoot sensor

        pressure = df[sensor_col].values
        timestamps = df['timestamp'].values

        # Adaptive threshold
        threshold = self.activation_threshold * np.max(pressure)

        events = []
        in_contact = False
        contact_start_idx = 0

        for i in range(1, len(pressure)):
            if not in_contact and pressure[i] > threshold:
                # Contact initiated
                in_contact = True
                contact_start_idx = i
            elif in_contact and pressure[i] < threshold and pressure[i] < pressure[i-1]:
                # Contact released (toe-off)
                contact_end_idx = i
                duration_ms = (timestamps[contact_end_idx] - timestamps[contact_start_idx]) * 1000

                if duration_ms >= self.min_event_duration:
                    peak_pressure = np.max(pressure[contact_start_idx:contact_end_idx])
                    confidence = min(1.0, peak_pressure / (np.max(pressure) + 1e-6))

                    event = GaitEventRecord(
                        event_type='toe_off',
                        leg=leg,
                        event_start=timestamps[contact_start_idx],
                        event_end=timestamps[contact_end_idx],
                        duration=duration_ms,
                        sensor_source=sensor_col,
                        frame_start=contact_start_idx,
                        frame_end=contact_end_idx,
                        confidence=confidence
                    )
                    events.append(event)

                in_contact = False

        return events

    def detect_all_events(self, df: pd.DataFrame, leg: str) -> Dict[str, List[GaitEventRecord]]:
        """
        Detect all gait events for one leg.

        Returns:
            Dictionary with keys: 'heel_strike', 'mid_stance', 'toe_off'
        """
        return {
            'heel_strike': self.detect_heel_strike_events(df, leg),
            'mid_stance': self.detect_mid_stance_events(df, leg),
            'toe_off': self.detect_toe_off_events(df, leg)
        }


# ============================================================================
# GAIT PHASE DETECTOR
# ============================================================================

class GaitPhaseDetector:
    """
    Anatomically grounded 8-phase gait detection using pressure + IMU fusion.

    Phases (Perry's gait model):
    1. Initial Contact (IC): Heel strike
    2. Loading Response (LR): Weight acceptance
    3. Mid-Stance (MSt): Single limb support
    4. Terminal Stance (TSt): Heel rise
    5. Pre-Swing (PSw): Toe-off preparation
    6. Initial Swing (ISw): Leg acceleration
    7. Mid-Swing (MSw): Leg advancement
    8. Terminal Swing (TSw): Leg deceleration
    """

    def __init__(self, config: InsoleConfig, precision_detector: Optional[PressureSensorZoneDetector] = None):
        self.config = config
        self.precision_detector = precision_detector
        self.precision_events = None  # Cache for detected events

    def detect_gait_cycles(self, df: pd.DataFrame, leg: str) -> List[GaitCycle]:
        """
        Detect complete gait cycles (IC to IC) for one leg.

        Returns:
            List of GaitCycle objects with phase annotations
        """
        print(f"\nDetecting gait cycles for {leg} leg...")

        timestamps = df['timestamp'].values

        # Use precision detector if available
        if self.precision_detector is not None:
            print(f"  Using high-precision sensor zone detection...")
            if self.precision_events is None:
                self.precision_events = self.precision_detector.detect_all_events(df, leg)

            heel_strikes = self.precision_events['heel_strike']
            ic_indices = [event.frame_start for event in heel_strikes]
            print(f"  Found {len(ic_indices)} precision heel strikes (avg confidence: {np.mean([e.confidence for e in heel_strikes]):.2f})")
        else:
            # Fallback to original detection
            pressure = df[f'{leg}_pressure_total'].values
            pressure_deriv = df[f'{leg}_pressure_deriv'].values
            ic_indices = self._detect_initial_contacts(pressure, pressure_deriv)
            print(f"  Found {len(ic_indices)} initial contacts (legacy method)")

        # Build gait cycles between consecutive ICs
        cycles = []
        for i in range(len(ic_indices) - 1):
            start_idx = ic_indices[i]
            end_idx = ic_indices[i + 1]

            start_time = timestamps[start_idx]
            end_time = timestamps[end_idx]
            duration = end_time - start_time

            # Validate duration
            if not (self.config.min_cycle_duration <= duration <= self.config.max_cycle_duration):
                continue

            # Detect phases within this cycle
            phases = self._detect_phases_in_cycle(
                df, leg, start_idx, end_idx, start_time
            )

            # Skip cycles with invalid phase detection
            if not phases or len(phases) != 8:
                continue

            # Compute stance and swing durations
            stance_phases = [p for p in phases if p.phase_number <= 5]  # IC to PSw
            swing_phases = [p for p in phases if p.phase_number > 5]     # ISw to TSw

            stance_duration = sum(p.duration for p in stance_phases)
            swing_duration = sum(p.duration for p in swing_phases)

            ratio = stance_duration / (swing_duration + 1e-6)

            cycle = GaitCycle(
                leg=leg,
                cycle_id=len(cycles) + 1,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                phases=phases,
                stance_duration=stance_duration,
                swing_duration=swing_duration,
                stance_swing_ratio=ratio
            )

            cycles.append(cycle)

        print(f"  Extracted {len(cycles)} valid gait cycles")
        return cycles

    def _detect_initial_contacts(self, pressure: np.ndarray,
                                 pressure_deriv: np.ndarray) -> List[int]:
        """
        Detect initial contact (heel strike) events with adaptive multi-loop optimization.

        Uses dynamic threshold adjustment to find optimal step detection parameters.
        Inspired by pressure sensor zone detection with region validation.

        Biomechanical signature: Sharp pressure rise with sustained contact
        """
        # Adaptive step detection with multi-loop optimization
        min_duration_frames = int(self.config.sampling_rate * self.config.min_stance_duration)
        validation_window = 5  # frames to check before/after for clean transitions

        best_steps = []
        best_count = 0
        max_loops = 5

        for loop in range(max_loops):
            # Dynamic threshold calculation (increases each iteration)
            base_threshold = self.config.pressure_threshold
            noise_threshold = base_threshold + loop * 0.1 * np.std(pressure)

            # Dynamic prominence for peak detection
            prominence = (0.12 + 0.05 * loop) * (np.max(pressure) - np.min(pressure))

            # Find peaks with adaptive parameters
            peaks, _ = find_peaks(
                pressure,
                height=noise_threshold,
                prominence=prominence,
                distance=min_duration_frames
            )

            # Detect contact periods using threshold crossing
            is_in_contact = pressure > noise_threshold
            diffs = np.diff(is_in_contact.astype(int))
            initial_starts = np.where(diffs == 1)[0]
            initial_ends = np.where(diffs == -1)[0]

            # Handle edge cases
            if is_in_contact[0]:
                initial_starts = np.insert(initial_starts, 0, 0)
            if is_in_contact[-1]:
                initial_ends = np.append(initial_ends, len(is_in_contact) - 1)

            # Align starts and ends
            min_len = min(len(initial_starts), len(initial_ends))
            initial_starts = initial_starts[:min_len]
            initial_ends = initial_ends[:min_len]

            # Validate each contact period
            candidate_steps = []
            for start, end in zip(initial_starts, initial_ends):
                if (end - start) < min_duration_frames:
                    continue

                # Check for clean start (low pressure before contact)
                is_clean_start = True
                pre_window_start = max(0, start - validation_window)
                if start > 0 and np.mean(pressure[pre_window_start:start]) >= noise_threshold:
                    is_clean_start = False

                # Check for clean end (low pressure after contact)
                is_clean_end = True
                post_window_end = min(len(pressure), end + 1 + validation_window)
                if end < len(pressure) - 1 and np.mean(pressure[end+1:post_window_end]) >= noise_threshold:
                    is_clean_end = False

                if is_clean_start and is_clean_end:
                    # Find peak within this contact period
                    if start < end:
                        peak_in_window = np.argmax(pressure[start:end])
                        candidate_steps.append(start + peak_in_window)
                    else:
                        candidate_steps.append(start)

            # Track best result across iterations
            if len(candidate_steps) > best_count:
                best_steps = candidate_steps
                best_count = len(candidate_steps)

            # Early exit if we found reasonable number of steps (8-40 is typical for 10MWT)
            if 8 <= len(candidate_steps) <= 40:
                break

        return best_steps

    def _detect_phases_in_cycle(self, df: pd.DataFrame, leg: str,
                                start_idx: int, end_idx: int,
                                start_time: float) -> List[GaitPhase]:
        """
        Detect all 8 gait phases using dynamic sensor-based segmentation.

        Uses high-precision pressure sensor events instead of time-based heuristics.
        """
        # CRITICAL: Must use end_idx+1 to match the slicing in _detect_phases_dynamically
        cycle_data = df.iloc[start_idx:end_idx+1].copy()
        timestamps = cycle_data['timestamp'].values

        # Use event-anchored detection if precision detector available
        if self.precision_detector is not None and self.precision_events is not None:
            return self._detect_phases_event_anchored(
                df, leg, start_idx, end_idx, timestamps
            )
        else:
            # Use dynamic sensor-based detection
            return self._detect_phases_dynamically(
                df, leg, start_idx, end_idx, timestamps
            )

    def _detect_phases_event_anchored(self, df: pd.DataFrame, leg: str,
                                      start_idx: int, end_idx: int,
                                      timestamps: np.ndarray) -> List[GaitPhase]:
        """
        Event-anchored phase detection using precision pressure sensor events.

        Phase Segmentation:
        - IC: Heel strike start
        - LR: IC → mid-stance start
        - MSt: mid-stance start → end
        - TSt: mid-stance end → 90% of stance
        - PSw: 90% stance → toe-off
        - ISw, MSw, TSw: Divide swing into thirds
        """
        # 1. Get high-precision events within this cycle window
        events = {
            event_type: [e for e in self.precision_events[event_type]
                        if start_idx <= e.frame_start <= end_idx]
            for event_type in ['heel_strike', 'mid_stance', 'toe_off']
        }

        ic_event = events['heel_strike'][0] if events['heel_strike'] else None
        ms_events = events['mid_stance'] if events['mid_stance'] else []
        to_event = events['toe_off'][0] if events['toe_off'] else None

        # 2. Validate event order
        if not ic_event or not to_event:
            # Missing critical events - fall back to dynamic detection
            return self._detect_phases_dynamically(df, leg, start_idx, end_idx, timestamps)

        if ic_event.frame_start >= to_event.frame_start:
            # Invalid event order - skip cycle
            return []

        # 3. Extract key timestamps
        ic_t = ic_event.event_start
        to_t = to_event.event_end
        next_ic_t = timestamps[-1]

        # Find the primary mid-stance event (longest duration or highest confidence)
        ms_event = None
        if ms_events:
            ms_event = max(ms_events, key=lambda e: e.duration * e.confidence)

        # 4. Build stance phases using real biomechanical events
        phases = []

        # Phase 1: Initial Contact (brief impact phase)
        ic_duration = 0.05  # 50ms typical IC duration
        ic_end_t = min(ic_t + ic_duration, timestamps[-1])
        ic_end_idx = np.argmin(np.abs(df['timestamp'].values - ic_end_t))

        phases.append(GaitPhase(
            phase_name='Initial Contact',
            leg=leg,
            start_time=ic_t,
            end_time=ic_end_t,
            start_idx=ic_event.frame_start,
            end_idx=ic_end_idx,
            duration=ic_end_t - ic_t,
            phase_number=1,
            support_type=GaitPhase.classify_support_type(1)
        ))

        # Phase 2: Loading Response (IC → mid-stance start)
        if ms_event:
            lr_start_t = ic_end_t
            lr_end_t = ms_event.event_start
            lr_start_idx = ic_end_idx
            lr_end_idx = ms_event.frame_start
        else:
            # Estimate LR as first 20% of stance if no mid-stance detected
            lr_start_t = ic_end_t
            lr_end_t = ic_t + (to_t - ic_t) * 0.2
            lr_start_idx = ic_end_idx
            lr_end_idx = np.argmin(np.abs(df['timestamp'].values - lr_end_t))

        phases.append(GaitPhase(
            phase_name='Loading Response',
            leg=leg,
            start_time=lr_start_t,
            end_time=lr_end_t,
            start_idx=lr_start_idx,
            end_idx=lr_end_idx,
            duration=lr_end_t - lr_start_t,
            phase_number=2,
            support_type=GaitPhase.classify_support_type(2)
        ))

        # Phase 3: Mid-Stance (mid-stance event duration)
        if ms_event:
            mst_start_t = ms_event.event_start
            mst_end_t = ms_event.event_end
            mst_start_idx = ms_event.frame_start
            mst_end_idx = ms_event.frame_end
        else:
            # Estimate MSt as 20-60% of stance
            mst_start_t = lr_end_t
            mst_end_t = ic_t + (to_t - ic_t) * 0.6
            mst_start_idx = lr_end_idx
            mst_end_idx = np.argmin(np.abs(df['timestamp'].values - mst_end_t))

        phases.append(GaitPhase(
            phase_name='Mid-Stance',
            leg=leg,
            start_time=mst_start_t,
            end_time=mst_end_t,
            start_idx=mst_start_idx,
            end_idx=mst_end_idx,
            duration=mst_end_t - mst_start_t,
            phase_number=3,
            support_type=GaitPhase.classify_support_type(3)
        ))

        # Phase 4: Terminal Stance (mid-stance end → 90% of stance)
        stance_duration = to_t - ic_t
        tst_start_t = mst_end_t
        tst_end_t = ic_t + stance_duration * 0.9
        tst_start_idx = mst_end_idx
        tst_end_idx = np.argmin(np.abs(df['timestamp'].values - tst_end_t))

        phases.append(GaitPhase(
            phase_name='Terminal Stance',
            leg=leg,
            start_time=tst_start_t,
            end_time=tst_end_t,
            start_idx=tst_start_idx,
            end_idx=tst_end_idx,
            duration=tst_end_t - tst_start_t,
            phase_number=4,
            support_type=GaitPhase.classify_support_type(4)
        ))

        # Phase 5: Pre-Swing (last 10% before toe-off)
        psw_start_t = tst_end_t
        psw_end_t = to_t
        psw_start_idx = tst_end_idx
        psw_end_idx = to_event.frame_end

        phases.append(GaitPhase(
            phase_name='Pre-Swing',
            leg=leg,
            start_time=psw_start_t,
            end_time=psw_end_t,
            start_idx=psw_start_idx,
            end_idx=psw_end_idx,
            duration=psw_end_t - psw_start_t,
            phase_number=5,
            support_type=GaitPhase.classify_support_type(5)
        ))

        # 5. Build swing phases (divide into thirds with validation)
        swing_duration = next_ic_t - to_t

        # Validate swing duration (0.3-0.6s typical at normal walking speed)
        if swing_duration < 0.25 or swing_duration > 0.8:
            # Invalid swing - might indicate detection error
            print(f"  Warning: Unusual swing duration {swing_duration:.3f}s for {leg} leg")

        swing_phase_duration = swing_duration / 3.0

        swing_phases = [
            ('Initial Swing', 6),
            ('Mid-Swing', 7),
            ('Terminal Swing', 8)
        ]

        for i, (name, num) in enumerate(swing_phases):
            phase_start_t = to_t + i * swing_phase_duration
            phase_end_t = to_t + (i + 1) * swing_phase_duration

            phase_start_idx = np.argmin(np.abs(df['timestamp'].values - phase_start_t))
            phase_end_idx = np.argmin(np.abs(df['timestamp'].values - phase_end_t))

            phases.append(GaitPhase(
                phase_name=name,
                leg=leg,
                start_time=phase_start_t,
                end_time=phase_end_t,
                start_idx=phase_start_idx,
                end_idx=phase_end_idx,
                duration=phase_end_t - phase_start_t,
                phase_number=num,
                support_type=GaitPhase.classify_support_type(num)
            ))

        return phases

    def _detect_phases_dynamically(self, df: pd.DataFrame, leg: str,
                                   start_idx: int, end_idx: int,
                                   timestamps: np.ndarray) -> List[GaitPhase]:
        """
        Dynamically detect 8 gait phases based on sensor transitions.

        Phase detection rules (NO fixed percentages):
        - IC: Hindfoot (value4) sharp rise
        - LR: Hind+Mid active, Fore off
        - MSt: Mid dominant, Hind decreasing
        - TSt: Fore rising
        - PSw: Fore only
        - ISw: All off + gyro rise
        - MSw: Zero pressure, gyro peak
        - TSw: Gyro fall pre-IC
        """
        leg_prefix = leg[0].upper()

        # Extract sensor data for this cycle
        cycle_data = df.iloc[start_idx:end_idx+1].copy()

        # Pressure sensors
        P1 = cycle_data.get(f'{leg_prefix}_value1', pd.Series(0, index=cycle_data.index)).values  # Toe
        P2 = cycle_data.get(f'{leg_prefix}_value2', pd.Series(0, index=cycle_data.index)).values  # Forefoot
        P3 = cycle_data.get(f'{leg_prefix}_value3', pd.Series(0, index=cycle_data.index)).values  # Midfoot
        P4 = cycle_data.get(f'{leg_prefix}_value4', pd.Series(0, index=cycle_data.index)).values  # Hindfoot

        # IMU (use filtered if available)
        gy_col = f'{leg_prefix}_GYRO_Y_filt' if f'{leg_prefix}_GYRO_Y_filt' in cycle_data.columns else f'{leg_prefix}_GYRO_Y'
        GY = cycle_data.get(gy_col, pd.Series(0, index=cycle_data.index)).values

        # Derivatives
        dP4 = np.gradient(P4)
        dP2 = np.gradient(P2)
        dGY = np.gradient(GY)

        # Thresholds
        p_th = np.nanmax([P1.max(), P2.max(), P3.max(), P4.max()]) * 0.3
        gyro_th = 50.0

        phases = []
        n = len(P4)

        # 1. IC (Initial Contact): Hindfoot sharp rise
        ic_candidates = np.where((dP4 > 0) & (P4 > p_th))[0]
        ic_start = 0
        ic_end = min(ic_candidates[0] + 5, n-1) if len(ic_candidates) > 0 else 5
        phases.append(self._create_phase("Initial Contact", leg, start_idx, ic_start, ic_end, timestamps, 1))

        # 2. LR (Loading Response): Hind+Mid active, Fore off
        lr_candidates = np.where(((P4 + P1 + P3) > p_th) & (P2 < p_th))[0]
        lr_start = ic_end
        lr_end = min(lr_candidates[-1] if len(lr_candidates) > 0 else int(0.1*n), n-1)
        phases.append(self._create_phase("Loading Response", leg, start_idx, lr_start, lr_end, timestamps, 2))

        # 3. MSt (Mid Stance): Mid dominant, Hind decreasing
        mst_candidates = np.where(((P1 + P3) > p_th) & (dP4 < 0))[0]
        mst_start = lr_end
        mst_end = min(mst_candidates[-1] if len(mst_candidates) > 0 else int(0.3*n), n-1)
        phases.append(self._create_phase("Mid-Stance", leg, start_idx, mst_start, mst_end, timestamps, 3))

        # 4. TSt (Terminal Stance): Fore rising
        tst_candidates = np.where((dP2 > 0) & ((P1 + P3) > p_th))[0]
        tst_start = mst_end
        tst_end = min(tst_candidates[-1] if len(tst_candidates) > 0 else int(0.5*n), n-1)
        phases.append(self._create_phase("Terminal Stance", leg, start_idx, tst_start, tst_end, timestamps, 4))

        # 5. PSw (Pre-Swing): Fore only
        psw_candidates = np.where((P2 > p_th) & ((P1 + P3) < p_th))[0]
        psw_start = tst_end
        psw_end = min(psw_candidates[-1] if len(psw_candidates) > 0 else int(0.6*n), n-1)
        phases.append(self._create_phase("Pre-Swing", leg, start_idx, psw_start, psw_end, timestamps, 5))

        # 6. ISw (Initial Swing): All off + gyro rise
        isw_candidates = np.where((np.max([P1, P2, P3, P4], axis=0) < p_th*0.5) & (np.abs(GY) > gyro_th))[0]
        isw_start = psw_end
        isw_end = min(isw_candidates[-1] if len(isw_candidates) > 0 else int(0.73*n), n-1)
        phases.append(self._create_phase("Initial Swing", leg, start_idx, isw_start, isw_end, timestamps, 6))

        # 7. MSw (Mid Swing): Zero pressure, gyro peak
        msw_candidates = np.where((np.max([P1, P2, P3, P4], axis=0) < p_th*0.5) & (np.abs(GY) == np.abs(GY).max()))[0]
        msw_start = isw_end
        msw_end = min(msw_candidates[-1] if len(msw_candidates) > 0 else int(0.87*n), n-1)
        phases.append(self._create_phase("Mid-Swing", leg, start_idx, msw_start, msw_end, timestamps, 7))

        # 8. TSw (Terminal Swing): Gyro fall pre-IC
        tsw_start = msw_end
        tsw_end = n - 1
        phases.append(self._create_phase("Terminal Swing", leg, start_idx, tsw_start, tsw_end, timestamps, 8))

        return phases

    def _create_phase(self, name: str, leg: str, cycle_start_idx: int,
                     start_rel: int, end_rel: int,
                     timestamps: np.ndarray, phase_num: int) -> GaitPhase:
        """Helper to create GaitPhase from relative indices."""
        start_idx = cycle_start_idx + start_rel
        end_idx = cycle_start_idx + end_rel

        # Use relative indices to access timestamps array (which is local to cycle)
        start_time = timestamps[start_rel]
        end_time = timestamps[end_rel]
        duration = end_time - start_time

        return GaitPhase(
            phase_name=name,
            leg=leg,
            start_time=start_time,
            end_time=end_time,
            start_idx=start_idx,
            end_idx=end_idx,
            duration=duration,
            phase_number=phase_num,
            support_type=GaitPhase.classify_support_type(phase_num)
        )


# ============================================================================
# CYCLOGRAM GENERATOR
# ============================================================================

class CyclogramGenerator:
    """Generate IMU-based coordination cyclograms for gait cycles."""

    def __init__(self, config: InsoleConfig):
        self.config = config

    def generate_cyclograms(self, df: pd.DataFrame,
                           cycles: List[GaitCycle],
                           leg: str,
                           has_real_gyro: bool = True) -> List[CyclogramData]:
        """
        Generate multiple cyclogram types for each gait cycle.

        Cyclogram types (6 2D + 2 3D = 8 total):
        2D Cyclograms:
        1. ACC_X vs ACC_Y (frontal-sagittal plane)
        2. ACC_X vs ACC_Z (frontal-vertical plane)
        3. ACC_Y vs ACC_Z (sagittal-vertical plane)
        4. GYRO_X vs GYRO_Y (roll-pitch plane) - ONLY if has_real_gyro=True
        5. GYRO_X vs GYRO_Z (roll-yaw plane) - ONLY if has_real_gyro=True
        6. GYRO_Y vs GYRO_Z (pitch-yaw plane) - ONLY if has_real_gyro=True

        3D Cyclograms:
        7. ACC 3D (X, Y, Z)
        8. GYRO 3D (X, Y, Z) - ONLY if has_real_gyro=True
        """
        cyclograms = []

        # Convert leg name to uppercase prefix
        leg_prefix = 'L' if leg == 'left' else 'R'

        # 2D cyclogram types: All 3 plane combinations for ACC (always) and GYRO (conditional)
        cyclogram_2d_types = [
            # Accelerometer planes (always included)
            (f'{leg_prefix}_ACC_X_filt', f'{leg_prefix}_ACC_Y_filt', None, 'ACC_X', 'ACC_Y', None, False),
            (f'{leg_prefix}_ACC_X_filt', f'{leg_prefix}_ACC_Z_filt', None, 'ACC_X', 'ACC_Z', None, False),
            (f'{leg_prefix}_ACC_Y_filt', f'{leg_prefix}_ACC_Z_filt', None, 'ACC_Y', 'ACC_Z', None, False),
            # Gyroscope planes (conditional - only if real GYRO data exists)
            (f'{leg_prefix}_GYRO_X_filt', f'{leg_prefix}_GYRO_Y_filt', None, 'GYRO_X', 'GYRO_Y', None, True),
            (f'{leg_prefix}_GYRO_X_filt', f'{leg_prefix}_GYRO_Z_filt', None, 'GYRO_X', 'GYRO_Z', None, True),
            (f'{leg_prefix}_GYRO_Y_filt', f'{leg_prefix}_GYRO_Z_filt', None, 'GYRO_Y', 'GYRO_Z', None, True),
        ]

        # 3D cyclogram types
        cyclogram_3d_types = [
            (f'{leg_prefix}_ACC_X_filt', f'{leg_prefix}_ACC_Y_filt', f'{leg_prefix}_ACC_Z_filt', 'ACC_X', 'ACC_Y', 'ACC_Z', False),
            (f'{leg_prefix}_GYRO_X_filt', f'{leg_prefix}_GYRO_Y_filt', f'{leg_prefix}_GYRO_Z_filt', 'GYRO_X', 'GYRO_Y', 'GYRO_Z', True),
        ]

        for cycle in cycles:
            # Generate 2D cyclograms
            for x_col, y_col, z_col, x_label, y_label, z_label, is_gyro in cyclogram_2d_types:
                # Skip GYRO cyclograms if no real GYRO data exists (prevents placeholder contamination)
                if is_gyro and not has_real_gyro:
                    continue

                cyclogram = self._extract_cyclogram(
                    df, cycle, x_col, y_col, z_col, x_label, y_label, z_label
                )
                cyclograms.append(cyclogram)

            # Generate 3D cyclograms
            for x_col, y_col, z_col, x_label, y_label, z_label, is_gyro in cyclogram_3d_types:
                # Skip GYRO cyclograms if no real GYRO data exists (prevents placeholder contamination)
                if is_gyro and not has_real_gyro:
                    continue

                cyclogram = self._extract_cyclogram(
                    df, cycle, x_col, y_col, z_col, x_label, y_label, z_label, is_3d=True
                )
                cyclograms.append(cyclogram)

        return cyclograms

    def _extract_cyclogram(self, df: pd.DataFrame, cycle: GaitCycle,
                          x_col: str, y_col: str, z_col: Optional[str],
                          x_label: str, y_label: str, z_label: Optional[str],
                          is_3d: bool = False) -> CyclogramData:
        """Extract single cyclogram for one cycle (2D or 3D)."""

        # Get cycle data
        start_idx = cycle.phases[0].start_idx
        end_idx = cycle.phases[-1].end_idx

        cycle_data = df.iloc[start_idx:end_idx]

        x_signal = cycle_data[x_col].values
        y_signal = cycle_data[y_col].values
        z_signal = cycle_data[z_col].values if z_col is not None else None

        # Normalize to percentage of gait cycle
        normalized_length = self.config.cyclogram_resolution

        if len(x_signal) < 2:
            # Handle edge case
            x_signal_norm = np.zeros(normalized_length)
            y_signal_norm = np.zeros(normalized_length)
            z_signal_norm = np.zeros(normalized_length) if is_3d else None
        else:
            time_original = np.linspace(0, 100, len(x_signal))
            time_normalized = np.linspace(0, 100, normalized_length)

            x_interp = interp1d(time_original, x_signal, kind='cubic',
                               fill_value='extrapolate')
            y_interp = interp1d(time_original, y_signal, kind='cubic',
                               fill_value='extrapolate')

            x_signal_norm = x_interp(time_normalized)
            y_signal_norm = y_interp(time_normalized)

            # Interpolate z-axis for 3D cyclograms
            if is_3d and z_signal is not None:
                z_interp = interp1d(time_original, z_signal, kind='cubic',
                                   fill_value='extrapolate')
                z_signal_norm = z_interp(time_normalized)
            else:
                z_signal_norm = None

        # Compute phase boundary indices in normalized space
        phase_indices = []
        phase_labels = []

        for phase in cycle.phases:
            # Convert absolute index to relative position in cycle
            relative_start = phase.start_idx - start_idx
            relative_pos = (relative_start / len(cycle_data)) * normalized_length
            phase_indices.append(int(relative_pos))
            phase_labels.append(phase.phase_name)

        return CyclogramData(
            cycle=cycle,
            x_signal=x_signal_norm,
            y_signal=y_signal_norm,
            z_signal=z_signal_norm,
            x_label=x_label,
            y_label=y_label,
            z_label=z_label,
            phase_indices=phase_indices,
            phase_labels=phase_labels,
            is_3d=is_3d
        )

    def compute_cyclogram_metrics(self, cyclogram: CyclogramData) -> Dict:
        """
        Compute geometric metrics for cyclogram.

        Returns:
        - area: Signed polygon area
        - curvature_mean: Average curvature
        - curvature_var: Curvature variability
        - smoothness_coeff: Trajectory smoothness
        - perimeter: Loop perimeter
        - compactness: 4π*area / perimeter²
        """
        x = cyclogram.x_signal
        y = cyclogram.y_signal

        # Area (shoelace formula)
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

        # Curvature
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = np.power(dx**2 + dy**2 + 1e-6, 1.5)
        curvature = numerator / denominator

        curvature_mean = np.mean(curvature)
        curvature_var = np.var(curvature)

        # Smoothness (inverse jerk)
        jerk = ddx**2 + ddy**2
        smoothness_coeff = 1.0 / (np.mean(jerk) + 1e-6)

        # Perimeter
        perimeter = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))

        # Compactness
        compactness = (4 * np.pi * area) / (perimeter**2 + 1e-6)

        # Closure error
        closure_error = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)

        return {
            'area': float(area),
            'normalized_area': float(area / (np.std(x) * np.std(y) + 1e-6)),
            'curvature_mean': float(curvature_mean),
            'curvature_var': float(curvature_var),
            'smoothness_coeff': float(smoothness_coeff),
            'perimeter': float(perimeter),
            'compactness': float(compactness),
            'closure_error': float(closure_error)
        }

    def compute_symmetry_metrics(self, left_cyclogram: CyclogramData,
                                right_cyclogram: CyclogramData) -> Dict:
        """Compute L-R symmetry metrics."""
        left_metrics = self.compute_cyclogram_metrics(left_cyclogram)
        right_metrics = self.compute_cyclogram_metrics(right_cyclogram)

        # Normalized differences (1.0 = perfect symmetry)
        area_diff = np.abs(left_metrics['area'] - right_metrics['area'])
        area_avg = (left_metrics['area'] + right_metrics['area']) / 2 + 1e-6
        area_symmetry = 1.0 - (area_diff / area_avg)

        curv_diff = np.abs(left_metrics['curvature_mean'] - right_metrics['curvature_mean'])
        curv_avg = (left_metrics['curvature_mean'] + right_metrics['curvature_mean']) / 2 + 1e-6
        curvature_symmetry = 1.0 - (curv_diff / curv_avg)

        smooth_diff = np.abs(left_metrics['smoothness_coeff'] - right_metrics['smoothness_coeff'])
        smooth_avg = (left_metrics['smoothness_coeff'] + right_metrics['smoothness_coeff']) / 2 + 1e-6
        smoothness_symmetry = 1.0 - (smooth_diff / smooth_avg)

        # DTW distance
        dtw_dist = self._compute_dtw(left_cyclogram, right_cyclogram)

        return {
            'area_symmetry': float(np.clip(area_symmetry, 0, 1)),
            'curvature_symmetry': float(np.clip(curvature_symmetry, 0, 1)),
            'smoothness_symmetry': float(np.clip(smoothness_symmetry, 0, 1)),
            'dtw_distance': float(dtw_dist)
        }

    def _compute_dtw(self, left: CyclogramData, right: CyclogramData) -> float:
        """Compute DTW distance between two cyclograms."""
        A = np.column_stack([left.x_signal, left.y_signal])
        B = np.column_stack([right.x_signal, right.y_signal])

        if HAS_FASTDTW:
            dist, _ = fastdtw(A, B, dist=lambda x, y: float(np.linalg.norm(x - y)))
            return float(dist) / max(len(A), len(B))

        # O(N²) DP fallback
        N, M = len(A), len(B)
        D = np.full((N+1, M+1), np.inf)
        D[0, 0] = 0.0
        for i in range(1, N+1):
            for j in range(1, M+1):
                cost = float(np.linalg.norm(A[i-1] - B[j-1]))
                D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
        return float(D[N, M]) / max(N, M)


# ============================================================================
# MORPHOLOGICAL MEAN CYCLOGRAM
# ============================================================================

class MorphologicalMeanCyclogramComputer:
    """
    Compute phase-aligned median cyclograms (MMC).

    Ported from Analysis.py MMC implementation with adaptations for insole data.
    """

    def __init__(self):
        pass

    def find_median_reference_loop(self, cyclograms: List[CyclogramData]) -> int:
        """Find median reference loop using DTW distances."""
        if len(cyclograms) == 1:
            return 0

        n = len(cyclograms)
        dtw_sum = np.zeros(n)

        for i in range(n):
            for j in range(n):
                if i != j:
                    dtw_sum[i] += self.compute_dtw_distance(cyclograms[i], cyclograms[j])

        return int(np.argmin(dtw_sum))

    def center_loop(self, cyclogram: CyclogramData) -> np.ndarray:
        """Center loop to centroid (removes position bias)."""
        x_centered = cyclogram.x_signal - np.mean(cyclogram.x_signal)
        y_centered = cyclogram.y_signal - np.mean(cyclogram.y_signal)

        if cyclogram.is_3d and cyclogram.z_signal is not None:
            z_centered = cyclogram.z_signal - np.mean(cyclogram.z_signal)
            return np.column_stack([x_centered, y_centered, z_centered])

        return np.column_stack([x_centered, y_centered])

    def rescale_loop(self, centered_loop: np.ndarray, target_area: float,
                    current_area: float) -> np.ndarray:
        """Rescale loop to target area (removes amplitude bias)."""
        if abs(current_area) < 1e-6:
            return centered_loop

        scale_factor = np.sqrt(abs(target_area) / abs(current_area))
        return centered_loop * scale_factor

    def compute_mmc(self, cyclograms: List[CyclogramData]) -> Optional[MorphologicalMeanCyclogram]:
        """
        Compute Morphological Mean Cyclogram.

        Algorithm:
        1. Find median reference loop
        2. Center all loops to centroid
        3. Rescale to median area
        4. Compute median shape
        5. Compute variance envelope
        6. Calculate metrics

        Returns MMC object or None if insufficient data.
        """
        if not cyclograms or len(cyclograms) < 2:
            return None

        # Step 1: Find median reference
        median_ref_idx = self.find_median_reference_loop(cyclograms)

        # Step 2: Compute areas
        areas = [self._compute_area(c) for c in cyclograms]
        median_area = float(np.median(np.abs(areas)))

        # Step 3: Center and rescale all loops
        centered_rescaled_loops = []
        for cyclogram, area in zip(cyclograms, areas):
            centered = self.center_loop(cyclogram)
            rescaled = self.rescale_loop(centered, median_area, area)
            centered_rescaled_loops.append(rescaled)

        # Step 4: Stack and compute median
        loops_array = np.stack(centered_rescaled_loops, axis=0)
        median_trajectory = np.median(loops_array, axis=0)

        # Step 5: Variance envelope
        std_trajectory = np.std(loops_array, axis=0)
        envelope_lower = median_trajectory - std_trajectory
        envelope_upper = median_trajectory + std_trajectory

        # Step 6: Shape dispersion index
        var_x = np.var(loops_array[:, :, 0], axis=0)
        var_y = np.var(loops_array[:, :, 1], axis=0)
        mean_variance = float(np.mean(var_x) + np.mean(var_y))
        shape_dispersion_index = mean_variance / max(median_area, 1e-6)

        # Step 7: Alignment quality
        dtw_distances = [self.compute_dtw_distance(cyclograms[i], cyclograms[median_ref_idx])
                        for i in range(len(cyclograms))]
        alignment_quality = float(np.mean(dtw_distances))

        # Step 8: Confidence ellipse
        confidence_ellipse = self._compute_confidence_ellipse(cyclograms)

        # Extract sensor pair from first cyclogram
        sensor_pair = (cyclograms[0].x_label, cyclograms[0].y_label)

        return MorphologicalMeanCyclogram(
            leg=cyclograms[0].cycle.leg,
            sensor_pair=sensor_pair,
            median_trajectory=median_trajectory,
            variance_envelope_lower=envelope_lower,
            variance_envelope_upper=envelope_upper,
            shape_dispersion_index=shape_dispersion_index,
            confidence_ellipse_params=confidence_ellipse,
            n_loops=len(cyclograms),
            alignment_quality=alignment_quality,
            median_area=median_area
        )

    def compute_dtw_distance(self, loop1: CyclogramData, loop2: CyclogramData) -> float:
        """DTW distance between two cyclograms."""
        A = np.column_stack([loop1.x_signal, loop1.y_signal])
        B = np.column_stack([loop2.x_signal, loop2.y_signal])

        if HAS_FASTDTW:
            dist, _ = fastdtw(A, B, dist=lambda x, y: float(np.linalg.norm(x - y)))
            return float(dist) / max(len(A), len(B))

        # O(N²) DP fallback
        N, M = len(A), len(B)
        D = np.full((N+1, M+1), np.inf)
        D[0, 0] = 0.0
        for i in range(1, N+1):
            for j in range(1, M+1):
                cost = float(np.linalg.norm(A[i-1] - B[j-1]))
                D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
        return float(D[N, M]) / max(N, M)

    def _compute_area(self, cyclogram: CyclogramData) -> float:
        """Compute signed area using shoelace formula."""
        x = cyclogram.x_signal
        y = cyclogram.y_signal
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def _compute_confidence_ellipse(self, cyclograms: List[CyclogramData]) -> Dict[str, float]:
        """Compute 95% confidence ellipse from loop centroids."""
        centroids = np.array([[np.mean(c.x_signal), np.mean(c.y_signal)]
                             for c in cyclograms])

        if len(centroids) < 2:
            return {"center_x": 0.0, "center_y": 0.0,
                   "major_axis": 0.0, "minor_axis": 0.0, "angle": 0.0}

        cov = np.cov(centroids.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        chi2_val = 5.991  # 95% CI for 2-DOF
        major_axis = 2 * np.sqrt(chi2_val * eigenvalues[0])
        minor_axis = 2 * np.sqrt(chi2_val * eigenvalues[1])

        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]) * 180 / np.pi
        center = np.mean(centroids, axis=0)

        return {
            "center_x": float(center[0]),
            "center_y": float(center[1]),
            "major_axis": float(major_axis),
            "minor_axis": float(minor_axis),
            "angle": float(angle)
        }


# ============================================================================
# VALIDATION ENGINE
# ============================================================================

class ValidationEngine:
    """Anatomical plausibility validation for detected gait patterns."""

    def __init__(self, config: InsoleConfig):
        self.config = config

    def validate_gait_cycles(self, left_cycles: List[GaitCycle],
                            right_cycles: List[GaitCycle]) -> ValidationMetrics:
        """
        Comprehensive validation of detected gait cycles.

        Checks:
        1. Left-right alternation
        2. Stance-to-swing ratio
        3. Bilateral symmetry
        4. Phase sequence integrity
        5. Duration constraints
        """
        issues = []

        # Check 1: Left-right alternation
        alternation_valid = self._check_alternation(left_cycles, right_cycles)
        if not alternation_valid:
            issues.append("Left-right IC events do not alternate properly")

        # Check 2: Stance-swing ratio
        ratio_valid = self._check_stance_swing_ratio(left_cycles + right_cycles)
        if not ratio_valid:
            issues.append("Stance-to-swing ratio outside expected range")

        # Check 3: Bilateral symmetry
        symmetry_valid = self._check_bilateral_symmetry(left_cycles, right_cycles)
        if not symmetry_valid:
            issues.append("Left-right cycle durations differ significantly")

        # Check 4: Phase sequence
        sequence_valid = self._check_phase_sequence(left_cycles + right_cycles)
        if not sequence_valid:
            issues.append("Phase sequence violated in some cycles")

        # Check 5: Duration constraints
        duration_valid = self._check_duration_constraints(left_cycles + right_cycles)
        if not duration_valid:
            issues.append("Phase durations outside biomechanical constraints")

        overall_valid = (alternation_valid and ratio_valid and
                        symmetry_valid and sequence_valid and duration_valid)

        return ValidationMetrics(
            left_right_alternation=alternation_valid,
            stance_swing_ratio_valid=ratio_valid,
            bilateral_symmetry_valid=symmetry_valid,
            phase_sequence_valid=sequence_valid,
            duration_constraints_valid=duration_valid,
            overall_valid=overall_valid,
            issues=issues
        )

    def _check_alternation(self, left_cycles: List[GaitCycle],
                          right_cycles: List[GaitCycle]) -> bool:
        """Verify left and right ICs alternate (not simultaneous)."""

        if not left_cycles or not right_cycles:
            return False

        # Collect all IC times with leg labels
        all_ics = []
        for cycle in left_cycles:
            all_ics.append((cycle.start_time, 'left'))
        for cycle in right_cycles:
            all_ics.append((cycle.start_time, 'right'))

        # Sort by time
        all_ics.sort(key=lambda x: x[0])

        # Check for alternation
        for i in range(len(all_ics) - 1):
            if all_ics[i][1] == all_ics[i+1][1]:
                # Same leg twice in a row
                time_diff = all_ics[i+1][0] - all_ics[i][0]
                if time_diff < 0.3:  # Too close, likely detection error
                    return False

        return True

    def _check_stance_swing_ratio(self, cycles: List[GaitCycle]) -> bool:
        """Verify stance-to-swing ratio is within expected range (1.2-2.0)."""

        for cycle in cycles:
            ratio = cycle.stance_swing_ratio
            if not (self.config.stance_swing_ratio_min <= ratio <=
                   self.config.stance_swing_ratio_max):
                return False

        return True

    def _check_bilateral_symmetry(self, left_cycles: List[GaitCycle],
                                  right_cycles: List[GaitCycle]) -> bool:
        """Verify left and right cycles have similar durations."""

        if not left_cycles or not right_cycles:
            return True

        left_mean = np.mean([c.duration for c in left_cycles])
        right_mean = np.mean([c.duration for c in right_cycles])

        diff_ratio = abs(left_mean - right_mean) / ((left_mean + right_mean) / 2)

        return diff_ratio <= self.config.bilateral_tolerance

    def _check_phase_sequence(self, cycles: List[GaitCycle]) -> bool:
        """Verify phases occur in correct anatomical order."""

        expected_sequence = list(range(1, 9))  # 1-8

        for cycle in cycles:
            actual_sequence = [p.phase_number for p in cycle.phases]
            if actual_sequence != expected_sequence:
                return False

        return True

    def _check_duration_constraints(self, cycles: List[GaitCycle]) -> bool:
        """Verify phase durations are biomechanically plausible."""

        for cycle in cycles:
            # Check stance duration
            if not (self.config.min_stance_duration <= cycle.stance_duration):
                return False

            # Check swing duration
            if not (self.config.min_swing_duration <= cycle.swing_duration):
                return False

        return True


# ============================================================================
# SYMMETRY ANALYZER
# ============================================================================

class SymmetryAnalyzer:
    """
    Bilateral symmetry and morphology metrics for gait analysis.

    Quantifies left-vs-right similarity across cyclogram types using:
    - Geometric metrics: Area, curvature distribution
    - Temporal metrics: Smoothness coefficient (inverse jerk)
    - Morphological comparison: Shape similarity indices
    """

    def __init__(self, config: InsoleConfig):
        self.config = config

    def compute_area(self, x: np.ndarray, y: np.ndarray, z: np.ndarray = None) -> float:
        """
        Compute area enclosed by 2D cyclogram or surface area of 3D cyclogram.

        For 2D: Uses shoelace formula for polygonal area
        For 3D: Computes approximate surface area of convex hull

        Args:
            x: X-coordinates of cyclogram trajectory
            y: Y-coordinates of cyclogram trajectory
            z: Z-coordinates (optional, for 3D cyclograms)

        Returns:
            Absolute area/surface area enclosed by the loop
        """
        if z is None:
            # 2D shoelace formula
            return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        else:
            # 3D: Approximate surface area using triangular segments
            # Sum of triangle areas formed by consecutive points and centroid
            centroid = np.array([x.mean(), y.mean(), z.mean()])
            total_area = 0.0

            for i in range(len(x)):
                p1 = np.array([x[i], y[i], z[i]])
                p2 = np.array([x[(i + 1) % len(x)], y[(i + 1) % len(x)], z[(i + 1) % len(x)]])

                # Triangle area = 0.5 * |cross product|
                v1 = p1 - centroid
                v2 = p2 - centroid
                cross = np.cross(v1, v2)
                total_area += 0.5 * np.linalg.norm(cross)

            return total_area

    def compute_curvature(self, x: np.ndarray, y: np.ndarray, z: np.ndarray = None) -> np.ndarray:
        """
        Compute curvature along 2D or 3D trajectory.

        2D: κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
        3D: κ = |r' × r''| / |r'|³

        Args:
            x: X-coordinates of trajectory
            y: Y-coordinates of trajectory
            z: Z-coordinates (optional, for 3D)

        Returns:
            Curvature array along trajectory
        """
        if z is None:
            # 2D curvature
            dx = np.gradient(x)
            dy = np.gradient(y)
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)

            numerator = np.abs(dx * ddy - dy * ddx)
            denominator = np.power(dx**2 + dy**2 + 1e-6, 1.5)

            curvature = numerator / denominator
        else:
            # 3D curvature
            dx = np.gradient(x)
            dy = np.gradient(y)
            dz = np.gradient(z)
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            ddz = np.gradient(dz)

            # First derivative (velocity)
            r_prime = np.column_stack([dx, dy, dz])
            # Second derivative (acceleration)
            r_double_prime = np.column_stack([ddx, ddy, ddz])

            # Cross product for each point
            cross = np.cross(r_prime, r_double_prime)
            cross_norm = np.linalg.norm(cross, axis=1)

            # Magnitude of first derivative cubed
            r_prime_norm = np.linalg.norm(r_prime, axis=1)
            r_prime_norm_cubed = np.power(r_prime_norm + 1e-6, 3)

            curvature = cross_norm / r_prime_norm_cubed

        return curvature

    def compute_smoothness(self, x: np.ndarray, y: np.ndarray, z: np.ndarray = None) -> float:
        """
        Compute smoothness coefficient as inverse of mean jerk.

        Smoothness = 1 / mean(|jerk|)
        Higher values indicate smoother, more controlled movement.

        Args:
            x: X-coordinates of trajectory
            y: Y-coordinates of trajectory
            z: Z-coordinates (optional, for 3D)

        Returns:
            Smoothness coefficient (higher = smoother)
        """
        if z is None:
            # 2D smoothness
            # Compute jerk (third derivative of position)
            dx = np.gradient(x)
            dy = np.gradient(y)
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)

            # Jerk magnitude
            jerk = ddx**2 + ddy**2
        else:
            # 3D smoothness
            dx = np.gradient(x)
            dy = np.gradient(y)
            dz = np.gradient(z)
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            ddz = np.gradient(dz)

            # Jerk magnitude in 3D
            jerk = ddx**2 + ddy**2 + ddz**2

        # Smoothness as inverse of mean jerk
        smoothness = 1.0 / (np.mean(jerk) + 1e-6)

        return smoothness

    def compare_cyclograms(self, left: CyclogramData, right: CyclogramData) -> Dict[str, float]:
        """
        Compare left vs right cyclogram geometry and morphology (supports 2D and 3D).

        Computes bilateral symmetry metrics:
        - Area symmetry: normalized difference in enclosed areas (or surface area for 3D)
        - Curvature symmetry: RMS difference in curvature profiles
        - Smoothness symmetry: difference in movement smoothness

        Args:
            left: Left leg cyclogram data (2D or 3D)
            right: Right leg cyclogram data (2D or 3D)

        Returns:
            Dictionary with symmetry metrics and individual leg measurements
        """
        # Check if 3D or 2D
        z_L = left.z_signal if left.is_3d else None
        z_R = right.z_signal if right.is_3d else None

        # Compute geometric properties for each leg
        A_L = self.compute_area(left.x_signal, left.y_signal, z_L)
        A_R = self.compute_area(right.x_signal, right.y_signal, z_R)

        κ_L = self.compute_curvature(left.x_signal, left.y_signal, z_L)
        κ_R = self.compute_curvature(right.x_signal, right.y_signal, z_R)

        S_L = self.compute_smoothness(left.x_signal, left.y_signal, z_L)
        S_R = self.compute_smoothness(right.x_signal, right.y_signal, z_R)

        # Compute symmetry indices (normalized differences)
        # Symmetry = 1 - normalized_difference (1.0 = perfect symmetry)

        area_diff = np.abs(A_L - A_R) / ((A_L + A_R) / 2 + 1e-6)
        area_symmetry = 1.0 - area_diff

        curvature_diff = np.mean(np.abs(κ_L - κ_R)) / (np.mean([κ_L.mean(), κ_R.mean()]) + 1e-6)
        curvature_symmetry = 1.0 - curvature_diff

        smoothness_diff = np.abs(S_L - S_R) / ((S_L + S_R) / 2 + 1e-6)
        smoothness_symmetry = 1.0 - smoothness_diff

        # Compute overall symmetry score (mean of z-scored metrics)
        symmetry_scores = np.array([area_symmetry, curvature_symmetry, smoothness_symmetry])
        overall_symmetry = np.mean(symmetry_scores)

        return {
            'area_symmetry': area_symmetry,
            'curvature_symmetry': curvature_symmetry,
            'smoothness_symmetry': smoothness_symmetry,
            'overall_symmetry': overall_symmetry,
            'area_L': A_L,
            'area_R': A_R,
            'curvature_mean_L': κ_L.mean(),
            'curvature_mean_R': κ_R.mean(),
            'curvature_var_L': κ_L.var(),
            'curvature_var_R': κ_R.var(),
            'smoothness_L': S_L,
            'smoothness_R': S_R
        }

    def compute_batch_symmetry(self, left_cyclograms: List[CyclogramData],
                               right_cyclograms: List[CyclogramData]) -> pd.DataFrame:
        """
        Compute symmetry metrics for all cyclogram pairs.

        Args:
            left_cyclograms: List of left leg cyclograms
            right_cyclograms: List of right leg cyclograms

        Returns:
            DataFrame with per-cycle and aggregate symmetry metrics
        """
        results = []

        # Group cyclograms by type (ACC_X_vs_ACC_Y, etc.)
        left_by_type = {}
        right_by_type = {}

        for cyclogram in left_cyclograms:
            key = (cyclogram.x_label, cyclogram.y_label)
            if key not in left_by_type:
                left_by_type[key] = []
            left_by_type[key].append(cyclogram)

        for cyclogram in right_cyclograms:
            key = (cyclogram.x_label, cyclogram.y_label)
            if key not in right_by_type:
                right_by_type[key] = []
            right_by_type[key].append(cyclogram)

        # Compare paired cyclograms
        for key in left_by_type.keys():
            if key not in right_by_type:
                continue

            left_cycles = left_by_type[key]
            right_cycles = right_by_type[key]

            # Compare cycle-by-cycle
            for i, (L, R) in enumerate(zip(left_cycles, right_cycles)):
                metrics = self.compare_cyclograms(L, R)
                metrics['cycle_id'] = i + 1
                metrics['cyclogram_type'] = f"{L.x_label}_vs_{L.y_label}"
                results.append(metrics)

        return pd.DataFrame(results)


# ============================================================================
# VISUALIZER WITH DUAL PNG+JSON OUTPUT
# ============================================================================

class InsoleVisualizer:
    """Visualization generation with dual PNG+JSON output for insole gait analysis."""

    def __init__(self, config: InsoleConfig, output_dir: Path, has_real_gyro: bool = True):
        self.config = config
        self.output_dir = output_dir
        self.has_real_gyro = has_real_gyro

        # Create categorized directory structure
        self.output_dir_summary = output_dir / "summary"
        self.output_dir_plots = output_dir / "plots"
        self.output_dir_json = output_dir / "json"

        # Subplot categories under plots/
        self.categories = {
            'gait_phases': self.output_dir_plots / "gait_phases",
            'stride_cyclograms': self.output_dir_plots / "stride_cyclograms",
            'gait_cyclograms': self.output_dir_plots / "gait_cyclograms",
            'mean_cyclograms': self.output_dir_plots / "mean_cyclograms",
            'symmetry': self.output_dir_plots / "symmetry"
        }

        # Mirror structure under json/
        self.json_categories = {
            'gait_phases': self.output_dir_json / "gait_phases",
            'stride_cyclograms': self.output_dir_json / "stride_cyclograms",
            'gait_cyclograms': self.output_dir_json / "gait_cyclograms",
            'mean_cyclograms': self.output_dir_json / "mean_cyclograms",
            'symmetry': self.output_dir_json / "symmetry"
        }

        # Create all directories
        self.output_dir_summary.mkdir(parents=True, exist_ok=True)
        for cat_dir in list(self.categories.values()) + list(self.json_categories.values()):
            cat_dir.mkdir(parents=True, exist_ok=True)

        # Category mapping: analysis_type → category folder
        self.category_mapping = {
            'gyro_stride': 'stride_cyclograms',
            'acc_stride': 'stride_cyclograms',
            '3d_stride': 'stride_cyclograms',
            'gyro_gait': 'gait_cyclograms',
            'acc_gait': 'gait_cyclograms',
            '3d_gait': 'gait_cyclograms',
            'mean_cyclogram': 'mean_cyclograms',
            'mmc_envelope': 'mean_cyclograms',
            'gait_events': 'gait_phases',
            'phase_timeline': 'gait_phases',
            'symmetry_analysis': 'symmetry',
            'lr_comparison': 'symmetry',
            'cyclogram_2d': 'stride_cyclograms'
        }

        # Backward compatibility
        self.output_dir_png = self.output_dir_plots
        self.output_dir_json = self.output_dir_json

        # Initialize MMC computer for mean cyclogram computation
        self.mmc_computer = MorphologicalMeanCyclogramComputer()

        self._setup_colors()

    def _setup_colors(self):
        """Define color scheme for phase segmentation."""
        # Stance phases: High-contrast color sequence (cool tones)
        self.stance_colors = [
            '#1f77b4',  # IC - bright blue
            '#2ca02c',  # LR - green
            '#17becf',  # MSt - cyan
            '#9467bd',  # TSt - purple
            '#8c564b'   # PSw - brown
        ]

        # Swing phases: High-contrast color sequence (warm tones)
        self.swing_colors = [
            '#d62728',  # ISw - red
            '#ff7f0e',  # MSw - orange
            '#e377c2'   # TSw - pink
        ]

        self.phase_colors = self.stance_colors + self.swing_colors

    def _add_phase_boundary_markers(self, ax, x, y, phase_indices, phase_labels, marker_size=8):
        """
        Add visual markers at phase transition boundaries on cyclogram.

        Args:
            ax: Matplotlib axes object
            x, y: Cyclogram signal arrays
            phase_indices: List of phase boundary indices
            phase_labels: List of phase names
            marker_size: Size of boundary markers
        """
        # Add diamond markers at each phase transition point
        for i, idx in enumerate(phase_indices):
            if idx > 0 and idx < len(x):  # Skip start (0) and end
                ax.plot(x[idx], y[idx], marker='D', color='black',
                       markersize=marker_size, markeredgewidth=1.5,
                       markerfacecolor='white', zorder=10,
                       label='Phase Transition' if i == 1 else '')

    def plot_phase_segmented_cyclogram(self, cyclogram: CyclogramData,
                                      output_path: Path, mmc: Optional[MorphologicalMeanCyclogram] = None,
                                      cyclogram_metrics: Optional[Dict] = None):
        """
        Plot cyclogram with phase-based color segmentation and optional MMC overlay.
        """
        fig, ax = plt.subplots(figsize=(10, 10), dpi=self.config.plot_dpi)

        x = cyclogram.x_signal
        y = cyclogram.y_signal

        # Plot full trajectory in light gray
        ax.plot(x, y, color='lightgray', linewidth=0.5, alpha=0.5)

        # Plot phase segments with colors
        phase_indices = cyclogram.phase_indices + [len(x)]  # Add endpoint

        for i in range(len(cyclogram.phase_indices)):
            start_idx = phase_indices[i]
            end_idx = phase_indices[i + 1]

            color = self.phase_colors[i % len(self.phase_colors)]

            ax.plot(x[start_idx:end_idx], y[start_idx:end_idx],
                   color=color, linewidth=2,
                   label=cyclogram.phase_labels[i])

        # Add phase boundary markers at transition points
        self._add_phase_boundary_markers(ax, x, y, cyclogram.phase_indices,
                                         cyclogram.phase_labels, marker_size=8)

        # Overlay MMC if provided
        if mmc is not None:
            ax.plot(mmc.median_x, mmc.median_y, 'k--', linewidth=3,
                   label=f'MMC (n={mmc.n_loops})', alpha=0.7)

            # Variance envelope
            ax.fill_between(mmc.median_x,
                           mmc.variance_envelope_lower[:, 1],
                           mmc.variance_envelope_upper[:, 1],
                           alpha=0.2, color='gray', label='±1 SD')

        # Mark start/end
        ax.plot(x[0], y[0], 'go', markersize=10, label='IC (Start)')
        ax.plot(x[-1], y[-1], 'rs', markersize=10, label='IC (End)')

        # Formatting
        ax.set_xlabel(f'{cyclogram.x_label} (filtered)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{cyclogram.y_label} (filtered)', fontsize=12, fontweight='bold')
        ax.set_title(f'{cyclogram.cycle.leg.title()} Leg Cyclogram - '
                    f'Cycle {cyclogram.cycle.cycle_id}\n'
                    f'Duration: {cyclogram.cycle.duration:.2f}s, '
                    f'Stance/Swing: {cyclogram.cycle.stance_swing_ratio:.2f}',
                    fontsize=14, fontweight='bold')

        ax.legend(loc='best', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()

        # Generate metadata
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        base_name = f"cyclogram_{cyclogram.x_label}_vs_{cyclogram.y_label}_{cyclogram.cycle.leg}_{timestamp}"

        metadata = self._generate_metadata(
            analysis_type='cyclogram_2d',
            leg=cyclogram.cycle.leg,
            sensor_type=f"{cyclogram.x_label}_vs_{cyclogram.y_label}",
            file_name=f"{base_name}.png",
            cyclogram=cyclogram,
            cyclogram_metrics=cyclogram_metrics,
            mmc=mmc
        )

        self.save_outputs(fig, metadata, base_name)

    def plot_aggregated_cyclogram(self, cyclograms: List[CyclogramData],
                                 output_path: Path):
        """
        Plot mean cyclogram with all individual cycles overlaid.
        """
        if not cyclograms:
            return

        fig, ax = plt.subplots(figsize=(10, 10), dpi=self.config.plot_dpi)

        # Plot individual cycles in light gray
        for cyclogram in cyclograms:
            ax.plot(cyclogram.x_signal, cyclogram.y_signal,
                   color='lightgray', linewidth=0.5, alpha=0.3)

        # Compute mean cyclogram
        x_mean = np.mean([c.x_signal for c in cyclograms], axis=0)
        y_mean = np.mean([c.y_signal for c in cyclograms], axis=0)

        # Plot mean with phase segmentation
        phase_indices = cyclograms[0].phase_indices + [len(x_mean)]

        for i in range(len(cyclograms[0].phase_indices)):
            start_idx = phase_indices[i]
            end_idx = phase_indices[i + 1]

            color = self.phase_colors[i % len(self.phase_colors)]

            ax.plot(x_mean[start_idx:end_idx], y_mean[start_idx:end_idx],
                   color=color, linewidth=3,
                   label=cyclograms[0].phase_labels[i])

        # Mark start/end
        ax.plot(x_mean[0], y_mean[0], 'go', markersize=12, label='IC (Start)')
        ax.plot(x_mean[-1], y_mean[-1], 'rs', markersize=12, label='IC (End)')

        # Formatting
        leg = cyclograms[0].cycle.leg
        x_label = cyclograms[0].x_label
        y_label = cyclograms[0].y_label

        ax.set_xlabel(f'{x_label} (filtered)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{y_label} (filtered)', fontsize=12, fontweight='bold')
        ax.set_title(f'{leg.title()} Leg Mean Cyclogram ({len(cyclograms)} cycles)\n'
                    f'{x_label} vs {y_label}',
                    fontsize=14, fontweight='bold')

        ax.legend(loc='best', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()

        # Save with metadata
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        base_name = f"mean_cyclogram_{x_label}_vs_{y_label}_{leg}_{timestamp}"

        metadata = self._generate_metadata(
            analysis_type='mean_cyclogram',
            leg=leg,
            sensor_type=f"{x_label}_vs_{y_label}",
            file_name=f"{base_name}.png",
            cycle_count=len(cyclograms)
        )

        self.save_outputs(fig, metadata, base_name)

    def save_outputs(self, fig: plt.Figure, metadata: Dict, base_name: str, category: str = None) -> None:
        """
        Save PNG + JSON pair to categorized directories.

        Args:
            fig: Figure to save
            metadata: Complete metadata dict
            base_name: Base filename (no extension)
            category: Category folder (auto-detected from metadata['analysis_type'] if None)
        """
        analysis_type = metadata.get('analysis_type', 'unknown')

        # Auto-detect category if not provided
        if category is None:
            category = self.category_mapping.get(analysis_type, 'plots')  # Default to root plots/

        # Determine output paths
        if category in self.categories:
            png_dir = self.categories[category]
            json_dir = self.json_categories[category]
        else:
            # Fallback to root plots/json directories
            png_dir = self.output_dir_plots
            json_dir = self.output_dir_json

        png_path = png_dir / f"{base_name}.png"
        json_path = json_dir / f"{base_name}.json"

        # Save PNG with high quality
        fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')

        # Save JSON with pretty formatting
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False, default=str)

        print(f"  Saved → {category}/{png_path.name} + {json_path.name}")

        plt.close(fig)

    def _generate_metadata(self, analysis_type: str, leg: str, sensor_type: str, **kwargs) -> Dict:
        """
        Generate complete metadata for JSON export.

        Enhanced schema includes 9 categories:
        - file_name, timestamp, subject, leg, analysis_type, sensor_type
        - event_timings (heel_strikes, toe_offs, frame indices, confidence scores)
        - phase_durations, phase_names, cycle_count
        - cyclogram_metrics, symmetry_metrics, mmc_metrics
        - filter_settings (adaptive FFT-based cutoff, calibration offsets)
        - visualization params, hardware_info
        """
        timestamp = datetime.now().isoformat()

        # Extract file name if provided
        file_name = kwargs.get("file_name", f"{analysis_type}_{leg}_{timestamp}.png")

        # Initialize metadata with complete structure
        metadata = {
            "file_name": file_name,
            "timestamp": timestamp,
            "subject": kwargs.get("subject", "unknown"),
            "leg": leg,
            "analysis_type": analysis_type,
            "sensor_type": sensor_type,
            "phase_names": kwargs.get("phase_names", []),
            "cycle_count": kwargs.get("cycle_count", 0),

            "event_timings": {
                "heel_strikes": kwargs.get("heel_strikes", []),
                "toe_offs": kwargs.get("toe_offs", []),
                "start_idx": kwargs.get("start_idx", 0),
                "end_idx": kwargs.get("end_idx", 0),
                "start_frame": kwargs.get("start_frame", 0),
                "end_frame": kwargs.get("end_frame", 0),
                "confidence_scores": kwargs.get("confidence_scores", [])
            },

            "phase_durations": kwargs.get("phase_durations", {}),
            "cyclogram_metrics": kwargs.get("cyclogram_metrics", {}),
            "symmetry_metrics": kwargs.get("symmetry_metrics", {}),
            "mmc_metrics": kwargs.get("mmc_metrics", {}),

            "filter_settings": {
                "filter_type": "butterworth_lowpass_adaptive",
                "cutoff_frequency": kwargs.get("cutoff_frequency", self.config.filter_cutoff),
                "filter_order": self.config.filter_order,
                "adaptive_gain": kwargs.get("adaptive_gain", 1.0),
                "gait_frequency": kwargs.get("gait_frequency", 1.0),
                "calibration_offsets": kwargs.get("calibration_offsets", {})
            },

            "hardware_info": {
                "sampling_rate": self.config.sampling_rate,
                "sensor_count": kwargs.get("sensor_count", 4),
                "num_cycles": kwargs.get("num_cycles", 0)
            },

            "visualization": {
                "figure_size": [12, 6],
                "dpi": 300,
                "color_scheme": kwargs.get("color_scheme", "viridis")
            }
        }

        # Subplot-specific metadata (if applicable)
        if hasattr(self, '_current_layout') and self._current_layout:
            metadata['subplot_layout'] = self._current_layout
            metadata['subplot_data'] = kwargs.get('subplot_metrics', [])
            metadata['synchronization'] = {
                'stride_ids': kwargs.get('stride_ids', []),
                'phase_boundaries': kwargs.get('phase_boundaries', {}),
                'dominant_frequencies': kwargs.get('dominant_frequencies', {})
            }

        # Add detailed stride events with cyclogram data if provided
        if 'cycles' in kwargs and kwargs['cycles']:
            cycles = kwargs['cycles']
            stride_events = []

            # Also get cyclograms if available
            cyclograms_by_cycle = {}
            if 'cyclograms' in kwargs and kwargs['cyclograms']:
                for cyc in kwargs['cyclograms']:
                    if cyc and hasattr(cyc, 'cycle'):
                        key = (cyc.cycle.leg, cyc.cycle.cycle_id)
                        if key not in cyclograms_by_cycle:
                            cyclograms_by_cycle[key] = []
                        cyclograms_by_cycle[key].append(cyc)

            for cycle in cycles:
                stride_event = {
                    'cycle_id': cycle.cycle_id,
                    'leg': cycle.leg,
                    'start_time': float(cycle.start_time),
                    'end_time': float(cycle.end_time),
                    'duration': float(cycle.duration),
                    'stance_duration': float(cycle.stance_duration),
                    'swing_duration': float(cycle.swing_duration),
                    'stance_swing_ratio': float(cycle.stance_swing_ratio),
                    'phases': [],
                    'cyclograms': []
                }

                # Add detailed phase information
                for phase in cycle.phases:
                    phase_info = {
                        'phase_number': phase.phase_number,
                        'phase_name': phase.phase_name,
                        'support_type': phase.support_type,
                        'start_time': float(phase.start_time),
                        'duration': float(phase.duration),
                        'start_idx': int(phase.start_idx),
                        'end_idx': int(phase.end_idx)
                    }
                    stride_event['phases'].append(phase_info)

                # Add cyclogram data for this stride
                key = (cycle.leg, cycle.cycle_id)
                if key in cyclograms_by_cycle:
                    for cyclogram in cyclograms_by_cycle[key]:
                        cyclogram_data = {
                            'type': f"{cyclogram.x_label}_vs_{cyclogram.y_label}",
                            'x_label': cyclogram.x_label,
                            'y_label': cyclogram.y_label,
                            'is_3d': cyclogram.is_3d
                        }

                        # Add z_label for 3D cyclograms
                        if cyclogram.is_3d and cyclogram.z_label:
                            cyclogram_data['z_label'] = cyclogram.z_label

                        # Add metrics if available
                        if hasattr(cyclogram, 'x_signal') and cyclogram.x_signal is not None:
                            # Compute basic metrics
                            x, y = cyclogram.x_signal, cyclogram.y_signal
                            cyclogram_data['signal_length'] = len(x)
                            cyclogram_data['x_range'] = [float(np.min(x)), float(np.max(x))]
                            cyclogram_data['y_range'] = [float(np.min(y)), float(np.max(y))]

                            if cyclogram.is_3d and cyclogram.z_signal is not None:
                                z = cyclogram.z_signal
                                cyclogram_data['z_range'] = [float(np.min(z)), float(np.max(z))]

                            # Compute geometric metrics
                            try:
                                # Area using shoelace formula
                                area = 0.5 * np.abs(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))
                                cyclogram_data['area'] = float(area)

                                # Perimeter
                                perimeter = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
                                cyclogram_data['perimeter'] = float(perimeter)

                                # Closure error
                                closure_error = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
                                cyclogram_data['closure_error'] = float(closure_error)

                                # For 3D cyclograms, add 3D-specific metrics
                                if cyclogram.is_3d and cyclogram.z_signal is not None:
                                    trajectory_length_3d = np.sum(np.sqrt(
                                        np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2
                                    ))
                                    cyclogram_data['trajectory_length_3d'] = float(trajectory_length_3d)

                                    closure_error_3d = np.sqrt(
                                        (x[-1] - x[0])**2 + (y[-1] - y[0])**2 + (z[-1] - z[0])**2
                                    )
                                    cyclogram_data['closure_error_3d'] = float(closure_error_3d)
                            except Exception:
                                pass  # Skip metrics if computation fails

                        stride_event['cyclograms'].append(cyclogram_data)

                stride_events.append(stride_event)

            metadata['stride_events'] = stride_events

        # Add cyclogram-specific data
        if 'cyclogram' in kwargs:
            cyclogram = kwargs['cyclogram']
            metadata["event_timings"]["start_time"] = float(cyclogram.cycle.start_time)
            metadata["event_timings"]["end_time"] = float(cyclogram.cycle.end_time)
            metadata["event_timings"]["duration"] = float(cyclogram.cycle.duration)
            metadata["event_timings"]["start_idx"] = int(cyclogram.cycle.phases[0].start_idx)
            metadata["event_timings"]["end_idx"] = int(cyclogram.cycle.phases[-1].end_idx)

            metadata["phase_durations"] = {
                "stance": float(cyclogram.cycle.stance_duration),
                "swing": float(cyclogram.cycle.swing_duration),
                "stance_swing_ratio": float(cyclogram.cycle.stance_swing_ratio)
            }

            metadata["phase_names"] = [phase.phase_name for phase in cyclogram.cycle.phases]
            metadata["cycle_count"] = cyclogram.cycle.cycle_id

        # Add cyclogram metrics
        if 'cyclogram_metrics' in kwargs and kwargs['cyclogram_metrics']:
            metadata["cyclogram_metrics"] = kwargs['cyclogram_metrics']

        # Add MMC data
        if 'mmc' in kwargs and kwargs['mmc']:
            mmc = kwargs['mmc']
            metadata["mmc_metrics"] = {
                "n_loops": mmc.n_loops,
                "median_area": float(mmc.median_area),
                "shape_dispersion_index": float(mmc.shape_dispersion_index),
                "alignment_quality": float(mmc.alignment_quality),
                "confidence_ellipse": mmc.confidence_ellipse_params
            }

        # Add cycles data
        if 'cycles' in kwargs and kwargs['cycles'] is not None:
            cycles = kwargs['cycles']
            metadata["hardware_info"]["num_cycles"] = len(cycles)

        return metadata

    def build_subplot_grid(self, analysis_type: str, data_dict: Dict,
                          title: str = None, subject_name: str = "Unknown") -> Tuple[plt.Figure, np.ndarray]:
        """
        Auto-generate subplot grid based on analysis type.

        Layouts:
        - gyro_stride: 2x3 (L/R × XY/XZ/YZ) = 6 subplots
        - acc_stride: 2x3 = 6 subplots
        - 3d_stride: 2x2 (L/R × Gyro3D/Acc3D) = 4 subplots
        - gyro_gait: 2x3 (L/R × XY/XZ/YZ) = 6 subplots
        - acc_gait: 2x3 (L/R × XY/XZ/YZ) = 6 subplots
        - 3d_gait: 1x2 (Gyro3D/Acc3D) = 2 subplots
        - gait_events: 2x1 (L/R feet complete) = 2 subplots (stacked by foot zone)

        Returns:
            fig: Figure object (12×6 in @ 300 DPI)
            axes: Array of subplot axes
        """
        # Layout configurations
        layouts = {
            'gyro_stride': (2, 3, ["Left X-Y", "Left X-Z", "Left Y-Z", "Right X-Y", "Right X-Z", "Right Y-Z"]),
            'acc_stride': (2, 3, ["Left X-Y", "Left X-Z", "Left Y-Z", "Right X-Y", "Right X-Z", "Right Y-Z"]),
            '3d_stride': (2, 2, ["Left Gyro 3D", "Left Acc 3D", "Right Gyro 3D", "Right Acc 3D"]),
            'gyro_gait': (2, 3, ["Left X-Y", "Left X-Z", "Left Y-Z", "Right X-Y", "Right X-Z", "Right Y-Z"]),
            'acc_gait': (2, 3, ["Left X-Y", "Left X-Z", "Left Y-Z", "Right X-Y", "Right X-Z", "Right Y-Z"]),
            'gait_events': (2, 1, ["Left Foot Complete", "Right Foot Complete"])
        }

        # Dynamic layout for 3d_gait based on GYRO availability
        if analysis_type == '3d_gait':
            if self.has_real_gyro:
                nrows, ncols, subplot_titles = (1, 2, ["Gyroscope 3D", "Accelerometer 3D"])
            else:
                nrows, ncols, subplot_titles = (1, 1, ["Accelerometer 3D"])
        elif analysis_type not in layouts:
            raise ValueError(f"Unknown analysis_type: {analysis_type}")
        else:
            nrows, ncols, subplot_titles = layouts[analysis_type]

        # Create figure with standard dimensions
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 10), dpi=300)

        # Ensure axes is always 2D array for consistent indexing
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1 or ncols == 1:
            axes = axes.reshape(nrows, ncols)

        # Set main title
        if title:
            fig.suptitle(f"{subject_name} - {title}", fontsize=16, fontweight='bold', y=0.98)

        # Store layout info for metadata
        self._current_layout = {
            'analysis_type': analysis_type,
            'grid_shape': [nrows, ncols],
            'subplot_titles': subplot_titles,
            'subplot_count': nrows * ncols
        }

        return fig, axes

    def _plot_gyro_stride_subplots(self, axes: np.ndarray, data_dict: Dict) -> List[Dict]:
        """
        Populate 2x3 grid with gyroscopic cyclograms (stride level).

        Layout: Left (row 0) and Right (row 1) × XY/XZ/YZ (cols 0-2)
        """
        subplot_metrics = []
        sensor_pairs = [('GYRO_X', 'GYRO_Y'), ('GYRO_X', 'GYRO_Z'), ('GYRO_Y', 'GYRO_Z')]
        pair_labels = ['X-Y Plane', 'X-Z Plane', 'Y-Z Plane']

        for leg_idx, leg in enumerate(['left', 'right']):
            leg_data = data_dict.get(leg, {})

            for col_idx, (pair, label) in enumerate(zip(sensor_pairs, pair_labels)):
                ax = axes[leg_idx, col_idx]
                cyclogram = leg_data.get(label)

                if cyclogram is None:
                    ax.text(0.5, 0.5, f'No data for {leg.title()} {label}',
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"{leg.title()} {label}", fontsize=10, fontweight='bold')
                    subplot_metrics.append({'subplot_index': leg_idx * 3 + col_idx, 'data': None})
                    continue

                # Plot cyclogram with phase segmentation
                x, y = cyclogram.x_signal, cyclogram.y_signal
                phase_indices = cyclogram.phase_indices + [len(x)]

                for i in range(len(cyclogram.phase_indices)):
                    start_idx = phase_indices[i]
                    end_idx = phase_indices[i + 1]
                    color = self.phase_colors[i % len(self.phase_colors)]
                    # Add legend labels to first column of each row (both left and right legs)
                    ax.plot(x[start_idx:end_idx], y[start_idx:end_idx],
                           color=color, linewidth=1.5, label=cyclogram.phase_labels[i] if col_idx == 0 else "")

                # Add phase boundary markers for clearer sub-phase divisions
                self._add_phase_boundary_markers(ax, x, y, cyclogram.phase_indices,
                                                 cyclogram.phase_labels, marker_size=4)

                # Mark start/end
                ax.plot(x[0], y[0], 'go', markersize=6)
                ax.plot(x[-1], y[-1], 'rs', markersize=6)

                # Annotations
                ax.set_xlabel(pair[0], fontsize=8)
                ax.set_ylabel(pair[1], fontsize=8)
                ax.set_title(f"{leg.title()} {label}", fontsize=10, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal', adjustable='box')

                # Compute metrics
                metrics = self._compute_cyclogram_metrics(cyclogram)
                subplot_metrics.append({
                    'subplot_index': leg_idx * 3 + col_idx,
                    'position': [leg_idx, col_idx],
                    'title': f"{leg.title()} Gyro {label}",
                    'leg': leg,
                    'sensor_pair': list(pair),
                    'metrics': metrics
                })

        # Add legend to first subplot of each row (both left and right legs)
        if axes[0, 0].get_lines():
            axes[0, 0].legend(loc='upper right', fontsize=6, ncol=2)
        if axes[1, 0].get_lines():
            axes[1, 0].legend(loc='upper right', fontsize=6, ncol=2)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return subplot_metrics

    def _plot_acc_stride_subplots(self, axes: np.ndarray, data_dict: Dict) -> List[Dict]:
        """
        Populate 2x3 grid with accelerometer cyclograms (stride level).
        Same structure as gyro but for ACC_X, ACC_Y, ACC_Z sensors.
        """
        subplot_metrics = []
        sensor_pairs = [('ACC_X', 'ACC_Y'), ('ACC_X', 'ACC_Z'), ('ACC_Y', 'ACC_Z')]
        pair_labels = ['X-Y Plane', 'X-Z Plane', 'Y-Z Plane']

        # Similar implementation to _plot_gyro_stride_subplots but for accelerometer data
        for leg_idx, leg in enumerate(['left', 'right']):
            leg_data = data_dict.get(leg, {})

            for col_idx, (pair, label) in enumerate(zip(sensor_pairs, pair_labels)):
                ax = axes[leg_idx, col_idx]
                cyclogram = leg_data.get(label)

                if cyclogram is None:
                    ax.text(0.5, 0.5, f'No data for {leg.title()} {label}',
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"{leg.title()} {label}", fontsize=10, fontweight='bold')
                    subplot_metrics.append({'subplot_index': leg_idx * 3 + col_idx, 'data': None})
                    continue

                # Plot cyclogram
                x, y = cyclogram.x_signal, cyclogram.y_signal
                phase_indices = cyclogram.phase_indices + [len(x)]

                for i in range(len(cyclogram.phase_indices)):
                    start_idx = phase_indices[i]
                    end_idx = phase_indices[i + 1]
                    color = self.phase_colors[i % len(self.phase_colors)]
                    # Add legend labels to first column of each row (both left and right legs)
                    ax.plot(x[start_idx:end_idx], y[start_idx:end_idx],
                           color=color, linewidth=1.5, label=cyclogram.phase_labels[i] if col_idx == 0 else "")

                # Add phase boundary markers for clearer sub-phase divisions
                self._add_phase_boundary_markers(ax, x, y, cyclogram.phase_indices,
                                                 cyclogram.phase_labels, marker_size=4)

                ax.plot(x[0], y[0], 'go', markersize=6)
                ax.plot(x[-1], y[-1], 'rs', markersize=6)

                ax.set_xlabel(pair[0], fontsize=8)
                ax.set_ylabel(pair[1], fontsize=8)
                ax.set_title(f"{leg.title()} {label}", fontsize=10, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal', adjustable='box')

                metrics = self._compute_cyclogram_metrics(cyclogram)
                subplot_metrics.append({
                    'subplot_index': leg_idx * 3 + col_idx,
                    'position': [leg_idx, col_idx],
                    'title': f"{leg.title()} Acc {label}",
                    'leg': leg,
                    'sensor_pair': list(pair),
                    'metrics': metrics
                })

        # Add legend to first subplot of each row (both left and right legs)
        if axes[0, 0].get_lines():
            axes[0, 0].legend(loc='upper right', fontsize=6, ncol=2)
        if axes[1, 0].get_lines():
            axes[1, 0].legend(loc='upper right', fontsize=6, ncol=2)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return subplot_metrics

    def _plot_gait_events_subplots(self, axes: np.ndarray, data_dict: Dict) -> List[Dict]:
        """
        Populate 2x1 grid with gait event timelines showing foot zone grouping.

        Layout: 2 rows (left foot top, right foot bottom) × 1 column
        Each subplot shows all 4 pressure sensors grouped by foot zone:
        - Forefoot: Sensor 2
        - Midfoot: Sensors 1, 3
        - Hindfoot: Sensor 4

        Shows heel strikes, toe-offs, and 8-phase boundaries.
        """
        subplot_metrics = []

        # Extract DataFrame and events from data_dict
        df = data_dict.get('df', None)
        left_events = data_dict.get('left', {})
        right_events = data_dict.get('right', {})

        # Define sensor grouping by foot zone
        # User specified: forefoot=2, midfoot=1,3, hindfoot=4
        foot_zones = {
            'Forefoot': [2],      # Sensor 2
            'Midfoot': [1, 3],    # Sensors 1, 3
            'Hindfoot': [4]       # Sensor 4
        }

        # Sensor column names
        sensor_columns = {
            'left': {1: 'L_value1', 2: 'L_value2', 3: 'L_value3', 4: 'L_value4'},
            'right': {1: 'R_value1', 2: 'R_value2', 3: 'R_value3', 4: 'R_value4'}
        }

        # Colors for each zone
        zone_colors = {
            'Forefoot': '#1f77b4',  # blue
            'Midfoot': '#ff7f0e',   # orange
            'Hindfoot': '#2ca02c'   # green
        }

        # If no DataFrame, display error
        if df is None:
            for leg_idx, leg in enumerate(['left', 'right']):
                ax = axes[leg_idx, 0] if axes.ndim > 1 else axes[leg_idx]
                ax.text(0.5, 0.5, f'No pressure data available for {leg.title()} foot',
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title(f"{leg.title()} Foot Complete", fontsize=11, fontweight='bold')
                subplot_metrics.append({
                    'subplot_index': leg_idx,
                    'position': [leg_idx, 0],
                    'leg': leg,
                    'data': None
                })
            return subplot_metrics

        # Get time column
        time_col = 'time' if 'time' in df.columns else df.columns[0]
        time_data = df[time_col].values

        # Plot each foot (left=top, right=bottom)
        for leg_idx, leg in enumerate(['left', 'right']):
            ax = axes[leg_idx, 0] if axes.ndim > 1 else axes[leg_idx]
            events = left_events if leg == 'left' else right_events

            heel_strikes = events.get('heel_strikes', [])
            toe_offs = events.get('toe_offs', [])
            phases = events.get('phases', [])

            # Prepare stacked plot data
            all_zone_data = []
            zone_labels = []
            y_offset = 0
            y_ticks = []
            y_tick_labels = []

            # Plot each foot zone (stacked vertically within the subplot)
            for zone_name, sensor_ids in foot_zones.items():
                zone_color = zone_colors[zone_name]

                for sensor_id in sensor_ids:
                    sensor_col = sensor_columns[leg][sensor_id]

                    if sensor_col in df.columns:
                        pressure_data = df[sensor_col].values

                        # Normalize and offset for stacking
                        if len(pressure_data) > 0 and pressure_data.max() > 0:
                            normalized = pressure_data / pressure_data.max() if pressure_data.max() > 0 else pressure_data
                        else:
                            normalized = pressure_data

                        # Plot pressure timeline with offset
                        ax.plot(time_data, normalized + y_offset, color=zone_color,
                               linewidth=1.0, alpha=0.8, label=f"{zone_name} S{sensor_id}" if sensor_id == sensor_ids[0] else "")
                        ax.fill_between(time_data, y_offset, normalized + y_offset,
                                       color=zone_color, alpha=0.15)

                        # Store tick position and label
                        y_ticks.append(y_offset + 0.5)
                        y_tick_labels.append(f"S{sensor_id}\n{zone_name}")

                        y_offset += 1.2  # Stack spacing

            # Plot phase backgrounds
            added_to_legend = set()
            for phase in phases:
                start_t = phase.get('start_time', 0)
                end_t = phase.get('end_time', 0)
                phase_name = phase.get('phase_name', '')
                phase_idx = phase.get('phase_number', 0) - 1

                color = self.phase_colors[phase_idx % len(self.phase_colors)]

                # Only add to legend for left foot (top subplot)
                label = phase_name if (leg_idx == 0 and phase_name not in added_to_legend) else ""
                if label:
                    added_to_legend.add(phase_name)

                ax.axvspan(start_t, end_t, alpha=0.1, color=color, label=label, zorder=0)

            # Overlay heel strikes and toe-offs
            for hs_time in heel_strikes:
                ax.axvline(hs_time, color='green', linestyle='--', linewidth=1.5, alpha=0.7, zorder=2)
            for to_time in toe_offs:
                ax.axvline(to_time, color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=2)

            # Add event markers at top
            if heel_strikes:
                ax.scatter(heel_strikes, [y_offset]*len(heel_strikes),
                         marker='v', s=80, c='green', label='Heel Strike', zorder=3, edgecolors='darkgreen', linewidths=1.5)
            if toe_offs:
                ax.scatter(toe_offs, [y_offset]*len(toe_offs),
                         marker='^', s=80, c='red', label='Toe-Off', zorder=3, edgecolors='darkred', linewidths=1.5)

            # Formatting
            ax.set_title(f"{leg.title()} Foot Complete (Grouped by Zone)", fontsize=10, fontweight='bold')
            ax.set_xlabel('Time (s)', fontsize=8)
            ax.set_ylabel('Normalized Pressure by Foot Zone', fontsize=8)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_tick_labels, fontsize=6)
            ax.set_ylim(-0.2, y_offset + 0.5)
            ax.grid(True, alpha=0.2, axis='x')
            ax.tick_params(axis='x', labelsize=7)

            # Add legend only to top subplot
            if leg_idx == 0:
                ax.legend(loc='upper right', fontsize=6, ncol=3, framealpha=0.9)

            subplot_metrics.append({
                'subplot_index': leg_idx,
                'position': [leg_idx, 0],
                'leg': leg,
                'foot_zones': list(foot_zones.keys()),
                'event_count': len(heel_strikes) + len(toe_offs),
                'heel_strikes': len(heel_strikes),
                'toe_offs': len(toe_offs)
            })

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return subplot_metrics

    def _plot_3d_stride_subplots(self, axes: np.ndarray, data_dict: Dict) -> List[Dict]:
        """
        Populate 2x2 grid with 3D cyclograms (stride level).

        Layout:
        - Row 0: Left Gyro 3D, Left Acc 3D
        - Row 1: Right Gyro 3D, Right Acc 3D
        """
        from mpl_toolkits.mplot3d import Axes3D
        subplot_metrics = []

        sensor_types = ['gyro', 'acc']
        sensor_labels = ['Gyroscope 3D', 'Accelerometer 3D']

        for leg_idx, leg in enumerate(['left', 'right']):
            leg_data = data_dict.get(leg, {})

            for col_idx, (sensor_type, label) in enumerate(zip(sensor_types, sensor_labels)):
                # Remove old 2D axis and create 3D axis
                ax = plt.subplot(2, 2, leg_idx * 2 + col_idx + 1, projection='3d')

                cyclogram_3d = leg_data.get(f'{sensor_type}_3d')

                if cyclogram_3d is None or not hasattr(cyclogram_3d, 'x_signal'):
                    ax.text2D(0.5, 0.5, f'No 3D data for {leg.title()} {label}',
                            ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"{leg.title()} {label}", fontsize=10, fontweight='bold')
                    subplot_metrics.append({'subplot_index': leg_idx * 2 + col_idx, 'data': None})
                    continue

                # Get 3D trajectory
                x = cyclogram_3d.x_signal  # Sensor X
                y = cyclogram_3d.y_signal  # Sensor Y
                z = cyclogram_3d.z_signal if hasattr(cyclogram_3d, 'z_signal') else np.zeros_like(x)

                # Plot with phase colors
                phase_indices = cyclogram_3d.phase_indices + [len(x)]

                for i in range(len(cyclogram_3d.phase_indices)):
                    start_idx = phase_indices[i]
                    end_idx = phase_indices[i + 1]
                    color = self.phase_colors[i % len(self.phase_colors)]
                    ax.plot(x[start_idx:end_idx], y[start_idx:end_idx], z[start_idx:end_idx],
                           color=color, linewidth=1.5)

                # Mark start/end
                ax.scatter(x[0], y[0], z[0], c='green', s=50, marker='o')
                ax.scatter(x[-1], y[-1], z[-1], c='red', s=50, marker='s')

                # Labels
                if sensor_type == 'gyro':
                    ax.set_xlabel('Gyro X', fontsize=7)
                    ax.set_ylabel('Gyro Y', fontsize=7)
                    ax.set_zlabel('Gyro Z', fontsize=7)
                else:
                    ax.set_xlabel('Acc X', fontsize=7)
                    ax.set_ylabel('Acc Y', fontsize=7)
                    ax.set_zlabel('Acc Z', fontsize=7)

                ax.set_title(f"{leg.title()} {label}", fontsize=10, fontweight='bold')

                # Compute 3D metrics
                trajectory_length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2))

                subplot_metrics.append({
                    'subplot_index': leg_idx * 2 + col_idx,
                    'position': [leg_idx, col_idx],
                    'title': f"{leg.title()} {label}",
                    'leg': leg,
                    'sensor_type': sensor_type,
                    'metrics': {
                        'trajectory_length': float(trajectory_length),
                        'closure_error_3d': float(np.sqrt((x[-1]-x[0])**2 + (y[-1]-y[0])**2 + (z[-1]-z[0])**2))
                    }
                })

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return subplot_metrics

    def _plot_gyro_gait_subplots(self, axes: np.ndarray, data_dict: Dict) -> List[Dict]:
        """
        Populate 2x3 grid with gyroscopic cyclograms (gait level - Left vs Right comparison).

        Layout (2 rows x 3 columns):
        Row 0 (Left Leg):  X-Y | X-Z | Y-Z
        Row 1 (Right Leg): X-Y | X-Z | Y-Z

        Each subplot shows all cycles (semi-transparent with phase colors) + mean + ±SD envelope.
        """
        subplot_metrics = []
        sensor_pairs = [('GYRO_X', 'GYRO_Y'), ('GYRO_X', 'GYRO_Z'), ('GYRO_Y', 'GYRO_Z')]
        pair_labels = ['GYRO_X-Y Plane', 'GYRO_X-Z Plane', 'GYRO_Y-Z Plane']
        display_labels = ['X-Y Plane', 'X-Z Plane', 'Y-Z Plane']

        for leg_idx, leg in enumerate(['left', 'right']):
            leg_color = 'blue' if leg == 'left' else 'red'
            leg_data = data_dict.get(leg, {})

            for col_idx, (pair, label, disp_label) in enumerate(zip(sensor_pairs, pair_labels, display_labels)):
                ax = axes[leg_idx, col_idx]

                # Get cyclogram data for this leg and plane (with GYRO_ prefix)
                plane_data = leg_data.get(label, [])

                if isinstance(plane_data, list) and len(plane_data) > 0:
                    # Plot all individual cycles with phase segmentation (semi-transparent)
                    for cyclogram in plane_data:
                        if hasattr(cyclogram, 'x_signal'):
                            # Check if phase information is available
                            if hasattr(cyclogram, 'phase_indices') and cyclogram.phase_indices:
                                x, y = cyclogram.x_signal, cyclogram.y_signal
                                phase_indices = cyclogram.phase_indices + [len(x)]

                                # Plot each phase segment with distinct color
                                for i in range(len(cyclogram.phase_indices)):
                                    start_idx = phase_indices[i]
                                    end_idx = phase_indices[i + 1]
                                    color = self.phase_colors[i % len(self.phase_colors)]
                                    ax.plot(x[start_idx:end_idx], y[start_idx:end_idx],
                                           color=color, linewidth=0.8, alpha=0.25)
                            else:
                                # Fallback: single-color plot if no phase data
                                ax.plot(cyclogram.x_signal, cyclogram.y_signal,
                                       color=leg_color, linewidth=0.5, alpha=0.2)

                    # Compute and overlay mean with envelope
                    if len(plane_data) > 1:
                        mmc = self.mmc_computer.compute_mmc(plane_data)
                        if mmc:
                            ax.plot(mmc.median_x, mmc.median_y,
                                   color=leg_color, linewidth=2.5,
                                   label=f'Mean (n={mmc.n_loops})', alpha=0.9)
                            ax.fill_between(mmc.median_x,
                                           mmc.variance_envelope_lower[:, 1],
                                           mmc.variance_envelope_upper[:, 1],
                                           alpha=0.15, color=leg_color, label='±SD')
                            ax.plot(mmc.median_x[0], mmc.median_y[0],
                                   color=leg_color, marker='o', markersize=8)
                            # Add phase boundary markers on mean cyclogram (use first cycle's phase info)
                            if plane_data and hasattr(plane_data[0], 'phase_indices'):
                                self._add_phase_boundary_markers(ax, mmc.median_x, mmc.median_y,
                                                                 plane_data[0].phase_indices,
                                                                 plane_data[0].phase_labels,
                                                                 marker_size=3)
                    else:
                        # Single cycle - plot bold
                        ax.plot(plane_data[0].x_signal, plane_data[0].y_signal,
                               color=leg_color, linewidth=2.5, label='Single Cycle', alpha=0.9)
                        ax.plot(plane_data[0].x_signal[0], plane_data[0].y_signal[0],
                               color=leg_color, marker='o', markersize=8)
                else:
                    ax.text(0.5, 0.5, 'No Data',
                           ha='center', va='center', transform=ax.transAxes, fontsize=10)

                ax.set_xlabel(pair[0], fontsize=8)
                ax.set_ylabel(pair[1], fontsize=8)
                ax.set_title(f"{leg.title()} - {disp_label}", fontsize=9, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal', adjustable='box')
                ax.legend(loc='best', fontsize=6, ncol=2)

                # Serialize cyclogram data for JSON export
                cycles_data = []
                if isinstance(plane_data, list):
                    for cyc in plane_data:
                        if hasattr(cyc, 'x_signal') and hasattr(cyc, 'y_signal'):
                            cycles_data.append({
                                'x_signal': cyc.x_signal.tolist() if hasattr(cyc.x_signal, 'tolist') else list(cyc.x_signal),
                                'y_signal': cyc.y_signal.tolist() if hasattr(cyc.y_signal, 'tolist') else list(cyc.y_signal)
                            })

                subplot_metrics.append({
                    'subplot_index': leg_idx * 3 + col_idx,
                    'position': [leg_idx, col_idx],
                    'title': f"{leg.title()} Gyro {disp_label}",
                    'leg': leg,
                    'sensor_pair': list(pair),
                    'cycle_count': len(cycles_data),
                    'cycles': cycles_data
                })

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return subplot_metrics

    def _plot_acc_gait_subplots(self, axes: np.ndarray, data_dict: Dict) -> List[Dict]:
        """
        Populate 2x3 grid with accelerometer cyclograms (gait level - Left vs Right comparison).

        Layout (2 rows x 3 columns):
        Row 0 (Left Leg):  X-Y | X-Z | Y-Z
        Row 1 (Right Leg): X-Y | X-Z | Y-Z

        Each subplot shows all cycles (semi-transparent with phase colors) + mean + ±SD envelope.
        """
        subplot_metrics = []
        sensor_pairs = [('ACC_X', 'ACC_Y'), ('ACC_X', 'ACC_Z'), ('ACC_Y', 'ACC_Z')]
        pair_labels = ['ACC_X-Y Plane', 'ACC_X-Z Plane', 'ACC_Y-Z Plane']
        display_labels = ['X-Y Plane', 'X-Z Plane', 'Y-Z Plane']

        for leg_idx, leg in enumerate(['left', 'right']):
            leg_color = 'blue' if leg == 'left' else 'red'
            leg_data = data_dict.get(leg, {})

            for col_idx, (pair, label, disp_label) in enumerate(zip(sensor_pairs, pair_labels, display_labels)):
                ax = axes[leg_idx, col_idx]

                # Get cyclogram data for this leg and plane (with ACC_ prefix)
                plane_data = leg_data.get(label, [])

                if isinstance(plane_data, list) and len(plane_data) > 0:
                    # Plot all individual cycles with phase segmentation (semi-transparent)
                    for cyclogram in plane_data:
                        if hasattr(cyclogram, 'x_signal'):
                            # Check if phase information is available
                            if hasattr(cyclogram, 'phase_indices') and cyclogram.phase_indices:
                                x, y = cyclogram.x_signal, cyclogram.y_signal
                                phase_indices = cyclogram.phase_indices + [len(x)]

                                # Plot each phase segment with distinct color
                                for i in range(len(cyclogram.phase_indices)):
                                    start_idx = phase_indices[i]
                                    end_idx = phase_indices[i + 1]
                                    color = self.phase_colors[i % len(self.phase_colors)]
                                    ax.plot(x[start_idx:end_idx], y[start_idx:end_idx],
                                           color=color, linewidth=0.8, alpha=0.25)
                            else:
                                # Fallback: single-color plot if no phase data
                                ax.plot(cyclogram.x_signal, cyclogram.y_signal,
                                       color=leg_color, linewidth=0.5, alpha=0.2)

                    # Compute and overlay mean with envelope
                    if len(plane_data) > 1:
                        mmc = self.mmc_computer.compute_mmc(plane_data)
                        if mmc:
                            ax.plot(mmc.median_x, mmc.median_y,
                                   color=leg_color, linewidth=2.5,
                                   label=f'Mean (n={mmc.n_loops})', alpha=0.9)
                            ax.fill_between(mmc.median_x,
                                           mmc.variance_envelope_lower[:, 1],
                                           mmc.variance_envelope_upper[:, 1],
                                           alpha=0.15, color=leg_color, label='±SD')
                            ax.plot(mmc.median_x[0], mmc.median_y[0],
                                   color=leg_color, marker='o', markersize=8)
                            # Add phase boundary markers on mean cyclogram (use first cycle's phase info)
                            if plane_data and hasattr(plane_data[0], 'phase_indices'):
                                self._add_phase_boundary_markers(ax, mmc.median_x, mmc.median_y,
                                                                 plane_data[0].phase_indices,
                                                                 plane_data[0].phase_labels,
                                                                 marker_size=3)
                    else:
                        # Single cycle - plot bold
                        ax.plot(plane_data[0].x_signal, plane_data[0].y_signal,
                               color=leg_color, linewidth=2.5, label='Single Cycle', alpha=0.9)
                        ax.plot(plane_data[0].x_signal[0], plane_data[0].y_signal[0],
                               color=leg_color, marker='o', markersize=8)
                else:
                    ax.text(0.5, 0.5, 'No Data',
                           ha='center', va='center', transform=ax.transAxes, fontsize=10)

                ax.set_xlabel(pair[0], fontsize=8)
                ax.set_ylabel(pair[1], fontsize=8)
                ax.set_title(f"{leg.title()} - {disp_label}", fontsize=9, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal', adjustable='box')
                ax.legend(loc='best', fontsize=6, ncol=2)

                # Serialize cyclogram data for JSON export
                cycles_data = []
                if isinstance(plane_data, list):
                    for cyc in plane_data:
                        if hasattr(cyc, 'x_signal') and hasattr(cyc, 'y_signal'):
                            cycles_data.append({
                                'x_signal': cyc.x_signal.tolist() if hasattr(cyc.x_signal, 'tolist') else list(cyc.x_signal),
                                'y_signal': cyc.y_signal.tolist() if hasattr(cyc.y_signal, 'tolist') else list(cyc.y_signal)
                            })

                subplot_metrics.append({
                    'subplot_index': leg_idx * 3 + col_idx,
                    'position': [leg_idx, col_idx],
                    'title': f"{leg.title()} Acc {disp_label}",
                    'leg': leg,
                    'sensor_pair': list(pair),
                    'cycle_count': len(cycles_data),
                    'cycles': cycles_data
                })

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return subplot_metrics

    def _plot_3d_gait_subplots(self, axes: np.ndarray, data_dict: Dict) -> List[Dict]:
        """
        Populate 1x2 grid with 3D cyclograms (gait level - combined L+R).

        Layout: Gyro 3D | Acc 3D (both with left and right overlaid)
        Shows all individual cycles (semi-transparent) + mean trajectory (bold).
        NOTE: Gyro 3D subplot only created if has_real_gyro=True
        """
        from mpl_toolkits.mplot3d import Axes3D
        subplot_metrics = []

        # Only include GYRO if real GYRO data exists
        sensor_types = []
        sensor_labels = []
        if self.has_real_gyro:
            sensor_types.append('gyro')
            sensor_labels.append('Gyroscope 3D')
        sensor_types.append('acc')
        sensor_labels.append('Accelerometer 3D')

        for col_idx, (sensor_type, label) in enumerate(zip(sensor_types, sensor_labels)):
            # Create 3D axis
            ax = plt.subplot(1, 2, col_idx + 1, projection='3d')

            # Plot left leg (blue)
            left_3d_data = data_dict.get('left', {}).get(f'{sensor_type}_3d')
            if isinstance(left_3d_data, list) and len(left_3d_data) > 0:
                # Plot all individual cycles (semi-transparent)
                for cyclogram in left_3d_data:
                    if hasattr(cyclogram, 'x_signal') and cyclogram.z_signal is not None:
                        ax.plot(cyclogram.x_signal, cyclogram.y_signal, cyclogram.z_signal,
                               color='blue', linewidth=0.5, alpha=0.2)

                # Compute and plot mean trajectory (bold)
                if len(left_3d_data) > 1:
                    x_mean = np.mean([c.x_signal for c in left_3d_data], axis=0)
                    y_mean = np.mean([c.y_signal for c in left_3d_data], axis=0)
                    z_mean = np.mean([c.z_signal for c in left_3d_data], axis=0)
                    ax.plot(x_mean, y_mean, z_mean,
                           color='blue', linewidth=2.5, label=f'Left Mean (n={len(left_3d_data)})', alpha=0.9)
                    ax.scatter(x_mean[0], y_mean[0], z_mean[0], c='blue', s=80, marker='o')
                else:
                    # Single cycle - plot bold
                    cyclogram = left_3d_data[0]
                    ax.plot(cyclogram.x_signal, cyclogram.y_signal, cyclogram.z_signal,
                           color='blue', linewidth=2.5, label='Left', alpha=0.9)
                    ax.scatter(cyclogram.x_signal[0], cyclogram.y_signal[0], cyclogram.z_signal[0],
                              c='blue', s=80, marker='o')
            elif left_3d_data and hasattr(left_3d_data, 'x_signal'):
                # Single cyclogram (not in list)
                ax.plot(left_3d_data.x_signal, left_3d_data.y_signal,
                       left_3d_data.z_signal if hasattr(left_3d_data, 'z_signal') else np.zeros_like(left_3d_data.x_signal),
                       color='blue', linewidth=2, label='Left', alpha=0.7)

            # Plot right leg (red)
            right_3d_data = data_dict.get('right', {}).get(f'{sensor_type}_3d')
            if isinstance(right_3d_data, list) and len(right_3d_data) > 0:
                # Plot all individual cycles (semi-transparent)
                for cyclogram in right_3d_data:
                    if hasattr(cyclogram, 'x_signal') and cyclogram.z_signal is not None:
                        ax.plot(cyclogram.x_signal, cyclogram.y_signal, cyclogram.z_signal,
                               color='red', linewidth=0.5, alpha=0.2)

                # Compute and plot mean trajectory (bold)
                if len(right_3d_data) > 1:
                    x_mean = np.mean([c.x_signal for c in right_3d_data], axis=0)
                    y_mean = np.mean([c.y_signal for c in right_3d_data], axis=0)
                    z_mean = np.mean([c.z_signal for c in right_3d_data], axis=0)
                    ax.plot(x_mean, y_mean, z_mean,
                           color='red', linewidth=2.5, label=f'Right Mean (n={len(right_3d_data)})', alpha=0.9)
                    ax.scatter(x_mean[0], y_mean[0], z_mean[0], c='red', s=80, marker='o')
                else:
                    # Single cycle - plot bold
                    cyclogram = right_3d_data[0]
                    ax.plot(cyclogram.x_signal, cyclogram.y_signal, cyclogram.z_signal,
                           color='red', linewidth=2.5, label='Right', alpha=0.9)
                    ax.scatter(cyclogram.x_signal[0], cyclogram.y_signal[0], cyclogram.z_signal[0],
                              c='red', s=80, marker='o')
            elif right_3d_data and hasattr(right_3d_data, 'x_signal'):
                # Single cyclogram (not in list)
                ax.plot(right_3d_data.x_signal, right_3d_data.y_signal,
                       right_3d_data.z_signal if hasattr(right_3d_data, 'z_signal') else np.zeros_like(right_3d_data.x_signal),
                       color='red', linewidth=2, label='Right', alpha=0.7)

            # Labels
            if sensor_type == 'gyro':
                ax.set_xlabel('Gyro X', fontsize=8)
                ax.set_ylabel('Gyro Y', fontsize=8)
                ax.set_zlabel('Gyro Z', fontsize=8)
            else:
                ax.set_xlabel('Acc X', fontsize=8)
                ax.set_ylabel('Acc Y', fontsize=8)
                ax.set_zlabel('Acc Z', fontsize=8)

            ax.set_title(label, fontsize=10, fontweight='bold')
            ax.legend(loc='best', fontsize=8)

            # Serialize cyclogram data for JSON export
            left_cycles_data = []
            if isinstance(left_3d_data, list):
                for cyc in left_3d_data:
                    if hasattr(cyc, 'x_signal') and hasattr(cyc, 'y_signal'):
                        left_cycles_data.append({
                            'x_signal': cyc.x_signal.tolist() if hasattr(cyc.x_signal, 'tolist') else list(cyc.x_signal),
                            'y_signal': cyc.y_signal.tolist() if hasattr(cyc.y_signal, 'tolist') else list(cyc.y_signal),
                            'z_signal': cyc.z_signal.tolist() if hasattr(cyc, 'z_signal') and cyc.z_signal is not None and hasattr(cyc.z_signal, 'tolist') else (list(cyc.z_signal) if hasattr(cyc, 'z_signal') and cyc.z_signal is not None else [0]*101)
                        })

            right_cycles_data = []
            if isinstance(right_3d_data, list):
                for cyc in right_3d_data:
                    if hasattr(cyc, 'x_signal') and hasattr(cyc, 'y_signal'):
                        right_cycles_data.append({
                            'x_signal': cyc.x_signal.tolist() if hasattr(cyc.x_signal, 'tolist') else list(cyc.x_signal),
                            'y_signal': cyc.y_signal.tolist() if hasattr(cyc.y_signal, 'tolist') else list(cyc.y_signal),
                            'z_signal': cyc.z_signal.tolist() if hasattr(cyc, 'z_signal') and cyc.z_signal is not None and hasattr(cyc.z_signal, 'tolist') else (list(cyc.z_signal) if hasattr(cyc, 'z_signal') and cyc.z_signal is not None else [0]*101)
                        })

            subplot_metrics.append({
                'subplot_index': col_idx,
                'position': [0, col_idx],
                'title': label,
                'sensor_type': sensor_type,
                'bilateral': True,
                'left_cycle_count': len(left_cycles_data),
                'left_cycles': left_cycles_data,
                'right_cycle_count': len(right_cycles_data),
                'right_cycles': right_cycles_data
            })

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return subplot_metrics

    def _compute_cyclogram_metrics(self, cyclogram: 'CyclogramData') -> Dict:
        """
        Compute comprehensive cyclogram metrics (basic + advanced).

        Integrates:
        - Basic metrics: area, perimeter, closure error
        - Level 1 (Geometric): compactness, aspect ratio, eccentricity, curvature, smoothness
        - Level 2 (Temporal): phase coupling, CRP, MARP, coupling angle variability
        """
        x, y = cyclogram.x_signal, cyclogram.y_signal

        # ===== BASIC METRICS (existing) =====
        # Area (shoelace formula)
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

        # Perimeter
        dx = np.diff(x)
        dy = np.diff(y)
        perimeter = np.sum(np.sqrt(dx**2 + dy**2))

        # Compactness: 4π*area / perimeter²
        compactness = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0

        # Closure error
        closure_error = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)

        basic_metrics = {
            'area': float(area),
            'perimeter': float(perimeter),
            'compactness': float(compactness),
            'closure_error': float(closure_error)
        }

        # ===== ADVANCED METRICS (new) =====
        try:
            advanced_computer = AdvancedCyclogramMetrics(cyclogram)

            # Level 1: Geometric/Morphology
            geometric_metrics = advanced_computer.compute_geometric_metrics()

            # Level 2: Temporal-Coupling
            temporal_metrics = advanced_computer.compute_temporal_metrics()

            # Combine all metrics
            all_metrics = {
                **basic_metrics,
                **geometric_metrics,
                **temporal_metrics
            }

            return all_metrics

        except Exception as e:
            # Fallback to basic metrics if advanced computation fails
            print(f"Warning: Advanced metrics computation failed: {e}")
            return basic_metrics

    def create_and_populate_subplot_figure(self, analysis_type: str, data_dict: Dict,
                                           subject_name: str = "Unknown") -> Tuple[plt.Figure, Dict, str]:
        """
        Complete workflow: create grid → populate subplots → generate metadata.

        Supports all 7 analysis types:
        - gyro_stride, acc_stride, 3d_stride
        - gyro_gait, acc_gait, 3d_gait
        - gait_events

        Returns:
            fig: Populated figure
            metadata: Complete metadata dict
            base_name: Generated filename base
        """
        # Title mapping
        title_map = {
            'gyro_stride': 'Gyroscopic Stride Cyclograms',
            'acc_stride': 'Accelerometer Stride Cyclograms',
            '3d_stride': '3D Stride Cyclograms',
            'gyro_gait': 'Gyroscopic Gait Cyclograms',
            'acc_gait': 'Accelerometer Gait Cyclograms',
            '3d_gait': '3D Gait Cyclograms',
            'gait_events': 'Gait Event Timeline'
        }

        title = title_map.get(analysis_type, analysis_type.replace('_', ' ').title())
        fig, axes = self.build_subplot_grid(analysis_type, data_dict, title, subject_name)

        # Populate based on type
        if analysis_type == 'gyro_stride':
            subplot_metrics = self._plot_gyro_stride_subplots(axes, data_dict)
        elif analysis_type == 'acc_stride':
            subplot_metrics = self._plot_acc_stride_subplots(axes, data_dict)
        elif analysis_type == '3d_stride':
            subplot_metrics = self._plot_3d_stride_subplots(axes, data_dict)
        elif analysis_type == 'gyro_gait':
            subplot_metrics = self._plot_gyro_gait_subplots(axes, data_dict)
        elif analysis_type == 'acc_gait':
            subplot_metrics = self._plot_acc_gait_subplots(axes, data_dict)
        elif analysis_type == '3d_gait':
            subplot_metrics = self._plot_3d_gait_subplots(axes, data_dict)
        elif analysis_type == 'gait_events':
            subplot_metrics = self._plot_gait_events_subplots(axes, data_dict)
        else:
            raise NotImplementedError(f"Subplot type {analysis_type} not yet implemented")

        # Generate metadata
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        base_name = f"{analysis_type}_{subject_name}_{timestamp}"

        # Extract cycles and cyclograms for stride event metadata
        all_cycles = []
        all_cyclograms = []

        for leg_key in ['left', 'right']:
            leg_data = data_dict.get(leg_key, {})
            # For cyclogram plots, leg_data is a dict of sensor_label -> cyclogram(s)
            # For gait_events, it's already in the right format
            if analysis_type != 'gait_events' and isinstance(leg_data, dict):
                for sensor_label, cyclogram_data in leg_data.items():
                    # Get cyclograms (could be single or list)
                    cyclograms = cyclogram_data if isinstance(cyclogram_data, list) else [cyclogram_data]
                    for cyc in cyclograms:
                        if cyc and hasattr(cyc, 'cycle'):
                            all_cycles.append(cyc.cycle)
                            all_cyclograms.append(cyc)

        # Remove duplicate cycles based on cycle_id and leg
        unique_cycles = []
        seen_cycles = set()
        for cycle in all_cycles:
            key = (cycle.leg, cycle.cycle_id)
            if key not in seen_cycles:
                seen_cycles.add(key)
                unique_cycles.append(cycle)

        metadata = self._generate_metadata(
            analysis_type=analysis_type,
            leg='bilateral',
            sensor_type=analysis_type.split('_')[0],  # gyro/acc/3d/gait
            file_name=f"{base_name}.png",
            subject=subject_name,
            subplot_metrics=subplot_metrics,
            cycles=unique_cycles if unique_cycles else None,
            cyclograms=all_cyclograms if all_cyclograms else None
        )

        return fig, metadata, base_name


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

class InsolePipeline:
    """Main orchestrator for insole gait analysis with all enhancements."""

    def __init__(self, config: InsoleConfig, use_precision_detection: bool = True):
        self.config = config
        self.data_handler = Data_handling(config)

        # Track if GYRO data is real or placeholder (set during load_data)
        self.has_real_gyro = True

        # Initialize precision detector if requested
        self.precision_detector = PressureSensorZoneDetector(config) if use_precision_detection else None

        # Pass precision detector to phase detector
        self.phase_detector = GaitPhaseDetector(config, precision_detector=self.precision_detector)

        self.cyclogram_generator = CyclogramGenerator(config)
        self.mmc_computer = MorphologicalMeanCyclogramComputer()
        self.validator = ValidationEngine(config)
        self.symmetry_analyzer = SymmetryAnalyzer(config)
        self.visualizer = None  # Initialized with output_dir

    def analyze_insole_data(self, input_csv: Path, output_dir: Path):
        """
        Complete analysis pipeline for smart insole data.

        Stages:
        1. Load and preprocess data
        2. Self-calibration (baseline correction + adaptive filter tuning)
        3. Feature extraction (with adaptive filtering)
        4. Gait phase detection
        5. Cyclogram generation
        6. MMC computation
        7. Bilateral symmetry analysis
        8. Validation and visualization with PNG+JSON output
        """
        print("="*70)
        print("SMART INSOLE GAIT ANALYSIS - COMPLETE MERGED VERSION")
        print("="*70)

        # Stage 1: Load data and detect gyro availability FIRST
        df = self.data_handler.load_data(input_csv)
        col_map = self.data_handler.detect_column_mapping(df)

        # Extract subject name from filename (needed for plot generation)
        subject_name = input_csv.stem

        # Get GYRO availability from data handler BEFORE initializing visualizer
        self.has_real_gyro = self.data_handler.has_real_gyro

        # NOW initialize visualizer with correct GYRO availability flag
        self.visualizer = InsoleVisualizer(self.config, output_dir, self.has_real_gyro)
        print(f"  Visualizer initialized (has_real_gyro={self.has_real_gyro})")

        # Save column detection diagnostics (nice-to-have for debugging CSV formats)
        columns_found = {
            'detected_format': self.data_handler.csv_format_version if hasattr(self.data_handler, 'csv_format_version') else 'unknown',
            'has_real_gyro': self.has_real_gyro,
            'column_mapping': {k: v for k, v in col_map.items()},
            'total_columns': len(df.columns),
            'detected_columns': list(df.columns),
            'timestamp': df['timestamp'].iloc[0] if 'timestamp' in df.columns else 'N/A',
            'sampling_rate': self.config.sampling_rate,
            'duration_seconds': df['timestamp'].iloc[-1] - df['timestamp'].iloc[0] if 'timestamp' in df.columns else 'N/A'
        }
        columns_json_path = output_dir / f"{subject_name}_columns_found.json"
        with open(columns_json_path, 'w') as f:
            json.dump(columns_found, f, indent=2, default=str)
        print(f"  Saved column diagnostics: {columns_json_path.name}")

        # Stage 2: Self-calibration
        print("\nPerforming self-calibration...")
        df = self.data_handler.calibrate_signals(df, col_map)

        # Stage 3: Feature extraction
        print("\nExtracting features...")
        df = self.data_handler.compute_pressure_features(df, col_map)
        df = self.data_handler.compute_imu_features(df, col_map)
        print("  Feature extraction complete")

        # Stage 4: Gait phase detection
        left_cycles = self.phase_detector.detect_gait_cycles(df, 'left')
        right_cycles = self.phase_detector.detect_gait_cycles(df, 'right')

        # Stage 5: Cyclogram generation
        print("\nGenerating cyclograms...")
        left_cyclograms = self.cyclogram_generator.generate_cyclograms(
            df, left_cycles, 'left', self.has_real_gyro
        )
        right_cyclograms = self.cyclogram_generator.generate_cyclograms(
            df, right_cycles, 'right', self.has_real_gyro
        )
        print(f"  Generated {len(left_cyclograms)} left cyclograms")
        print(f"  Generated {len(right_cyclograms)} right cyclograms")

        # Stage 6: MMC computation
        print("\nComputing Morphological Mean Cyclograms...")
        left_mmc = self._generate_mmc(left_cyclograms, 'left')
        right_mmc = self._generate_mmc(right_cyclograms, 'right')

        # Stage 6b: Generate MMC plots
        print("\nGenerating Mean Cyclogram plots...")
        self._generate_mmc_plots(left_mmc, right_mmc, left_cyclograms, right_cyclograms, subject_name)

        # Stage 7: Validation
        print("\nValidating anatomical plausibility...")
        validation = self.validator.validate_gait_cycles(left_cycles, right_cycles)

        self._print_validation_results(validation)

        # Stage 8: Bilateral Symmetry Analysis (2D and 3D cyclograms)
        print("\nComputing bilateral symmetry metrics...")
        # Compute symmetry for 2D cyclograms
        left_cyclograms_2d = [c for c in left_cyclograms if not c.is_3d]
        right_cyclograms_2d = [c for c in right_cyclograms if not c.is_3d]
        symmetry_metrics_2d = self._compute_symmetry(left_cyclograms_2d, right_cyclograms_2d, "2D")

        # Compute symmetry for 3D cyclograms (gait sub-phase based)
        left_cyclograms_3d = [c for c in left_cyclograms if c.is_3d]
        right_cyclograms_3d = [c for c in right_cyclograms if c.is_3d]
        symmetry_metrics_3d = self._compute_symmetry(left_cyclograms_3d, right_cyclograms_3d, "3D")

        # Combine both 2D and 3D symmetry metrics
        symmetry_metrics = pd.concat([symmetry_metrics_2d, symmetry_metrics_3d], ignore_index=True)

        # Stage 8b: Generate symmetry plots (separate for 2D and 3D)
        print("\nGenerating bilateral symmetry plots...")
        if not symmetry_metrics_2d.empty:
            self._generate_symmetry_plots(symmetry_metrics_2d, subject_name, dimension="2D")
        if not symmetry_metrics_3d.empty:
            self._generate_symmetry_plots(symmetry_metrics_3d, subject_name, dimension="3D")

        # Stage 9: Visualization
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\nGenerating visualizations with PNG+JSON output...")

        # NOTE: Individual cyclogram plots are now generated as organized subplots only
        # (removed individual file generation to reduce clutter and improve organization)

        # ====================================================================
        # SUBPLOT FIGURE GENERATION (Organized Multi-Panel Visualizations)
        # ====================================================================
        print("\nGenerating organized subplot figures...")

        self._generate_subplot_figures(
            left_cyclograms, right_cyclograms,
            left_cycles, right_cycles,
            subject_name,
            df  # Pass DataFrame for individual pressure sensor access in gait events
        )

        # Save summary
        self._save_summary(left_cycles, right_cycles, validation, output_dir)

        # Save symmetry metrics
        self._save_symmetry_results(symmetry_metrics, output_dir)

        # Save advanced cyclogram metrics (Levels 1-3)
        self._save_advanced_cyclogram_metrics(left_cyclograms, right_cyclograms, output_dir)

        # Save precision events if available
        if self.precision_detector is not None:
            self._save_precision_events(df, output_dir)
            # Create debug validation plots
            self.create_debug_event_validation_plot(df, output_dir)

        # Save detailed per-leg gait cycle phases (8-phase segmentation per leg)
        self._save_detailed_gait_phases(left_cycles, right_cycles, output_dir)

        # Export comprehensive Excel workbook with all results
        self._export_comprehensive_excel(output_dir)

        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)

    def _generate_mmc(self, cyclograms: List[CyclogramData], leg: str) -> Optional[MorphologicalMeanCyclogram]:
        """Compute MMC for 2D cyclograms of same sensor pair (3D cyclograms excluded)."""
        # Filter out 3D cyclograms - MMC only works with 2D data
        cyclograms_2d = [c for c in cyclograms if not c.is_3d]

        if not cyclograms_2d:
            print(f"  No 2D cyclograms available for {leg} MMC computation")
            return None

        # Group by sensor pair
        by_pair = {}
        for c in cyclograms_2d:
            key = (c.x_label, c.y_label)
            if key not in by_pair:
                by_pair[key] = []
            by_pair[key].append(c)

        # Compute MMC for first sensor pair (can expand to all pairs)
        if by_pair:
            first_pair = list(by_pair.keys())[0]
            mmc = self.mmc_computer.compute_mmc(by_pair[first_pair])
            if mmc:
                print(f"  {leg.title()} MMC: {mmc.n_loops} loops, dispersion={mmc.shape_dispersion_index:.3f}")
            return mmc
        return None

    def _generate_mmc_plots(self, left_mmc, right_mmc, left_cyclograms, right_cyclograms, subject_name):
        """Generate and save mean cyclogram plots with MMC overlays.

        NOTE: left_mmc and right_mmc parameters are deprecated and ignored.
        MMC is now computed per sensor pair for accurate visualization.
        """
        try:
            # Group cyclograms by sensor pair
            left_by_pair = {}
            for c in [c for c in left_cyclograms if not c.is_3d]:
                key = f"{c.x_label}_vs_{c.y_label}"
                if key not in left_by_pair:
                    left_by_pair[key] = []
                left_by_pair[key].append(c)

            right_by_pair = {}
            for c in [c for c in right_cyclograms if not c.is_3d]:
                key = f"{c.x_label}_vs_{c.y_label}"
                if key not in right_by_pair:
                    right_by_pair[key] = []
                right_by_pair[key].append(c)

            # Generate plots for each sensor pair
            for sensor_pair_key in set(list(left_by_pair.keys()) + list(right_by_pair.keys())):
                left_loops = left_by_pair.get(sensor_pair_key, [])
                right_loops = right_by_pair.get(sensor_pair_key, [])

                if left_loops or right_loops:
                    # Compute MMC for THIS specific sensor pair
                    left_mmc_for_pair = self.mmc_computer.compute_mmc(left_loops) if left_loops else None
                    right_mmc_for_pair = self.mmc_computer.compute_mmc(right_loops) if right_loops else None

                    # Use first cyclogram as template for plotting
                    template = left_loops[0] if left_loops else right_loops[0]

                    # Create figure with standardized dimensions
                    fig, axes = plt.subplots(1, 2, figsize=(12, 10), dpi=300)
                    fig.suptitle(f"{subject_name} - Mean Cyclograms ({sensor_pair_key.replace('_', ' ')})",
                                fontsize=14, fontweight='bold')

                    # Plot left leg with sensor-pair-specific MMC
                    if left_loops and left_mmc_for_pair:
                        self._plot_mmc_subplot(axes[0], left_loops, left_mmc_for_pair, 'Left', template)
                    else:
                        axes[0].text(0.5, 0.5, 'No Left Leg Data', ha='center', va='center', transform=axes[0].transAxes)
                        axes[0].set_title('Left Leg', fontweight='bold')

                    # Plot right leg with sensor-pair-specific MMC
                    if right_loops and right_mmc_for_pair:
                        self._plot_mmc_subplot(axes[1], right_loops, right_mmc_for_pair, 'Right', template)
                    else:
                        axes[1].text(0.5, 0.5, 'No Right Leg Data', ha='center', va='center', transform=axes[1].transAxes)
                        axes[1].set_title('Right Leg', fontweight='bold')

                    plt.tight_layout()

                    # Save to mean_cyclograms directory
                    timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
                    filename = f"mmc_{sensor_pair_key.lower()}_{subject_name}_{timestamp}"

                    # Create metadata with MMC info
                    metadata = {
                        'sensor_pair': sensor_pair_key,
                        'left_mmc_dispersion': left_mmc_for_pair.shape_dispersion_index if left_mmc_for_pair else None,
                        'right_mmc_dispersion': right_mmc_for_pair.shape_dispersion_index if right_mmc_for_pair else None,
                        'left_n_loops': left_mmc_for_pair.n_loops if left_mmc_for_pair else 0,
                        'right_n_loops': right_mmc_for_pair.n_loops if right_mmc_for_pair else 0
                    }

                    self.visualizer.save_outputs(fig, metadata, filename, category='mean_cyclograms')
                    plt.close(fig)
                    print(f"  ✓ Saved: {filename}.png (L:{metadata['left_n_loops']} cycles, R:{metadata['right_n_loops']} cycles)")

        except Exception as e:
            print(f"  ✗ Error generating MMC plots: {str(e)}")

    def _plot_mmc_subplot(self, ax, loops, mmc, leg_name, template):
        """Plot individual cycles with MMC overlay on a subplot."""
        # Plot all individual cycles (semi-transparent)
        for loop in loops:
            ax.plot(loop.x_signal, loop.y_signal, 'gray', alpha=0.15, linewidth=0.8)

        # Plot MMC median trajectory
        if mmc and mmc.median_trajectory is not None:
            median_x = mmc.median_trajectory[:, 0]
            median_y = mmc.median_trajectory[:, 1]
            ax.plot(median_x, median_y,
                   'b-', linewidth=3, label=f'MMC Median (n={mmc.n_loops})')

            # Plot ±SD envelope using variance bounds
            if mmc.variance_envelope_lower is not None and mmc.variance_envelope_upper is not None:
                lower_x = mmc.variance_envelope_lower[:, 0]
                lower_y = mmc.variance_envelope_lower[:, 1]
                upper_x = mmc.variance_envelope_upper[:, 0]
                upper_y = mmc.variance_envelope_upper[:, 1]

                # Fill between upper and lower bounds
                ax.fill(np.concatenate([upper_x, lower_x[::-1]]),
                       np.concatenate([upper_y, lower_y[::-1]]),
                       alpha=0.2, color='blue', label='±1 SD')

        ax.set_xlabel(template.x_label, fontsize=10)
        ax.set_ylabel(template.y_label, fontsize=10)
        ax.set_title(f'{leg_name} Leg (SDI={mmc.shape_dispersion_index:.3f})' if mmc else f'{leg_name} Leg',
                    fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        ax.set_aspect('equal', adjustable='box')

    def _generate_symmetry_plots(self, symmetry_df, subject_name, dimension="2D"):
        """Generate bilateral symmetry comparison plots for 2D or 3D cyclograms."""
        try:
            if symmetry_df.empty:
                print(f"  No {dimension} symmetry data to plot")
                return

            # Group by cyclogram type
            for cyclogram_type in symmetry_df['cyclogram_type'].unique():
                type_data = symmetry_df[symmetry_df['cyclogram_type'] == cyclogram_type]

                # Create figure with 4 subplots (standardized dimensions)
                fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
                fig.suptitle(f"{subject_name} - Bilateral Symmetry Analysis [{dimension}] ({cyclogram_type})",
                            fontsize=14, fontweight='bold')

                # 1. Area comparison (Left vs Right)
                axes[0, 0].scatter(type_data['area_L'], type_data['area_R'], alpha=0.6, s=50)
                axes[0, 0].plot([type_data[['area_L', 'area_R']].min().min(),
                                type_data[['area_L', 'area_R']].max().max()],
                               [type_data[['area_L', 'area_R']].min().min(),
                                type_data[['area_L', 'area_R']].max().max()],
                               'r--', label='Perfect Symmetry')
                axes[0, 0].set_xlabel('Left Leg Area', fontsize=10)
                axes[0, 0].set_ylabel('Right Leg Area', fontsize=10)
                axes[0, 0].set_title('Area Symmetry (Left vs Right Leg)', fontweight='bold')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)

                # 2. Symmetry metric distributions
                symmetry_cols = ['area_symmetry', 'curvature_symmetry', 'smoothness_symmetry', 'overall_symmetry']
                axes[0, 1].boxplot([type_data[col].dropna() for col in symmetry_cols],
                                  labels=['Area', 'Curvature', 'Smoothness', 'Overall'])
                axes[0, 1].set_ylabel('Symmetry Score', fontsize=10)
                axes[0, 1].set_title('Symmetry Metric Distributions', fontweight='bold')
                axes[0, 1].grid(True, alpha=0.3, axis='y')
                axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Perfect Symmetry')

                # 3. Cycle-by-cycle symmetry trends
                axes[1, 0].plot(type_data.index, type_data['overall_symmetry'], 'o-', alpha=0.7)
                axes[1, 0].axhline(y=type_data['overall_symmetry'].mean(), color='r',
                                  linestyle='--', label=f'Mean={type_data["overall_symmetry"].mean():.3f}')
                axes[1, 0].set_xlabel('Cycle Index', fontsize=10)
                axes[1, 0].set_ylabel('Overall Symmetry', fontsize=10)
                axes[1, 0].set_title('Cycle-by-Cycle Symmetry', fontweight='bold')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)

                # 4. Summary statistics table with variance heatmap
                axes[1, 1].axis('off')

                # Collect statistics for heatmap coloring
                metrics_data = {
                    'Area Sym (L vs R)': type_data['area_symmetry'],
                    'Curv Sym': type_data['curvature_symmetry'],
                    'Smooth Sym': type_data['smoothness_symmetry'],
                    'Overall': type_data['overall_symmetry']
                }

                summary_stats = [['Metric', 'Mean', 'Std', 'Min', 'Max']]
                std_values = []  # For color normalization

                for name, data in metrics_data.items():
                    summary_stats.append([
                        name,
                        f"{data.mean():.3f}",
                        f"{data.std():.3f}",
                        f"{data.min():.3f}",
                        f"{data.max():.3f}"
                    ])
                    std_values.append(data.std())

                table = axes[1, 1].table(cellText=summary_stats, cellLoc='center',
                                        loc='center', bbox=[0, 0, 1, 1])
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 2)

                # Color header row
                for j in range(5):
                    table[(0, j)].set_facecolor('#4a4a4a')
                    table[(0, j)].set_text_props(color='white', weight='bold')

                # Color-code Std column using variance heatmap (higher std = redder)
                if std_values:
                    max_std = max(std_values)
                    min_std = min(std_values)
                    std_range = max_std - min_std if max_std > min_std else 1.0

                    for i, std_val in enumerate(std_values, start=1):
                        # Normalize std to 0-1 range
                        normalized = (std_val - min_std) / std_range if std_range > 0 else 0

                        # Green (low variance) → Yellow → Red (high variance)
                        if normalized < 0.5:
                            # Green to Yellow
                            r = int(255 * (normalized * 2))
                            g = 255
                            b = 0
                        else:
                            # Yellow to Red
                            r = 255
                            g = int(255 * (2 - normalized * 2))
                            b = 0

                        color = f'#{r:02x}{g:02x}{b:02x}'
                        table[(i, 2)].set_facecolor(color)  # Std column

                        # Light background for other data cells
                        table[(i, 0)].set_facecolor('#f5f5f5')
                        table[(i, 1)].set_facecolor('#ffffff')
                        table[(i, 3)].set_facecolor('#ffffff')
                        table[(i, 4)].set_facecolor('#ffffff')

                axes[1, 1].set_title('Summary Statistics (Std colored by variance)',
                                   fontweight='bold', y=0.95, fontsize=10)

                plt.tight_layout()

                # Save to symmetry directory with dimension prefix
                timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
                filename = f"symmetry_{dimension.lower()}_{cyclogram_type.lower()}_{subject_name}_{timestamp}"
                metadata = {'dimension': dimension, 'cyclogram_type': cyclogram_type}
                self.visualizer.save_outputs(fig, metadata, filename, category='symmetry')
                plt.close(fig)
                print(f"  ✓ Saved: {filename}.png")

        except Exception as e:
            print(f"  ✗ Error generating symmetry plots: {str(e)}")

    def _print_validation_results(self, validation: ValidationMetrics):
        """Print validation results to console."""

        print("\n  Validation Results:")
        print(f"    Left-Right Alternation: {'✓' if validation.left_right_alternation else '✗'}")
        print(f"    Stance-Swing Ratio: {'✓' if validation.stance_swing_ratio_valid else '✗'}")
        print(f"    Bilateral Symmetry: {'✓' if validation.bilateral_symmetry_valid else '✗'}")
        print(f"    Phase Sequence: {'✓' if validation.phase_sequence_valid else '✗'}")
        print(f"    Duration Constraints: {'✓' if validation.duration_constraints_valid else '✗'}")
        print(f"\n    Overall Valid: {'✓ PASS' if validation.overall_valid else '✗ FAIL'}")

        if validation.issues:
            print("\n  Issues Found:")
            for issue in validation.issues:
                print(f"    - {issue}")

    def _save_summary(self, left_cycles: List[GaitCycle],
                     right_cycles: List[GaitCycle],
                     validation: ValidationMetrics,
                     output_dir: Path):
        """Save analysis summary to CSV with bilateral comparison."""

        summary_data = []

        for cycle in left_cycles + right_cycles:
            summary_data.append({
                'leg': cycle.leg,
                'cycle_id': cycle.cycle_id,
                'duration': cycle.duration,
                'stance_duration': cycle.stance_duration,
                'swing_duration': cycle.swing_duration,
                'stance_swing_ratio': cycle.stance_swing_ratio,
                'start_time': cycle.start_time,
                'end_time': cycle.end_time
            })

        df_summary = pd.DataFrame(summary_data)
        summary_path = output_dir / 'gait_cycle_summary.csv'
        df_summary.to_csv(summary_path, index=False)

        print(f"\n  Saved summary: {summary_path}")

        # Add bilateral comparison summary
        if len(left_cycles) > 0 and len(right_cycles) > 0:
            left_avg = {
                'duration': np.mean([c.duration for c in left_cycles]),
                'stance_duration': np.mean([c.stance_duration for c in left_cycles]),
                'swing_duration': np.mean([c.swing_duration for c in left_cycles]),
                'stance_swing_ratio': np.mean([c.stance_swing_ratio for c in left_cycles])
            }

            right_avg = {
                'duration': np.mean([c.duration for c in right_cycles]),
                'stance_duration': np.mean([c.stance_duration for c in right_cycles]),
                'swing_duration': np.mean([c.swing_duration for c in right_cycles]),
                'stance_swing_ratio': np.mean([c.stance_swing_ratio for c in right_cycles])
            }

            bilateral_comparison = []
            for metric in ['duration', 'stance_duration', 'swing_duration', 'stance_swing_ratio']:
                left_val = left_avg[metric]
                right_val = right_avg[metric]
                asymmetry = abs(left_val - right_val) / ((left_val + right_val) / 2) * 100 if (left_val + right_val) > 0 else 0

                bilateral_comparison.append({
                    'metric': metric,
                    'left_mean': left_val,
                    'right_mean': right_val,
                    'difference': left_val - right_val,
                    'asymmetry_percent': asymmetry
                })

            df_bilateral = pd.DataFrame(bilateral_comparison)
            bilateral_path = output_dir / 'bilateral_comparison_summary.csv'
            df_bilateral.to_csv(bilateral_path, index=False)

            print(f"  Saved bilateral comparison: {bilateral_path}")
            print("\n  Bilateral Asymmetry Summary:")
            for _, row in df_bilateral.iterrows():
                print(f"    {row['metric']}: {row['asymmetry_percent']:.2f}% asymmetry")

    def _compute_symmetry(self, left_cyclograms: List[CyclogramData],
                         right_cyclograms: List[CyclogramData], dimension: str = "2D") -> pd.DataFrame:
        """
        Compute bilateral symmetry metrics for all cyclogram pairs.

        Args:
            left_cyclograms: List of left leg cyclograms
            right_cyclograms: List of right leg cyclograms
            dimension: "2D" or "3D" for reporting purposes

        Returns:
            DataFrame with per-cycle symmetry metrics
        """
        if not left_cyclograms or not right_cyclograms:
            print(f"  No {dimension} cyclograms available for symmetry analysis")
            return pd.DataFrame()

        symmetry_df = self.symmetry_analyzer.compute_batch_symmetry(
            left_cyclograms, right_cyclograms
        )

        # Print summary statistics
        if not symmetry_df.empty:
            print(f"  Computed {dimension} symmetry for {len(symmetry_df)} cyclogram pairs")

            # Aggregate by cyclogram type
            type_summary = symmetry_df.groupby('cyclogram_type').agg({
                'area_symmetry': ['mean', 'std'],
                'curvature_symmetry': ['mean', 'std'],
                'smoothness_symmetry': ['mean', 'std'],
                'overall_symmetry': ['mean', 'std']
            })

            print("\n  Symmetry Summary (mean ± std):")
            for cyclogram_type in type_summary.index:
                area_sym = type_summary.loc[cyclogram_type, ('area_symmetry', 'mean')]
                curv_sym = type_summary.loc[cyclogram_type, ('curvature_symmetry', 'mean')]
                smooth_sym = type_summary.loc[cyclogram_type, ('smoothness_symmetry', 'mean')]
                overall = type_summary.loc[cyclogram_type, ('overall_symmetry', 'mean')]

                print(f"    {cyclogram_type}:")
                print(f"      Area: {area_sym:.3f}, Curvature: {curv_sym:.3f}, "
                      f"Smoothness: {smooth_sym:.3f}, Overall: {overall:.3f}")

        return symmetry_df

    def _save_symmetry_results(self, symmetry_df: pd.DataFrame, output_dir: Path):
        """
        Save bilateral symmetry metrics to CSV.

        Args:
            symmetry_df: DataFrame with symmetry metrics
            output_dir: Directory to save results
        """
        if symmetry_df.empty:
            print("  No symmetry metrics to save")
            return

        symmetry_path = output_dir / 'symmetry_metrics.csv'
        symmetry_df.to_csv(symmetry_path, index=False)

        print(f"\n  Saved symmetry metrics: {symmetry_path}")

        # Compute and save aggregate summary
        aggregate = symmetry_df.groupby('cyclogram_type').agg({
            'area_L': 'mean',
            'area_R': 'mean',
            'area_symmetry': ['mean', 'std'],
            'curvature_mean_L': 'mean',
            'curvature_mean_R': 'mean',
            'curvature_symmetry': ['mean', 'std'],
            'smoothness_L': 'mean',
            'smoothness_R': 'mean',
            'smoothness_symmetry': ['mean', 'std'],
            'overall_symmetry': ['mean', 'std']
        })

        aggregate_path = output_dir / 'symmetry_aggregate.csv'
        aggregate.to_csv(aggregate_path)

        print(f"  Saved aggregate summary: {aggregate_path}")

    def _save_advanced_cyclogram_metrics(self, left_cyclograms: List[CyclogramData],
                                        right_cyclograms: List[CyclogramData],
                                        output_dir: Path):
        """
        Compute and save advanced cyclogram metrics (Levels 1-3) to CSV.

        Args:
            left_cyclograms: List of left leg cyclograms
            right_cyclograms: List of right leg cyclograms
            output_dir: Directory to save results
        """
        print("\n  Computing advanced cyclogram metrics...")

        metrics_data = []

        # Process all cyclograms
        all_cyclograms = left_cyclograms + right_cyclograms

        for cyclogram in all_cyclograms:
            # Compute metrics for this cyclogram
            metrics = self.visualizer._compute_cyclogram_metrics(cyclogram)

            # Add identifying information
            metrics_row = {
                'cycle_id': cyclogram.cycle.cycle_id,
                'leg': cyclogram.cycle.leg,
                'cyclogram_type': f"{cyclogram.x_label}_vs_{cyclogram.y_label}",
                'duration': cyclogram.cycle.duration,
                **metrics  # Unpack all computed metrics
            }

            metrics_data.append(metrics_row)

        # Create DataFrame
        df_metrics = pd.DataFrame(metrics_data)

        # Save to CSV
        metrics_path = output_dir / 'cyclogram_advanced_metrics.csv'
        df_metrics.to_csv(metrics_path, index=False)

        print(f"  Saved advanced metrics: {metrics_path}")
        print(f"  Total cyclograms analyzed: {len(metrics_data)}")

        # Compute and save aggregate statistics by cyclogram type
        if not df_metrics.empty:
            # Group by cyclogram type and leg
            aggregate = df_metrics.groupby(['cyclogram_type', 'leg']).agg({
                'area': ['mean', 'std'],
                'compactness_ratio': ['mean', 'std'],
                'aspect_ratio': ['mean', 'std'],
                'eccentricity': ['mean', 'std'],
                'mean_curvature': ['mean', 'std'],
                'trajectory_smoothness': ['mean', 'std'],
                'mean_relative_phase': ['mean', 'std'],
                'marp': ['mean', 'std'],
                'coupling_angle_variability': ['mean', 'std']
            }).round(4)

            aggregate_path = output_dir / 'cyclogram_metrics_aggregate.csv'
            aggregate.to_csv(aggregate_path)

            print(f"  Saved aggregate statistics: {aggregate_path}")

        # Compute Level 3 (Symmetry) metrics for paired left-right cyclograms
        symmetry_metrics_data = []

        # Group cyclograms by type
        cyclogram_dict = {'left': {}, 'right': {}}
        for cyclogram in all_cyclograms:
            cyc_type = f"{cyclogram.x_label}_vs_{cyclogram.y_label}"
            leg = cyclogram.cycle.leg

            if cyc_type not in cyclogram_dict[leg]:
                cyclogram_dict[leg][cyc_type] = []
            cyclogram_dict[leg][cyc_type].append(cyclogram)

        # Compute bilateral symmetry for each cyclogram type
        for cyc_type in set(cyclogram_dict['left'].keys()) & set(cyclogram_dict['right'].keys()):
            left_list = cyclogram_dict['left'][cyc_type]
            right_list = cyclogram_dict['right'][cyc_type]

            # Pair cyclograms by cycle_id
            for left_cyc in left_list:
                for right_cyc in right_list:
                    if left_cyc.cycle.cycle_id == right_cyc.cycle.cycle_id:
                        # Compute bilateral symmetry
                        symmetry = AdvancedCyclogramMetrics.compute_bilateral_symmetry(
                            left_cyc, right_cyc
                        )

                        symmetry_row = {
                            'cycle_id': left_cyc.cycle.cycle_id,
                            'cyclogram_type': cyc_type,
                            **symmetry
                        }

                        symmetry_metrics_data.append(symmetry_row)
                        break  # Only pair once

        if symmetry_metrics_data:
            df_symmetry = pd.DataFrame(symmetry_metrics_data)
            symmetry_path = output_dir / 'cyclogram_bilateral_symmetry.csv'
            df_symmetry.to_csv(symmetry_path, index=False)

            print(f"  Saved bilateral symmetry metrics: {symmetry_path}")
            print(f"  Total cyclogram pairs analyzed: {len(symmetry_metrics_data)}")

    def _export_comprehensive_excel(self, output_dir: Path):
        """
        Export all results to a comprehensive Excel workbook (Result_output.xlsx).

        Creates multiple sheets:
        - Gait Cycles: Cycle timing and duration metrics
        - Advanced Metrics: All cyclogram metrics (Levels 1-3)
        - Bilateral Symmetry: Left-right symmetry analysis
        - Aggregated Stats: Summary statistics by cyclogram type
        - Precision Events: High-precision gait events
        """
        print("\n  Creating comprehensive Excel export...")

        excel_path = output_dir / 'Result_output.xlsx'

        # Load all CSV files
        csv_files = {
            'Gait Cycles': output_dir / 'gait_cycle_summary.csv',
            'Bilateral Comparison': output_dir / 'bilateral_comparison_summary.csv',
            'Advanced Metrics': output_dir / 'cyclogram_advanced_metrics.csv',
            'Cyclogram Symmetry': output_dir / 'cyclogram_bilateral_symmetry.csv',
            'Aggregated Stats': output_dir / 'cyclogram_metrics_aggregate.csv',
            'Symmetry Metrics': output_dir / 'symmetry_metrics.csv',
            'Symmetry Aggregate': output_dir / 'symmetry_aggregate.csv',
            'Precision Events': output_dir / 'precision_gait_events.csv'
        }

        # Create Excel writer
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            for sheet_name, csv_path in csv_files.items():
                if csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path)
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        print(f"    Added sheet: {sheet_name} ({len(df)} rows)")
                    except Exception as e:
                        print(f"    Warning: Could not add {sheet_name}: {e}")
                else:
                    print(f"    Skipping {sheet_name} (file not found)")

        print(f"  Saved comprehensive Excel: {excel_path}")

    def _generate_subplot_figures(self, left_cyclograms: List[CyclogramData],
                                  right_cyclograms: List[CyclogramData],
                                  left_cycles: List[GaitCycle],
                                  right_cycles: List[GaitCycle],
                                  subject_name: str,
                                  df: pd.DataFrame = None):
        """
        Generate all organized subplot figures for comprehensive gait analysis.

        Creates 7 types of multi-panel visualizations:
        1. Gyroscopic Stride Cyclograms (2×3 grid: L/R × X-Y/X-Z/Y-Z)
        2. Accelerometer Stride Cyclograms (2×3 grid)
        3. 3D Stride Cyclograms (2×2 grid: L/R × Gyro3D/Acc3D)
        4. Gyroscopic Gait Cyclograms (1×3 grid: X-Y/X-Z/Y-Z)
        5. Accelerometer Gait Cyclograms (1×3 grid)
        6. 3D Gait Cyclograms (1×2 grid: Gyro3D/Acc3D)
        7. Gait Event Timeline (1×2 grid: Left/Right events)

        Each figure is saved as PNG with accompanying JSON metadata.
        """
        # =================================================================
        # DATA ORGANIZATION
        # =================================================================

        # For STRIDE-LEVEL plots: Use first cyclogram of each type per leg
        # CRITICAL: Separate GYRO and ACC to avoid key collisions
        stride_dict = {
            'left': {'gyro': {}, 'acc': {}},
            'right': {'gyro': {}, 'acc': {}}
        }

        # For GAIT-LEVEL plots: Collect ALL cyclograms of each type per leg
        gait_dict = {
            'left': {},
            'right': {}
        }

        # Map sensor pair labels to subplot labels (simple without sensor prefix)
        sensor_mapping_stride_simple = {
            'GYRO_X_vs_GYRO_Y': 'X-Y Plane',
            'GYRO_X_vs_GYRO_Z': 'X-Z Plane',
            'GYRO_Y_vs_GYRO_Z': 'Y-Z Plane',
            'ACC_X_vs_ACC_Y': 'X-Y Plane',
            'ACC_X_vs_ACC_Z': 'X-Z Plane',
            'ACC_Y_vs_ACC_Z': 'Y-Z Plane'
        }

        # CRITICAL: Must include sensor type prefix to avoid mixing GYRO and ACC data in gait dict
        sensor_mapping_gait = {
            'GYRO_X_vs_GYRO_Y': 'GYRO_X-Y Plane',
            'GYRO_X_vs_GYRO_Z': 'GYRO_X-Z Plane',
            'GYRO_Y_vs_GYRO_Z': 'GYRO_Y-Z Plane',
            'ACC_X_vs_ACC_Y': 'ACC_X-Y Plane',
            'ACC_X_vs_ACC_Z': 'ACC_X-Z Plane',
            'ACC_Y_vs_ACC_Z': 'ACC_Y-Z Plane'
        }

        # Organize cyclograms by sensor type and leg
        for cyclogram in left_cyclograms + right_cyclograms:
            leg = cyclogram.cycle.leg

            # Handle 3D cyclograms separately
            if cyclogram.is_3d and cyclogram.z_signal is not None:
                # Determine sensor type from x_label (ACC_X → acc, GYRO_X → gyro)
                sensor_type = 'acc' if 'ACC' in cyclogram.x_label else 'gyro'
                label_3d = f'{sensor_type}_3d'

                # For stride-level: store in nested structure by sensor type
                if label_3d not in stride_dict[leg][sensor_type]:
                    stride_dict[leg][sensor_type][label_3d] = cyclogram

                # For gait-level: collect all 3D cyclograms
                if label_3d not in gait_dict[leg]:
                    gait_dict[leg][label_3d] = []
                gait_dict[leg][label_3d].append(cyclogram)
            else:
                # Handle 2D cyclograms
                key = f"{cyclogram.x_label}_vs_{cyclogram.y_label}"
                label_stride_simple = sensor_mapping_stride_simple.get(key, key)
                label_gait = sensor_mapping_gait.get(key, key)

                # Determine sensor type for proper categorization
                is_gyro = 'GYRO' in key
                is_acc = 'ACC' in key
                sensor_type = 'gyro' if is_gyro else 'acc'

                # For stride-level: keep only first cyclogram in appropriate sensor category
                if label_stride_simple not in stride_dict[leg][sensor_type]:
                    stride_dict[leg][sensor_type][label_stride_simple] = cyclogram

                # For gait-level: collect all cyclograms (use prefixed labels to avoid collision)
                # PERFORMANCE: Limit to max 30 cycles per sensor pair for large files
                MAX_GAIT_CYCLES = 30
                if label_gait not in gait_dict[leg]:
                    gait_dict[leg][label_gait] = []
                if len(gait_dict[leg][label_gait]) < MAX_GAIT_CYCLES:
                    gait_dict[leg][label_gait].append(cyclogram)

                # Also track sensor type for plot generation checks
                # CRITICAL: Only mark GYRO as available if we have REAL gyro data
                if is_gyro and self.has_real_gyro:
                    stride_dict[leg]['_has_gyro'] = True
                    gait_dict[leg]['_has_gyro'] = True
                if is_acc:
                    stride_dict[leg]['_has_acc'] = True
                    gait_dict[leg]['_has_acc'] = True

        # =================================================================
        # 1. GYROSCOPIC STRIDE CYCLOGRAMS (if available)
        # =================================================================
        if stride_dict['left'].get('_has_gyro', False) or stride_dict['right'].get('_has_gyro', False):
            try:
                # Extract gyro data into flat dict for plotting
                gyro_stride_data = {
                    'left': stride_dict['left']['gyro'],
                    'right': stride_dict['right']['gyro']
                }
                fig, metadata, base_name = self.visualizer.create_and_populate_subplot_figure(
                    'gyro_stride', gyro_stride_data, subject_name
                )
                self.visualizer.save_outputs(fig, metadata, base_name)
                plt.close(fig)
                print(f"  ✓ Generated: Gyroscopic Stride Cyclograms (Plot Set 1)")
            except Exception as e:
                print(f"  ✗ Skipped Gyroscopic Stride: {str(e)}")

        # =================================================================
        # 2. ACCELEROMETER STRIDE CYCLOGRAMS (if available)
        # =================================================================
        if stride_dict['left'].get('_has_acc', False) or stride_dict['right'].get('_has_acc', False):
            try:
                # Extract acc data into flat dict for plotting
                acc_stride_data = {
                    'left': stride_dict['left']['acc'],
                    'right': stride_dict['right']['acc']
                }
                fig, metadata, base_name = self.visualizer.create_and_populate_subplot_figure(
                    'acc_stride', acc_stride_data, subject_name
                )
                self.visualizer.save_outputs(fig, metadata, base_name)
                plt.close(fig)
                print(f"  ✓ Generated: Accelerometer Stride Cyclograms (Plot Set 2)")
            except Exception as e:
                print(f"  ✗ Skipped Accelerometer Stride: {str(e)}")

        # =================================================================
        # 3. 3D STRIDE CYCLOGRAMS (if available)
        # =================================================================
        # Check if 3D data exists in either gyro or acc categories
        has_3d_data = (any('_3d' in k for k in stride_dict['left']['gyro'].keys()) or
                      any('_3d' in k for k in stride_dict['left']['acc'].keys()) or
                      any('_3d' in k for k in stride_dict['right']['gyro'].keys()) or
                      any('_3d' in k for k in stride_dict['right']['acc'].keys()))
        if has_3d_data:
            try:
                # Merge gyro and acc 3D data for plotting
                stride_3d_data = {
                    'left': {**stride_dict['left']['gyro'], **stride_dict['left']['acc']},
                    'right': {**stride_dict['right']['gyro'], **stride_dict['right']['acc']}
                }
                fig, metadata, base_name = self.visualizer.create_and_populate_subplot_figure(
                    '3d_stride', stride_3d_data, subject_name
                )
                self.visualizer.save_outputs(fig, metadata, base_name)
                plt.close(fig)
                print(f"  ✓ Generated: 3D Stride Cyclograms (Plot Set 3)")
            except Exception as e:
                print(f"  ✗ Skipped 3D Stride: {str(e)}")

        # =================================================================
        # 4. GYROSCOPIC GAIT CYCLOGRAMS (if available)
        # =================================================================
        if gait_dict['left'].get('_has_gyro', False) or gait_dict['right'].get('_has_gyro', False):
            try:
                fig, metadata, base_name = self.visualizer.create_and_populate_subplot_figure(
                    'gyro_gait', gait_dict, subject_name
                )
                self.visualizer.save_outputs(fig, metadata, base_name)
                plt.close(fig)
                print(f"  ✓ Generated: Gyroscopic Gait Cyclograms (Plot Set 4)")
            except Exception as e:
                print(f"  ✗ Skipped Gyroscopic Gait: {str(e)}")

        # =================================================================
        # 5. ACCELEROMETER GAIT CYCLOGRAMS (if available)
        # =================================================================
        if gait_dict['left'].get('_has_acc', False) or gait_dict['right'].get('_has_acc', False):
            try:
                fig, metadata, base_name = self.visualizer.create_and_populate_subplot_figure(
                    'acc_gait', gait_dict, subject_name
                )
                self.visualizer.save_outputs(fig, metadata, base_name)
                plt.close(fig)
                print(f"  ✓ Generated: Accelerometer Gait Cyclograms (Plot Set 5)")
            except Exception as e:
                print(f"  ✗ Skipped Accelerometer Gait: {str(e)}")

        # =================================================================
        # 6. 3D GAIT CYCLOGRAMS (if available)
        # =================================================================
        has_3d_gait_data = any('_3d' in k for k in gait_dict['left'].keys()) or any('_3d' in k for k in gait_dict['right'].keys())
        if has_3d_gait_data:
            try:
                fig, metadata, base_name = self.visualizer.create_and_populate_subplot_figure(
                    '3d_gait', gait_dict, subject_name
                )
                self.visualizer.save_outputs(fig, metadata, base_name)
                plt.close(fig)
                print(f"  ✓ Generated: 3D Gait Cyclograms (Plot Set 6)")
            except Exception as e:
                print(f"  ✗ Skipped 3D Gait: {str(e)}")

        # =================================================================
        # 7. GAIT EVENT TIMELINE
        # =================================================================
        try:
            # Extract events from cycles for timeline visualization
            def extract_events_from_cycles(cycles: List[GaitCycle]) -> Dict:
                if not cycles:
                    return {
                        'heel_strikes': [],
                        'toe_offs': [],
                        'phases': []
                    }

                heel_strikes = []
                toe_offs = []
                phases_data = []

                for cycle in cycles:
                    if cycle and hasattr(cycle, 'start_time'):
                        heel_strikes.append(cycle.start_time)

                        # Toe-off is typically the first phase boundary after heel strike
                        if hasattr(cycle, 'phases') and cycle.phases and len(cycle.phases) > 0:
                            first_phase = cycle.phases[0]
                            if hasattr(first_phase, 'start_time') and hasattr(first_phase, 'duration'):
                                toe_off_time = first_phase.start_time + first_phase.duration
                                if hasattr(cycle, 'end_time') and toe_off_time < cycle.end_time:
                                    toe_offs.append(toe_off_time)

                            # Add phase data
                            for phase in cycle.phases:
                                if hasattr(phase, 'start_time') and hasattr(phase, 'duration'):
                                    phases_data.append({
                                        'start_time': phase.start_time,
                                        'end_time': phase.start_time + phase.duration,
                                        'phase_name': getattr(phase, 'phase_name', 'Unknown'),
                                        'phase_number': getattr(phase, 'phase_number', 0),
                                        'support_type': getattr(phase, 'support_type', None)
                                    })

                return {
                    'heel_strikes': heel_strikes,
                    'toe_offs': toe_offs,
                    'phases': phases_data
                }

            event_data = {
                'df': df,  # Add DataFrame for individual pressure sensor access
                'left': extract_events_from_cycles(left_cycles) if left_cycles else {'heel_strikes': [], 'toe_offs': [], 'phases': []},
                'right': extract_events_from_cycles(right_cycles) if right_cycles else {'heel_strikes': [], 'toe_offs': [], 'phases': []}
            }

            fig, metadata, base_name = self.visualizer.create_and_populate_subplot_figure(
                'gait_events', event_data, subject_name
            )
            self.visualizer.save_outputs(fig, metadata, base_name)
            plt.close(fig)
            print(f"  ✓ Generated: Gait Event Timeline (Plot Set 7)")
        except Exception as e:
            import traceback
            print(f"  ✗ Skipped Gait Event Timeline: {str(e)}")
            # traceback.print_exc()  # Uncomment for debugging

        print(f"  \n  Comprehensive subplot figure generation complete!")
        print(f"  Generated all 7 core plot sets (6 cyclogram types + gait events)")

    def _save_precision_events(self, df: pd.DataFrame, output_dir: Path):
        """Save high-precision gait events to CSV for validation."""

        print("\n  Exporting precision gait events...")

        all_events = []

        for leg in ['left', 'right']:
            events = self.precision_detector.detect_all_events(df, leg)

            for event_type, event_list in events.items():
                for event in event_list:
                    all_events.append({
                        'leg': event.leg,
                        'event_type': event.event_type,
                        'event_start_time': event.event_start,
                        'event_end_time': event.event_end,
                        'duration_ms': event.duration,
                        'sensor_source': event.sensor_source,
                        'frame_start': event.frame_start,
                        'frame_end': event.frame_end,
                        'confidence': event.confidence
                    })

        df_events = pd.DataFrame(all_events)
        df_events = df_events.sort_values(['leg', 'event_start_time'])

        events_path = output_dir / 'precision_gait_events.csv'
        df_events.to_csv(events_path, index=False)

        print(f"  Saved {len(all_events)} precision events: {events_path}")

        # Print summary statistics
        for leg in ['left', 'right']:
            leg_events = df_events[df_events['leg'] == leg]
            for event_type in ['heel_strike', 'mid_stance', 'toe_off']:
                type_events = leg_events[leg_events['event_type'] == event_type]
                if len(type_events) > 0:
                    avg_duration = type_events['duration_ms'].mean()
                    avg_confidence = type_events['confidence'].mean()
                    print(f"    {leg.title()} {event_type}: {len(type_events)} events, "
                          f"avg duration={avg_duration:.1f}ms, avg confidence={avg_confidence:.3f}")

    def _save_detailed_gait_phases(self, left_cycles: List[GaitCycle],
                                   right_cycles: List[GaitCycle],
                                   output_dir: Path):
        """
        Save detailed per-leg gait cycle phase breakdown (8-phase segmentation).

        Exports separate CSVs for left and right legs with all gait sub-phases per cycle.
        """
        print("\n  Exporting detailed per-leg gait phases...")

        for leg, cycles in [('left', left_cycles), ('right', right_cycles)]:
            all_phases = []

            for cycle in cycles:
                for phase in cycle.phases:
                    all_phases.append({
                        'leg': leg,
                        'cycle_id': cycle.cycle_id,
                        'cycle_start_time': cycle.start_time,
                        'cycle_duration': cycle.duration,
                        'phase_number': phase.phase_number,
                        'phase_name': phase.phase_name,
                        'support_type': phase.support_type,
                        'phase_start_time': phase.start_time,
                        'phase_end_time': phase.end_time,
                        'phase_duration': phase.duration,
                        'phase_start_idx': phase.start_idx,
                        'phase_end_idx': phase.end_idx,
                        'percentage_of_cycle': (phase.duration / cycle.duration * 100) if cycle.duration > 0 else 0
                    })

            df_phases = pd.DataFrame(all_phases)
            phases_path = output_dir / f'{leg}_leg_gait_phases_detailed.csv'
            df_phases.to_csv(phases_path, index=False)

            print(f"    {leg.title()} leg: {len(all_phases)} phases across {len(cycles)} cycles → {phases_path.name}")

            # Print phase statistics
            if not df_phases.empty:
                phase_summary = df_phases.groupby('phase_name')['phase_duration'].agg(['mean', 'std', 'count'])
                print(f"      Phase timing summary for {leg} leg:")
                for phase_name, row in phase_summary.iterrows():
                    print(f"        {phase_name}: {row['mean']:.3f}s ± {row['std']:.3f}s (n={int(row['count'])})")

    def create_debug_event_validation_plot(self, df: pd.DataFrame, output_dir: Path):
        """
        Create debug visualization to validate precision event detection timing.

        Shows pressure signal with overlaid event markers for visual QA.
        """
        print("\n  Creating debug event validation plots...")

        debug_dir = output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        for leg in ['left', 'right']:
            leg_prefix = 'L' if leg == 'left' else 'R'
            pressure_col = f'{leg}_pressure_total'

            if pressure_col not in df.columns:
                continue

            # Get all events for this leg
            events = self.precision_detector.detect_all_events(df, leg)

            # Create figure
            fig, ax = plt.subplots(figsize=(16, 6), dpi=150)

            # Plot pressure signal
            timestamps = df['timestamp'].values
            pressure = df[pressure_col].values

            ax.plot(timestamps, pressure, color='black', linewidth=1, alpha=0.7,
                   label='Total Pressure')

            # Overlay heel strike events (green)
            for event in events['heel_strike']:
                ax.axvline(event.event_start, color='green', linestyle='--',
                          linewidth=1.5, alpha=0.7, label='HS' if event == events['heel_strike'][0] else '')
                ax.axvspan(event.event_start, event.event_end, color='green',
                          alpha=0.1)

            # Overlay mid-stance events (blue)
            for event in events['mid_stance']:
                ax.axvspan(event.event_start, event.event_end, color='blue',
                          alpha=0.2, label='MSt' if event == events['mid_stance'][0] else '')

            # Overlay toe-off events (red)
            for event in events['toe_off']:
                ax.axvline(event.event_end, color='red', linestyle='--',
                          linewidth=1.5, alpha=0.7, label='TO' if event == events['toe_off'][0] else '')
                ax.axvspan(event.event_start, event.event_end, color='red',
                          alpha=0.1)

            # Formatting
            ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Pressure (normalized)', fontsize=12, fontweight='bold')
            ax.set_title(f'{leg.title()} Leg - Event Validation (Pressure + Detected Events)',
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Legend (show only unique labels)
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)

            plt.tight_layout()

            # Save plot
            plot_path = debug_dir / f'event_validation_{leg}.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"    Saved: {plot_path.name}")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point with batch processing support."""
    parser = argparse.ArgumentParser(
        description='Smart Insole Gait Analysis - Complete Merged Version with Batch Processing'
    )

    parser.add_argument(
        '--input',
        type=str,
        help='Path to single input CSV file (optional, overrides batch mode)'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output directory for results (optional)'
    )

    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process all CSVs in INPUT_DIR'
    )

    parser.add_argument(
        '--sampling-rate',
        type=float,
        default=100.0,
        help='Sampling rate in Hz (default: 100)'
    )

    parser.add_argument(
        '--filter-cutoff',
        type=float,
        default=20.0,
        help='Filter cutoff frequency in Hz (default: 20.0)'
    )

    args = parser.parse_args()

    # Create configuration
    config = InsoleConfig(
        sampling_rate=args.sampling_rate,
        filter_cutoff=args.filter_cutoff
    )

    # Determine processing mode
    if args.batch or (not args.input and len(CSV_FILES) > 0):
        # Batch processing mode
        print(f"\n{'='*70}")
        print(f"BATCH PROCESSING MODE")
        print(f"{'='*70}")
        print(f"Input directory: {INPUT_DIR}")
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"Found {len(CSV_FILES)} CSV files\n")

        successful = 0
        failed = 0

        for csv_file in CSV_FILES:
            print(f"\n{'='*70}")
            print(f"Processing: {csv_file.name}")
            print(f"{'='*70}")

            output_subdir = OUTPUT_DIR / csv_file.stem
            output_subdir.mkdir(parents=True, exist_ok=True)

            try:
                pipeline = InsolePipeline(config)
                pipeline.analyze_insole_data(csv_file, output_subdir)
                print(f"Completed: {csv_file.name}")
                successful += 1
            except Exception as e:
                print(f"Failed: {csv_file.name} - {str(e)}")
                failed += 1
                continue

        print(f"\n{'='*70}")
        print(f"BATCH PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"  Total files: {len(CSV_FILES)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"{'='*70}\n")

    else:
        # Single file mode
        if not args.input:
            print("Error: No input file specified and no CSVs found in INPUT_DIR")
            print(f"       INPUT_DIR: {INPUT_DIR}")
            print("\nUsage:")
            print("  Single file: python3 insole-analysis.py --input <file.csv> --output <output_dir>")
            print("  Batch mode:  python3 insole-analysis.py --batch")
            return

        input_file = Path(args.input)
        if not input_file.exists():
            print(f"Error: Input file not found: {input_file}")
            return

        output_dir = Path(args.output) if args.output else OUTPUT_DIR / input_file.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"SINGLE FILE MODE")
        print(f"{'='*70}")
        print(f"Input file: {input_file}")
        print(f"Output directory: {output_dir}\n")

        pipeline = InsolePipeline(config)
        pipeline.analyze_insole_data(input_file, output_dir)

        print(f"\nAnalysis complete: {input_file.name}\n")


if __name__ == '__main__':
    main()
