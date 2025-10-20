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

INPUT_DIR = Path("/home/shivam/Desktop/Human_Pose/PROJECT CYCLOGRAM/insole-sample")
OUTPUT_DIR = Path("/home/shivam/Desktop/Human_Pose/PROJECT CYCLOGRAM/insole-output")

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
    max_cycle_duration: float = 2.0   # Maximum gait cycle duration
    min_stance_duration: float = 0.5  # Minimum stance phase duration
    min_swing_duration: float = 0.3   # Minimum swing phase duration

    # Phase detection thresholds
    pressure_threshold: float = 0.5    # Normalized pressure for contact detection
    pressure_derivative_threshold: float = 0.1  # For IC detection
    gyro_swing_threshold: float = 50.0  # deg/s for swing detection

    # Validation thresholds
    stance_swing_ratio_min: float = 1.2  # Expected 60:40 = 1.5
    stance_swing_ratio_max: float = 2.0
    bilateral_tolerance: float = 0.15    # 15% duration difference allowed

    # Visualization
    plot_dpi: int = 300
    cyclogram_resolution: int = 101  # Points per gait cycle


@dataclass
class GaitPhase:
    """Single gait phase annotation."""
    phase_name: str          # IC, LR, MSt, TSt, PSw, ISw, MSw, TSw
    leg: str                 # 'left' or 'right'
    start_time: float        # seconds
    end_time: float          # seconds
    start_idx: int           # frame index
    end_idx: int             # frame index
    duration: float          # seconds
    phase_number: int        # 1-8


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

        Expected columns:
        - timestamp: Time in milliseconds
        - L_value1-4, R_value1-4: Pressure sensors
        - L_ACC_X/Y/Z, R_ACC_X/Y/Z: Accelerometers
        - L_GYRO_X/Y/Z, R_GYRO_X/Y/Z: Gyroscopes
        """
        print(f"Loading insole data from {csv_path}")

        df = pd.read_csv(csv_path, skiprows=3)

        # Convert timestamp from ms to seconds
        if 'timestamp' in df.columns:
            df['timestamp'] = df['timestamp'] / 1000.0

        print(f"Loaded {len(df)} samples ({df['timestamp'].iloc[-1]:.1f} seconds)")

        # Verify required columns
        required_cols = self._get_required_columns()
        missing = [col for col in required_cols if col not in df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        return df

    def _get_required_columns(self) -> List[str]:
        """Define required columns for analysis."""
        pressure_cols = [f'L_value{i}' for i in range(1, 5)] + \
                       [f'R_value{i}' for i in range(1, 5)]

        acc_cols = [f'{side}_ACC_{axis}'
                   for side in ['L', 'R']
                   for axis in ['X', 'Y', 'Z']]

        gyro_cols = [f'{side}_GYRO_{axis}'
                    for side in ['L', 'R']
                    for axis in ['X', 'Y', 'Z']]

        return ['timestamp'] + pressure_cols + acc_cols + gyro_cols

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
        Detect initial contact (heel strike) events.

        Biomechanical signature: Sharp pressure rise
        """
        # Find peaks in pressure derivative (rapid increase)
        ic_candidates, _ = find_peaks(
            pressure_deriv,
            height=self.config.pressure_derivative_threshold,
            distance=int(self.config.sampling_rate * 0.5)  # Min 0.5s between ICs
        )

        # Filter by pressure level (must be significant contact)
        ic_indices = [idx for idx in ic_candidates
                     if pressure[idx] > self.config.pressure_threshold]

        return ic_indices

    def _detect_phases_in_cycle(self, df: pd.DataFrame, leg: str,
                                start_idx: int, end_idx: int,
                                start_time: float) -> List[GaitPhase]:
        """
        Detect all 8 gait phases using dynamic sensor-based segmentation.

        Uses high-precision pressure sensor events instead of time-based heuristics.
        """
        cycle_data = df.iloc[start_idx:end_idx].copy()
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
            phase_number=1
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
            phase_number=2
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
            phase_number=3
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
            phase_number=4
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
            phase_number=5
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
                phase_number=num
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
        start_time = timestamps[start_idx]
        end_time = timestamps[end_idx]
        duration = end_time - start_time

        return GaitPhase(
            phase_name=name,
            leg=leg,
            start_time=start_time,
            end_time=end_time,
            start_idx=start_idx,
            end_idx=end_idx,
            duration=duration,
            phase_number=phase_num
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
                           leg: str) -> List[CyclogramData]:
        """
        Generate multiple cyclogram types for each gait cycle.

        Cyclogram types:
        1. ACC_X vs ACC_Y (sagittal-frontal acceleration)
        2. ACC_Z vs GYRO_Y (vertical acc vs pitch rate)
        3. GYRO_X vs GYRO_Z (roll-yaw coordination)
        """
        cyclograms = []

        # Convert leg name to uppercase prefix
        leg_prefix = 'L' if leg == 'left' else 'R'

        cyclogram_types = [
            (f'{leg_prefix}_ACC_X_filt', f'{leg_prefix}_ACC_Y_filt', 'ACC_X', 'ACC_Y'),
            (f'{leg_prefix}_ACC_Z_filt', f'{leg_prefix}_GYRO_Y_filt', 'ACC_Z', 'GYRO_Y'),
            (f'{leg_prefix}_GYRO_X_filt', f'{leg_prefix}_GYRO_Z_filt', 'GYRO_X', 'GYRO_Z')
        ]

        for cycle in cycles:
            for x_col, y_col, x_label, y_label in cyclogram_types:
                cyclogram = self._extract_cyclogram(
                    df, cycle, x_col, y_col, x_label, y_label
                )
                cyclograms.append(cyclogram)

        return cyclograms

    def _extract_cyclogram(self, df: pd.DataFrame, cycle: GaitCycle,
                          x_col: str, y_col: str,
                          x_label: str, y_label: str) -> CyclogramData:
        """Extract single cyclogram for one cycle."""

        # Get cycle data
        start_idx = cycle.phases[0].start_idx
        end_idx = cycle.phases[-1].end_idx

        cycle_data = df.iloc[start_idx:end_idx]

        x_signal = cycle_data[x_col].values
        y_signal = cycle_data[y_col].values

        # Normalize to percentage of gait cycle
        normalized_length = self.config.cyclogram_resolution

        if len(x_signal) < 2:
            # Handle edge case
            x_signal_norm = np.zeros(normalized_length)
            y_signal_norm = np.zeros(normalized_length)
        else:
            time_original = np.linspace(0, 100, len(x_signal))
            time_normalized = np.linspace(0, 100, normalized_length)

            x_interp = interp1d(time_original, x_signal, kind='cubic',
                               fill_value='extrapolate')
            y_interp = interp1d(time_original, y_signal, kind='cubic',
                               fill_value='extrapolate')

            x_signal_norm = x_interp(time_normalized)
            y_signal_norm = y_interp(time_normalized)

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
            x_label=x_label,
            y_label=y_label,
            phase_indices=phase_indices,
            phase_labels=phase_labels,
            is_3d=False
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

    def compute_area(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute polygonal area enclosed by cyclogram using shoelace formula.

        Args:
            x: X-coordinates of cyclogram trajectory
            y: Y-coordinates of cyclogram trajectory

        Returns:
            Absolute area enclosed by the loop
        """
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def compute_curvature(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute curvature along 2D trajectory.

        Curvature κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)

        Args:
            x: X-coordinates of trajectory
            y: Y-coordinates of trajectory

        Returns:
            Curvature array along trajectory
        """
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = np.power(dx**2 + dy**2 + 1e-6, 1.5)

        curvature = numerator / denominator
        return curvature

    def compute_smoothness(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute smoothness coefficient as inverse of mean jerk.

        Smoothness = 1 / mean(|jerk|)
        Higher values indicate smoother, more controlled movement.

        Args:
            x: X-coordinates of trajectory
            y: Y-coordinates of trajectory

        Returns:
            Smoothness coefficient (higher = smoother)
        """
        # Compute jerk (third derivative of position)
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        # Jerk magnitude
        jerk = ddx**2 + ddy**2

        # Smoothness as inverse of mean jerk
        smoothness = 1.0 / (np.mean(jerk) + 1e-6)

        return smoothness

    def compare_cyclograms(self, left: CyclogramData, right: CyclogramData) -> Dict[str, float]:
        """
        Compare left vs right cyclogram geometry and morphology.

        Computes bilateral symmetry metrics:
        - Area symmetry: normalized difference in enclosed areas
        - Curvature symmetry: RMS difference in curvature profiles
        - Smoothness symmetry: difference in movement smoothness

        Args:
            left: Left leg cyclogram data
            right: Right leg cyclogram data

        Returns:
            Dictionary with symmetry metrics and individual leg measurements
        """
        # Compute geometric properties for each leg
        A_L = self.compute_area(left.x_signal, left.y_signal)
        A_R = self.compute_area(right.x_signal, right.y_signal)

        κ_L = self.compute_curvature(left.x_signal, left.y_signal)
        κ_R = self.compute_curvature(right.x_signal, right.y_signal)

        S_L = self.compute_smoothness(left.x_signal, left.y_signal)
        S_R = self.compute_smoothness(right.x_signal, right.y_signal)

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

    def __init__(self, config: InsoleConfig, output_dir: Path):
        self.config = config
        self.output_dir = output_dir

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

        self._setup_colors()

    def _setup_colors(self):
        """Define color scheme for phase segmentation."""
        # Stance phases: Blue gradient
        self.stance_colors = [
            '#08519c',  # IC - dark blue
            '#3182bd',  # LR
            '#6baed6',  # MSt
            '#9ecae1',  # TSt
            '#c6dbef'   # PSw - light blue
        ]

        # Swing phases: Red gradient
        self.swing_colors = [
            '#a50f15',  # ISw - dark red
            '#de2d26',  # MSw
            '#fc9272'   # TSw - light red
        ]

        self.phase_colors = self.stance_colors + self.swing_colors

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
        if 'cycles' in kwargs:
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
        - gyro_gait: 1x3 (XY/XZ/YZ) = 3 subplots
        - acc_gait: 1x3 = 3 subplots
        - 3d_gait: 1x2 (Gyro3D/Acc3D) = 2 subplots
        - gait_events: 1x2 (Left/Right timeline) = 2 subplots

        Returns:
            fig: Figure object (12×6 in @ 300 DPI)
            axes: Array of subplot axes
        """
        # Layout configurations
        layouts = {
            'gyro_stride': (2, 3, ["Left X-Y", "Left X-Z", "Left Y-Z", "Right X-Y", "Right X-Z", "Right Y-Z"]),
            'acc_stride': (2, 3, ["Left X-Y", "Left X-Z", "Left Y-Z", "Right X-Y", "Right X-Z", "Right Y-Z"]),
            '3d_stride': (2, 2, ["Left Gyro 3D", "Left Acc 3D", "Right Gyro 3D", "Right Acc 3D"]),
            'gyro_gait': (1, 3, ["X-Y Plane", "X-Z Plane", "Y-Z Plane"]),
            'acc_gait': (1, 3, ["X-Y Plane", "X-Z Plane", "Y-Z Plane"]),
            '3d_gait': (1, 2, ["Gyroscope 3D", "Accelerometer 3D"]),
            'gait_events': (1, 2, ["Left Leg Events", "Right Leg Events"])
        }

        if analysis_type not in layouts:
            raise ValueError(f"Unknown analysis_type: {analysis_type}")

        nrows, ncols, subplot_titles = layouts[analysis_type]

        # Create figure with standard dimensions
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 6), dpi=300)

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
        pair_labels = ['X-Y', 'X-Z', 'Y-Z']

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
                    ax.plot(x[start_idx:end_idx], y[start_idx:end_idx],
                           color=color, linewidth=1.5, label=cyclogram.phase_labels[i] if leg_idx == 0 and col_idx == 0 else "")

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

        # Add legend to first subplot
        if axes[0, 0].get_lines():
            axes[0, 0].legend(loc='upper right', fontsize=6, ncol=2)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return subplot_metrics

    def _plot_acc_stride_subplots(self, axes: np.ndarray, data_dict: Dict) -> List[Dict]:
        """
        Populate 2x3 grid with accelerometer cyclograms (stride level).
        Same structure as gyro but for ACC_X, ACC_Y, ACC_Z sensors.
        """
        subplot_metrics = []
        sensor_pairs = [('ACC_X', 'ACC_Y'), ('ACC_X', 'ACC_Z'), ('ACC_Y', 'ACC_Z')]
        pair_labels = ['X-Y', 'X-Z', 'Y-Z']

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
                    ax.plot(x[start_idx:end_idx], y[start_idx:end_idx],
                           color=color, linewidth=1.5)

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

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return subplot_metrics

    def _plot_gait_events_subplots(self, axes: np.ndarray, data_dict: Dict) -> List[Dict]:
        """
        Populate 1x2 grid with gait event timelines (left/right comparison).

        Shows heel strikes, toe-offs, and 8-phase boundaries across full trial.
        """
        subplot_metrics = []

        for leg_idx, leg in enumerate(['left', 'right']):
            ax = axes[0, leg_idx]
            events = data_dict.get(leg, {})

            heel_strikes = events.get('heel_strikes', [])
            toe_offs = events.get('toe_offs', [])
            phases = events.get('phases', [])

            if not heel_strikes:
                ax.text(0.5, 0.5, f'No events for {leg.title()} leg',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{leg.title()} Leg Events", fontsize=10, fontweight='bold')
                subplot_metrics.append({'subplot_index': leg_idx, 'data': None})
                continue

            # Plot heel strikes
            ax.scatter(heel_strikes, [1]*len(heel_strikes),
                      marker='v', s=100, c='green', label='Heel Strike', zorder=3)

            # Plot toe-offs
            ax.scatter(toe_offs, [1]*len(toe_offs),
                      marker='^', s=100, c='red', label='Toe-Off', zorder=3)

            # Plot phase backgrounds
            for phase in phases:
                start_t = phase.get('start_time', 0)
                end_t = phase.get('end_time', 0)
                phase_name = phase.get('phase_name', '')
                phase_idx = phase.get('phase_number', 0) - 1

                color = self.phase_colors[phase_idx % len(self.phase_colors)]
                ax.axvspan(start_t, end_t, alpha=0.2, color=color, label=phase_name if leg_idx == 0 else "")

            ax.set_xlabel('Time (s)', fontsize=8)
            ax.set_ylabel('Events', fontsize=8)
            ax.set_title(f"{leg.title()} Leg Events", fontsize=10, fontweight='bold')
            ax.set_ylim(0.5, 1.5)
            ax.set_yticks([])
            ax.grid(True, alpha=0.3, axis='x')
            ax.legend(loc='upper right', fontsize=6)

            subplot_metrics.append({
                'subplot_index': leg_idx,
                'position': [0, leg_idx],
                'title': f"{leg.title()} Events",
                'leg': leg,
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
        Populate 1x3 grid with gyroscopic cyclograms (gait level - combined L+R).

        Layout: XY / XZ / YZ planes with left (blue) and right (red) overlaid
        """
        subplot_metrics = []
        sensor_pairs = [('GYRO_X', 'GYRO_Y'), ('GYRO_X', 'GYRO_Z'), ('GYRO_Y', 'GYRO_Z')]
        pair_labels = ['X-Y Plane', 'X-Z Plane', 'Y-Z Plane']

        for col_idx, (pair, label) in enumerate(zip(sensor_pairs, pair_labels)):
            ax = axes[0, col_idx]

            # Plot left leg (blue)
            left_data = data_dict.get('left', {}).get(label)
            if left_data and hasattr(left_data, 'x_signal'):
                ax.plot(left_data.x_signal, left_data.y_signal,
                       color='blue', linewidth=2, label='Left', alpha=0.7)
                ax.plot(left_data.x_signal[0], left_data.y_signal[0], 'bo', markersize=8)

            # Plot right leg (red)
            right_data = data_dict.get('right', {}).get(label)
            if right_data and hasattr(right_data, 'x_signal'):
                ax.plot(right_data.x_signal, right_data.y_signal,
                       color='red', linewidth=2, label='Right', alpha=0.7)
                ax.plot(right_data.x_signal[0], right_data.y_signal[0], 'ro', markersize=8)

            ax.set_xlabel(pair[0], fontsize=9)
            ax.set_ylabel(pair[1], fontsize=9)
            ax.set_title(label, fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            ax.legend(loc='best', fontsize=8)

            subplot_metrics.append({
                'subplot_index': col_idx,
                'position': [0, col_idx],
                'title': f"Gyro {label}",
                'sensor_pair': list(pair),
                'bilateral': True
            })

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return subplot_metrics

    def _plot_acc_gait_subplots(self, axes: np.ndarray, data_dict: Dict) -> List[Dict]:
        """
        Populate 1x3 grid with accelerometer cyclograms (gait level - combined L+R).

        Same structure as gyro_gait but for accelerometer sensors.
        """
        subplot_metrics = []
        sensor_pairs = [('ACC_X', 'ACC_Y'), ('ACC_X', 'ACC_Z'), ('ACC_Y', 'ACC_Z')]
        pair_labels = ['X-Y Plane', 'X-Z Plane', 'Y-Z Plane']

        for col_idx, (pair, label) in enumerate(zip(sensor_pairs, pair_labels)):
            ax = axes[0, col_idx]

            # Plot left leg (blue)
            left_data = data_dict.get('left', {}).get(label)
            if left_data and hasattr(left_data, 'x_signal'):
                ax.plot(left_data.x_signal, left_data.y_signal,
                       color='blue', linewidth=2, label='Left', alpha=0.7)
                ax.plot(left_data.x_signal[0], left_data.y_signal[0], 'bo', markersize=8)

            # Plot right leg (red)
            right_data = data_dict.get('right', {}).get(label)
            if right_data and hasattr(right_data, 'x_signal'):
                ax.plot(right_data.x_signal, right_data.y_signal,
                       color='red', linewidth=2, label='Right', alpha=0.7)
                ax.plot(right_data.x_signal[0], right_data.y_signal[0], 'ro', markersize=8)

            ax.set_xlabel(pair[0], fontsize=9)
            ax.set_ylabel(pair[1], fontsize=9)
            ax.set_title(label, fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            ax.legend(loc='best', fontsize=8)

            subplot_metrics.append({
                'subplot_index': col_idx,
                'position': [0, col_idx],
                'title': f"Acc {label}",
                'sensor_pair': list(pair),
                'bilateral': True
            })

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return subplot_metrics

    def _plot_3d_gait_subplots(self, axes: np.ndarray, data_dict: Dict) -> List[Dict]:
        """
        Populate 1x2 grid with 3D cyclograms (gait level - combined L+R).

        Layout: Gyro 3D | Acc 3D (both with left and right overlaid)
        """
        from mpl_toolkits.mplot3d import Axes3D
        subplot_metrics = []

        sensor_types = ['gyro', 'acc']
        sensor_labels = ['Gyroscope 3D', 'Accelerometer 3D']

        for col_idx, (sensor_type, label) in enumerate(zip(sensor_types, sensor_labels)):
            # Create 3D axis
            ax = plt.subplot(1, 2, col_idx + 1, projection='3d')

            # Plot left leg (blue)
            left_3d = data_dict.get('left', {}).get(f'{sensor_type}_3d')
            if left_3d and hasattr(left_3d, 'x_signal'):
                ax.plot(left_3d.x_signal, left_3d.y_signal,
                       left_3d.z_signal if hasattr(left_3d, 'z_signal') else np.zeros_like(left_3d.x_signal),
                       color='blue', linewidth=2, label='Left', alpha=0.7)

            # Plot right leg (red)
            right_3d = data_dict.get('right', {}).get(f'{sensor_type}_3d')
            if right_3d and hasattr(right_3d, 'x_signal'):
                ax.plot(right_3d.x_signal, right_3d.y_signal,
                       right_3d.z_signal if hasattr(right_3d, 'z_signal') else np.zeros_like(right_3d.x_signal),
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

            subplot_metrics.append({
                'subplot_index': col_idx,
                'position': [0, col_idx],
                'title': label,
                'sensor_type': sensor_type,
                'bilateral': True
            })

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return subplot_metrics

    def _compute_cyclogram_metrics(self, cyclogram: 'CyclogramData') -> Dict:
        """Helper to compute standard metrics for a cyclogram."""
        x, y = cyclogram.x_signal, cyclogram.y_signal

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

        return {
            'area': float(area),
            'perimeter': float(perimeter),
            'compactness': float(compactness),
            'closure_error': float(closure_error)
        }

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

        metadata = self._generate_metadata(
            analysis_type=analysis_type,
            leg='bilateral',
            sensor_type=analysis_type.split('_')[0],  # gyro/acc/3d/gait
            file_name=f"{base_name}.png",
            subject=subject_name,
            subplot_metrics=subplot_metrics
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

        # Initialize visualizer
        self.visualizer = InsoleVisualizer(self.config, output_dir)

        # Stage 1: Load data
        df = self.data_handler.load_data(input_csv)
        col_map = self.data_handler.detect_column_mapping(df)

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
            df, left_cycles, 'left'
        )
        right_cyclograms = self.cyclogram_generator.generate_cyclograms(
            df, right_cycles, 'right'
        )
        print(f"  Generated {len(left_cyclograms)} left cyclograms")
        print(f"  Generated {len(right_cyclograms)} right cyclograms")

        # Stage 6: MMC computation
        print("\nComputing Morphological Mean Cyclograms...")
        left_mmc = self._generate_mmc(left_cyclograms, 'left')
        right_mmc = self._generate_mmc(right_cyclograms, 'right')

        # Stage 7: Validation
        print("\nValidating anatomical plausibility...")
        validation = self.validator.validate_gait_cycles(left_cycles, right_cycles)

        self._print_validation_results(validation)

        # Stage 8: Bilateral Symmetry Analysis
        print("\nComputing bilateral symmetry metrics...")
        symmetry_metrics = self._compute_symmetry(left_cyclograms, right_cyclograms)

        # Stage 9: Visualization
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\nGenerating visualizations with PNG+JSON output...")

        # Plot individual cyclograms (first 3 of each type)
        cyclogram_types = {}
        for cyclogram in left_cyclograms + right_cyclograms:
            key = (cyclogram.cycle.leg, cyclogram.x_label, cyclogram.y_label)
            if key not in cyclogram_types:
                cyclogram_types[key] = []
            cyclogram_types[key].append(cyclogram)

        for (leg, x_label, y_label), cyclograms in cyclogram_types.items():
            # Plot first 3 individual cycles
            for i, cyclogram in enumerate(cyclograms[:3]):
                output_path = output_dir / f"{leg}_{x_label}_vs_{y_label}_cycle_{i+1}.png"
                metrics = self.cyclogram_generator.compute_cyclogram_metrics(cyclogram)
                mmc = left_mmc if leg == 'left' else right_mmc
                self.visualizer.plot_phase_segmented_cyclogram(cyclogram, output_path, mmc, metrics)

            # Plot aggregated mean
            output_path = output_dir / f"{leg}_{x_label}_vs_{y_label}_mean.png"
            self.visualizer.plot_aggregated_cyclogram(cyclograms, output_path)

        # ====================================================================
        # SUBPLOT FIGURE GENERATION (Organized Multi-Panel Visualizations)
        # ====================================================================
        print("\nGenerating organized subplot figures...")
        subject_name = input_csv.stem  # Extract subject name from filename

        self._generate_subplot_figures(
            left_cyclograms, right_cyclograms,
            left_cycles, right_cycles,
            subject_name
        )

        # Save summary
        self._save_summary(left_cycles, right_cycles, validation, output_dir)

        # Save symmetry metrics
        self._save_symmetry_results(symmetry_metrics, output_dir)

        # Save precision events if available
        if self.precision_detector is not None:
            self._save_precision_events(df, output_dir)
            # Create debug validation plots
            self.create_debug_event_validation_plot(df, output_dir)

        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)

    def _generate_mmc(self, cyclograms: List[CyclogramData], leg: str) -> Optional[MorphologicalMeanCyclogram]:
        """Compute MMC for cyclograms of same sensor pair."""
        # Group by sensor pair
        by_pair = {}
        for c in cyclograms:
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
        """Save analysis summary to CSV."""

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

    def _compute_symmetry(self, left_cyclograms: List[CyclogramData],
                         right_cyclograms: List[CyclogramData]) -> pd.DataFrame:
        """
        Compute bilateral symmetry metrics for all cyclogram pairs.

        Args:
            left_cyclograms: List of left leg cyclograms
            right_cyclograms: List of right leg cyclograms

        Returns:
            DataFrame with per-cycle symmetry metrics
        """
        symmetry_df = self.symmetry_analyzer.compute_batch_symmetry(
            left_cyclograms, right_cyclograms
        )

        # Print summary statistics
        if not symmetry_df.empty:
            print(f"  Computed symmetry for {len(symmetry_df)} cyclogram pairs")

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

    def _generate_subplot_figures(self, left_cyclograms: List[CyclogramData],
                                  right_cyclograms: List[CyclogramData],
                                  left_cycles: List[GaitCycle],
                                  right_cycles: List[GaitCycle],
                                  subject_name: str):
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
        # Organize cyclograms by sensor type and leg
        cyclogram_dict = {
            'left': {},
            'right': {}
        }

        # Group cyclograms by sensor pair labels
        for cyclogram in left_cyclograms + right_cyclograms:
            leg = cyclogram.cycle.leg
            key = f"{cyclogram.x_label}_vs_{cyclogram.y_label}"

            if key not in cyclogram_dict[leg]:
                cyclogram_dict[leg][key] = cyclogram

        # Extract first cyclogram of each type for each leg
        left_data = {}
        right_data = {}

        # Map sensor pair labels to subplot labels
        sensor_mapping = {
            'GYRO_X_vs_GYRO_Y': 'X-Y',
            'GYRO_X_vs_GYRO_Z': 'X-Z',
            'GYRO_Y_vs_GYRO_Z': 'Y-Z',
            'ACC_X_vs_ACC_Y': 'X-Y',
            'ACC_X_vs_ACC_Z': 'X-Z',
            'ACC_Y_vs_ACC_Z': 'Y-Z'
        }

        for key, cyclogram in cyclogram_dict['left'].items():
            label = sensor_mapping.get(key, key)
            left_data[label] = cyclogram

        for key, cyclogram in cyclogram_dict['right'].items():
            label = sensor_mapping.get(key, key)
            right_data[label] = cyclogram

        # =================================================================
        # 1. GYROSCOPIC STRIDE CYCLOGRAMS (if available)
        # =================================================================
        if any('GYRO' in k for k in cyclogram_dict['left'].keys()):
            try:
                data_dict = {'left': left_data, 'right': right_data}
                fig, metadata, base_name = self.visualizer.create_and_populate_subplot_figure(
                    'gyro_stride', data_dict, subject_name
                )
                self.visualizer.save_outputs(fig, metadata, base_name)
                plt.close(fig)
                print(f"  ✓ Generated: Gyroscopic Stride Cyclograms")
            except Exception as e:
                print(f"  ✗ Skipped Gyroscopic Stride: {str(e)}")

        # =================================================================
        # 2. ACCELEROMETER STRIDE CYCLOGRAMS (if available)
        # =================================================================
        if any('ACC' in k for k in cyclogram_dict['left'].keys()):
            try:
                data_dict = {'left': left_data, 'right': right_data}
                fig, metadata, base_name = self.visualizer.create_and_populate_subplot_figure(
                    'acc_stride', data_dict, subject_name
                )
                self.visualizer.save_outputs(fig, metadata, base_name)
                plt.close(fig)
                print(f"  ✓ Generated: Accelerometer Stride Cyclograms")
            except Exception as e:
                print(f"  ✗ Skipped Accelerometer Stride: {str(e)}")

        # =================================================================
        # 3. 3D STRIDE CYCLOGRAMS (if 3D data available)
        # =================================================================
        # Note: 3D cyclograms require specific 3D trajectory data structure
        # This can be extended when 3D cyclogram generation is implemented

        # =================================================================
        # 4. GAIT EVENT TIMELINE
        # =================================================================
        try:
            event_data = {
                'left': left_cycles,
                'right': right_cycles
            }
            fig, metadata, base_name = self.visualizer.create_and_populate_subplot_figure(
                'gait_events', event_data, subject_name
            )
            self.visualizer.save_outputs(fig, metadata, base_name)
            plt.close(fig)
            print(f"  ✓ Generated: Gait Event Timeline")
        except Exception as e:
            print(f"  ✗ Skipped Gait Event Timeline: {str(e)}")

        print(f"  Subplot figure generation complete")

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
