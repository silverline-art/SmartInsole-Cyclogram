#!/usr/bin/env python3
"""
Test script for GPU-accelerated MMC computation.

Validates:
1. GPU module imports correctly
2. CPU and GPU produce numerically equivalent results
3. GPU provides expected speedup
4. Fallback mechanism works correctly
"""

import numpy as np
import sys
import time
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from gpu_acceleration import (
        is_gpu_available,
        compute_mmc_gpu,
        benchmark_cpu_vs_gpu,
        GPU_AVAILABLE
    )
    print("✓ GPU module imported successfully")
except ImportError as e:
    print(f"✗ Failed to import GPU module: {e}")
    sys.exit(1)

try:
    from insole_data_process import (
        InsoleConfig,
        MorphologicalMeanCyclogramComputer,
        CyclogramData,
        GaitCycle
    )
    print("✓ Insole analysis module imported successfully")
except ImportError:
    # Fallback: try insole-analysis.py
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "insole_analysis",
            Path(__file__).parent / "insole-analysis.py"
        )
        insole_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(insole_module)

        InsoleConfig = insole_module.InsoleConfig
        MorphologicalMeanCyclogramComputer = insole_module.MorphologicalMeanCyclogramComputer
        CyclogramData = insole_module.CyclogramData
        GaitCycle = insole_module.GaitCycle
        print("✓ Insole analysis module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import insole analysis module: {e}")
        sys.exit(1)


def generate_test_cyclograms(n_cycles: int = 20, n_points: int = 101) -> list:
    """Generate synthetic gait cyclograms for testing."""
    cyclograms = []

    for i in range(n_cycles):
        # Generate realistic cyclogram shape (elliptical with some noise)
        t = np.linspace(0, 2*np.pi, n_points)

        # Base ellipse with some variation per cycle
        a = 50 + np.random.randn() * 5  # x-axis radius
        b = 30 + np.random.randn() * 3  # y-axis radius
        phase = np.random.randn() * 0.2  # phase shift

        x = a * np.cos(t + phase) + np.random.randn(n_points) * 2
        y = b * np.sin(t + phase) + np.random.randn(n_points) * 2

        # Create minimal dummy gait cycle
        dummy_cycle = GaitCycle(
            leg='left',
            cycle_id=i,
            start_time=i * 1.0,
            end_time=(i + 1) * 1.0,
            duration=1.0,
            phases=[],
            stance_duration=0.6,
            swing_duration=0.4,
            stance_swing_ratio=1.5
        )

        cyclogram = CyclogramData(
            cycle=dummy_cycle,
            x_signal=x,
            y_signal=y,
            z_signal=None,
            x_label='gyro_x',
            y_label='gyro_y',
            is_3d=False,
            phase_indices=[],
            phase_labels=[]
        )

        cyclograms.append(cyclogram)

    return cyclograms


def test_gpu_availability():
    """Test 1: Check GPU availability."""
    print("\n" + "="*60)
    print("TEST 1: GPU Availability")
    print("="*60)

    print(f"GPU module available: {GPU_AVAILABLE}")
    print(f"GPU device accessible: {is_gpu_available()}")

    if not is_gpu_available():
        print("⚠ GPU not available. Tests will verify fallback behavior only.")
        return False
    else:
        print("✓ GPU is available and accessible")
        return True


def test_numerical_equivalence(gpu_available: bool):
    """Test 2: Verify CPU and GPU produce equivalent results."""
    print("\n" + "="*60)
    print("TEST 2: Numerical Equivalence (CPU vs GPU)")
    print("="*60)

    # Generate test data
    n_cycles = 30
    test_cyclograms = generate_test_cyclograms(n_cycles)
    print(f"Generated {n_cycles} test cyclograms")

    # CPU computation
    config_cpu = InsoleConfig()
    config_cpu.use_gpu = False
    computer_cpu = MorphologicalMeanCyclogramComputer(config_cpu)

    print("Computing MMC on CPU...")
    start_cpu = time.perf_counter()
    result_cpu = computer_cpu.compute_mmc(test_cyclograms)
    time_cpu = time.perf_counter() - start_cpu
    print(f"  CPU time: {time_cpu:.3f} seconds")

    if not gpu_available:
        print("⚠ Skipping GPU comparison (GPU not available)")
        return

    # GPU computation
    config_gpu = InsoleConfig()
    config_gpu.use_gpu = True
    computer_gpu = MorphologicalMeanCyclogramComputer(config_gpu)

    print("Computing MMC on GPU...")
    start_gpu = time.perf_counter()
    result_gpu = computer_gpu.compute_mmc(test_cyclograms)
    time_gpu = time.perf_counter() - start_gpu
    print(f"  GPU time: {time_gpu:.3f} seconds")

    # Compare results
    if result_cpu is None or result_gpu is None:
        print("✗ One or both computations failed")
        return

    # Compare median trajectories
    diff_median = np.abs(result_cpu.median_trajectory - result_gpu.median_trajectory)
    max_diff = np.max(diff_median)
    mean_diff = np.mean(diff_median)

    print(f"\nNumerical Differences:")
    print(f"  Max difference: {max_diff:.6e}")
    print(f"  Mean difference: {mean_diff:.6e}")
    print(f"  Relative error: {mean_diff / np.mean(np.abs(result_cpu.median_trajectory)):.6e}")

    # Compare SDI
    sdi_diff = abs(result_cpu.shape_dispersion_index - result_gpu.shape_dispersion_index)
    print(f"  SDI difference: {sdi_diff:.6e}")

    # Tolerance check
    tolerance = 1e-6
    if max_diff < tolerance and sdi_diff < tolerance:
        print(f"✓ Results are numerically equivalent (tolerance: {tolerance})")
    else:
        print(f"⚠ Results differ beyond tolerance (tolerance: {tolerance})")

    # Speedup
    speedup = time_cpu / time_gpu
    print(f"\nPerformance:")
    print(f"  Speedup: {speedup:.1f}×")

    if speedup > 5:
        print(f"✓ Significant speedup achieved")
    else:
        print(f"⚠ Speedup lower than expected (target: >5×)")


def test_fallback_mechanism():
    """Test 3: Verify graceful fallback when GPU unavailable."""
    print("\n" + "="*60)
    print("TEST 3: Fallback Mechanism")
    print("="*60)

    test_cyclograms = generate_test_cyclograms(10)

    # Force fallback by requesting GPU when it may not be available
    config = InsoleConfig()
    config.use_gpu = True
    computer = MorphologicalMeanCyclogramComputer(config)

    print(f"GPU requested: {config.use_gpu}")
    print(f"GPU actually used: {computer.use_gpu}")

    result = computer.compute_mmc(test_cyclograms)

    if result is not None:
        print("✓ Computation succeeded (CPU or GPU)")
        print(f"  - Median trajectory shape: {result.median_trajectory.shape}")
        print(f"  - Shape Dispersion Index: {result.shape_dispersion_index:.4f}")
    else:
        print("✗ Computation failed")


def main():
    """Run all tests."""
    print("GPU-Accelerated MMC Test Suite")
    print("="*60)

    # Test 1: GPU availability
    gpu_available = test_gpu_availability()

    # Test 2: Numerical equivalence
    test_numerical_equivalence(gpu_available)

    # Test 3: Fallback mechanism
    test_fallback_mechanism()

    print("\n" + "="*60)
    print("All tests completed")
    print("="*60)


if __name__ == "__main__":
    main()
