# SmartInsole-Cyclogram

**Advanced Gait Analysis System Using Cyclograms for Biomechanical Assessment**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

A comprehensive research pipeline for analyzing human gait patterns using **cyclograms** (angle-angle phase diagrams) from dual sensor modalities: MediaPipe pose estimation and smart insole pressure/IMU sensors. This system enables quantitative assessment of bilateral symmetry, gait coordination, and recovery patterns in lower extremity pathologies.

### Key Features

- **Dual-Modality Analysis**: Process both pose-based (MediaPipe) and pressure-based (smart insole) gait data
- **Cyclogram Generation**: Automated extraction of angle-angle phase plots from gait cycles
- **Advanced Metrics**: 30+ biomechanical metrics including area, compactness, curvature, temporal coupling
- **Morphological Mean Cyclogram (MMC)**: Robust median-based averaging resistant to outliers
- **Statistical Phenotyping**: Cluster analysis for gait impairment pattern discovery
- **GPU Acceleration**: 30-40× speedup for MMC computation using CuPy (optional)
- **Publication-Ready Output**: High-resolution plots (300 DPI) with comprehensive metadata

## Research Applications

- **Clinical Gait Analysis**: Quantitative assessment of pathological gait patterns
- **Injury Recovery**: Track bilateral asymmetry during rehabilitation
- **Fracture Research**: Compare injured vs. contralateral limb coordination
- **Phenotype Discovery**: Identify distinct gait impairment clusters
- **Longitudinal Studies**: Monitor gait changes over time with robust metrics

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│         CYCLOGRAM GENERATION PIPELINE               │
├─────────────────────────────────────────────────────┤
│  Raw Sensor Data (Pose/Insole)                     │
│    ↓                                                │
│  Gait Event Detection (Heel Strike, Toe Off)       │
│    ↓                                                │
│  Quality Control & Enhancement                      │
│    ↓                                                │
│  Cyclogram Extraction (Angle-Angle Diagrams)       │
│    ↓                                                │
│  Metrics Computation (Area, Symmetry, Curvature)   │
│    ↓                                                │
│  Morphological Mean Cyclogram (MMC)                │
│    ↓                                                │
│  Export (PNG + JSON metadata)                      │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│         STATISTICAL ANALYSIS PIPELINE               │
├─────────────────────────────────────────────────────┤
│  Excel Cyclogram Metrics                           │
│    ↓                                                │
│  Effect Feature Engineering (Δ/ρ transforms)       │
│    ↓                                                │
│  Preprocessing (Imputation, Scaling)               │
│    ↓                                                │
│  Multi-Algorithm Clustering                        │
│    ↓                                                │
│  Phenotype Characterization                        │
│    ↓                                                │
│  Statistical Testing (FDR-corrected)               │
│    ↓                                                │
│  Publication Figures                               │
└─────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.12+
- NVIDIA GPU + CUDA 12.x (optional, for GPU acceleration)

### Setup

```bash
# Clone repository
git clone https://github.com/silverline-art/SmartInsole-Cyclogram.git
cd SmartInsole-Cyclogram

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Optional: GPU acceleration (requires NVIDIA GPU + CUDA)
pip install cupy-cuda12x

# Optional: Fast DTW for temporal metrics
pip install fastdtw
```

## Quick Start

### Pose-Based Cyclogram Analysis

```bash
# Analyze all subjects (publication mode: 10×8" @ 300 DPI)
python3 "CYCLOGRAM-PROCESSING Script/Pose-Analysis.py"

# Single subject analysis
python3 "CYCLOGRAM-PROCESSING Script/Pose-Analysis.py" \
  --subject-name "Openpose_subject_001"

# Custom parameters
python3 "CYCLOGRAM-PROCESSING Script/Pose-Analysis.py" \
  --smooth-window 15 \
  --smooth-threshold 7.0 \
  --lock-axis-limits
```

### Insole Pressure Analysis

```bash
# Single file
python3 "CYCLOGRAM-PROCESSING Script/insole-analysis.py" \
  --input insole-sample/10MWT.csv \
  --output insole-output/10MWT

# Batch processing
python3 "CYCLOGRAM-PROCESSING Script/insole-analysis.py" --batch
```

### Statistical Analysis

```bash
cd "DATA ANALYSIS Script"

# Run clustering analysis
python3 injured_vs_contralateral_cluster_analysis.py

# Generate publication statistics
python3 cyclogram_publication_analysis.py

# Create characterization figures
python3 cluster_analysis_complete.py
```

## Sample Data

The repository includes a sample 10-meter walk test (10MWT) dataset:

```
insole-sample/
└── 10MWT.csv  # Smart insole sensor data (100 Hz sampling)
```

**Data Format**: Synchronized pressure sensors (4 per foot) + 3-axis accelerometers + 3-axis gyroscopes

## Output Structure

### Cyclogram Generation Outputs

```
Output/SUBJECT_ID/
├── CK_hip_knee_AllStrides.png          # Hip-knee cyclograms
├── CK_knee_ankle_AllStrides.png        # Knee-ankle cyclograms
├── LR_Similarity_Summary.png           # Bilateral symmetry visualization
├── cyclogram_stride_metrics.csv        # Per-stride quantitative metrics
└── cyclogram_session_summary.csv       # Session-level statistics
```

### Insole Analysis Outputs

```
insole-output/SUBJECT_ID/
├── plots/
│   ├── gait_phases/              # Gait event timelines
│   ├── stride_cyclograms/        # Individual stride cyclograms
│   ├── gait_cyclograms/          # Multi-cycle overlays with MMC
│   └── symmetry/                 # Bilateral comparison
├── json/                         # Metadata for all plots
├── precision_gait_events.csv     # High-precision event timing
└── detailed_gait_phases_*.csv    # 8-phase gait annotations
```

## Key Metrics

### Level 1: Geometric/Morphology
- **Area**: Cyclogram enclosed area (range of motion proxy)
- **Perimeter**: Trajectory length (coordination complexity)
- **Compactness**: `4πA/P²` (shape efficiency, 0-1 scale)
- **Eccentricity**: Ellipse fit eccentricity (directional bias)
- **Curvature**: Mean and variability of trajectory curvature

### Level 2: Temporal-Coupling
- **Continuous Relative Phase (CRP)**: Hilbert transform-based phase coupling
- **MARP**: Mean absolute relative phase (coordination strength)
- **Coupling Angle Variability**: Stability of inter-limb coordination
- **Phase Shift**: Timing lag between signals

### Level 3: Bilateral Symmetry
- **Symmetry Index**: `[(L-R)/mean(L,R)] × 100`
- **Mirror Correlation**: Correlation of horizontally mirrored cyclograms
- **RMS Trajectory Difference**: Functional asymmetry measure

## Methodology

### Cyclogram Definition

A **cyclogram** is an angle-angle phase diagram that visualizes joint coordination by plotting one joint angle against another throughout a complete gait cycle (heel strike → heel strike).

- **X-axis**: Proximal joint (e.g., hip flexion)
- **Y-axis**: Distal joint (e.g., knee flexion)
- **Trajectory**: Temporal evolution of joint coordination
- **Loop Closure**: Naturally closed for valid gait cycles

### Morphological Mean Cyclogram (MMC)

Robust alternative to naive averaging that preserves gait morphology:

1. **Median Reference Selection**: DTW-based selection of representative cycle
2. **Centroid Alignment**: Center all loops to remove translation variance
3. **Area Rescaling**: Normalize to median area for shape comparison
4. **Median Shape**: Point-wise median trajectory (resistant to outliers)
5. **Variance Envelope**: ±SD bands for cycle-to-cycle variability
6. **Shape Dispersion Index (SDI)**: Quantifies consistency (< 0.1 = excellent)

**Advantages over mean**: Robust to outliers, preserves biomechanical shape, quantifies variability

## Citation

If you use this software in your research, please cite:

```bibtex
@software{smartinsole_cyclogram,
  title = {SmartInsole-Cyclogram: Advanced Gait Analysis Using Cyclograms},
  author = {[Your Research Group]},
  year = {2025},
  url = {https://github.com/silverline-art/SmartInsole-Cyclogram},
  version = {1.0.0}
}
```

## Project Structure

```
SmartInsole-Cyclogram/
├── CYCLOGRAM-PROCESSING Script/     # Raw data → cyclograms
│   ├── Pose-Analysis.py            # MediaPipe pose analysis
│   ├── insole-analysis.py          # Smart insole analysis
│   └── gpu_acceleration.py         # CuPy GPU acceleration
├── DATA ANALYSIS Script/           # Cyclogram metrics → statistics
│   ├── cyclogram_publication_analysis.py
│   ├── injured_vs_contralateral_cluster_analysis.py
│   └── cluster_analysis_complete.py
├── claudedocs/                     # Technical documentation
│   ├── CLAUDE.md                   # Development guide
│   ├── cyclogram_system_design.md  # Architecture spec
│   └── MMC_METHODOLOGY.md          # MMC algorithm details
├── insole-sample/                  # Sample dataset
│   └── 10MWT.csv                   # 10-meter walk test
├── CLAUDE.md                       # AI assistant guide
├── README.md                       # This file
└── requirements.txt                # Python dependencies
```

## Technical Details

### Data Processing Pipeline

**Quality Control Gates**:
1. Coverage: ≥70% non-NaN frames per joint
2. Gaps: Max contiguous NaN ≤30 frames
3. Stability: Pelvis motion std ≤15 pixels
4. Sanity: Max frame-to-frame change ≤45°

**Angle Enhancement** (automatic for poor quality data):
- Tier 1: PCHIP interpolation for small gaps
- Tier 2: Geometric recalculation from landmarks
- Tier 3: Temporal smoothing with jump limiting

**Statistical Analysis**:
- Effect features: Delta (Δ) for bounded, Rho (ρ) for unbounded metrics
- Preprocessing: KNN imputation, robust scaling, collinearity removal
- Clustering: GMM, K-Means, Hierarchical with bootstrap validation
- Testing: FDR-corrected (Benjamini-Hochberg) for all p-values

### Performance

**Pose Analysis**:
- Runtime: ~45-75 sec/subject (with enhancement)
- Memory: ~200-500 MB/subject
- Bottleneck: DTW computation (use `fastdtw` for acceleration)

**Insole Analysis**:
- Runtime: ~8 sec/subject (GPU), ~23 sec (CPU)
- Memory: ~50 MB/file
- GPU Speedup: 30-40× for MMC computation

**Statistical Analysis**:
- Runtime: ~2-5 min (143 patients, 4 planes)
- Memory: ~1 GB peak
- Bottleneck: Bootstrap clustering (100 iterations)

## GPU Acceleration

Optional CuPy acceleration provides 30-40× speedup for MMC computation:

```bash
# Check GPU availability
python3 -c "from CYCLOGRAM-PROCESSING\ Script.gpu_acceleration import is_gpu_available; print(is_gpu_available())"

# Install CuPy (requires NVIDIA GPU + CUDA 12.x)
pip install cupy-cuda12x
```

## Documentation

- **CLAUDE.md**: Comprehensive development guide for AI assistants
- **claudedocs/**: Technical documentation
  - `cyclogram_system_design.md`: System architecture
  - `MMC_METHODOLOGY.md`: Morphological Mean Cyclogram algorithm
  - `GPU_ACCELERATION_GUIDE.md`: CuPy setup and usage
  - `PUBLICATION_MODE_IMPLEMENTATION.md`: Publication-quality plots

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MediaPipe for pose estimation framework
- SciPy community for signal processing tools
- CuPy developers for GPU acceleration support
- Research participants who contributed gait data

## Contact

For questions or collaboration inquiries, please open an issue on GitHub.

---

**Note**: This repository contains scripts and sample data for academic publication. Complete datasets are available upon request for research purposes.
