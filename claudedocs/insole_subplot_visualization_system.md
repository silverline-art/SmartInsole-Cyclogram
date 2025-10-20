# Smart Insole Subplot Visualization System

## Overview

Comprehensive multi-panel visualization system for smart insole gait analysis, generating organized subplot figures with PNG+JSON output pairs for clinical assessment and EXMO portal ingestion.

**Date**: 2025-10-20
**File**: `Code-Script/insole-analysis.py`
**System**: Step-Cyclogram Integrated Analysis

---

## Architecture

### Directory Structure

```
insole-output/
├── summary/                    # Analysis summaries and aggregates
├── plots/                      # All PNG visualizations
│   ├── gait_phases/           # Gait event timelines
│   ├── stride_cyclograms/     # Stride-level cyclograms
│   ├── gait_cyclograms/       # Gait-level (aggregated) cyclograms
│   ├── mean_cyclograms/       # Morphological mean cyclograms
│   └── symmetry/              # Bilateral symmetry analyses
└── json/                       # Metadata companions
    ├── gait_phases/
    ├── stride_cyclograms/
    ├── gait_cyclograms/
    ├── mean_cyclograms/
    └── symmetry/
```

**Key Principle**: Every PNG has a JSON twin in the mirrored directory structure.

---

## Subplot Figure Types

### 1. Gyroscopic Stride Cyclograms
**Layout**: 2×3 grid (6 subplots)
**Organization**: Left leg (row 0) vs Right leg (row 1) × X-Y / X-Z / Y-Z (cols 0-2)

| Subplot | Position | Content |
|---------|----------|---------|
| [0, 0] | Top-left | Left Leg X-Y cyclogram |
| [0, 1] | Top-center | Left Leg X-Z cyclogram |
| [0, 2] | Top-right | Left Leg Y-Z cyclogram |
| [1, 0] | Bottom-left | Right Leg X-Y cyclogram |
| [1, 1] | Bottom-center | Right Leg X-Z cyclogram |
| [1, 2] | Bottom-right | Right Leg Y-Z cyclogram |

**File naming**: `gyro_stride_{subject_name}_{timestamp}.png`

### 2. Accelerometer Stride Cyclograms
**Layout**: 2×3 grid (6 subplots)
**Organization**: Same as gyroscopic but for accelerometer sensors

| Subplot | Position | Content |
|---------|----------|---------|
| [0, 0] | Top-left | Left Leg ACC X-Y |
| [0, 1] | Top-center | Left Leg ACC X-Z |
| [0, 2] | Top-right | Left Leg ACC Y-Z |
| [1, 0] | Bottom-left | Right Leg ACC X-Y |
| [1, 1] | Bottom-center | Right Leg ACC X-Z |
| [1, 2] | Bottom-right | Right Leg ACC Y-Z |

**File naming**: `acc_stride_{subject_name}_{timestamp}.png`

### 3. 3D Stride Cyclograms
**Layout**: 2×2 grid (4 subplots)
**Organization**: Left/Right × Gyro3D/Acc3D

| Subplot | Position | Content |
|---------|----------|---------|
| [0, 0] | Top-left | Left Leg Gyro 3D trajectory |
| [0, 1] | Top-right | Left Leg Acc 3D trajectory |
| [1, 0] | Bottom-left | Right Leg Gyro 3D trajectory |
| [1, 1] | Bottom-right | Right Leg Acc 3D trajectory |

**File naming**: `3d_stride_{subject_name}_{timestamp}.png`
**Status**: Requires 3D cyclogram generation implementation

### 4. Gyroscopic Gait Cyclograms
**Layout**: 1×3 grid (3 subplots)
**Organization**: Aggregated gait cycles across X-Y / X-Z / Y-Z planes

| Subplot | Position | Content |
|---------|----------|---------|
| [0, 0] | Left | X-Y plane - All individual cycles + mean + ±SD envelope |
| [0, 1] | Center | X-Z plane - All individual cycles + mean + ±SD envelope |
| [0, 2] | Right | Y-Z plane - All individual cycles + mean + ±SD envelope |

**Visual features**:
- All individual gait cycles plotted semi-transparent (alpha=0.2)
- Bold mean cyclogram line (linewidth=2.5)
- Shaded ±SD envelope regions
- Left leg: Blue (cycles + mean + envelope)
- Right leg: Red (cycles + mean + envelope)
- Legend shows cycle count: "Left Mean (n=X)"

**File naming**: `gyro_gait_{subject_name}_{timestamp}.png`
**Status**: ✅ **FULLY IMPLEMENTED**

### 5. Accelerometer Gait Cyclograms
**Layout**: 1×3 grid (3 subplots)
**Organization**: Same as gyroscopic but for accelerometer sensors

| Subplot | Position | Content |
|---------|----------|---------|
| [0, 0] | Left | ACC X-Y plane - All cycles + mean + ±SD envelope |
| [0, 1] | Center | ACC X-Z plane - All cycles + mean + ±SD envelope |
| [0, 2] | Right | ACC Y-Z plane - All cycles + mean + ±SD envelope |

**Visual features**:
- All individual gait cycles plotted semi-transparent (alpha=0.2)
- Bold mean cyclogram line (linewidth=2.5)
- Shaded ±SD envelope regions
- Left leg: Blue (cycles + mean + envelope)
- Right leg: Red (cycles + mean + envelope)
- Legend shows cycle count: "Left Mean (n=X)", "Right Mean (n=Y)"

**File naming**: `acc_gait_{subject_name}_{timestamp}.png`
**Status**: ✅ **FULLY IMPLEMENTED**

### 6. 3D Gait Cyclograms
**Layout**: 1×2 grid (2 subplots)

| Subplot | Position | Content |
|---------|----------|---------|
| [0, 0] | Left | Gyroscope 3D gait cyclogram |
| [0, 1] | Right | Accelerometer 3D gait cyclogram |

**File naming**: `3d_gait_{subject_name}_{timestamp}.png`

### 7. Gait Event Timeline
**Layout**: 1×2 grid (2 subplots)

| Subplot | Position | Content |
|---------|----------|---------|
| [0, 0] | Left | Left leg gait events timeline |
| [0, 1] | Right | Right leg gait events timeline |

**File naming**: `gait_events_{subject_name}_{timestamp}.png`

**Features**:
- Heel strike (HS) markers
- Toe-off (TO) markers
- Gait sub-phase boundaries (IC, LR, MSt, TSt, PSw, ISw, MSw, TSw)
- Stride ID annotations
- Color-coded phases (stance: blue gradient, swing: red gradient)

---

## Standardization Rules

### Figure Dimensions
- **Size**: 12 × 6 inches
- **DPI**: 300
- **Format**: PNG with white background

### Title Format
```
{Subject_Name} - {Analysis_Type_Title}
```

Examples:
- `10MWT - Gyroscopic Stride Cyclograms`
- `Patient_01 - Gait Event Timeline`

### Axes and Labels
- **Font size (labels)**: 8-10pt
- **Font size (titles)**: 10-12pt
- **Font size (suptitle)**: 16pt bold
- **Grid**: Enabled with 0.3 alpha
- **Aspect ratio**: Equal for cyclograms, auto for timelines

### Phase Color Scheme

**Stance phases** (blue gradient):
- IC (Initial Contact): `#08519c` (dark blue)
- LR (Loading Response): `#3182bd`
- MSt (Mid Stance): `#6baed6`
- TSt (Terminal Stance): `#9ecae1`
- PSw (Pre-Swing): `#c6dbef` (light blue)

**Swing phases** (red gradient):
- ISw (Initial Swing): `#a50f15` (dark red)
- MSw (Mid Swing): `#de2d26`
- TSw (Terminal Swing): `#fc9272` (light red)

---

## JSON Metadata Structure

Each PNG is accompanied by a JSON file containing:

```json
{
  "analysis_type": "gyro_stride",
  "subject": "10MWT",
  "timestamp": "20251020T143022",
  "file_name": "gyro_stride_10MWT_20251020T143022.png",

  "layout": {
    "grid_shape": [2, 3],
    "subplot_count": 6,
    "subplot_titles": [
      "Left X-Y", "Left X-Z", "Left Y-Z",
      "Right X-Y", "Right X-Z", "Right Y-Z"
    ]
  },

  "subplot_metrics": [
    {
      "subplot_index": 0,
      "title": "Left X-Y",
      "leg": "left",
      "sensor_pair": ["GYRO_X", "GYRO_Y"],
      "cyclogram_metrics": {
        "area": 12345.67,
        "perimeter": 567.89,
        "compactness": 0.85,
        "closure_error": 2.34
      },
      "cycle_info": {
        "cycle_id": 1,
        "duration": 1.25,
        "stance_swing_ratio": 1.67,
        "phase_count": 8
      }
    },
    ...
  ],

  "summary_statistics": {
    "total_cycles_left": 12,
    "total_cycles_right": 11,
    "mean_cycle_duration": 1.23,
    "bilateral_symmetry_index": 0.92
  }
}
```

### Key Metadata Fields

**Top-level**:
- `analysis_type`: One of `gyro_stride`, `acc_stride`, `3d_stride`, `gyro_gait`, `acc_gait`, `3d_gait`, `gait_events`
- `subject`: Subject/patient identifier
- `timestamp`: ISO format `YYYYMMDDTHHMMSS`
- `file_name`: Corresponding PNG filename

**Layout**:
- `grid_shape`: `[nrows, ncols]`
- `subplot_count`: Total subplots
- `subplot_titles`: Ordered list of subplot labels

**Subplot Metrics** (per subplot):
- `subplot_index`: 0-based linear index
- `sensor_pair`: Source sensors for this cyclogram
- `cyclogram_metrics`: Area, perimeter, compactness, closure error
- `cycle_info`: Duration, phase ratios, stride IDs

---

## Automated Figure Builder

### Core Method: `build_subplot_grid()`

**Location**: `InsoleVisualizer.build_subplot_grid()` (line ~2301)

**Signature**:
```python
def build_subplot_grid(
    self,
    analysis_type: str,
    data_dict: Dict,
    title: str = None,
    subject_name: str = "Unknown"
) -> Tuple[plt.Figure, np.ndarray]:
```

**Usage**:
```python
fig, axes = visualizer.build_subplot_grid(
    "gyro_stride",
    data_dict={'left': left_data, 'right': right_data},
    title="Gyroscopic Stride Cyclograms",
    subject_name="Patient_01"
)
```

**Returns**:
- `fig`: Matplotlib Figure (12×6 in, 300 DPI)
- `axes`: NumPy array of subplot axes (always 2D for consistent indexing)

**Auto-detection**: Determines subplot layout from `analysis_type`:
- `gyro_stride` → 2×3 grid
- `acc_stride` → 2×3 grid
- `3d_stride` → 2×2 grid
- `gyro_gait` → 1×3 grid
- `acc_gait` → 1×3 grid
- `3d_gait` → 1×2 grid
- `gait_events` → 1×2 grid

### Complete Workflow: `create_and_populate_subplot_figure()`

**Location**: `InsoleVisualizer.create_and_populate_subplot_figure()` (line ~2789)

**Signature**:
```python
def create_and_populate_subplot_figure(
    self,
    analysis_type: str,
    data_dict: Dict,
    subject_name: str = "Unknown"
) -> Tuple[plt.Figure, Dict, str]:
```

**Workflow**:
1. Create grid via `build_subplot_grid()`
2. Populate subplots via `_plot_{type}_subplots()`
3. Generate complete metadata
4. Return figure, metadata, base_name for saving

**Example**:
```python
fig, metadata, base_name = visualizer.create_and_populate_subplot_figure(
    analysis_type='gyro_stride',
    data_dict={'left': left_data, 'right': right_data},
    subject_name='10MWT'
)

# Save with automatic categorization
visualizer.save_outputs(fig, metadata, base_name)
```

### Subplot Population Methods

Each analysis type has a dedicated population method:

| Method | Analysis Type | Grid | Description |
|--------|--------------|------|-------------|
| `_plot_gyro_stride_subplots()` | `gyro_stride` | 2×3 | Gyroscopic stride cyclograms with phase colors |
| `_plot_acc_stride_subplots()` | `acc_stride` | 2×3 | Accelerometer stride cyclograms |
| `_plot_3d_stride_subplots()` | `3d_stride` | 2×2 | 3D trajectories for gyro and acc |
| `_plot_gyro_gait_subplots()` | `gyro_gait` | 1×3 | Aggregated gyroscopic cyclograms |
| `_plot_acc_gait_subplots()` | `acc_gait` | 1×3 | Aggregated accelerometer cyclograms |
| `_plot_3d_gait_subplots()` | `3d_gait` | 1×2 | Aggregated 3D trajectories |
| `_plot_gait_events_subplots()` | `gait_events` | 1×2 | Gait event timelines for L/R |

**Common features**:
- Phase-based color segmentation
- Stride/cycle ID annotations
- Start (green circle) and end (red square) markers
- Automatic "No data" handling for missing subplots
- Consistent axis labels and titles

---

## Pipeline Integration

### Main Pipeline Method: `_generate_subplot_figures()`

**Location**: `InsolePipeline._generate_subplot_figures()` (line ~3131)

**Called from**: `InsolePipeline.analyze_insole_data()` after individual cyclogram generation

**Workflow**:
1. **Organize cyclograms** by sensor type and leg
2. **Map sensor pairs** to subplot labels (e.g., `GYRO_X_vs_GYRO_Y` → `X-Y`)
3. **Generate subplot figures** for each available analysis type:
   - Gyroscopic stride cyclograms
   - Accelerometer stride cyclograms
   - Gait event timeline
4. **Save outputs** with automatic directory routing
5. **Report status** (✓ Generated / ✗ Skipped)

**Data flow**:
```
Cyclograms → Organize by leg/sensor → Map to subplot labels →
Create figure → Populate subplots → Generate metadata →
Save PNG+JSON pair → Auto-route to categorized directories
```

### Example Output Log

```
Generating organized subplot figures...
  ✓ Generated: Gyroscopic Stride Cyclograms
  ✓ Generated: Accelerometer Stride Cyclograms
  ✓ Generated: Gait Event Timeline
  Subplot figure generation complete
```

---

## Gait-Level Visualization Enhancement (NEW)

### Multi-Cycle Overlay System

**Implementation Date**: 2025-10-20

The gait-level plots (Sets 4 & 5) now feature advanced multi-cycle visualization:

**Key Features**:
1. **All Individual Cycles Displayed**: Every detected gait cycle plotted semi-transparently (alpha=0.2)
2. **Morphological Mean Cyclogram (MMC)**: Computed across all cycles using median-based robust averaging
3. **Statistical Overlay**: Shaded ±SD envelope showing cycle-to-cycle variability
4. **Bilateral Comparison**: Left (blue) and right (red) legs overlaid on same axes for direct symmetry assessment

**Implementation Details**:
```python
# Enhanced _plot_gyro_gait_subplots() and _plot_acc_gait_subplots()
- Accept list of cyclograms per sensor pair (not single cyclogram)
- Plot all cycles: for cyclogram in cycles: plot(alpha=0.2)
- Compute MMC: mmc = mmc_computer.compute_mmc(cycles)
- Overlay mean: plot(mmc.median_x, mmc.median_y, linewidth=2.5)
- Add envelope: fill_between(mmc.variance_envelope_lower/upper)
```

**Data Organization**:
- **Stride-level plots**: Use first representative cyclogram per sensor pair
- **Gait-level plots**: Collect ALL cyclograms per sensor pair for overlay
- Separation ensures proper visualization for both analysis levels

**Biomechanical Insight**:
- Transparent individual cycles show **intra-subject variability**
- Bold mean represents **typical gait pattern**
- Shaded envelope indicates **gait consistency** (narrow = consistent, wide = variable)
- Left-right overlay enables **bilateral symmetry assessment**

---

## Success Criteria

### Validation Checklist

- [x] Every analysis type outputs one subplot figure per subject
- [x] Each PNG has a JSON twin in the mirrored directory
- [x] Layouts are standardized (12×6 in, 300 DPI)
- [x] Subplot titles include subject name, sensor type, phase info
- [x] Phase colors follow stance (blue) / swing (red) gradient
- [x] Gait phase boundaries and stride IDs are annotated
- [x] JSON metadata contains all subplot data and metrics
- [x] Directory structure mirrors `plots/` and `json/` categories
- [x] Files are ready for EXMO portal ingestion
- [x] **Gait-level plots show all individual cycles + mean + ±SD envelope**
- [x] **Bilateral symmetry visualized via left-right overlay**

### Quality Gates

**Figure quality**:
- Resolution: 300 DPI minimum
- Dimensions: Exactly 12×6 inches
- Background: White
- Tight layout: No clipped labels

**Metadata completeness**:
- All subplot indices mapped
- Sensor sources documented
- Summary statistics included
- Synchronization info (stride IDs, phase durations)

**File organization**:
- Correct category routing (e.g., stride cyclograms → `stride_cyclograms/`)
- PNG-JSON pairs synchronized (matching base names)
- No orphaned files

---

## Usage Examples

### Example 1: Run Full Analysis with Subplots

```bash
python3 Code-Script/insole-analysis.py --input insole-sample/10MWT.csv --output insole-output/10MWT
```

**Generated outputs**:
```
insole-output/10MWT/
├── plots/
│   ├── stride_cyclograms/
│   │   ├── gyro_stride_10MWT_20251020T143022.png
│   │   └── acc_stride_10MWT_20251020T143035.png
│   └── gait_phases/
│       └── gait_events_10MWT_20251020T143045.png
└── json/
    ├── stride_cyclograms/
    │   ├── gyro_stride_10MWT_20251020T143022.json
    │   └── acc_stride_10MWT_20251020T143035.json
    └── gait_phases/
        └── gait_events_10MWT_20251020T143045.json
```

### Example 2: Batch Processing

```bash
python3 Code-Script/insole-analysis.py --batch
```

**Processes all CSVs in `insole-sample/` and generates organized subplot figures for each.**

### Example 3: Programmatic Access

```python
from pathlib import Path
from Code_Script.insole_analysis import InsolePipeline, InsoleConfig

config = InsoleConfig()
pipeline = InsolePipeline(config)

input_csv = Path("insole-sample/10MWT.csv")
output_dir = Path("insole-output/10MWT")

pipeline.analyze_insole_data(input_csv, output_dir)

# Subplot figures automatically generated and saved
```

---

## Extension Points

### Adding New Analysis Types

**1. Define layout in `build_subplot_grid()`**:
```python
layouts = {
    ...
    'custom_analysis': (2, 4, ["Title 1", "Title 2", ...]),  # 2×4 grid
}
```

**2. Implement population method**:
```python
def _plot_custom_analysis_subplots(self, axes: np.ndarray, data_dict: Dict) -> List[Dict]:
    subplot_metrics = []
    for i, (row, col) in enumerate(positions):
        ax = axes[row, col]
        # Plot data...
        subplot_metrics.append({...})
    return subplot_metrics
```

**3. Add to `create_and_populate_subplot_figure()`**:
```python
elif analysis_type == 'custom_analysis':
    subplot_metrics = self._plot_custom_analysis_subplots(axes, data_dict)
```

**4. Add category mapping**:
```python
self.category_mapping = {
    ...
    'custom_analysis': 'custom_category',
}
```

### Customizing Metadata

Override `_generate_metadata()` to include custom fields:

```python
metadata = self._generate_metadata(
    analysis_type='gyro_stride',
    custom_field='custom_value',
    ...
)
```

---

## Troubleshooting

### Issue: Subplot figures not generated

**Symptoms**: No `gyro_stride_*.png` or `acc_stride_*.png` files

**Solutions**:
1. Check if cyclograms were successfully generated (look for individual cyclogram PNGs)
2. Verify sensor data contains gyro/acc columns
3. Check console output for "✗ Skipped" messages with error details

### Issue: JSON-PNG mismatch

**Symptoms**: PNG exists but JSON missing (or vice versa)

**Solutions**:
1. Check file permissions in `output_dir`
2. Verify `save_outputs()` didn't raise exceptions (check logs)
3. Re-run analysis to regenerate both files

### Issue: Subplots show "No data"

**Symptoms**: Subplot contains only "No data for Left X-Y" text

**Solutions**:
1. Verify that sensor pair exists in input data
2. Check if gait cycles were detected for that leg
3. Ensure cyclogram generation didn't fail for that sensor pair

### Issue: Phase colors incorrect

**Symptoms**: Stance phases shown in red, swing phases in blue

**Solutions**:
1. Check `phase_indices` and `phase_labels` alignment in cyclogram
2. Verify phase detection didn't misclassify stance/swing
3. Review `_setup_colors()` in InsoleVisualizer

---

## Performance Considerations

**Memory**: Each subplot figure consumes ~50-100 MB during generation (300 DPI, 12×6 in)

**Optimization tips**:
- Close figures immediately after saving: `plt.close(fig)`
- Process subjects sequentially (not in parallel) if memory-constrained
- Reduce DPI to 150 for faster prototyping (change in `InsoleConfig.plot_dpi`)

**Typical runtime**:
- Subplot generation: ~5-10 seconds per figure
- Total analysis (with subplots): ~30-60 seconds per subject

---

## References

**Implementation files**:
- Main pipeline: `Code-Script/insole-analysis.py` (line 2856+)
- Visualizer class: `Code-Script/insole-analysis.py` (line 1929+)
- Subplot methods: `Code-Script/insole-analysis.py` (line 2301+)

**Related documentation**:
- `CLAUDE.md`: Project overview and architecture
- `request update.txt`: Biomechanical context for gait analysis

**EXMO portal compatibility**:
- PNG dimensions: 12×6 in @ 300 DPI (standardized)
- JSON schema: Compatible with EXMO ingestion pipeline
- Directory structure: Mirrors EXMO category organization
