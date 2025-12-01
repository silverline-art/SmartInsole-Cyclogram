# Complete Cyclogram Sheet Column Structure

**Script**: `cyclogram_extractor_ACC.py` v2.1.0
**Total Expected Columns**: ~144

## Column Organization

### Patient Information (5 columns)
```
1. Dataset
2. Patient Name
3. Patient ID
4. Fracture Site
5. Fracture Name
```

### Cycle Counts (2 columns)
```
6. Total_Cycles_Right_Leg
7. Total_Cycles_Left_Leg
```

---

## Section A: Advanced Cyclogram Metrics (34 columns)

### Left Leg (17 metrics)
```
8. left_duration
9. left_area
10. left_perimeter
11. left_compactness
12. left_closure_error
13. left_compactness_ratio
14. left_aspect_ratio
15. left_eccentricity
16. left_orientation_angle
17. left_mean_curvature
18. left_curvature_std
19. left_trajectory_smoothness
20. left_mean_relative_phase
21. left_marp
22. left_coupling_angle_variability
23. left_deviation_phase
24. left_phase_shift
```

### Right Leg (17 metrics)
```
25-41. right_duration through right_phase_shift (same as left)
```

---

## Section B: Bilateral Symmetry (5 columns)

```
42. bilateral_symmetry_index
43. bilateral_mirror_correlation
44. bilateral_rms_trajectory_diff_x
45. bilateral_rms_trajectory_diff_y
46. bilateral_rms_trajectory_diff
```

---

## Section C: Aggregate Metrics (18 columns)

### Left Leg Aggregates (9 metrics)
```
47. agg_left_area
48. agg_left_compactness_ratio
49. agg_left_aspect_ratio
50. agg_left_eccentricity
51. agg_left_mean_curvature
52. agg_left_trajectory_smoothness
53. agg_left_mean_relative_phase
54. agg_left_marp
55. agg_left_coupling_angle_variability
```

### Right Leg Aggregates (9 metrics)
```
56-64. agg_right_area through agg_right_coupling_angle_variability
```

---

## Section D: Symmetry Aggregates (14 columns)

```
65. sym_agg_area_L
66. sym_agg_area_R
67. sym_agg_area_symmetry
68. sym_agg_curvature_mean_L
69. sym_agg_curvature_mean_R
70. sym_agg_curvature_symmetry
71. sym_agg_smoothness_L
72. sym_agg_smoothness_R
73. sym_agg_smoothness_symmetry
74. sym_agg_overall_symmetry
75-78. [Additional symmetry aggregate metrics if present]
```

---

## Section E: Symmetry Metrics (8 columns)

```
79. sym_met_area_symmetry
80. sym_met_curvature_symmetry
81. sym_met_smoothness_symmetry
82. sym_met_overall_symmetry
83. sym_met_area_L
84. sym_met_area_R
85. sym_met_curvature_mean_L
86. sym_met_curvature_mean_R
```

**Note**: Original implementation ended at column ~86

---

## Section F: Curvature Phase-Variation (58 columns) - NEW

### Left Leg Peak Phase (2 columns)
```
87. left_curv_phase_peak_phase_%
88. left_curv_phase_peak_value
```

### Left Leg Statistical (3 columns)
```
89. left_curv_phase_variability_index
90. left_curv_phase_entropy
91. left_curv_phase_rms
```

### Left Leg Phase-Binned (10 columns)
```
92. left_curv_p00    (0-10% phase)
93. left_curv_p10    (10-20% phase)
94. left_curv_p20    (20-30% phase)
95. left_curv_p30    (30-40% phase)
96. left_curv_p40    (40-50% phase)
97. left_curv_p50    (50-60% phase)
98. left_curv_p60    (60-70% phase)
99. left_curv_p70    (70-80% phase)
100. left_curv_p80   (80-90% phase)
101. left_curv_p90   (90-100% phase)
```

### Right Leg Peak Phase (2 columns)
```
102. right_curv_phase_peak_phase_%
103. right_curv_phase_peak_value
```

### Right Leg Statistical (3 columns)
```
104. right_curv_phase_variability_index
105. right_curv_phase_entropy
106. right_curv_phase_rms
```

### Right Leg Phase-Binned (10 columns)
```
107. right_curv_p00   (0-10% phase)
108. right_curv_p10   (10-20% phase)
109. right_curv_p20   (20-30% phase)
110. right_curv_p30   (30-40% phase)
111. right_curv_p40   (40-50% phase)
112. right_curv_p50   (50-60% phase)
113. right_curv_p60   (60-70% phase)
114. right_curv_p70   (70-80% phase)
115. right_curv_p80   (80-90% phase)
116. right_curv_p90   (90-100% phase)
```

### Bilateral Curvature Symmetry (4 columns)
```
117. curv_phase_rms_diff
118. curv_phase_circular_corr
119. curv_phase_peak_phase_diff_%
120. curv_phase_variability_index_diff
```

---

## Column Count Summary

| Section | Description | Count |
|---------|-------------|-------|
| Header | Patient info + cycle counts | 7 |
| A | Advanced cyclogram metrics (L+R) | 34 |
| B | Bilateral symmetry | 5 |
| C | Aggregate metrics (L+R) | 18 |
| D | Symmetry aggregates | 14 |
| E | Symmetry metrics | 8 |
| **F** | **Curvature phase-variation (NEW)** | **58** |
| **Total** | | **~144** |

## Format Reference

### Column Value Formats

| Metric Type | Format | Example |
|-------------|--------|---------|
| Peak/Statistical metrics | `{value:.4f}` | `0.1234` |
| Phase-binned metrics | `{mean:.4f} ± {std:.4f}` | `0.1234 ± 0.0567` |
| Bilateral metrics | `{value:.4f}` | `0.0234` |
| Cycle counts | Integer | `15` |
| Empty/missing | Empty string | `` |

## Section Dependencies

### Data Source Files Required

**Sections A-F**:
- `cyclogram_advanced_metrics.csv` ✓
  - Contains: duration, area, perimeter, etc.
  - Contains: continuous_relative_phase (for Section F)
  - Filter by: cyclogram_type, leg

**Section B**:
- `cyclogram_bilateral_symmetry.csv` ✓
  - Contains: symmetry_index, mirror_correlation, etc.
  - Filter by: cyclogram_type

**Section C**:
- `cyclogram_metrics_aggregate.csv` ✓
  - Contains: aggregate stats for each metric
  - Filter by: cyclogram_type

**Section D**:
- `symmetry_aggregate.csv` ✓
  - Contains: overall symmetry aggregates
  - Filter by: cyclogram_type

**Section E**:
- `symmetry_metrics.csv` ✓
  - Contains: per-metric symmetry stats
  - Filter by: cyclogram_type

## Extraction Pipeline

```
Patient Directory
├─> cyclogram_advanced_metrics.csv
│   ├─> Sections A (metrics per cycle)
│   └─> Section F (curvature from phase)
│
├─> cyclogram_bilateral_symmetry.csv
│   └─> Section B
│
├─> cyclogram_metrics_aggregate.csv
│   └─> Section C
│
├─> symmetry_aggregate.csv
│   └─> Section D
│
└─> symmetry_metrics.csv
    └─> Section E

Combined → Single Excel Row (144 columns)
```

## Excel File Structure

### File: ACC_XY.xlsx
**Sheet 1: Cyclogram** (~144 columns)
- All sections A-F as described above

**Sheet 2: Gait** (separate metric set)
- Gait phases, support types, etc.

### File: ACC_XZ.xlsx
Same structure as ACC_XY.xlsx

### File: ACC_YZ.xlsx
Same structure as ACC_XY.xlsx

## Column Sorting Recommendations

### For Analysis
1. Sort by Dataset → Compare cohorts
2. Sort by Fracture Site → Group by pathology
3. Sort by curv_phase_rms_diff → Identify asymmetric patients
4. Sort by left_curv_phase_variability_index → Find unstable gaits

### For Quality Control
1. Check empty columns → Missing data
2. Filter by Total_Cycles < 5 → Insufficient data
3. Look for extreme values → Data quality issues

## Common Excel Formulas

### Aggregate Metrics
```excel
# Average RMS across legs
=(left_curv_phase_rms + right_curv_phase_rms)/2

# Asymmetry percentage
=curv_phase_rms_diff / AVERAGE(left_curv_phase_rms, right_curv_phase_rms) * 100

# Combined variability score
=AVERAGE(left_curv_phase_variability_index, right_curv_phase_variability_index)
```

### Conditional Logic
```excel
# Flag high asymmetry
=IF(curv_phase_rms_diff > 0.15, "Asymmetric", "Symmetric")

# Flag unstable gait
=IF(left_curv_phase_variability_index > 0.7, "Unstable", "Stable")

# Overall quality score
=IF(AND(curv_phase_circular_corr > 0.7, curv_phase_rms_diff < 0.15), "Good", "Review")
```

## Data Validation Rules

### Expected Value Ranges

| Metric | Min | Max | Typical |
|--------|-----|-----|---------|
| curv_phase_peak_phase_% | 0 | 100 | 30-70 |
| curv_phase_peak_value | 0 | 1 | 0.01-0.5 |
| curv_phase_variability_index | 0 | 2 | 0.2-0.6 |
| curv_phase_entropy | 0 | 3.3 | 1.5-2.5 |
| curv_phase_rms | 0 | 1 | 0.01-0.3 |
| curv_phase_circular_corr | -1 | 1 | 0.5-0.9 |
| curv_phase_rms_diff | 0 | 1 | 0-0.2 |
| curv_phase_peak_phase_diff_% | 0 | 100 | 0-25 |
| curv_phase_variability_index_diff | 0 | 2 | 0-0.4 |

### Quality Flags

**Red Flags** (data quality issues):
- All curvature metrics empty → Missing phase data
- Entropy = 0 → Single value (check data)
- Correlation > 1 or < -1 → Computational error
- RMS < mean curvature → Computational error

**Clinical Flags** (potential pathology):
- VI > 0.7 → High instability
- RMS diff > 0.2 → Significant asymmetry
- Correlation < 0.4 → Poor bilateral coordination
- Peak phase diff > 30% → Timing asynchrony

## Version Compatibility

### v2.1.0 (Current)
- Columns 1-144 populated
- Section F metrics included
- Compatible with existing analysis scripts

### v2.0.0 (Previous)
- Columns 1-86 populated
- Section F columns will be empty if using old data
- Backward compatible (no breaking changes)

### Migration Notes
- Old Excel files: 86 columns
- New Excel files: 144 columns
- Analysis scripts: Update column references if needed
- No data loss: All original metrics retained

---

**Generated**: 2025-10-23
**Script Version**: 2.1.0
**Status**: Production Ready
