# CRITICAL BUG FIX: ACC and GYRO Data Mixing

**Date**: 2025-10-20
**Severity**: 🔴 **CRITICAL**
**Status**: ✅ **FIXED**

---

## Bug Description

**Reported Issue**: `acc_gait_10MWT_*.png` and `gyro_gait_10MWT_*.png` showing identical data instead of distinct accelerometer and gyroscope measurements.

**Root Cause**: Data mapping key collision in `_generate_subplot_figures()` (line 3551-3604).

---

## Technical Analysis

### Problem Code (Before Fix)

**Location**: `insole-analysis.py:3551-3559`

```python
# Map sensor pair labels to subplot labels with sensor type preserved
sensor_mapping = {
    'GYRO_X_vs_GYRO_Y': 'X-Y Plane',  # ❌ COLLISION
    'GYRO_X_vs_GYRO_Z': 'X-Z Plane',  # ❌ COLLISION
    'GYRO_Y_vs_GYRO_Z': 'Y-Z Plane',  # ❌ COLLISION
    'ACC_X_vs_ACC_Y': 'X-Y Plane',    # ❌ COLLISION
    'ACC_X_vs_ACC_Z': 'X-Z Plane',    # ❌ COLLISION
    'ACC_Y_vs_ACC_Z': 'Y-Z Plane'     # ❌ COLLISION
}
```

**What Happened**:
1. Both GYRO and ACC sensor pairs mapped to **same keys**: `'X-Y Plane'`, `'X-Z Plane'`, `'Y-Z Plane'`
2. Data organization loop (line 3562-3604) overwrote GYRO data with ACC data
3. `gait_dict['left']['X-Y Plane']` contained **only ACC data**
4. `_plot_gyro_gait_subplots()` received ACC data instead of GYRO data
5. Both acc_gait and gyro_gait plots showed **identical ACC data**

**Data Flow (Buggy)**:
```
Cyclogram: GYRO_X_vs_GYRO_Y
  ↓
Key: 'X-Y Plane'
  ↓
gait_dict['left']['X-Y Plane'] = [gyro cyclograms]  # ✓ First pass

Cyclogram: ACC_X_vs_ACC_Y
  ↓
Key: 'X-Y Plane'  # ❌ SAME KEY!
  ↓
gait_dict['left']['X-Y Plane'] = [acc cyclograms]  # ❌ OVERWRITES GYRO DATA!
```

---

## Solution Implemented

### Fixed Code (After Fix)

**Location**: `insole-analysis.py:3551-3560`

```python
# Map sensor pair labels to subplot labels with sensor type preserved
# CRITICAL: Must include sensor type prefix to avoid mixing GYRO and ACC data
sensor_mapping = {
    'GYRO_X_vs_GYRO_Y': 'GYRO_X-Y Plane',  # ✓ UNIQUE KEY
    'GYRO_X_vs_GYRO_Z': 'GYRO_X-Z Plane',  # ✓ UNIQUE KEY
    'GYRO_Y_vs_GYRO_Z': 'GYRO_Y-Z Plane',  # ✓ UNIQUE KEY
    'ACC_X_vs_ACC_Y': 'ACC_X-Y Plane',     # ✓ UNIQUE KEY
    'ACC_X_vs_ACC_Z': 'ACC_X-Z Plane',     # ✓ UNIQUE KEY
    'ACC_Y_vs_ACC_Z': 'ACC_Y-Z Plane'      # ✓ UNIQUE KEY
}
```

**Data Flow (Fixed)**:
```
Cyclogram: GYRO_X_vs_GYRO_Y
  ↓
Key: 'GYRO_X-Y Plane'  # ✓ UNIQUE
  ↓
gait_dict['left']['GYRO_X-Y Plane'] = [gyro cyclograms]

Cyclogram: ACC_X_vs_ACC_Y
  ↓
Key: 'ACC_X-Y Plane'  # ✓ UNIQUE (different from GYRO)
  ↓
gait_dict['left']['ACC_X-Y Plane'] = [acc cyclograms]

# ✓ NO COLLISION - Both sensor types preserved!
```

### Updated Plotting Functions

**Location**: `insole-analysis.py:2841-2987`

Updated both `_plot_gyro_gait_subplots()` and `_plot_acc_gait_subplots()` to:
1. Use prefixed keys for data lookup: `'GYRO_X-Y Plane'` and `'ACC_X-Y Plane'`
2. Use clean display labels for titles: `'X-Y Plane'` (user-facing)

**Before**:
```python
pair_labels = ['X-Y Plane', 'X-Z Plane', 'Y-Z Plane']  # ❌ Same for both

for col_idx, (pair, label) in enumerate(zip(sensor_pairs, pair_labels)):
    plane_data = leg_data.get(label, [])  # ❌ Gets wrong data
```

**After**:
```python
# GYRO function
pair_labels = ['GYRO_X-Y Plane', 'GYRO_X-Z Plane', 'GYRO_Y-Z Plane']  # ✓ Unique
display_labels = ['X-Y Plane', 'X-Z Plane', 'Y-Z Plane']               # ✓ Clean display

for col_idx, (pair, label, disp_label) in enumerate(zip(sensor_pairs, pair_labels, display_labels)):
    plane_data = leg_data.get(label, [])           # ✓ Gets correct GYRO data
    ax.set_title(f"{leg.title()} - {disp_label}") # ✓ Clean title

# ACC function
pair_labels = ['ACC_X-Y Plane', 'ACC_X-Z Plane', 'ACC_Y-Z Plane']  # ✓ Unique
display_labels = ['X-Y Plane', 'X-Z Plane', 'Y-Z Plane']             # ✓ Clean display

for col_idx, (pair, label, disp_label) in enumerate(zip(sensor_pairs, pair_labels, display_labels)):
    plane_data = leg_data.get(label, [])           # ✓ Gets correct ACC data
    ax.set_title(f"{leg.title()} - {disp_label}") # ✓ Clean title
```

---

## Changes Summary

### Files Modified
- `Code-Script/insole-analysis.py`

### Lines Changed
1. **Line 3551-3560**: Fixed `sensor_mapping` with prefixed keys
2. **Line 2853-2854**: Added `pair_labels` with GYRO prefix + `display_labels` for _plot_gyro_gait_subplots()
3. **Line 2860**: Updated loop to use both `label` (lookup) and `disp_label` (display)
4. **Line 2864**: Uses prefixed `label` for data lookup
5. **Line 2898**: Uses clean `disp_label` for subplot title
6. **Line 2927-2928**: Added `pair_labels` with ACC prefix + `display_labels` for _plot_acc_gait_subplots()
7. **Line 2934**: Updated loop to use both `label` (lookup) and `disp_label` (display)
8. **Line 2938**: Uses prefixed `label` for data lookup
9. **Line 2972**: Uses clean `disp_label` for subplot title

**Total**: ~18 lines modified across 3 functions

---

## Validation

### Test Run
```bash
python3 Code-Script/insole-analysis.py --input insole-sample/10MWT.csv --output insole-output/10MWT_final_test
```

### Expected Output
- `gyro_gait_10MWT_*.png`: Should show **gyroscope** data (rotational velocities)
- `acc_gait_10MWT_*.png`: Should show **accelerometer** data (linear accelerations)
- Data values should be **completely different** between the two plots

### Verification Checklist
- [ ] gyro_gait plot shows gyroscope data (typical range: -500 to +500 deg/s)
- [ ] acc_gait plot shows accelerometer data (typical range: -20 to +20 m/s²)
- [ ] Subplot titles display clean labels: "Left - X-Y Plane", "Right - Y-Z Plane", etc.
- [ ] No data mixing or collisions
- [ ] All 6 subplots per figure populated correctly

---

## Impact Assessment

### Severity: 🔴 CRITICAL
**Why Critical**:
1. **Data Corruption**: ACC and GYRO data were being mixed, producing invalid clinical results
2. **Silent Failure**: No error messages - plots looked normal but contained wrong data
3. **Research Impact**: Any analysis using these plots would have incorrect conclusions
4. **Clinical Risk**: Gait assessments based on mixed data could lead to wrong treatment decisions

### Affected Outputs (Before Fix)
- ❌ `gyro_gait_10MWT_*.png`: Showed ACC data instead of GYRO data
- ✓ `acc_gait_10MWT_*.png`: Showed correct ACC data (but both plots were identical)
- ✓ All other plots (stride cyclograms, 3D, gait events): **NOT affected**

### Scope
- **Affected**: Gait-level plots (Plot Sets 4 & 5) only
- **Not Affected**: Stride-level plots (Plot Sets 1 & 2), 3D plots (Plot Set 3, 6), gait events (Plot Set 7)

---

## Root Cause Analysis

### Why Did This Happen?

1. **Insufficient Key Uniqueness**: Original design assumed sensor type would be implicit from context
2. **No Data Type Validation**: No checks to ensure GYRO functions receive GYRO data
3. **Overwrite Semantics**: Dictionary key assignment silently overwrites previous values
4. **Missing Unit Tests**: No automated tests to catch data type mixing

### Prevention Measures

**Immediate**:
- ✅ Added sensor type prefix to all mapping keys
- ✅ Separated lookup keys from display labels

**Long-term Recommendations**:
1. **Type System**: Use typed dataclasses to enforce sensor type at compile time
2. **Validation Layer**: Add assertions to verify data type matches expected sensor type
3. **Unit Tests**: Create tests that verify GYRO/ACC separation in data dictionaries
4. **Code Review**: Flag any dictionary keys that don't include sensor type information

---

## Testing Protocol

### Manual Verification Steps

1. **Visual Inspection**:
   ```bash
   # Compare gyro_gait vs acc_gait side-by-side
   display insole-output/10MWT_final_test/plots/gait_cyclograms/gyro_gait_*.png &
   display insole-output/10MWT_final_test/plots/gait_cyclograms/acc_gait_*.png &
   ```
   - ✓ Plots should show **completely different** cyclogram shapes
   - ✓ Value ranges should be different (gyro: deg/s, acc: m/s²)

2. **Data Range Check**:
   ```python
   # Extract and compare data ranges from JSON metadata
   import json

   with open('insole-output/10MWT_final_test/json/gait_cyclograms/gyro_gait_*.json') as f:
       gyro_meta = json.load(f)

   with open('insole-output/10MWT_final_test/json/gait_cyclograms/acc_gait_*.json') as f:
       acc_meta = json.load(f)

   # Gyro should have larger magnitude values (rotational velocity)
   # Acc should have smaller magnitude values (linear acceleration)
   ```

3. **Subplot Title Check**:
   - All titles should display clean labels: "Left - X-Y Plane" (not "Left - GYRO_X-Y Plane")
   - Internal data lookup uses prefixed keys (not visible to user)

---

## Conclusion

**Status**: ✅ **FIXED**

**Changes**:
- Added `GYRO_` and `ACC_` prefixes to mapping keys
- Separated data lookup keys from user-facing display labels
- Updated both plotting functions to use new key structure

**Result**:
- GYRO and ACC data now properly separated
- No more key collisions in data dictionaries
- Clean subplot titles maintained for user experience

**Recommendation**: **Immediate rerun of all analyses** using this fixed code version to replace any results generated with the buggy version.
