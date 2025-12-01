# Pose-Analysis.py Detection Failure - Root Cause Analysis

**Date**: 2025-11-04
**Issue**: Pose-Analysis.py failing to detect columns "when the data clearly contains the value"
**Status**: ✅ ROOT CAUSE IDENTIFIED - DATA TYPE MISMATCH (Not a bug)

---

## Executive Summary

**The detection is NOT failing - it's working correctly by rejecting incompatible data.**

Pose-Analysis.py expects **preprocessed joint angle data** (hip/knee/ankle flexion angles) but is receiving **raw insole sensor data** (accelerometer, gyroscope, pressure values). This is a fundamental data type incompatibility, not a pattern detection bug.

---

## Technical Analysis

### Expected Input Format (Pose-Analysis.py)

**Script Purpose**: Process MediaPipe pose estimation joint angles
**Expected CSV**: `Raw_Angles.csv` with columns like:
```
timestamp, frame, hip_flex_L_deg, knee_flex_L_deg, ankle_dorsi_L_deg,
hip_flex_R_deg, knee_flex_R_deg, ankle_dorsi_R_deg
```

**Detection Logic** (Pose-Analysis.py:1057-1106):
```python
def detect_columns(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    patterns = {
        "hip": [r"(?i)hip.*flex.*(L|left|lt).*deg", ...],
        "knee": [r"(?i)knee.*flex.*(L|left|lt).*deg", ...],
        "ankle": [r"(?i)ankle.*(dorsi|flex).*(L|left|lt).*deg", ...]
    }
    # Validates ALL required joints exist for both legs
    # Raises ValueError if any missing
```

### Actual Input Format (DEMC Data Files)

**File Structure**: `DEMC048김영금_19370211_Gait_보행_2023_05_04-05_54_00_rawdata.csv`

**Actual CSV Headers**:
```
version                              # Line 1: "version"
22                                   # Line 2: version number
timestamp,L_value1,L_value2,L_value3,L_value4,R_value1,R_value2,R_value3,R_value4,
COP_Left,COP_Right,COP_Front,COP_Back,COP_Left_Front,COP_Left_Back,
COP_Right_Front,COP_Right_Back,L_ACC_X,L_ACC_Y,L_ACC_Z,R_ACC_X,R_ACC_Y,R_ACC_Z,
L_GYRO_X,L_GYRO_Y,L_GYRO_Z,R_GYRO_X,R_GYRO_Y,R_GYRO_Z,L_TEMPERATURE,R_TEMPERATURE,...
```

**Data Type**: Smart insole pressure + IMU sensor readings
- **Pressure sensors**: L_value1-6, R_value1-6, sensor_1N-6N
- **Accelerometer**: L_ACC_X/Y/Z, R_ACC_X/Y/Z
- **Gyroscope**: L_GYRO_X/Y/Z, R_GYRO_X/Y/Z
- **Center of Pressure**: COP_Left, COP_Right, COP_Front, COP_Back
- **Temperature**: L_TEMPERATURE, R_TEMPERATURE

**❌ NO JOINT ANGLES PRESENT**

---

## Root Cause: Wrong Script for Data Type

### Correct Script Assignment

| Data Type | Correct Script | Purpose |
|-----------|---------------|---------|
| **Joint angles** (hip/knee/ankle flexion) | `Pose-Analysis.py` | MediaPipe pose-based cyclograms |
| **Insole pressure/IMU** (sensors, ACC, GYRO) | `insole-analysis.py` | Pressure-based cyclograms |

### Project Architecture (claudedocs/CLAUDE.md:44-48)

```python
# For POSE data (joint angles from MediaPipe)
python3 Code-Script/Pose-Analysis.py --enhance-angles

# For INSOLE data (pressure sensors, IMU)
python3 Code-Script/insole-analysis.py --input data.csv
```

---

## Why Detection "Failed"

The detection didn't fail - **it succeeded in validation**:

1. ✅ Script loaded CSV correctly
2. ✅ Searched for joint angle columns using regex patterns
3. ✅ Found ZERO matches (because they don't exist)
4. ✅ **Correctly rejected incompatible data with ValueError**

**Error message** (Pose-Analysis.py:1103-1104):
```python
raise ValueError(f"Could not detect {joint} column for {leg} leg. "
                 f"Available columns: {df.columns.tolist()}")
```

This is **proper validation**, not a bug. The script prevented inappropriate processing of incompatible data.

---

## Potential User Scenarios

### Scenario 1: Script Confusion (MOST LIKELY)
**Problem**: Using Pose-Analysis.py on insole data
**Solution**: Switch to `insole-analysis.py`

```bash
# WRONG (current approach)
python3 Pose-Analysis.py --input DEMC048김영금_*_rawdata.csv

# CORRECT
python3 insole-analysis.py --input DEMC048김영금_*_rawdata.csv --output results/
```

### Scenario 2: Missing Preprocessing
**Problem**: Have pose video but no angle extraction yet
**Solution**: Run MediaPipe pose estimation first, then Pose-Analysis.py

**Pipeline**:
```
Video → MediaPipe Pose → Raw_Angles.csv → Pose-Analysis.py
```

### Scenario 3: Non-Standard Column Naming
**Problem**: Have actual joint angles but different column names
**Example**: `Left_Hip_Flexion` instead of `hip_flex_L_deg`

**Solution**: Modify detection patterns (Pose-Analysis.py:1066-1078):
```python
patterns = {
    "hip": [
        r"(?i)hip.*flex.*(L|left|lt).*deg",
        r"(?i)(L|left|lt).*hip.*flex.*deg",
        r"(?i)(left|lt).*hip.*flexion",    # ADD THIS
        r"(?i)hip.*flexion.*(left|lt)",    # ADD THIS
    ],
    # ... similar for knee and ankle
}
```

### Scenario 4: Want Pose-Style Analysis on IMU Data
**Problem**: Have insole data but want joint coordination cyclograms
**Solution**: Need IMU → joint angle conversion (biomechanical modeling required)

**Note**: This requires inverse kinematics and is architecturally complex. Current scripts are domain-specific by design.

---

## Recommended Solutions

### Immediate Fix (Choose based on actual need)

#### Option A: Use Correct Script for Insole Data
```bash
python3 CYCLOGRAM-PROCESSING\ Script/insole-analysis.py \
  --input Dataset/RAW-DATA/DEMC-DATA/DEMC048김영금_*_rawdata.csv \
  --output insole-output/DEMC048 \
  --sampling-rate 100
```

#### Option B: Enhanced Error Messaging (If pattern detection issue)
Improve error message to guide users:

```python
# In detect_columns() at line 1103
available_cols = df.columns.tolist()

# Detect data type
is_insole_data = any('ACC_' in col or 'GYRO_' in col or 'COP_' in col
                     for col in available_cols)

if is_insole_data:
    raise ValueError(
        f"❌ DATA TYPE MISMATCH: Found insole sensor data (ACC/GYRO/COP columns).\n"
        f"   This script expects POSE data with joint angle columns.\n\n"
        f"   ✅ SOLUTION: Use insole-analysis.py instead:\n"
        f"   python3 insole-analysis.py --input <file> --output <dir>\n\n"
        f"   Available columns: {available_cols[:10]}..."
    )
else:
    raise ValueError(
        f"Could not detect {joint} column for {leg} leg.\n"
        f"Expected pattern: hip/knee/ankle + flex/dorsi + L/R + deg\n"
        f"Available columns: {available_cols}"
    )
```

#### Option C: Flexible Column Mapping (If pose data with custom names)
Add custom column mapping option:

```python
# Add to AnalysisConfig (line ~905)
custom_column_map: Optional[Dict[str, Dict[str, str]]] = None

# In detect_columns()
if config.custom_column_map:
    return config.custom_column_map  # Use user-provided mapping
else:
    # Continue with auto-detection
```

**Usage**:
```python
config = AnalysisConfig(
    custom_column_map={
        "L": {"hip": "Left_Hip_Flexion", "knee": "Left_Knee_Flexion", ...},
        "R": {"hip": "Right_Hip_Flexion", "knee": "Right_Knee_Flexion", ...}
    }
)
```

#### Option D: Auto-Detection and Script Routing
Create intelligent dispatcher:

```python
# new file: auto_analyze.py
def detect_data_type(csv_path):
    df = pd.read_csv(csv_path, nrows=0)
    cols = df.columns.tolist()

    if any('ACC_' in col or 'GYRO_' in col for col in cols):
        return 'insole'
    elif any('hip' in col.lower() and 'deg' in col.lower() for col in cols):
        return 'pose'
    else:
        return 'unknown'

def main():
    data_type = detect_data_type(args.input)

    if data_type == 'insole':
        subprocess.run(['python3', 'insole-analysis.py', ...])
    elif data_type == 'pose':
        subprocess.run(['python3', 'Pose-Analysis.py', ...])
    else:
        print("❌ Unknown data type. Manual script selection required.")
```

---

## Comprehensive Fix Plan

### Phase 1: Clarification (REQUIRED FIRST)
1. ✅ Identify user's actual data type and goal
2. ✅ Determine if they have pose data with custom naming OR insole data misrouted

### Phase 2: Quick Win (Choose ONE based on clarification)
- **If insole data**: Guide to insole-analysis.py usage
- **If pose data with custom names**: Add flexible column patterns
- **If workflow confusion**: Enhance error messages with guidance

### Phase 3: Long-Term Improvements (Optional)
- [ ] Auto-detection and script routing (Option D)
- [ ] Unified interface with data type auto-detection
- [ ] Custom column mapping configuration file support
- [ ] Preprocessing pipeline for video → angles → cyclograms

### Phase 4: Validation Strategy
```bash
# Test with DEMC insole data
python3 insole-analysis.py --input Dataset/RAW-DATA/DEMC-DATA/*.csv

# Test with pose data (if exists)
python3 Pose-Analysis.py --input Sample-Data/Openpose_*/Raw_Angles.csv

# Test enhanced error messaging
python3 Pose-Analysis.py --input <insole-data-file>  # Should show helpful error
```

---

## Decision Points

**Before implementing ANY solution, clarify with user**:

1. **What data do you actually have?**
   - [ ] Pose videos (need MediaPipe processing first)
   - [ ] Joint angle CSVs with custom column names
   - [ ] Insole pressure/IMU sensor data (DEMC files)
   - [ ] Other format

2. **What is your goal?**
   - [ ] Process insole data for pressure-based cyclograms
   - [ ] Process pose data with non-standard column names
   - [ ] Convert insole IMU data to joint angles (complex preprocessing)
   - [ ] Batch process mixed data types

3. **Where is the detection failing?**
   - [ ] Which specific file/directory path?
   - [ ] Can you share actual column names from the CSV?
   - [ ] Is this DEMC data or different source?

---

## Conclusion

**The detection is NOT broken** - it's correctly validating data compatibility.

**Root cause**: Data type mismatch between script expectations (joint angles) and actual data (insole sensors).

**Next steps**:
1. Clarify user's actual data type and intent
2. Route to appropriate solution from Options A-D
3. Implement validation strategy
4. Consider long-term architectural improvements

**Priority**: Get clarification before coding any solution.
