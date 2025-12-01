# Sample Data Directory

This directory should contain the 10MWT.csv sample file for testing and demonstration.

## Expected File

- `10MWT.csv` - Smart insole sensor data from a 10-meter walk test

## Data Format

The 10MWT.csv file should contain synchronized sensor data with the following columns:

### Timestamp
- `timestamp` - Milliseconds (converted to seconds during processing)

### Pressure Sensors (4 per foot)
- `L_value1`, `L_value2`, `L_value3`, `L_value4` - Left foot pressure sensors
- `R_value1`, `R_value2`, `R_value3`, `R_value4` - Right foot pressure sensors

Sensor positions:
- value1, value3: Midfoot
- value2: Forefoot
- value4: Hindfoot (heel)

### Accelerometers (3-axis per foot)
- `L_ACC_X`, `L_ACC_Y`, `L_ACC_Z` - Left foot accelerometer
- `R_ACC_X`, `R_ACC_Y`, `R_ACC_Z` - Right foot accelerometer

### Gyroscopes (3-axis per foot)
- `L_GYRO_X`, `L_GYRO_Y`, `L_GYRO_Z` - Left foot gyroscope
- `R_GYRO_X`, `R_GYRO_Y`, `R_GYRO_Z` - Right foot gyroscope

## Sampling Rate

Default: 100 Hz (configurable via `--sampling-rate` parameter)

## Usage

```bash
# Analyze the 10MWT sample
python3 "CYCLOGRAM-PROCESSING Script/insole-analysis.py" \
  --input insole-sample/10MWT.csv \
  --output insole-output/10MWT
```

## Note

If you don't have the 10MWT.csv file, please contact the repository maintainers for access to sample data for academic purposes.
