# Active Size Grasp

This repository provides a function to execute a trained grasping policy for boxes of various sizes using Isaac Sim.

## Files
- `grasp_active_size.py`: Contains the `grasp_active_size` function which loads the model and runs the inference environment.
- `active_size.zip`: The pre-trained SAC model for grasping.

## Usage

You can import and use the function in your Python scripts:

```python
from grasp_active_size import grasp_active_size

# Define box position (x, y, z) and half-extents (hx, hy, hz) in meters
# Example: Box at (0.6, 0.0, 0.45) with size 5cm x 5cm x 5cm
success = grasp_active_size(
    x=0.6, y=0.0, z=0.45,
    hx=0.025, hy=0.025, hz=0.025,
    headless=False
)

if success:
    print("Grasping successful!")
else:
    print("Grasping failed.")
```

### Arguments for `grasp_active_size`:
- `x`, `y`, `z` (float): Center position of the box in world coordinates [meters]
- `hx`, `hy`, `hz` (float): Half-extents of the box [meters] (e.g. `0.025` = 5cm width)
- `app`: Optional existing Isaac Sim SimulationApp instance
- `headless` (bool): If True, runs without GUI rendering

## Requirements
- Isaac Sim
- stable-baselines3
- pybullet
- gymnasium
