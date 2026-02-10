# Emergency Response Simulation Studio v44

An interactive 3D scenario builder and GPU-accelerated pollutant dispersion simulator for emergency response planning and training.

https://youtu.be/Dy1IKznezwY

## Overview

This tool allows users to:

1. **Build scenarios** - Place obstacles (buildings, tanks, barriers) and leak sources in a 3D environment
2. **Configure physics** - Set wind conditions, diffusion rates, and leak parameters
3. **Run simulations** - Execute GPU-accelerated fluid dynamics on NVIDIA GPUs
4. **Visualize results** - Watch pollutant clouds evolve and interact with obstacles in real-time 3D

## Physics Model

The simulation solves the **advection-diffusion-decay equation** for pollutant transport:

```
dC/dt = -u dot grad(C) + D * laplacian(C) - k*C + S
```

Where:
- **C** - Pollutant concentration field
- **u** - Wind velocity vector (u, v, w)
- **D** - Diffusion coefficient (turbulent spreading)
- **k** - Decay rate (chemical degradation/deposition)
- **S** - Source term (leak injection)

The leak source follows a temporal profile:
- **Growth phase** (0 to growth_time): Linear ramp from 0 to leak_rate
- **Steady phase** (growth_time to decay_start): Constant at leak_rate  
- **Decay phase** (after decay_start): Exponential decay

Obstacles are treated as no-flux boundaries - the pollutant cloud flows around them realistically.

## CUDA Fortran Engine

The simulation runs on NVIDIA GPUs using CUDA Fortran for maximum performance:

- **Grid**: 128 x 128 x 128 cells (2m resolution, 256m domain)
- **Timestep**: 0.05s (adaptive for stability)
- **Output**: Binary concentration fields at configurable intervals

The Fortran code is auto-generated from scene configurations and compiled on-the-fly using `nvfortran`.

## Requirements

### Hardware
- NVIDIA GPU with Compute Capability 8.0+ (RTX 30xx, 40xx, A100, etc.)

### Software
- NVIDIA HPC SDK (for `nvfortran` compiler)
- Python 3.8+
- Modern web browser (Chrome, Firefox, Edge)

### Python Dependencies

```bash
pip install fastapi uvicorn websockets lz4 numpy
```

## Setup

1. **Install NVIDIA HPC SDK**
   
   Download from: https://developer.nvidia.com/hpc-sdk
   
   Ensure `nvfortran` is in your PATH:
   ```bash
   nvfortran --version
   ```

2. **Install Python dependencies**
   ```bash
   cd emergency_response/v44_interactive_studio/web_viewer
   pip install fastapi uvicorn websockets lz4 numpy
   ```

3. **Start the server**
   ```bash
   python server.py
   ```

4. **Open the builder**
   
   Navigate to: http://localhost:8000/index_studio.html

## Usage Workflow

### 1. Build Your Scene

- **Add obstacles**: Click shape buttons (Box, Cylinder, Panel) then click in the 3D view to place
- **Position objects**: Select and drag with transform controls, or use Scale/Rotate modes
- **Place leak source**: Click "Place Leak" then click to position the red leak ball
- **Configure parameters**: Adjust leak rate, wind speed, diffusion, etc. in the side panel

### 2. Run Simulation

- Click **RUN SIMULATION** to generate Fortran code, compile, and execute
- Progress is shown in the status bar
- Typical simulation takes 10-30 seconds depending on GPU

### 3. View Results

- Click **VIEW RESULTS** to open the 3D viewer in a new tab
- Use playback controls to step through time
- Adjust threshold and opacity to explore the concentration field

### 4. Save/Load Scenes

- **Save Scene**: Downloads current configuration as JSON to your browser's default download directory
- **Load Scene**: Opens a file picker to load a previously saved scene
- **New Scene**: Clears everything and resets to defaults

Note: Scene files are saved to your browser's default download location (typically ~/Downloads). This is a browser security restriction that cannot be changed.

## Example Scenes

Three starter scenes are included in `example_scenes/`:

1. **factory_simple.json** - Single building with ground-level leak, moderate wind
2. **tank_farm.json** - Multiple cylindrical tanks with elevated leak source
3. **urban_canyon.json** - Street canyon between buildings demonstrating channeling effects

Load these via the "Load Scene" button to explore different scenarios.

## File Structure

```
v44_interactive_studio/
├── README.md
├── .gitignore
├── web_viewer/
│   ├── server.py              # FastAPI server + WebSocket streaming
│   ├── index_studio.html      # 3D scenario builder interface
│   ├── index_3d.html          # 3D results viewer
│   └── fortran_generator.py   # Scene-to-Fortran code generator
├── fortran/
│   ├── template_scenario.cuf  # Fortran template with placeholders
│   ├── boundary_pinn_module_only.cuf  # GPU kernels module
│   └── output_custom_*/       # Simulation output directories (generated)
└── example_scenes/
    └── *.json                 # Example scenario files
```

## Diagnostics

The viewer includes diagnostic displays for debugging:

- **Leak Source (from scene.json)**: Shows the leak position loaded from configuration
- **Cloud Center (from data)**: Computes center-of-mass of the concentration field
- **Offset from Leak**: Distance between cloud center and leak position (should be small initially, grows with wind advection)

These help verify that simulations are running correctly and data is being loaded properly.

## Known Limitations

- **Obstacle rotation**: The Fortran simulation engine currently does not support rotated obstacles. Rotation values are stored in scene files but ignored during simulation. To achieve different orientations, adjust the scale dimensions instead (e.g., swap X and Z scale values to rotate 90 degrees around Y).

- **Scene files download location**: When saving scenes, files are saved to your browser's default download directory. This is a browser security restriction.

- **Limited leakage**: This first version is only 120 seconds at most.
- Current Limitations:

### 1. Grid Resolution Bottleneck
- **2m cells can't resolve small gaps** - A 1m gap doesn't exist in the simulation
- Sub-grid features are completely invisible to the solver
- This explains why gas doesn't seep through narrow openings

### 2. Staircase Boundaries
- Obstacles are voxelized to the grid
- Curved surfaces become blocky stairs
- Flow around cylinders is poorly represented

### 3. No Turbulence Modeling
- Laminar flow only (no eddies, no mixing enhancement)
- Unrealistic for real emergency scenarios with complex terrain

### 4. Uniform Wind Field
- Constant wind everywhere
- No building wake effects, no channeling computed dynamically

## Where CNN/PINN Could Help

### Option 1: Sub-Grid CNN for Gap Flow
Train a CNN to predict **effective permeability** through sub-grid features.

### Option 2: Super-Resolution Post-Processing
Train a CNN to upscale results.

### Option 3: Learned Turbulence Closure
Replace the simple diffusion coefficient D with a CNN-predicted field.

### Option 4: Full PINN Approach
Train a neural network to satisfy the PDE directly. PINNs struggle with advection-dominated problems and sharp gradients. Our initial efforts failed.

In short, this first version is only a baseline for an eventual PINN, not the PINN we were hoping for. That work remains to be solved.

## Troubleshooting

### "nvfortran not found"
Ensure NVIDIA HPC SDK is installed and `nvfortran` is in your PATH.

### Simulation runs but cloud appears in wrong location
Restart the server to ensure it loads the latest simulation output.

### View Results shows old data
After loading a new scene, you must run a simulation before viewing results. The View Results button is hidden until a simulation completes.

### Popup blocked when clicking View Results
Allow popups for localhost:8000 in your browser settings.

## License

MIT License - See LICENSE file for details.
