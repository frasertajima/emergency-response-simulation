#!/usr/bin/env python3
"""
WebSocket Server for Real-Time 3D Visualization

Serves simulation data to browser-based WebGL viewer.

Install:
    pip install fastapi uvicorn websockets lz4

Usage:
    python server.py
    # Then open browser to: http://localhost:8000/index_studio.html
"""

import json
from pathlib import Path

import numpy as np

try:
    import lz4.frame
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, HTMLResponse
    from fastapi.staticfiles import StaticFiles
except ImportError:
    print("ERROR: Required packages not installed!")
    print("Install with:")
    print("  pip install fastapi uvicorn websockets lz4")
    exit(1)

app = FastAPI(title="BoundaryPINN 3D Viewer")

# Paths
BASE_DIR = Path(__file__).parent.parent
# Find the most recent output directory (custom simulations only)
_all_outputs = sorted(
    [
        d
        for d in (BASE_DIR / "fortran").glob("output_custom_*")
        if d.is_dir() and list(d.glob("concentration_*.bin"))
    ],
    key=lambda d: d.name,
    reverse=True,
)
DATA_DIR = _all_outputs[0] if _all_outputs else None
WEB_DIR = Path(__file__).parent

print("=" * 80)
print("BoundaryPINN WebSocket Server")
print("=" * 80)
print(f"Data directory: {DATA_DIR if DATA_DIR else '(none - run a simulation first)'}")
print(f"Web directory: {WEB_DIR}")
print()


def auto_detect_metadata():
    """Auto-detect simulation parameters from output directory"""
    if not DATA_DIR or not DATA_DIR.exists():
        print(f"INFO: No simulation output directory found yet.")
        return None

    # Find all concentration files
    conc_files = sorted(DATA_DIR.glob("concentration_*.bin"))
    if not conc_files:
        print(f"ERROR: No concentration files found in {DATA_DIR}")
        return None

    # Extract step numbers from filenames
    step_numbers = []
    for f in conc_files:
        # Extract number from concentration_XXXX.bin
        step_num = int(f.stem.split("_")[1])
        step_numbers.append(step_num)

    step_numbers.sort()
    num_timesteps = len(step_numbers)
    max_step = step_numbers[-1]
    step_interval = step_numbers[1] - step_numbers[0] if len(step_numbers) > 1 else 1

    # Load first file to get grid dimensions
    first_file = conc_files[0]
    data = np.fromfile(first_file, dtype=np.float32)
    total_cells = len(data)

    # Assume cubic grid (128³ = 2,097,152)
    nx = int(round(total_cells ** (1 / 3)))

    # Calculate time parameters
    # dt_simulation = 0.25 (from CUDA code)
    # Files saved every step_interval simulation steps
    dt_save = 0.25 * step_interval  # Time between saved frames
    total_time = max_step * 0.25  # Total simulation time

    metadata = {
        "grid": {"nx": nx, "ny": nx, "nz": nx},
        "domain": {"size": 256.0, "dx": 2.0},
        "timesteps": num_timesteps,
        "dt": dt_save,
        "total_time": total_time,
        "step_interval": step_interval,
        "max_step": max_step,
        "physics": {
            "diffusivity": 0.5,
            "decay": 0.05,
            "wind_u": 2.0,
            "wind_v": 0.3,
            "wind_w": 0.0,
        },
    }

    print("Auto-detected configuration:")
    print(f"  Grid: {nx}³ = {total_cells:,} cells")
    print(f"  Timesteps: {num_timesteps} (steps 0 to {max_step})")
    print(f"  Step interval: {step_interval} (dt = {dt_save}s per frame)")
    print(f"  Total time: {total_time}s")
    print()

    return metadata, step_numbers


# Auto-detect metadata on startup
METADATA = None
STEP_NUMBERS = []

# Load obstacle mask (static)
obstacle_data = None


def load_obstacle():
    global obstacle_data
    if obstacle_data is None:
        print("Loading obstacle mask...")
        obstacle_file = DATA_DIR / "obstacle_mask.bin"
        if obstacle_file.exists():
            obstacle = np.fromfile(obstacle_file, dtype=np.int8)
            nx = METADATA["grid"]["nx"]
            obstacle = obstacle.reshape((nx, nx, nx), order="F")
            obstacle_data = obstacle
            print(f"  Obstacle cells: {obstacle.sum():,}")
        else:
            print("  WARNING: obstacle_mask.bin not found")
            nx = METADATA["grid"]["nx"] if METADATA else 128
            obstacle_data = np.zeros((nx, nx, nx), dtype=np.int8)
    return obstacle_data


@app.on_event("startup")
async def startup():
    """Load static data on startup"""
    global METADATA, STEP_NUMBERS

    # Auto-detect simulation configuration (may be None if no outputs exist yet)
    result = auto_detect_metadata()
    if result is None:
        print("NOTE: No simulation outputs found yet.")
        print("Use the Studio to build a scene and run a simulation.")
        METADATA = None
        STEP_NUMBERS = []
    else:
        METADATA, STEP_NUMBERS = result
        # Load obstacle mask
        load_obstacle()

    print()
    # ANSI color codes: \033[92m = bright green, \033[1m = bold, \033[0m = reset
    GREEN = "\033[92m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    print(f"{GREEN}{'=' * 60}{RESET}")
    print(f"{GREEN}{BOLD}  SERVER READY!{RESET}")
    print(f"{GREEN}  Open browser to:{RESET}")
    print()
    print(f"{GREEN}{BOLD}    >>> http://localhost:8000/index_studio.html <<<{RESET}")
    print()
    print(f"{GREEN}{'=' * 60}{RESET}")
    print()


@app.get("/")
async def root():
    """Serve main HTML page"""
    html_file = WEB_DIR / "index.html"
    if html_file.exists():
        return FileResponse(html_file)
    else:
        # Return basic HTML if file doesn't exist
        return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>BoundaryPINN 3D Viewer</title>
    <meta charset="utf-8">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f0f0f0;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }
        .status {
            padding: 15px;
            background: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 5px;
            color: #155724;
            margin: 20px 0;
        }
        .error {
            background: #f8d7da;
            border-color: #f5c6cb;
            color: #721c24;
        }
        pre {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }
        a {
            color: #007bff;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 BoundaryPINN 3D Viewer</h1>

        <div class="status">
            ✓ Server is running!
        </div>

        <h2>Quick Start</h2>
        <p>The WebGL viewer is not yet implemented. This is the backend server.</p>

        <h3>API Endpoints:</h3>
        <ul>
            <li><a href="/metadata">/metadata</a> - Simulation metadata</li>
            <li><a href="/docs">/docs</a> - API documentation</li>
            <li><code>/ws</code> - WebSocket endpoint for data streaming</li>
        </ul>

        <h3>Next Steps:</h3>
        <ol>
            <li>Create <code>web_viewer/index.html</code> with WebGL viewer</li>
            <li>Use Three.js or vtk.js for 3D rendering</li>
            <li>Connect to WebSocket at <code>ws://localhost:8000/ws</code></li>
        </ol>

        <h3>Test WebSocket:</h3>
        <pre>
# Python test client
import asyncio
import websockets
import json

async def test():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as ws:
        # Request timestep 0
        await ws.send(json.dumps({"timestep": 0}))
        data = await ws.recv()
        print(f"Received {len(data)} bytes")

asyncio.run(test())
        </pre>

        <hr>
        <p><small>BoundaryPINN v42 - Emergency Response Simulation</small></p>
    </div>
</body>
</html>
        """)


@app.get("/index_studio.html")
async def serve_studio():
    """Serve the interactive studio page"""
    html_file = WEB_DIR / "index_studio.html"
    if html_file.exists():
        return FileResponse(html_file)
    else:
        return HTMLResponse("<h1>Studio page not found</h1>", status_code=404)


@app.get("/index_2d.html")
async def serve_2d_viewer():
    """Serve the 2D viewer page"""
    html_file = WEB_DIR / "index_2d.html"
    if html_file.exists():
        return FileResponse(html_file)
    else:
        return HTMLResponse("<h1>2D viewer page not found</h1>", status_code=404)


@app.get("/index_3d.html")
async def serve_3d_viewer():
    """Serve the 3D volumetric viewer page"""
    html_file = WEB_DIR / "index_3d.html"
    if html_file.exists():
        return FileResponse(html_file)
    else:
        return HTMLResponse("<h1>3D viewer page not found</h1>", status_code=404)


@app.get("/index_original.html")
async def serve_original_viewer():
    """Serve the original working 3D volumetric viewer"""
    html_file = WEB_DIR / "index_original.html"
    if html_file.exists():
        return FileResponse(html_file)
    else:
        return HTMLResponse("<h1>Original viewer page not found</h1>", status_code=404)


@app.get("/metadata")
async def get_metadata():
    """Return simulation metadata"""
    return METADATA


@app.get("/scene")
async def get_scene():
    """Return scene configuration (obstacles, leak, wind, etc.)"""
    scene_file = DATA_DIR / "scene.json"
    if scene_file.exists():
        with open(scene_file, "r") as f:
            scene = json.load(f)
        # Add data directory name for diagnostics
        scene["_data_dir"] = DATA_DIR.name
        return scene
    else:
        # Return empty scene if not found
        return {
            "obstacles": [],
            "leak": {"position": [128, 128, 128], "rate": 0},
            "wind": {"u": 0, "v": 0, "w": 0},
            "physics": {"diffusivity": 0.5, "decay": 0.05},
        }


@app.get("/obstacle")
async def get_obstacle():
    """Return obstacle mask as compressed binary"""
    obstacle = load_obstacle()
    compressed = lz4.frame.compress(obstacle.tobytes())

    return {
        "shape": list(obstacle.shape),
        "dtype": str(obstacle.dtype),
        "compressed_size": len(compressed),
        "uncompressed_size": obstacle.nbytes,
        "compression_ratio": obstacle.nbytes / len(compressed),
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming concentration data"""
    await websocket.accept()
    print(f"Client connected: {websocket.client}")

    try:
        while True:
            # Receive request from client
            message = await websocket.receive_text()
            request = json.loads(message)

            if "timestep" in request:
                timestep = request["timestep"]
                print(f"  Request: timestep {timestep}")

                # Map timestep index to actual step number
                if timestep < 0 or timestep >= len(STEP_NUMBERS):
                    await websocket.send_json(
                        {
                            "error": f"Timestep {timestep} out of range [0, {len(STEP_NUMBERS) - 1}]",
                        }
                    )
                    continue

                step_num = STEP_NUMBERS[timestep]
                filename = DATA_DIR / f"concentration_{step_num:04d}.bin"

                if filename.exists():
                    C = np.fromfile(filename, dtype=np.float32)
                    nx = METADATA["grid"]["nx"]
                    C = C.reshape((nx, nx, nx), order="F")

                    # Sanitize data - replace inf/nan with 0 to prevent visualization crashes
                    C = np.nan_to_num(C, nan=0.0, posinf=1e10, neginf=0.0)

                    # Send uncompressed for simplicity (8MB is fine for local network)
                    # Use order='F' to preserve Fortran column-major ordering for JavaScript
                    raw_bytes = C.tobytes(order="F")

                    # Send metadata first
                    await websocket.send_json(
                        {
                            "timestep": timestep,
                            "step_num": step_num,
                            "shape": list(C.shape),
                            "dtype": "float32",
                            "compressed": False,
                            "data_size": len(raw_bytes),
                            "max_value": float(C.max()),
                            "min_value": float(C.min()),
                            "mean_value": float(C.mean()),
                        }
                    )

                    # Send raw binary data
                    await websocket.send_bytes(raw_bytes)

                    print(
                        f"    Sent {len(raw_bytes) / (1024 * 1024):.1f} MB "
                        f"(max: {C.max():.4f})"
                    )
                else:
                    await websocket.send_json(
                        {
                            "error": f"Timestep {timestep} not found",
                            "filename": str(filename),
                        }
                    )

            elif "obstacle" in request:
                # Send obstacle mask
                obstacle = load_obstacle()
                raw_bytes = obstacle.tobytes()

                await websocket.send_json(
                    {
                        "type": "obstacle",
                        "shape": list(obstacle.shape),
                        "dtype": "int8",
                        "compressed": False,
                        "data_size": len(raw_bytes),
                    }
                )

                await websocket.send_bytes(raw_bytes)
                print(f"  Sent obstacle mask: {len(raw_bytes) / (1024 * 1024):.1f} MB")

            else:
                await websocket.send_json(
                    {
                        "error": "Unknown request",
                        "valid_requests": ["timestep", "obstacle"],
                    }
                )

    except WebSocketDisconnect:
        print(f"Client disconnected: {websocket.client}")
    except Exception as e:
        print(f"WebSocket error: {e}")


# ================================================================================
# Run Simulation Endpoint (Phase 2)
# ================================================================================


@app.post("/run_simulation")
async def run_simulation(scene: dict):
    """
    Generate Fortran code from scene, compile, and execute simulation

    Returns metadata for loading results
    """
    import subprocess
    import time
    from datetime import datetime

    from fortran_generator import FortranGenerator

    try:
        print("\n" + "=" * 80)
        print("🚀 RUNNING CUSTOM SIMULATION")
        print("=" * 80)

        # Generate timestamp for unique output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir_name = f"output_custom_{timestamp}"
        output_dir = BASE_DIR / "fortran" / output_dir_name
        fortran_file = BASE_DIR / "fortran" / f"custom_scenario_{timestamp}.cuf"
        executable = BASE_DIR / "fortran" / f"custom_scenario_{timestamp}"

        # 1. Generate Fortran code
        print(f"\n📝 Step 1: Generating Fortran code...")
        print(f"   Output will go to: {output_dir}")

        generator = FortranGenerator()
        fortran_code = generator.generate(scene, output_dir_name)

        # Write Fortran code to file
        with open(fortran_file, "w") as f:
            f.write(fortran_code)

        print(f"   ✅ Generated: {fortran_file.name}")
        print(f"   Code length: {len(fortran_code)} characters")

        # Save scene configuration JSON for viewer to load obstacles
        scene_json_file = BASE_DIR / "fortran" / f"scene_{timestamp}.json"
        with open(scene_json_file, "w") as f:
            json.dump(scene, f, indent=2)
        print(f"   ✅ Saved scene config: {scene_json_file.name}")

        # 2. Compile
        print(f"\n🔨 Step 2: Compiling with nvfortran...")

        module_file = BASE_DIR / "fortran" / "boundary_pinn_module_only.cuf"

        compile_cmd = [
            "nvfortran",
            "-cuda",
            "-gpu=cc80,cc86,cc89,cc90",
            "-O3",
            "-fast",
            str(module_file),
            str(fortran_file),
            "-o",
            str(executable),
        ]

        print(f"   Command: {' '.join(compile_cmd)}")

        compile_result = subprocess.run(
            compile_cmd, capture_output=True, text=True, timeout=120
        )

        if compile_result.returncode != 0:
            print(f"   ❌ Compilation FAILED!")
            print(f"   STDERR: {compile_result.stderr}")
            return {
                "success": False,
                "error": "Compilation failed",
                "details": compile_result.stderr,
                "stdout": compile_result.stdout,
            }

        print(f"   ✅ Compilation successful!")

        # 3. Execute
        print(f"\n⚡ Step 3: Running simulation...")

        start_time = time.time()

        run_result = subprocess.run(
            [str(executable)],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=str(BASE_DIR / "fortran"),
        )

        elapsed = time.time() - start_time

        if run_result.returncode != 0:
            print(f"   ❌ Simulation FAILED!")
            print(f"   STDERR: {run_result.stderr}")
            return {
                "success": False,
                "error": "Simulation failed",
                "details": run_result.stderr,
                "stdout": run_result.stdout,
            }

        print(f"   ✅ Simulation complete in {elapsed:.2f}s!")
        print(f"\n📊 Simulation Output:")
        print(run_result.stdout)

        # 4. Copy scene JSON to output directory for viewer
        import shutil

        scene_dest = output_dir / "scene.json"
        shutil.copy(scene_json_file, scene_dest)
        print(f"   ✅ Copied scene config to output directory")

        # 5. Switch DATA_DIR to new output
        global DATA_DIR, METADATA, STEP_NUMBERS
        DATA_DIR = output_dir

        # Auto-detect new metadata
        result = auto_detect_metadata()
        if result:
            METADATA, STEP_NUMBERS = result

        print(f"\n✅ SUCCESS! Results ready for visualization")
        print("=" * 80)

        return {
            "success": True,
            "message": "Simulation complete!",
            "output_dir": str(output_dir),
            "timesteps": METADATA["timesteps"] if METADATA else 0,
            "elapsed_time": elapsed,
            "stdout": run_result.stdout,
        }

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Simulation timeout (exceeded 5 minutes)"}
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


if __name__ == "__main__":
    import uvicorn

    print("Starting server...")
    print("Press Ctrl+C to stop")
    print()

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
