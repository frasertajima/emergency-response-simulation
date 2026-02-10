#!/usr/bin/env python3
"""
Fortran Code Generator for Interactive Studio

Converts 3D scene JSON from browser into CUDA Fortran code.
Generates obstacle initialization code from user-placed shapes.
"""

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class FortranGenerator:
    """Generate CUDA Fortran code from scene configuration"""

    def __init__(self, template_path: str = None):
        if template_path is None:
            template_path = (
                Path(__file__).parent.parent / "fortran" / "template_scenario.cuf"
            )

        with open(template_path, "r") as f:
            self.template = f.read()

    def generate(
        self, scene_config: Dict[str, Any], output_dir: str = "output_custom_scenario"
    ) -> str:
        """
        Generate complete Fortran code from scene configuration

        Args:
            scene_config: Dictionary with obstacle, leak, wind, physics, simulation params
            output_dir: Output directory name (relative to fortran/ directory)

        Returns:
            Complete Fortran source code as string
        """

        # Extract parameters
        leak = scene_config["leak"]
        wind = scene_config["wind"]
        physics = scene_config["physics"]
        sim = scene_config["simulation"]

        # Calculate save interval (dt = 0.05 in template for stability)
        save_interval = max(1, int((sim["total_time"] / sim["frames"]) / 0.05))

        # Generate obstacle initialization code
        obstacle_code = self._generate_obstacle_code(scene_config["obstacles"])

        # Replace placeholders
        code = self.template
        code = code.replace("{{GENERATION_TIME}}", datetime.now().isoformat())
        code = code.replace("{{DIFFUSIVITY}}", str(physics["diffusivity"]))
        code = code.replace("{{DECAY_RATE}}", str(physics["decay"]))
        code = code.replace("{{TOTAL_TIME}}", str(sim["total_time"]))
        code = code.replace("{{SAVE_INTERVAL}}", str(save_interval))
        code = code.replace("{{WIND_U}}", str(wind["u"]))
        code = code.replace("{{WIND_V}}", str(wind["v"]))
        code = code.replace("{{WIND_W}}", str(wind["w"]))
        # Leak position from builder (in world coordinates 0-256, need to convert to grid 1-128)
        # World coordinate / 2.0 gives grid index (since domain is 256m with 128 cells = 2m per cell)
        leak_x = max(1, min(128, int(leak["position"][0] / 2.0)))
        leak_y = max(1, min(128, int(leak["position"][1] / 2.0)))
        leak_z = max(1, min(128, int(leak["position"][2] / 2.0)))

        code = code.replace("{{LEAK_X}}", str(leak_x))
        code = code.replace("{{LEAK_Y}}", str(leak_y))
        code = code.replace("{{LEAK_Z}}", str(leak_z))
        code = code.replace("{{LEAK_RATE}}", str(leak["rate"]))
        code = code.replace("{{LEAK_GROWTH}}", str(leak["growth_time"]))
        code = code.replace("{{LEAK_DECAY}}", str(leak["decay_start"]))
        code = code.replace("{{OBSTACLE_INITIALIZATION}}", obstacle_code)
        code = code.replace("{{OUTPUT_DIR}}", output_dir)

        return code

    def _generate_obstacle_code(self, obstacles: List[Dict[str, Any]]) -> str:
        """Generate Fortran code to initialize obstacles from 3D shapes"""

        if not obstacles:
            return "    ! No obstacles defined\n"

        code_lines = []
        code_lines.append("    ! User-placed obstacles")
        code_lines.append("")

        for idx, obj in enumerate(obstacles):
            shape_type = obj["type"]
            pos = obj["position"]  # [x, y, z] in world coordinates
            rot = obj.get("rotation", [0, 0, 0])  # [x, y, z] rotations in radians
            scale = obj.get("scale", [1, 1, 1])  # [x, y, z] scale factors
            is_obstacle = obj.get("isObstacle", True)

            # Skip if not an actual obstacle (e.g., openings)
            if not is_obstacle:
                code_lines.append(
                    f"    ! Shape {idx + 1}: {shape_type} (opening - not solid)"
                )
                continue

            code_lines.append(f"    ! Shape {idx + 1}: {shape_type}")

            if shape_type == "box":
                code_lines.extend(self._generate_box_code(pos, scale))
            elif shape_type == "cylinder":
                code_lines.extend(self._generate_cylinder_code(pos, scale))
            elif shape_type == "panel":
                code_lines.extend(self._generate_panel_code(pos, scale))
            else:
                code_lines.append(f"    ! Unknown shape type: {shape_type}")

            code_lines.append("")

        return "\n".join(code_lines)

    def _generate_box_code(self, pos: List[float], scale: List[float]) -> List[str]:
        """Generate Fortran code for a box obstacle"""

        # Default box size: 30x40x20 (from createBox function)
        base_size = [30, 40, 20]

        # Apply scale
        size = [base_size[i] * scale[i] for i in range(3)]

        # Convert from world coordinates to grid indices
        # Grid is 128³ with dx=2.0, so grid_index = world_coord / 2.0
        # Fortran convention: i=X, j=Y, k=Z (same as v42)
        dx = 2.0  # 256m domain / 128 cells

        i_min = max(1, int((pos[0] - size[0] / 2) / dx))
        i_max = min(128, int((pos[0] + size[0] / 2) / dx))
        j_min = max(1, int((pos[1] - size[1] / 2) / dx))
        j_max = min(128, int((pos[1] + size[1] / 2) / dx))
        k_min = max(1, int((pos[2] - size[2] / 2) / dx))
        k_max = min(128, int((pos[2] + size[2] / 2) / dx))

        code = [
            f"    ! Box at world ({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f}), size ({size[0]:.0f}, {size[1]:.0f}, {size[2]:.0f})",
            f"    do k = {k_min}, {k_max}",
            f"        do j = {j_min}, {j_max}",
            f"            do i = {i_min}, {i_max}",
            f"                obstacle(i, j, k) = 1",
            f"            end do",
            f"        end do",
            f"    end do",
        ]

        return code

    def _generate_cylinder_code(
        self, pos: List[float], scale: List[float]
    ) -> List[str]:
        """Generate Fortran code for a cylindrical obstacle"""

        # Default cylinder: radius 15, height 60 (from createCylinder function)
        base_radius = 15.0
        base_height = 60.0

        # Apply scale (assume uniform scaling for radius, Y scale for height)
        radius = base_radius * math.sqrt(
            scale[0] * scale[2]
        )  # Average of X and Z scale
        height = base_height * scale[1]

        # Fortran convention: i=X, j=Y, k=Z (same as v42)
        # Cylinder axis is vertical (Y), so loop over j for height
        # Use world coordinates for distance check (like v42)
        center_x = pos[0]
        center_z = pos[2]
        j_min = max(1, int((pos[1] - height / 2) / 2.0))
        j_max = min(128, int((pos[1] + height / 2) / 2.0))

        code = [
            f"    ! Cylinder at world ({center_x:.1f}, {pos[1]:.1f}, {center_z:.1f}), radius={radius:.1f}, height={height:.1f}",
            f"    do k = 1, nz",
            f"        do j = {j_min}, {j_max}",
            f"            do i = 1, nx",
            f"                x = (i - 0.5) * dx",
            f"                z = (k - 0.5) * dx",
            f"                r = sqrt((x - {center_x:.1f})**2 + (z - {center_z:.1f})**2)",
            f"                if (r < {radius:.1f}) obstacle(i, j, k) = 1",
            f"            end do",
            f"        end do",
            f"    end do",
        ]

        return code

    def _generate_panel_code(self, pos: List[float], scale: List[float]) -> List[str]:
        """Generate Fortran code for a panel (thin wall/floor)"""

        # Default panel: 50x2x50 (from createPanel function)
        base_size = [50, 2, 50]

        # Apply scale
        size = [base_size[i] * scale[i] for i in range(3)]

        # Convert to grid indices (world / dx)
        # Fortran convention: i=X, j=Y, k=Z (same as v42)
        dx = 2.0  # 256m domain / 128 cells

        i_min = max(1, int((pos[0] - size[0] / 2) / dx))
        i_max = min(128, int((pos[0] + size[0] / 2) / dx))
        j_min = max(1, int((pos[1] - size[1] / 2) / dx))
        j_max = min(128, int((pos[1] + size[1] / 2) / dx))
        k_min = max(1, int((pos[2] - size[2] / 2) / dx))
        k_max = min(128, int((pos[2] + size[2] / 2) / dx))

        code = [
            f"    ! Panel at world ({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f}), size ({size[0]:.0f}, {size[1]:.0f}, {size[2]:.0f})",
            f"    do k = {k_min}, {k_max}",
            f"        do j = {j_min}, {j_max}",
            f"            do i = {i_min}, {i_max}",
            f"                obstacle(i, j, k) = 1",
            f"            end do",
            f"        end do",
            f"    end do",
        ]

        return code


def test_generator():
    """Test the generator with sample scene"""

    sample_scene = {
        "domain": {"nx": 256, "ny": 256, "nz": 256, "dx": 1.0},
        "obstacles": [
            {
                "type": "box",
                "position": [128, 64, 128],
                "rotation": [0, 0, 0],
                "scale": [1, 1, 1],
                "isObstacle": True,
            },
            {
                "type": "cylinder",
                "position": [180, 30, 180],
                "rotation": [0, 0, 0],
                "scale": [1, 1, 1],
                "isObstacle": True,
            },
            {
                "type": "panel",
                "position": [128, 0, 128],
                "rotation": [0, 0, 0],
                "scale": [2, 1, 2],
                "isObstacle": True,
            },
        ],
        "leak": {
            "position": [128, 32, 128],
            "rate": 5.0,
            "growth_time": 40,
            "decay_start": 60,
        },
        "wind": {"u": 2.0, "v": 0.3, "w": 0.0},
        "physics": {"diffusivity": 0.5, "decay": 0.05},
        "simulation": {"total_time": 120, "frames": 60},
    }

    generator = FortranGenerator()
    code = generator.generate(sample_scene)

    print("Generated Fortran code:")
    print("=" * 80)
    print(code[:2000])  # Print first 2000 chars
    print("...")
    print("=" * 80)

    # Save to file
    output_file = Path("test_generated_scenario.cuf")
    with open(output_file, "w") as f:
        f.write(code)

    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    test_generator()
