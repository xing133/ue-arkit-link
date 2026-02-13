"""
Blender import script for expression-detect blendshape data.

Usage:
  1. In Blender, select the mesh object that has ARKit-compatible shape keys
  2. Open this script in Blender's Script Editor
  3. Edit TARGET_OBJECT and DATA_FILE below
  4. Run the script

The mesh must already have shape keys matching ARKit names
(e.g. "eyeBlinkLeft", "mouthSmileRight").
You can create them with the ARKitBlendshapeHelper addon:
  https://github.com/elijah-atkins/ARKitBlendshapeHelper
"""

import json

# === EDIT THESE ===
TARGET_OBJECT = "YourMeshName"  # Name of the Blender mesh with shape keys
DATA_FILE = "/path/to/output/video_blendshapes.json"  # Path to exported JSON
# ==================

try:
    import bpy
except ImportError:
    print("This script must be run inside Blender's Python environment.")
    print("Open it in Blender's Script Editor and run from there.")
    raise SystemExit(1)


def import_blendshapes(filepath: str, target_object_name: str) -> None:
    obj = bpy.data.objects.get(target_object_name)
    if obj is None:
        print(f"Error: Object '{target_object_name}' not found in scene")
        return

    shape_keys = obj.data.shape_keys
    if shape_keys is None:
        print(f"Error: Object '{target_object_name}' has no shape keys")
        return

    with open(filepath) as f:
        data = json.load(f)

    fps = data["metadata"]["fps"]
    bpy.context.scene.render.fps = int(fps)
    bpy.context.scene.frame_end = data["metadata"]["total_frames"]

    applied = 0
    skipped = set()

    for frame_data in data["frames"]:
        frame_num = frame_data["frame_index"]
        if not frame_data["faces"]:
            continue

        bs_values = frame_data["faces"][0]["blendshapes"]

        for bs_name, bs_value in bs_values.items():
            if bs_name in shape_keys.key_blocks:
                kb = shape_keys.key_blocks[bs_name]
                kb.value = bs_value
                kb.keyframe_insert(data_path="value", frame=frame_num)
                applied += 1
            elif bs_name not in skipped:
                skipped.add(bs_name)

    print(f"Imported {applied} keyframes across {data['metadata']['total_frames']} frames")
    if skipped:
        print(f"Skipped blendshapes (no matching shape key): {', '.join(sorted(skipped))}")


if __name__ == "__main__":
    import_blendshapes(DATA_FILE, TARGET_OBJECT)
