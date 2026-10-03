"""
training_data_encoder.py

Reads terrain_model_#.rbxmx files from model_files/, extracts legacy terrain
chunk data, generates synthetic smooth terrain counterparts, and saves paired
training samples to training_data/terrain_sample_#.npz

Usage:
    python training_data_encoder.py

Requirements:
    pip install numpy
"""

import os
import re
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

MODEL_DIR   = Path("model_files")
OUTPUT_DIR  = Path("training_data")
CHUNK_SIZE  = 16

# ── Material mapping ──────────────────────────────────────────────────────────
LEGACY_MAT_TO_SMOOTH = {
    0:  0,     # Air -> Air
    1:  1280,  # Grass -> Grass
    2:  1536,  # Sand -> Sand
    3:  1040,  # Brick -> Brick
    4:  1360,  # Granite -> Granite
    5:  1312,  # Asphalt -> Asphalt
    6:  816,   # Iron -> Concrete (merged)
    7:  816,   # Aluminum -> Concrete (merged)
    8:  816,   # Gold -> Concrete (merged)
    9:  1200,  # WoodPlank -> WoodPlanks
    10: 1300,  # WoodLog -> Wood
    11: 1312,  # Gravel -> Asphalt (absorbed)
    12: 816,   # CinderBlock -> Concrete (merged)
    13: 1032,  # MossyStone -> Cobblestone
    14: 816,   # Cement -> Concrete
    15: 1344,  # RedPlastic -> Sandstone
    16: 1648,  # BluePlastic -> Ice
    17: 2112,  # Water -> Water
    18: 1280,  # Lava -> Grass (no smooth equivalent)
    19: 1328,  # Snow -> Snow
}

WATER_MAT = 17  # Legacy material int for Water

SHAPE_TO_OCCUPANCY = {
    0: 1.0,
    1: 0.5,
    2: 1.0 / 3.0,
    3: 2.0 / 3.0,
    4: 0.5,
}

# ── Cell packing ──────────────────────────────────────────────────────────────
def unpack_cell(value: int) -> tuple[int, int, int]:
    mat    = value % 100
    shape  = (value // 100) % 5
    orient = (value // 500) % 6
    return mat, shape, orient

def verify_unpacking():
    assert unpack_cell(1)    == (1, 0, 0), f"Got {unpack_cell(1)}"
    assert unpack_cell(101)  == (1, 1, 0), f"Got {unpack_cell(101)}"
    assert unpack_cell(501)  == (1, 0, 1), f"Got {unpack_cell(501)}"
    assert unpack_cell(1701) == (1, 2, 3), f"Got {unpack_cell(1701)}"
    print("  Cell unpacking verification passed")

# ── RLE decoder ───────────────────────────────────────────────────────────────
def decode_rle(rle_string: str) -> np.ndarray:
    parts = rle_string.strip().split("/")
    result = []
    i = 0
    while i + 1 < len(parts):
        try:
            value = int(parts[i])
            count = int(parts[i + 1])
            result.extend([value] * count)
            i += 2
        except ValueError:
            i += 1

    expected = CHUNK_SIZE ** 3
    if len(result) < expected:
        result.extend([0] * (expected - len(result)))
    elif len(result) > expected:
        result = result[:expected]

    return np.array(result, dtype=np.uint32)

# ── rbxmx parser ─────────────────────────────────────────────────────────────
def parse_rbxmx(filepath: Path) -> dict:
    tree = ET.parse(filepath)
    root = tree.getroot()

    parts = {}
    for item in root.iter("Item"):
        if item.get("class") == "StringValue":
            name_el  = item.find(".//Properties/string[@name='Name']")
            value_el = item.find(".//Properties/string[@name='Value']")
            if name_el is not None and value_el is not None:
                name  = name_el.text or ""
                value = value_el.text or ""
                if name.startswith("Part"):
                    try:
                        parts[int(name[4:])] = value
                    except ValueError:
                        pass

    if not parts:
        print(f"  [WARN] No Part# StringValues found in {filepath.name}")
        return {}

    full_text = "".join(parts[k] for k in sorted(parts.keys()))
    print(f"  Reassembled {len(parts)} parts, {len(full_text)} chars")

    chunks = {}
    bad_lines = 0
    for line in full_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            header, rle = line.split("|", 1)
            cx, cy, cz, non_empty = [int(x) for x in header.split(",")]
            cell_values = decode_rle(rle)
            chunks[(cx, cy, cz)] = cell_values
        except Exception as e:
            bad_lines += 1
            if bad_lines <= 3:
                print(f"  [WARN] Bad line: {e} | {line[:60]}")

    if bad_lines > 0:
        print(f"  [WARN] {bad_lines} lines failed to parse")

    return chunks

# ── Training sample generator ─────────────────────────────────────────────────
def generate_training_samples(chunks: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates paired (smooth_input, legacy_label) arrays.

    smooth_input: float32 (N, 54)
        26 neighbors x 2 channels (occupancy, is_resolved) + center_occ + center_smooth_mat_normalized

    legacy_label: int16 (N, 3)
        [shape, orientation, material] — only for non-solid, non-air, non-water cells

    Water cells are treated as air throughout — excluded as training samples
    and treated as occupancy=0 / resolved when appearing as neighbors.
    """
    CHUNK_VOL = CHUNK_SIZE ** 3

    print("  Building cell lookup...")
    cell_map = {}  # (gx,gy,gz) -> (mat, shape, orient)

    for (cx, cy, cz), buf in chunks.items():
        base_x = cx * CHUNK_SIZE
        base_y = cy * CHUNK_SIZE
        base_z = cz * CHUNK_SIZE
        for idx in range(CHUNK_VOL):
            v = int(buf[idx])
            if v == 0:
                continue
            mat, shape, orient = unpack_cell(v)
            if mat == 0:
                continue
            # Treat water cells as air — don't add them to cell_map at all
            if mat == WATER_MAT:
                continue
            lx = idx % CHUNK_SIZE
            lz = (idx // CHUNK_SIZE) % CHUNK_SIZE
            ly = idx // (CHUNK_SIZE * CHUNK_SIZE)
            cell_map[(base_x+lx, base_y+ly, base_z+lz)] = (mat, shape, orient)

    print(f"  {len(cell_map)} non-air non-water cells in lookup")

    NEIGHBOR_OFFSETS = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]

    smooth_rows = []
    label_rows  = []

    for (gx, gy, gz), (mat, shape, orient) in cell_map.items():
        if shape == 0:  # Solid — no orientation to predict
            continue

        center_occ  = SHAPE_TO_OCCUPANCY.get(shape, 0.5)
        center_smat = LEGACY_MAT_TO_SMOOTH.get(mat, 1280)

        neighbor_feats = []
        for (dx, dy, dz) in NEIGHBOR_OFFSETS:
            nb = cell_map.get((gx+dx, gy+dy, gz+dz))
            if nb is None:
                # Air, water (excluded from cell_map), or out of bounds —
                # all treated identically as empty/resolved
                neighbor_feats.append(0.0)  # occupancy = air
                neighbor_feats.append(1.0)  # is_resolved = yes
            else:
                nb_mat, nb_shape, _ = nb
                nb_occ = SHAPE_TO_OCCUPANCY.get(nb_shape, 0.5)
                is_resolved = 0.0 if nb_shape in (1, 4) else 1.0
                neighbor_feats.append(nb_occ)
                neighbor_feats.append(is_resolved)

        feature = neighbor_feats + [center_occ, center_smat / 2112.0]
        smooth_rows.append(feature)
        label_rows.append([shape, orient, mat])

    if not smooth_rows:
        return np.array([]), np.array([])

    return np.array(smooth_rows, dtype=np.float32), np.array(label_rows, dtype=np.int16)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    model_files = sorted(MODEL_DIR.glob("terrain_model_*.rbxmx"))
    if not model_files:
        print(f"No terrain_model_*.rbxmx files found in {MODEL_DIR}/")
        return

    print(f"Found {len(model_files)} model file(s)\n")

    print("Verifying cell unpacking...")
    verify_unpacking()
    print()

    for model_path in model_files:
        match = re.search(r"terrain_model_(\d+)", model_path.name)
        if not match:
            print(f"Skipping {model_path.name} (unexpected filename)")
            continue
        idx      = match.group(1)
        out_path = OUTPUT_DIR / f"terrain_sample_{idx}.npz"

        if out_path.exists():
            print(f"[{model_path.name}] Already encoded -> {out_path.name}, skipping")
            continue

        print(f"[{model_path.name}] Parsing...")
        chunks = parse_rbxmx(model_path)
        if not chunks:
            print("  No chunk data found, skipping\n")
            continue
        print(f"  {len(chunks)} chunks loaded")

        print("  Generating training samples...")
        smooth_arr, label_arr = generate_training_samples(chunks)

        if smooth_arr.size == 0:
            print("  No non-solid cells found — terrain may be entirely flat. Skipping.\n")
            continue

        np.savez_compressed(out_path, smooth=smooth_arr, labels=label_arr)

        shape_names = {1:"VWedge", 2:"Corner", 3:"InvCorner", 4:"HWedge"}
        dist = " ".join(
            f"{shape_names[s]}={int(np.sum(label_arr[:,0]==s))}"
            for s in range(1, 5) if np.sum(label_arr[:,0]==s) > 0
        )
        print(f"  Saved {out_path.name}")
        print(f"    smooth : {smooth_arr.shape} float32 ({smooth_arr.nbytes/1024:.1f} KB)")
        print(f"    labels : {label_arr.shape} int16   ({label_arr.nbytes/1024:.1f} KB)")
        print(f"    shapes : {dist}")
        print()

    print("All done!")

if __name__ == "__main__":
    main()