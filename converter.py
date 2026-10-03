"""
converter.py

Reads smooth_#.rbxmx files from smooth_terrain/, runs each through the trained
neural network weights, and writes the predicted legacy terrain as legacy_#.rbxmx
into legacy_results/.

Usage:
    python converter.py

Requirements:
    pip install numpy
"""

import re
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

SMOOTH_DIR  = Path("smooth_terrain")
LEGACY_DIR  = Path("legacy_results")
WEIGHTS_FILE = Path("weights.npz")

# ── Smooth material enum value -> legacy material int ─────────────────────────
SMOOTH_TO_LEGACY_MAT = {
    1280: 1,   # Grass -> Grass
    1296: 2,   # Sand -> Sand
    848:  3,   # Brick -> Brick
    800:  4,   # Slate -> Granite
    1376: 5,   # Asphalt -> Asphalt (May have been Gravel)
    816:  14,  # Concrete -> Cement (May have been CinderBlock, Iron, Aluminum, or Gold)
    880:  13,  # Cobblestone -> MossyStone
    912:  15,  # Sandstone -> RedPlastic
    1536: 16,  # Ice -> BluePlastic
    528:  9,   # WoodPlanks -> WoodPlank (May have been WoodLog)
    2048: 17,  # Water -> Water
}
DEFAULT_LEGACY_MAT = 1  # Grass

# Occupancy thresholds -> block type
def occ_to_block(occ: float) -> str:
    if occ >= 0.9:       return "solid"
    elif occ >= 0.58:    return "inversecorner"
    elif occ >= 0.42:    return "wedge"
    elif occ > 0.05:     return "corner"
    else:                return "air"

BLOCK_OCC = {"solid":1.0, "wedge":0.5, "corner":1/3, "inversecorner":2/3, "air":0.0}

# ── Weights loader ────────────────────────────────────────────────────────────
def load_weights(path: Path) -> dict:
    """Loads weights.npz produced by trainer.py."""
    data = np.load(path)
    return {k: data[k] for k in data.files}

# ── MLP inference ─────────────────────────────────────────────────────────────
def relu(x): return np.maximum(0, x)

def mlp_forward(W: dict, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs a batch of inputs through the MLP.
    inputs: (N, 54) float32
    Returns: orient_preds (N,) int, is_vertical (N,) bool
    """
    h1 = relu(inputs @ W["w1"].T + W["b1"])
    h2 = relu(h1 @ W["w2"].T + W["b2"])
    h3 = relu(h2 @ W["w3"].T + W["b3"])
    orient_logits = h3 @ W["w_orient"].T + W["b_orient"]  # (N, 4)
    shape_logits  = h3 @ W["w_shape"].T  + W["b_shape"]   # (N, 2)
    orient_preds  = orient_logits.argmax(axis=1)
    is_vertical   = shape_logits[:, 0] > shape_logits[:, 1]
    return orient_preds, is_vertical

# ── rbxmx parser ──────────────────────────────────────────────────────────────
def parse_rbxmx(filepath: Path) -> str:
    """Extracts and reassembles the Part# StringValue text from a .rbxmx file."""
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
                m = re.match(r"^Part(\d+)$", name)
                if m:
                    parts[int(m.group(1))] = value
    return "".join(parts[k] for k in sorted(parts))

# ── Smooth data parser ────────────────────────────────────────────────────────
def parse_smooth(text: str) -> tuple[dict, tuple]:
    """
    Parses smooth export text.
    Returns:
        voxels: dict {(gx,gy,gz): (mat_enum_int, occ_float)}
        region: (minX,minY,minZ, maxX,maxY,maxZ) in studs
    """
    voxels = {}
    region = None
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("REGION|"):
            nums = [int(x) for x in line[7:].split(",")]
            region = tuple(nums)
        else:
            parts = line.split(",")
            if len(parts) == 5:
                gx,gy,gz = int(parts[0]), int(parts[1]), int(parts[2])
                mat_enum  = int(parts[3])
                occ255    = int(parts[4])
                occ       = occ255 / 255.0
                if mat_enum == 2048:
                    # Store real occupancy so output pass can use it,
                    # but build_features will skip water regardless
                    voxels[(gx,gy,gz)] = (mat_enum, occ)
                else:
                    voxels[(gx,gy,gz)] = (mat_enum, occ)
    return voxels, region

# ── Feature builder ───────────────────────────────────────────────────────────
NEIGHBOR_OFFSETS = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if not (dx == 0 and dy == 0 and dz == 0)
]

def build_features(voxels: dict) -> tuple[list, list]:
    """
    Builds MLP input features for all non-solid, non-air, non-water cells.
    Returns: (features list, cell_keys list) — parallel arrays.
    """
    features  = []
    cell_keys = []

    for (gx,gy,gz), (mat_enum, occ) in voxels.items():
        bt = occ_to_block(occ)
        if bt in ("air", "solid"):
            continue
        if mat_enum == 2048:  # water
            continue

        center_occ  = BLOCK_OCC[bt]
        center_norm = mat_enum / 2112.0

        feats = []
        for (dx,dy,dz) in NEIGHBOR_OFFSETS:
            nb = voxels.get((gx+dx, gy+dy, gz+dz))
            if nb is None:
                feats.append(0.0)  # air occupancy
                feats.append(1.0)  # resolved
            else:
                nb_mat, nb_occ = nb
                nb_occ_used = 0.0 if nb_mat == 2048 else nb_occ
                is_resolved = 0.0 if abs(nb_occ_used - 0.5) <= 0.05 else 1.0
                feats.append(nb_occ_used)
                feats.append(is_resolved)

        feats.append(center_occ)
        feats.append(center_norm)
        features.append(feats)
        cell_keys.append((gx, gy, gz, bt, mat_enum))

    return features, cell_keys

# ── Cell value packer (matches LegacyTerrain ChunkGrid) ──────────────────────
# value = (orientation * 500) + (shape * 100) + material
BLOCK_SHAPE = {"solid":0, "wedge_v":1, "corner":2, "inversecorner":3, "wedge_h":4}

def pack_cell(mat: int, shape: int, orient: int) -> int:
    return (orient * 500) + (shape * 100) + mat

# ── Legacy rbxmx writer ───────────────────────────────────────────────────────
CHUNK_SIZE = 16
CHUNK_VOL  = CHUNK_SIZE ** 3

def build_legacy_rbxmx(cell_data: dict) -> str:
    """
    cell_data: {(gx,gy,gz): packed_value}  — global voxel coords
    Returns a .rbxmx XML string containing a LegacyImport Folder with Part# StringValues.
    """
    # Group cells by chunk
    chunks: dict[tuple, dict] = {}
    for (gx,gy,gz), value in cell_data.items():
        cx = gx // CHUNK_SIZE
        cy = gy // CHUNK_SIZE
        cz = gz // CHUNK_SIZE
        lx = gx % CHUNK_SIZE
        ly = gy % CHUNK_SIZE
        lz = gz % CHUNK_SIZE
        # Buffer index: x + z*chunkSize + y*chunkSize^2
        idx = lx + lz * CHUNK_SIZE + ly * CHUNK_SIZE * CHUNK_SIZE
        key = (cx, cy, cz)
        if key not in chunks:
            chunks[key] = {}
        chunks[key][idx] = value

    # Build RLE strings per chunk (value/count format matching LegacyTerrain)
    lines = []
    for (cx,cy,cz), cells in chunks.items():
        non_empty = len(cells)
        # Build flat buffer of 4096 values
        buf = [0] * CHUNK_VOL
        for idx, val in cells.items():
            if 0 <= idx < CHUNK_VOL:
                buf[idx] = val
        # RLE encode: value/count/value/count/...
        rle_parts = []
        i = 0
        while i < len(buf):
            val = buf[i]
            count = 1
            while i + count < len(buf) and buf[i+count] == val:
                count += 1
            rle_parts.append(f"{val}/{count}")
            i += count
        rle = "/".join(rle_parts)
        lines.append(f"{cx},{cy},{cz},{non_empty}|{rle}")

    export_str = "\n".join(lines)

    # Split into Part# StringValues (190000 char limit)
    parts_xml = []
    CSIZ = 190000
    pi = 1
    ps = 0
    while ps < len(export_str):
        pe = min(ps + CSIZ, len(export_str))
        if pe < len(export_str):
            # Walk back to last newline
            while pe > ps and export_str[pe-1] != "\n":
                pe -= 1
        chunk_text = export_str[ps:pe]
        safe = chunk_text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        parts_xml.append(f"""		<Item class="StringValue">
			<Properties>
				<string name="Name">Part{pi}</string>
				<string name="Value">{safe}</string>
			</Properties>
		</Item>""")
        pi += 1
        ps = pe

    parts_joined = "\n".join(parts_xml)

    return f"""<roblox xmlns:xmime="http://www.w3.org/2005/05/xmlmime" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://www.roblox.com/roblox.xsd" version="4">
	<External>null</External>
	<External>nil</External>
	<Item class="Folder">
		<Properties>
			<string name="Name">LegacyImport</string>
		</Properties>
{parts_joined}
	</Item>
</roblox>"""

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    LEGACY_DIR.mkdir(exist_ok=True)

    smooth_files = sorted(SMOOTH_DIR.glob("smooth_*.rbxmx"))
    if not smooth_files:
        print(f"No smooth_*.rbxmx files found in {SMOOTH_DIR}/")
        return

    print(f"Loading weights from {WEIGHTS_FILE}...")
    if not WEIGHTS_FILE.exists():
        print(f"[ERROR] {WEIGHTS_FILE} not found. Run trainer.py first.")
        return
    W = load_weights(WEIGHTS_FILE)
    print(f"  Weights loaded: {list(W.keys())}")

    print(f"\nFound {len(smooth_files)} smooth file(s)\n")

    for smooth_path in smooth_files:
        m = re.search(r"smooth_(.+)\.rbxmx$", smooth_path.name)
        if not m:
            print(f"Skipping {smooth_path.name}")
            continue
        idx      = m.group(1)
        out_path = LEGACY_DIR / f"legacy_{idx}.rbxmx"

        if out_path.exists():
            print(f"[{smooth_path.name}] Already converted -> {out_path.name}, skipping")
            continue

        print(f"[{smooth_path.name}] Parsing...")
        text = parse_rbxmx(smooth_path)
        if not text:
            print("  No data found, skipping")
            continue

        voxels, region = parse_smooth(text)
        print(f"  {len(voxels)} non-air voxels, region={region}")

        print("  Building features...")
        features, cell_keys = build_features(voxels)
        print(f"  {len(features)} non-solid cells need orientation prediction")

        # Run MLP on all non-solid cells at once
        cell_data = {}

        if features:
            feat_arr = np.array(features, dtype=np.float32)
            orient_preds, is_vertical = mlp_forward(W, feat_arr)

            for i, (gx,gy,gz,bt,mat_enum) in enumerate(cell_keys):
                legacy_mat = SMOOTH_TO_LEGACY_MAT.get(mat_enum, DEFAULT_LEGACY_MAT)
                orient     = int(orient_preds[i])
                if bt == "wedge":
                    shape = BLOCK_SHAPE["wedge_v"] if is_vertical[i] else BLOCK_SHAPE["wedge_h"]
                elif bt == "corner":
                    shape = BLOCK_SHAPE["corner"]
                else:
                    shape = BLOCK_SHAPE["inversecorner"]
                cell_data[(gx,gy,gz)] = pack_cell(legacy_mat, shape, orient)

        # Add solid cells and water
        for (gx,gy,gz), (mat_enum, occ) in voxels.items():
            if mat_enum == 2048:
                # Only output water cells with non-zero occupancy
                if occ > 0:
                    cell_data[(gx,gy,gz)] = pack_cell(17, 0, 0)
            else:
                bt = occ_to_block(occ)
                if bt == "solid":
                    legacy_mat = SMOOTH_TO_LEGACY_MAT.get(mat_enum, DEFAULT_LEGACY_MAT)
                    cell_data[(gx,gy,gz)] = pack_cell(legacy_mat, 0, 0)
                # air: skip

        print(f"  {len(cell_data)} total legacy cells")
        print(f"  Writing {out_path.name}...")
        xml = build_legacy_rbxmx(cell_data)
        out_path.write_text(xml, encoding="utf-8")
        print(f"  Done. ({out_path.stat().st_size/1024:.1f} KB)")
        print()

    print("All done!")
    print()
    print("To import into Studio:")
    print("  1. In Explorer, right-click on ReplicatedStorage (or Workspace)")
    print("  2. Click 'Insert from File'")
    print("  3. Select the legacy_#.rbxmx file")
    print("  4. A 'LegacyImport' folder will appear")
    print("  5. Run: require(game:GetService('ReplicatedStorage').TerrainConverter).LoadLegacy()")

if __name__ == "__main__":
    main()