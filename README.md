# Roblox Terrain Resolver

Convert Roblox **Smooth Terrain** back into **Legacy (block) Terrain**, using a small neural network to work out how each wedge and corner block should be oriented.

Smooth Terrain stores each 4×4×4 voxel as a material plus an *occupancy* value (0–1). Legacy Terrain stores each cell as a material, a **shape** (solid, vertical wedge, corner wedge, inverse corner wedge, horizontal wedge) and an **orientation** (one of four rotations). You can estimate the shape from occupancy, but you can't get the orientation from one voxel. This project trains an MLP on real Legacy Terrain builds so it learns to predict orientation from a cell's 26 neighbors.

```
 Legacy builds ──► training_data_encoder.py ──► trainer.py ──► weights.npz
                                                                   │
 Smooth Terrain ──► ExportSmooth() ──► converter.py ◄──────────────┘
                                           │
                                           ▼
                               legacy_*.rbxmx ──► LoadLegacy() in Studio
```

Current weights (`weights.npz` / `weights.lua`) score **99.7%** orientation accuracy and **100%** wedge-shape accuracy on the training set.

---

## Repository layout

| Path | Description |
| --- | --- |
| `training_data_encoder.py` | Reads Legacy Terrain exports from `model_files/` and writes paired feature/label samples to `training_data/`. |
| `trainer.py` | Trains the MLP in plain NumPy and writes `weights.npz` (used by the converter) and `weights.lua` (Lua table form). |
| `converter.py` | Reads Smooth Terrain exports from `smooth_terrain/`, runs inference and writes Legacy Terrain to `legacy_results/`. |
| `weights.npz`, `weights.lua` | Pre-trained network weights. |
| `model_files/` | Legacy Terrain training exports (`terrain_model_#.rbxmx`). |
| `training_data/` | Encoded training samples (`terrain_sample_#.npz`). |
| `smooth_terrain/` | Smooth Terrain exports to convert (`smooth_<name>.rbxmx`). **Git-ignored** because the files are large. Put your own exports here. |
| `legacy_results/` | Converted output (`legacy_<name>.rbxmx`), ready to import into Studio. |
| `resources/terrain_converter.rbxm` | `TerrainConverter` ModuleScript: Studio-side export/import functions. |
| `resources/legacy_terrain_package.rbxm` | `LegacyTerrain` runtime (chunk grid, mesher, multithreaded meshing, textures). |
| `resources/water_system.rbxm` | Legacy water: `LegacyWaterServer` / `WaterTerrain`, `WaterTerrainPhysics` and `LegacyWaterClient`. |
| `resources/conversion_test_region_part.rbxm` | `ConversionTestRegion` part that marks the area `ExportSmooth()` reads. |
| `resources/convert_weights.py` | One-time utility that rebuilds `weights.npz` from `weights.lua`. |

---

## Requirements

- Python 3.10+
- NumPy

```bash
pip install numpy
```

- Roblox Studio, for exporting and importing terrain

Run every Python script from the repository root, because they use relative paths.

---

## Studio setup

Insert the models from `resources/` into your place (**right-click → Insert from File**):

| Model | Put it in |
| --- | --- |
| `TerrainConverter` (ModuleScript) | `ReplicatedStorage` |
| `LegacyTerrain` | `ReplicatedStorage` |
| `WaterTerrainPhysics` (ModuleScript) | `ReplicatedStorage` |
| `LegacyWaterServer` (Script, contains `WaterTerrain`) | `ServerScriptService` |
| `LegacyWaterClient` (LocalScript) | `StarterPlayer.StarterPlayerScripts` |
| `ConversionTestRegion` (Part) | `Workspace` (only needed for smooth exports) |

Run the `TerrainConverter` functions from the Studio command bar.

---

## Usage

### 1. Convert Smooth Terrain to Legacy Terrain (using the pre-trained weights)

**Export the smooth terrain from Studio**

1. Move and resize `ConversionTestRegion` until it covers the area you want to convert.
2. Run:
   ```lua
   require(game.ReplicatedStorage.TerrainConverter).ExportSmooth(workspace.ConversionTestRegion)
   ```
3. Right-click `ReplicatedStorage.SmoothExport` → **Save to File** → `smooth_terrain/smooth_<name>.rbxmx`.

**Convert**

```bash
python converter.py
```

Each `smooth_<name>.rbxmx` becomes `legacy_results/legacy_<name>.rbxmx`. The converter skips files that already have an output, so delete an output file if you want it regenerated.

**Import the result into Studio**

1. Right-click `ReplicatedStorage` → **Insert from File** → choose `legacy_<name>.rbxmx`. A `LegacyImport` folder appears.
2. Run:
   ```lua
   local TC = require(game.ReplicatedStorage.TerrainConverter)
   TC.LoadLegacy()      -- builds Legacy Terrain chunks and places water voxels
   TC.ApplyWaterlog()   -- optional: fills non-solid cells that sit next to water
   ```

### 2. Retrain the model (optional)

1. In a place with Legacy Terrain loaded, run:
   ```lua
   require(game.ReplicatedStorage.TerrainConverter).ExportLegacy()
   ```
   Right-click `ReplicatedStorage.LegacyExport` → **Save to File** → `model_files/terrain_model_<n>.rbxmx`.
2. Encode the training samples:
   ```bash
   python training_data_encoder.py
   ```
3. Train:
   ```bash
   python trainer.py
   ```
   This overwrites `weights.npz` and `weights.lua`. It keeps the checkpoint with the best orientation accuracy.

If you only have `weights.lua`, rebuild `weights.npz` with `python resources/convert_weights.py`. Run it from a directory that contains `weights.lua`.

---

## How it works

### Smooth to Legacy shape mapping

Each smooth voxel's occupancy picks a Legacy block shape:

| Occupancy | Legacy shape |
| --- | --- |
| ≥ 0.90 | Solid |
| 0.58 – 0.90 | Inverse corner wedge |
| 0.42 – 0.58 | Wedge (the network picks vertical or horizontal) |
| 0.05 – 0.42 | Corner wedge |
| ≤ 0.05 | Air |

Water is handled separately and goes through the legacy water system.

### Network

- **Input (54 features):** each of the 26 neighbors contributes its occupancy and an "is resolved" flag (whether its shape is ambiguous). The last two features are the center cell's occupancy and its normalized material.
- **Body:** `54 → 128 → 128 → 64`, ReLU.
- **Heads:**
  - `orient` (4 classes: −Z, +X, +Z, −X)
  - `shape` (2 classes: vertical or horizontal wedge; only used for cells with occupancy ≈ 0.5)
- **Training:** Adam (lr 1e-3), batch size 256, 200 epochs, L2 regularization 1e-4. Written from scratch in NumPy with no ML framework.

### Legacy cell encoding

Cells are packed as `orientation * 500 + shape * 100 + material` and stored per 16³ chunk as run-length-encoded `value/count` pairs. Each line has the form `cx,cy,cz,nonEmpty|rle`. The text is split across `Part#` StringValues of at most 190,000 characters each so it fits inside Roblox's string limits.

### Material mapping

Smooth materials map to the closest Legacy material: Grass→Grass, Sand→Sand, Brick→Brick, Slate→Granite, Asphalt→Asphalt, Concrete→Cement, Cobblestone→MossyStone, Sandstone→RedPlastic, Ice→BluePlastic, WoodPlanks→WoodPlank, Water→Water. Any other material falls back to Grass. Edit `SMOOTH_TO_LEGACY_MAT` in `converter.py` to change the mapping.

---

## Credits

- The `LegacyTerrain` runtime in `resources/legacy_terrain_package.rbxm` was written by **@dylanleonardo**. All rights to that module belong to its author.
- Conversion pipeline, training code and water system by [@brandonzzz](https://github.com/brandonzzz).
