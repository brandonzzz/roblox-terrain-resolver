"""
convert_weights.py

One-time conversion of existing weights.lua -> weights.npz.
Run this once so converter.py can use the trained weights without retraining.

Usage:
    python convert_weights.py
"""

import re
import numpy as np
from pathlib import Path

WEIGHTS_LUA = Path("weights.lua")
WEIGHTS_NPZ = Path("weights.npz")

def parse_lua_vector(s):
    """Parse a Lua vector literal like {1.0,2.0,...} into a numpy array."""
    s = s.strip().strip("{}")
    return np.array([float(x) for x in s.split(",")], dtype=np.float32)

def parse_lua_matrix(s):
    """Parse a Lua matrix literal like {{...},{...},...} into a numpy 2D array."""
    # Extract each row {...}
    rows = re.findall(r'\{([^{}]+)\}', s)
    return np.array([[float(x) for x in row.split(",")] for row in rows], dtype=np.float32)

def main():
    print(f"Reading {WEIGHTS_LUA}...")
    text = WEIGHTS_LUA.read_text(encoding="utf-8")

    # Strip comments and the outer return { ... }
    # We'll extract each named key's value using regex

    def extract_key(name, text):
        """Extract the raw value string for a given key from the Lua table."""
        # Match: name = <value>, where value is either {...} (possibly nested) or a scalar
        # We need to handle nested braces for matrices
        pattern = rf'\b{re.escape(name)}\s*=\s*'
        m = re.search(pattern, text)
        if not m:
            raise ValueError(f"Key '{name}' not found in weights.lua")
        start = m.end()
        # Find the matching closing brace
        if text[start] != '{':
            raise ValueError(f"Expected '{{' after key '{name}'")
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
        raise ValueError(f"Unmatched braces for key '{name}'")

    # Determine number of hidden layers by looking for w1, w2, w3...
    layer_weights = []
    layer_biases = []
    i = 1
    while True:
        try:
            wval = extract_key(f"w{i}", text)
            bval = extract_key(f"b{i}", text)
        except ValueError:
            break
        print(f"  Parsing w{i}, b{i}...")
        layer_weights.append(parse_lua_matrix(wval))
        layer_biases.append(parse_lua_vector(bval))
        i += 1

    print(f"  Found {len(layer_weights)} hidden layers")

    print("  Parsing w_orient, b_orient...")
    w_orient = parse_lua_matrix(extract_key("w_orient", text))
    b_orient = parse_lua_vector(extract_key("b_orient", text))

    print("  Parsing w_shape, b_shape...")
    w_shape = parse_lua_matrix(extract_key("w_shape", text))
    b_shape = parse_lua_vector(extract_key("b_shape", text))

    # Print shapes for verification
    for idx, (w, b) in enumerate(zip(layer_weights, layer_biases)):
        print(f"  w{idx+1}: {w.shape}  b{idx+1}: {b.shape}")
    print(f"  w_orient: {w_orient.shape}  b_orient: {b_orient.shape}")
    print(f"  w_shape:  {w_shape.shape}   b_shape:  {b_shape.shape}")

    # Build save dict
    save_dict = {}
    for idx, (w, b) in enumerate(zip(layer_weights, layer_biases)):
        save_dict[f"w{idx+1}"] = w
        save_dict[f"b{idx+1}"] = b
    save_dict["w_orient"] = w_orient
    save_dict["b_orient"] = b_orient
    save_dict["w_shape"]  = w_shape
    save_dict["b_shape"]  = b_shape

    np.savez_compressed(WEIGHTS_NPZ, **save_dict)
    print(f"\nSaved {WEIGHTS_NPZ} ({WEIGHTS_NPZ.stat().st_size/1024:.1f} KB)")
    print("Done! You can now run converter.py.")

if __name__ == "__main__":
    main()