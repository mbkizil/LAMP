#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path
from typing import List, Tuple
import sys
# ===================== AYARLAR ===================== #
# All relative paths (relative to the directory this script is in):
INPUT_TXT = Path("./generated_bbox_trajectories_test.txt")  # her satır: "tx ty tz, tx ty tz, ..."
OUT_DIR   = Path("./generated_bbox_trajectories_test")          # JSON'lar buraya yazılır
START     = 0                            # 0000.json için 0
PAD       = 4                            # dosya adı basamak sayısı

# Quantization transformation
TRANS_OFFSET = 256.0
TRANS_SCALE  = 1.0 / 256.0  # QUAT_SCALE ile aynı

# ===================== TRANSFORMATION ===================== #
def dequantize_tx_ty_tz(tx_i: float, ty_i: float, tz_i: float) -> Tuple[float, float, float]:
    """ tx =  (tx_i - 256)/256; ty = -(ty_i - 256)/256; tz = -(tz_i - 256)/256 """
    tx = (tx_i - TRANS_OFFSET) * TRANS_SCALE
    ty = (ty_i - TRANS_OFFSET) * TRANS_SCALE
    tz = (tz_i - TRANS_OFFSET) * TRANS_SCALE -100/256
    return tx, -ty, -tz

def parse_line_to_triplets(line: str) -> List[Tuple[int, int, int]]:
    """
    Parses triplets in a line. Flexible: ',', ';', '|' or just space.
    Example line:
      256 256 256, 299 235 277 | 342 214 298 ; 342 235 277 342 256 256
    """
    s = line.strip()
    if not s or s.startswith("#"):
        return []
    parts = re.split(r"[,\|;]+", s)  # strong separators
    triplets: List[Tuple[int, int, int]] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        nums = part.split()
        if len(nums) == 0:
            continue
        if len(nums) % 3 != 0:
            raise ValueError(f"Piece is not a multiple of 3: '{part}' (number of tokens {len(nums)})")
        for i in range(0, len(nums), 3):
            tx_i, ty_i, tz_i = nums[i:i+3]
            triplets.append((int(tx_i), int(ty_i), int(tz_i)))
    return triplets

def build_json_from_triplets(triplets: List[Tuple[int, int, int]]) -> dict:
    frames = []
    for idx, (tx_i, ty_i, tz_i) in enumerate(triplets, start=1):
        tx, ty, tz = dequantize_tx_ty_tz(tx_i, ty_i, tz_i)
        transform_matrix = [
            [1.0, 0.0, 0.0, tx],
            [0.0, 1.0, 0.0, ty],
            [0.0, 0.0, 1.0, tz],
            [0.0, 0.0, 0.0, 1.0],
        ]
        frames.append({
            "transform_matrix": transform_matrix,
            "monst3r_im_id": idx
        })
    return {
        "w": 0, "h": 0,
        "fl_x": 0.0, "fl_y": 0.0,
        "cx": 0.0, "cy": 0.0,
        "frames": frames
    }

def write_numbered_json(data: dict, out_dir: Path, index: int, pad: int = 4) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{index:0{pad}d}.json"
    out_path = out_dir / name
    out_path.write_text(json.dumps(data, indent=4), encoding="utf-8")
    return out_path

# ===================== ANA AKIŞ ===================== #
def run_default_mode():
    """ 
    Original behavior: Reads INPUT_TXT, writes one JSON per line to OUT_DIR.
    """
    print(f"Running in default mode...")
    print(f"Input: {INPUT_TXT.resolve()}")
    print(f"Output: {OUT_DIR.resolve()}")
    
    if not INPUT_TXT.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_TXT.resolve()}")

    lines = [ln.rstrip("\n") for ln in INPUT_TXT.read_text(encoding="utf-8").splitlines()]
    file_index = START
    created = 0
    skipped = 0

    for line_no, line in enumerate(lines, start=1):
        try:
            triplets = parse_line_to_triplets(line)
            if not triplets:
                skipped += 1
                continue
            data = build_json_from_triplets(triplets)
            out_path = write_numbered_json(data, OUT_DIR, file_index, pad=PAD)
            print(f"[OK] line {line_no}: -> {out_path.name} (frames={len(triplets)})")
            file_index += 1
            created += 1
        except Exception as e:
            print(f"[WARN] line {line_no} skipped: {e}")
            skipped += 1

    print("\n--- Summary (Default Mode) ---")
    print(f"Created JSON: {created}")
    print(f"Skipped lines: {skipped}")
    print(f"Output directory: {OUT_DIR.resolve()}")


def run_single_file_mode(input_file_path: Path, output_file_path: Path):
    """
    New behavior: Reads the given <input.txt>, combines all lines,
    and builds a single JSON object.
    """
    print(f"Running in single file mode...")
    print(f"Input: {input_file_path.resolve()}")
    print(f"Output: {output_file_path.resolve()}")

    if not input_file_path.exists():
        print(f"Error: Input file not found: {input_file_path.resolve()}")
        sys.exit(1)

    lines = [ln.rstrip("\n") for ln in input_file_path.read_text(encoding="utf-8").splitlines()]
    
    all_triplets: List[Tuple[int, int, int]] = []
    skipped = 0
    processed_lines = 0

    for line_no, line in enumerate(lines, start=1):
        try:
            triplets = parse_line_to_triplets(line)
            if not triplets:
                skipped += 1
                continue
            all_triplets.extend(triplets) # <--- Tümünü tek listeye ekliyoruz
            processed_lines += 1
        except Exception as e:
            print(f"[WARN] Line {line_no} skipped (error): {e}")
            skipped += 1

    if not all_triplets:
        print("Error: No valid data found in input file.")
        sys.exit(1)

    print(f"Total {processed_lines} lines processed, {len(all_triplets)} frames found.")
    
    # Create a single JSON object from all frames
    final_data = build_json_from_triplets(all_triplets)

    try:
        # Ensure the output directory exists
        output_file_path.parent.mkdir(parents=True, exist_ok=True) 
        output_file_path.write_text(json.dumps(final_data, indent=4), encoding="utf-8")
    except Exception as e:
        print(f"Error: Output file could not be written: {output_file_path.resolve()} - {e}")
        sys.exit(1)

    print("\n--- Summary (Single File Mode) ---")
    print(f"Processed lines:   {processed_lines}")
    print(f"Skipped lines:   {skipped}")
    print(f"Total frames:    {len(all_triplets)}")
    print(f"Output file:   {output_file_path.resolve()}")


def main():
    
    if len(sys.argv) == 1:
        run_default_mode()
    
    # len(sys.argv) == 3 means there are 2 arguments (script_name, input, output)
    elif len(sys.argv) == 3:
        input_file = Path(sys.argv[1])
        output_file = Path(sys.argv[2])
        run_single_file_mode(input_file, output_file)
    
    else:
        # Invalid usage
        print("Invalid usage!", file=sys.stderr)
        print(f"  Default mode: python3 {sys.argv[0]}", file=sys.stderr)
        print(f"  Single file mode: python3 {sys.argv[0]} <input.txt> <output.json>", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()