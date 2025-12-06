#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import sys

START_INDEX = 0

INTRINSICS = {
    "w": 0,
    "h": 0,
    "fl_x": 0.0,
    "fl_y": 0.0,
    "cx": 0.0,
    "cy": 0.0,
}

def parse_frame_12floats(chunk: str):
    """
    Parses a chunk string expecting a color code followed by 12 float values.
    Returns (color_code, [12 floats]) or (None, None) if parsing fails.
    """
    tokens = [t for t in chunk.strip().split() if t]
    if len(tokens) < 13:
        return None, None
    color_code = tokens[0]
    try:
        vals = [float(t) for t in tokens[1:13]]
    except ValueError:
        return None, None
    return color_code, vals

def vals12_to_4x4(vals12):
    """
    Converts 12 float values into a 4x4 transformation matrix.
    The 12 values represent the first 3 rows (4 values each), 
    and the 4th row is set to [0, 0, 0, 1] (homogeneous coordinates).
    """
    r0 = vals12[0:4]
    r1 = vals12[4:8]
    r2 = vals12[8:12]
    r3 = [0.0, 0.0, 0.0, 1.0]
    return [r0, r1, r2, r3]

def build_json_for_line(line: str):
    """
    Builds a JSON structure from a line containing frame data separated by '|'.
    Each frame should contain a color code and 12 float values.
    """
    frames = []
    parts = [p for p in line.strip().split('|') if p.strip()]
    frame_id = 1
    for part in parts:
        color_code, vals12 = parse_frame_12floats(part)
        if vals12 is None:
            continue
        T = vals12_to_4x4(vals12)
        frames.append({
            "transform_matrix": T,
            "monst3r_im_id": frame_id,
            "color_code": color_code
        })
        frame_id += 1

    return {**INTRINSICS, "frames": frames}

def main(input_txt, output_dir):
    """
    Converts trajectory text file to individual JSON files.
    Each line in the input file becomes a separate JSON file in the output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = input_txt.read_text(encoding="utf-8").splitlines()
    file_index = 0
    ok, skipped = 0, 0

    for ln in lines:
        if not ln.strip():
            skipped += 1
            continue
        data = build_json_for_line(ln)
        if not data["frames"]:
            skipped += 1
            continue
        out_path = output_dir / f"{file_index:04d}.json"
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")
        print(f"[OK] {out_path.name} -> {len(data['frames'])} frame")
        ok += 1
        file_index += 1

    print(f"\nCompleted: {ok} JSON files, {skipped} skipped -> {output_dir.resolve()}")

if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
