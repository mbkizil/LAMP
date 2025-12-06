#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import re

def parse_qwen_line(line: str) -> str:
    """
    Parses a line according to Qwen format.

    If the line starts with "free-form", extracts patterns like:
    "t_x: v1 t_y: v2 t_z: v3 yaw: v4 tilt: v5 roll: v6"
    and converts them to "v1 v2 v3 v4 v5 v6".

    If it doesn't start with "free-form", returns the line as-is.
    """
    line = line.strip()
    
    if not line.startswith("free-form"):
        return line

    content_to_parse = line[len("free-form"):].strip()

    pattern = re.compile(
        r"t_x:\s*(\S+)\s+"
        r"t_y:\s*(\S+)\s+"
        r"t_z:\s*(\S+)\s+"
        r"yaw:\s*(\S+)\s+"
        r"tilt:\s*(\S+)\s+"
        r"roll:\s*(\S+)"
    )

    matches = pattern.findall(content_to_parse)

    if not matches:
        return content_to_parse

    output_parts = []
    for match_tuple in matches:
        output_parts.append(" ".join(match_tuple))

    return " ".join(output_parts)

def main():
    """Main function: converts Qwen format input file to tag format output file."""
    if len(sys.argv) != 3:
        print(f"Error: Invalid usage.", file=sys.stderr)
        print(f"Usage: python3 {sys.argv[0]} <input.txt> <output.txt>", file=sys.stderr)
        sys.exit(1)

    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    try:
        with open(input_file_path, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Error reading input file: {e}", file=sys.stderr)
        sys.exit(1)

    processed_lines = []
    for ln in lines:
        ln = ln.rstrip('\n')
        processed = parse_qwen_line(ln)
        processed_lines.append(processed)

    try:
        with open(output_file_path, 'w', encoding='utf-8') as f_out:
            for i, pl in enumerate(processed_lines):
                f_out.write(pl)
                if i != len(processed_lines) - 1:
                    f_out.write('\n')
    except Exception as e:
        print(f"Error: Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing completed.")
    print(f"Input: {input_file_path}")
    print(f"Output: {output_file_path}")

if __name__ == "__main__":
    main()