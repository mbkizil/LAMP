import numpy as np
from scipy.spatial.transform import Rotation
import sys
import os

# --- Ayarlar ---
#INPUT_FILENAME = 'generated_relative_trajectories_test.txt'  
#OUTPUT_FILENAME = 'generated_relative_trajectories_test_matrix.txt'
# ----------------

def map_int_to_float(val_int):
    
    val_01 = val_int / 512.0
    val_float = (val_01 * 2.0) - 1.0
    return val_float

def process_reverse_line(line):
    if not line.strip():
        return None

    str_frames = line.strip().split(' | ')
    output_matrix_frames = []

    for frame_str in str_frames:
        try:
            tokens = frame_str.strip().split()
            if len(tokens) < 8:
                color_code = "#000000"
                values_int = [int(v) for v in tokens[0:7]]                
            else:
                color_code = tokens[0]
                values_int = [int(v) for v in tokens[1:8]]

            rx_int, ry_int, rw_int, rz_int, tx_int, ty_int, tz_int = values_int

            rw_f = -map_int_to_float(rw_int)
            rx_f = map_int_to_float(rx_int)
            ry_f = -map_int_to_float(ry_int)
            rz_f = map_int_to_float(rz_int)
            tx_f = map_int_to_float(tx_int)
            ty_f = -map_int_to_float(ty_int)
            tz_f = -map_int_to_float(tz_int)

            r = Rotation.from_quat([rx_f, ry_f, rz_f, rw_f])
            R_matrix_3x3 = r.as_matrix()
            t_vector_1x3 = np.array([tx_f, ty_f, tz_f])
            pose_matrix_3x4 = np.hstack((R_matrix_3x3, t_vector_1x3.reshape(3, 1)))
            values_flat_12 = pose_matrix_3x4.flatten()
            frame_out_str = f"{color_code} " + " ".join(str(v) for v in values_flat_12)
            output_matrix_frames.append(frame_out_str)

        except Exception as e:
            print(f"Warning: Invalid frame skipped. ({e}) -> '{frame_str}'")
            continue

    return " | ".join(output_matrix_frames)

def main(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return

    print(f"Input: {input_file}")
    print(f"Output: {output_file}")

    processed_count = 0
    with open(input_file, 'r') as in_f, open(output_file, 'w') as out_f:
        for line in in_f:
            processed = process_reverse_line(line)
            if processed:
                out_f.write(processed + '\n')
                processed_count += 1

    print(f"Completed. {processed_count} lines processed -> {output_file}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
