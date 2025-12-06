import math
import os
import sys
from collections import Counter
import random

SHOT_LENGTH = 21
SEGMENT_COUNT = 4
PARAMS_PER_SEGMENT = 6
ORIGINAL_SEGMENT_LENGTH = 5


def clamp(value, lo=0, hi=512):
    """Clamps a value to the specified range and returns it as an integer."""
    return int(round(max(lo, min(hi, value))))

def interpolate_sequence(start, end, num_frames):
    """Creates a linear interpolation sequence between start and end values over num_frames."""
    if num_frames < 1:
        return [end]
    sequence = []
    for frame_idx in range(1, num_frames + 1):
        t = frame_idx / num_frames
        sequence.append(start + (end - start) * t)
    return sequence

def quaternion_from_euler(yaw, pitch, roll):
    """
    Converts Euler angles (in degrees) to quaternion using YXZ rotation order.
    Rotation order: Y (yaw) -> X (pitch) -> Z (roll).
    """
    cy, sy = math.cos(math.radians(yaw) * 0.5), math.sin(math.radians(yaw) * 0.5)
    cp, sp = math.cos(math.radians(pitch) * 0.5), math.sin(math.radians(pitch) * 0.5)
    cr, sr = math.cos(math.radians(roll) * 0.5), math.sin(math.radians(roll) * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return {"rw": w, "rx": x, "ry": y, "rz": z}

def format_traj_line(traj):
    """Formats trajectory data for file output. Each frame contains quaternion (rw, rx, ry, rz) and translation (tx, ty, tz) values separated by spaces, frames separated by '|'."""
    frame_strs = []
    for f in traj:
        frame_vals = [f["rw"], f["rx"], f["ry"], f["rz"], f["tx"], f["ty"], f["tz"]]
        frame_strs.append(" ".join(str(v) for v in frame_vals))
    return " | ".join(frame_strs)

def determine_pattern(segments):
    """
    Determines the structural pattern (e.g., AABB, ABCD) from the given 4 segments.
    Returns a pattern string where identical segments are represented by the same letter.
    """
    if not segments or len(segments) != 4:
        return "ABCD"
    full_segments = [tuple(seg) for seg in segments]
    counts = Counter(full_segments)
    
    if len(counts) == 1:
        return "AAAA"
    elif len(counts) == 4:
        return "ABCD"
    elif len(counts) == 3:
        if full_segments[0] == full_segments[1]: return "AABC"
        if full_segments[1] == full_segments[2]: return "ABBC"
        if full_segments[2] == full_segments[3]: return "ABCC"
        else: return "ABCD"
    elif len(counts) == 2:
        if full_segments[0] == full_segments[1] and full_segments[2] == full_segments[3]:
            return "AABB"
        elif full_segments[0] == full_segments[1] == full_segments[2]:
            return "AAAB"
        elif full_segments[1] == full_segments[2] == full_segments[3]:
            return "ABBB"
        else:
            return "ABCD"
    return "ABCD"

def get_move_distance(move_str, segment_strength=1):
    """
    Determines movement distance magnitude based on movement tag and segment strength.
    Returns only a positive magnitude value. Direction is determined in the transform function.
    """
    if move_str == "no":
        return 0

    base_distance = 64
    max_distance = base_distance * segment_strength
    
    if "far" in move_str:
        distance = max_distance
    elif "near" in move_str:
        distance = max_distance / 3.0
    else:
        distance = (max_distance * 2.0) / 3.0
    
    return distance

def create_rotation_matrix(yaw, pitch, roll):
    """
    Converts Euler angles (in degrees) to a 3x3 rotation matrix using YXZ rotation order.
    This uses the exact same rotation standard as 'quaternion_from_euler'.
    """
    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(pitch)
    roll_rad = math.radians(roll)
    
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)
    
    r11 = cy * cr + sy * sp * sr
    r12 = -cy * sr + sy * sp * cr
    r13 = sy * cp
    
    r21 = cp * sr
    r22 = cp * cr
    r23 = -sp
    
    r31 = -sy * cr + cy * sp * sr
    r32 = sy * sr + cy * sp * cr
    r33 = cy * cp
    
    return [
        [r11, r12, r13],
        [r21, r22, r23],
        [r31, r32, r33]
    ]

def transform_movement_to_world(move_x_str, move_y_str, move_z_str, segment_strength, current_yaw, current_pitch, current_roll):
    """
    Transforms movement strings to world space based on camera's current rotation.
    Uses standard YXZ rotation matrix and standard local axis convention:
    Right=+X, Up=+Y, Forward=-Z
    """
    distance_x = get_move_distance(move_x_str, segment_strength)
    distance_y = get_move_distance(move_y_str, segment_strength)
    distance_z = get_move_distance(move_z_str, segment_strength)
    
    if distance_x == 0 and distance_y == 0 and distance_z == 0:
        return 0, 0, 0
    
    camera_right = 0
    camera_up = 0
    camera_forward = 0
    
    if "left" in move_x_str:
        camera_right = -distance_x
    elif "right" in move_x_str:
        camera_right = distance_x
    
    if "up" in move_y_str:
        camera_up = distance_y
    elif "down" in move_y_str:
        camera_up = -distance_y
        
    if "in" in move_z_str:
        camera_forward = -distance_z
    elif "out" in move_z_str:
        camera_forward = distance_z
    
    rot_matrix = create_rotation_matrix(current_yaw, current_pitch, current_roll)
    
    world_dx = (rot_matrix[0][0] * camera_right + 
                rot_matrix[0][1] * camera_up + 
                rot_matrix[0][2] * camera_forward)
    
    world_dy = (rot_matrix[1][0] * camera_right + 
                rot_matrix[1][1] * camera_up + 
                rot_matrix[1][2] * camera_forward)
    
    world_dz = (rot_matrix[2][0] * camera_right + 
                rot_matrix[2][1] * camera_up + 
                rot_matrix[2][2] * camera_forward)
    
    return world_dx, world_dy, world_dz

def group_consecutive_segments(pattern):
    """Groups consecutive identical characters in the pattern string."""
    if not pattern:
        return []
    groups = []
    current_char = pattern[0]
    count = 1
    for i in range(1, len(pattern)):
        if pattern[i] == current_char:
            count += 1
        else:
            groups.append((current_char, count))
            current_char = pattern[i]
            count = 1
    groups.append((current_char, count))
    return groups

def generate_trajectory_from_tags(tag_params):
    """
    Generates a consistent trajectory from the given tag list.
    Coordinate conventions:
    - X: 0=Left, 512=Right
    - Y: 0=Down, 512=Up
    - Z: 0=In, 512=Out
    - Yaw: positive = Right (CW)
    - Pitch: positive = Up (CCW)
    - Roll: positive = Clockwise (CW)
    """
    
    segments = [tag_params[i*PARAMS_PER_SEGMENT : (i+1)*PARAMS_PER_SEGMENT] for i in range(SEGMENT_COUNT)]
    pattern = determine_pattern(segments)
    groups = group_consecutive_segments(pattern)
    
    traj = []
    current_tx, current_ty, current_tz = 256, 256, 256
    current_yaw, current_pitch, current_roll = 0, 0, 0
    
    quat = quaternion_from_euler(current_yaw, current_pitch, current_roll)
    traj.append({
        "tx": clamp(current_tx), "ty": clamp(current_ty), "tz": clamp(current_tz),
        "rw": clamp((quat["rw"] + 1) * 256), "rx": clamp((quat["rx"] + 1) * 256),
        "ry": clamp((quat["ry"] + 1) * 256), "rz": clamp((quat["rz"] + 1) * 256),
    })

    segment_idx = 0
    for char, consecutive_count in groups:
        seg = segments[segment_idx]
        move_x, move_y, move_z, rot_yaw, rot_pitch, rot_roll = seg
        
        start_yaw, start_pitch, start_roll = current_yaw, current_pitch, current_roll
        
        target_yaw = current_yaw - rot_yaw
        target_pitch = current_pitch + rot_pitch
        target_roll = current_roll - rot_roll
        
        group_frames = consecutive_count * ORIGINAL_SEGMENT_LENGTH
        
        yaw_sequence = interpolate_sequence(start_yaw, target_yaw, group_frames)
        pitch_sequence = interpolate_sequence(start_pitch, target_pitch, group_frames)
        roll_sequence = interpolate_sequence(start_roll, target_roll, group_frames)
        
        for i in range(group_frames):
            frame_yaw = yaw_sequence[i]
            frame_pitch = pitch_sequence[i]
            frame_roll = roll_sequence[i]
            
            frame_move_strength = consecutive_count / float(group_frames)
            
            world_dx, world_dy, world_dz = transform_movement_to_world(
                move_x, move_y, move_z, frame_move_strength,
                frame_yaw, frame_pitch, frame_roll
            )
            
            current_tx += world_dx
            current_ty += world_dy
            current_tz += world_dz
            
            quat = quaternion_from_euler(frame_yaw, frame_pitch, frame_roll)
            traj.append({
                "tx": clamp(current_tx), "ty": clamp(current_ty), "tz": clamp(current_tz),
                "rw": clamp((quat["rw"] + 1) * 256), "rx": clamp((quat["rx"] + 1) * 256),
                "ry": clamp((quat["ry"] + 1) * 256), "rz": clamp((quat["rz"] + 1) * 256),
            })
            
        current_yaw, current_pitch, current_roll = target_yaw, target_pitch, target_roll
        segment_idx += consecutive_count
        
    return traj


if __name__ == "__main__":
    tags_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    random.seed(99)

    if not os.path.exists(tags_file_path):
        print(f"Error: Input file not found -> '{tags_file_path}'"); sys.exit(1)

    if os.path.exists(output_file_path):
        os.remove(output_file_path)
        print(f"Existing output file deleted: {output_file_path}")

    with open(tags_file_path, "r", encoding="utf-8") as f:
        tag_lines = f.readlines()

    total_lines = len(tag_lines)
    print(f"Reading {total_lines} tags from '{tags_file_path}'...")
    print(f"Trajectories will be written to '{output_file_path}'...")

    for i, line in enumerate(tag_lines):
        parts = line.strip().split()
        
        if len(parts) != PARAMS_PER_SEGMENT * SEGMENT_COUNT:
            print(f"Warning: Line {i+1} has missing parameters (expected: {PARAMS_PER_SEGMENT * SEGMENT_COUNT}, found: {len(parts)}), skipping.")
            sys.stderr.write(f"Warning: Line {i+1} has missing parameters (expected: {PARAMS_PER_SEGMENT * SEGMENT_COUNT}, found: {len(parts)}), skipping.\n")
            sys.exit(1)
            
        parsed_params = []
        for j in range(SEGMENT_COUNT):
            base_idx = j * PARAMS_PER_SEGMENT
            parsed_params.extend(parts[base_idx : base_idx + 3])
            parsed_params.extend(map(int, parts[base_idx + 3 : base_idx + 6]))

        trajectory = generate_trajectory_from_tags(parsed_params)
        traj_line = format_traj_line(trajectory)
        
        with open(output_file_path, "a", encoding="utf-8") as out_f:
            out_f.write(traj_line + "\n")

        if (i + 1) % 500 == 0 or (i + 1) == total_lines:
            print(f"Progress: {i + 1}/{total_lines} ({100*(i + 1)/total_lines:.1f}%)")
            
    print("\nCompleted! All trajectories generated with consistent rotation logic.")