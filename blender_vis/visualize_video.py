# -*- coding: utf-8 -*-
import os
import sys
import logging
from pathlib import Path
import pathlib
import json
import shutil
import numpy as np
import bpy
import re
import trimesh
import tempfile
from mathutils import Vector

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
if bpy.data.filepath:
    sys.path.append(os.path.dirname(bpy.data.filepath))

try:
    from PIL import Image
except Exception:
    print("[FATAL] Pillow (PIL).")
    raise

logger = logging.getLogger(__name__)

def reset_scene():
    if bpy.context.active_object and bpy.context.active_object.mode == 'EDIT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for _ in range(4):
        bpy.ops.outliner.orphans_purge()
    for world in bpy.data.worlds:
        bpy.data.worlds.remove(world)
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    
    # --- Tam Beyaz Arka Plan ---
    bg_node = nt.nodes.new(type='ShaderNodeBackground')
    bg_node.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0) 
    bg_node.inputs['Strength'].default_value = 1.0
    output_node = nt.nodes.new(type='ShaderNodeOutputWorld')
    nt.links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])
    
    # --- Renk Yönetimi (Standard) ---
    bpy.context.scene.view_settings.view_transform = 'Standard'
    bpy.context.scene.view_settings.look = 'None'
    
    bpy.ops.object.camera_add(location=(0, 0, 0))
    bpy.context.active_object.name = "Camera"
    
    bpy.ops.object.light_add(type='SUN', align='WORLD', location=(0, 0, 10))
    light = bpy.context.active_object
    light.name = "Sun"
    light.data.energy = 3.0
    light.rotation_euler[0] = np.deg2rad(45)
    light.rotation_euler[1] = np.deg2rad(-30)


# === RENK VE YARDIMCI FONKSİYONLAR ===
def hex_to_rgb(hex_str):
    h = hex_str.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def rgb_to_hex(rgb_tuple):
    r, g, b = [int(c * 255) for c in rgb_tuple]
    return f"#{r:02x}{g:02x}{b:02x}"

SEGMENT_PALETTES = [
    {"gradient_hex": ["#fe9929", "#d95f0e", "#993404"], "caption_hex": "#993404"},
    {"gradient_hex": ["#41b6c4", "#2c7fb8", "#253494"], "caption_hex": "#253494"},
    {"gradient_hex": ["#df65b0", "#dd1c77", "#980043"], "caption_hex": "#980043"},
    {"gradient_hex": ["#78c679", "#31a354", "#006837"], "caption_hex": "#006837"}
]
for palette in SEGMENT_PALETTES:
    palette["gradient_rgb"] = [hex_to_rgb(h) for h in palette["gradient_hex"]]

def generate_segmented_color_gradient(num_steps: int, palette_rgb):
    if num_steps <= 1: return [palette_rgb[0]]
    gradient = []
    num_sections = len(palette_rgb) - 1
    for i in range(num_steps):
        t_global = i / (num_steps - 1) if num_steps > 1 else 0
        section = int(t_global * num_sections)
        if section >= num_sections: section = num_sections - 1
        t_local = (t_global - section / num_sections) * num_sections if num_sections > 0 else 0
        start_rgb, end_rgb = palette_rgb[section], palette_rgb[section + 1]
        r, g, b = [(1 - t_local) * s + t_local * e for s, e in zip(start_rgb, end_rgb)]
        gradient.append((r, g, b))
    return gradient

def generate_simple_gradient(num_steps: int, start_rgb, end_rgb):
    if num_steps <= 1: return [start_rgb]
    gradient = []
    for i in range(num_steps):
        t = i / (num_steps - 1)
        r, g, b = [(1 - t) * s + t * e for s, e in zip(start_rgb, end_rgb)]
        gradient.append((r, g, b))
    return gradient

def load_json(file):
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)

def load_transforms_raw_with_colors(transform_p: str):
    data = load_json(transform_p)
    frames = data.get('frames', [])
    if not frames: raise ValueError(f"JSON ERROR: {transform_p}")
    transform = [np.array(item['transform_matrix']) for item in frames]
    color_codes = [item.get("color_code", "#FFFFFF") for item in frames]
    c2ws = np.stack(transform, axis=0)[:, :3, :]
    c2ws[:,:,0] = -c2ws[:,:,0]
    c2ws[:,:,2] = -c2ws[:,:,2]
    return c2ws, color_codes

def normalize_scene(cam_c2ws, obj_c2ws):
    all_positions = np.vstack([cam_c2ws[:, :3, 3], obj_c2ws[:, :3, 3]])
    center = np.mean(all_positions, axis=0)
    scale = np.max(np.linalg.norm(all_positions - center, axis=1)) + 1e-6
    final_scale = scale / 5.0
    cam_c2ws_norm = cam_c2ws.copy()
    cam_c2ws_norm[:, :3, 3] = (cam_c2ws[:, :3, 3] - center) / final_scale
    obj_c2ws_norm = obj_c2ws.copy()
    obj_c2ws_norm[:, :3, 3] = (obj_c2ws[:, :3, 3] - center) / final_scale
    return cam_c2ws_norm, obj_c2ws_norm

def get_meshes_bounds(mesh_objects):
    all_vertices = []
    for obj in mesh_objects:
        if obj.type in ('MESH', 'CURVE') and obj.visible_get() and not obj.hide_render:
            depsgraph = bpy.context.evaluated_depsgraph_get()
            eval_obj = obj.evaluated_get(depsgraph)
            try:
                mesh = eval_obj.to_mesh()
                all_vertices.extend([obj.matrix_world @ v.co for v in mesh.vertices])
                eval_obj.to_mesh_clear()
            except: continue
    if not all_vertices: return np.zeros(3), np.zeros(3)
    all_vertices = np.array(all_vertices)
    return np.min(all_vertices, axis=0), np.max(all_vertices, axis=0)

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def look_at_rotation(camera_pos, target_pos, up=np.array([0, 0, 1])):
    direction = normalize(np.array(target_pos) - np.array(camera_pos))
    right = normalize(np.cross(direction, up))
    new_up = np.cross(right, direction)
    return np.array([right, new_up, -direction]).T

def rotation_matrix_to_euler(matrix):
    sy = np.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2)
    if sy > 1e-6:
        x, y, z = np.arctan2(matrix[2, 1], matrix[2, 2]), np.arctan2(-matrix[2, 0], sy), np.arctan2(matrix[1, 0], matrix[0, 0])
    else:
        x, y, z = np.arctan2(-matrix[1, 2], matrix[1, 1]), np.arctan2(-matrix[2, 0], sy), 0
    return np.array([x, y, z])

def setup_camera(mesh_objects):
    camera = bpy.data.objects.get("Camera")
    if camera is None:
        bpy.ops.object.camera_add(); camera = bpy.context.active_object; camera.name = "Camera"
    min_xyz, max_xyz = get_meshes_bounds(mesh_objects)
    center = (min_xyz + max_xyz) / 2
    max_extent = float(np.max(max_xyz - min_xyz))
    if max_extent < 1e-6: max_extent = 1.0
    cam_dist = max(3.5, 1.3 * max_extent)
    camera.location = center + np.array([cam_dist * 0.6, -cam_dist * 1.2, cam_dist * 0.6])
    camera.rotation_euler = rotation_matrix_to_euler(look_at_rotation(camera.location, center))
    bpy.context.scene.camera = camera
    if max_extent > 0:
        camera.data.clip_end = cam_dist * 5
        camera.data.clip_start = 0.1
    return camera


def create_camera_icon(scale=0.3):

    stl_path = Path("./blender/cam_marker.stl")
    
    if not stl_path.exists():
        stl_path = PROJECT_ROOT / "blender/cam_marker.stl"
    
    if not stl_path.exists():
        raise FileNotFoundError(f"Cam Icon not found: {stl_path.resolve()}")

    cam_marker = trimesh.load_mesh(str(stl_path))
    cam_vertices = (cam_marker.vertices / np.abs(cam_marker.vertices).max()) * 1
    cam_vertices[:, 2] *= -1
    cam_vertices[:, 2] += 1
    cam_faces = cam_marker.faces

    mesh_data = bpy.data.meshes.new("CameraIconMesh")
    mesh_data.from_pydata(cam_vertices, [], cam_faces)
    mesh_data.update()

    parent_empty = bpy.data.objects.new("CameraIconParent", None)
    bpy.context.collection.objects.link(parent_empty)

    icon_obj = bpy.data.objects.new("CameraIcon", mesh_data)
    bpy.context.collection.objects.link(icon_obj)
    icon_obj.parent = parent_empty
    
    parent_empty.scale = (scale, scale, scale)

    return parent_empty, [icon_obj]


class Renderer:
    def __init__(self):
        self.all_created_objs = []

    def render_single(self, cam_traj_json_path: str, obj_traj_json_path: str, total_anim_frames: int, grid_subdivisions: int = 15, grid_line_color: tuple = (0.3, 0.3, 0.3)):
        Z_FIGHTING_OFFSET = 1e-5
        self.all_created_objs = []
        CAM_ICON_FORWARD_OFFSET = 0 
        OBJ_ICON_FORWARD_OFFSET = 0.0

        cam_traj_c2ws_raw, cam_color_codes = load_transforms_raw_with_colors(cam_traj_json_path)
        obj_traj_c2ws_raw, obj_color_codes = load_transforms_raw_with_colors(obj_traj_json_path)
        cam_traj_c2ws_norm, obj_traj_c2ws_norm = normalize_scene(cam_traj_c2ws_raw, obj_traj_c2ws_raw)
        
        cam_traj_final = cam_traj_c2ws_norm[:, [0, 2, 1]]; cam_traj_final[:, 2] = -cam_traj_final[:, 2]
        obj_traj_final = obj_traj_c2ws_norm[:, [0, 2, 1]]; obj_traj_final[:, 2] = -obj_traj_final[:, 2]
        
        use_simple_gradient = not cam_color_codes or (cam_color_codes[0] == "0" or cam_color_codes[0] == "#000000")
        if use_simple_gradient:
            cam_gradient_rgb = generate_simple_gradient(len(cam_traj_final), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
            obj_gradient_rgb = generate_simple_gradient(len(obj_traj_final), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        else:
            def generate_full_gradient(color_codes):
                full_gradient_rgb, color_to_palette_map, next_palette_idx = [], {}, 0; i = 0
                while i < len(color_codes):
                    current_color = color_codes[i]
                    if current_color not in color_to_palette_map:
                        palette_idx = next_palette_idx % len(SEGMENT_PALETTES)
                        color_to_palette_map[current_color], next_palette_idx = palette_idx, next_palette_idx + 1
                    palette = SEGMENT_PALETTES[color_to_palette_map[current_color]]
                    j = i
                    while j < len(color_codes) and color_codes[j] == current_color: j += 1
                    segment_gradient = generate_segmented_color_gradient(j - i, palette['gradient_rgb'])
                    full_gradient_rgb.extend(segment_gradient)
                    i = j
                return full_gradient_rgb
            cam_gradient_rgb = generate_full_gradient(cam_color_codes)
            obj_gradient_rgb = generate_full_gradient(obj_color_codes)

        scene = bpy.context.scene
        scene.frame_start = 1
        scene.frame_end = total_anim_frames
        
        cam_icon_parent, cam_icon_parts = create_camera_icon()
        cam_icon_parent.name = "AnimatedCameraIcon"
        self.all_created_objs.extend([cam_icon_parent.name] + [p.name for p in cam_icon_parts])
        
        icon_mat_cam = bpy.data.materials.new("IconMat_Cam"); icon_mat_cam.use_nodes = True
        bsdf_cam = icon_mat_cam.node_tree.nodes.get("Principled BSDF")
        for part in cam_icon_parts: part.data.materials.append(icon_mat_cam)

        cam_path_segments = []
        cam_path_points = cam_traj_final[:, :3, 3]
        for i in range(len(cam_path_points) - 1):
            segment_points = cam_path_points[i : i+2]
            curve_data = bpy.data.curves.new(f'CamPathStepData_{i}', type='CURVE'); curve_data.dimensions = '3D'
            polyline = curve_data.splines.new('POLY'); polyline.points.add(1)
            z_offset = i * Z_FIGHTING_OFFSET
            p0, p1 = segment_points[0], segment_points[1]
            polyline.points[0].co = (p0[0], p0[1], p0[2] + z_offset, 1)
            polyline.points[1].co = (p1[0], p1[1], p1[2] + z_offset, 1)
            
            curve_obj = bpy.data.objects.new(f"CamPathStepCurve_{i}", curve_obj_data := curve_data)
            curve_obj.data.bevel_depth = 0.016
            
            color = cam_gradient_rgb[i]
            mat = bpy.data.materials.new(name=f"CamPathMat_{i}"); mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF"); bsdf.inputs["Base Color"].default_value = (*color, 1.0)
            curve_obj.data.materials.append(mat)
            bpy.context.collection.objects.link(curve_obj); self.all_created_objs.append(curve_obj.name)
            cam_path_segments.append(curve_obj)

        bpy.ops.mesh.primitive_cube_add(size=0.25, location=(0,0,0))
        obj_icon = bpy.context.active_object; obj_icon.name = "AnimatedObjectIcon"
        self.all_created_objs.append(obj_icon.name)
        icon_mat_obj = bpy.data.materials.new("IconMat_Obj"); icon_mat_obj.use_nodes = True
        bsdf_obj = icon_mat_obj.node_tree.nodes.get("Principled BSDF")
        obj_icon.data.materials.append(icon_mat_obj)

        obj_path_segments = []
        obj_path_points = obj_traj_final[:, :3, 3]
        for i in range(len(obj_path_points) - 1):
            segment_points = obj_path_points[i : i+2]
            curve_data = bpy.data.curves.new(f'ObjPathStepData_{i}', type='CURVE'); curve_data.dimensions = '3D'
            polyline = curve_data.splines.new('POLY'); polyline.points.add(1)
            z_offset = i * Z_FIGHTING_OFFSET
            p0, p1 = segment_points[0], segment_points[1]
            polyline.points[0].co = (p0[0], p0[1], p0[2] + z_offset, 1)
            polyline.points[1].co = (p1[0], p1[1], p1[2] + z_offset, 1)
            
            curve_obj = bpy.data.objects.new(f"ObjPathStepCurve_{i}", curve_data)
            curve_obj.data.bevel_depth = 0.016
            
            color = obj_gradient_rgb[i]
            mat = bpy.data.materials.new(name=f"ObjPathMat_{i}"); mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            bsdf.inputs["Base Color"].default_value = (*color, 1.0)
            curve_obj.data.materials.append(mat)
            bpy.context.collection.objects.link(curve_obj); self.all_created_objs.append(curve_obj.name)
            obj_path_segments.append(curve_obj)

        n_cam_points = len(cam_traj_final)
        n_obj_points = len(obj_traj_final)

        for frame_num in range(scene.frame_start, scene.frame_end + 1):
            scene.frame_set(frame_num)

            cam_progress = (frame_num - scene.frame_start) / (scene.frame_end - scene.frame_start) if (scene.frame_end - scene.frame_start) > 0 else 0
            cam_data_idx_float = cam_progress * (n_cam_points - 1)
            cam_idx_floor = int(np.floor(cam_data_idx_float))
            cam_idx_ceil = min(cam_idx_floor + 1, n_cam_points - 1)
            cam_interp_factor = cam_data_idx_float - cam_idx_floor
            
            loc_floor_cam = cam_traj_final[cam_idx_floor][:3, 3]
            loc_ceil_cam = cam_traj_final[cam_idx_ceil][:3, 3]
            interpolated_loc_cam = (1 - cam_interp_factor) * loc_floor_cam + cam_interp_factor * loc_ceil_cam
            
            safe_cam_idx = max(0, min(n_cam_points - 1, int(round(cam_data_idx_float))))
            transform_matrix_cam = cam_traj_final[safe_cam_idx]
            rotation_matrix_cam = transform_matrix_cam[:3, :3]
            forward_vector_cam = rotation_matrix_cam @ np.array([0.0, 0.0, -1.0])
            position_offset_cam = forward_vector_cam * CAM_ICON_FORWARD_OFFSET
            
            cam_icon_parent.location = interpolated_loc_cam + position_offset_cam
            cam_icon_parent.rotation_euler = rotation_matrix_to_euler(rotation_matrix_cam)
            bsdf_cam.inputs["Base Color"].default_value = (*cam_gradient_rgb[safe_cam_idx], 1.0)
            
            cam_icon_parent.keyframe_insert(data_path="location", frame=frame_num)
            cam_icon_parent.keyframe_insert(data_path="rotation_euler", frame=frame_num)
            bsdf_cam.inputs["Base Color"].keyframe_insert(data_path="default_value", frame=frame_num)

            for seg_idx, segment_obj in enumerate(cam_path_segments):
                if cam_data_idx_float < seg_idx: bevel_val = 0.0
                elif cam_data_idx_float >= seg_idx + 1: bevel_val = 1.0
                else: bevel_val = cam_data_idx_float - seg_idx
                segment_obj.data.bevel_factor_end = bevel_val
                segment_obj.data.keyframe_insert(data_path="bevel_factor_end", frame=frame_num)

            obj_progress = (frame_num - scene.frame_start) / (scene.frame_end - scene.frame_start) if (scene.frame_end - scene.frame_start) > 0 else 0
            obj_data_idx_float = obj_progress * (n_obj_points - 1)
            obj_idx_floor = int(np.floor(obj_data_idx_float))
            obj_idx_ceil = min(obj_idx_floor + 1, n_obj_points - 1)
            obj_interp_factor = obj_data_idx_float - obj_idx_floor
            
            loc_floor_obj = obj_traj_final[obj_idx_floor][:3, 3]
            loc_ceil_obj = obj_traj_final[obj_idx_ceil][:3, 3]
            interpolated_loc_obj = (1 - obj_interp_factor) * loc_floor_obj + obj_interp_factor * loc_ceil_obj
            
            safe_obj_idx = max(0, min(n_obj_points - 1, int(round(obj_data_idx_float))))
            transform_matrix_obj = obj_traj_final[safe_obj_idx]
            rotation_matrix_obj = transform_matrix_obj[:3, :3]
            forward_vector_obj = rotation_matrix_obj @ np.array([0.0, 0.0, -1.0])
            position_offset_obj = forward_vector_obj * OBJ_ICON_FORWARD_OFFSET
            
            obj_icon.location = interpolated_loc_obj + position_offset_obj
            obj_icon.rotation_euler = rotation_matrix_to_euler(rotation_matrix_obj)
            bsdf_obj.inputs["Base Color"].default_value = (*obj_gradient_rgb[safe_obj_idx], 1.0)
            
            obj_icon.keyframe_insert(data_path="location", frame=frame_num)
            obj_icon.keyframe_insert(data_path="rotation_euler", frame=frame_num)
            bsdf_obj.inputs["Base Color"].keyframe_insert(data_path="default_value", frame=frame_num)
            
            for seg_idx, segment_obj in enumerate(obj_path_segments):
                if obj_data_idx_float < seg_idx: bevel_val = 0.0
                elif obj_data_idx_float >= seg_idx + 1: bevel_val = 1.0
                else: bevel_val = obj_data_idx_float - seg_idx
                segment_obj.data.bevel_factor_end = bevel_val
                segment_obj.data.keyframe_insert(data_path="bevel_factor_end", frame=frame_num)

        renderable_objects = [bpy.data.objects[n] for n in self.all_created_objs if n in bpy.data.objects]
        bpy.context.view_layer.update()
        setup_camera(renderable_objects)
        
        min_xyz, max_xyz = get_meshes_bounds(renderable_objects)
        if (max_xyz - min_xyz).sum() > 1e-5:
            center_xy = ((min_xyz[0] + max_xyz[0]) * 0.5, (min_xyz[1] + max_xyz[1]) * 0.5)
            plane_z = min_xyz[2] - 0.05 
            grid_size = np.max(max_xyz - min_xyz) * 2.0
            
            bpy.ops.mesh.primitive_grid_add(
                size=max(grid_size, 10.0), x_subdivisions=grid_subdivisions, y_subdivisions=grid_subdivisions, 
                location=(center_xy[0], center_xy[1], plane_z))
            plane = bpy.context.active_object; plane.name = "GroundPlane"
            
            grid_mat = bpy.data.materials.new(name="GridMaterial"); grid_mat.use_nodes = True
            bsdf_grid = grid_mat.node_tree.nodes.get("Principled BSDF")
            if bsdf_grid:
                bsdf_grid.inputs['Base Color'].default_value = (*grid_line_color, 1.0)
                bsdf_grid.inputs['Roughness'].default_value = 1.0
            plane.data.materials.append(grid_mat)
            wf_modifier = plane.modifiers.new(name="Wireframe", type='WIREFRAME')
            wf_modifier.thickness = 0.015; wf_modifier.use_replace = True 
        

        temp_dir = Path(tempfile.gettempdir()) / "blender_render_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = f"render_{Path(cam_traj_json_path).stem}"
        temp_out_prefix = temp_dir / base_name
        
        for f in temp_dir.glob(f"{base_name}*.mp4"):
            try: f.unlink()
            except: pass
            

        bpy.context.scene.render.filepath = str(temp_out_prefix)

        # Render başlat
        bpy.ops.render.render(animation=True)

        generated_files = list(temp_dir.glob(f"{base_name}*.mp4"))
        
        if not generated_files:
            raise FileNotFoundError(f"Error: {temp_out_prefix}*.mp4")
            
        final_video_file = sorted(generated_files, key=lambda x: x.stat().st_mtime)[-1]
        
        return final_video_file


if __name__ == "__main__":

    TRAJ_INPUT_DIR = Path("./vis/example_traj/")  
    OUT_DIR = Path("./results")           
    ANIMATION_TOTAL_FRAMES = 80
    
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 64
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.render.film_transparent = False
    
    bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
    bpy.context.scene.render.ffmpeg.format = "MPEG4"
    bpy.context.scene.render.ffmpeg.codec = "H264"
    bpy.context.scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
    
    bpy.context.scene.render.resolution_x = 720
    bpy.context.scene.render.resolution_y = 480
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.fps = 16

    if not TRAJ_INPUT_DIR.exists():
        raise FileNotFoundError(f"Directory not found: {TRAJ_INPUT_DIR}")
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def extract_cam_id(path):
        name = path.name
        m = re.match(r'^(\d+)\.(json|jsn)$', name, flags=re.IGNORECASE)
        if m: return int(m.group(1))
        m2 = re.search(r'(\d+)', name)
        if m2: return int(m2.group(1))
        return -1

    trajectory_pairs = []

    for case_dir in sorted([p for p in TRAJ_INPUT_DIR.iterdir() if p.is_dir()]):
        bbox_path = case_dir / "bbox.json"
        cam_dir = case_dir / "cam_json"

        if not bbox_path.exists() or not cam_dir.exists():
            continue

        cam_files = [p for p in cam_dir.iterdir() if p.is_file() and p.suffix.lower() in {".json", ".jsn"}]
        cam_files_with_id = [(p, extract_cam_id(p)) for p in cam_files]
        cam_files_with_id.sort(key=lambda x: (x[1] < 0, x[1] if x[1] >= 0 else x[0].name))

        for cam_path, cam_id in cam_files_with_id:
            if cam_id >= 0: prefix = f"{case_dir.name}_cam{cam_id:04d}"
            else: prefix = f"{case_dir.name}_{cam_path.stem}"
            trajectory_pairs.append((cam_path, bbox_path, prefix))

    if not trajectory_pairs:
        print("[ERROR].")
        sys.exit(1)

    renderer = Renderer()
    
    for cam_path, bbox_path, prefix in trajectory_pairs:
        reset_scene()
        
        try:
            final_path = OUT_DIR / f"{prefix}_animation.mp4"

            rendered_video = renderer.render_single(
                cam_traj_json_path=str(cam_path), 
                obj_traj_json_path=str(bbox_path),
                total_anim_frames=ANIMATION_TOTAL_FRAMES,
                grid_subdivisions=15
            )
            

            shutil.move(str(rendered_video), str(final_path))
            print(f"[SUCCESS] Created: {final_path}")

        except Exception as e:
            print(f"[ERROR] : {prefix} ({e})")
            continue

