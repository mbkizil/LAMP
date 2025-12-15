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
from mathutils import Vector

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
if bpy.data.filepath:
    sys.path.append(os.path.dirname(bpy.data.filepath))


try:
    from blender.src.render import render
    from blender.src.tools import delete_objs
except ImportError:
    print("[ERROR] 'blender.src.render' and 'blender.src.tools' not found.")
    def render(**kwargs):
        print("[Warning]")
        return []
    def delete_objs(*args, **kwargs):
        print("[Warning]")


try:
    from PIL import Image
except Exception:
    print("[FATAL] Pillow (PIL) library not found.")
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
    bg_node = nt.nodes.new(type='ShaderNodeBackground')
    bg_node.inputs['Color'].default_value = (0.6, 0.8, 1.0, 1.0)
    output_node = nt.nodes.new(type='ShaderNodeOutputWorld')
    nt.links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])
    bpy.ops.object.camera_add(location=(0, 0, 0))
    bpy.context.active_object.name = "Camera"
    bpy.ops.object.light_add(type='SUN', align='WORLD', location=(0, 0, 10))
    light = bpy.context.active_object
    light.name = "Sun"
    light.data.energy = 4
    light.rotation_euler[0] = np.deg2rad(45)
    light.rotation_euler[1] = np.deg2rad(-30)

def create_simple_arrow(name: str, size: float):

    points = [
        (size * 0.6, 0, size * 0.84),
        (0, 0, 0),
        (-size * 0.6, 0, size * 0.84)
    ]
    curve_data = bpy.data.curves.new(name, type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.fill_mode = 'FULL'
    curve_data.bevel_depth = size * 0.25
    curve_data.bevel_resolution = 2
    polyline = curve_data.splines.new('POLY')
    polyline.points.add(len(points) - 1)
    for i, coords in enumerate(points):
        polyline.points[i].co = (*coords, 1.0)
        if i == 1:
            polyline.points[i].radius = 0.8
        else:
            polyline.points[i].radius = 0.2
    arrow_obj = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(arrow_obj)
    return arrow_obj


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
    gradient, num_sections = [], len(palette_rgb) - 1
    for i in range(num_steps):
        t_global = i / (num_steps - 1) if num_steps > 1 else 0.0
        section = int(t_global * num_sections)
        if section >= num_sections: section = num_sections - 1
        t_local = (t_global - section / num_sections) * num_sections if num_sections > 0 else 0.0
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
    with open(file, "r", encoding="utf-8") as f: return json.load(f)
def load_transforms_raw_with_colors(transform_p: str):
    data = load_json(transform_p)
    frames = data.get('frames', [])
    if not frames: raise ValueError(f"JSON dosyasında 'frames' listesi bulunamadı: {transform_p}")
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
    cam_c2ws_norm = cam_c2ws.copy(); cam_c2ws_norm[:, :3, 3] = (cam_c2ws[:, :3, 3] - center) / final_scale
    obj_c2ws_norm = obj_c2ws.copy(); obj_c2ws_norm[:, :3, 3] = (obj_c2ws[:, :3, 3] - center) / final_scale
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
    x = np.arctan2(matrix[2, 1], matrix[2, 2])
    y = np.arctan2(-matrix[2, 0], sy)
    z = np.arctan2(matrix[1, 0], matrix[0, 0]) if sy > 1e-6 else 0
    return np.array([x, y, z])
def setup_camera(mesh_objects):
    camera = bpy.data.objects.get("Camera")
    if camera is None:
        bpy.ops.object.camera_add(); camera = bpy.context.active_object; camera.name = "Camera"
    min_xyz, max_xyz = get_meshes_bounds(mesh_objects)
    center = (min_xyz + max_xyz) / 2
    max_extent = float(np.max(max_xyz - min_xyz))
    cam_dist = max(3.5, 1.3 * max_extent)
    camera.location = center + np.array([cam_dist * 0.6, -cam_dist * 1.2, cam_dist * 0.6])
    camera.rotation_euler = rotation_matrix_to_euler(look_at_rotation(camera.location, center))
    bpy.context.scene.camera = camera
    return camera
def apply_overlap_offset(traj, camera_obj):
    DISTANCE_THRESHOLD, OFFSET_STRENGTH = 0.001, 0.0001
    positions = traj[:, :3, 3]; num_points = len(positions)
    clusters, visited_indices = [], [False] * num_points
    for i in range(num_points):
        if visited_indices[i]: continue
        current_cluster = [i]
        for j in range(i + 1, num_points):
            if not visited_indices[j] and np.linalg.norm(positions[i] - positions[j]) < DISTANCE_THRESHOLD:
                current_cluster.append(j)
        if len(current_cluster) > 1:
            clusters.append(current_cluster)
            for idx in current_cluster: visited_indices[idx] = True
    if not clusters: return traj
    cam_pos = np.array(camera_obj.location)
    view_direction = normalize(np.mean(positions, axis=0) - cam_pos)
    right_vector = normalize(np.cross(view_direction, np.array([0.0, 0.0, 1.0])))
    traj_modified = traj.copy()
    for cluster in clusters:
        n = len(cluster)
        for i, idx in enumerate(cluster):
            spread_factor = i - (n - 1) / 2.0
            offset = right_vector * spread_factor * OFFSET_STRENGTH
            traj_modified[idx, :3, 3] += offset
    return traj_modified


class Renderer:
    def __init__(self):
        self.all_created_objs = []

    def render_single(
        self,
        input_dir: str, sample_id: str, max_frames: int, mode: str,
        cam_traj_json_path: str, obj_traj_json_path: str, output_filepath: Path,
        object_icon_size: float = 0.2,
        camera_icon_scale: float = 3.0,  # <-- YENİ PARAMETRE
        grid_subdivisions: int = 20, grid_line_color: tuple = (0.3, 0.3, 0.3),
        simple_gradient_start: tuple = (0.1, 0.9, 0.0), simple_gradient_end: tuple = (0.1, 0.0, 0.9)
    ):
        self.all_created_objs = []
        
        cam_traj_c2ws_raw, cam_color_codes = load_transforms_raw_with_colors(cam_traj_json_path)
        obj_traj_c2ws_raw, obj_color_codes = load_transforms_raw_with_colors(obj_traj_json_path)
        cam_traj_c2ws_norm, obj_c2ws_norm = normalize_scene(cam_traj_c2ws_raw, obj_traj_c2ws_raw)
        cam_traj_final = cam_traj_c2ws_norm[:, [0, 2, 1]]; cam_traj_final[:, 2] = -cam_traj_final[:, 2]
        obj_traj_final = obj_c2ws_norm[:, [0, 2, 1]]; obj_traj_final[:, 2] = -obj_traj_final[:, 2]
        
        nframes_cam, nframes_obj = cam_traj_final.shape[0], obj_traj_final.shape[0]
        use_simple_gradient = not cam_color_codes or (cam_color_codes[0] == "0" or cam_color_codes[0] == "#000000")
        
        def generate_full_gradient(color_codes):
            full_gradient_rgb, color_to_palette_map, next_palette_idx = [], {}, 0
            i, n_frames = 0, len(color_codes)
            while i < n_frames:
                current_color = color_codes[i]
                if current_color not in color_to_palette_map:
                    palette_idx = next_palette_idx % len(SEGMENT_PALETTES); color_to_palette_map[current_color] = palette_idx; next_palette_idx += 1
                palette = SEGMENT_PALETTES[color_to_palette_map[current_color]]
                j = i
                while j < n_frames and color_codes[j] == current_color: j += 1
                segment_gradient = generate_segmented_color_gradient(j - i, palette['gradient_rgb'])
                full_gradient_rgb.extend(segment_gradient)
                i = j
            return full_gradient_rgb
        
        if use_simple_gradient:
            cam_gradient_rgb = generate_simple_gradient(nframes_cam, simple_gradient_start, simple_gradient_end)
            obj_gradient_rgb = generate_simple_gradient(nframes_obj, simple_gradient_start, simple_gradient_end)
        else:
            cam_gradient_rgb = generate_full_gradient(cam_color_codes)
            obj_gradient_rgb = generate_full_gradient(obj_color_codes)

        cam_gradient_hex = [rgb_to_hex(c) for c in cam_gradient_rgb]
        camera_for_offset = setup_camera([])
        cam_traj_final_offset = apply_overlap_offset(cam_traj_final, camera_for_offset)
        
        char_path = Path(input_dir) / "vert_raw" / f"{sample_id}.npy"
        char = np.load(char_path, allow_pickle=True)[()]
        vertices = char["vertices"][..., [0, 2, 1]]; vertices[..., 2] = -vertices[..., 2]
        faces = char["faces"][..., [0, 2, 1]]

        num_cam_icons = min(nframes_cam, max_frames)
        created_by_render = render(
            traj=cam_traj_final_offset, colors=cam_gradient_hex, vertices=vertices, faces=faces,
            denoising=True, oldrender=True, res="low", canonicalize=False,
            exact_frame=0.5, num=num_cam_icons, mode=mode, init=False,
        )
        self.all_created_objs.extend(created_by_render)
     
        for obj_name in created_by_render:
            obj = bpy.data.objects.get(obj_name)
            if obj: 
                obj.visible_shadow = False
                if "Curve" in obj_name and obj.data.materials:
                    mat = obj.data.materials[0]
                    if mat.use_nodes:
                        emis_node = next((node for node in mat.node_tree.nodes if node.type == 'EMISSION'), None)
                        if emis_node:
                            original_color = emis_node.inputs["Color"].default_value
                            mat.node_tree.nodes.clear()
                            out = mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
                            bsdf = mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
                            bsdf.inputs["Base Color"].default_value = original_color
                            mat.node_tree.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        
        if num_cam_icons > 1:
            icon_frame_indices = [int(i * (nframes_cam - 1) / (num_cam_icons - 1)) for i in range(num_cam_icons)]
            for i in range(len(icon_frame_indices) - 1):
                start_frame_idx, end_frame_idx = icon_frame_indices[i], icon_frame_indices[i+1]
                mid_point_idx = (start_frame_idx + end_frame_idx) // 2
                if mid_point_idx + 1 >= nframes_cam: continue
                pos_start_local = Vector(cam_traj_final_offset[mid_point_idx, :3, 3])
                pos_end_local = Vector(cam_traj_final_offset[mid_point_idx + 1, :3, 3])
                direction = pos_end_local - pos_start_local
                if direction.length > 1e-1:
                    arrow_obj = create_simple_arrow(f"CameraPathArrow_{i}", 0.21)
                    self.all_created_objs.append(arrow_obj.name)
                    faktor = 0.50 
                    arrow_obj.location = pos_start_local + (pos_end_local - pos_start_local) * faktor
                    print("(pos_end_local - pos_start_local) * faktor", (pos_end_local - pos_start_local) * faktor)
                    print("pos_start_local", pos_start_local)
                    print("pos_end_local", pos_end_local)
                    arrow_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
                    color = cam_gradient_rgb[start_frame_idx]
                    mat = bpy.data.materials.new(name=f"CamArrowMat_{i}"); mat.use_nodes = True
                    bsdf = mat.node_tree.nodes.get("Principled BSDF"); bsdf.inputs["Base Color"].default_value = (*color, 1.0)
                    arrow_obj.data.materials.append(mat)

        i, seg_idx = 0, 0
        while i < nframes_obj:
            start_idx = i
            current_color_code = obj_color_codes[i] if not use_simple_gradient else "simple"
            while i < nframes_obj and (obj_color_codes[i] == current_color_code if not use_simple_gradient else True): i += 1
            segment_points = obj_traj_final[start_idx:i, :3, 3]
            curve_data = bpy.data.curves.new(f'ObjectPathData_{seg_idx}', type='CURVE'); curve_data.dimensions = '3D'
            polyline = curve_data.splines.new('POLY'); polyline.points.add(len(segment_points) - 1)
            for pt_idx, coords in enumerate(segment_points): polyline.points[pt_idx].co = (*coords, 1)
            curve_obj = bpy.data.objects.new(f"ObjectPathCurve_{seg_idx}", curve_data)
            curve_obj.visible_shadow = False; self.all_created_objs.append(curve_obj.name)
            curve_obj.data.bevel_depth = 0.016
            bpy.context.collection.objects.link(curve_obj)
            color = obj_gradient_rgb[i - 1]
            mat = bpy.data.materials.new(name=f"ObjCurveMat_{seg_idx}"); mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF"); bsdf.inputs["Base Color"].default_value = (1,1,1, 0.6)
            curve_obj.data.materials.append(mat)
            seg_idx += 1

        num_obj_cubes = min(nframes_obj, max_frames)
        if num_obj_cubes > 0:
            for i in range(num_obj_cubes):
                frame_idx = int(i * (nframes_obj - 1) / (num_obj_cubes - 1)) if num_obj_cubes > 1 else 0
                transform_matrix = obj_traj_final[frame_idx]
                location = transform_matrix[:3, 3]
                bpy.ops.mesh.primitive_cube_add(size=object_icon_size, location=location)
                cube = bpy.context.active_object; cube.visible_shadow = False
                cube.rotation_euler = rotation_matrix_to_euler(transform_matrix[:3, :3])
                cube.name = f"ObjectCube_{i}"; self.all_created_objs.append(cube.name)
                color = obj_gradient_rgb[frame_idx]
                mat = bpy.data.materials.new(name=f"ObjCubeMat_{i}"); mat.use_nodes = True
                bsdf = mat.node_tree.nodes.get("Principled BSDF"); bsdf.inputs["Base Color"].default_value = (*color, 1.0)
                cube.data.materials.append(mat)

        if num_obj_cubes > 1:
            icon_frame_indices = [int(i * (nframes_obj - 1) / (num_obj_cubes - 1)) for i in range(num_obj_cubes)]
            for i in range(len(icon_frame_indices) - 1):
                start_frame_idx, end_frame_idx = icon_frame_indices[i], icon_frame_indices[i+1]
                mid_point_idx = (start_frame_idx + end_frame_idx) // 2
                if mid_point_idx + 1 >= nframes_obj: continue
                pos_start_local = Vector(obj_traj_final[mid_point_idx, :3, 3])
                pos_end_local = Vector(obj_traj_final[mid_point_idx + 1, :3, 3])
                direction = pos_end_local - pos_start_local
                if direction.length > 1e-6:
                    arrow_obj = create_simple_arrow(f"ObjectPathArrow_{i}", 0.21)
                    self.all_created_objs.append(arrow_obj.name)
                    faktor = 0.50
                    arrow_obj.location = pos_start_local + (pos_end_local - pos_start_local) * faktor
                    arrow_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
                    color = obj_gradient_rgb[start_frame_idx]
                    mat = bpy.data.materials.new(name=f"ObjArrowMat_{i}"); mat.use_nodes = True
                    bsdf = mat.node_tree.nodes.get("Principled BSDF"); bsdf.inputs["Base Color"].default_value = (*color, 1.0)
                    arrow_obj.data.materials.append(mat)

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
            self.all_created_objs.append(plane.name)
            grid_mat = bpy.data.materials.new(name="GridMaterial"); grid_mat.use_nodes = True
            bsdf_grid = grid_mat.node_tree.nodes.get("Principled BSDF")
            if bsdf_grid:
                bsdf_grid.inputs['Base Color'].default_value = (*grid_line_color, 1.0)
                bsdf_grid.inputs['Roughness'].default_value = 1.0
            plane.data.materials.append(grid_mat)
            wf_modifier = plane.modifiers.new(name="Wireframe", type='WIREFRAME')
            wf_modifier.thickness = 0.015; wf_modifier.use_replace = True
            
        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        bpy.context.scene.render.filepath = str(output_filepath.resolve())
        print(f"[RENDER] Render başlatılıyor. Çıktı: {bpy.context.scene.render.filepath}")
        bpy.ops.render.render(write_still=True)
        if not output_filepath.exists():
            raise RuntimeError(f"Render işlemi tamamlandı ancak çıktı dosyası bulunamadı.")
        return output_filepath



if __name__ == "__main__":
    
    INPUT_DIR = "demo"
    SAMPLE_ID = "2011_F_EuMeT2wBo_00014_00001"


    TRAJ_INPUT_DIR = Path("./vis/example_traj/")
    OUT_DIR = Path("./results")
    

    MAX_VISUALIZED_FRAMES = 5
    MODE = "image"

    CAMERA_ICON_SCALE = 5.0

    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.image_settings.color_mode = "RGBA"
    bpy.context.scene.render.resolution_percentage = 400
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 64
    bpy.context.scene.cycles.device = 'GPU'

    if not TRAJ_INPUT_DIR.exists():
        raise FileNotFoundError(f"Belirtilen trajektör klasörü bulunamadı: {TRAJ_INPUT_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import re

    def extract_cam_id(path):

        name = path.name
        m = re.match(r'^(\d+)\.(json|jsn)$', name, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
        m2 = re.search(r'(\d+)', name)
        if m2:
            return int(m2.group(1))
        return -1

    trajectory_pairs = []

    # Expected:
    # TRAJ_INPUT_DIR/
    #   ├── case_A/
    #   │     ├── bbox.json
    #   │     └── cam_json/
    #   │           ├── 0000.json (veya .jsn)
    #   └── case_B/
    #         ├── bbox.json
    #         └── cam_json/...

    for case_dir in sorted([p for p in TRAJ_INPUT_DIR.iterdir() if p.is_dir()]):
        bbox_path = case_dir / "bbox.json"
        cam_dir = case_dir / "cam_json"

        if not bbox_path.exists():
            print(f"  -> [WARNING] '{case_dir.name}' bbox.json not found.")
            continue
        if not cam_dir.exists() or not cam_dir.is_dir():
            print(f"  -> [WARNING] '{case_dir.name}' cam_json/ directory not found.")
            continue

        cam_files = [p for p in cam_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in {".json", ".jsn"}]

        if not cam_files:
            print(f"  -> [WARNING] '{case_dir.name}/cam_json' trajectory file not found.")
            continue

        cam_files_with_id = [(p, extract_cam_id(p)) for p in cam_files]
        cam_files_with_id.sort(key=lambda x: (x[1] < 0, x[1] if x[1] >= 0 else x[0].name))

        print(f"  -> Case: {case_dir.name} | {len(cam_files_with_id)} cam found.")
        for cam_path, cam_id in cam_files_with_id:
            if cam_id >= 0:
                prefix = f"{case_dir.name}_cam{cam_id:04d}"
            else:
                prefix = f"{case_dir.name}_{cam_path.stem}"

            trajectory_pairs.append((cam_path, bbox_path, prefix))
            print(f"     - Match: {prefix}  (cam: {cam_path.name}, bbox: bbox.json)")

    if not trajectory_pairs:
        print("[ERROR] No (cam, bbox) .")
        sys.exit(1)

    renderer = Renderer()
    index = 0
    for cam_path, bbox_path, prefix in trajectory_pairs:
        index += 1
        if index < 0:
            continue
        reset_scene()

        try:
            start_color, end_color = hex_to_rgb("#ff0000"), hex_to_rgb("#0000ff")
            out_dir_idx = OUT_DIR / prefix
            out_dir_idx.mkdir(parents=True, exist_ok=True)
            render_path = out_dir_idx / f"{prefix}.png"

            rendered_png = renderer.render_single(
                input_dir=INPUT_DIR,
                sample_id=SAMPLE_ID,
                max_frames=MAX_VISUALIZED_FRAMES,
                mode=MODE,
                cam_traj_json_path=str(cam_path),
                obj_traj_json_path=str(bbox_path),
                output_filepath=render_path,
                simple_gradient_start=start_color,
                simple_gradient_end=end_color,
                object_icon_size=0.3,
                grid_subdivisions=15,
                camera_icon_scale=CAMERA_ICON_SCALE,
            )

        except Exception as e:
            print(f"\n[error] '{prefix}'  {e}")
            logger.exception(f"Render error ({prefix}):")