import bpy
from .scene import setup_scene
from .floor import plot_floor
from .camera import Camera
from .sampler import get_frameidx
from .meshes_w_cams import MeshesWithCameras
from .materials import MESH_TO_MATERIAL
import numpy as np
# YENİ YARDIMCI FONKSİYON: Hex renk kodunu Blender'ın anladığı formata çevirir.
def _hex_to_rgb_float(hex_str):
    """'#RRGGBB' -> (r, g, b) [0-1 aralığında float]"""
    hex_str = hex_str.lstrip('#')
    if len(hex_str) != 6:
        return (1.0, 0.0, 1.0)  # Hata durumunda magenta
    try:
        return tuple(int(hex_str[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        return (1.0, 0.0, 1.0)

# ANA RENDER FONKSİYONU (GÜNCELLENDİ)
def render(
    traj,
    vertices,
    faces,
    colors=None,
    cam_segments=None,
    char_segments=None,
    # ----------------- #
    mode="image",
    exact_frame=None,
    num=8,
    canonicalize=True,
    denoising=True,
    oldrender=True,
    res="high",
    mesh_color="Blues",
    cam_color="Purples",
    init=False,
):

    #vertices = np.array(vertices) * 3
    assert mode in ["image", "video", "video_accumulate"]
    print(f"Render modu: {mode}")
    if not init:
        setup_scene(res=res, denoising=denoising, oldrender=oldrender)

    materials_dict = {}
    if colors:
        print(f"Toplam {len(set(colors))} benzersiz renk kodu için materyal oluşturuluyor...")
        unique_colors = set(colors)
        for color_hex in unique_colors:
            mat_name = f"CamMat_{color_hex.lstrip('#')}"
            if mat_name in bpy.data.materials:
                mat = bpy.data.materials[mat_name]
            else:
                mat = bpy.data.materials.new(name=mat_name)
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes.get('Principled BSDF')
                if bsdf:
                    rgb_float = _hex_to_rgb_float(color_hex)
                    bsdf.inputs['Base Color'].default_value = (*rgb_float, 1.0)
            materials_dict[color_hex] = mat

    data = MeshesWithCameras(
        cams=traj,
        vertices=vertices,
        faces=faces,
        mode=mode,
        canonicalize=canonicalize,
        always_on_floor=False,
        cam_color=cam_color,
        mesh_color=mesh_color,
    )

    if not init:
        # === DEĞİŞİKLİK BURADA ===
        # Zemin oluşturma ana betik tarafından yapıldığı için bu satır devre dışı bırakıldı.
        # plot_floor(data.data)
        pass

    Camera(first_root=data.get_root(0), mode=mode, is_mesh=True)

    nframes = len(data)
    frames_to_keep = num if mode == "image" else nframes
    frame_indices = get_frameidx(
        mode=mode, nframes=nframes, exact_frame=exact_frame, frames_to_keep=frames_to_keep
    )
    frame_to_keep_indices = list(
        get_frameidx(
            mode=mode, nframes=nframes, exact_frame=exact_frame, frames_to_keep=num
        )
    )
    nframes_to_render = len(frame_indices)

    imported_obj_names = []
    for index, frame_index in enumerate(frame_indices):
        
        if colors and materials_dict:
            current_color_hex = colors[frame_index]
            cam_mat = materials_dict[current_color_hex]
        else:
            if mode == "image":
                frac = index / (nframes_to_render - 1) if nframes_to_render > 1 else 1.0
                cam_mat = data.get_cam_sequence_mat(frac)
            else:
                cam_mat = data.get_cam_sequence_mat(1.0)

        if char_segments is not None:
            mesh_mat = MESH_TO_MATERIAL[char_segments[frame_index]]
        else:
            if mode == "image":
                frac = index / (nframes_to_render - 1) if nframes_to_render > 1 else 1.0
                mesh_mat = data.get_mesh_sequence_mat(frac)
            else:
                mesh_mat = data.get_mesh_sequence_mat(1.0)

        keep_frame = frame_index in frame_to_keep_indices
        _, cam_name = data.load_in_blender(
            frame_index, cam_mat, mesh_mat, mode, keep_frame
        )
        curve_name = data.show_cams(frame_index + 1, mode)

        imported_obj_names.extend([cam_name, curve_name])

    return imported_obj_names