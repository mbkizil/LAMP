#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import math
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
from PIL import Image


try:
    from scipy.spatial.transform import Rotation as R
    from scipy.interpolate import interp1d
    from scipy.spatial.transform import Slerp
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[WARNING] Scipy not found. Interpolation will be 'linear'.")

logger = logging.getLogger(__name__)


class BBoxProjectionPipeline:
    # --- CONSTANTS AND SETUP ---
    COORD_SCALE = 5.0
    DEFAULT_CAMERA_Z_OFFSET = 0.0

    # Interpolation targets
    INITIAL_FRAMES_ESTIMATE = 21
    TARGET_FRAMES = 81

    # Video Background Colors
    VIDEO_BACKGROUND_3D = (255, 255, 255)
    VIDEO_BACKGROUND_2D = (255, 255, 255)
    VIDEO_BACKGROUND_INVERTED = (0, 0, 0)
    VIDEO_BACKGROUND_INVERTED_FF = (255, 255, 255)
    VIDEO_BACKGROUND_GRID = (255, 255, 255)

    # Background Grid Configuration
    GRID_COLOR = (200, 200, 200)
    GRID_THICKNESS = 1

    # 2D BBox Configuration
    BBOX_2D_OUTLINE_COLOR = (30, 30, 30)
    BBOX_2D_THICKNESS = 1
    BBOX_2D_EXPAND_PIXELS = 20

    # Inverted Video BBox Configuration
    BBOX_INVERTED_COLOR = (0,0, 0)
    BBOX_INVERTED_EXPAND_PIXELS = 45

    def __init__(self, width=720, height=480, fov_y=60):
        self.width = width
        self.height = height
        self.fov_y = fov_y

        self.fx = self.fy = (height / 2) / math.tan(math.radians(fov_y / 2))
        self.cx = width / 2
        self.cy = height / 2

        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])

        print("Camera Intrinsics:")
        print(f"  Image: {width}x{height}, FOV_Y: {fov_y}°")
        print(f"  fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.1f}, cy={self.cy:.1f}")

    def _reflect_y_left(self) -> np.ndarray:
        S = np.diag([1.0, -1.0, 1.0])
        A = np.eye(4)
        A[:3, :3] = S
        return A

    def _reflect_z_left(self) -> np.ndarray:
        S = np.diag([1.0, 1.0, -1.0])
        A = np.eye(4)
        A[:3, :3] = S
        return A

    def _reflect_x_left(self) -> np.ndarray:
        S = np.diag([-1.0, 1.0, 1.0])
        A = np.eye(4)
        A[:3, :3] = S
        return A

    def _axis_change_left(self) -> np.ndarray:
        """4x4 left-multiplication matrix A for (x, y, z) -> (x, z, -y) transformation."""
        P = np.array([[1, 0, 0],
                      [0, -1, 0],
                      [0, 0, -1]], dtype=float)
        A = np.eye(4, dtype=float)
        A[:3, :3] = P
        return A

    def load_trajectories_from_json(self, json_path: str) -> List[np.ndarray]:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        frames = data.get('frames', [])
        if not frames:
            raise ValueError(f"no frames in {json_path}")

        A = self._axis_change_left()

        matrices = []
        for item in frames:
            M = np.array(item['transform_matrix'], dtype=float)
            M = A @ M
            matrices.append(M)
        return matrices

    def _interpolate_matrices(self, matrices: List[np.ndarray], target_frames: int) -> List[np.ndarray]:
        n = len(matrices)
        if n == 0: return []
        if n == target_frames: return matrices
        if n == 1: return [matrices[0] for _ in range(target_frames)]

        if not SCIPY_AVAILABLE:
            print("  [WARNING] Scipy not found -> linear interpolation.")
            interp = []
            orig = np.linspace(0, 1, n)
            tgt = np.linspace(0, 1, target_frames)
            for t in tgt:
                idx_f = np.interp(t, orig, np.arange(n))
                i0 = int(idx_f)
                i1 = min(i0 + 1, n - 1)
                a = idx_f - i0
                interp.append((1 - a) * matrices[i0] + a * matrices[i1])
            return interp

        rots = R.from_matrix(np.array([M[:3, :3] for M in matrices]))
        trans = np.array([M[:3, 3] for M in matrices])
        trans[:, 0] = -trans[:, 0]
        trans[:, 1] = -trans[:, 1]
        trans[:, 2] = -trans[:, 2]
        orig = np.linspace(0, 1, n)
        tgt = np.linspace(0, 1, target_frames)

        interp_trans_fn = interp1d(orig, trans, axis=0, kind='cubic')
        smooth_trans = interp_trans_fn(tgt)
        slerp = Slerp(orig, rots)
        smooth_rots = slerp(tgt)

        out = []
        for Rm, t in zip(smooth_rots, smooth_trans):
            M = np.eye(4)
            M[:3, :3] = Rm.as_matrix()
            M[:3, 3] = t
            out.append(M)
        print("  Using scipy cubic/SLERP interpolation for smooth motion")
        return out

    def _interpolate_matrices_object(self, matrices: List[np.ndarray], target_frames: int) -> List[np.ndarray]:
        n = len(matrices)
        if n == 0: return []
        if n == target_frames: return matrices
        if n == 1: return [matrices[0] for _ in range(target_frames)]

        if not SCIPY_AVAILABLE:
            print("  [WARNING] Scipy not found -> linear interpolation.")
            interp = []
            orig = np.linspace(0, 1, n)
            tgt = np.linspace(0, 1, target_frames)
            for t in tgt:
                idx_f = np.interp(t, orig, np.arange(n))
                i0 = int(idx_f)
                i1 = min(i0 + 1, n - 1)
                a = idx_f - i0
                interp.append((1 - a) * matrices[i0] + a * matrices[i1])
            return interp

        rots = R.from_matrix(np.array([M[:3, :3] for M in matrices]))
        trans = np.array([M[:3, 3] for M in matrices])
        trans[:, 0] = trans[:, 0]
        trans[:, 1] = -trans[:, 1]
        trans[:, 2] = -trans[:, 2]
        orig = np.linspace(0, 1, n)
        tgt = np.linspace(0, 1, target_frames)

        interp_trans_fn = interp1d(orig, trans, axis=0, kind='cubic')
        smooth_trans = interp_trans_fn(tgt)
        slerp = Slerp(orig, rots)
        smooth_rots = slerp(tgt)

        out = []
        for Rm, t in zip(smooth_rots, smooth_trans):
            M = np.eye(4)
            M[:3, :3] = Rm.as_matrix()
            M[:3, 3] = t
            out.append(M)
        print("  Using scipy cubic/SLERP interpolation for smooth motion")
        return out

    def _normalize_scene_lists(self,
                             cam_c2w_list: List[np.ndarray],
                             obj_o2w_list: List[np.ndarray],
                             target_radius: float = 5.0,
                             eps: float = 1e-6,
                             center_mode: str = "scene"
                             ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, float]:

        cam_pos = np.array([M[:3, 3] for M in cam_c2w_list]) if cam_c2w_list else np.zeros((0, 3))
        obj_pos = np.array([M[:3, 3] for M in obj_o2w_list]) if obj_o2w_list else np.zeros((0, 3))

        if cam_pos.size == 0 and obj_pos.size == 0:
            return cam_c2w_list, obj_o2w_list, np.zeros(3), 1.0

        center = np.zeros(3, dtype=float)
        all_positions = np.vstack([p for p in (cam_pos, obj_pos) if p.size])
        centered = all_positions - center
        scale = float(np.max(np.linalg.norm(centered, axis=1))) + eps
        final_scale = scale / float(target_radius)

        if final_scale < eps:
            final_scale = 1.0

        cam_norm, obj_norm = [], []
        for M in cam_c2w_list:
            N = M.copy()
            N[:3, 3] = (N[:3, 3] - center) / final_scale
            cam_norm.append(N)
        for M in obj_o2w_list:
            N = M.copy()
            N[:3, 3] = (N[:3, 3] - center) / final_scale
            obj_norm.append(N)

        return cam_norm, obj_norm, center, final_scale

    def _prepare_trajectories(self, cam_json_path: str, obj_json_path: str, camera_z_offset: float
                              ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        cam_c2w_raw = self.load_trajectories_from_json(cam_json_path)
        obj_o2w_raw = self.load_trajectories_from_json(obj_json_path)

        print(f"Loaded trajectories: cam={len(cam_c2w_raw)} frames, obj={len(obj_o2w_raw)} frames.")
        cam_c2w_raw, obj_o2w_raw, _, _ = self._normalize_scene_lists(cam_c2w_raw, obj_o2w_raw,target_radius=5.0,center_mode="object_xy")
        

        cam_w2c_raw = [np.linalg.inv(M) for M in cam_c2w_raw]
        
        print(f"Interpolating to {self.TARGET_FRAMES} frames...")
        cam_w2c = self._interpolate_matrices(cam_w2c_raw, self.TARGET_FRAMES)
        obj_o2w = self._interpolate_matrices_object(obj_o2w_raw, self.TARGET_FRAMES)
        
        print("  Syncing object orientation to camera world orientation...")
        cam_c2w_interp = [np.linalg.inv(M) for M in cam_w2c]

        obj_o2w_synced = []
        for i in range(self.TARGET_FRAMES):
            obj_matrix = obj_o2w[i].copy()
            obj_o2w_synced.append(obj_matrix)
            
        S = self._reflect_x_left()
        cam_w2c_final = [S @ M for M in cam_w2c]
        obj_o2w_final = [S @ M for M in obj_o2w_synced]
        
        return cam_w2c_final, obj_o2w_final

    def get_object_world_position(self, object_world_matrix: np.ndarray) -> np.ndarray:
        return object_world_matrix[:3, 3]

    def get_xy_plane_square_bbox_2d(self, object_world_pos: np.ndarray, extrinsic_matrix: np.ndarray,
                                 side_length: float = 1.0) -> Optional[Tuple[int, int, int, int]]:
        """
        Defines a square in the XY plane centered on the object's center coordinates,
        projects the square's 4 corners to 2D, and returns the 2D BBox.
        The square is always parallel to the XY plane, unaffected by the object's 3D rotation.
        """
        half_side = side_length / 2.0

        # Define the square's 4 corners in the world space
        # Using the object's center (x,y,z) coordinates
        # The Z coordinate will remain constant, the square will always be in the XY plane
        square_corners_3d = np.array([
            [object_world_pos[0] - half_side, object_world_pos[1] - half_side, object_world_pos[2]],
            [object_world_pos[0] + half_side, object_world_pos[1] - half_side, object_world_pos[2]],
            [object_world_pos[0] - half_side, object_world_pos[1] + half_side, object_world_pos[2]],
            [object_world_pos[0] + half_side, object_world_pos[1] + half_side, object_world_pos[2]]
        ], dtype=float)

        # Project the corners to 2D
        pts2d, in_front, _ = self.project_3d_to_2d(square_corners_3d, extrinsic_matrix)

        # Only take points in front of the camera (visible)
        visible_points = pts2d[in_front]

        if visible_points.shape[0] == 0:
            # All corners are behind the camera
            return None

        # Find the min/max values of the visible 2D points
        x_min = np.min(visible_points[:, 0])
        y_min = np.min(visible_points[:, 1])
        x_max = np.max(visible_points[:, 0])
        y_max = np.max(visible_points[:, 1])

        # Clamp the screen boundaries
        x_min_c = max(0, int(x_min))
        y_min_c = max(0, int(y_min))
        x_max_c = min(self.width - 1, int(x_max))
        y_max_c = min(self.height - 1, int(y_max))

        if x_max_c <= x_min_c or y_max_c <= y_min_c:
            # No valid area left
            return None

        return x_min_c, y_min_c, x_max_c, y_max_c

    def project_3d_to_2d(self, points_3d: np.ndarray, extrinsic_matrix: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pts_h = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
        cam_h = (extrinsic_matrix @ pts_h.T).T
        cam = cam_h[:, :3]

        in_front = cam[:, 2] > 0.01
        pts2d = np.zeros((len(points_3d), 2), dtype=float)

        if np.any(in_front):
            pix_h = (self.K @ cam[in_front].T).T
            pts2d_valid = pix_h[:, :2] / pix_h[:, 2:3]
            pts2d[in_front] = pts2d_valid

        return pts2d, in_front, cam[:, 2]

    def draw_line_simple(self, img, p1_3d, p2_3d, extrinsic_matrix, color, thickness):
        num_segments = 20
        t_vals = np.linspace(0, 1, num_segments + 1)
        
        line_points_3d = np.array([p1_3d + t * (p2_3d - p1_3d) for t in t_vals])
        pts2d, valid, _ = self.project_3d_to_2d(line_points_3d, extrinsic_matrix)
        
        for i in range(len(valid) - 1):
            if valid[i] and valid[i+1]:
                pt1 = tuple(pts2d[i].astype(int))
                pt2 = tuple(pts2d[i+1].astype(int))
                
                if (0 <= pt1[0] < self.width and 0 <= pt1[1] < self.height) or \
                   (0 <= pt2[0] < self.width and 0 <= pt2[1] < self.height):
                    cv2.line(img, pt1, pt2, color, thickness)

    def create_background(self, background_color):
        bgr = tuple(int(c) for c in background_color)
        return np.full((self.height, self.width, 3), bgr, dtype=np.uint8)

    def draw_3d_grid(self, img, extrinsic_matrix, trajectory_bounds: Dict):
            """
            Draws a dynamic floor grid and vertical pillars based on the scene boundaries.
            Grid is placed below the lowest point in the scene.
            """
            if not trajectory_bounds or 'all_min' not in trajectory_bounds:
                return

            scene_min = trajectory_bounds['all_min']
            scene_max = trajectory_bounds['all_max']

            # --- FLOOR GRID ---
            # The lowest point has the highest Y value. Place the grid below this point.
            y_floor = scene_max[1] + 0.2

            padding = 5.0
            x_min, x_max = scene_min[0] - padding, scene_max[0] + padding
            z_min, z_max = scene_min[2] - padding, scene_max[2] + padding

            num_lines = 4
            x_coords = np.linspace(x_min, x_max, num_lines)
            z_coords = np.linspace(z_min, z_max, num_lines)

            # Lines along the Z axis
            for x in x_coords:
                p1 = np.array([x, y_floor, z_min])
                p2 = np.array([x, y_floor, z_max])
                self.draw_line_simple(img, p1, p2, extrinsic_matrix, self.GRID_COLOR, self.GRID_THICKNESS)

            # Lines along the Z axis
            for z in z_coords:
                p1 = np.array([x_min, y_floor, z])
                p2 = np.array([x_max, y_floor, z])
                self.draw_line_simple(img, p1, p2, extrinsic_matrix, self.GRID_COLOR, self.GRID_THICKNESS)

            
            # === VERTICAL LINES (UPDATED SECTION) ===
            y_ceiling = scene_min[1] - 5.0

            # Determine the density of vertical lines (less than the floor grid)
            num_vertical_lines = 2
            vertical_x_coords = np.linspace(x_min, x_max, num_vertical_lines)
            vertical_z_coords = np.linspace(z_min, z_max, num_vertical_lines)

            # Draw vertical lines on the "back" and "front" walls
            for x in vertical_x_coords:
                # Back wall (along z_min)
                p_bottom_back = np.array([x, y_floor, z_min])
                p_top_back = np.array([x, y_ceiling, z_min])
                self.draw_line_simple(img, p_bottom_back, p_top_back, extrinsic_matrix, self.GRID_COLOR, self.GRID_THICKNESS)
                
                # Front wall (along z_max)
                p_bottom_front = np.array([x, y_floor, z_max])
                p_top_front = np.array([x, y_ceiling, z_max])
                self.draw_line_simple(img, p_bottom_front, p_top_front, extrinsic_matrix, self.GRID_COLOR, self.GRID_THICKNESS)

            # Draw vertical lines on the "left" and "right" walls
            # Don't redraw corners, skip the first and last elements ([1:-1])
            for z in vertical_z_coords[1:-1]:
                # Left wall (along x_min)
                p_bottom_left = np.array([x_min, y_floor, z])
                p_top_left = np.array([x_min, y_ceiling, z])
                self.draw_line_simple(img, p_bottom_left, p_top_left, extrinsic_matrix, self.GRID_COLOR, self.GRID_THICKNESS)

                # Right wall (along x_max)
                p_bottom_right = np.array([x_max, y_floor, z])
                p_top_right = np.array([x_max, y_ceiling, z])
                self.draw_line_simple(img, p_bottom_right, p_top_right, extrinsic_matrix, self.GRID_COLOR, self.GRID_THICKNESS)

    def draw_2d_bbox(self, img, bbox_2d, color=(0, 0, 0), thickness=2, fill=False, expand_pixels=0):
        if bbox_2d is None: return
        x_min, y_min, x_max, y_max = bbox_2d
        if expand_pixels > 0:
            x_min = max(0, x_min - expand_pixels)
            y_min = max(0, y_min - expand_pixels)
            x_max = min(img.shape[1], x_max + expand_pixels)
            y_max = min(img.shape[0], y_max + expand_pixels)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness if not fill else -1)

    def _setup_video_writers(self, output_path: Path, fps: int, flags: Dict[str, bool]) -> Dict[str, cv2.VideoWriter]:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writers = {'3d': cv2.VideoWriter(str(output_path), fourcc, fps, (self.width, self.height))}

        if flags.get('create_2d_video'):
            path_2d = str(output_path.with_name(f"{output_path.stem}_2d_bbox.mp4"))
            writers['2d'] = cv2.VideoWriter(path_2d, fourcc, fps, (self.width, self.height))
        if flags.get('create_inverted_video'):
            path_inv = str(output_path.with_name(f"{output_path.stem}_2d_bbox_inverted.mp4"))
            writers['inverted'] = cv2.VideoWriter(path_inv, fourcc, fps, (self.width, self.height))
            path_inv_ff = str(output_path.with_name(f"{output_path.stem}_2d_bbox_inverted_ff.mp4"))
            writers['inverted_ff'] = cv2.VideoWriter(path_inv_ff, fourcc, fps, (self.width, self.height))
        if flags.get('create_grid_only_video'):
            path_grid = str(output_path.with_name(f"{output_path.stem}_grid_only.mp4"))
            writers['grid'] = cv2.VideoWriter(path_grid, fourcc, fps, (self.width, self.height))
        return writers

    def _release_video_writers(self, writers: Dict[str, cv2.VideoWriter]):
        for w in writers.values():
            if w: w.release()
        print("\nAll video writers released.")

    def _render_single_frame(self, frame_idx: int, extrinsic: np.ndarray, object_matrix: np.ndarray,
                         world_bbox_size: float, flags: Dict[str, bool],
                         trajectory_bounds: Dict) -> Dict[str, np.ndarray]:
        img_3d = self.create_background(self.VIDEO_BACKGROUND_3D)
        img_grid = self.create_background(self.VIDEO_BACKGROUND_GRID)
        img_2d = self.create_background(self.VIDEO_BACKGROUND_2D) if flags.get('create_2d_video') else None
        img_inverted = self.create_background(self.VIDEO_BACKGROUND_INVERTED) if flags.get('create_inverted_video') else None
        
        img_inverted_ff = None
        if flags.get('create_inverted_video'):
            bg_color = self.VIDEO_BACKGROUND_INVERTED if frame_idx == 0 else self.VIDEO_BACKGROUND_INVERTED_FF
            img_inverted_ff = self.create_background(bg_color)
        
        self.draw_3d_grid(img_3d, extrinsic, trajectory_bounds)
        if flags.get('create_grid_only_video'):
            self.draw_3d_grid(img_grid, extrinsic, trajectory_bounds)

        object_world_pos = self.get_object_world_position(object_matrix)
        bbox2d = self.get_xy_plane_square_bbox_2d(object_world_pos, extrinsic, side_length=world_bbox_size)

        if bbox2d:
            self.draw_2d_bbox(img_3d, bbox2d, self.BBOX_2D_OUTLINE_COLOR, self.BBOX_2D_THICKNESS, expand_pixels=self.BBOX_2D_EXPAND_PIXELS)
            if img_2d is not None:
                self.draw_2d_bbox(img_2d, bbox2d, self.BBOX_2D_OUTLINE_COLOR, self.BBOX_2D_THICKNESS, expand_pixels=self.BBOX_2D_EXPAND_PIXELS, fill=True)
            if img_inverted is not None:
                self.draw_2d_bbox(img_inverted, bbox2d, self.BBOX_INVERTED_COLOR, fill=True, expand_pixels=self.BBOX_INVERTED_EXPAND_PIXELS)
            if img_inverted_ff is not None:
                self.draw_2d_bbox(img_inverted_ff, bbox2d, self.BBOX_INVERTED_COLOR, fill=True, expand_pixels=self.BBOX_INVERTED_EXPAND_PIXELS)

        images = {'3d': img_3d}
        if flags.get('create_grid_only_video'): images['grid'] = img_grid
        if img_2d is not None: images['2d'] = img_2d
        if img_inverted is not None: images['inverted'] = img_inverted
        if img_inverted_ff is not None: images['inverted_ff'] = img_inverted_ff
        return images

    def _calculate_trajectory_bounds(self, cam_w2c_list: List[np.ndarray], obj_o2w_list: List[np.ndarray]) -> Dict:
        """Calculates the min/max bounds of the camera and object trajectories, and the overall scene bounds."""
        cam_c2w_list = [np.linalg.inv(m) for m in cam_w2c_list]
        cam_positions = np.array([m[:3, 3] for m in cam_c2w_list])
        obj_positions = np.array([m[:3, 3] for m in obj_o2w_list])

        if cam_positions.size == 0 or obj_positions.size == 0:
            return {}

        bounds = {
            'cam_min': np.min(cam_positions, axis=0),
            'cam_max': np.max(cam_positions, axis=0),
            'obj_min': np.min(obj_positions, axis=0),
            'obj_max': np.max(obj_positions, axis=0)
        }
        
        bounds['all_min'] = np.minimum(bounds['cam_min'], bounds['obj_min'])
        bounds['all_max'] = np.maximum(bounds['cam_max'], bounds['obj_max'])
        
        return bounds

    def create_video(self, output_path: str, cam_json_path: str, obj_json_path: str,
                     base_bbox_size: float, fps: int, camera_z_offset: float,
                     create_2d_video: bool, create_inverted_video: bool, create_grid_only_video: bool):
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        cam_extrinsics, obj_world = self._prepare_trajectories(cam_json_path, obj_json_path, camera_z_offset)

        trajectory_bounds = self._calculate_trajectory_bounds(cam_extrinsics, obj_world)
        if trajectory_bounds:
            print("--- Trajectory Bounds ---")
            obj_min, obj_max = trajectory_bounds['obj_min'], trajectory_bounds['obj_max']
            cam_min, cam_max = trajectory_bounds['cam_min'], trajectory_bounds['cam_max']
            all_min, all_max = trajectory_bounds['all_min'], trajectory_bounds['all_max']
            
            print(f"  Object Min (x,y,z): ({obj_min[0]:.2f}, {obj_min[1]:.2f}, {obj_min[2]:.2f})")
            print(f"  Object Max (x,y,z): ({obj_max[0]:.2f}, {obj_max[1]:.2f}, {obj_max[2]:.2f})")
            print(f"  Camera Min (x,y,z): ({cam_min[0]:.2f}, {cam_min[1]:.2f}, {cam_min[2]:.2f})")
            print(f"  Camera Max (x,y,z): ({cam_max[0]:.2f}, {cam_max[1]:.2f}, {cam_max[2]:.2f})")
            print(f"  ALL SCENE Min (x,y,z): ({all_min[0]:.2f}, {all_min[1]:.2f}, {all_min[2]:.2f})")
            print(f"  ALL SCENE Max (x,y,z): ({all_max[0]:.2f}, {all_max[1]:.2f}, {all_max[2]:.2f})")
            print("--------------------------")

        flags = {
            'create_2d_video': create_2d_video,
            'create_inverted_video': create_inverted_video,
            'create_grid_only_video': create_grid_only_video
        }
        writers = self._setup_video_writers(out, fps, flags)

        print("\nProcessing frames...")
        for i in range(self.TARGET_FRAMES):
            if i % 10 == 0 or i < 5:
                print(f"  Frame {i+1}/{self.TARGET_FRAMES}")
            
            imgs = self._render_single_frame(
                i, cam_extrinsics[i], obj_world[i], base_bbox_size, flags,
                trajectory_bounds=trajectory_bounds
            )
            for name, img in imgs.items():
                if name in writers:
                    writers[name].write(img)

        self._release_video_writers(writers)

        print("Video generation complete. Outputs:")
        print(f"  3D Video: {out}")
        if create_2d_video:
            print(f"  2D BBox Video: {out.with_name(f'{out.stem}_2d_bbox.mp4')}")
        if create_inverted_video:
            print(f"  Inverted Video: {out.with_name(f'{out.stem}_2d_bbox_inverted.mp4')}")
            print(f"  Inverted FF Video: {out.with_name(f'{out.stem}_2d_bbox_inverted_ff.mp4')}")
        if create_grid_only_video:
            print(f"  Grid Only Video: {out.with_name(f'{out.stem}_grid_only.mp4')}")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent
    RENDER_FPS = 24
    RESOLUTION_X = 720
    RESOLUTION_Y = 480
    CAMERA_FOV_Y = 55
    CAMERA_Z_OFFSET = -0.0
    BASE_BBOX_SIZE = 0.3

    pipeline = BBoxProjectionPipeline(width=RESOLUTION_X, height=RESOLUTION_Y, fov_y=CAMERA_FOV_Y)

    def run_default_mode():
        print("Running in default mode (reading lists)...")
        CAM_LIST_TXT = Path("./vis/et/split_relative_test2.txt")
        OBJ_LIST_TXT = Path("./vis/et/split_relative_test_bbox2.txt")
        OUT_DIR = Path("./split_relative_test_video_opencv_gpt2")

        try:
            cam_json_lines = [ln.strip() for ln in CAM_LIST_TXT.read_text(encoding="utf-8").splitlines() if ln.strip()]
            obj_json_lines = [ln.strip() for ln in OBJ_LIST_TXT.read_text(encoding="utf-8").splitlines() if ln.strip()]
        except FileNotFoundError as e:
            print(f"[ERROR] Input list file not found: {e.filename}")
            sys.exit(1)

        if len(cam_json_lines) != len(obj_json_lines):
            raise ValueError("Camera and object list lines do not match!")

        OUT_DIR.mkdir(parents=True, exist_ok=True)

        for idx, (cam_raw, obj_raw) in enumerate(zip(cam_json_lines, obj_json_lines), 1):
            cam_traj_json = (PROJECT_ROOT / cam_raw).resolve() if not Path(cam_raw).is_absolute() else Path(cam_raw)
            obj_traj_json = (PROJECT_ROOT / obj_raw).resolve() if not Path(obj_raw).is_absolute() else Path(obj_raw)

            if not cam_traj_json.exists() or not obj_traj_json.exists():
                print(f"[ERROR] JSON file not found: {cam_traj_json.name} or {obj_traj_json.name}")
                continue

            prefix = cam_traj_json.stem.replace("_transforms_pred", "").replace("_transforms_cleaning", "")
            print(f"\n--- [PROCESSING] {idx}/{len(cam_json_lines)} -> {prefix} ---")

            out_path = OUT_DIR / f"{prefix}_projection.mp4"
            try:
                pipeline.create_video(
                    output_path=str(out_path),
                    cam_json_path=str(cam_traj_json),
                    obj_json_path=str(obj_traj_json),
                    base_bbox_size=BASE_BBOX_SIZE,
                    fps=RENDER_FPS,
                    camera_z_offset=CAMERA_Z_OFFSET,
                    create_2d_video=True,
                    create_inverted_video=True,
                    create_grid_only_video=True
                )
                print(f"[VIDEO] Created: {out_path}")
            except Exception as e:
                print(f"[HATA] Render hatası: {cam_traj_json} ({e})")
                logger.exception("Render hatası:")
                continue
        print(f"\n[COMPLETED] Outputs: {OUT_DIR.resolve()}")

    def run_single_file_mode(bbox_json_str: str, cam_json_str: str, output_dir_str: str):
        print("Tek dosya modunda çalışılıyor (argümanlar kullanılıyor)...")
        cam_traj_json = Path(cam_json_str).resolve()
        obj_traj_json = Path(bbox_json_str).resolve()
        OUT_DIR = Path(output_dir_str)

        if not cam_traj_json.exists() or not obj_traj_json.exists():
            print(f"[HATA] JSON dosyası bulunamadı: {cam_traj_json if not cam_traj_json.exists() else obj_traj_json}")
            sys.exit(1)
        
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        prefix = cam_traj_json.stem.replace("_transforms_pred", "").replace("_transforms_cleaning", "")
        print(f"\n--- [İŞLEM] -> {prefix} ---")
        out_path = OUT_DIR / f"{prefix}_projection.mp4"

        try:
            pipeline.create_video(
                output_path=str(out_path),
                cam_json_path=str(cam_traj_json),
                obj_json_path=str(obj_traj_json),
                base_bbox_size=BASE_BBOX_SIZE,
                fps=RENDER_FPS,
                camera_z_offset=CAMERA_Z_OFFSET,
                create_2d_video=False,
                create_inverted_video=False,
                create_grid_only_video=False
            )
            print(f"[VIDEO] Created: {out_path}")
        except Exception as e:
            print(f"[ERROR] Render error: {cam_traj_json} ({e})")
            logger.exception("Render error:")
            sys.exit(1)
        print(f"\n[COMPLETED] Outputs: {OUT_DIR.resolve()}")

    if len(sys.argv) == 1:
        run_default_mode()
    elif len(sys.argv) == 4:
        run_single_file_mode(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Invalid usage!", file=sys.stderr)
        print(f"  Default mode: python3 {sys.argv[0]}", file=sys.stderr)
        print(f"  Single file mode: python3 {sys.argv[0]} <bbox.json> <cam.json> <output_dir>", file=sys.stderr)
        sys.exit(1)