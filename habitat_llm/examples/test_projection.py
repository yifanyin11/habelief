import os
import sys
import numpy as np
import imageio
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from pathlib import Path
import pathlib
from PIL import Image

ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
sys.path.append("..")

from habitat_llm.perception.perception_sim import (
    UNKNOWN_SEMANTIC_ID,
    compute_2d_bbox_from_aabb,
)

# --- Config ---
room_names = ["bathroom_1", "bedroom_1", "bedroom_2", "closet_1", "closet_2", "dining_room_1", "entryway_1", "hallway_1", "kitchen_1", "living_room_1", "office_1"]
# room_names = ["bedroom_2"]
ROOM_PATH_ROOT = "/home/zheyuanzhang/Documents/GitHub/habelief/data/trajectories/test/epidx_0_scene_106878960_174887073/agent_1/"
DRAW_CENTER = True

def get_intrinsic_matrix(intrinsics_vector):
    fx, fy, cx, cy = intrinsics_vector[0], intrinsics_vector[1], intrinsics_vector[2], intrinsics_vector[3]
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float32)

def world_to_camera(T_wc_inv, p_world):
    p = np.array([p_world[0], p_world[1], p_world[2], 1.0], dtype=np.float32)
    return (T_wc_inv @ p)[:3]

def project_cam_to_pixel(K, pc, W, H):
    x, y, z = float(pc[0]), float(pc[1]), float(pc[2])
    if z >= -1e-8:
        return None
    uvw = K @ np.array([x, y, z], dtype=np.float32)
    u = float(uvw[0] / z)
    v = float(uvw[1] / z)
    u = float(K[0, 2] * 2.0 - u)
    if not (0 <= u < W and 0 <= v < H):
        return None
    return (u, v)

def aabb_corners(local_aabb):
    return np.array([
        local_aabb.front_bottom_left, local_aabb.front_bottom_right,
        local_aabb.front_top_left,    local_aabb.front_top_right,
        local_aabb.back_bottom_left,  local_aabb.back_bottom_right,
        local_aabb.back_top_left,     local_aabb.back_top_right
    ], dtype=np.float32)

# def project_aabb_to_bbox2d(local_aabb, global_T, K, T_wc_inv, W, H):
#     corners = aabb_corners(local_aabb)
#     corners_h = np.concatenate([corners, np.ones((8, 1), dtype=np.float32)], axis=1)
#     corners_world = (np.array(global_T, dtype=np.float32) @ corners_h.T).T[:, :3]

#     pixels = []
#     for p in corners_world:
#         pc = world_to_camera(T_wc_inv, p)
#         uv = project_cam_to_pixel(K, pc, W, H)
#         if uv is not None:
#             pixels.append(uv)

#     if not pixels:
#         return None

#     us = [p[0] for p in pixels]; vs = [p[1] for p in pixels]
#     xmin = int(np.clip(np.floor(min(us)), 0, W - 1))
#     xmax = int(np.clip(np.ceil(max(us)),  0, W - 1))
#     ymin = int(np.clip(np.floor(min(vs)), 0, H - 1))
#     ymax = int(np.clip(np.ceil(max(vs)),  0, H - 1))
#     cx = float(np.mean(us)); cy = float(np.mean(vs))
#     return (xmin, ymin, xmax, ymax), (cx, cy)

def project_aabb_to_bbox2d(local_aabb, global_T, K, T_wc_inv, W, H):
    corners = aabb_corners(local_aabb)
    corners_h = np.concatenate([corners, np.ones((8, 1), dtype=np.float32)], axis=1)
    corners_world = (np.array(global_T, dtype=np.float32) @ corners_h.T).T[:, :3]

    pixels = []
    for pos_3d in corners_world:
        # print("pos_3d:", pos_3d)
        pos_2d = project_3d_to_2d_from_perspective_camera(pos_3d, K, T_wc_inv)
        # print("pos_2d:", pos_2d)
        pos_2d = (W - 1 - pos_2d[0], pos_2d[1])
        pixels.append(pos_2d)

    us = [p[0] for p in pixels]; vs = [p[1] for p in pixels]
    xmin = int(np.clip(np.floor(min(us)), 0, W - 1))
    xmax = int(np.clip(np.ceil(max(us)),  0, W - 1))
    ymin = int(np.clip(np.floor(min(vs)), 0, H - 1))
    ymax = int(np.clip(np.ceil(max(vs)),  0, H - 1))
    # cx = float(np.mean(us)); cy = float(np.mean(vs))
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    return (xmin, ymin, xmax, ymax), (cx, cy)


def project_3d_to_2d_from_perspective_camera(pos_3d, intrinsic_K, camera_extrinsics):
	extrinsic = camera_extrinsics[:3, :4]
	P_world = np.append(pos_3d, 1.0)
	P_camera = extrinsic @ P_world
	P_image = intrinsic_K @ P_camera
	pixel_x = int(P_image[0] / P_image[2])
	pixel_y = int(P_image[1] / P_image[2])
	return pixel_x, pixel_y

def aabb_volume_from_local(local_aabb):
    corners = aabb_corners(local_aabb)
    mins = corners.min(axis=0)
    maxs = corners.max(axis=0)
    extents = maxs - mins
    vol = float(np.prod(extents))
    return vol, mins, maxs, extents

def aabb_volume_from_global(local_aabb, global_T):
    corners = aabb_corners(local_aabb)
    corners_h = np.concatenate([corners, np.ones((8, 1), dtype=np.float32)], axis=1)
    corners_global = (np.array(global_T, dtype=np.float32) @ corners_h.T).T[:, :3]
    mins = corners_global.min(axis=0)
    maxs = corners_global.max(axis=0)
    extents = maxs - mins
    vol = float(np.prod(extents))
    return vol, mins, maxs, extents

def main(room_path):
    
    intrinsics = np.load(os.path.join(room_path, "..", "intrinsics.npy"), allow_pickle=True)[0]
    K = get_intrinsic_matrix(intrinsics)

    rgb_dir = os.path.join(room_path, "rgb")
    pose_dir = os.path.join(room_path, "pose")
    pan_dir  = os.path.join(room_path, "panoptic")
    assert os.path.isdir(rgb_dir) and os.path.isdir(pose_dir) and os.path.isdir(pan_dir), "Missing rgb/pose/panoptic dirs"

    frame_ids = sorted([int(Path(f).stem) for f in os.listdir(rgb_dir) if f.endswith(".jpg")])
    assert frame_ids, "No frames in rgb/"
    frame = frame_ids[0]
    print("frame:", frame)

    img = imageio.v2.imread(os.path.join(rgb_dir, f"{frame}.jpg"))
    H, W = img.shape[:2]
    pan = imageio.v2.imread(os.path.join(pan_dir, f"{frame}.png"))
    T_cw = np.load(os.path.join(pose_dir, f"{frame}.npy"))
    T_wc_inv = np.linalg.inv(T_cw)

    # print("rgb:", img.shape, "pan:", pan.shape)
    # print("K:", K)

    # if pan.shape[0] != H or pan.shape[1] != W:
    #     pan = np.array(Image.fromarray(pan).resize((W, H), resample=Image.NEAREST))

    unique = np.unique(pan)
    visible_ids = [int(i) - 100 for i in unique if i != UNKNOWN_SEMANTIC_ID]

    obj_id_to_handle = np.load(os.path.join(room_path, "..", "object_id_to_handle.npy"), allow_pickle=True).item()
    ao_id_to_handle = np.load(os.path.join(room_path, "..", "ao_id_to_handle.npy"), allow_pickle=True).item()
    ao_ids = set(ao_id_to_handle.values())
    print("ao_ids:", ao_ids)
    print("visible_ids before:", visible_ids)
    visible_ids = [i for i in visible_ids if i not in ao_ids]
    print("visible_ids:", visible_ids)
    all_objects = np.load(os.path.join(room_path, "all_objects", sorted(os.listdir(os.path.join(room_path, "all_objects")))[0]), allow_pickle=True)
    all_furnitures = np.load(os.path.join(room_path, "all_furnitures", sorted(os.listdir(os.path.join(room_path, "all_furnitures")))[0]), allow_pickle=True)
    all_entities = list(all_objects) + list(all_furnitures)
    obj_id_to_name = {
        obj_id: next((obj.name for obj in all_entities if getattr(obj, "sim_handle", None) == handle), None)
        for obj_id, handle in obj_id_to_handle.items()
    }
    obj_id_to_name = {k: v for k, v in obj_id_to_name.items() if v is not None}
    # print("obj_id_to_name:", obj_id_to_name)

    all_bboxes = np.load(os.path.join(room_path, "..", "all_bb.npy"), allow_pickle=True).item()
    # print("ids of bboxes:", list(all_bboxes.keys()))

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.imshow(img)
    ax.set_axis_off()

    drawn = 0
    for oid in visible_ids:
        if oid not in all_bboxes: # or oid not in obj_id_to_name:
            continue
        print(f"Drawing bbox for object id: {oid}")
        name = obj_id_to_name.get(oid, f"id:{oid}")
        local_aabb, global_T = all_bboxes[oid]
        # print(f"{name}'s aabb:", local_aabb)
        # print(f"{name}'s global_T:", global_T)
        print(f"{name}'s local volume:", aabb_volume_from_local(local_aabb)[0])
        print(f"{name}'s global volume:", aabb_volume_from_global(local_aabb, global_T)[0])
        bb = compute_2d_bbox_from_aabb(local_aabb, np.array(global_T), np.array(K), np.array(T_wc_inv))
        if bb is None:
            continue
        x1, y1, x2, y2 = bb["x_min"], bb["y_min"], bb["x_max"], bb["y_max"]
        print(f"{name}'s 2D bbox area:", bb["area"])

        rect = Rectangle((x1, y1), (x2 - x1), (y2 - y1),
                         linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)

        if DRAW_CENTER:
            ax.add_patch(Circle(((x1 + x2) / 2, (y1 + y2) / 2), radius=3.0, color='red'))

        ax.text(x1, max(0, y1 - 4), name,
                fontsize=8, color='yellow', backgroundcolor='black')

        drawn += 1

    fig.tight_layout(pad=0)
    # fig.savefig(out_path, bbox_inches='tight')
    # plt.close(fig)
    plt.show()
    # print(f"saved to: {os.path.abspath(out_path)}")

if __name__ == "__main__":
    for room_name in room_names:
        print("Room:", room_name)
        ROOM_PATH = os.path.join(ROOM_PATH_ROOT, room_name)
        # OUT_PATH = os.path.join(ROOM_PATH_ROOT, room_name, "annotated_first_frame.jpg")
        main(ROOM_PATH)