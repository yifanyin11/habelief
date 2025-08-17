import os
from pydoc import text
import re
import json
import shutil
import pathlib
import numpy as np
import imageio
from numpy.linalg import norm
from math import acos
import sys
import time
from tqdm import tqdm
from math import pi, atan2
import cv2
import matplotlib.pyplot as plt

ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
sys.path.append("..")

from habitat_llm.perception.perception_sim import (
    UNKNOWN_SEMANTIC_ID,
    compute_2d_bbox_from_aabb,
)

DEBUG_MODE = True
DRAW_CENTER_USING_BBOX = True

AGENT_NAME = "agent_1"
VISIBLE_BBOX_RATIO_MIN = 0.05
VISIBLE_BBOX_AREA_PX_MIN = 1200
MIN_FORWARD_DEPTH_M = 0.05
DIRECTION_COMPONENT_MIN = 0.05

DIR_HALF_ANGLES = {
    "front": pi / 12,
    "back":  pi / 12,
    "left":  pi / 12,
    "right": pi / 12,
}

DIR_MIN_XZ_DIST_M = 0.15
LARGER_RATIO_MIN = 1.35
CLOSER_DELTA_MIN = 1

DIR_FOLDER = "direction"
LARGER_FOLDER = "larger"
CLOSER_FOLDER = "closer"

def clean_text(text: str) -> str:
    return ''.join([c for c in text if not c.isdigit()]).replace("_", " ").strip()

def get_intrinsic_matrix(intrinsics_vector):
    fx, fy, cx, cy = intrinsics_vector[0], intrinsics_vector[1], intrinsics_vector[2], intrinsics_vector[3]
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float32)
    return K

def world_to_camera(T_wc_inv, p_world):
    p = np.array([p_world[0], p_world[1], p_world[2], 1.0], dtype=np.float32)
    pc = T_wc_inv @ p
    return pc[:3]

def projected_bbox_ratio(local_aabb, global_transform, K, T_wc_inv, img_size):
    bbox = compute_2d_bbox_from_aabb(local_aabb, np.array(global_transform), np.array(K), np.array(T_wc_inv))
    area = 0 if bbox["area"] == np.inf else bbox["area"]
    H, W = img_size
    return float(area) / float(H * W + 1e-6)

def list_visible_entity_ids_for_frame(panoptic_img, obj_id_to_name, ao_handle_to_id, furns_in_room_ids):
    unique = np.unique(panoptic_img)
    ids = [i - 100 for i in unique if i != UNKNOWN_SEMANTIC_ID]
    ids += furns_in_room_ids
    ids = [i for i in ids if i in obj_id_to_name]
    return list(set(ids))

def classify_direction_sector(vx: float, vz: float) -> str:
    r = (vx * vx + vz * vz) ** 0.5
    if r < DIR_MIN_XZ_DIST_M:
        return None

    theta = atan2(vx, -vz)
    diffs = {
        "front": min(abs(theta - pi), abs(theta + pi)), # note that front/back is inverted under english convention
        "right": abs(theta - pi/2),
        "back":  abs(theta),
        "left":  abs(theta + pi/2),
    }

    best_dir = min(diffs, key=diffs.get)
    if diffs[best_dir] <= DIR_HALF_ANGLES[best_dir]:
        return best_dir
    return None

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def append_jsonl(path, record):
    def _to_native(o):
        if isinstance(o, (np.floating, np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.integer, np.int32, np.int64)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        try:
            return o.item()
        except Exception:
            return str(o)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=_to_native) + "\n")

def project_cam_to_pixel(K, pc):
    x, y, z = float(pc[0]), float(pc[1]), float(pc[2])
    if z >= -1e-8:
        return None
    uvw = K @ np.array([x, y, z], dtype=np.float32)
    u = float(uvw[0] / z)
    v = float(uvw[1] / z)
    u = float(K[0, 2] * 2.0 - u)
    return (u, v)

def compute_object_pixel_center(local_aabb, global_T, K, T_wc_inv, W, H):
    corners_local = np.array([
        local_aabb.front_bottom_left, local_aabb.front_bottom_right,
        local_aabb.front_top_left,    local_aabb.front_top_right,
        local_aabb.back_bottom_left,  local_aabb.back_bottom_right,
        local_aabb.back_top_left,     local_aabb.back_top_right
    ], dtype=np.float32)
    corners_local_h = np.concatenate([corners_local, np.ones((8,1), dtype=np.float32)], axis=1)
    corners_world_h = (np.array(global_T, dtype=np.float32) @ corners_local_h.T).T
    uv_list = []
    for k in range(8):
        pw = corners_world_h[k, :3]
        pc = world_to_camera(T_wc_inv, pw)
        uv = project_cam_to_pixel(K, pc)
        if uv is None:
            continue
        u, v = uv
        if 0 <= u < W and 0 <= v < H:
            uv_list.append((u, v))
    if not uv_list:
        return None
    u_mean = int(round(np.mean([p[0] for p in uv_list])))
    v_mean = int(round(np.mean([p[1] for p in uv_list])))
    return (u_mean, v_mean)

def draw_markers_rgb(rgb_img, A_px, B_px, A_bbox=None, B_bbox=None):
    img = rgb_img.copy()
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def _draw_px(pt, text):
        if pt is None:
            return
        r = 10
        cv2.circle(img_bgr, pt, radius=r, color=(0, 0, 255), thickness=-1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = int(pt[0] - tw / 2)
        text_y = int(pt[1] + th / 2)
        cv2.putText(img_bgr, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    
    def _draw_px_using_bbox(bbox, text):
        pt = (int((bbox["x_min"] + bbox["x_max"]) / 2), int((bbox["y_min"] + bbox["y_max"]) / 2))
        r = 10
        cv2.circle(img_bgr, pt, radius=r, color=(0, 0, 255), thickness=-1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = int(pt[0] - tw / 2)
        text_y = int(pt[1] + th / 2)
        cv2.putText(img_bgr, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    
    def _draw_bbox(bbox, text):
        x1, y1, x2, y2 = bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]
        h, w = img_bgr.shape[:2]
        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(0, min(w - 1, int(x2)))
        y2 = max(0, min(h - 1, int(y2)))

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        text_x = max(0, min(w - tw - 2, x1))
        text_y = max(th + 2, y1 - 4)

        bg_tl = (text_x - 2, text_y - th - 2)
        bg_br = (text_x + tw + 2, text_y + 2)
        bg_tl = (max(0, bg_tl[0]), max(0, bg_tl[1]))
        bg_br = (min(w - 1, bg_br[0]), min(h - 1, bg_br[1]))
        cv2.rectangle(img_bgr, bg_tl, bg_br, (0, 0, 0), thickness=-1)

        cv2.putText(img_bgr, text, (text_x, text_y), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

    if A_bbox is not None and DRAW_CENTER_USING_BBOX:
        _draw_px_using_bbox(A_bbox, "A")
    else:
        _draw_px(A_px, "A")
    if B_bbox is not None and DRAW_CENTER_USING_BBOX:
        _draw_px_using_bbox(B_bbox, "B")
    else:
        _draw_px(B_px, "B")
    if A_bbox is not None and DEBUG_MODE:
        _draw_bbox(A_bbox, "A")
    if B_bbox is not None and DEBUG_MODE:
        _draw_bbox(B_bbox, "B")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def relation_statement(subject_name: str, object_name: str, relation: str) -> str:
    s = clean_text(subject_name)
    o = clean_text(object_name)
    if relation == "left":
        return f"The {s} (mark A) to the left of the {o} (mark B)."
    if relation == "right":
        return f"The {s} (mark A) to the right of the {o} (mark B)."
    if relation == "front":
        return f"The {s} (mark A) in front of the {o} (mark B)."
    if relation == "back":
        return f"The {s} (mark A) behind the {o} (mark B)."
    if relation == "larger":
        return f"The {s} (mark A) larger than the {o} (mark B)."
    if relation == "closer":
        return f"The {s} (mark A) closer than the {o} (mark B)."
    return


def process_frame_for_relationships(
    episode_id,
    room_name,
    frame_idx,
    room_path,
    K,
    T_wc_inv,
    all_bboxes,
    obj_id_to_handle,
    obj_id_to_name,
    ao_handle_to_id,
    furn_ids_in_room,
    output_root
):
    frame_counts = {
        "left": 0,
        "right": 0,
        "front": 0,
        "back": 0,
        "larger": 0,
        "closer": 0,
    }

    rgb_path = os.path.join(room_path, "rgb", f"{frame_idx}.jpg")
    panoptic_path = os.path.join(room_path, "panoptic", f"{frame_idx}.png")
    if not (os.path.exists(rgb_path) and os.path.exists(panoptic_path)):
        return frame_counts

    rgb = imageio.v2.imread(rgb_path)
    H, W = rgb.shape[0], rgb.shape[1]
    pan = imageio.v2.imread(panoptic_path)

    visible_ids = list_visible_entity_ids_for_frame(
        pan, obj_id_to_name, ao_handle_to_id, furn_ids_in_room
    )
    if len(visible_ids) < 2:
        return frame_counts

    visible_keep = []
    for oid in visible_ids:
        if oid not in all_bboxes:
            continue
        local_aabb, global_T = all_bboxes[oid]
        bbox = compute_2d_bbox_from_aabb(
            local_aabb, np.array(global_T), np.array(K), np.array(T_wc_inv)
        )
        area_px = 0.0 if bbox["area"] == np.inf else float(bbox["area"])
        ratio = area_px / float(H * W + 1e-6)

        corners_local = np.array([
            local_aabb.front_bottom_left, local_aabb.front_bottom_right,
            local_aabb.front_top_left,    local_aabb.front_top_right,
            local_aabb.back_bottom_left,  local_aabb.back_bottom_right,
            local_aabb.back_top_left,     local_aabb.back_top_right
        ], dtype=np.float32)
        corners_local_h = np.concatenate([corners_local, np.ones((8,1), dtype=np.float32)], axis=1)
        corners_world_h = (np.array(global_T, dtype=np.float32) @ corners_local_h.T).T
        center_world = corners_world_h[:, :3].mean(axis=0)
        pc_center = world_to_camera(T_wc_inv, center_world)
        depth_forward = max(0.0, -pc_center[2])

        if (
            ratio >= VISIBLE_BBOX_RATIO_MIN
            and area_px >= VISIBLE_BBOX_AREA_PX_MIN
            and depth_forward >= MIN_FORWARD_DEPTH_M
        ):
            visible_keep.append((oid, ratio))

    if len(visible_keep) < 2:
        return frame_counts

    centers_cam = {}
    centers_px = {}
    distances = {}
    depths = {}
    bboxes = {}
    areas = {}

    for oid, _ in visible_keep:
        local_aabb, global_T = all_bboxes[oid]

        corners_local = np.array([
            local_aabb.front_bottom_left, local_aabb.front_bottom_right,
            local_aabb.front_top_left,    local_aabb.front_top_right,
            local_aabb.back_bottom_left,  local_aabb.back_bottom_right,
            local_aabb.back_top_left,     local_aabb.back_top_right
        ], dtype=np.float32)
        corners_local_h = np.concatenate([corners_local, np.ones((8,1), dtype=np.float32)], axis=1)
        corners_world_h = (np.array(global_T, dtype=np.float32) @ corners_local_h.T).T
        center_world = corners_world_h[:, :3].mean(axis=0)

        pc = world_to_camera(T_wc_inv, center_world)
        T_cw = np.linalg.inv(T_wc_inv)
        cam_world = T_cw[:3, 3].astype(np.float32)
        dist_world = float(norm(center_world - cam_world))
        distances[oid] = dist_world
        centers_cam[oid] = pc
        depths[oid] = max(0.0, -pc[2])

        bbox = compute_2d_bbox_from_aabb(
            local_aabb, np.array(global_T), np.array(K), np.array(T_wc_inv)
        )
        bboxes[oid] = bbox
        areas[oid] = 0.0 if bbox["area"] == np.inf else float(bbox["area"])

        centers_px[oid] = compute_object_pixel_center(local_aabb, global_T, K, T_wc_inv, W, H)

    ep_room = f"{episode_id}_{room_name}"
    basename = f"{ep_room}_{frame_idx}"

    for i in range(len(visible_keep)):
        oidA = visible_keep[i][0]
        for j in range(i + 1, len(visible_keep)):
            oidB = visible_keep[j][0]
            nameA = obj_id_to_name.get(oidA, f"id{oidA}")
            nameB = obj_id_to_name.get(oidB, f"id{oidB}")

            v = centers_cam[oidA] - centers_cam[oidB]
            card = classify_direction_sector(float(v[0]), float(v[2]))
            if card is not None:
                out_dir = os.path.join(output_root, DIR_FOLDER, card)
                ensure_dir(out_dir)
                ensure_dir(os.path.join(out_dir, "rgb"))

                out_name = f"{basename}__{sanitize(nameA)}__{card}__{sanitize(nameB)}.jpg"
                out_path = os.path.join(out_dir, "rgb", out_name)

                img_marked = draw_markers_rgb(
                    rgb_img=rgb,
                    A_px=centers_px.get(oidA),
                    B_px=centers_px.get(oidB),
                    A_bbox=bboxes.get(oidA),
                    B_bbox=bboxes.get(oidB),
                )
                imageio.v2.imwrite(out_path, img_marked)

                rec = {
                    "episode_id": episode_id,
                    "room": room_name,
                    "frame": int(frame_idx),
                    "image": out_name,
                    "relation_type": "direction",
                    "relation": card,
                    "subject": nameA,
                    "object": nameB,
                    "camera_depth_subject": float(depths[oidA]),
                    "camera_depth_object": float(depths[oidB]),
                    "subject_pixel": centers_px.get(oidA),
                    "object_pixel": centers_px.get(oidB),
                    "statement": relation_statement(nameA, nameB, card),
                }
                append_jsonl(os.path.join(out_dir, "index.jsonl"), rec)
                frame_counts[card] += 1

            areaA = float(areas[oidA]); areaB = float(areas[oidB])
            if areaA > LARGER_RATIO_MIN * max(areaB, 1e-6):
                out_dir = os.path.join(output_root, LARGER_FOLDER)
                ensure_dir(out_dir); ensure_dir(os.path.join(out_dir, "rgb"))
                out_name = f"{basename}__{sanitize(nameA)}__larger__{sanitize(nameB)}.jpg"
                out_path = os.path.join(out_dir, "rgb", out_name)

                img_marked = draw_markers_rgb(rgb, centers_px.get(oidA), centers_px.get(oidB), A_bbox=bboxes.get(oidA), B_bbox=bboxes.get(oidB))
                imageio.v2.imwrite(out_path, img_marked)

                rec = {
                    "episode_id": episode_id,
                    "room": room_name,
                    "frame": int(frame_idx),
                    "image": out_name,
                    "relation_type": "larger",
                    "relation": "larger",
                    "subject": nameA,
                    "object": nameB,
                    "area_subject_px": float(areaA),
                    "area_object_px": float(areaB),
                    "area_ratio": float(areaA / max(areaB, 1e-6)),
                    "subject_pixel": centers_px.get(oidA),
                    "object_pixel": centers_px.get(oidB),
                    "statement": relation_statement(nameA, nameB, "larger"),
                }
                append_jsonl(os.path.join(out_dir, "index.jsonl"), rec)
                frame_counts["larger"] += 1

            elif areaB > LARGER_RATIO_MIN * max(areaA, 1e-6):
                out_dir = os.path.join(output_root, LARGER_FOLDER)
                ensure_dir(out_dir); ensure_dir(os.path.join(out_dir, "rgb"))
                out_name = f"{basename}__{sanitize(nameB)}__larger__{sanitize(nameA)}.jpg"
                out_path = os.path.join(out_dir, "rgb", out_name)

                img_marked = draw_markers_rgb(rgb, centers_px.get(oidB), centers_px.get(oidA), A_bbox=bboxes.get(oidB), B_bbox=bboxes.get(oidA))
                imageio.v2.imwrite(out_path, img_marked)

                rec = {
                    "episode_id": episode_id,
                    "room": room_name,
                    "frame": int(frame_idx),
                    "image": out_name,
                    "relation_type": "larger",
                    "relation": "larger",
                    "subject": nameB,
                    "object": nameA,
                    "area_subject_px": float(areaB),
                    "area_object_px": float(areaA),
                    "area_ratio": float(areaB / max(areaA, 1e-6)),
                    "subject_pixel": centers_px.get(oidB),
                    "object_pixel": centers_px.get(oidA),
                    "statement": relation_statement(nameB, nameA, "larger"),
                }
                append_jsonl(os.path.join(out_dir, "index.jsonl"), rec)
                frame_counts["larger"] += 1

            disA = float(distances[oidA]); disB = float(distances[oidB])
            if (disA - disB) > CLOSER_DELTA_MIN:
                # print("distance to A:", disA)
                # print("distance to B:", disB)
                out_dir = os.path.join(output_root, CLOSER_FOLDER)
                ensure_dir(out_dir); ensure_dir(os.path.join(out_dir, "rgb"))
                out_name = f"{basename}__{sanitize(nameA)}__closer__{sanitize(nameB)}.jpg"
                out_path = os.path.join(out_dir, "rgb", out_name)

                img_marked = draw_markers_rgb(rgb, centers_px.get(oidA), centers_px.get(oidB), A_bbox=bboxes.get(oidA), B_bbox=bboxes.get(oidB))
                imageio.v2.imwrite(out_path, img_marked)
                # plt.imshow(img_marked)
                # plt.axis("off")
                # plt.show()

                rec = {
                    "episode_id": episode_id,
                    "room": room_name,
                    "frame": int(frame_idx),
                    "image": out_name,
                    "relation_type": "closer",
                    "relation": "closer",
                    "subject": nameA,
                    "object": nameB,
                    "distance_subject": float(disA),
                    "distance_object": float(disB),
                    "distance_delta": float(disA - disB),
                    "subject_pixel": centers_px.get(oidA),
                    "object_pixel": centers_px.get(oidB),
                    "statement": relation_statement(nameA, nameB, "closer"),
                }
                append_jsonl(os.path.join(out_dir, "index.jsonl"), rec)
                frame_counts["closer"] += 1

            elif (disA - disB) > CLOSER_DELTA_MIN:
                # print("distance to A:", disA)
                # print("distance to B:", disB)
                out_dir = os.path.join(output_root, CLOSER_FOLDER)
                ensure_dir(out_dir); ensure_dir(os.path.join(out_dir, "rgb"))
                out_name = f"{basename}__{sanitize(nameB)}__closer__{sanitize(nameA)}.jpg"
                out_path = os.path.join(out_dir, "rgb", out_name)

                img_marked = draw_markers_rgb(rgb, centers_px.get(oidB), centers_px.get(oidA), A_bbox=bboxes.get(oidB), B_bbox=bboxes.get(oidA))
                imageio.v2.imwrite(out_path, img_marked)
                # plt.imshow(img_marked)
                # plt.axis("off")
                # plt.show()

                rec = {
                    "episode_id": episode_id,
                    "room": room_name,
                    "frame": int(frame_idx),
                    "image": out_name,
                    "relation_type": "closer",
                    "relation": "closer",
                    "subject": nameB,
                    "object": nameA,
                    "distance_subject": float(disB),
                    "distance_object": float(disA),
                    "distance_delta": float(disB - disA),
                    "subject_pixel": centers_px.get(oidB),
                    "object_pixel": centers_px.get(oidA),
                    "statement": relation_statement(nameB, nameA, "closer"),
                }
                append_jsonl(os.path.join(out_dir, "index.jsonl"), rec)
                frame_counts["closer"] += 1

    return frame_counts


def sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\-]+", "_", name).strip("_")

def generate_relationship_dataset(data_root, output_root):
    from tqdm import tqdm
    import time

    stats = {
        "direction_left": 0,
        "direction_right": 0,
        "direction_front": 0,
        "direction_back": 0,
        "larger": 0,
        "closer": 0,
        "rooms_processed": 0,
        "episodes_processed": 0,
        "frames_seen": 0,
    }

    ensure_dir(output_root)

    episodes = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    episodes.sort()

    print(f"[START] Scanning dataset root: {data_root}")

    for episode_id in episodes[:1]:
        episode_path = os.path.join(data_root, episode_id, AGENT_NAME)
        if not os.path.isdir(episode_path):
            continue

        room_names = [
            name for name in os.listdir(episode_path)
            if os.path.isdir(os.path.join(episode_path, name)) and "unknown" not in name
        ]
        room_names.sort()
        print(f"Episode: {episode_id} — rooms: {len(room_names)}")

        for room_name in room_names[:1]:
            room_path = os.path.join(episode_path, room_name)
            intrinsics_path = os.path.join(room_path, "..", "intrinsics.npy")
            if not os.path.exists(intrinsics_path):
                print(f"[Skip] Missing intrinsics for {episode_id}/{room_name}")
                continue
            K = get_intrinsic_matrix(np.load(intrinsics_path, allow_pickle=True)[0])
            all_objects = np.load(os.path.join(room_path, "all_objects", sorted(os.listdir(os.path.join(room_path, "all_objects")))[0]), allow_pickle=True)
            all_furnitures = np.load(os.path.join(room_path, "all_furnitures", sorted(os.listdir(os.path.join(room_path, "all_furnitures")))[0]), allow_pickle=True)
            all_entities = list(all_objects) + list(all_furnitures)
            obj_id_to_handle = np.load(os.path.join(room_path, "..", "object_id_to_handle.npy"), allow_pickle=True).item()
            ao_id_to_handle  = np.load(os.path.join(room_path, "..", "ao_id_to_handle.npy"), allow_pickle=True).item()
            obj_id_to_name = {
                obj_id: next((obj.name for obj in all_entities if getattr(obj, "sim_handle", None) == handle), None)
                for obj_id, handle in obj_id_to_handle.items()
            }
            obj_id_to_name = {k: v for k, v in obj_id_to_name.items() if v is not None}
            ao_handle_to_id = {v: k for k, v in ao_id_to_handle.items()}
            furn_ids_in_room = list(ao_handle_to_id.values())
            all_bboxes = np.load(os.path.join(room_path, "..", "all_bb.npy"), allow_pickle=True).item()
            frames = [
                int(os.path.splitext(f)[0]) for f in os.listdir(os.path.join(room_path, "rgb"))
                if f.endswith(".jpg") and os.path.exists(os.path.join(room_path, "pose", f"{os.path.splitext(f)[0]}.npy"))
            ]
            frames.sort()
            if not frames:
                continue

            pbar = tqdm(total=len(frames), desc=f"{episode_id}/{room_name}", leave=False)
            start_ts = time.time()

            for frame_idx in frames:
                pose_path = os.path.join(room_path, "pose", f"{frame_idx}.npy")
                T_wc_inv = np.linalg.inv(np.load(pose_path))

                frame_counts = process_frame_for_relationships(
                    episode_id=episode_id,
                    room_name=room_name,
                    frame_idx=frame_idx,
                    room_path=room_path,
                    K=K,
                    T_wc_inv=T_wc_inv,
                    all_bboxes=all_bboxes,
                    obj_id_to_handle=obj_id_to_handle,
                    obj_id_to_name=obj_id_to_name,
                    ao_handle_to_id=ao_handle_to_id,
                    furn_ids_in_room=furn_ids_in_room,
                    output_root=output_root
                )

                if frame_counts is not None:
                    stats["direction_left"] += frame_counts.get("left", 0)
                    stats["direction_right"] += frame_counts.get("right", 0)
                    stats["direction_front"] += frame_counts.get("front", 0)
                    stats["direction_back"] += frame_counts.get("back", 0)
                    stats["larger"] += frame_counts.get("larger", 0)
                    stats["closer"] += frame_counts.get("closer", 0)
                    stats["frames_seen"] += 1

                pbar.update(1)

            pbar.close()
            print(f"[{episode_id}/{room_name}] done {len(frames)} frames in {time.time()-start_ts:.1f}s")
            stats["rooms_processed"] += 1

        print(f"Finished episode {episode_id}")
        stats["episodes_processed"] += 1

    print("\n[SUMMARY]")
    print(f"Episodes processed: {stats['episodes_processed']}")
    print(f"Rooms processed: {stats['rooms_processed']}")
    print(f"Frames seen: {stats['frames_seen']}")
    print(f"Direction:")
    print(f"left: {stats['direction_left']}")
    print(f"right: {stats['direction_right']}")
    print(f"front: {stats['direction_front']}")
    print(f"back: {stats['direction_back']}")
    print(f"Larger: {stats['larger']}")
    print(f"Closer: {stats['closer']}")
    print(f"[DONE] Results written to: {output_root}")

if __name__ == "__main__":
    input_dataset_path = "/home/zheyuanzhang/Documents/GitHub/habelief/data/trajectories/test"
    output_dataset_path = "/home/zheyuanzhang/Documents/GitHub/habelief/data/trajectories/relationships"

    generate_relationship_dataset(input_dataset_path, output_dataset_path)