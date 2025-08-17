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

DEBUG_MODE = False
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
CLOSER_DELTA_MIN = 0.3

DIR_FOLDER = "direction"
LARGER_FOLDER = "larger"
CLOSER_FOLDER = "closer"

MIN_PAIR_FORWARD_M = 0.5
MIN_TURN_DEG_FOR_ACTION = 5.0
PAIR_FOLDER = "pairs"

def sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\-]+", "_", name).strip("_")

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
    # ids += furns_in_room_ids
    ids = [i for i in ids if i in obj_id_to_name]
    return list(set(ids))

def classify_direction_sector(vx: float, vz: float) -> str:
    r = (vx * vx + vz * vz) ** 0.5
    if r < DIR_MIN_XZ_DIST_M:
        return None
    theta = atan2(vx, -vz)
    diffs = {
        "front": min(abs(theta - pi), abs(theta + pi)),
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
    ensure_dir(os.path.dirname(path))
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

def draw_markers_rgb(rgb_img, A_px, B_px, A_bbox=None, B_bbox=None, A_name=None, B_name=None):
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
        font_scale = 0.3
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
    if A_bbox is not None:
        _draw_px_using_bbox(A_bbox, "A")
        # _draw_px(A_px, "A")
    if B_bbox is not None:
        _draw_px_using_bbox(B_bbox, "B")
        # _draw_px(B_px, "B")
    if DEBUG_MODE:
        if A_bbox is not None:
            _draw_bbox(A_bbox, "".join(A_name.split('_')[:-1]))
    if DEBUG_MODE:
        if B_bbox is not None:
            _draw_bbox(B_bbox, "".join(B_name.split('_')[:-1]))
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

def yaw_from_Tcw(T_cw: np.ndarray) -> float:
    R = T_cw[:3, :3].astype(np.float32)
    yaw = atan2(R[0,2], -R[2,2])
    return float(yaw)

def angle_diff_rad(a: float, b: float) -> float:
    d = (b - a + pi) % (2*pi) - pi
    return float(d)

def rad2deg(a: float) -> float:
    return float(a * 180.0 / pi)

def plan_actions(T_cw_start: np.ndarray, T_cw_end: np.ndarray):
    p0 = T_cw_start[:3, 3].astype(np.float32)
    p1 = T_cw_end[:3, 3].astype(np.float32)
    dp = p1 - p0
    dx, dz = float(dp[0]), float(dp[2])
    forward_dist = (dx*dx + dz*dz) ** 0.5
    yaw0 = yaw_from_Tcw(T_cw_start)
    yaw1 = yaw_from_Tcw(T_cw_end)
    desired_heading = atan2(dx, -dz)
    first_turn = angle_diff_rad(yaw0, desired_heading)
    final_turn = angle_diff_rad(desired_heading, yaw1)
    actions = []
    if abs(rad2deg(first_turn)) >= MIN_TURN_DEG_FOR_ACTION:
        if first_turn > 0:
            actions.append(f"turn-left {rad2deg(first_turn):.1f} degrees")
        else:
            actions.append(f"turn-right {abs(rad2deg(first_turn)):.1f} degrees")
    else:
        actions.append(f"turn-left {max(0.1, rad2deg(abs(first_turn))):.1f} degrees")
    actions.append(f"move-forward {forward_dist:.2f} m")
    if abs(rad2deg(final_turn)) >= MIN_TURN_DEG_FOR_ACTION:
        if final_turn > 0:
            actions.append(f"turn-left {rad2deg(final_turn):.1f} degrees")
        else:
            actions.append(f"turn-right {abs(rad2deg(final_turn)):.1f} degrees")
    else:
        actions.append(f"turn-right {max(0.1, rad2deg(abs(final_turn))):.1f} degrees")
    return actions, forward_dist

def collect_frame_facts(
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
    facts = {
        "visible_pairs": [],
        "visible_solo":  [],
        "rgb": None,
        "pose_T_cw": None,
    }
    rgb_path = os.path.join(room_path, "rgb", f"{frame_idx}.jpg")
    panoptic_path = os.path.join(room_path, "panoptic", f"{frame_idx}.png")
    if not (os.path.exists(rgb_path) and os.path.exists(panoptic_path)):
        return None
    rgb = imageio.v2.imread(rgb_path)
    H, W = rgb.shape[0], rgb.shape[1]
    pan = imageio.v2.imread(panoptic_path)
    # print(f"frame idx: {frame_idx}")
    visible_ids = list_visible_entity_ids_for_frame(
        pan, obj_id_to_name, ao_handle_to_id, furn_ids_in_room
    )
    # print(f"Visible ids: {visible_ids}")
    if len(visible_ids) < 1:
        return None
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
    if not visible_keep:
        return None
    centers_cam = {}
    centers_px = {}
    bboxes = {}
    areas = {}
    distances = {}
    depths = {}
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
        depths[oid]    = max(0.0, -pc[2])
        centers_cam[oid] = pc
        bbox = compute_2d_bbox_from_aabb(local_aabb, np.array(global_T), np.array(K), np.array(T_wc_inv))
        bboxes[oid] = bbox
        areas[oid] = 0.0 if bbox["area"] == np.inf else float(bbox["area"])
        centers_px[oid] = compute_object_pixel_center(local_aabb, global_T, K, T_wc_inv, W, H)
        facts["visible_solo"].append((oid, areas[oid], centers_px[oid], bboxes[oid]))
    for i in range(len(visible_keep)):
        oidA = visible_keep[i][0]
        for j in range(i + 1, len(visible_keep)):
            oidB = visible_keep[j][0]
            v = centers_cam[oidA] - centers_cam[oidB]
            card = classify_direction_sector(float(v[0]), float(v[2]))
            facts["visible_pairs"].append(
                (oidA, oidB, card, float(distances[oidA]), float(distances[oidB]),
                 float(areas[oidA]), float(areas[oidB]),
                 centers_px.get(oidA), centers_px.get(oidB),
                 bboxes.get(oidA), bboxes.get(oidB))
            )
    facts["rgb"] = rgb
    facts["pose_T_cw"] = np.linalg.inv(T_wc_inv)
    return facts

def is_flip_direction(r1, r2):
    return (r1, r2) in {("left","right"), ("right","left"), ("front","back"), ("back","front")}

def is_different_direction(r1, r2):
    return r1 != r2

def write_pair_sample(
    out_root, kind, ep_room, frame_i, frame_j,
    img_i, img_j,
    A_meta_i, B_meta_i, A_meta_j, B_meta_j,
    nameA, nameB,
    T_cw_i, T_cw_j,
    stats=None
):
    pair_id = f"{frame_i}_to_{frame_j}__{sanitize(nameA)}__{sanitize(nameB)}"
    out_dir = os.path.join(out_root, PAIR_FOLDER, kind, ep_room, pair_id)
    ensure_dir(os.path.join(out_dir, "rgb"))
    ann_i = draw_markers_rgb(img_i, A_meta_i[0], B_meta_i[0], A_meta_i[1], B_meta_i[1], A_name=nameA, B_name=nameB)
    imageio.v2.imwrite(os.path.join(out_dir, "rgb", "1.jpg"), ann_i)
    ann_j = draw_markers_rgb(img_j, A_meta_j[0], B_meta_j[0], A_meta_j[1], B_meta_j[1], A_name=nameA, B_name=nameB)
    imageio.v2.imwrite(os.path.join(out_dir, "rgb", "2.jpg"), ann_j)
    actions, forward_d = plan_actions(T_cw_i, T_cw_j)
    rec = {
        "episode_room": ep_room,
        "frames": [int(frame_i), int(frame_j)],
        "rgb_folder": os.path.join(PAIR_FOLDER, kind, ep_room, pair_id, "rgb"),
        "relation_type": kind,
        "objects": [nameA, nameB],
        "frame1_relation": A_meta_i[2],
        "frame2_relation": A_meta_j[2],
        "path_actions": actions,
        "forward_distance_m": float(forward_d),
    }
    append_jsonl(os.path.join(out_root, PAIR_FOLDER, kind, "index.jsonl"), rec)
    if stats is not None:
        if kind == "direction":
            stats["direction_pairs"] += 1
            for rel in (A_meta_i[2], A_meta_j[2]):
                if rel in ("left","right","front","back"):
                    stats[f"direction_{rel}"] += 1
        elif kind == "distance":
            stats["distance_pairs"] += 1

def build_pairs_for_room(
    episode_id, room_name, frames, per_frame_facts, obj_id_to_name, out_root, stats=None
):
    ep_room = f"{episode_id}_{room_name}"
    dir_obs = {}
    dist_obs = {}
    size_obs = {}
    for frame in frames:
        facts = per_frame_facts.get(frame)
        if facts is None:
            continue
        for (oidA, oidB, card, dA, dB, areaA, areaB, cpxA, cpxB, bbA, bbB) in facts["visible_pairs"]:
            dir_obs[(oidA, oidB, frame)] = (card, cpxA, cpxB, bbA, bbB, areaA, areaB, dA, dB)
            delta = float(dB - dA)
            if delta >= CLOSER_DELTA_MIN:
                closer = "A-closer"
            elif delta <= -CLOSER_DELTA_MIN:
                closer = "B-closer"
            else:
                closer = None
            dist_obs[(oidA, oidB, frame)] = (closer, cpxA, cpxB, bbA, bbB, areaA, areaB, dA, dB)
        for (oid, area_px, cpx, bbox) in facts["visible_solo"]:
            size_obs[(oid, frame)] = (area_px, cpx, bbox)
    for i_idx in range(0, len(frames), 10):
        fi = frames[i_idx]
        if per_frame_facts.get(fi) is None:
            continue
        Ti = per_frame_facts[fi]["pose_T_cw"]
        pi = Ti[:3,3]
        for j_idx in range(i_idx+10, len(frames), 10):
            fj = frames[j_idx]
            if per_frame_facts.get(fj) is None:
                continue
            Tj = per_frame_facts[fj]["pose_T_cw"]
            pj = Tj[:3,3]
            forward = float(np.linalg.norm([pj[0]-pi[0], pj[2]-pi[2]]))
            if forward < MIN_PAIR_FORWARD_M:
                continue
            keys_i = {(A,B) for (A,B,f) in dir_obs.keys() if f==fi}
            keys_j = {(A,B) for (A,B,f) in dir_obs.keys() if f==fj}
            common_pairs = keys_i & keys_j
            for (A,B) in common_pairs:
                (card_i, cAi, cBi, bbAi, bbBi, areaAi, areaBi, dAi, dBi) = dir_obs[(A,B,fi)]
                (card_j, cAj, cBj, bbAj, bbBj, areaAj, areaBj, dAj, dBj) = dir_obs[(A,B,fj)]
                if card_i is None or card_j is None:
                    continue
                if not is_different_direction(card_i, card_j):
                    continue
                nameA = obj_id_to_name.get(A, f"id{A}")
                nameB = obj_id_to_name.get(B, f"id{B}")
                write_pair_sample(
                    out_root, "direction", ep_room, fi, fj,
                    per_frame_facts[fi]["rgb"], per_frame_facts[fj]["rgb"],
                    (cAi, bbAi, card_i, areaAi, dAi),
                    (cBi, bbBi, card_i, areaBi, dBi),
                    (cAj, bbAj, card_j, areaAj, dAj),
                    (cBj, bbBj, card_j, areaBj, dBj),
                    nameA, nameB,
                    Ti, Tj,
                    stats=stats
                )
            keys_i = {(A,B) for (A,B,f) in dist_obs.keys() if f==fi}
            keys_j = {(A,B) for (A,B,f) in dist_obs.keys() if f==fj}
            common_pairs = keys_i & keys_j
            for (A,B) in common_pairs:
                (closer_i, cAi, cBi, bbAi, bbBi, areaAi, areaBi, dAi, dBi) = dist_obs[(A,B,fi)]
                (closer_j, cAj, cBj, bbAj, bbBj, areaAj, areaBj, dAj, dBj) = dist_obs[(A,B,fj)]
                if closer_i is None or closer_j is None:
                    continue
                if (closer_i == "A-closer" and closer_j == "B-closer") or (closer_i == "B-closer" and closer_j == "A-closer"):
                    nameA = obj_id_to_name.get(A, f"id{A}")
                    nameB = obj_id_to_name.get(B, f"id{B}")
                    rel_i = "A-closer" if closer_i == "A-closer" else "B-closer"
                    rel_j = "A-closer" if closer_j == "A-closer" else "B-closer"
                    write_pair_sample(
                        out_root, "distance", ep_room, fi, fj,
                        per_frame_facts[fi]["rgb"], per_frame_facts[fj]["rgb"],
                        (cAi, bbAi, rel_i, areaAi, dAi),
                        (cBi, bbBi, rel_i, areaBi, dBi),
                        (cAj, bbAj, rel_j, areaAj, dAj),
                        (cBj, bbBj, rel_j, areaBj, dBj),
                        nameA, nameB,
                        Ti, Tj,
                        stats=stats
                    )
            oids_i = {oid for (oid,f) in size_obs.keys() if f==fi}
            oids_j = {oid for (oid,f) in size_obs.keys() if f==fj}
            common_oids = oids_i & oids_j
            for oid in common_oids:
                area_i, cpi, bbi = size_obs[(oid,fi)]
                area_j, cpj, bbj = size_obs[(oid,fj)]
                if area_i > LARGER_RATIO_MIN * max(area_j,1e-6):
                    nameO = obj_id_to_name.get(oid, f"id{oid}")
                    pair_id = f"{fi}_to_{fj}__{sanitize(nameO)}"
                    out_dir = os.path.join(out_root, PAIR_FOLDER, "size", ep_room, pair_id); ensure_dir(os.path.join(out_dir,"rgb"))
                    ann_i = draw_markers_rgb(per_frame_facts[fi]["rgb"], cpi, None, A_bbox=bbi, B_bbox=None, A_name=nameO, B_name=None)
                    ann_j = draw_markers_rgb(per_frame_facts[fj]["rgb"], cpj, None, A_bbox=bbj, B_bbox=None, A_name=nameO, B_name=None)
                    imageio.v2.imwrite(os.path.join(out_dir, "rgb", "1.jpg"), ann_i)
                    imageio.v2.imwrite(os.path.join(out_dir, "rgb", "2.jpg"), ann_j)
                    actions, forward_d = plan_actions(Ti, Tj)
                    rec = {
                        "episode_room": ep_room,
                        "frames": [int(fi), int(fj)],
                        "rgb_folder": os.path.join(PAIR_FOLDER, "size", ep_room, pair_id, "rgb"),
                        "relation_type": "size",
                        "object": nameO,
                        "frame1_relation": "A-larger",
                        "frame2_relation": "A-smaller",
                        "path_actions": actions,
                        "forward_distance_m": float(forward_d),
                        "area_ratio_1_over_2": float(area_i/max(area_j,1e-6)),
                    }
                    append_jsonl(os.path.join(out_root, PAIR_FOLDER, "size", "index.jsonl"), rec)
                    if stats is not None:
                        stats["size_pairs"] += 1
                elif area_j > LARGER_RATIO_MIN * max(area_i,1e-6):
                    nameO = obj_id_to_name.get(oid, f"id{oid}")
                    pair_id = f"{fi}_to_{fj}__{sanitize(nameO)}"
                    out_dir = os.path.join(out_root, PAIR_FOLDER, "size", ep_room, pair_id); ensure_dir(os.path.join(out_dir,"rgb"))
                    ann_i = draw_markers_rgb(per_frame_facts[fi]["rgb"], cpi, None, A_bbox=bbi, B_bbox=None, A_name=nameO, B_name=None)
                    ann_j = draw_markers_rgb(per_frame_facts[fj]["rgb"], cpj, None, A_bbox=bbj, B_bbox=None, A_name=nameO, B_name=None)
                    imageio.v2.imwrite(os.path.join(out_dir, "rgb", "1.jpg"), ann_i)
                    imageio.v2.imwrite(os.path.join(out_dir, "rgb", "2.jpg"), ann_j)
                    actions, forward_d = plan_actions(Ti, Tj)
                    rec = {
                        "episode_room": ep_room,
                        "frames": [int(fi), int(fj)],
                        "rgb_folder": os.path.join(PAIR_FOLDER, "size", ep_room, pair_id, "rgb"),
                        "relation_type": "size",
                        "object": nameO,
                        "frame1_relation": "A-smaller",
                        "frame2_relation": "A-larger",
                        "path_actions": actions,
                        "forward_distance_m": float(forward_d),
                        "area_ratio_2_over_1": float(area_j/max(area_i,1e-6)),
                    }
                    append_jsonl(os.path.join(out_root, PAIR_FOLDER, "size", "index.jsonl"), rec)
                    if stats is not None:
                        stats["size_pairs"] += 1

def sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\-]+", "_", name).strip("_")

def generate_relationship_dataset(data_root, output_root):
    stats = {
        "rooms_processed": 0,
        "episodes_processed": 0,
        "frames_seen": 0,
        "direction_pairs": 0,
        "direction_left": 0,
        "direction_right": 0,
        "direction_front": 0,
        "direction_back": 0,
        "size_pairs": 0,
        "distance_pairs": 0,
    }
    ensure_dir(output_root)
    episodes = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    episodes.sort()
    print(f"[START] Scanning dataset root: {data_root}")
    for episode_id in episodes:
        episode_path = os.path.join(data_root, episode_id, AGENT_NAME)
        if not os.path.isdir(episode_path):
            continue
        room_names = [
            name for name in os.listdir(episode_path)
            if os.path.isdir(os.path.join(episode_path, name)) and "unknown" not in name
        ]
        room_names.sort()
        print(f"Episode: {episode_id} — rooms: {len(room_names)}")
        for room_name in room_names:
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
            per_frame_facts = {}
            for frame_idx in frames:
                pose_path = os.path.join(room_path, "pose", f"{frame_idx}.npy")
                T_wc_inv = np.linalg.inv(np.load(pose_path))
                facts = collect_frame_facts(
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
                per_frame_facts[frame_idx] = facts
                stats["frames_seen"] += 1
                pbar.update(1)
            pbar.close()
            build_pairs_for_room(
                episode_id=episode_id,
                room_name=room_name,
                frames=frames,
                per_frame_facts=per_frame_facts,
                obj_id_to_name=obj_id_to_name,
                out_root=output_root,
                stats=stats
            )
            print(f"[{episode_id}/{room_name}] done {len(frames)} frames in {time.time()-start_ts:.1f}s")
            stats["rooms_processed"] += 1
        print(f"Finished episode {episode_id}")
        stats["episodes_processed"] += 1
    print("\n[SUMMARY]")
    print(f"Episodes processed: {stats['episodes_processed']}")
    print(f"Rooms processed: {stats['rooms_processed']}")
    print(f"Frames seen: {stats['frames_seen']}")
    print("Pairs:")
    print(f"Direction (total): {stats['direction_pairs']}")
    print(f"left: {stats['direction_left']}")
    print(f"right: {stats['direction_right']}")
    print(f"front: {stats['direction_front']}")
    print(f"back: {stats['direction_back']}")
    print(f"Larger (size) pairs: {stats['size_pairs']}")
    print(f"Closer (distance) pairs: {stats['distance_pairs']}")
    print(f"[DONE] Pair results written to: {output_root}/{PAIR_FOLDER}")

if __name__ == "__main__":
    input_dataset_path = "/home/zheyuanzhang/Documents/GitHub/habelief/data/trajectories/test"
    output_dataset_path = "/home/zheyuanzhang/Documents/GitHub/habelief/data/trajectories/relationships"
    generate_relationship_dataset(input_dataset_path, output_dataset_path)