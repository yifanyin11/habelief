import os
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

ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
sys.path.append("..")

from habitat_llm.perception.perception_sim import (
    UNKNOWN_SEMANTIC_ID,
    compute_2d_bbox_from_aabb,
)

AGENT_NAME = "agent_1"
VISIBLE_BBOX_RATIO_MIN = 0.06
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
CLOSER_DELTA_MIN = 0.35

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
        "front": abs(theta),
        "right": abs(theta - pi/2),
        "back":  min(abs(theta - pi), abs(theta + pi)),
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
        return

    rgb = imageio.v2.imread(rgb_path)
    H, W = rgb.shape[0], rgb.shape[1]
    pan = imageio.v2.imread(panoptic_path)

    visible_ids = list_visible_entity_ids_for_frame(
        pan, obj_id_to_name, ao_handle_to_id, furn_ids_in_room
    )
    if len(visible_ids) < 2:
        return

    visible_keep = []
    for oid in visible_ids:
        if oid not in all_bboxes:
            continue
        local_aabb, global_T = all_bboxes[oid]
        bbox = compute_2d_bbox_from_aabb(local_aabb, np.array(global_T), np.array(K), np.array(T_wc_inv))
        area_px = 0 if bbox["area"] == np.inf else float(bbox["area"])
        H, W = rgb.shape[0], rgb.shape[1]
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
        pc = world_to_camera(T_wc_inv, center_world)
        depth_forward = max(0.0, -pc[2])

        if ratio >= VISIBLE_BBOX_RATIO_MIN and area_px >= VISIBLE_BBOX_AREA_PX_MIN and depth_forward >= MIN_FORWARD_DEPTH_M:
            visible_keep.append((oid, ratio))

    if len(visible_keep) < 2:
        return

    centers_cam = {}
    depths = {}
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
        centers_cam[oid] = pc
        depths[oid] = max(0.0, -pc[2])

        bbox = compute_2d_bbox_from_aabb(local_aabb, np.array(global_T), np.array(K), np.array(T_wc_inv))
        area = 0.0 if bbox["area"] == np.inf else float(bbox["area"])
        areas[oid] = area
    
    ep_room = f"{episode_id}_{room_name}"
    basename = f"{ep_room}_{frame_idx}"

    for i in range(len(visible_keep)):
        oidA = visible_keep[i][0]
        for j in range(i+1, len(visible_keep)):
            oidB = visible_keep[j][0]

            nameA = obj_id_to_name.get(oidA, f"id{oidA}")
            nameB = obj_id_to_name.get(oidB, f"id{oidB}")

            v = centers_cam[oidA] - centers_cam[oidB]
            card = classify_direction_sector(v[0], v[2])
            if card is not None:
                out_dir = os.path.join(output_root, DIR_FOLDER, card)
                ensure_dir(out_dir)
                ensure_dir(os.path.join(out_dir, "rgb"))

                out_name = f"{basename}__{sanitize(nameA)}__{card}__{sanitize(nameB)}.jpg"
                out_path = os.path.join(out_dir, "rgb", out_name)
                if not os.path.exists(out_path):
                    shutil.copy(rgb_path, out_path)

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
                }
                append_jsonl(os.path.join(out_dir, "index.jsonl"), rec)
                frame_counts[card] += 1


            areaA = areas[oidA]
            areaB = areas[oidB]
            if areaA > LARGER_RATIO_MIN * max(areaB, 1e-6):
                out_dir = os.path.join(output_root, LARGER_FOLDER)
                ensure_dir(out_dir)
                ensure_dir(os.path.join(out_dir, "rgb"))
                out_name = f"{basename}__{sanitize(nameA)}__larger__{sanitize(nameB)}.jpg"
                out_path = os.path.join(out_dir, "rgb", out_name)
                if not os.path.exists(out_path):
                    shutil.copy(rgb_path, out_path)
                rec = {
                    "episode_id": episode_id,
                    "room": room_name,
                    "frame": int(frame_idx),
                    "image": out_name,
                    "relation_type": "larger",
                    "relation": "larger",
                    "subject": nameA,
                    "object": nameB,
                    "area_subject_px": areaA,
                    "area_object_px": areaB,
                    "area_ratio": areaA / max(areaB, 1e-6)
                }
                append_jsonl(os.path.join(out_dir, "index.jsonl"), rec)
                frame_counts["larger"] += 1
            elif areaB > LARGER_RATIO_MIN * max(areaA, 1e-6):
                out_dir = os.path.join(output_root, LARGER_FOLDER)
                ensure_dir(out_dir)
                ensure_dir(os.path.join(out_dir, "rgb"))
                out_name = f"{basename}__{sanitize(nameB)}__larger__{sanitize(nameA)}.jpg"
                out_path = os.path.join(out_dir, "rgb", out_name)
                if not os.path.exists(out_path):
                    shutil.copy(rgb_path, out_path)
                rec = {
                    "episode_id": episode_id,
                    "room": room_name,
                    "frame": int(frame_idx),
                    "image": out_name,
                    "relation_type": "larger",
                    "relation": "larger",
                    "subject": nameB,
                    "object": nameA,
                    "area_subject_px": areaB,
                    "area_object_px": areaA,
                    "area_ratio": areaB / max(areaA, 1e-6)
                }
                append_jsonl(os.path.join(out_dir, "index.jsonl"), rec)
                frame_counts["larger"] += 1

            dA, dB = depths[oidA], depths[oidB]
            if (dA - dB) > CLOSER_DELTA_MIN:
                out_dir = os.path.join(output_root, CLOSER_FOLDER)
                ensure_dir(out_dir)
                ensure_dir(os.path.join(out_dir, "rgb"))
                out_name = f"{basename}__{sanitize(nameA)}__closer__{sanitize(nameB)}.jpg"
                out_path = os.path.join(out_dir, "rgb", out_name)
                if not os.path.exists(out_path):
                    shutil.copy(rgb_path, out_path)
                rec = {
                    "episode_id": episode_id,
                    "room": room_name,
                    "frame": int(frame_idx),
                    "image": out_name,
                    "relation_type": "closer",
                    "relation": "closer",
                    "subject": nameA,
                    "object": nameB,
                    "depth_subject": dA,
                    "depth_object": dB,
                    "depth_delta": dA - dB
                }
                append_jsonl(os.path.join(out_dir, "index.jsonl"), rec)
                frame_counts["closer"] += 1
            elif (dB - dA) > CLOSER_DELTA_MIN:
                out_dir = os.path.join(output_root, CLOSER_FOLDER)
                ensure_dir(out_dir)
                ensure_dir(os.path.join(out_dir, "rgb"))
                out_name = f"{basename}__{sanitize(nameB)}__closer__{sanitize(nameA)}.jpg"
                out_path = os.path.join(out_dir, "rgb", out_name)
                if not os.path.exists(out_path):
                    shutil.copy(rgb_path, out_path)
                rec = {
                    "episode_id": episode_id,
                    "room": room_name,
                    "frame": int(frame_idx),
                    "image": out_name,
                    "relation_type": "closer",
                    "relation": "closer",
                    "subject": nameB,
                    "object": nameA,
                    "depth_subject": dB,
                    "depth_object": dA,
                    "depth_delta": dB - dA
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