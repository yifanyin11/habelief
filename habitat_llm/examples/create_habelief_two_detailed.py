import os
import numpy as np
import shutil
import random
import cv2
import imageio
from math import pi, acos
from numpy.linalg import norm
import sys
import re
import pathlib
from habitat.sims.habitat_simulator.sim_utilities import (
    get_obj_from_handle,
    get_obj_from_id,
    get_bb_for_object_id,
)
from habitat_llm.perception.perception_sim import (
    HUMAN_SEMANTIC_ID, UNKNOWN_SEMANTIC_ID, 
    compute_2d_bbox_from_aabb
)

from habitat_llm.utils import setup_config

from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_sensors,
)

from habitat_llm.utils.core import get_config
from habitat_llm.agent.env.dataset import CollaborationDatasetV0

ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# Constants
ANGLE_THRESHOLD = {"on the left": pi / 4, "on the right": pi / 4, "on the back of self": pi / 4, "on the back of ref": pi / 4}
AGENT_ENTITY_DISTANCE_THRESHOLD = 3.0
NEAR_DISTANCE_THRESHOLD = 3.0
MAX_CLIPS_PER_COMBINATION = 5
AGENT_NAME = "agent_1"
combination_control = {}

def calculate_angle(v1, v2):
    """Calculate the angle between two vectors."""
    cos_theta = np.dot(v1, v2) / (norm(v1) * norm(v2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Avoid numerical errors
    return acos(cos_theta)

def load_world_graph(world_graph_path):
    """Load the world graph."""
    return np.load(world_graph_path, allow_pickle=True).item()  # Load as dictionary

def calculate_distance(pos1, pos2):
    return norm(np.array(pos1) - np.array(pos2))

def clean_text(text):
    # remove "_" and all numbers
    return ''.join([char for char in text if not char.isdigit()]).replace("_", " ").strip()

def get_instrinsic_matrix(intrinsics):
    fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]
    intrinsics_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    return intrinsics_matrix

def recover_scene_id_and_room_name(room_path):
    # Extract the last two components from the path
    data_root, episode_id, agent, room_name = room_path.rsplit(os.sep, 3)
    
    # Extract scene_id from episode_id
    match = re.search(r'epidx_\d+_scene_(.+)', episode_id)
    if match:
        scene_id = match.group(1)
    else:
        raise ValueError("Invalid episode_id format in the path.")
    
    return scene_id, room_name

def pass_combination_control(room_path, relation, obj1, obj2):
    """Check if the combination is allowed to be processed."""
    obj_name1 = obj1.name
    obj_name2 = obj2.name
    scene_id, room_name = recover_scene_id_and_room_name(room_path)
    key = f"{scene_id}_{room_name}_{obj_name1}_{obj_name2}_{relation}"
    if key not in combination_control:
        combination_control[key] = 0
    if combination_control[key] > MAX_CLIPS_PER_COMBINATION:
        return False
    combination_control[key] += 1
    return True

def parse_world_description(file_path):
    """Parses a world description file."""
    with open(file_path, 'r') as file:
        data = file.readlines()

    furniture = {}
    objects = {}

    current_category = None
    for line in data:
        line = line.strip()
        if line.startswith("Furniture:"):
            current_category = "furniture"
        elif line.startswith("Objects:"):
            current_category = "objects"
        elif current_category == "furniture" and ":" in line:
            room, items = line.split(":")
            furniture[room.strip()] = [item.strip() for item in items.split(",")]
        elif current_category == "objects" and ":" in line:
            obj, loc = line.split(":")
            objects[obj.strip()] = loc.strip()
    return furniture, objects

def construct_segment_data(raw_ctxt_frames, raw_trgt_frames, ref_entity, target_entity, room_path, ctxt_ref_upper_thred=0.3, ctxt_trgt_upper_thred=0.001, trgt_trgt_lower_thred=0.3):
    """
    Construct segment data for 'ctxt' and 'trgt' frames based on parsed world description and proximity check.
    """
    ctxt_frames = []
    trgt_frames = []
    # Apply filters to the ctxt and trgt frames
    panoptic_path = os.path.join(room_path, "panoptic")
    depth_path = os.path.join(room_path, "depth")
    # Read intrinsics
    intrinsics = np.load(os.path.join(room_path, "..", "intrinsics.npy"), allow_pickle=True)[0]
    intrinsics_array = get_instrinsic_matrix(intrinsics)
    # Load all furniture and object lists
    all_objects_files = sorted(os.listdir(os.path.join(room_path, "all_objects")))
    all_furnitures_files = sorted(os.listdir(os.path.join(room_path, "all_furnitures")))
    all_objects_path = os.path.join(room_path, "all_objects", all_objects_files[0])
    all_furnitures_path = os.path.join(room_path, "all_furnitures", all_furnitures_files[0])
    all_objects = np.load(all_objects_path, allow_pickle=True)
    all_furnitures = np.load(all_furnitures_path, allow_pickle=True)
    all_entities = list(all_objects) + list(all_furnitures)
    # Load object_id_to_handle
    object_id_to_handle = np.load(os.path.join(room_path, "..", "object_id_to_handle.npy"), allow_pickle=True).item()
    object_handle_to_id = {v: k for k, v in object_id_to_handle.items()}
    # Load ao_id_to_handle
    ao_id_to_handle = np.load(os.path.join(room_path, "..", "ao_id_to_handle.npy"), allow_pickle=True).item()
    ao_handle_to_id = {v: k for k, v in ao_id_to_handle.items()}
    # Obtain object_id to name mapping
    object_id_to_name = {
        obj_id: next((obj.name for obj in all_entities if obj.sim_handle == handle), None)
        for obj_id, handle in object_id_to_handle.items()
    }
    object_id_to_name = {k: v for k, v in object_id_to_name.items() if v is not None}
    # Load all bounding boxes
    all_bboxes = np.load(os.path.join(room_path, "..", "all_bb.npy"), allow_pickle=True).item()
    # Extract ref and trgt entity names
    ref_name = ref_entity.name
    target_name = target_entity.name
    # Filter out all ctxt images in which the target entity is present
    for i in range(len(raw_ctxt_frames)):
        world_graph_path = os.path.join(room_path, "world_graph", f"{raw_ctxt_frames[i]}.npy")
        world_graph = np.load(world_graph_path, allow_pickle=True).item()
        # Read all images using imageio
        ctxt_panoptic = imageio.v2.imread(os.path.join(panoptic_path, f"{raw_ctxt_frames[i]}.png"))
        # Read depth using np.load
        ctxt_depth = np.load(os.path.join(depth_path, f"{raw_ctxt_frames[i]}.npy"))
        # Grab the agent's camera pose
        extrinsics = np.linalg.inv(np.load(os.path.join(room_path, "pose", f"{raw_ctxt_frames[i]}.npy")))
        # Check if the target entity is present in the image
        unique_obj_ids = np.unique(ctxt_panoptic)
        unique_obj_ids = [
            idx - 100 for idx in unique_obj_ids if idx != UNKNOWN_SEMANTIC_ID
        ]
        # Get articulated furniture in the room
        furns_in_room = world_graph.get_furniture_in_room(world_graph.get_room_for_entity(world_graph.get_human()))
        furns_in_room = [furn for furn in furns_in_room if furn.sim_handle in ao_handle_to_id]
        extra_furn_ids = [ao_handle_to_id[furn.sim_handle] for furn in furns_in_room]
        unique_obj_ids += extra_furn_ids
        # Check if other entities with the same types as the ref and trgt are in the image
        obj_names = [object_id_to_name[one_obj_id] for one_obj_id in unique_obj_ids if one_obj_id in object_id_to_name]
        obj_clean_names = [clean_text(obj_name) for obj_name in obj_names]
        if ref_name not in obj_names:
            continue
        if obj_clean_names.count(clean_text(ref_name)) > 1:
            continue
        else:
            if target_name in obj_names and obj_clean_names.count(clean_text(target_name)) > 1:
                continue
            elif target_name not in obj_names and obj_clean_names.count(clean_text(target_name)) > 0:
                continue
        # make sure the ref entity is in the image
        obj_ids = [one_obj_id for one_obj_id in unique_obj_ids if object_id_to_handle[one_obj_id] == ref_entity.sim_handle]
        if len(obj_ids)!=0:
            obj_id = obj_ids[0]
        else:
            continue
        if obj_id in extra_furn_ids:
            in_view = is_ao_in_view(obj_id, ctxt_depth, intrinsics_array, extrinsics, all_bboxes)
            if not in_view:
                continue
        bbox_ratio = get_bbox_ratio(obj_id, intrinsics_array, extrinsics, all_bboxes)
        if bbox_ratio < ctxt_ref_upper_thred:
            continue
        # make sure the target entity is not in the image
        obj_ids = [one_obj_id for one_obj_id in unique_obj_ids if object_id_to_handle[one_obj_id] == target_entity.sim_handle]
        if len(obj_ids)!=0:
            obj_id = obj_ids[0]
        else:
            ctxt_frames.append(raw_ctxt_frames[i])
            continue
        # if obj_id in extra_furn_ids:
        #     in_view = is_ao_in_view(obj_id, ctxt_depth, intrinsics_array, extrinsics, all_bboxes)
        #     if in_view:
        #         continue
        bbox_ratio = get_bbox_ratio(obj_id, intrinsics_array, extrinsics, all_bboxes)
        if bbox_ratio < ctxt_trgt_upper_thred:
            ctxt_frames.append(raw_ctxt_frames[i])
    
    # Filter out all trgt images in which the target entity is not present
    for i in range(len(raw_trgt_frames)):
        world_graph_path = os.path.join(room_path, "world_graph", f"{raw_trgt_frames[i]}.npy")
        world_graph = np.load(world_graph_path, allow_pickle=True).item()
        # Read all images using imageio
        trgt_panoptic = imageio.v2.imread(os.path.join(panoptic_path, f"{raw_trgt_frames[i]}.png"))
        # Read depth using np.load
        trgt_depth = np.load(os.path.join(depth_path, f"{raw_trgt_frames[i]}.npy"))
        # Grab the agent's camera pose
        extrinsics = np.linalg.inv(np.load(os.path.join(room_path, "pose", f"{raw_trgt_frames[i]}.npy")))
        # Check if the target entity is present in the image
        unique_obj_ids = np.unique(trgt_panoptic)
        unique_obj_ids = [
            idx - 100 for idx in unique_obj_ids if idx != UNKNOWN_SEMANTIC_ID
        ]
        # Get articulated furniture in the room
        furns_in_room = world_graph.get_furniture_in_room(world_graph.get_room_for_entity(world_graph.get_human()))
        furns_in_room = [furn for furn in furns_in_room if furn.sim_handle in ao_handle_to_id]
        extra_furn_ids = [ao_handle_to_id[furn.sim_handle] for furn in furns_in_room]
        unique_obj_ids += extra_furn_ids
        # Check if other entities with the same types as the ref and trgt are in the image
        obj_names = [object_id_to_name[one_obj_id] for one_obj_id in unique_obj_ids if one_obj_id in object_id_to_name]
        obj_clean_names = [clean_text(obj_name) for obj_name in obj_names]

        if target_name not in obj_names:
            continue
        if obj_clean_names.count(clean_text(target_name)) > 1:
            continue
        else:
            if ref_name in obj_names and obj_clean_names.count(clean_text(ref_name)) > 1:
                continue
            elif ref_name not in obj_names and obj_clean_names.count(clean_text(ref_name)) > 0:
                continue

        # make sure the trgt entity is in the image
        obj_ids = [one_obj_id for one_obj_id in unique_obj_ids if object_id_to_handle[one_obj_id] == target_entity.sim_handle]
        if len(obj_ids)==0:
            continue
        obj_id = obj_ids[0]
        if obj_id in extra_furn_ids:
            in_view = is_ao_in_view(obj_id, trgt_depth, intrinsics_array, extrinsics, all_bboxes)
            if not in_view:
                continue
        bbox_ratio = get_bbox_ratio(obj_id, intrinsics_array, extrinsics, all_bboxes)
        
        if ref_name in obj_names and len(obj_names)==2 and bbox_ratio > 1/2:    
            trgt_frames.append(raw_trgt_frames[i])
            continue
        if bbox_ratio > trgt_trgt_lower_thred:
            trgt_frames.append(raw_trgt_frames[i])
    
    # Collect file paths for each modality
    segment_data = {
        "ctxt": {"frames": ctxt_frames, "data": []},
        "trgt": {"frames": trgt_frames, "data": []},
    }

    for key in ["ctxt", "trgt"]:
        modality_data = []
        for frame in segment_data[key]["frames"]:
            frame_data = {}
            for modality in ["rgb", "pose"]:
                modality_dir = os.path.join(room_path, modality)
                if modality == "rgb":
                    frame_data[f"{modality}_jpg"] = os.path.join(modality_dir, f"{frame}.jpg")
                elif modality == "pose":
                    frame_data[f"{modality}_npy"] = os.path.join(modality_dir, f"{frame}.npy")
            modality_data.append(frame_data)
        segment_data[key]["data"] = modality_data

    return segment_data

def save_segment(segment_dir, segment_data, ref_name, target_name, relationship):
    """
    Save a segment to the output directory in the specified structure, 
    only if both ctxt and trgt have at least one step.
    """
    # Ensure both ctxt and trgt have at least one frame
    if not segment_data["ctxt"]["frames"] or not segment_data["trgt"]["frames"]:
        print(f"Skipping segment {segment_dir}: Missing frames in ctxt or trgt.")
        return False # Skip saving if any list is empty

    os.makedirs(segment_dir, exist_ok=True)
    
    # Create ctxt and trgt directories
    ctxt_dir = os.path.join(segment_dir, "ctxt")
    trgt_dir = os.path.join(segment_dir, "trgt")
    os.makedirs(os.path.join(ctxt_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(ctxt_dir, "pose"), exist_ok=True)
    os.makedirs(os.path.join(trgt_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(trgt_dir, "pose"), exist_ok=True)

    # Copy files for ctxt
    for frame_data in segment_data["ctxt"]["data"]:
        for modality, file_path in frame_data.items():
            modality_type, ext = modality.rsplit("_", 1)
            save_path = os.path.join(ctxt_dir, modality_type, f"{os.path.basename(file_path)}")
            shutil.copy(file_path, save_path)

    # Copy files for trgt
    for frame_data in segment_data["trgt"]["data"]:
        for modality, file_path in frame_data.items():
            modality_type, ext = modality.rsplit("_", 1)
            save_path = os.path.join(trgt_dir, modality_type, f"{os.path.basename(file_path)}")
            shutil.copy(file_path, save_path)

    # Save entity description
    target_name = clean_text(target_name)
    ref_name = clean_text(ref_name)
    desc_file = os.path.join(segment_dir, "entity_desc.txt")
    if target_name==ref_name:
        description = f"There is another {target_name} {relationship} the {ref_name}."
    else:
        description = f"A {target_name} is {relationship} the {ref_name}."
    with open(desc_file, "w") as f:
        f.write(description)
    return True

def find_depth_value(points_3d, intrinsics, extrinsics):
    """Projects 3D points to the 2D image plane, conditionally flipping Z values."""
    # Transform points from world to camera coordinates
    points_camera = (extrinsics @ (np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))).T).T
    points_camera = points_camera[points_camera[:, 2] < 0]  # Filter out points behind the camera
    if points_camera.shape[0] == 0:
        return None, None
    z_values = -points_camera[:, 2]  # Extract Z values
    points_image = (intrinsics @ points_camera[:, :3].T).T  # Apply intrinsics
    points_image = points_image[:, :2] / points_camera[:, 2:3]  # Normalize by depth (Z)
    # x=width-x
    points_image[:, 0] = intrinsics[0, 2] * 2 - points_image[:, 0]
    return z_values, points_image

def is_ao_in_view(obj_id, depth, intrinsics, extrinsics, all_bboxes):
    """
    Check if the articulated object is in the view of the camera.
    """
    local_aabb, global_transform = all_bboxes[obj_id]
    # Get corners of the local AABB
    corners_local = np.array([
        np.array(local_aabb.front_bottom_left),
        np.array(local_aabb.front_bottom_right),
        np.array(local_aabb.front_top_left),
        np.array(local_aabb.front_top_right),
        np.array(local_aabb.back_bottom_left),
        np.array(local_aabb.back_bottom_right),
        np.array(local_aabb.back_top_left),
        np.array(local_aabb.back_top_right),
    ])
    # Transform corners to global coordinates
    corners_global = (np.array(global_transform) @ (np.hstack((corners_local, np.ones((corners_local.shape[0], 1))))).T).T[:, :3]
    depth_values, points_image = find_depth_value(corners_global, intrinsics, extrinsics)
    if depth_values is None:
        return False
    num_points = points_image.shape[0]
    thred = num_points // 2
    depth_min = np.min(depth_values)
    depth_max = np.max(depth_values)
    # For each point in points_image, check is there any point +-5 pixels around it (inclusive) has depth value within the range of depth_min and depth_max
    valid_points = 0  # Counter for valid points
    # Check each projected point in image space
    for i in range(num_points):
        u, v = int(points_image[i, 0]), int(points_image[i, 1])  # Pixel coordinates
        # Ensure pixel coordinates are within image bounds
        if u < 0 or v < 0 or u >= depth.shape[1] or v >= depth.shape[0]:
            continue
        # Check surrounding pixels (±2 pixels) for valid depth values
        for du in range(-2, 3):
            for dv in range(-2, 3):
                u_neighbor = u + du
                v_neighbor = v + dv
                # Ensure neighbor is within image bounds
                if u_neighbor < 0 or v_neighbor < 0 or u_neighbor >= depth.shape[1] or v_neighbor >= depth.shape[0]:
                    continue
                # Check if depth is within range
                neighbor_depth = depth[v_neighbor, u_neighbor]
                if depth_min <= neighbor_depth <= depth_max:
                    valid_points += 1
                    break  # Break inner loop if a valid neighbor is found
            else:
                continue  # Continue if inner loop wasn't broken
            break  # Break outer loop if a valid neighbor is found
    # Check if valid points meet the threshold
    return valid_points >= thred

def get_bbox_ratio(
    obj_id,
    intrinsics_array,
    extrinsics,
    all_bboxes,
):
    """
    This method uses the instance segmentation output to
    get the bbox ratio, measuring the visible area of the object.
    """
    local_aabb, global_transform = all_bboxes[obj_id]
    # Compute the 2D bounding box area
    bbox = compute_2d_bbox_from_aabb(local_aabb, np.array(global_transform), np.array(intrinsics_array), np.array(extrinsics))
    if bbox["area"] == np.inf:
        bbox["area"] = 0
    bbox_ratio = bbox["area"]/bbox["large_area"]
    return bbox_ratio

def on_the_left(ref_camera_pose, obj, furniture, angle_thred):
    """Check if object is on the left of furniture using the reference camera pose."""
    obj_pos = np.array(obj.properties["translation"])
    furn_pos = np.array(furniture.properties["translation"])
    ref_pos = ref_camera_pose[:3, 3]  # Extract translation from the pose matrix
    ref_dir = -ref_camera_pose[:3, 0]  # Extract the x-axis (left direction) from the pose matrix

    angle = calculate_angle(obj_pos - furn_pos, ref_dir)

    return angle <= angle_thred

def on_the_right(ref_camera_pose, obj, furniture, angle_thred):
    """Check if object is on the right of furniture using the reference camera pose."""
    obj_pos = np.array(obj.properties["translation"])
    furn_pos = np.array(furniture.properties["translation"])
    ref_pos = ref_camera_pose[:3, 3]  # Extract translation from the pose matrix
    ref_dir = ref_camera_pose[:3, 0]  # Negative x-axis (right direction)

    angle = calculate_angle(obj_pos - furn_pos, ref_dir)

    return angle <= angle_thred

def on_the_back_of_self(ref_camera_pose, obj, furniture, angle_thred):
    """Check if object is on the back of the camera frame using the reference camera pose."""
    obj_pos = np.array(obj.properties["translation"])
    ref_pos = ref_camera_pose[:3, 3]  # Extract translation from the pose matrix
    ref_dir = ref_camera_pose[:3, 2]  # Negative z-axis (back direction)

    angle = calculate_angle(obj_pos - ref_pos, ref_dir)

    return angle <= angle_thred

def on_the_back_of_ref(ref_camera_pose, obj, furniture, angle_thred):
    """Check if object is on the back of the furniture using the reference camera pose."""
    obj_pos = np.array(obj.properties["translation"])
    furn_pos = np.array(furniture.properties["translation"])
    ref_pos = ref_camera_pose[:3, 3]  # Extract translation from the pose matrix
    ref_dir = -ref_camera_pose[:3, 2]  # Positive z-axis (front direction)

    angle = calculate_angle(obj_pos - furn_pos, ref_dir)

    return angle <= angle_thred

def near(world_graph, obj, furniture, distance_thred=NEAR_DISTANCE_THRESHOLD):
    """Check if object is near furniture in the world graph."""
    """ Using 
    get_closest_object_or_furniture(
        self, obj_node, n: int, dist_threshold: float = NEAR_DISTANCE_THRESHOLD
    )
    """
    obj_name = obj.name  # Access the object's name
    furn_name = furniture.name  # Access the furniture's name

    # Iterate through the keys in world_graph.graph, which are instances
    for graph_obj, graph_furns in world_graph.graph.items():
        if graph_obj.name == obj_name:  # Match object name
            # Get closest object or furniture
            closest_furns = world_graph.get_closest_object_or_furniture(graph_obj, 20, distance_thred)
            # Get names
            closest_furns_names = [furn.name for furn in closest_furns]
            if furn_name in closest_furns_names:
                print(f"Found relationship: near")
                return True
    return False

def check_graph_relationship(world_graph, obj, furn, target_rel):
    """
    Check if the relationship between object and furniture in the world graph matches the target relationship.
    """
    obj_name = obj.name  # Access the object's name
    furn_name = furn.name  # Access the furniture's name

    # Iterate through the keys in world_graph.graph, which are instances
    for graph_obj, graph_furns in world_graph.graph.items():
        if graph_obj.name == obj_name:  # Match object name
            for graph_furn, rel in graph_furns.items():
                if furn_name in graph_furn.name:  # Match furniture name
                    print(f"Found relationship: {rel}")
                    return rel == target_rel  # Check if the relationship matches
    return False

def generate_segments(data_root, new_root):
    """Generate segments based on the updated criteria."""
    # Counter for statistics
    stats = {
        "on": 0,
        "in": 0,
        "inside": 0,
        "near": 0,
        "on the left": 0,
        "on the right": 0,
        "on the back of self": 0,
        "on the back of ref": 0,
    }
    for episode_id in os.listdir(data_root):
        print(f"Processing episode {episode_id}...")
        episode_path = os.path.join(data_root, episode_id, AGENT_NAME)

        if not os.path.isdir(episode_path):
            continue
        
        room_names = [name for name in os.listdir(episode_path) if os.path.isdir(os.path.join(episode_path, name)) and "unknown" not in name]
        for room_name in room_names:
            print(f"\tProcessing room {room_name}...")
            room_path = os.path.join(episode_path, room_name)
            world_desc_path = os.path.join(room_path, "world_desc")
            pose_path = os.path.join(room_path, "pose")
            world_graph_path = os.path.join(room_path, "world_graph.npy")

            all_objects_files = sorted(os.listdir(os.path.join(room_path, "all_objects")))
            all_furnitures_files = sorted(os.listdir(os.path.join(room_path, "all_furnitures")))

            if not all_objects_files or not all_furnitures_files or not os.path.exists(world_graph_path):
                print(f"\t\tMissing required files for room {room_name}.")
                continue

            all_objects_path = os.path.join(room_path, "all_objects", all_objects_files[0])
            all_furnitures_path = os.path.join(room_path, "all_furnitures", all_furnitures_files[0])

            # Load object and furniture metadata
            all_objects = np.load(all_objects_path, allow_pickle=True)
            all_furnitures = np.load(all_furnitures_path, allow_pickle=True)

            # Load the world graph
            world_graph = load_world_graph(world_graph_path)

            # Track discovery steps for objects and furniture
            object_discovery = {}
            furniture_discovery = {}
            desc_files = sorted(
                [f for f in os.listdir(world_desc_path) if f.endswith(".txt")],
                key=lambda x: int(os.path.splitext(x)[0])  # Sort by numeric index
            )
            # Get sorted list of frame indices
            sorted_frame_indices = sorted(int(os.path.splitext(desc_file)[0]) for desc_file in desc_files)
            frame_idx_to_relative = {frame_idx: i for i, frame_idx in enumerate(sorted_frame_indices)}  # Map to relative indices
            num_frames = len(sorted_frame_indices)  # Total number of frames

            for desc_file in desc_files:
                frame_idx = int(os.path.splitext(desc_file)[0])  # Extract numeric index
                relative_idx = frame_idx_to_relative[frame_idx]  # Map to relative index
                current_desc = parse_world_description(os.path.join(world_desc_path, desc_file))

                ##############
                ##  Filter  ##
                ##############
                one_world_graph_path = os.path.join(room_path, "world_graph", f"{frame_idx}.npy")
                one_world_graph = np.load(one_world_graph_path, allow_pickle=True).item()
                if (AGENT_NAME=='main_agent' or AGENT_NAME=='agent_0') and one_world_graph.get_room_for_entity(one_world_graph.get_spot_robot()).name != room_name:
                    continue
                if (AGENT_NAME=='agent_1') and one_world_graph.get_room_for_entity(one_world_graph.get_human()).name != room_name:
                    continue
                ##############
                ##  Filter  ##
                ##############

                # Objects
                for obj in current_desc[1].keys():
                    if "unknown" in obj:
                        continue
                    if obj not in object_discovery:
                        # Initialize an array of zeros with size equal to the number of frames
                        object_discovery[obj] = np.zeros(num_frames, dtype=int)
                    # Mark the relative index as 1
                    object_discovery[obj][relative_idx] = 1

                # Furniture
                if room_name not in current_desc[0]:
                    continue
                furn_names = current_desc[0][room_name]
                for furn in furn_names:
                    if "unknown" in furn:
                        continue
                    if furn not in furniture_discovery:
                        # Initialize an array of zeros with size equal to the number of frames
                        furniture_discovery[furn] = np.zeros(num_frames, dtype=int)
                    # Mark the relative index as 1
                    furniture_discovery[furn][relative_idx] = 1

            # Define all relationships to process
            relationships = [
                ("on", lambda ref_camera_pose, obj, furn: check_graph_relationship(world_graph, obj, furn, "on")),
                ("in", lambda ref_camera_pose, obj, furn: check_graph_relationship(world_graph, obj, furn, "in")),
                ("inside", lambda ref_camera_pose, obj, furn: check_graph_relationship(world_graph, obj, furn, "inside")),
                ("near", lambda ref_camera_pose, entity1, entity2: near(world_graph, entity1, entity2)),
                ("on the left", lambda ref_camera_pose, furn1, furn2: on_the_left(ref_camera_pose, furn1, furn2, ANGLE_THRESHOLD["on the left"])),
                ("on the right", lambda ref_camera_pose, furn1, furn2: on_the_right(ref_camera_pose, furn1, furn2, ANGLE_THRESHOLD["on the right"])),
                ("on the back of self", lambda ref_camera_pose, furn1, furn2: on_the_back_of_self(ref_camera_pose, furn1, furn2, ANGLE_THRESHOLD["on the back of self"])),
                ("on the back of ref", lambda ref_camera_pose, furn1, furn2: on_the_back_of_ref(ref_camera_pose, furn1, furn2, ANGLE_THRESHOLD["on the back of ref"])),
            ]

            ############
            ##  TEMP  ##
            ############
            # # Handle "on", "in", "inside" relationships (Object ↔ Furniture)
            # for furn_name, furn_discovery_steps in furniture_discovery.items():
            #     furn_entity = [furn for furn in all_furnitures if furn.name == furn_name][0]
            #     ##############
            #     ##  Filter  ##
            #     ##############
            #     if world_graph.get_room_for_entity(furn_entity).name != room_name:
            #         continue
            #     ##############
            #     ##  Filter  ##
            #     ##############
            #     # Process each discovered entity based on relationships
            #     for obj_name, obj_discovery_steps in object_discovery.items():
            #         obj_entity = [obj for obj in all_objects if obj.name == obj_name][0]
            #         try:
            #             detected_room_name = world_graph.get_room_for_entity(world_graph.find_furniture_for_object(obj_entity)).name
            #         except:
            #             print(f"Object {obj_name} not found in the world graph.")
            #             continue
            #         if detected_room_name != room_name:
            #             print(f"Object {obj_name} not found in room {room_name}.")
            #             continue
                    
            #         for relationship, func in relationships[:3]:  # on, in, inside
            #             if func(None, obj_entity, furn_entity) and pass_combination_control(room_path, relationship, obj_entity, furn_entity):
            #                 # Save the segment and description
            #                 segment_dir = os.path.join(new_root, episode_id+f"_{room_name}_{furn_name}_{obj_name}_{relationship.replace(' ', '_')}")
            #                 ctxt_relative_indices = np.where((furn_discovery_steps == 1) & (obj_discovery_steps == 0))[0]
            #                 # trgt_relative_indices = np.where((furn_discovery_steps == 1) & (obj_discovery_steps == 1))[0]
            #                 trgt_relative_indices = np.where(obj_discovery_steps == 1)[0]
            #                 ctxt_frames = [sorted_frame_indices[i] for i in ctxt_relative_indices]
            #                 trgt_frames = [sorted_frame_indices[i] for i in trgt_relative_indices]

            #                 segment_data = construct_segment_data(ctxt_frames, trgt_frames, obj_entity, furn_entity, room_path)
            #                 if save_segment(segment_dir, segment_data, furn_name, obj_name, relationship):
            #                     stats[relationship] += 1
            ############
            ##  TEMP  ##
            ############

            ############
            ##  TEMP  ##
            ############
            # # Handle "near" relationships
            # # Objects ↔ Objects
            # for obj_name1, discovery_steps1 in object_discovery.items():
                
            #     obj_entity1 = [obj for obj in all_objects if obj.name == obj_name1][0]
                
            #     try:
            #         detected_room_name = world_graph.get_room_for_entity(world_graph.find_furniture_for_object(obj_entity1)).name
            #     except:
            #         print(f"Object {obj_name1} not found in the world graph.")
            #         continue
            #     if detected_room_name != room_name:
            #         print(f"Object {obj_name1} not found in room {room_name}.")
            #         continue

            #     for obj_name2, discovery_steps2 in object_discovery.items():
                    
            #         if obj_name2 == obj_name1:
            #             continue

            #         obj_entity2 = [obj for obj in all_objects if obj.name == obj_name2][0]
                    
            #         try:
            #             detected_room_name = world_graph.get_room_for_entity(world_graph.find_furniture_for_object(obj_entity2)).name
            #         except:
            #             print(f"Object {obj_name2} not found in the world graph.")
            #             continue
            #         if detected_room_name != room_name:
            #             print(f"Object {obj_name2} not found in room {room_name}.")
            #             continue
                        
            #         if near(world_graph, obj_entity1, obj_entity2) and pass_combination_control(room_path, "near", obj_entity1, obj_entity2):
            #             # Save segment
            #             segment_dir = os.path.join(new_root, episode_id+f"_{room_name}_{obj_name1}_{obj_name2}_near")
            #             ctxt_relative_indices = np.where((discovery_steps1 == 1) & (discovery_steps2 == 0))[0]
            #             # trgt_relative_indices = np.where((discovery_steps1 == 1) & (discovery_steps2 == 1))[0]
            #             trgt_relative_indices = np.where(discovery_steps2 == 1)[0]
            #             ctxt_frames = [sorted_frame_indices[i] for i in ctxt_relative_indices]
            #             trgt_frames = [sorted_frame_indices[i] for i in trgt_relative_indices]
            #             segment_data = construct_segment_data(ctxt_frames, trgt_frames, obj_entity1, obj_entity2, room_path)
            #             if save_segment(segment_dir, segment_data, obj_name1, obj_name2, "near"):
            #                 stats["near"] += 1
            ############
            ##  TEMP  ##
            ############

            # Handle "near", "on the left", "on the right", etc. (Furniture ↔ Furniture)
            for furn_name1, discovery_steps1 in furniture_discovery.items():
                furn_entity1 = [furn for furn in all_furnitures if furn.name == furn_name1][0]
                ##############
                ##  Filter  ##
                ##############
                if world_graph.get_room_for_entity(furn_entity1).name != room_name:
                    continue
                ##############
                ##  Filter  ##
                ##############
                for furn_name2, discovery_steps2 in furniture_discovery.items():
                    ############
                    ##  TEMP  ##
                    ############
                    name1, _ = furn_name1.rsplit("_", 1)
                    name2, _ = furn_name2.rsplit("_", 1)
                    if name1 == name2:
                        continue
                    ############
                    ##  TEMP  ##
                    ############

                    furn_entity2 = [furn for furn in all_furnitures if furn.name == furn_name2][0]
                    ##############
                    ##  Filter  ##
                    ##############
                    if world_graph.get_room_for_entity(furn_entity2).name != room_name:
                        continue
                    ##############
                    ##  Filter  ##
                    ##############
                    ref_camera_pose = np.load(os.path.join(pose_path, f"{sorted_frame_indices[0]}.npy"))  # Load camera pose
                    for relationship, func in relationships[3:]:
                        if func(ref_camera_pose, furn_entity1, furn_entity2) and pass_combination_control(room_path, relationship, furn_entity1, furn_entity2):
                            if relationship != "near":
                                # print(f"Found relationship: {relationship}")
                                ############
                                ##  TEMP  ##
                                ############
                                continue
                                ############
                                ##  TEMP  ##
                                ############
                            # Save segment
                            segment_dir = os.path.join(new_root, episode_id+f"_{room_name}_{furn_name1}_{furn_name2}_{relationship.replace(' ', '_')}")
                            ctxt_relative_indices = np.where((discovery_steps1 == 1) & (discovery_steps2 == 0))[0]
                            # trgt_relative_indices = np.where((discovery_steps1 == 1) & (discovery_steps2 == 1))[0]
                            trgt_relative_indices = np.where(discovery_steps2 == 1)[0]
                            ctxt_frames = [sorted_frame_indices[i] for i in ctxt_relative_indices]
                            trgt_frames = [sorted_frame_indices[i] for i in trgt_relative_indices]
                            segment_data = construct_segment_data(ctxt_frames, trgt_frames, furn_entity1, furn_entity2, room_path)
                            if save_segment(segment_dir, segment_data, furn_name1, furn_name2, relationship):
                                stats[relationship] += 1

    # Print statistics
    print(stats)
    total_segments = sum(stats.values())
    print(f"Total number of segments: {total_segments}")


# Directories
input_dataset_path = "/scratch/tshu2/yyin34/projects/3d_belief/partnr-planner/data/trajectories/habelief/test"
output_dataset_path = "/scratch/tshu2/yyin34/projects/3d_belief/DFM/data/habelief_nl_one/test"

# Generate the segments
generate_segments(input_dataset_path, output_dataset_path)
