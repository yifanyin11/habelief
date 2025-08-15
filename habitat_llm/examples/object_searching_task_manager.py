import sys
import os
import random
import pathlib
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import imageio
import re
from pathlib import Path
from copy import deepcopy

ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
# append the path of the
# parent directory
sys.path.append("..")
from typing import Any, Dict
from omegaconf import DictConfig, OmegaConf
from hydra import initialize_config_dir, compose
from habitat.tasks.utils import get_angle

from habitat_llm.utils import cprint, setup_config

from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_sensors,
)

from habitat_llm.evaluation import (
    CentralizedEvaluationRunner,
)
from habitat_llm.world_model import Room
from habitat_llm.utils.core import get_config
from habitat_llm.agent.env.dataset import CollaborationDatasetV0

def parse_episode_path(path_str: str) -> Dict[str, Any]:
    """
    Extract:
      - episode_path: the input path itself
      - episode_name: 'epidx_<id>_scene_<scene_number>'
      - scene_number: '<digits>' or '<digits>_<digits>'
      - room_name:    '<word_word_...>_<digit>'
    """
    EPISODE_RE = re.compile(r'^(?P<episode_name>epidx_\d+_scene_(?P<scene_number>\d+(?:_\d+)*))$')
    ROOM_RE = re.compile(r'^(?P<room_name>[A-Za-z]+(?:_[A-Za-z]+)*_\d+)$')
    FURN_RE = re.compile(r'^.+_\d+-\d+$')

    p = Path(path_str).resolve()
    parts = list(p.parts)

    # Find episode_name & scene_number
    episode_name = None
    scene_number = None
    for part in parts:
        m = EPISODE_RE.match(part)
        if m:
            episode_name = m.group('episode_name')
            scene_number = m.group('scene_number')
            break
    if episode_name is None:
        raise ValueError(f"Could not find episode_name in path: {path_str}")

    # Find room_name:
    room_name = None
    if parts:
        last = parts[-1]
        if FURN_RE.match(last) and len(parts) >= 2:
            candidate = parts[-2]
            if ROOM_RE.match(candidate):
                room_name = candidate

    if room_name is None:
        # Fallback: scan all parts to find a room-looking segment
        for part in parts:
            m = ROOM_RE.match(part)
            if m:
                room_name = m.group('room_name')
                break

    if room_name is None:
        raise ValueError(f"Could not find room_name in path: {path_str}")

    return {
        "episode_path": str(p),
        "episode_name": episode_name,
        "scene_number": scene_number,
        "room_name":    room_name,
    }

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

def get_instrinsic_matrix(intrinsics):
    fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]
    intrinsics_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    return intrinsics_matrix

def is_obj_in_view(obj_id, depth, intrinsics, extrinsics, all_bboxes):
    """
    Check if the object is in the view of the camera.
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

def extract_obs(env_interface: EnvironmentInterface, obs: Dict[str, Any]):

    curr_agent, camera_source = env_interface.trajectory_agent_names[0], env_interface.conf.trajectory.camera_prefixes[0]

    if env_interface._single_agent_mode:
        rgb = obs[f"{camera_source}_rgb"]
        depth = obs[f"{camera_source}_depth"]
        pose = np.linalg.inv(
            env_interface.sim.agents[0]
            ._sensors[f"{camera_source}_rgb"]
            .render_camera.camera_matrix
        )
    else:
        rgb = obs[f"{curr_agent}_{camera_source}_rgb"]
        depth = obs[f"{curr_agent}_{camera_source}_depth"]
        pose = np.linalg.inv(
            env_interface.sim.agents[0]
            ._sensors[f"{curr_agent}_{camera_source}_rgb"]
            .render_camera.camera_matrix
        )
    
    extracted_obs = {
        "rgb": rgb,
        "depth": depth,
        "pose": pose
    }

    return extracted_obs

class ObjectSearchingTaskManager:
    def __init__(self, config: DictConfig, env_interface: EnvironmentInterface, dataset: CollaborationDatasetV0, eval_runner: CentralizedEvaluationRunner):
        self.config = config
        self.habelief_episode_root = config.habelief_episode_root
        self.close_enough_distance = config.close_enough_distance
        self.face_to_angle_threshold = config.face_to_angle_threshold
        self.agent_id = config.agent_id
        self.episode_list = sorted([p for p in Path(self.habelief_episode_root).glob("*/agent_1/*/*/") if p.is_dir()])
        self.episodes = [parse_episode_path(episode_path) for episode_path in self.episode_list]
        self.env_interface = env_interface
        self.eval_runner = eval_runner
        self.dataset = dataset
        self.current_episode_index = -1
        self.target_obj = None
        self.target_obj_name = None
        self.target_obj_id = None
        self.episode_path = None
        self.episode_name = None
        self.scene_number = None
        self.room_name = None
        self.all_bboxes = None
        self.last_position = None

    def reset(self, idx=None):
        # Extract the next episode
        if idx is not None:
            if idx < 0 or idx >= len(self.episodes):
                raise IndexError(f"Index {idx} is out of bounds for episodes list.")
            self.current_episode_index = idx
        else:
            self.current_episode_index += 1
        if self.current_episode_index >= len(self.episodes):
            cprint("No more episodes to process.", "red")
            return
        # Extract agent starting pose and target object
        episode_info = self.episodes[self.current_episode_index]
        self.episode_path = episode_info["episode_path"]
        print(f"Resetting to episode: {self.episode_path}")
        self.episode_name = episode_info["episode_name"]
        self.scene_number = episode_info["scene_number"]
        room_name = episode_info["room_name"]
        self.room_name = room_name

        pose_path = os.path.join(self.episode_path, "pose")
        start_pose_file = sorted(os.listdir(pose_path))[0]
        start_pose_path = os.path.join(pose_path, start_pose_file)
        start_pose = np.load(start_pose_path, allow_pickle=True)
        target_obj_path = os.path.join(self.episode_path, "entity_desc.txt")
        # First line is the target object, second line is the target object id
        with open(target_obj_path, "r") as file:
            lines = file.readlines()
            if len(lines) < 2:
                raise ValueError(f"Invalid target object file: {target_obj_path}")
            self.target_obj = lines[0].strip()
            self.target_obj_name = lines[1].strip()
        # Extract intrinsics, all_bbox, all_obj_id 
        intrinsics = np.load(os.path.join(self.habelief_episode_root, self.episode_name, "agent_1", "intrinsics.npy"), allow_pickle=True)[0]
        self.intrinsics = get_instrinsic_matrix(intrinsics)

        self.all_bboxes = np.load(os.path.join(self.habelief_episode_root, self.episode_name, "agent_1", "all_bb.npy"), allow_pickle=True).item()
        object_id_to_handle = np.load(os.path.join(self.habelief_episode_root, self.episode_name, "agent_1", "object_id_to_handle.npy"), allow_pickle=True).item()
        object_handle_to_id = {v: k for k, v in object_id_to_handle.items()}
        ao_id_to_handle = np.load(os.path.join(self.habelief_episode_root, self.episode_name, "agent_1", "ao_id_to_handle.npy"), allow_pickle=True).item()
        ao_handle_to_id = {v: k for k, v in ao_id_to_handle.items()}
        world_graph = np.load(os.path.join(self.episode_path, "world_graph", "0.npy"), allow_pickle=True).item()
        all_furns = world_graph.get_all_furnitures()

        ao_furns = [furn for furn in all_furns if furn.sim_handle in ao_handle_to_id]
        furns = [furn for furn in all_furns if furn.sim_handle in object_handle_to_id]

        object_id_to_name = {
            obj_id: next((obj.name for obj in furns if obj.sim_handle == handle), None)
            for obj_id, handle in object_id_to_handle.items()
        }
        object_id_to_name = {k: v for k, v in object_id_to_name.items() if v is not None}
        object_name_to_id = {v: k for k, v in object_id_to_name.items()}
        ao_id_to_name = {
            obj_id: next((obj.name for obj in ao_furns if obj.sim_handle == handle), None)
            for obj_id, handle in ao_id_to_handle.items()
        }
        ao_id_to_name = {k: v for k, v in ao_id_to_name.items() if v is not None}
        ao_name_to_id = {v: k for k, v in ao_id_to_name.items()}
        all_obj_name_to_id = {**object_name_to_id, **ao_name_to_id}

        # Extract target object id
        self.target_obj_id = all_obj_name_to_id.get(self.target_obj_name, None)
        if self.target_obj_id is None:
            raise ValueError(f"Target object {self.target_obj_name} not found in the dataset.")

        # Set the next dataset episode
        episode_id = self.dataset.get_scene_episodes(self.scene_number)[0].episode_id
        self.env_interface.env.habitat_env.episode_iterator.set_next_episode_by_id(
            str(episode_id)
        )
        # Reset the environment
        self.env_interface.reset_environment()
        self.eval_runner.reset()
        observations = self.env_interface.get_observations()
        # Set agent to the starting position
        hl_action_name = "NavigatePose"
        hl_action_input = (start_pose[:3, 3], True, False, (0.05, 0.05))
        hl_action_done = False
        print(f"Resetting the environment to the starting pose: {hl_action_input}")
        while not hl_action_done:
            low_level_action, response = self.eval_runner.planner.agents[
                self.agent_id
            ].process_high_level_action(
                hl_action_name, hl_action_input, observations
            )
            low_level_action = {self.agent_id: low_level_action}

            obs, _, _, _ = self.env_interface.step(
                low_level_action
            )
            observations = self.env_interface.parse_observations(obs)
            
            if response:
                print(f"\tResponse: {response}")
                hl_action_done = True

        if not response:
            raise RuntimeError("Failed to reset the environment to the starting pose.")
        
        R = start_pose[:3, :3]
        t = start_pose[:3, 3]
        z_forward = R[:, 2]
        t_new = t + 0.1 * z_forward
        new_pose = deepcopy(start_pose)
        new_pose[:3, 3] = t_new

        hl_action_name = "NavigatePose"
        hl_action_input = (new_pose[:3, 3], True, False, (0.05, 0.05))
        hl_action_done = False
        while not hl_action_done:
            low_level_action, response = self.eval_runner.planner.agents[
                self.agent_id
            ].process_high_level_action(
                hl_action_name, hl_action_input, observations
            )
            low_level_action = {self.agent_id: low_level_action}

            obs, _, _, _ = self.env_interface.step(
                low_level_action
            )
            observations = self.env_interface.parse_observations(obs)
            
            if response:
                print(f"\tResponse: {response}")
                hl_action_done = True

        if not response:
            raise RuntimeError("Failed to reset the environment to the starting pose.")
        
        # Update the last position
        self.last_position = new_pose[:3, 3].copy()
        
        return obs

    def set_last_position(self, position):
        if isinstance(position, np.ndarray):
            self.last_position = position.copy()
        else:
            raise ValueError("Position must be a numpy array.")

    def is_done(self) -> bool:
        # Get the agent's position
        base_T = self.env_interface.sim.agents_mgr[self.agent_id].articulated_agent.base_transformation
        agent_position = np.array(base_T[3])[:3]
        # Get the position of the target object
        target_obj_T = self.all_bboxes[self.target_obj_id][1]
        target_obj_position = np.array(target_obj_T[3])[:3]
        distance = np.linalg.norm((target_obj_position - agent_position)[[0, 2]])
        # Check if the target object is close enough
        close_enough = False
        if distance < self.close_enough_distance:
            cprint(f"Target object {self.target_obj_name} is in view and close enough.", "green")
            close_enough = True
        
        return close_enough