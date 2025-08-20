#!/usr/bin/env python3
# isort: skip_file

import sys
import os
import random
import pathlib
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import imageio
from copy import deepcopy
from pathlib import Path
import math
import json
import time
import magnum as mn
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
# append the path of the
# parent directory
sys.path.append("..")
from typing import Any, Dict
from omegaconf import DictConfig, OmegaConf
from hydra import initialize_config_dir, compose

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
from habitat_llm.examples.object_searching_task_manager import ObjectSearchingTaskManager

from agents.perception import object_detection
from agents.vlm_agent import VLMAgent
from pixelbelief.belief_agent import BeliefAgent, prepare_video
from pixelbelief.occupancy import OccupancyMap
from pixelsplat.ply_export import export_gaussians_to_ply

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

def convert_to_belief_obs(habitat_obs, first_pose, image_size=64):
    # rgb
    rgb = (
        torch.tensor(
            np.asarray(Image.fromarray(habitat_obs["rgb"])).astype(np.float32)
        ).permute(2, 0, 1)
        / 255.0
    )
    rgb = F.interpolate(
        rgb.unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        antialias=True,
    )[0]
    # depth
    depth = torch.tensor(habitat_obs["depth"], dtype=torch.float32).permute(2, 0, 1)
    depth = F.interpolate(
        depth.unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        antialias=True,
    )[0]
    # pose
    pose = pose_habitat2belief(habitat_obs["pose"], first_pose)

    belief_obs = {
        "rgb": rgb,
        "depth": depth,
        "pose": pose
    }

    return belief_obs

def pose_habitat2belief(extrinsics, first_pose):
    conversion = np.diag([1, -1, -1, 1])

    extrinsics = np.linalg.inv(extrinsics)
    extrinsics = conversion @ extrinsics
    c2w_mat = [np.linalg.inv(extrinsics)]
    c2w = torch.tensor(np.array(c2w_mat)).float()

    first_pose = np.linalg.inv(first_pose)
    first_pose = conversion @ first_pose
    c2w_mat_first = [np.linalg.inv(first_pose)]
    c2w_first = torch.tensor(np.array(c2w_mat_first)).float()

    inv_first_c2w = torch.inverse(c2w_first[0])
    inv_first_c2w_repeat = inv_first_c2w.unsqueeze(0).repeat(1, 1, 1)

    pose_belief = torch.einsum(
            "ijk, ikl -> ijl", inv_first_c2w_repeat, c2w
        )
    pose_belief = pose_belief[0]
    print("pose_belief shape", pose_belief.shape)
    return pose_belief

def point_in_room(env_interface: EnvironmentInterface, point: np.ndarray, room_name: str):
    room_region = [region for region in env_interface.perception.sim.semantic_scene.regions if env_interface.perception.region_id_to_name[region.id] == room_name][0]
    return room_region.contains(mn.Vector3(point[0], point[1], point[2]))

def is_too_close_to_wall(env_interface: EnvironmentInterface, point: np.ndarray, forward: np.ndarray, buffer: float = 0.3):
    d = env_interface.perception.sim.pathfinder.distance_to_closest_obstacle(point, 10.0)
    hit = env_interface.perception.sim.pathfinder.closest_obstacle_surface_point(point, 10.0)
    hit_normal = -hit.hit_normal
    # convert to np.array
    hit_normal = np.array([hit_normal[0], hit_normal[1], hit_normal[2]])
    # check if forward is close to the hit normal
    forward = np.array(forward)
    forward = forward / np.linalg.norm(forward)
    hit_normal = hit_normal / np.linalg.norm(hit_normal)
    angle = np.arccos(np.clip(np.dot(forward, hit_normal), -1.0, 1.0))
    if (d < buffer and angle < np.pi / 4) or d < buffer / 10:
        return True
    return False

def get_current_position_and_forward(env_interface: EnvironmentInterface):
    camera_source = env_interface.conf.trajectory.camera_prefixes[0]
    current_pose = np.linalg.inv(
        env_interface.sim.agents[0]
        ._sensors[f"{camera_source}_rgb"]
        .render_camera.camera_matrix
    )
    current_position = current_pose[:3, 3]
    current_forward = current_pose[:3, 2]
    return current_position, current_forward

def rotation_angle(initial_direction, target_direction):
    # Normalize the vectors
    initial_direction_normalized = initial_direction / np.linalg.norm(initial_direction)
    target_direction_normalized = target_direction / np.linalg.norm(target_direction)
    
    # Find the rotation angle (arc cosine of the dot product)
    cos_angle = np.dot(initial_direction_normalized, target_direction_normalized)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    return angle

# Method to load agent planner from the config
def run_planner(cfg: DictConfig):
    run_dir = cfg.results_folder
    save_scene = cfg.agent.save_scene
    num_imagined_trajectories = cfg.agent.num_imagined_trajectories
    semantic_thred = cfg.agent.semantic_thred
    adjacent_angle = cfg.adjacent_angle
    adjacent_distance = cfg.adjacent_distance
    # Initialize the belief agent
    belief_agent = BeliefAgent(cfg)

    # Setup a seed
    seed = 47668090

    # setup required overrides
    DATASET_OVERRIDES = [
        "habitat.dataset.data_path=data/datasets/partnr_episodes/v0_0/val.json.gz",
        "habitat.dataset.scenes_dir=data/hssd-hab/",
    ]
    SENSOR_OVERRIDES = [
        "habitat.simulator.agents.main_agent.sim_sensors.jaw_depth_sensor.normalize_depth=False",
    ]
    LLM_OVERRIDES = [
        "llm@evaluation.planner.plan_config.llm=mock",
    ]
    TRAJECTORY_OVERRIDES = [
        "evaluation.save_video=True",
        "evaluation.output_dir=./outputs",
        "trajectory.save=True",
        "trajectory.agent_names=[main_agent]",
        "trajectory.save_path=data/trajectories/habelief/test/",
    ]

    EPISODE_OVERRIDES = [
        # "+episode_indices=[2,87,370,444,515,590,435,390,555,50,452,355]"
        "+episode_indices=[2]"
    ]  # USE FOR VAL SCENES

    # Setup config
    config_base = get_config(
        "examples/single_agent_scene_mapping.yaml",
        overrides=DATASET_OVERRIDES
        + SENSOR_OVERRIDES
        + LLM_OVERRIDES
        + TRAJECTORY_OVERRIDES
        + EPISODE_OVERRIDES,
    )
    config = setup_config(config_base, seed)

    if config == None:
        cprint("Failed to setup config. Exiting", "red")
        return

    # We register the dynamic habitat sensors
    register_sensors(config)

    # We register custom actions
    register_actions(config)

    # Initialize the environment interface for the agent
    dataset = CollaborationDatasetV0(config.habitat.dataset)
    # if config.get("episode_indices", None) is not None:
    #     episode_subset = [dataset.episodes[x] for x in config.episode_indices]
    #     dataset = CollaborationDatasetV0(
    #         config=config.habitat.dataset, episodes=episode_subset
    #     )
    env_interface = EnvironmentInterface(config, dataset=dataset)

    # Instantiate the agent planner
    eval_runner = CentralizedEvaluationRunner(config.evaluation, env_interface)

    # Initialize the task manager
    manager_config = get_config(
        "examples/object_searching.yaml",
    )
    task_manager = ObjectSearchingTaskManager(
        env_interface=env_interface,
        eval_runner=eval_runner,
        dataset=dataset,
        config=manager_config,
    )

    vlm = VLMAgent(vlm_model_name="gpt-4o")

    # Highlight the mode of operation
    cprint("\n---------------------------------------", "blue")
    cprint(f"Planner Mode: {config.evaluation.type.capitalize()}", "blue")
    cprint(f"Partial Observability: {config.world_model.partial_obs}", "blue")
    # Print the agent list
    cprint(f"Agent List: {eval_runner.agent_list}", "blue")
    if env_interface._single_agent_mode:
        cprint("Single agent mode", "green")
    cprint("---------------------------------------\n", "blue")
    num_episodes = len(task_manager.episodes)
    robot_agent_uid = manager_config.agent_id
    max_steps = manager_config.max_steps

    # initial reset to load first episode
    for idx in range(num_episodes):
        obs = task_manager.reset()
        belief_agent.reset()

        target_obj = task_manager.target_obj
        # DEBUG
        print(f"Target object: {target_obj}")
        ## DEBUG

        assert isinstance(target_obj, str), "Target object should be a string"

        # create save folders
        save_folder_sample = os.path.join(run_dir, f"{task_manager.scene_number}_{target_obj}_{idx}")
        os.makedirs(
            save_folder_sample, exist_ok=True,
        )

        save_folder_nav_video = os.path.join(save_folder_sample, f'nav_video')
        os.makedirs(
            save_folder_nav_video, exist_ok=True,
        )

        save_folder_observation = os.path.join(save_folder_sample, f'observation')
        os.makedirs(
            save_folder_observation, exist_ok=True,
        )

        save_folder_planning = os.path.join(save_folder_sample, f'planning')
        os.makedirs(
            save_folder_planning, exist_ok=True,
        )

        save_folder_obs = os.path.join(save_folder_observation, f'obs_frames')
        os.makedirs(
            save_folder_obs, exist_ok=True,
        )

        save_folder_obs_obs_map = os.path.join(save_folder_observation, f'obs_maps')
        os.makedirs(
            save_folder_obs_obs_map, exist_ok=True,
        )

        save_folder_obs_height_map = os.path.join(save_folder_observation, f'height_maps')
        os.makedirs(
            save_folder_obs_height_map, exist_ok=True,
        )

        save_folder_height_map = os.path.join(save_folder_planning, f'height_map')
        os.makedirs(
            save_folder_height_map, exist_ok=True,
        )

        save_folder_imagine = os.path.join(save_folder_planning, f'imagined_frames')
        os.makedirs(
            save_folder_imagine, exist_ok=True,
        )

        save_folder_obs_map = os.path.join(save_folder_planning, f'obs_map')
        os.makedirs(
            save_folder_obs_map, exist_ok=True,
        )

        # get current observation
        observations = env_interface.get_observations()
        all_frames = []
        
        first_pose_habitat = None
        step = 0
        done = False
        start_time = time.time()
        while step < max_steps and not done:
            # step start time
            step_start_time = time.time()
            # Extract current obs
            habitat_obs = extract_obs(env_interface, obs)

            if step == 0:
                first_pose_habitat = habitat_obs["pose"]

            Image.fromarray(habitat_obs["rgb"]).save(
                os.path.join(save_folder_obs, f"observed_{step}.png")
            )
            
            belief_obs = BeliefAgent.convert_to_belief_obs(habitat_obs, first_pose_habitat)

            current_location = belief_obs["pose"][:3, 3].detach().cpu().numpy()
            
            # observe with the current observation
            belief_agent.observe([belief_obs["rgb"]], [belief_obs["pose"]])
            # save obs map
            belief_agent.obs_map.save_occupancy_map(
                os.path.join(save_folder_obs_obs_map, f"obs_map_{step}.png"),
            )
            # save height map
            belief_agent.obs_map.save_height_map(
                os.path.join(save_folder_obs_height_map, f"height_map_{step}.png"),
            )
            # use vlm TODO
            success = vlm.prompt_score_obj_image(
                image_file=os.path.join(save_folder_obs, f"observed_{step}.png"),
                object_name=target_obj,
            )

            # If new observation contains the target object, set success
            if success:
                step_log = {
                    "step": idx,
                    "is_direct": success,
                    "target_obj": target_obj,
                    "step_time": time.time() - step_start_time,
                }
                # dump the step log
                with open(os.path.join(save_folder_sample, f"step_log_{step}.json"), "w") as f:
                    json.dump(step_log, f, indent=4)
                
                done = True
                print(f"Found target object {target_obj} in observation {step}.")
                continue
            
            # Otherwise, apply the pure rotation policy

            current_position, current_forward = get_current_position_and_forward(env_interface)
            z_previous = current_forward
            t_previous = current_position

            # navigate following a subset of the path
            hl_action_name = "NavigatePose"

            debug_frames = []

            observations = env_interface.get_observations()

            hl_action_input = [] ## TODO
            hl_action_done = False
            print(f"Navigating to {hl_action_input}")
            hl_action_input = (hl_action_input, True, False, (0.05, 0.1))

            while not hl_action_done:
                low_level_action = np.zeros((39,), dtype=np.float32)
                low_level_action[8] = -10.0  # Pure rotation action
                low_level_action = {robot_agent_uid: low_level_action}

                obs, _, _, _ = env_interface.step(
                    low_level_action,
                )
                observations = env_interface.parse_observations(obs)
                frames_concat = eval_runner.dvu._DebugVideoUtil__get_combined_frames(observations)
                frames_concat = np.ascontiguousarray(frames_concat)
                debug_frames.append(frames_concat)
                t_current, z_current = get_current_position_and_forward(env_interface)
                angle = rotation_angle(z_previous, z_current)
                distance = np.linalg.norm(t_current - t_previous)
                if angle > adjacent_angle or distance > adjacent_distance:
                    break

            # step end time
            step_end_time = time.time()

            # save video
            video_path = os.path.join(save_folder_nav_video, f"nav_video_{step}.mp4")
            imageio.mimwrite(
                video_path,
                debug_frames,
                fps=10,
                quality=10,
            )
            # append elements in debug_frames to all_frames
            all_frames.extend(debug_frames)

            is_done = task_manager.is_done()
            if is_done:
                print(f"Episode {idx} completed.")
                done = True

            # log everything
            step_log = {
                "step": idx,
                "is_direct": False,
                "target_obj": target_obj,
                "semantic_thred": semantic_thred,
                "step_time": step_end_time - step_start_time,
            }

            # dump the step log
            with open(os.path.join(save_folder_sample, f"step_log_{step}.json"), "w") as f:
                json.dump(step_log, f, indent=4)
            
            # increment step
            step+=1
    
        # save final nav video
        video_path = os.path.join(save_folder_nav_video, f"full_nav_video.mp4")
        imageio.mimwrite(
            video_path,
            all_frames,
            fps=10,
            quality=10,
        )
        
        # setup final log
        final_log = {
            "scene": task_manager.scene_number,
            "target_obj": target_obj,
            "num_steps": step,
            "success": done,
            "time_taken": time.time() - start_time,
        }
        # dump final log
        with open(os.path.join(save_folder_sample, f"final_log.json"), "w") as f:
            json.dump(final_log, f, indent=4)

    env_interface.sim.close()

if __name__ == "__main__":
    cprint(
        "\nStart of the belief agent exploration",
        "blue",
    )
    cfg_path = "/home/ubuntu/VLMP/tianmin-project/yyin34/codebase/embodied_tasks/DFM/configurations"
    with initialize_config_dir(config_dir=cfg_path, version_base="1.2"):
        cfg = compose(
            config_name="sp_reason.yaml",
            overrides=[
                "sampling_steps=10",
                "semantic_mode=embed",
                "semantic_viz=query",
                "adjacent_angle=0.785",
                "adjacent_distance=1.0",
                "clean_target=False",
                "use_history=False",
                "model.encoder.use_epipolar_transformer=False",
                "model.encoder.use_image_condition=True",
                "model.encoder.depth_predictor_time_embed=True",
                "model.encoder.evolve_ctxt=True",
                "model.encoder.use_camera_pose=True",
                "model.encoder.use_semantic=False",
                "model.encoder.use_reg_model=False",
                "model.encoder.d_semantic=512",
                "model.encoder.d_semantic_reg=384",
                "model.encoder.gaussians_per_pixel=3",
                "model.encoder.inference_mode=False",
                "model.encoder.backbone.use_diff_pos_embed=True",
                "model.encoder.backbone.pose_condition_type=prope",
                "agent.save_scene=False",
            ]
        )
    cfg.checkpoint_path = "/home/ubuntu/VLMP/tianmin-project/yyin34/codebase/DFM/outputs/training/habelief_room/dfm/pose_cond_prope_evolve_ctxt/model-8.pt"
    cfg.results_folder = "/home/ubuntu/VLMP/tianmin-project/yyin34/codebase/embodied_tasks/DFM/outputs/belief_agent_pose_prope_evolve_ctxt_pure_rotation"
    cfg.semantic_config = "/home/ubuntu/VLMP/tianmin-project/yyin34/codebase/embodied_tasks/DFM/configurations/semantic/onehot.yaml"

    # Run planner
    run_planner(cfg)

    cprint(
        "\nEnd of the belief agent exploration",
        "blue",
    )
