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
            
            visual_0 = habitat_obs["rgb"]

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
            # render at the current pose
            rgb, depth, _ = belief_agent.render_image(extrinsics=belief_obs["pose"], query_label=target_obj)

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
                
                # Create and save a final visualization with just the observation where object was found
                vis_path = os.path.join(save_folder_sample, "visualization.png")
                prev_vis = Image.open(vis_path) if os.path.exists(vis_path) and step > 0 else None
                
                # Create a simplified visualization with only the observation image
                final_visuals = {
                    'visual_0': visual_0,
                    'visual_1': None,
                    'visual_2': [],
                    'visual_3': None
                }
                
                # Create a special version of visualization for success state
                create_success_visualization(final_visuals, step, vis_path, prev_vis, target_obj)
                
                done = True
                print(f"Found target object {target_obj} in observation {step}.")
                continue
            else: # Otherwise, continue exploring and imagining
                goals = belief_agent.sample_next_exploration_goals(
                    belief_agent.obs_map, 
                    belief_agent.current_pose[:3, 3].detach().cpu().numpy(),
                    plot_path=os.path.join(save_folder_obs_map, f"map_{step}.png")
                )   
                print("# Goals", len(goals))

                backup_goal = goals[-1]

                # filter out goals that not in the room
                goals = [goal for goal in goals if point_in_room(
                    env_interface, 
                    BeliefAgent.points_belief2habitat([goal["pose"][-1][:3, 3]], first_pose_habitat)[0],
                    task_manager.room_name
                )]
                # filter out goals that are too close to a wall
                goals = [goal for goal in goals if not is_too_close_to_wall(
                    env_interface,
                    BeliefAgent.points_belief2habitat([goal["pose"][-1][:3, 3]], first_pose_habitat)[0],
                    forward=BeliefAgent.pose_belief2habitat(goal["pose"][-1], first_pose_habitat)[:3, 2],
                    buffer=0.2
                )]

                goals.append(backup_goal)

                # keep at most num_imagined_trajectories goals
                if len(goals) > num_imagined_trajectories-1:
                    goals = random.sample(goals, num_imagined_trajectories-1)
                # append the backup goal

                save_folder_imagine_step = os.path.join(save_folder_imagine, f'step_{step}')
                os.makedirs(
                    save_folder_imagine_step, exist_ok=True,
                )
                
                optimal_goal = None
                optimal_belief_scene = None
                optimal_key_poses = None
                optimal_frames = None
                optimal_scores = None
                best_semantic_score = -1
                for gidx, goal_dict in enumerate(goals):
                    path = goal_dict["path"]
                    poses = goal_dict["pose"]

                    imagined_frames = []

                    belief_agent.obs_map.save_height_map(
                        os.path.join(save_folder_height_map, f"height_map_with_goals_{step}_{gidx}.png"), path=path
                    )

                    save_folder_imagine_step_goal = os.path.join(save_folder_imagine_step, f'goal_{gidx}')
                    os.makedirs(
                        save_folder_imagine_step_goal, exist_ok=True,
                    )

                    # 3D imagination
                    key_output, _, belief_scene = belief_agent.imagine_in_place(
                                                        poses[-1], 
                                                        query_label=target_obj,
                                                        return_belief_scene=True
                                                    )
                    frames = key_output.rgb
                    depths = key_output.depth
                    key_poses = key_output.pose

                    for p, frame in enumerate(frames):
                        Image.fromarray(frame).save(
                            os.path.join(save_folder_imagine_step_goal, f"rendered_{p}.png")
                        )
                        imagined_frames.append(frame)

                    # object detection on the imagined frames
                    results = vlm.prompt_score_obj_folder(
                        image_folder=save_folder_imagine_step_goal,
                        object_name=target_obj,
                    )
                    presences = [ele[0] for ele in results]
                    scores = [ele[1] for ele in results]
                    max_idx = np.argmax(scores)
                    semantic_score = scores[max_idx]
                    
                    # save the belief scene
                    if save_scene:
                        ply_path = Path(f"{save_folder_imagine_step_goal}/scene_goal_{gidx}.ply")
                        export_gaussians_to_ply(
                            belief_scene[-1].float(),
                            key_poses[-1].detach().unsqueeze(0).to("cuda"),
                            ply_path
                        )
                    
                    if semantic_score > best_semantic_score:
                        best_semantic_score = semantic_score
                        optimal_belief_scene = belief_scene
                        optimal_key_poses = key_poses
                        optimal_frames = imagined_frames
                        optimal_scores = scores
                        optimal_goal = optimal_key_poses[max_idx].detach().cpu().numpy()[:3, 3]

                # Set imagined occupancy map
                obs_map = deepcopy(belief_agent.obs_map)
                assert len(optimal_belief_scene) == len(optimal_key_poses)
                for i in range(len(optimal_key_poses)):
                    inc_map = OccupancyMap(resolution=belief_agent.step_size, obstacle_height_thresh=belief_agent.obstacle_height_thresh)
                    obs_pose_np = optimal_key_poses[i].detach().cpu().numpy()
                    Rcw = obs_pose_np[:3, :3]
                    forward = Rcw @ np.array([0.0, 0.0, 1.0])
                    yaw = -math.atan2(forward[0], forward[2])
                    yaw = yaw % (2*math.pi)
                    inc_map.set_point_cloud(
                        pcd=optimal_belief_scene[i].float().means.squeeze(0).detach().cpu().numpy(), 
                        sensor_origin=tuple(obs_pose_np[:3, 3]), 
                        yaw=yaw, 
                        intrinsics=belief_agent.camera.intrinsics
                    )
                    obs_map.merge(inc_map)
                
                # DEBUG save occupancy map
                obs_map.save_occupancy_map(
                    os.path.join(save_folder_obs_map, f"imagined_plan_obs_map_{step}.png"),
                    goals=[optimal_goal],
                )

                imagined_key_points = [key.detach().cpu().numpy()[:3, 3] for key in optimal_key_poses]
                # add current_location as the first point
                imagined_key_points.insert(0, current_location)

                visual_1 = belief_agent.obs_map.save_occupancy_map(
                    os.path.join(save_folder_obs_map, f"imagined_path_obs_map_{step}.png"),
                    goals=[current_location],
                    path=imagined_key_points,
                    return_image=True
                )

                visual_2 = optimal_frames
                visual_2_scores = optimal_scores

                # plan a path to the goal
                path = obs_map.plan(tuple(belief_obs["pose"][:3, 3].detach().cpu().numpy()), tuple(optimal_goal))
                if path is None:
                    print("Failed to plan a path to the goal, using interpolation.")
                    path = belief_agent.interpolate_path(
                        belief_obs["pose"][:3, 3].detach().cpu().numpy(),
                        optimal_goal,
                        step_size=0.05, # TODO set a step size
                    )
            
            # Calculate the path distance between start and goal
            if path is not None:
                path_distance = np.linalg.norm(
                    np.array(path[-1]) - np.array(path[0])
                )
                if path_distance < 1.5:
                    face_to_object = True
                else:
                    face_to_object = False

            visual_3 = belief_agent.obs_map.save_occupancy_map(
                os.path.join(save_folder_planning, f"path_obs_map_{step}.png"),
                goals=[current_location],
                path=path[:len(path)+1],
                return_image=True
            )
            
            # Create and save visualization with all visuals
            vis_path = os.path.join(save_folder_sample, "visualization.png")
            prev_vis = Image.open(vis_path) if os.path.exists(vis_path) and step > 0 else None
            
            # Create a dictionary of visuals
            visuals = {
                'visual_0': visual_0,
                'visual_1': visual_1,
                'visual_2': visual_2,
                'visual_2_scores': visual_2_scores,
                'visual_3': visual_3
            }
            
            # Update the visualization - pass target_obj to create_step_visualization
            _ = create_step_visualization(visuals, step, vis_path, prev_vis, target_obj)

            path_habitat = BeliefAgent.points_belief2habitat(path, first_pose_habitat)
            path_habitat_exe = path_habitat[:len(path_habitat)+1] # TODO find a subset of the path with step size

            # record current forward direction
            current_position, current_forward = get_current_position_and_forward(env_interface)
            z_previous = current_forward
            t_previous = current_position

            # navigate following a subset of the path
            hl_action_name = "NavigatePose"

            debug_frames = []

            observations = env_interface.get_observations()

            hl_action_input = path_habitat_exe[-1]
            hl_action_done = False
            print(f"Navigating to {hl_action_input}")
            hl_action_input = (hl_action_input, face_to_object, False, (0.05, 0.1))

            while not hl_action_done:
                low_level_action, response = eval_runner.planner.agents[
                    robot_agent_uid
                ].process_high_level_action(
                    hl_action_name, hl_action_input, observations
                )
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
                
                if response:
                    print(f"\tResponse: {response}")
                    hl_action_done = True

            # check if the agent is still in the room
            world_graph = env_interface.full_world_graph
            if world_graph.get_room_for_entity(world_graph.get_spot_robot()).name != task_manager.room_name:
                cprint(f"Agent is outside the room {task_manager.room_name}, resetting to the last position.", "red")
                hl_action_name = "NavigatePose"
                hl_action_input = (task_manager.last_position, True, False, (0.05, 0.1))
                hl_action_done = False
                while not hl_action_done:
                    low_level_action, response = eval_runner.planner.agents[
                        task_manager.agent_id
                    ].process_high_level_action(
                        hl_action_name, hl_action_input, env_interface.get_observations()
                    )
                    low_level_action = {task_manager.agent_id: low_level_action}

                    obs, _, _, _ = env_interface.step(
                        low_level_action
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

                    if response:
                        print(f"\tResponse: {response}")
                        hl_action_done = True

            t_current, z_current = get_current_position_and_forward(env_interface)

            # check if the agent is too close to a wall
            if is_too_close_to_wall(
                env_interface,
                t_current,
                z_current,
                buffer=0.2,
            ):
                cprint("Agent is too close to a wall, resetting to the last position.", "red")
                hl_action_name = "NavigatePose"
                hl_action_input = (task_manager.last_position, True, False, (0.05, 0.1))
                hl_action_done = False
                while not hl_action_done:
                    low_level_action, response = eval_runner.planner.agents[
                        task_manager.agent_id
                    ].process_high_level_action(
                        hl_action_name, hl_action_input, env_interface.get_observations()
                    )
                    low_level_action = {task_manager.agent_id: low_level_action}

                    obs, _, _, _ = env_interface.step(
                        low_level_action
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

                    if response:
                        print(f"\tResponse: {response}")
                        hl_action_done = True

            step_position, _ = get_current_position_and_forward(env_interface)
            task_manager.set_last_position(step_position)

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

            step_log["imagined_goal"] = {
                "semantic_score": best_semantic_score,
                "goal": optimal_goal.tolist(),
                "path": [ele.tolist() for ele in path_habitat_exe],
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

def create_step_visualization(visuals, step, save_path, prev_img=None, target_obj=None):
    """
    Create a visualization that combines multiple visual elements.
    
    Args:
        visuals: Dictionary with visual elements:
            - visual_0: observation image
            - visual_1: imagined path image
            - visual_2: list of imagined frames
            - visual_2_scores: list of scores for imagined frames
            - visual_3: path image
        step: Current step number
        save_path: Path to save the visualization
        prev_img: Previous visualization to append to (if not the first step)
        target_obj: Name of the target object being searched for
    """
    # Convert PIL images to numpy arrays if needed
    for key in ['visual_0', 'visual_1', 'visual_3']:
        if key in visuals and isinstance(visuals[key], Image.Image):
            visuals[key] = np.array(visuals[key])
    
    # Calculate how many frames in visual_2
    num_frames = len(visuals['visual_2']) if 'visual_2' in visuals else 0
    total_cols = 3 + num_frames  # visual_0, visual_1, visual_2 (multiple frames), visual_3
    
    # Create a new figure for this step
    fig = plt.figure(figsize=(20, 5))
    
    # Add step number as a title for the entire row (inside the bounding box)
    fig.suptitle(f"Step {step}", fontsize=20, y=0.95)
    
    # Define grid - we'll create a special layout to show visuals in order 0,1,2,3
    if num_frames > 0:
        # Create a grid with proper proportions for the visual_2 frames
        gs = GridSpec(1, total_cols)
        
        # Get reference height from visual_0 or first frame of visual_2
        ref_height = visuals['visual_0'].shape[0] if 'visual_0' in visuals else visuals['visual_2'][0].shape[0]
        
        # Resize visual_1 and visual_3 to match reference height
        if 'visual_1' in visuals and visuals['visual_1'] is not None:
            h, w = visuals['visual_1'].shape[:2]
            new_w = int(w * (ref_height / h))
            visuals['visual_1'] = np.array(Image.fromarray(visuals['visual_1']).resize((new_w, ref_height)))
            
        if 'visual_3' in visuals and visuals['visual_3'] is not None:
            h, w = visuals['visual_3'].shape[:2]
            new_w = int(w * (ref_height / h))
            visuals['visual_3'] = np.array(Image.fromarray(visuals['visual_3']).resize((new_w, ref_height)))
        
        # Add visual_0
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(visuals['visual_0'])
        ax0.set_title('Obs', fontsize=18)
        ax0.axis('off')
        
        # Add visual_1
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.imshow(visuals['visual_1'])
        ax1.set_title('Imagined Path', fontsize=18)
        ax1.axis('off')
        
        # Add visual_2 (multiple frames)
        for i, frame in enumerate(visuals['visual_2']):
            ax = fig.add_subplot(gs[0, 2 + i])
            ax.imshow(frame)
            
            # Add score as subtitle if available
            if 'visual_2_scores' in visuals and i < len(visuals['visual_2_scores']):
                score = visuals['visual_2_scores'][i]
                title = f'Imag. Score: {score}'
            else:
                title = 'Imag. Frames' if i == 0 else ''
                
            ax.set_title(title, fontsize=18)
            ax.axis('off')
        
        # Add visual_3
        ax3 = fig.add_subplot(gs[0, total_cols - 1])
        ax3.imshow(visuals['visual_3'])
        ax3.set_title('Path', fontsize=18)
        ax3.axis('off')
        
    else:
        # If no frames in visual_2, just show the other visuals
        gs = GridSpec(1, 3)
        
        # Get reference height from visual_0
        if 'visual_0' in visuals and visuals['visual_0'] is not None:
            ref_height = visuals['visual_0'].shape[0]
            
            # Resize visual_1 and visual_3 to match reference height
            if 'visual_1' in visuals and visuals['visual_1'] is not None:
                h, w = visuals['visual_1'].shape[:2]
                new_w = int(w * (ref_height / h))
                visuals['visual_1'] = np.array(Image.fromarray(visuals['visual_1']).resize((new_w, ref_height)))
                
            if 'visual_3' in visuals and visuals['visual_3'] is not None:
                h, w = visuals['visual_3'].shape[:2]
                new_w = int(w * (ref_height / h))
                visuals['visual_3'] = np.array(Image.fromarray(visuals['visual_3']).resize((new_w, ref_height)))
        
        # Add titles and images in order
        titles = {
            'visual_0': 'Obs',
            'visual_1': 'Imag. Path',
            'visual_3': 'Path'
        }
        
        # Add the individual images
        for i, (key, title) in enumerate(titles.items()):
            if key in visuals and visuals[key] is not None:
                ax = fig.add_subplot(gs[0, i])
                ax.imshow(visuals[key])
                ax.set_title(title, fontsize=18)
                ax.axis('off')
    
    # Add a bounding box around the entire row
    from matplotlib.patches import Rectangle
    fig.patches.extend([Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', 
                                 linewidth=2, transform=fig.transFigure)])
    
    # Save this step's visualization temporarily
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.95])  # Adjust layout to account for the bounding box
    temp_path = save_path.replace('.png', f'_temp_{step}.png')
    plt.savefig(temp_path)
    plt.close()
    
    # Now combine with previous visualization if it exists
    step_img = Image.open(temp_path)
    
    if prev_img is None:
        # First step - add target object title at the top
        if target_obj:
            # Create a separate figure just for the title with a large font
            title_height = 120  # Much larger height for the title section
            title_fig = plt.figure(figsize=(step_img.width/100, title_height/100))  # Convert pixels to inches
            title_fig.patch.set_facecolor('white')
            
            # Add a large, centered title
            plt.figtext(0.5, 0.5, f"Target Object: {target_obj}", 
                      fontsize=36, fontweight='bold', ha='center', va='center')
            
            # No axes for the title
            plt.axis('off')
            
            # Save the title image
            title_path = save_path.replace('.png', '_title.png')
            plt.savefig(title_path, bbox_inches='tight', pad_inches=0.3)
            plt.close(title_fig)
            
            # Open the title image
            title_img = Image.open(title_path)
            
            # Create a new combined image
            combined_height = title_img.height + step_img.height
            combined_img = Image.new('RGB', (max(title_img.width, step_img.width), combined_height), color=(255, 255, 255))
            
            # Paste the title and step images
            combined_img.paste(title_img, ((combined_img.width - title_img.width) // 2, 0))
            combined_img.paste(step_img, ((combined_img.width - step_img.width) // 2, title_img.height))
            
            # Save the combined image
            combined_img.save(save_path)
            
            # Clean up temporary title image
            import os
            if os.path.exists(title_path):
                os.remove(title_path)
                
            return combined_img
        else:
            # No target object specified
            step_img.save(save_path)
            return step_img
    else:
        # Append this step to previous visualization
        combined_height = prev_img.height + step_img.height
        combined_img = Image.new('RGB', (max(prev_img.width, step_img.width), combined_height))
        combined_img.paste(prev_img, (0, 0))
        combined_img.paste(step_img, (0, prev_img.height))
        combined_img.save(save_path)
        
        # Clean up temporary file
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return combined_img

def create_success_visualization(visuals, step, save_path, prev_img=None, target_obj=None):
    """
    Create a visualization for the success case, showing only the observation image
    where the target object was found.
    
    Args:
        visuals: Dictionary with visual elements (only visual_0 will be used)
        step: Current step number
        save_path: Path to save the visualization
        prev_img: Previous visualization to append to
        target_obj: Name of the target object that was found
    """
    # Convert PIL image to numpy array if needed
    if 'visual_0' in visuals and isinstance(visuals['visual_0'], Image.Image):
        visuals['visual_0'] = np.array(visuals['visual_0'])
    
    # Create a new figure for this step
    fig = plt.figure(figsize=(10, 5))
    
    # Add step number and success message as a title (inside the bounding box)
    success_message = f"Step {step} - Found {target_obj}!"
    fig.suptitle(success_message, fontsize=20, y=0.95, color='green')
    
    # Create a grid with just one column
    gs = GridSpec(1, 1)
    
    # Add the observation image
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(visuals['visual_0'])
    ax.set_title('Target Object Found', fontsize=18, color='green')
    ax.axis('off')
    
    # Add a green bounding box to indicate success
    from matplotlib.patches import Rectangle
    fig.patches.extend([Rectangle((0, 0), 1, 1, fill=False, edgecolor='green', 
                                 linewidth=3, transform=fig.transFigure)])
    
    # Save this step's visualization temporarily
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.95])
    temp_path = save_path.replace('.png', f'_temp_{step}.png')
    plt.savefig(temp_path)
    plt.close()
    
    # Now combine with previous visualization if it exists
    step_img = Image.open(temp_path)
    
    if prev_img is None:
        # First step - add target object title at the top if needed
        if target_obj:
            # Create a separate figure just for the title with a large font
            title_height = 120  # Much larger height for the title section
            title_fig = plt.figure(figsize=(step_img.width/100, title_height/100))  # Convert pixels to inches
            title_fig.patch.set_facecolor('white')
            
            # Add a large, centered title
            plt.figtext(0.5, 0.5, f"Target Object: {target_obj}", 
                      fontsize=36, fontweight='bold', ha='center', va='center')
            
            # No axes for the title
            plt.axis('off')
            
            # Save the title image
            title_path = save_path.replace('.png', '_title.png')
            plt.savefig(title_path, bbox_inches='tight', pad_inches=0.3)
            plt.close(title_fig)
            
            # Open the title image
            title_img = Image.open(title_path)
            
            # Create a new combined image
            combined_height = title_img.height + step_img.height
            combined_img = Image.new('RGB', (max(title_img.width, step_img.width), combined_height), color=(255, 255, 255))
            
            # Paste the title and step images
            combined_img.paste(title_img, ((combined_img.width - title_img.width) // 2, 0))
            combined_img.paste(step_img, ((combined_img.width - step_img.width) // 2, title_img.height))
            
            # Save the combined image
            combined_img.save(save_path)
            
            # Clean up temporary title image
            import os
            if os.path.exists(title_path):
                os.remove(title_path)
                
            return combined_img
        else:
            step_img.save(save_path)
            return step_img
    else:
        # Append this step to previous visualization
        combined_height = prev_img.height + step_img.height
        combined_img = Image.new('RGB', (max(prev_img.width, step_img.width), combined_height))
        combined_img.paste(prev_img, (0, 0))
        combined_img.paste(step_img, (0, prev_img.height))
        combined_img.save(save_path)
        
        # Clean up temporary file
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return combined_img

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
    cfg.checkpoint_path = "/home/ubuntu/VLMP/tianmin-project/yyin34/codebase/DFM/outputs/weights/habelief/dfm_prope_evolve_ctxt_semantic_room_ft/model-2.pt"
    cfg.results_folder = "/home/ubuntu/VLMP/tianmin-project/yyin34/codebase/embodied_tasks/DFM/outputs/belief_agent_prope_evolve_ctxt_semantic_room_ft"
    cfg.semantic_config = "/home/ubuntu/VLMP/tianmin-project/yyin34/codebase/embodied_tasks/DFM/configurations/semantic/onehot.yaml"

    # Run planner
    run_planner(cfg)

    cprint(
        "\nEnd of the belief agent exploration",
        "blue",
    )
