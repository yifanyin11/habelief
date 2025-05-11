#!/usr/bin/env python3
# isort: skip_file

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This script implements structured episodes over a collection of scenes, which
ask the agent to go to each furniture within the scene and save a RGBD+pose trajectory.
This trajectory is then used to create a map of the scenes through Concept-Graphs.
"""

import sys
import os
import random
import pathlib
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import imageio

ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
# append the path of the
# parent directory
sys.path.append("..")
from typing import Any, Dict
from omegaconf import DictConfig, OmegaConf
from hydra import initialize_config_dir, compose
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower  
from habitat.sims.habitat_simulator.actions import HabitatSimActions

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

from pixelbelief.belief_agent import BeliefAgent, prepare_video
from pixelbelief.occupancy import OccupancyMap

def get_agent_room_name(env_interface: EnvironmentInterface):
    world_graph = env_interface.full_world_graph
    return world_graph.get_room_for_entity(world_graph.get_human()).name

def extract_obs(env_interface: EnvironmentInterface, obs: Dict[str, Any]):

    curr_agent, camera_source = env_interface.trajectory_agent_names[0], env_interface.conf.trajectory.camera_prefixes[0]

    assert curr_agent=='agent_1'

    if env_interface._single_agent_mode:
        rgb = obs[f"{camera_source}_rgb"]
        depth = obs[f"{camera_source}_depth"]
        panoptic = obs[f"{camera_source}_panoptic"]
        pose = np.linalg.inv(
            env_interface.sim.agents[0]
            ._sensors[f"{camera_source}_rgb"]
            .render_camera.camera_matrix
        )
    else:
        rgb = obs[f"{curr_agent}_{camera_source}_rgb"]
        depth = obs[f"{curr_agent}_{camera_source}_depth"]
        panoptic = obs[f"{curr_agent}_{camera_source}_panoptic"]
        pose = np.linalg.inv(
            env_interface.sim.agents[0]
            ._sensors[f"{curr_agent}_{camera_source}_rgb"]
            .render_camera.camera_matrix
        )
    
    extracted_obs = {
        "rgb": rgb,
        "depth": depth,
        "panoptic": panoptic,
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
        "panoptic": habitat_obs["panoptic"], # unchanged for now
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

# Method to load agent planner from the config
def run_planner(cfg: DictConfig):
    run_dir = cfg.results_folder
    save_scene = cfg.agent.save_scene
    # Initialize the belief agent
    belief_agent = BeliefAgent(cfg)
    belief_agent.reset()

    # Setup a seed
    seed = 47668090

    # setup required overrides
    DATASET_OVERRIDES = [
        "habitat.dataset.data_path=data/datasets/partnr_episodes/v0_0/val.json.gz",
        "habitat.dataset.scenes_dir=data/hssd-hab/",
    ]
    SENSOR_OVERRIDES = [
        "habitat.simulator.agents.agent_0.sim_sensors.jaw_depth_sensor.normalize_depth=False",
        "habitat.simulator.agents.agent_1.sim_sensors.head_depth_sensor.normalize_depth=False"
    ]
    LLM_OVERRIDES = [
        "llm@evaluation.planner.plan_config.llm=mock",
    ]
    TRAJECTORY_OVERRIDES = [
        "evaluation.save_video=True",
        "evaluation.output_dir=./outputs",
        "trajectory.save=True",
        "trajectory.agent_names=[agent_1]",
        "trajectory.save_path=data/trajectories/habelief/test/",
    ]

    EPISODE_OVERRIDES = [
        # "+episode_indices=[2,87,370,444,515,590,435,390,555,50,452,355]"
        "+episode_indices=[2]"
    ]  # USE FOR VAL SCENES

    # Setup config
    config_base = get_config(
        "examples/multi_agent_scene_mapping.yaml",
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

    # Highlight the mode of operation
    cprint("\n---------------------------------------", "blue")
    cprint(f"Planner Mode: {config.evaluation.type.capitalize()}", "blue")
    cprint(f"Partial Observability: {config.world_model.partial_obs}", "blue")
    # Print the agent list
    cprint(f"Agent List: {eval_runner.agent_list}", "blue")
    if env_interface._single_agent_mode:
        cprint("Single agent mode", "green")
    cprint("---------------------------------------\n", "blue")
    num_episodes = len(env_interface.env.episodes)
    processed_scenes = {}
    robot_agent_uid = config.robot_agent_uid
    room_type = "kitchen"

    follower = ShortestPathFollower(
        sim=env_interface.env.env.env._env.sim,
        goal_radius=0.2,
        return_one_hot=False,
        stop_on_error=True
    )

    env = env_interface.env.env.env._env

    # initial reset to load first episode
    for idx in range(num_episodes):
        env_interface.reset_environment()
        eval_runner.reset()
        cur_episode = env_interface.env.env.env._env.current_episode
        cur_episode.episode_id = idx
        scene_id = cur_episode.scene_id

        print(
            f"Processing scene: {scene_id}, episode: {idx+1}/{num_episodes}, processed scenes: {len(processed_scenes)}"
        )

        save_folder_sample = os.path.join(run_dir, f"visuals_{idx}")
        os.makedirs(
            save_folder_sample, exist_ok=True,
        )

        observations = env_interface.get_observations()

        # get the list of all rooms in this house
        rooms = env_interface.world_graph[robot_agent_uid].get_all_nodes_of_type(Room)
        for current_room in rooms:
            if room_type in current_room.name:
                break
        
        hl_action_name = "Explore"
        hl_action_input = current_room.name
        hl_action_done = False
        print(f"Navigating to {hl_action_input}")

        while not hl_action_done:
            # Get response and/or low level actions
            env_interface.reset_world_graph()
            low_level_action, response = eval_runner.planner.agents[
                1
            ].process_high_level_action(
                hl_action_name, hl_action_input, observations
            )
            low_level_action = {1: low_level_action}

            obs, _, _, _ = env_interface.step(
                low_level_action, room_name=current_room.name
            )
            break
        
        first_pose_habitat = None
        step = 0

        for step in range(5):
            # Extract obs
            habitat_obs = extract_obs(env_interface, obs)

            if step==0:
                first_pose_habitat = habitat_obs["pose"]
            
            save_folder_obs = os.path.join(save_folder_sample, f'obs_frames')
            os.makedirs(
                save_folder_obs, exist_ok=True,
            )

            Image.fromarray(habitat_obs["rgb"]).save(
                os.path.join(save_folder_obs, f"rendered_{step}.png")
            )

            belief_obs = BeliefAgent.convert_to_belief_obs(habitat_obs, first_pose_habitat)
            
            belief_agent.observe([belief_obs["rgb"]], [belief_obs["pose"]])

            save_folder_obs_map = os.path.join(save_folder_sample, f'obs_map')
            os.makedirs(
                save_folder_obs_map, exist_ok=True,
            )
            belief_agent.obs_map.save_occupancy_map(os.path.join(save_folder_obs_map, f"map_{step}.png"))

            goals = belief_agent.sample_next_exploration_goals(belief_agent.obs_map, belief_agent.current_pose[:3, 3].detach().cpu().numpy())
            print("# Goals", len(goals))
            goal_dict = goals[0]
            path = goal_dict["path"]
            poses = goal_dict["pose"]

            save_folder_height_map = os.path.join(save_folder_sample, f'height_map')
            os.makedirs(
                save_folder_height_map, exist_ok=True,
            )
            belief_agent.obs_map.save_height_map(
                os.path.join(save_folder_height_map, f"height_map_with_goals{step}.png"), path=path
            )

            save_folder_imagine = os.path.join(save_folder_sample, f'imagined_frames{step}')
            os.makedirs(
                save_folder_imagine, exist_ok=True,
            )
            frames = belief_agent.imagine_in_place(poses[-1])

            # save all frames
            for p, frame in enumerate(frames):
                Image.fromarray(frame).save(
                    os.path.join(save_folder_imagine, f"rendered_{p}.png")
                )

            # poses_habitat = [BeliefAgent.pose_belief2habitat(pose, first_pose_habitat) for pose in poses]
            path_habitat = BeliefAgent.points_belief2habitat(path, first_pose_habitat)
            path_habitat_exe = path_habitat[:len(path_habitat)//4+1]

            # figure out your action/arg keys once
            act_name = "agent_1_oracle_coord_action"
            # prefix = env.task.get_action(act_name)._action_arg_prefix

            trajectory = []
            for goal in path_habitat_exe:
                # choose mode:
                #   0 -> step the agent incrementally along the computed path
                #   1 -> teleport the agent instantly to `goal`
                mode = 1

                action = {
                    "action": act_name,
                    "action_args": {
                        # supply the 3D target
                        f"agent_1_coord": goal.tolist(),
                        # tell the action whether to step or teleport
                        f"agent_1_mode": mode
                    }
                }

                # this single env.step will either drive the agent all the way to 'goal'
                # (mode=0) or teleport it there (mode=1)
                obs = env.step(action)
                trajectory.append((action, obs))
            step+=1

        break

    env_interface.sim.close()


if __name__ == "__main__":
    cprint(
        "\nStart of the belief agent exploration",
        "blue",
    )
    cfg_path = "/scratch/tshu2/yyin34/projects/3d_belief/embodied_belief/DFM/configurations"
    with initialize_config_dir(config_dir=cfg_path, version_base="1.2"):
        cfg = compose(config_name="sp_reason.yaml")
    cfg.checkpoint_path = "/scratch/tshu2/yyin34/projects/3d_belief/DFM/outputs/training/pixelsplat/habitat/full_cond/model-132.pt"
    cfg.results_folder = "/scratch/tshu2/yyin34/projects/3d_belief/embodied_belief/DFM/outputs/belief_agent"
    cfg.semantic_config = "/scratch/tshu2/yyin34/projects/3d_belief/embodied_belief/DFM/configurations/semantic/onehot.yaml"
    cfg.model.encoder.evolve_ctxt = False
    cfg.agent.save_scene = True

    # Run planner
    run_planner(cfg)

    cprint(
        "\nEnd of the belief agent exploration",
        "blue",
    )
