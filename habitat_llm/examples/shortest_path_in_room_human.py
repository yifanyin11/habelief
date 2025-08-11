import sys
import os
import pathlib
import yaml
import imageio
import random
from copy import deepcopy

ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
print(f"Changed working directory to {ROOT_DIR}")
sys.path.append("..")

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


# Method to load agent planner from the config
def run_planner():
    # Setup a seed
    seed = 47668090
    split = "val"
    num_sequences_per_room = 5

    # setup required overrides
    DATASET_OVERRIDES = [
        f"habitat.dataset.data_path=data/datasets/partnr_episodes/v0_0/{split}.json.gz",
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
        "evaluation.output_dir=/tmp/outputs",
        "trajectory.save=True",
        "trajectory.agent_names=[agent_1]",
        f"trajectory.save_path=data/trajectories/habelief_in_room_human/{split}/",
    ]

    # Setup config
    config_base = get_config(
        "examples/multi_agent_scene_mapping.yaml",
        overrides=DATASET_OVERRIDES
        + SENSOR_OVERRIDES
        + LLM_OVERRIDES
        + TRAJECTORY_OVERRIDES
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

    env_interface = EnvironmentInterface(config, dataset=dataset)

    # Instantiate the agent planner
    eval_runner = CentralizedEvaluationRunner(config.evaluation, env_interface)

    # Load splits
    split_meta_file = config.paths.splits_file_path
    if split_meta_file is not None:
        with open(split_meta_file, "r") as f:
            split_meta = yaml.safe_load(f)
            cprint(f"Loaded splits from {split_meta_file}", "green")
    else:
        cprint(f"No splits file found at {split_meta_file}", "red")

    scenes = split_meta[split]

    # book-keeping and verbosity
    # Highlight the mode of operation
    cprint("\n---------------------------------------", "blue")
    cprint(f"Planner Mode: {config.evaluation.type.capitalize()}", "blue")
    # cprint(f"LLM model: {config.planner.llm.llm._target_}", "blue")
    cprint(f"Partial Observability: {config.world_model.partial_obs}", "blue")
    # Print the agent list
    cprint(f"Agent List: {eval_runner.agent_list}", "blue")
    if env_interface._single_agent_mode:
        cprint("Single agent mode", "green")
    cprint("---------------------------------------\n", "blue")
    human_agent_uid = config.human_agent_uid
    for scene_number in scenes:
        cprint(f"Running routine for scene: {scene_number}", "green")
        episodes = dataset.get_scene_episodes(scene_number)
        if not episodes:
            cprint(f"No episodes found for scene {scene_number}. Skipping.", "red")
            continue
        episode_id = episodes[0].episode_id
        env_interface.env.habitat_env.episode_iterator.set_next_episode_by_id(
            str(episode_id)
        )
        # Reset the environment
        env_interface.reset_environment()
        eval_runner.reset()
        rooms = env_interface.world_graph[human_agent_uid].get_all_nodes_of_type(Room)
        while rooms:
            print(f"{len(rooms)} more room to go...")
            current_room = rooms.pop()
            furns_in_room = env_interface.world_graph[human_agent_uid].get_furniture_in_room(current_room)
            if not furns_in_room:
                print(f"No furniture found in the room {current_room.name}. Skipping.")
                continue
            print(f"Current room: {current_room.name}, contains {len(furns_in_room)} furnitures")
            explored_pairs = set()
            for seq in range(num_sequences_per_room):
                # Shuffle to get an unique furniture sequence
                random.shuffle(furns_in_room)
                furns_in_room_stream = deepcopy(furns_in_room)
                start_furn = furns_in_room_stream.pop()
                while furns_in_room_stream:
                    current_furn = furns_in_room_stream.pop()
                    if "floor" in current_furn.name.lower():
                        continue
                    pair = (start_furn.name, current_furn.name)
                    if pair in explored_pairs:
                        print(f"Pair {pair} already explored. Skipping.")
                        start_furn = current_furn
                        continue
                    explored_pairs.add(pair)
                    print(f"Current furniture: {current_furn.name}")
                    # Set the agent to the start furniture position
                    hl_action_name = "Navigate"
                    hl_action_input = start_furn.name # navigate to the starting pose
                    hl_action_done = False
                    print(f"Set agent to start position")
                    observations = env_interface.get_observations()
                    while not hl_action_done:
                        low_level_action, response = eval_runner.planner.agents[
                            human_agent_uid
                        ].process_high_level_action(
                            hl_action_name, hl_action_input, observations
                        )
                        low_level_action = {human_agent_uid: low_level_action}

                        obs, _, _, _ = env_interface.step(
                            low_level_action
                        )
                        observations = env_interface.parse_observations(obs)
                        
                        if response:
                            print(f"\tResponse: {response}")
                            hl_action_done = True
                    # Execute high-level action to navigate to the current furniture
                    env_interface.set_save_trigger(True)

                    hl_action_name = "Navigate"
                    hl_action_input = current_furn.name
                    hl_action_done = False
                    print(f"Executing high-level action: {hl_action_name} on {hl_action_input}")
                    observations = env_interface.get_observations()
                    video_frames = []
                    
                    while not hl_action_done:
                        low_level_action, response = eval_runner.planner.agents[
                            human_agent_uid
                        ].process_high_level_action(
                            hl_action_name, hl_action_input, observations
                        )
                        low_level_action = {human_agent_uid: low_level_action}

                        obs, _, _, _ = env_interface.step(
                            low_level_action, room_name=current_room.name, furn_name=current_furn.name+ f"-{seq}"
                        )
                        rgb = obs[f"agent_1_{env_interface.conf.trajectory.camera_prefixes[0]}_rgb"]
                        video_frames.append(rgb)
                        observations = env_interface.parse_observations(obs)
                        
                        if response:
                            print(f"\tResponse: {response}")
                            hl_action_done = True
                    
                    video_root = os.path.join(
                        env_interface.trajectory_save_paths[env_interface.trajectory_agent_names[0]],
                        current_room.name,
                        current_furn.name + f"-{seq}",
                    )
                    os.makedirs(video_root, exist_ok=True)
                    video_path = os.path.join(video_root, "navigation.mp4")
                    imageio.mimwrite(video_path, video_frames, fps=10, quality=8)

                    start_furn = current_furn  # Update start_furn for the next iteration

                    # reset the logging
                    env_interface.reset_logging()

if __name__ == "__main__":
    cprint(
        "\nStart of the shortest path routine",
        "blue",
    )

    # Run planner
    run_planner()

    cprint(
        "\nEnd of the single-agent, shortest path routine",
        "blue",
    )
