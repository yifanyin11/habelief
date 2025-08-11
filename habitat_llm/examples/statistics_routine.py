import sys
import os
import pathlib
import yaml
import imageio

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
    split = "train"

    # setup required overrides
    DATASET_OVERRIDES = [
        f"habitat.dataset.data_path=data/datasets/partnr_episodes/v0_0/{split}.json.gz",
        "habitat.dataset.scenes_dir=data/hssd-hab/",
    ]
    SENSOR_OVERRIDES = [
        "habitat.simulator.agents.main_agent.sim_sensors.jaw_depth_sensor.normalize_depth=False"
    ]
    LLM_OVERRIDES = [
        "llm@evaluation.planner.plan_config.llm=mock",
    ]
    TRAJECTORY_OVERRIDES = [
        "evaluation.save_video=True",
        "evaluation.output_dir=/tmp/outputs",
        "trajectory.save=True",
        "trajectory.agent_names=[main_agent]",
        "trajectory.save_path=data/trajectories/test/",
    ]

    # Setup config
    config_base = get_config(
        "examples/single_agent_scene_mapping.yaml",
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

    # Stats
    valid_scene_count = 0
    valid_room_count = 0
    total_episodes = 0

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
    robot_agent_uid = config.robot_agent_uid
    for scene_number in scenes:
        cprint(f"Running routine for scene: {scene_number}", "green")
        episodes = dataset.get_scene_episodes(scene_number)
        if not episodes:
            cprint(f"No episodes found for scene {scene_number}. Skipping.", "red")
            continue
        valid_scene_count += 1
        episode_id = episodes[0].episode_id
        env_interface.env.habitat_env.episode_iterator.set_next_episode_by_id(
            str(episode_id)
        )
        # Reset the environment
        env_interface.reset_environment()
        eval_runner.reset()
        rooms = env_interface.world_graph[robot_agent_uid].get_all_nodes_of_type(Room)
        valid_room_count += len(rooms)
        while rooms:
            print(f"{len(rooms)} more room to go...")
            current_room = rooms.pop()
            furns_in_room = env_interface.world_graph[robot_agent_uid].get_furniture_in_room(current_room)
            print(f"Current room: {current_room.name}, contains {len(furns_in_room)} furnitures")
            while furns_in_room:
                current_furn = furns_in_room.pop()
                if "floor" in current_furn.name.lower():
                    continue
                total_episodes += 1
        
    # Print statistics
    cprint("\n---------------------------------------", "blue")
    cprint(f"Total valid scenes: {valid_scene_count}", "green")
    cprint(f"Total valid rooms: {valid_room_count}", "green")
    cprint(f"Total episodes processed: {total_episodes}", "green")
    cprint("---------------------------------------\n", "blue")

if __name__ == "__main__":
    cprint(
        "\nStart of the statistics routine",
        "blue",
    )

    # Run planner
    run_planner()

    cprint(
        "\nEnd of the single-agent, statistics routine",
        "blue",
    )
