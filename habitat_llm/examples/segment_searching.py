import os
import shutil
from random import randint, random

MAX_ENTITIES_EACH_ROOM = 5


def parse_world_description(file_path):
    """Parses a world description file."""
    with open(file_path, "r") as file:
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


def weighted_sample_without_replacement(entities, weights, k):
    """Samples without replacement based on weights."""
    assert len(entities) == len(
        weights
    ), "Entities and weights must have the same length"
    assert k <= len(entities), "Cannot sample more items than available entities"

    # Pair entities with their weights and shuffle randomly to break ties
    weighted_entities = list(zip(entities, weights))
    weighted_entities.sort(key=lambda x: random() * x[1], reverse=True)

    # Select the top-k entities
    return [entity for entity, _ in weighted_entities[:k]]


def save_entity_description(segment_dir, entity_name, entity_type, location):
    """Saves a description file for the entity."""
    desc_file = os.path.join(segment_dir, "entity_desc.txt")
    if entity_type == "furniture":
        description = f"{entity_name} is in {location}."
    elif entity_type == "object":
        description = f"{entity_name} is on {location}."
    else:
        description = f"No specific spatial relationship found for {entity_name}."

    with open(desc_file, "w") as f:
        f.write(description)


def save_new_episode(
    new_root, episode_id, room_name, entity_name, segment_frames, segment_data
):
    """Saves a new segmented episode with the updated structure."""
    segment_dir = os.path.join(
        new_root, episode_id, "main_agent", room_name, entity_name
    )
    os.makedirs(segment_dir, exist_ok=True)

    # Create modality folders
    modality_folders = {
        modality: os.path.join(segment_dir, modality)
        for modality in ["depth", "panoptic", "pose", "rgb", "world_desc"]
    }
    for folder in modality_folders.values():
        os.makedirs(folder, exist_ok=True)

    # Save files for each modality
    for frame, frame_data in zip(segment_frames, segment_data):
        for modality, file_path in frame_data.items():
            # Split modality and extension safely
            modality_type, ext = modality.rsplit(
                "_", 1
            )  # Use rsplit to split only on the last underscore
            save_path = os.path.join(modality_folders[modality_type], f"{frame}.{ext}")
            shutil.copy(file_path, save_path)

    return segment_dir  # Return the path to the segment directory


def generate_segments(data_root, new_root):
    """Processes the dataset and generates segmented episodes."""
    for episode_id in os.listdir(data_root):
        print(f"Processing episode {episode_id}...")
        episode_path = os.path.join(data_root, episode_id, "main_agent")

        if not os.path.isdir(episode_path):
            continue

        room_names = [
            name
            for name in os.listdir(episode_path)
            if os.path.isdir(os.path.join(episode_path, name))
        ]
        for room_name in room_names:
            print(f"\tProcessing room {room_name}...")
            room_path = os.path.join(episode_path, room_name)
            world_desc_path = os.path.join(room_path, "world_desc")

            # Collect all frame indices and descriptions
            frame_indices = sorted(
                [
                    int(f.split(".")[0])
                    for f in os.listdir(world_desc_path)
                    if f.endswith(".txt")
                ]
            )
            descriptions = {}
            for frame_idx in frame_indices:
                descriptions[frame_idx] = parse_world_description(
                    os.path.join(world_desc_path, f"{frame_idx}.txt")
                )

            # Get changes in descriptions
            first_desc = descriptions[frame_indices[0]]
            last_desc = descriptions[frame_indices[-1]]

            furniture_diff = {
                room: [f for f in items if f not in first_desc[0].get(room, [])]
                for room, items in last_desc[0].items()
            }
            objects_diff = {
                obj: loc
                for obj, loc in last_desc[1].items()
                if obj not in first_desc[1]
            }

            # Adjust weights for furniture and objects
            object_weight = 3  # Higher weight for objects
            furniture_weight = 1  # Lower weight for furniture

            # Generate a weighted list of entities
            object_candidates = list(objects_diff.keys())
            furniture_candidates = [
                furn for items in furniture_diff.values() for furn in items
            ]

            if object_candidates or furniture_candidates:
                # Prepare entities and weights
                entities = object_candidates + furniture_candidates
                weights = [object_weight] * len(object_candidates) + [
                    furniture_weight
                ] * len(furniture_candidates)

                # Sample entities without replacement
                focus_entities = set(
                    weighted_sample_without_replacement(
                        entities, weights, min(len(entities), MAX_ENTITIES_EACH_ROOM)
                    )
                )
                print(f"\t\tEntities to focus on: {focus_entities}")
            else:
                focus_entities = set()  # No entities to focus on

            max_frame_idx = len(frame_indices) - 1
            for entity in focus_entities:
                start_frame, end_frame = None, None
                for frame_idx in frame_indices:
                    desc = descriptions[frame_idx]
                    if (
                        entity
                        not in [furn for items in desc[0].values() for furn in items]
                        and entity not in desc[1].keys()
                    ):
                        start_frame = frame_idx
                    if (
                        entity in [furn for items in desc[0].values() for furn in items]
                        or entity in desc[1]
                    ):
                        end_frame = frame_idx
                        break

                if start_frame is not None and end_frame is not None:
                    transition = end_frame
                    end_of_interest = end_frame + randint(
                        0, abs(max_frame_idx - transition)
                    )
                    start_of_interest = start_frame - randint(0, abs(0 - start_frame))
                    segment_frames = [
                        f
                        for f in frame_indices
                        if start_of_interest <= f <= end_of_interest
                    ]

                    # Collect segment data
                    segment_data = []
                    for frame in segment_frames:
                        frame_data = {}
                        for modality in [
                            "depth",
                            "panoptic",
                            "pose",
                            "rgb",
                            "world_desc",
                        ]:
                            modality_dir = os.path.join(room_path, modality)

                            if modality in ["depth", "panoptic"]:
                                # Save both .npy and .png for depth and panoptic
                                frame_data[f"{modality}_npy"] = os.path.join(
                                    modality_dir, f"{frame}.npy"
                                )
                                frame_data[f"{modality}_png"] = os.path.join(
                                    modality_dir, f"{frame}.png"
                                )
                            elif modality == "pose":
                                # Save only .npy for pose
                                frame_data[f"{modality}_npy"] = os.path.join(
                                    modality_dir, f"{frame}.npy"
                                )
                            elif modality == "rgb":
                                # Save both .npy and .jpg for rgb
                                frame_data[f"{modality}_npy"] = os.path.join(
                                    modality_dir, f"{frame}.npy"
                                )
                                frame_data[f"{modality}_jpg"] = os.path.join(
                                    modality_dir, f"{frame}.jpg"
                                )
                            elif modality == "world_desc":
                                # Save only .txt for world_desc
                                frame_data[f"{modality}_txt"] = os.path.join(
                                    modality_dir, f"{frame}.txt"
                                )

                        segment_data.append(frame_data)

                    # Save the segment
                    segment_dir = save_new_episode(
                        new_root,
                        episode_id,
                        room_name,
                        entity,
                        segment_frames,
                        segment_data,
                    )

                    # Determine entity type and location for description
                    if entity in objects_diff:
                        entity_type = "object"
                        location = objects_diff[entity]
                    else:
                        entity_type = "furniture"
                        location = next(
                            (
                                room
                                for room, items in furniture_diff.items()
                                if entity in items
                            ),
                            "unknown_room",
                        )

                    # Save entity description
                    save_entity_description(segment_dir, entity, entity_type, location)


# Directories
data_root = "/media/yyin34/ExtremePro/projects/home_robot/partnr-planner/data/trajectories/small"
new_root = "/media/yyin34/ExtremePro/projects/home_robot/partnr-planner/data/trajectories/dataset_small"

# Generate the segments
generate_segments(data_root, new_root)
