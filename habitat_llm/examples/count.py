import os


def count_entity_folders(dataset_root):
    """Counts the total number of entity folders in the dataset."""
    entity_count = 0

    # Traverse the dataset
    for episode_id in os.listdir(dataset_root):
        episode_path = os.path.join(dataset_root, episode_id, "main_agent")
        if not os.path.isdir(episode_path):
            continue

        # Traverse each room
        for room_name in os.listdir(episode_path):
            room_path = os.path.join(episode_path, room_name)
            if not os.path.isdir(room_path):
                continue

            # Count entity folders in the room
            entity_count += len(
                [
                    name
                    for name in os.listdir(room_path)
                    if os.path.isdir(os.path.join(room_path, name))
                ]
            )

    return entity_count


# Specify the dataset root
dataset_root = "/media/yyin34/ExtremePro/projects/home_robot/partnr-planner/data/trajectories/dataset_small"

# Count entity folders
total_entities = count_entity_folders(dataset_root)
print(f"Total number of entity folders: {total_entities}")
