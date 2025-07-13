# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import List

import habitat_sim
import magnum as mn
import numpy as np
import torch
from habitat.datasets.rearrange.navmesh_utils import (
    SimpleVelocityControlEnv,
    compute_turn,
    embodied_unoccluded_navmesh_snap,
)
from habitat.tasks.utils import get_angle

from habitat_llm.agent.env.actions import find_action_range
from habitat_llm.tools.motor_skills.skill import SkillPolicy
from habitat_llm.utils.grammar import NAV_POSE


class OracleNavPoseSkill(SkillPolicy):
    def __init__(
        self, config, observation_space, action_space, batch_size, env, agent_uid
    ):
        super().__init__(
            config,
            action_space,
            batch_size,
            should_keep_hold_state=True,
            agent_uid=agent_uid,
        )
        self.env = env
        # TODO: there may be cleaner ways to do this
        if f"agent_{self.agent_uid}_humanoidjoint_action" in action_space.spaces:
            self.motion_type = "human_joints"
        else:
            self.motion_type = "base_velocity"

        # pre-computed target pose for the ArticulatedAgent. See set_target.
        self.target_base_pos: mn.Vector3 = None
        self.target_base_rot: float = None
        self.facing_direction: float = None  # New variable to store facing direction
        self._has_reached_goal = torch.zeros(self._batch_size)

        # Define the velocity controller
        self.base_vel_ctrl = habitat_sim.physics.VelocityControl()
        self.base_vel_ctrl.controlling_lin_vel = True
        self.base_vel_ctrl.lin_vel_is_local = True
        self.base_vel_ctrl.controlling_ang_vel = True
        self.base_vel_ctrl.ang_vel_is_local = True

        self.dist_thresh = config.dist_thresh
        self.turn_thresh = config.turn_thresh
        self.forward_velocity = config.forward_velocity
        self.turn_velocity = config.turn_velocity
        self.sim_freq = config.sim_freq

        self.enable_backing_up = config.enable_backing_up

        # Get articulated agent
        self.articulated_agent = self.env.sim.agents_mgr[
            self.agent_uid
        ].articulated_agent

        self.face_to_obj = False

        self.do_teleport = False
        if "teleport" in config:
            self.do_teleport = config.teleport

        # Get indices for teleport action
        target_pos_ends = find_action_range(
            self.action_space, f"agent_{self.agent_uid}_teleport"
        )
        self.target_pos_range = range(target_pos_ends[0], target_pos_ends[1])

        # Get indices for linear and angular velocities in the action tensor
        if self.motion_type != "human_joints":
            self.action_range = find_action_range(
                self.action_space, f"agent_{self.agent_uid}_base_velocity"
            )
        else:
            self.action_range = find_action_range(
                self.action_space, f"agent_{self.agent_uid}_humanoid_base_velocity"
            )
        self.linear_velocity_index = self.action_range[0]
        self.angular_velocity_index = self.action_range[1] - 1

    def reset(self, batch_idxs):
        super().reset(batch_idxs)
        self._has_reached_goal = torch.zeros(self._batch_size)
        self.target_is_set = False
        self.target_base_pos = None
        self.target_base_rot = None
        self.facing_direction = None  # Reset facing direction
        return

    def get_state_description(self):
        """Method to get a string describing the state for this tool"""

        # Get room for agent
        room_node = self.env.world_graph[self.agent_uid].get_room_for_entity(
            f"agent_{self.agent_uid}"
        )
        return f"Walking to a specific pose in {room_node.name}"

    def _path_to_point(self, point):
        """
        Obtain path to reach the coordinate point. If agent_pos is not given
        the path starts at the agent base pos, otherwise it starts at the agent_pos
        value
        :param point: Vector3 indicating the target point
        """
        agent_pos = self.articulated_agent.base_pos

        path = habitat_sim.ShortestPath()
        path.requested_start = agent_pos
        path.requested_end = point
        found_path = self.env.sim.pathfinder.find_path(path)
        if not found_path:
            return [agent_pos, point]
        return path.points

    def is_collision(self, trans) -> bool:
        """
        The function checks if the agent collides with the object
        given the navmesh.
        """
        nav_pos_3d = [
            np.array([xz[0], 0.0, xz[1]])
            for xz in self.articulated_agent.params.navmesh_offsets
        ]  # type: ignore
        cur_pos = [trans.transform_point(xyz) for xyz in nav_pos_3d]
        cur_pos = [
            np.array([xz[0], self.articulated_agent.base_pos[1], xz[2]])
            for xz in cur_pos
        ]

        for pos in cur_pos:  # noqa: SIM110
            # Return true if the pathfinder says it is not navigable
            if not self.env.sim.pathfinder.is_navigable(pos):
                return True

        return False

    def fix_robot_leg(self):
        """
        Fix the robot leg's joint position
        """
        self.articulated_agent.leg_joint_pos = (
            self.articulated_agent.params.leg_init_params
        )

    def set_target(self, target_position, env):
        """
        Set the target position and calculate the optimal robot base position and rotation using embodied_unoccluded_navmesh_snap.

        :param target_position: The target position (array-like, will be converted to mn.Vector3), 
                               face_to_obj flag, teleport flag, and optional facing direction.
        """
        # Early return if the target is already set
        if self.target_is_set:
            return
        
        # Unpack parameters, now with an optional facing_direction parameter
        if len(target_position) == 4:
            target_position, face_to_obj, teleport, facing_direction = target_position
            self.facing_direction = facing_direction  # Store the facing direction
        else:
            target_position, face_to_obj, teleport = target_position
            self.facing_direction = None
            
        self.face_to_obj = face_to_obj
        self.do_teleport = teleport

        # Convert target_position to mn.Vector3 if it's not already
        if not isinstance(target_position, mn.Vector3):
            try:
                # Handle various array-like formats
                if hasattr(target_position, "__len__") and len(target_position) >= 3:
                    target_position = mn.Vector3(
                        target_position[0], target_position[1], target_position[2]
                    )
                else:
                    self.termination_message = f"Invalid target position format: {target_position}"
                    self.failed = True
                    return
            except Exception as e:
                self.termination_message = f"Error converting target position: {e}"
                self.failed = True
                return
        # Set the target position
        self.target_pos = target_position

        # Set flag to True to avoid resetting the target
        self.target_is_set = True

        # Get the object_id for all links associated with all articulated agents so they can be ignored in navigation placement sampling
        agent_object_ids = []
        other_agent_object_ids = []
        for articulated_agent in self.env.sim.agents_mgr.articulated_agents_iter:
            agent_object_ids.extend(
                [articulated_agent.sim_obj.object_id]
                + [*articulated_agent.sim_obj.link_object_ids.keys()]
            )
            if articulated_agent != self.articulated_agent:
                other_agent_object_ids = [articulated_agent.sim_obj.object_id] + [
                    *articulated_agent.sim_obj.link_object_ids.keys()
                ]

        # Try multiple times to find a valid navigation target
        attempts = 0
        # Track the nav point to target distance as we do rejection sampling
        pose_to_nav_point_dist = 0
        success = False
        # Maximum acceptable distance between target pose and navigation point
        max_pose_to_nav_point_dist = 1.8

        # Try up to 200 times to find a valid navigation target
        while (
            pose_to_nav_point_dist > max_pose_to_nav_point_dist or not success
        ) and attempts < 200:
            # Find the optimal robot base position and rotation
            (
                self.target_base_pos,
                self.target_base_rot,
                success,
            ) = embodied_unoccluded_navmesh_snap(
                target_position=target_position,
                height=1.3,  # TODO: hardcoded everywhere, should be config
                sim=self.env.sim,
                ignore_object_ids=agent_object_ids,  # ignore the agent's body in occlusion checking
                ignore_object_collision_ids=other_agent_object_ids,  # ignore the other agent's body in contact testing
                island_id=self.env.sim._largest_indoor_island_idx,  # from RearrangeSim
                min_sample_dist=0.25,  # approximates agent radius, doesn't need to be precise
                agent_embodiment=self.articulated_agent,
                orientation_noise=0.0,  # allow a bit of variation in body orientation
            )
            if success:
                pose_to_nav_point_dist = (
                    mn.Vector3(self.target_base_pos) - mn.Vector3(self.target_pos)
                ).length()
            attempts += 1

        if success and pose_to_nav_point_dist <= max_pose_to_nav_point_dist:
            # Make target visible in simulator for debugging
            self.env.sim.dynamic_target = self.target_base_pos
            return
        else:
            # Failed to find a suitable navigation target
            self.termination_message = f"Could not find a suitable nav target for position {target_position}. Possibly inaccessible."
            self.failed = True
            return

    def rotation_collision_check(
        self,
        next_pos,
    ):
        """
        This function checks if the robot needs to do backing-up action
        """
        # Make a copy of agent trans
        trans = mn.Matrix4(self.articulated_agent.sim_obj.transformation)
        # Initialize the velocity controller
        vc = SimpleVelocityControlEnv(120.0)
        angle = float("inf")
        # Get the current location of the agent
        cur_pos = self.articulated_agent.base_pos
        # Set the trans to be agent location
        trans.translation = self.articulated_agent.base_pos

        while abs(angle) > self.turn_thresh:
            # Compute the robot facing orientation
            # Convert Vector3 objects to numpy arrays before indexing
            rel_pos = np.array(
                [next_pos[0] - cur_pos[0], next_pos[2] - cur_pos[2]], dtype=np.float32
            )
            forward = np.array([1.0, 0, 0])
            robot_forward = np.array(trans.transform_vector(forward))
            robot_forward = robot_forward[[0, 2]]
            angle = get_angle(robot_forward, rel_pos)
            vel = compute_turn(rel_pos, self.turn_velocity, robot_forward)
            trans = vc.act(trans, vel)
            cur_pos = trans.translation

            if self.is_collision(trans):
                return True

        return False

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        # We do not feed any velocity command
        action = torch.zeros(prev_actions.shape, device=masks.device)

        if self.failed:
            return action, self.termination_message

        if self.target_base_rot is None:
            self.failed = True
            return action, "Target pose not set properly."

        # The location of the target position
        obj_targ_pos = np.array(self.target_pos)

        if self.do_teleport:
            # One hot indicator stating that agent should take teleport action.
            action[cur_batch_idx, self.target_pos_range[0]] = 1.0

            # Set the agent's base_pos and base_rot from the pre-computed target pose
            action[cur_batch_idx, self.target_pos_range[1:4]] = torch.tensor(
                self.target_base_pos, dtype=torch.float32
            ).to(action.device)
            
            # Use facing_direction if provided, otherwise use target_base_rot
            if self.facing_direction is not None:
                action[cur_batch_idx, self.target_pos_range[4]] = self.facing_direction
            else:
                action[cur_batch_idx, self.target_pos_range[4]] = self.target_base_rot

            # teleported agent has reached the goal
            self._has_reached_goal[cur_batch_idx] = 1
            return action, None

        # Compute the shortest path from the current position to the target position
        # Get the base transformation for the robot
        base_T = self.articulated_agent.base_transformation
        # Find the paths
        curr_path_points = self._path_to_point(self.target_base_pos)
        # Get the robot position
        robot_pos = np.array(self.articulated_agent.base_pos)

        if curr_path_points is None:
            raise RuntimeError("Pathfinder returns empty list")

        # Compute distance and angle to target
        if len(curr_path_points) == 1:
            curr_path_points += curr_path_points

        cur_nav_targ = curr_path_points[1]
        forward = np.array([1.0, 0, 0])
        robot_forward = np.array(base_T.transform_vector(forward))

        # Compute relative target
        rel_targ = cur_nav_targ - robot_pos

        # Compute heading angle (2D calculation)
        robot_forward = robot_forward[[0, 2]]
        rel_targ = np.array([rel_targ[0], rel_targ[2]], dtype=np.float32)
        rel_pos = (obj_targ_pos - robot_pos)[[0, 2]]
        # Get the angles
        angle_to_target = get_angle(robot_forward, rel_targ)
        angle_to_obj = get_angle(robot_forward, rel_pos)
        # Compute the distance
        robot_pos_np = np.array(robot_pos, dtype=np.float32)
        target_base_pos_np = np.array(self.target_base_pos, dtype=np.float32)
        delta = target_base_pos_np - robot_pos_np
        dist_to_final_nav_targ = np.linalg.norm(delta[[0, 2]])
        if self.face_to_obj:
            at_goal = (
                dist_to_final_nav_targ < self.dist_thresh
                and angle_to_obj < self.turn_thresh
            )
        else:
            at_goal = (
                dist_to_final_nav_targ < self.dist_thresh
            )

        if self.motion_type == "base_velocity":
            # Planning to see if the robot needs to do back-up
            need_move_backward = False
            if (
                dist_to_final_nav_targ >= self.dist_thresh
                and angle_to_target >= self.turn_thresh
                and not at_goal
            ):
                # check if there is a collision caused by rotation
                # if it does, we should block the rotation, and
                # only move backward
                need_move_backward = self.rotation_collision_check(
                    cur_nav_targ,
                )

            if need_move_backward and self.enable_backing_up:
                # Backward direction
                forward = np.array([-1.0, 0, 0])
                robot_forward = np.array(base_T.transform_vector(forward))
                # Compute relative target
                rel_targ = cur_nav_targ - robot_pos
                # Compute heading angle (2D calculation)
                robot_forward = robot_forward[[0, 2]]
                rel_targ = np.array([rel_targ[0], rel_targ[2]], dtype=np.float32)
                rel_pos = (obj_targ_pos - robot_pos)[[0, 2]]
                # Get the angles
                angle_to_target = get_angle(robot_forward, rel_targ)
                angle_to_obj = get_angle(robot_forward, rel_pos)
                # Compute the distance
                robot_pos_np = np.array(robot_pos, dtype=np.float32)
                target_base_pos_np = np.array(self.target_base_pos, dtype=np.float32)
                delta = target_base_pos_np - robot_pos_np
                dist_to_final_nav_targ = np.linalg.norm(delta[[0, 2]])
                if self.face_to_obj:
                    at_goal = (
                        dist_to_final_nav_targ < self.dist_thresh
                        and angle_to_obj < self.turn_thresh
                    )
                else:
                    at_goal = (
                        dist_to_final_nav_targ < self.dist_thresh
                    )

            if not at_goal:
                if dist_to_final_nav_targ < self.dist_thresh:
                    # TODO: this does not account for the sampled pose's final rotation
                    # Look at the object target position when getting close
                    vel = compute_turn(
                        rel_pos,
                        self.turn_velocity,
                        robot_forward,
                    )
                elif angle_to_target < self.turn_thresh:
                    # Move forward towards the target
                    vel = [self.forward_velocity, 0]
                else:
                    # Look at the target waypoint
                    vel = compute_turn(
                        rel_targ,
                        self.turn_velocity,
                        robot_forward,
                    )
                self._has_reached_goal[cur_batch_idx] = 0.0
            else:
                vel = [0, 0]
                self._has_reached_goal[cur_batch_idx] = 1.0

            if need_move_backward:
                vel[0] = -1 * vel[0]

            # # Reset the robot's leg joints
            # self.fix_robot_leg()

            # Populate the actions tensor
            action[cur_batch_idx, self.linear_velocity_index] = vel[0]
            action[cur_batch_idx, self.angular_velocity_index] = vel[1]
        else:
            if not at_goal:
                if dist_to_final_nav_targ < self._config.dist_thresh:
                    # Look at the object
                    vel = compute_turn(
                        rel_pos,
                        self.turn_velocity,
                        robot_forward,
                    )
                elif angle_to_target < self.turn_thresh:
                    # Move forward towards the target
                    vel = [self.forward_velocity, 0]
                else:
                    # Look at the target waypoint
                    vel = compute_turn(
                        rel_targ,
                        self.turn_velocity,
                        robot_forward,
                    )
                self._has_reached_goal[cur_batch_idx] = 0.0
            else:
                vel = [0, 0]
                self._has_reached_goal[cur_batch_idx] = 1.0

            # Populate the actions tensor
            action[cur_batch_idx, self.linear_velocity_index] = vel[0]
            action[cur_batch_idx, self.angular_velocity_index] = vel[1]

        return action, None

    def _is_skill_done(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        batch_idx,
    ) -> torch.BoolTensor:
        return (self._has_reached_goal[batch_idx] > 0.0).to(masks.device)

    @property
    def argument_types(self) -> List[str]:
        """
        Returns the types of arguments required for the OracleNavPoseSkill.

        :return: List of argument types.
        """
        return [NAV_POSE]
