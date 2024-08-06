"""
DEVELOPED BY:
Group: el_grupo_87

Group members:
1. André Moreira Lopes : 20230570
2. Luís Queiroz : 20230584
3. André Filipe Silva : 20230972
4. Pedro Cerejeira : 20230442
5. João Gonçalves : 20230560
"""

from typing import Dict, Text

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

Observation = np.ndarray


class CustomHighwayEnv(AbstractEnv):

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,  # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
            # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
            # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,  # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "normalize_reward": True,
            "offroad_terminal": False
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()
        self.previous_lane_index = self.vehicle.lane_index[2]  # Track the initial lane index

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config['normalize_reward']:
            reward = utils.lmap(reward,
                                [self.config['collision_reward'],
                                 self.config['speed_reward'] + self.config['right_lane_reward']],
                                [-float(self.config['slower_than_others_penalty']), 1])
            # y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
        reward *= rewards['on_road_reward']
        return reward

    def _rewards(self, action) -> Dict[Text, float]:
        _, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)

        # Define individual reward components here
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]

        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed,
                                  self.config['reward_speed_range'],
                                  [-float(self.config['slower_than_others_penalty']), 1]
                                  )
        # utils.lmap(v,x,y) = y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])

        if forward_speed <= 0.0:
            self.stationary_penalty_counter += 1
        else:
            self.stationary_penalty_counter = 0

        # Check for lane change
        if lane != self.previous_lane_index:
            self.previous_lane_index = lane  # Update the previous lane index

        return {
            'collision_reward': float(self.vehicle.crashed),
            'right_lane_reward': lane / max(len(neighbours) - 1, 1),
            # 'high_speed_reward': np.clip(scaled_speed, 0, 1),
            'on_road_reward': float(self.vehicle.on_road),

            'lane_centering_reward': 1 / (1 + self.config['lane_centering_cost'] * lateral ** 2),
            'speed_reward': np.clip(scaled_speed, -float(self.config['slower_than_others_penalty']), 1),
            'offroad_penalty': abs(1 - float(self.vehicle.on_road)),
            'lane_change_penalty': float(lane != self.previous_lane_index),
            'stationary_penalty': float(self.stationary_penalty_counter > 32),
            'action_reward': np.linalg.norm(action)
        }

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (self.vehicle.crashed or
                self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]
