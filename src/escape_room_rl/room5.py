"""Room 5: a configurable one-way multi-lane driving environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


MAX_LANES = 6
OBSERVATION_SIZE = (MAX_LANES * 3) + 1
RoadObservation = tuple[float, ...]


class Action5(IntEnum):
    LEFT = 0
    KEEP_LANE = 1
    RIGHT = 2


@dataclass(frozen=True)
class TrafficCar:
    car_id: int
    lane: int
    distance: float
    speed: float


@dataclass(frozen=True)
class RoadSnapshot:
    progress: float
    ego_lane: int
    traffic: tuple[TrafficCar, ...]


DEFAULT_ROOM5_REWARDS: dict[str, float] = {
    "step": -0.02,
    "forward_progress": 0.02,
    "overtake": 8.0,
    "lane_change": -0.05,
    "invalid_lane_change": -0.5,
    "collision": -40.0,
    "goal_reached": 40.0,
}


@dataclass
class Room5Config:
    lane_count: int = 4
    vision_distance: float = 120.0
    road_length: float = 600.0
    dt: float = 0.2
    ego_speed: float = 30.0
    traffic_speed_min: float = 12.0
    traffic_speed_max: float = 24.0
    traffic_count: int = 10
    car_length: float = 4.5
    rewards: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_ROOM5_REWARDS)
    )
    seed: int = 42

    def validate(self) -> None:
        if not 2 <= self.lane_count <= MAX_LANES:
            raise ValueError("lane_count must be between 2 and 6.")
        if self.vision_distance <= 0:
            raise ValueError("vision_distance must be positive.")
        if self.road_length <= 0 or self.dt <= 0:
            raise ValueError("road_length and dt must be positive.")
        if self.ego_speed <= 0:
            raise ValueError("ego_speed must be positive.")
        if not 0 < self.traffic_speed_min <= self.traffic_speed_max:
            raise ValueError("Invalid traffic speed range.")
        if self.traffic_speed_max >= self.ego_speed:
            raise ValueError("Traffic must be slower than the agent car.")
        if self.traffic_count <= 0:
            raise ValueError("traffic_count must be positive.")
        if self.car_length <= 0:
            raise ValueError("car_length must be positive.")


@dataclass(frozen=True)
class Room5Transition:
    next_state: RoadObservation
    reward: float
    done: bool
    events: tuple[str, ...]
    snapshot: RoadSnapshot


class Room5Environment:
    """Same-direction traffic that approaches in the faster ego car's frame."""

    observation_size = OBSERVATION_SIZE
    action_size = len(Action5)

    def __init__(self, config: Room5Config | None = None):
        self.config = config or Room5Config()
        self.config.validate()
        self._rng = np.random.default_rng(self.config.seed)
        self.ego_lane = self.config.lane_count // 2
        self.progress = 0.0
        self.traffic: list[TrafficCar] = []
        self._next_car_id = 0
        self.reset(self.config.seed)

    def reset(self, seed: int | None = None) -> RoadObservation:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.ego_lane = self.config.lane_count // 2
        self.progress = 0.0
        self._next_car_id = 0
        self.traffic = []

        base_distances = np.linspace(
            max(18.0, self.config.car_length * 3.0),
            self.config.vision_distance * 1.45,
            self.config.traffic_count,
        )
        jitter_limit = min(4.0, self.config.vision_distance * 0.025)
        for distance in base_distances:
            self.traffic.append(
                self._new_car(
                    distance=float(distance + self._rng.uniform(-jitter_limit, jitter_limit))
                )
            )
        return self.observation()

    def _new_car(self, distance: float) -> TrafficCar:
        car = TrafficCar(
            car_id=self._next_car_id,
            lane=int(self._rng.integers(0, self.config.lane_count)),
            distance=float(distance),
            speed=float(
                self._rng.uniform(
                    self.config.traffic_speed_min,
                    self.config.traffic_speed_max,
                )
            ),
        )
        self._next_car_id += 1
        return car

    def snapshot(self) -> RoadSnapshot:
        return RoadSnapshot(
            progress=float(self.progress),
            ego_lane=int(self.ego_lane),
            traffic=tuple(sorted(self.traffic, key=lambda car: car.distance)),
        )

    def observation(self) -> RoadObservation:
        one_hot_lane = [0.0] * MAX_LANES
        one_hot_lane[self.ego_lane] = 1.0

        distances: list[float] = []
        closing_speeds: list[float] = []
        for lane in range(MAX_LANES):
            if lane >= self.config.lane_count:
                distances.append(-1.0)
                closing_speeds.append(-1.0)
                continue

            visible = [
                car
                for car in self.traffic
                if car.lane == lane
                and 0.0 <= car.distance <= self.config.vision_distance
            ]
            if not visible:
                distances.append(1.0)
                closing_speeds.append(0.0)
                continue

            nearest = min(visible, key=lambda car: car.distance)
            distances.append(float(nearest.distance / self.config.vision_distance))
            closing_speeds.append(
                float((self.config.ego_speed - nearest.speed) / self.config.ego_speed)
            )

        progress = min(1.0, self.progress / self.config.road_length)
        return tuple(one_hot_lane + distances + closing_speeds + [float(progress)])

    def step(self, action: Action5) -> Room5Transition:
        action = Action5(action)
        events: list[str] = ["step"]
        reward = float(self.config.rewards.get("step", -0.02))

        previous_lane = self.ego_lane
        if action == Action5.LEFT:
            if self.ego_lane > 0:
                self.ego_lane -= 1
            else:
                events.append("invalid_lane_change")
                reward += self.config.rewards.get("invalid_lane_change", -0.5)
        elif action == Action5.RIGHT:
            if self.ego_lane < self.config.lane_count - 1:
                self.ego_lane += 1
            else:
                events.append("invalid_lane_change")
                reward += self.config.rewards.get("invalid_lane_change", -0.5)

        if self.ego_lane != previous_lane:
            events.append("lane_changed")
            reward += self.config.rewards.get("lane_change", -0.05)

        forward_distance = self.config.ego_speed * self.config.dt
        self.progress += forward_distance
        reward += forward_distance * self.config.rewards.get("forward_progress", 0.02)
        events.append("forward_progress")

        updated_traffic: list[TrafficCar] = []
        overtaken = 0
        collision = False
        farthest_distance = max(
            [car.distance for car in self.traffic]
            + [self.config.vision_distance]
        )

        for car in self.traffic:
            closing_distance = (self.config.ego_speed - car.speed) * self.config.dt
            next_distance = car.distance - closing_distance
            if (
                car.lane == self.ego_lane
                and abs(next_distance) <= self.config.car_length
            ):
                collision = True

            if next_distance < -self.config.car_length:
                overtaken += 1
                farthest_distance += float(
                    self._rng.uniform(
                        max(12.0, self.config.car_length * 2.5),
                        max(25.0, self.config.vision_distance * 0.3),
                    )
                )
                updated_traffic.append(self._new_car(farthest_distance))
            else:
                updated_traffic.append(
                    TrafficCar(
                        car_id=car.car_id,
                        lane=car.lane,
                        distance=float(next_distance),
                        speed=car.speed,
                    )
                )

        self.traffic = updated_traffic
        if overtaken:
            events.extend(["overtake"] * overtaken)
            reward += overtaken * self.config.rewards.get("overtake", 8.0)

        done = False
        if collision:
            done = True
            events.append("collision")
            reward += self.config.rewards.get("collision", -40.0)
        elif self.progress >= self.config.road_length:
            done = True
            events.append("goal_reached")
            reward += self.config.rewards.get("goal_reached", 40.0)

        return Room5Transition(
            next_state=self.observation(),
            reward=float(reward),
            done=done,
            events=tuple(events),
            snapshot=self.snapshot(),
        )
