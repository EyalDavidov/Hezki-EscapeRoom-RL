"""Room 5: a configurable one-way multi-lane driving environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


MAX_LANES = 6
MIN_TRAFFIC_CLEARANCE_METERS = 3.0
SPAWN_DISTANCE = 120.0
# Six values identify the agent's own lane. The remaining 7 values describe
# the left, current, and right lanes (distance & closing speed), plus road progress.
OBSERVATION_SIZE = MAX_LANES + 7
RoadObservation = tuple[float, ...]


class Action5(IntEnum):
    LEFT = 0
    KEEP_LANE = 1
    RIGHT = 2


@dataclass(frozen=True)
class TrafficCar:
    car_id: int
    lane: int
    # Longitudinal center-to-center offset from the ego car. User-facing
    # distances are converted to physical edge-to-edge clearance.
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
    "safer_lane_change": 1.0,
    "riskier_lane_change": -1.0,
    "invalid_lane_change": -0.5,
    "collision": -40.0,
    "goal_reached": 40.0,
}


@dataclass
class Room5Config:
    lane_count: int = 4
    vision_distance: float = 50.0
    road_length: float = 50.0
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
        if not 0 <= self.traffic_speed_min <= self.traffic_speed_max:
            raise ValueError("Invalid traffic speed range.")
        if self.traffic_speed_max >= self.ego_speed:
            raise ValueError("Traffic must be slower than the agent car.")
        if self.traffic_count < 0:
            raise ValueError("traffic_count cannot be negative.")
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
        self.elapsed_time = 0.0
        self._next_spawn_time = 1.0
        self.reset(self.config.seed)

    def reset(self, seed: int | None = None) -> RoadObservation:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.ego_lane = self.config.lane_count // 2
        self.progress = 0.0
        self._next_car_id = 0
        self.elapsed_time = 0.0
        self.traffic = []
        for _ in range(self.config.traffic_count):
            dist = float(self._rng.uniform(20.0, SPAWN_DISTANCE))
            self._spawn_arriving_car(distance=dist)
        self.traffic = self._enforce_minimum_traffic_clearance(self.traffic)
        return self.observation()

    def _spawn_arriving_car(self, distance: float | None = None) -> None:
        # Pick the lane with the fewest cars currently to ensure even distribution
        lane_counts = [
            sum(1 for car in self.traffic if car.lane == l)
            for l in range(self.config.lane_count)
        ]
        min_count = min(lane_counts)
        candidate_lanes = [l for l, c in enumerate(lane_counts) if c == min_count]
        lane = int(self._rng.choice(candidate_lanes))
        
        if distance is None:
            cars_in_lane = [car.distance for car in self.traffic if car.lane == lane]
            farthest = max(cars_in_lane, default=SPAWN_DISTANCE)
            spacing = float(self._rng.uniform(10.0, 30.0))
            distance = max(SPAWN_DISTANCE, farthest + spacing)

        car = TrafficCar(
            car_id=self._next_car_id,
            lane=lane,
            distance=float(distance),
            speed=float(
                self._rng.uniform(
                    self.config.traffic_speed_min,
                    self.config.traffic_speed_max,
                )
            ),
        )
        self._next_car_id += 1
        self.traffic.append(car)

    def forward_clearance(self, car: TrafficCar) -> float:
        """Return the distance from the ego rear edge to the traffic front edge."""
        return float(car.distance + self.config.car_length)

    def _enforce_minimum_traffic_clearance(
        self,
        cars: list[TrafficCar],
    ) -> list[TrafficCar]:
        """Prevent same-lane traffic cars from getting closer than three meters."""
        minimum_center_spacing = (
            self.config.car_length + MIN_TRAFFIC_CLEARANCE_METERS
        )
        constrained: list[TrafficCar] = []
        for lane in range(self.config.lane_count):
            lane_cars = sorted(
                (car for car in cars if car.lane == lane),
                key=lambda car: car.distance,
                reverse=True,
            )
            front_car: TrafficCar | None = None
            for car in lane_cars:
                adjusted = car
                if front_car is not None:
                    maximum_distance = front_car.distance - minimum_center_spacing
                    if car.distance > maximum_distance:
                        adjusted = TrafficCar(
                            car_id=car.car_id,
                            lane=car.lane,
                            distance=float(maximum_distance),
                            speed=float(min(car.speed, front_car.speed)),
                        )
                constrained.append(adjusted)
                front_car = adjusted
        return sorted(constrained, key=lambda car: car.car_id)

    def nearest_ahead_distance(self, lane: int) -> float:
        """Return edge-to-edge clearance ahead, capped at the field of view."""
        visible_distances = [
            self.forward_clearance(car)
            for car in self.traffic
            if car.lane == lane
            and 0.0 <= self.forward_clearance(car) <= self.config.vision_distance
        ]
        return float(min(visible_distances, default=self.config.vision_distance))

    def snapshot(self) -> RoadSnapshot:
        return RoadSnapshot(
            progress=float(self.progress),
            ego_lane=int(self.ego_lane),
            traffic=tuple(sorted(self.traffic, key=lambda car: car.distance)),
        )

    def _get_lane_observation(self, lane_idx: int) -> tuple[float, float]:
        if lane_idx < 0 or lane_idx >= self.config.lane_count:
            return 1.0, 0.0
        
        visible = [
            car for car in self.traffic
            if car.lane == lane_idx
            and 0.0 <= self.forward_clearance(car) <= self.config.vision_distance
        ]
        if visible:
            nearest = min(visible, key=self.forward_clearance)
            dist = float(self.forward_clearance(nearest) / self.config.vision_distance)
            speed = float((self.config.ego_speed - nearest.speed) / self.config.ego_speed)
            return dist, speed
        return 1.0, 0.0

    def observation(self) -> RoadObservation:
        one_hot_lane = [0.0] * MAX_LANES
        one_hot_lane[self.ego_lane] = 1.0

        left_dist, left_speed = self._get_lane_observation(self.ego_lane - 1)
        curr_dist, curr_speed = self._get_lane_observation(self.ego_lane)
        right_dist, right_speed = self._get_lane_observation(self.ego_lane + 1)

        progress = min(1.0, self.progress / self.config.road_length)
        return tuple(
            one_hot_lane
            + [left_dist, left_speed, curr_dist, curr_speed, right_dist, right_speed, float(progress)]
        )

    def step(self, action: Action5) -> Room5Transition:
        action = Action5(action)
        events: list[str] = ["step"]
        reward = float(self.config.rewards.get("step", -0.02))

        previous_lane = self.ego_lane
        previous_clearance = self.nearest_ahead_distance(previous_lane)
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
            new_clearance = self.nearest_ahead_distance(self.ego_lane)
            if new_clearance > previous_clearance + 1e-9:
                events.append("safer_lane_change")
                reward += self.config.rewards.get("safer_lane_change", 1.0)
            elif new_clearance < previous_clearance - 1e-9:
                events.append("riskier_lane_change")
                reward += self.config.rewards.get("riskier_lane_change", -1.0)

        forward_distance = self.config.ego_speed * self.config.dt
        self.progress += forward_distance
        self.elapsed_time += self.config.dt
        reward += forward_distance * self.config.rewards.get("forward_progress", 0.02)
        events.append("forward_progress")

        moved_traffic: list[TrafficCar] = []
        for car in self.traffic:
            closing_distance = (self.config.ego_speed - car.speed) * self.config.dt
            moved_traffic.append(
                TrafficCar(
                    car_id=car.car_id,
                    lane=car.lane,
                    distance=float(car.distance - closing_distance),
                    speed=car.speed,
                )
            )

        moved_traffic = self._enforce_minimum_traffic_clearance(moved_traffic)
        updated_traffic: list[TrafficCar] = []
        overtaken = 0
        collision = False
        for car in moved_traffic:
            next_distance = car.distance
            if (
                car.lane == self.ego_lane
                and abs(next_distance) <= self.config.car_length
            ):
                collision = True

            if next_distance <= 0:
                overtaken += 1
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
        while len(self.traffic) < self.config.traffic_count:
            self._spawn_arriving_car()
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
