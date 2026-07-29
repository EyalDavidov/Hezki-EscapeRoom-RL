# Room 5 Specification — One-way Traffic Avoidance

## Goal

Room 5 is a one-way road-driving task. The agent is a faster car that must change lanes to avoid slower traffic, safely overtake vehicles, and reach the configured road-completion distance without a collision.

## Environment

- The road contains 2–6 lanes, selected in the dashboard.
- All vehicles travel in the same direction.
- The agent travels at a constant speed greater than the traffic speed range. In the agent-relative view, slower traffic therefore approaches from ahead.
- The dashboard controls lane count, forward vision distance, road length, traffic count, agent speed, traffic-speed range, random seed, and all reward values.
- The episode terminates on collision, successful road completion, or the configured maximum timestep limit.

## Observation and actions

The fixed 19-value observation supports every lane count:

- A six-value one-hot representation of the agent lane.
- The normalized distance to the nearest visible car in each of six possible lanes.
- The normalized closing speed for that car in each lane.
- Normalized progress toward the road-completion distance.

The discrete actions are `LEFT`, `KEEP_LANE`, and `RIGHT`.

## Rewards

The dashboard exposes rewards for each timestep, forward progress per meter, overtaking, valid lane changes, invalid boundary lane changes, collision, and successful road completion.

## Algorithm

Room 5 uses PPO (Proximal Policy Optimization), a different algorithm from Rooms 1–4. The actor and critic share configurable hidden layers and have separate policy and value heads. The dashboard controls the learning rate, discount factor, GAE lambda, clipping epsilon, entropy and value coefficients, PPO update epochs, mini-batch size, architecture, activation function, episode count, timestep limit, and seed.

## Dashboard and acceptance criteria

- Environment, Training, Testing, and Models control sections are available.
- Every control has contextual help.
- Training shows live reward, overtakes, policy/value loss, entropy, duration, and action distribution.
- Testing shows success rate, collision rate, overtakes, rewards, action distribution, episode data, and automatic replay.
- Trained PPO models can be downloaded and restored as JSON artifacts.
- Python tests cover lane limits, field of view, collision, overtaking, PPO training/evaluation, visualization, and artifact round trips.
