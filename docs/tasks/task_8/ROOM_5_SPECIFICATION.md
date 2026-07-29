# Room 5 Specification — One-way Traffic Avoidance

## Goal

Room 5 is a one-way road-driving task. The agent is a faster car that must change lanes to avoid slower traffic, safely overtake vehicles, and reach the configured road-completion distance without a collision.

## Environment

- The road contains 2–6 lanes, selected in the dashboard.
- All vehicles travel in the same direction.
- Traffic vehicles in the same lane always keep at least 3 meters of physical edge-to-edge clearance from one another.
- The agent travels at a constant speed greater than the traffic speed range. In the agent-relative view, slower traffic therefore approaches from ahead.
- The dashboard controls lane count, forward vision distance, road length, traffic count (including zero for an empty road), agent speed, traffic-speed range, random seed, and all reward values.
- The episode terminates on collision, successful road completion, or the configured maximum timestep limit.

## Observation and actions

The fixed 9-value observation supports every lane count while restricting perception to the agent's current lane:

- A six-value one-hot representation of the agent lane.
- The normalized physical clearance to the nearest visible car in the current lane only, measured from the agent car's front edge to the traffic car's rear edge.
- The normalized closing speed for that same car.
- Normalized progress toward the road-completion distance.

Cars in adjacent lanes are deliberately excluded from the observation. Once the agent changes lanes, the distance and closing-speed fields describe the nearest car in the new lane.

The discrete actions are `LEFT`, `KEEP_LANE`, and `RIGHT`.

## Rewards

Every episode starts with an empty road. Traffic is introduced progressively from beyond the forward-vision boundary until the configured target car count is reached. The nearest visible car in the agent's current lane is annotated with its edge-to-edge clearance in meters, and a forward cone shows the current-lane-only field of view.

The dashboard exposes rewards for each timestep, forward progress per meter, overtaking, valid lane changes, invalid boundary lane changes, collision, and successful road completion. A valid lane change also receives a configurable safer-lane reward when it increases clearance to the nearest visible car ahead, or a configurable riskier-lane penalty when it decreases that clearance.

## Algorithm

Room 5 uses PPO (Proximal Policy Optimization), a different algorithm from Rooms 1–4. The actor and critic share configurable hidden layers and have separate policy and value heads. The dashboard controls the learning rate, discount factor, GAE lambda, clipping epsilon, entropy and value coefficients, PPO update epochs, mini-batch size, architecture, activation function, episode count, timestep limit, and seed.

## Dashboard and acceptance criteria

- Environment, Training, Testing, and Models control sections are available.
- Environment and reward values use compact numeric text inputs with explicit plus/minus buttons. Labels and inputs share a row where the sidebar width allows it.
- Reward values are unbounded and may be positive, negative, or zero. Environment values keep only structural or logical constraints, such as 2–6 lanes, non-negative traffic, positive road dimensions, and traffic speeds below the agent speed.
- Every control has contextual help.
- Training shows live reward, overtakes, signed policy loss, value loss, entropy, duration, and action distribution. Policy and value losses use separate axes because PPO's normalized signed policy objective is commonly close to zero.
- Testing shows success rate, collision rate, overtakes, rewards, action distribution, episode data, and automatic replay.
- Trained PPO models can be downloaded and restored as JSON artifacts.
- Python tests cover lane limits, field of view, collision, overtaking, PPO training/evaluation, visualization, and artifact round trips.
