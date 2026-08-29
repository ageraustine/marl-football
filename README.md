# MARL Football

A multi-agent reinforcement learning environment and training pipeline for
11-a-side football. Built on PettingZoo, Gymnasium, RLlib, and raylib.

![gameplay demo](assets/demo.gif)

## Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Rules of Play](#rules-of-play)
5. [Environment Specification](#environment-specification)
6. [Rendering](#rendering)
7. [Training](#training)
8. [Project Structure](#project-structure)
9. [Testing](#testing)
10. [Limitations](#limitations)
11. [Roadmap](#roadmap)

## Overview

`FootballEnv` simulates a two-team football match on a regulation-size
pitch. Squad size is configurable from 1v1 up to full 11v11. Each player
is an independent agent controlled by an external policy through the
PettingZoo `ParallelEnv` API; the environment itself contains no
policies or scripted behavior.

The project includes:

- A physics-based simulation core (`football_env/`) with no rendering
  or training dependencies, so it can run headless at full speed.
- A raylib renderer (`render/`) for real-time visualization of a match.
- An RLlib training pipeline (`training/`) implementing team-level
  policy sharing and a staged curriculum from 1v1 to 11v11.

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/ageraustine/marl-football
cd marl_football
pip install -r requirements.txt
```

`football_env/constants.py`, `physics.py`, `entities.py`,
`observations.py`, and `rewards.py` depend only on NumPy and can be
imported and tested without installing PettingZoo, raylib, or Ray.

## Quick Start

```bash
# Verify the simulation core
python -m tests.test_physics

# Watch a match with random actions
python -m scripts.watch_random --team-size 11

# Record a short GIF of a match
python -m scripts.record_demo --seconds 5 --team-size 11 --out assets/demo.gif

# Train the full curriculum (1v1 -> 11v11)
python -m training.train

# Watch a trained checkpoint
python -m scripts.watch_trained --checkpoint checkpoints/11v11_full/<dir> --team-size 11
```

## Rules of Play

This section describes the ruleset implemented by the simulation. It is
a simplified subset of the FIFA Laws of the Game; differences from the
official laws are listed under [Limitations](#limitations).

### Pitch

| Property | Value |
|---|---|
| Length | 105 m |
| Width | 68 m |
| Goal width | 7.32 m |
| Penalty area | 40.32 m x 16.5 m |
| Center circle radius | 9.15 m |

### Teams

Each side fields between 1 and 11 players, set by the `team_size`
parameter at environment construction. Team 0 defends the goal at
`x = -52.5` and attacks toward `x = +52.5`; Team 1 is mirrored. Sides
are not swapped at any point during a match.

Players are assigned one of four roles at kickoff: Goalkeeper,
Defender, Midfielder, or Forward. Roles determine starting position and
are included in each agent's observation; they do not restrict what
actions a player can take.

### Match Duration

An episode runs for a fixed number of simulated seconds (180s by
default, configurable via `max_steps`). The simulation steps at 20 Hz.
A goal does not end the episode -- play resumes immediately from
kickoff, and the running score is tracked for the remainder of the
episode.

### Kickoff

At the start of an episode and after every goal:

1. All players return to their formation start positions for the
   current squad size.
2. The ball is placed at the center spot.
3. Possession is awarded to the team that conceded (or to Team 0 at the
   start of the episode).

### Scoring

A goal is awarded when the ball's position crosses a team's goal line
(`x = +/-52.5`) within the width of the goal (`|y| <= 3.66`). The
scoring team's counter increments and kickoff is triggered as described
above.

### Possession

The ball is either **free** or **owned** by exactly one player at any
time.

- A free ball within 0.9 m of a player is automatically collected by
  the nearest eligible player. A fast-moving ball is proportionally
  harder to collect (the effective radius shrinks with ball speed).
- A player in possession carries the ball: it follows just ahead of
  their movement direction as they run.
- Possession transfers when the ball is kicked, or when an opposing
  player collects a loose ball.

### Kicking

Only the player currently in possession may kick. Three kick types are
available, each releasing possession and applying a fixed initial
speed to the ball in a direction chosen by the kicking player:

| Kick type | Speed |
|---|---|
| Pass | 12 m/s |
| Shot | 22 m/s |
| Clear | 18 m/s |

### Out of Bounds

If the ball crosses a touchline or goal line without a goal being
scored, play is restarted as one of three types, matching which
boundary was crossed and who touched it last:

| Restart | Trigger | Restart position | Possession awarded to |
|---|---|---|---|
| Throw-in | Ball crosses a touchline | On the touchline, at the point of exit | Team that did not touch it last |
| Corner kick | Ball crosses a goal line; defending team touched it last | Corner arc nearest the exit point | Attacking team |
| Goal kick | Ball crosses a goal line; attacking team touched it last | Inside the goal area, on the defending team's side | Defending team |

No stoppage time or opposing-player positioning (e.g. the 9.15 m
distance on a corner) is simulated; the ball and restart possession are
placed directly.

### Offside

Offside is evaluated at the moment a player plays a pass. A teammate of
the passer is in an offside position if, at that moment, they are:

1. In the opponents' half, and
2. Ahead of both the ball and the second-deepest opponent (goalkeeper
   included in that count, following the standard practical
   approximation — the goalkeeper is almost always the deepest player
   in any case).

If the pass is subsequently received by a flagged teammate, play stops
and a free kick is awarded to the defending team at the point of
receipt. The flag is cleared as soon as the ball is next controlled by
anyone, whether or not an infringement occurred. Offside is evaluated
only on passes; shots and clearances do not trigger it.

### Stamina

Each player has a stamina value from 0-100, starting at 100. Moving
above 60% of base top speed drains stamina; moving slower recovers it.
Top speed scales with current stamina, from 100% at full stamina down
to 60% at zero — sustained sprinting reduces a player's top speed for
the remainder of the period until they recover.

### Movement

Players accelerate toward a commanded direction and speed rather than
moving instantaneously, subject to a maximum speed (8 m/s for outfield
players, 7.2 m/s for goalkeepers) and fixed acceleration/deceleration
limits.

## Environment Specification

### Action Space

Each agent's action is a dictionary:

| Key | Type | Description |
|---|---|---|
| `move` | `Box(2,)`, range [-1, 1] | Target movement direction and intensity |
| `kick_type` | `Discrete(4)` | 0 = none, 1 = pass, 2 = shoot, 3 = clear |
| `kick_dir` | `Box(2,)`, range [-1, 1] | Kick direction; ignored unless the agent is in possession and `kick_type != 0` |

### Observation Space

Each agent receives a fixed-length `Box(126,)` vector, independent of
squad size:

| Segment | Size | Contents |
|---|---|---|
| Self | 8 | Position, velocity, possession flag, time remaining, team id, stamina fraction |
| Ball | 5 | Relative position, relative velocity, free/owned flag |
| Goals | 4 | Relative vector to own and opponent goal |
| Role | 4 | One-hot: GK / DF / MF / FW |
| Teammates | 50 | 10 nearest teammates x (relative position, relative velocity, mask), distance-sorted, zero-padded |
| Opponents | 55 | 11 nearest opponents x (relative position, relative velocity, mask), distance-sorted, zero-padded |

Fixed-length, zero-padded teammate/opponent slots allow a policy
trained on a smaller squad size to be used as a checkpoint for training
on a larger squad size without a change in network architecture.

### Reward Function

| Term | Value | Applies to |
|---|---|---|
| Goal scored | +10.0 | Every player on the scoring team |
| Goal conceded | -10.0 | Every player on the conceding team |
| Ball proximity | +0.01 per meter closed | Players not in possession, closing distance to the ball |
| Possession advance | +0.02 per meter | Team in possession, per meter the ball moves toward the opponent goal |
| Out of bounds | -0.05 | Players on the team that last touched the ball |
| Time step | -0.0005 | Every agent, every step |

## Rendering

`render/renderer.py` draws the match state using raylib: pitch
markings, goal frames, players with team colors and role labels,
possession indicator, ball, and a scoreboard. Rendering is invoked by
setting `render_mode="human"` on `FootballEnv` and calling `env.render()`
after each step; it is not required for training.

To record a clip:

```bash
python -m scripts.record_demo --seconds 5 --team-size 11 --out assets/demo.gif
```

This captures one frame per simulated step via raylib's screenshot
function and assembles them into an animated GIF. Run it from the
project root -- frames are written to a temporary folder relative to
the working directory, since `take_screenshot()` resolves paths
relative to the process's working directory rather than accepting an
arbitrary absolute path on all platforms.

## Training

Training uses RLlib's multi-agent PPO. Each team shares a single
policy (`team_0_policy`, `team_1_policy`) across all of its players;
this is the mechanism that keeps 22-agent training computationally
tractable, at the cost of per-player specialization within a team.

Training proceeds through a fixed curriculum defined in
`training/curriculum.py`:

| Stage | Squad size | Target steps |
|---|---|---|
| `1v1_ball_chase` | 1 | 2,000,000 |
| `2v2_basics` | 2 | 3,000,000 |
| `3v3_passing` | 3 | 4,000,000 |
| `5v5_shape` | 5 | 6,000,000 |
| `11v11_full` | 11 | 15,000,000 |

Each stage restores the previous stage's policy checkpoint before
training continues at the larger squad size. Policy IDs are stable
across stages, which is what makes this restore valid.

```bash
python -m training.train                    # run the full curriculum
python -m training.train --start-stage 3     # resume from the 5v5 stage
```

Training metrics are written to `~/ray_results/` and can be viewed with
TensorBoard:

```bash
tensorboard --logdir ~/ray_results
```

RLlib's configuration API differs between major versions. This code
targets Ray 2.9+, as pinned in `requirements.txt`.

## Project Structure

```
football_env/
    constants.py        Pitch geometry, physics constants, formations, colors
    formations.py        Kickoff formation builder for a given squad size
    entities.py            Player and Ball state
    physics.py               Movement, ball dynamics, possession, kicks, scoring
    observations.py          Per-agent observation construction
    rewards.py                Reward computation
    environment.py             FootballEnv (pettingzoo.ParallelEnv)

render/
    renderer.py           raylib rendering

training/
    curriculum.py         Curriculum stage definitions
    rllib_wrapper.py        PettingZoo-to-RLlib adapter and policy mapping
    train.py                  Training entrypoint

scripts/
    watch_random.py       Render a match with random actions
    watch_trained.py        Render a match using a trained checkpoint
    record_demo.py            Record a match to GIF

tests/
    test_physics.py       Simulation core tests (NumPy only)

assets/
    demo.gif              Recorded gameplay clip
```

## Testing

```bash
python -m tests.test_physics
```

Covers player acceleration/deceleration, possession pickup radius,
kicks, goal detection, out-of-bounds detection, stamina drain and
recovery, offside detection, and restart classification (throw-in /
corner / goal kick).

## Limitations

The following are not implemented in the current ruleset:

- Fouls, cards, and physical contact between players (possession is
  proximity-based, not contested via a tackle action)
- Opposing-player positioning during restarts (e.g. the 9.15 m distance
  on a corner or free kick)
- Stoppage time
- Half-time end swap (each team defends the same goal for the entire
  episode)

## Roadmap

- Contested possession (tackling) instead of proximity-based pickup
- Fouls and cards
- Half-time end swap
- Opponent pool for self-play, rather than a single fixed opponent policy
- Episode replay recording independent of a live policy