# MARL Football ⚽

A multi-agent reinforcement learning project simulating full **11v11 football**,
built on **PettingZoo** + **Gymnasium** for the environment API, **RLlib** for
multi-agent training, and **raylib** for real-time, high-quality 2D rendering.

Two teams of 11 learn to move, pass, and shoot from scratch through
self-play — no scripted behavior, no hand-coded tactics. What you see on
screen is entirely emergent from reward signals and curriculum training.

```
                    TEAM A   2  -  1   TEAM B          04:12
      ┌──────────────────────────────────────────────────────┐
      │░░░░░░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓░░░│
      │░░░  ╔═══╗              .                    ╔═══╗  ░░░│
      │░░░  ║ ● │      ⚫F      |      ⚪F           │ ● ║  ░░░│
      │░░░  ╚═══╝    ⚫M  ⚫M    |    ⚪M   ⚪M       ╚═══╝  ░░░│
      │▓▓▓        ⚫D    ⚫D     |      ⚪D    ⚪D          ▓▓▓│
      │░░░░░░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓( )▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░│
      └──────────────────────────────────────────────────────┘
```
*(actual rendering is a live, animated raylib window with smooth motion,
possession rings, direction indicators, and a scoreboard — the above is
just a text sketch of the layout. Run `scripts/watch_random.py` to see it.)*

---

## Why this project is interesting

Full 11v11 is the hard version of this problem. With 22 agents and a
sparse goal reward, naive independent training almost never converges to
anything watchable — credit assignment across ~90 decisions per agent
per episode, with a reward that only fires when *someone* scores, is
brutal. This project's design is built around making that tractable:

- **Parameter sharing**: one policy per team (not 22 separate networks).
  Every outfield player on Team A runs the *same* network, conditioned on
  a role embedding, so learning from one player's experience benefits
  all 11.
- **Curriculum learning**: train 1v1 → 2v2 → 3v3 → 5v5 → 11v11, carrying
  policy weights forward at each stage. Going straight to 11v11 is the
  single most common way projects like this stall out.
- **Reward shaping that doesn't fight the goal reward**: dense shaping
  terms (closing down the ball, advancing it upfield) are small relative
  to the sparse goal reward, so the policy can't "cheat" by farming
  shaping reward instead of actually trying to score.
- **A hybrid action space**: continuous movement (for motion that
  actually looks like football, not a grid-world) plus a small discrete
  set of kick actions gated by ball possession (for a kicking problem
  that's actually learnable).

## Quickstart

```bash
git clone <this-repo>
cd marl_football
pip install -r requirements.txt

# Sanity-check the simulation logic (no raylib/pettingzoo needed for this part)
python -m tests.test_physics

# Watch a full 11v11 match with random actions, rendered live in raylib
python -m scripts.watch_random --team-size 11

# Train the full curriculum (1v1 -> 11v11) — this takes hours to days
# depending on hardware; see "Training" below
python -m training.train

# Watch a trained checkpoint play
python -m scripts.watch_trained --checkpoint checkpoints/11v11_full/<checkpoint-dir> --team-size 11
```

## Project structure

```
football_env/            The simulation — no rendering or training deps
    constants.py            Pitch geometry, physics tuning, formations, colors
    formations.py            Kickoff formation builder for any squad size
    entities.py               Player / Ball dataclasses (plain state, no behavior)
    physics.py                 Movement, ball dynamics, possession, kicks, goals
    observations.py            Egocentric, fixed-size observation vector per agent
    rewards.py                  Sparse goal reward + dense shaping terms
    environment.py              FootballEnv(pettingzoo.ParallelEnv) — ties it together

render/
    renderer.py               raylib rendering: pitch, players, ball, scoreboard, HUD

training/
    curriculum.py             The 1v1 -> 11v11 stage definitions
    rllib_wrapper.py           PettingZoo -> RLlib MultiAgentEnv adapter, policy mapping
    train.py                    Curriculum training entrypoint (RLlib MAPPO/PPO)

scripts/
    watch_random.py           Render a match with random actions (no training needed)
    watch_trained.py           Load a checkpoint and render it playing

tests/
    test_physics.py            Fast, dependency-light sanity tests for the sim core
```

`football_env/` deliberately has **zero dependency on raylib or ray** —
it only needs `pettingzoo` + `gymnasium` (and `numpy` even without
those, for the physics core). That's what lets training run headless at
full speed on a cluster with no display, while `render/renderer.py` is
only ever imported when you actually want to watch.

## Design decisions

### Action space
Each agent's action is a `Dict`:
- `move`: `Box(2,)` in `[-1, 1]` — direction × intensity. Players
  *accelerate* towards this target velocity (not teleport to it), which
  is what makes motion look like a football player and not a
  twin-stick-shooter character.
- `kick_type`: `Discrete(4)` — `none / pass / shoot / clear`. Only has
  any effect if the agent currently has the ball.
- `kick_dir`: `Box(2,)` — direction for the kick, magnitude ignored
  (power comes from `kick_type`).

This hybrid mirrors how most football-RL research (e.g. Google Research
Football) structures the problem: continuous locomotion is what needs to
look and feel natural, while ball actions are a much smaller decision
that's far easier to learn as a discrete choice than as unconstrained
continuous kicking.

### Observations
Egocentric and **fixed-size regardless of squad size** — this is the
detail that makes curriculum learning work. Every agent sees:
own state, ball state (relative), both goals (relative), a role
one-hot, and its `MAX_TEAMMATES` / `MAX_OPPONENTS` nearest teammates and
opponents sorted by distance and zero-padded when the squad is smaller
(e.g. during 3v3 training). Because the observation *shape* never
changes, a policy trained at 3v3 can be dropped straight into a 5v5
environment as a warm start.

### Rewards
```
GOAL_REWARD            = +10.0  (shared by every player on the scoring team)
CONCEDE_PENALTY         = -10.0  (shared by every player on the conceding team)
BALL_PROXIMITY_WEIGHT   =  +0.01  per meter closed, only when not in possession
POSSESSION_ADVANCE_WEIGHT = +0.02  per meter the ball advances upfield, in possession
OUT_OF_BOUNDS_PENALTY   =  -0.05  charged to whoever last touched it
TIME_PENALTY             = -0.0005 per step, discourages pure stalling
```
The shaping terms are intentionally ~100-1000x smaller than the goal
reward. This is a deliberate anti-reward-hacking choice: a policy that
just jogs towards the ball forever without ever passing or shooting
should score *worse* over an episode than one that actually plays, and
the weights are tuned so that holds.

### Parameter sharing + self-play
Both teams train against each other in the same environment
(`training/rllib_wrapper.py`), with one policy per team
(`team_0_policy`, `team_1_policy`) shared across all 11 players. This
is a standard MARL simplification (independent-but-shared-weights PPO,
sometimes called "parameter-sharing IPPO") that trades off individual
player specialization for tractability — it's what makes 22-agent
training feasible on modest hardware. Role information is still fed
into the observation, so the shared policy *can* learn
role-conditioned behavior (a striker and a center-back behave very
differently despite sharing weights).

### Curriculum
See `training/curriculum.py`. Each stage restores the previous stage's
policy checkpoint before continuing training with a larger squad. The
policy IDs (`team_0_policy` / `team_1_policy`) stay constant across
every stage specifically so this restore works cleanly.

## Simplifications (scope cuts, on purpose)

This is a portfolio-focused simulation, not a certified football-rules
engine. Known simplifications, so nothing here is a "bug":

- **No offside rule.**
- **No fouls, cards, or physical contact/tackling** — the ball changes
  possession purely by proximity (`POSSESSION_RADIUS`), not by a tackle
  action.
- **Throw-ins / goal kicks / corners are all one simplified restart**:
  the ball is clamped just inside the line and handed to whichever team
  didn't touch it last, with no stoppage or set-piece positioning.
- **No stamina model** — top speed is constant for the full match.
- **Half-time side swap is not simulated** — each team defends the same
  goal for the whole episode.

All of these are natural "future work" extensions and the codebase is
structured (see `physics.py` / `environment.py`) so each one can be
added independently without a rewrite.

## Training

Training uses RLlib's PPO with the multi-agent API
(`training/train.py`), running the curriculum end to end:

```bash
python -m training.train                    # full curriculum from 1v1
python -m training.train --start-stage 3    # resume from the 5v5 stage
```

Each stage prints `steps` and `reward_mean` per training iteration.
Rough sizing: the 11v11 stage alone (`training/curriculum.py`) defaults
to 15M environment steps — plan for a multi-GPU machine or a cloud
cluster if you want to reach that stage in a reasonable amount of wall
time; the earlier small-squad stages are cheap and a good place to
first confirm the pipeline works end to end.

RLlib's exact config API has shifted across major versions; `train.py`
targets Ray 2.9+ as pinned in `requirements.txt`. If a call like
`.env_runners(...)` doesn't match your installed version, check that
version's RLlib migration notes — the curriculum/training *logic* won't
need to change, just the config-builder call names.

### Monitoring training
RLlib writes results to `~/ray_results/` by default, which you can point
TensorBoard at:
```bash
tensorboard --logdir ~/ray_results
```
Watch `episode_reward_mean` per policy — it should climb noticeably
faster in the small-squad stages (there's more goal-scoring signal per
episode) and more slowly, but still upward, once you reach 11v11.

## Testing

```bash
python -m tests.test_physics
```
Covers acceleration/deceleration, possession pickup radius, kicks, goal
detection, and out-of-bounds detection. These run with nothing but
`numpy` installed — deliberately independent of `pettingzoo` / `raylib`
/ `ray` — so you can validate the simulation core before setting up the
full stack.

## Roadmap
- [ ] Tackling / contested possession instead of pure-proximity pickup
- [ ] Offside detection
- [ ] Stamina model affecting max speed late in a match
- [ ] Proper set-piece positioning (corners, throw-ins, free kicks)
- [ ] TrueSkill-style opponent pool for self-play instead of a single
      fixed opponent policy (reduces strategy collapse / overfitting to
      one opponent style)
- [ ] Replay recording (serialize an episode's positions to disk) so
      good matches can be replayed without re-running a policy

## Requirements

See `requirements.txt`. Summary: Python 3.10+, `numpy`, `pettingzoo`,
`gymnasium`, `raylib` (the `pyray` bindings), `ray[rllib]`, `torch`.
