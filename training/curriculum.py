"""Curriculum stages for training up to full 11v11.

Rationale (from project planning): training 22 agents from scratch on
sparse goal reward rarely converges. Each stage trains a shared-per-team
policy on a smaller squad, then the NEXT stage restores those same
policy weights (policy ids "team_0_policy"/"team_1_policy" are stable
across stages) and continues training with more players on the pitch.

Timesteps below are starting points, not guarantees -- watch the reward
curves (see README > Monitoring training) and extend a stage if the
policy hasn't plateaued yet.
"""

from dataclasses import dataclass


@dataclass
class Stage:
    name: str
    team_size: int
    timesteps: int
    max_episode_steps: int


CURRICULUM = [
    Stage(name="1v1_ball_chase", team_size=1, timesteps=2_000_000, max_episode_steps=1200),
    Stage(name="2v2_basics", team_size=2, timesteps=3_000_000, max_episode_steps=1800),
    Stage(name="3v3_passing", team_size=3, timesteps=4_000_000, max_episode_steps=2400),
    Stage(name="5v5_shape", team_size=5, timesteps=6_000_000, max_episode_steps=3000),
    Stage(name="11v11_full", team_size=11, timesteps=15_000_000, max_episode_steps=3600),
]
