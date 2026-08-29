"""FootballEnv: a PettingZoo ParallelEnv for N-vs-N football.

Design notes (see README for the full write-up):
  * Continuous movement + gated discrete kick actions (see action_space).
  * Team size is a constructor argument so the same class powers every
    curriculum stage (2v2 through 11v11) -- only the formation changes.
  * A goal does NOT end the episode; it increments the score and the
    match continues (kickoff reset) until MAX_EPISODE_STEPS. This makes
    training signal denser (multiple goals per episode) and is also just
    how football actually works.
"""

import functools

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from football_env import constants as C
from football_env import physics
from football_env.entities import Ball, Player
from football_env.formations import build_formation
from football_env.observations import OBS_SIZE, build_observation
from football_env.rewards import compute_rewards


class FootballEnv(ParallelEnv):
    metadata = {"render_modes": ["human", None], "name": "marl_football_v0"}

    def __init__(self, team_size: int = 11, max_steps: int = C.MAX_EPISODE_STEPS, render_mode=None):
        assert 1 <= team_size <= 11, "team_size must be between 1 and 11"
        self.team_size = team_size
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.possible_agents = [f"team_0_player_{i:02d}" for i in range(team_size)] + \
                                [f"team_1_player_{i:02d}" for i in range(team_size)]
        self.agents = list(self.possible_agents)

        self._roles = {}   # agent_id -> role, fixed for the env's lifetime
        self.players = {}
        self.ball = Ball()
        self.score = {0: 0, 1: 0}
        self.step_count = 0
        self._renderer = None  # lazily created only if render_mode == "human"

    # -- PettingZoo boilerplate -------------------------------------------------
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(low=-10.0, high=10.0, shape=(OBS_SIZE,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Dict({
            "move": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            "kick_type": spaces.Discrete(4),  # 0=none 1=pass 2=shoot 3=clear
            "kick_dir": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        })

    # -- core lifecycle -----------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.agents = list(self.possible_agents)
        self.players = {}
        for team_id, attacking_right in [(0, True), (1, False)]:
            formation = build_formation(self.team_size, attacking_right)
            for i, (role, pos) in enumerate(formation):
                agent_id = f"team_{team_id}_player_{i:02d}"
                self._roles[agent_id] = role
                self.players[agent_id] = Player(
                    agent_id=agent_id, team_id=team_id, role=role,
                    position=pos.copy(), velocity=C.np_zeros2(), home_position=pos.copy(),
                )

        self.ball = Ball(position=C.np_zeros2(), velocity=C.np_zeros2())
        self.score = {0: 0, 1: 0}
        self.step_count = 0

        obs = {aid: build_observation(aid, self.players, self.ball, 0, self.max_steps) for aid in self.agents}
        infos = {aid: {} for aid in self.agents}
        return obs, infos

    def step(self, actions: dict):
        prev_positions = {aid: p.position.copy() for aid, p in self.players.items()}
        prev_ball_pos = self.ball.position.copy()

        # 1) movement for everyone
        for agent_id, action in actions.items():
            move = np.asarray(action.get("move", (0.0, 0.0)), dtype=np.float32)
            physics.step_player(self.players[agent_id], move, C.DT)

        # 2) dribbling: ball follows its owner
        physics.snap_ball_to_owner(self.ball, self.players)

        # 3) kicks: only the current owner's kick action has any effect
        if self.ball.owner_id is not None and self.ball.owner_id in actions:
            action = actions[self.ball.owner_id]
            kick_type = int(action.get("kick_type", 0))
            if kick_type != C.KICK_NONE:
                kick_dir = np.asarray(action.get("kick_dir", (1.0, 0.0)), dtype=np.float32)
                physics.apply_kick(self.ball, self.players[self.ball.owner_id], kick_type, kick_dir)

        # 4) free-ball physics + pickup
        physics.step_ball(self.ball, C.DT)
        physics.resolve_possession(self.ball, self.players)

        # 5) goals / out-of-bounds
        scoring_team = physics.check_goal(self.ball)
        went_out = physics.check_out_of_bounds(self.ball) and scoring_team is None

        if scoring_team is not None:
            self.score[scoring_team] += 1
            self._kickoff_reset(conceding_team=1 - scoring_team)
        elif went_out:
            self._handle_out_of_bounds()

        self.step_count += 1

        rewards = compute_rewards(
            self.players, self.ball, prev_positions, prev_ball_pos,
            scoring_team, went_out, {0: self.team_size, 1: self.team_size},
        )

        truncated = self.step_count >= self.max_steps
        terminations = {aid: False for aid in self.agents}
        truncations = {aid: truncated for aid in self.agents}
        obs = {aid: build_observation(aid, self.players, self.ball, self.step_count, self.max_steps) for aid in self.agents}
        infos = {aid: {"score": dict(self.score)} for aid in self.agents}

        if truncated:
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    # -- internal helpers -----------------------------------------------------
    def _kickoff_reset(self, conceding_team: int):
        for team_id, attacking_right in [(0, True), (1, False)]:
            formation = build_formation(self.team_size, attacking_right)
            for i, (role, pos) in enumerate(formation):
                agent_id = f"team_{team_id}_player_{i:02d}"
                p = self.players[agent_id]
                p.position = pos.copy()
                p.velocity = C.np_zeros2()
        self.ball.position = C.np_zeros2()
        self.ball.velocity = C.np_zeros2()
        self.ball.owner_id = None
        # Kickoff possession goes to the team that conceded, mirroring real rules.
        kicker = f"team_{conceding_team}_player_00"
        self.ball.owner_id = kicker
        self.ball.last_touch_team = conceding_team

    def _handle_out_of_bounds(self):
        """Simplified restart: clamp the ball just inside the line and hand
        possession to the team that did NOT touch it last (like a throw-in
        or goal kick), without simulating the stoppage itself.
        """
        self.ball.position[0] = np.clip(self.ball.position[0], -C.HALF_LENGTH + 0.5, C.HALF_LENGTH - 0.5)
        self.ball.position[1] = np.clip(self.ball.position[1], -C.HALF_WIDTH + 0.5, C.HALF_WIDTH - 0.5)
        self.ball.velocity = C.np_zeros2()

        restart_team = 1 - (self.ball.last_touch_team if self.ball.last_touch_team is not None else 0)
        candidates = [p for p in self.players.values() if p.team_id == restart_team]
        nearest = min(candidates, key=lambda p: np.linalg.norm(p.position - self.ball.position))
        self.ball.owner_id = nearest.agent_id
        self.ball.last_touch_team = restart_team

    def render(self):
        if self.render_mode != "human":
            return
        if self._renderer is None:
            from render.renderer import Renderer
            self._renderer = Renderer(self.team_size)
        return self._renderer.draw(self.players, self.ball, self.score, self.step_count, self.max_steps)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
