"""Thin adapter: PettingZoo ParallelEnv -> RLlib MultiAgentEnv.

Written by hand instead of relying on ray.rllib's built-in PettingZoo
wrapper because that wrapper's import path/signature has changed across
Ray versions -- this ~30 line version is easier to keep working than to
chase RLlib's compatibility shims.
"""

from ray.rllib.env.multi_agent_env import MultiAgentEnv

from football_env import FootballEnv


def team_policy_mapping_fn(agent_id, episode=None, **kwargs):
    """One shared policy per team -- this is the parameter-sharing choice
    discussed in the README, and it's what makes 11v11 tractable.
    """
    return "team_0_policy" if agent_id.startswith("team_0") else "team_1_policy"


class FootballMultiAgentEnv(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        self.env = FootballEnv(
            team_size=config.get("team_size", 11),
            max_steps=config.get("max_steps", None) or __import__("football_env.constants", fromlist=["MAX_EPISODE_STEPS"]).MAX_EPISODE_STEPS,
            render_mode=config.get("render_mode", None),
        )
        self.agents = self.env.possible_agents
        self.possible_agents = self.env.possible_agents
        self.observation_space = self.env.observation_space(self.agents[0])
        self.action_space = self.env.action_space(self.agents[0])

    def reset(self, *, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        return obs, infos

    def step(self, action_dict):
        obs, rewards, terms, truncs, infos = self.env.step(action_dict)
        terms["__all__"] = all(terms.values()) if terms else False
        truncs["__all__"] = all(truncs.values()) if truncs else False
        return obs, rewards, terms, truncs, infos

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
