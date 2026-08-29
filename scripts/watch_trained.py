"""Load a trained checkpoint and watch it play, rendered live.

    python -m scripts.watch_trained --checkpoint checkpoints/11v11_full/... --team-size 11
"""

import argparse

import numpy as np
import torch
from ray.rllib.algorithms.algorithm import Algorithm

from football_env import FootballEnv
from training.rllib_wrapper import team_policy_mapping_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--team-size", type=int, default=11)
    parser.add_argument("--max-steps", type=int, default=3600)
    parser.add_argument("--deterministic", action="store_true", help="disable action sampling noise")
    args = parser.parse_args()

    algo = Algorithm.from_checkpoint(args.checkpoint)

    env = FootballEnv(team_size=args.team_size, max_steps=args.max_steps, render_mode="human")
    obs, infos = env.reset(seed=0)

    running = True
    while running and env.agents:
        actions = {}
        for agent_id in env.agents:
            policy_id = team_policy_mapping_fn(agent_id)
            action = algo.compute_single_action(
                obs[agent_id], policy_id=policy_id, explore=not args.deterministic,
            )
            actions[agent_id] = action

        obs, rewards, terms, truncs, infos = env.step(actions)
        running = env.render()

    env.close()


if __name__ == "__main__":
    main()
