"""Watch a full match with random actions, rendered live in raylib.

This is the fastest way to confirm your install works and to see the
graphics before touching any training code:

    python -m scripts.watch_random --team-size 11
"""

import argparse

import numpy as np

from football_env import FootballEnv


def random_action(env, agent_id):
    space = env.action_space(agent_id)
    return {
        "move": space["move"].sample(),
        "kick_type": space["kick_type"].sample(),
        "kick_dir": space["kick_dir"].sample(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team-size", type=int, default=11)
    parser.add_argument("--max-steps", type=int, default=3600)
    args = parser.parse_args()

    env = FootballEnv(team_size=args.team_size, max_steps=args.max_steps, render_mode="human")
    obs, infos = env.reset(seed=0)

    running = True
    while running and env.agents:
        actions = {aid: random_action(env, aid) for aid in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        running = env.render()

    env.close()


if __name__ == "__main__":
    main()
