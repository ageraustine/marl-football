"""Curriculum training entrypoint.

    python -m training.train                # full curriculum, 1v1 -> 11v11
    python -m training.train --start-stage 2 # resume from 3v3 onward

Each stage trains two shared policies (team_0_policy / team_1_policy) via
self-play against each other, then hands the resulting weights to the
next, larger-squad stage. See training/curriculum.py for the stage list
and training/rllib_wrapper.py for why policies are shared per-team.

Note on RLlib API stability: this targets Ray 2.9+ (see requirements.txt).
RLlib's config API has shifted between major versions in the past --
if `.env_runners(...)` or `algo.save(...).checkpoint.path` don't match
your installed version, check `ray.rllib` changelogs for the equivalent
call; the training logic/curriculum structure itself won't need to change.
"""

import argparse
import os

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from training.curriculum import CURRICULUM
from training.rllib_wrapper import FootballMultiAgentEnv, team_policy_mapping_fn

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")


def env_creator(config):
    return FootballMultiAgentEnv(config)


def build_config(stage, obs_space, act_space, num_env_runners):
    return (
        PPOConfig()
        .environment(
            "football_multiagent",
            env_config={"team_size": stage.team_size, "max_steps": stage.max_episode_steps},
        )
        .multi_agent(
            policies={
                "team_0_policy": (None, obs_space, act_space, {}),
                "team_1_policy": (None, obs_space, act_space, {}),
            },
            policy_mapping_fn=team_policy_mapping_fn,
        )
        .env_runners(num_env_runners=num_env_runners, rollout_fragment_length=200)
        .training(
            train_batch_size=8000,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            model={"fcnet_hiddens": [256, 256], "fcnet_activation": "relu"},
        )
        .framework("torch")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-stage", type=int, default=0, help="index into CURRICULUM to resume from")
    parser.add_argument("--num-env-runners", type=int, default=4)
    parser.add_argument("--resume-checkpoint", type=str, default=None,
                         help="explicit checkpoint path to load before --start-stage, "
                              "for resuming after a crash rather than starting the stage fresh")
    args = parser.parse_args()

    ray.init()
    register_env("football_multiagent", env_creator)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    probe_env = FootballMultiAgentEnv({"team_size": 11})
    obs_space, act_space = probe_env.observation_space, probe_env.action_space
    probe_env.close()

    checkpoint_path = args.resume_checkpoint

    for stage in CURRICULUM[args.start_stage:]:
        print(f"\n=== Curriculum stage: {stage.name} (team_size={stage.team_size}v{stage.team_size}) ===")
        config = build_config(stage, obs_space, act_space, args.num_env_runners)
        algo = config.build()

        if checkpoint_path is not None:
            algo.restore(checkpoint_path)
            print(f"Restored policy weights from {checkpoint_path}")

        steps_done = 0
        while steps_done < stage.timesteps:
            result = algo.train()
            steps_done = result.get("num_env_steps_sampled_lifetime", steps_done + result.get("num_env_steps_sampled", 1))
            reward_mean = result.get("env_runners", {}).get("episode_reward_mean", float("nan"))
            print(f"[{stage.name}] steps={steps_done:>10}/{stage.timesteps} reward_mean={reward_mean:.3f}")

        stage_dir = os.path.join(CHECKPOINT_DIR, stage.name)
        saved = algo.save(stage_dir)
        checkpoint_path = saved.checkpoint.path if hasattr(saved, "checkpoint") else saved
        print(f"Saved checkpoint: {checkpoint_path}")
        algo.stop()

    ray.shutdown()


if __name__ == "__main__":
    main()
