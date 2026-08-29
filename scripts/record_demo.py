"""Record a short GIF of a match for the README.

    python -m scripts.record_demo --seconds 5 --team-size 11 --out assets/demo.gif

Captures one PNG per simulated frame via raylib's own take_screenshot(),
then assembles them into an animated GIF with Pillow and deletes the
intermediate PNGs. One frame per env.step() means the GIF naturally
plays back at the simulation rate (SIM_HZ, default 20fps) with no need
to throttle to wall-clock time.

Note: raylib's take_screenshot() always resolves its filename relative
to the process's *current working directory* -- on some platforms it
does this even if you pass it an absolute path elsewhere (e.g. a system
temp dir), which produces a broken concatenated path. To sidestep that
entirely, frames are written to a relative subfolder under wherever you
run this script from (normally the project root), not to tempfile's
system temp directory.

Requires a real display -- this opens the same raylib window as
watch_random.py, it just also saves frames while it runs.
"""

import argparse
import os
import shutil

import pyray as rl
from PIL import Image

from football_env import FootballEnv
from football_env import constants as C


def random_action(env, agent_id):
    space = env.action_space(agent_id)
    return {
        "move": space["move"].sample(),
        "kick_type": space["kick_type"].sample(),
        "kick_dir": space["kick_dir"].sample(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--team-size", type=int, default=11)
    parser.add_argument("--out", type=str, default="assets/demo.gif")
    parser.add_argument("--fps", type=int, default=None,
                         help=f"GIF playback fps; defaults to the sim rate ({C.SIM_HZ})")
    args = parser.parse_args()

    n_frames = int(args.seconds * C.SIM_HZ)
    fps = args.fps or C.SIM_HZ

    env = FootballEnv(team_size=args.team_size, max_steps=n_frames + 5, render_mode="human")
    obs, infos = env.reset(seed=0)

    # Relative to cwd on purpose -- see module docstring.
    frame_dir_rel = "_football_record_tmp"
    os.makedirs(frame_dir_rel, exist_ok=True)
    frame_paths = []

    try:
        for i in range(n_frames):
            if not env.agents:
                break
            actions = {aid: random_action(env, aid) for aid in env.agents}
            obs, rewards, terms, truncs, infos = env.step(actions)

            still_open = env.render()
            if not still_open:
                break

            rel_path = os.path.join(frame_dir_rel, f"frame_{i:04d}.png")
            rl.take_screenshot(rel_path)
            frame_paths.append(os.path.abspath(rel_path))

        env.close()

        if not frame_paths:
            print("No frames captured (window closed immediately?). Nothing to save.")
            return

        missing = [p for p in frame_paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} frame(s) were not written to disk, e.g. {missing[0]}. "
                f"This usually means take_screenshot() couldn't resolve the path -- "
                f"try running this script from the project root."
            )

        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        frames = [Image.open(p).convert("RGB").convert("P", palette=Image.ADAPTIVE) for p in frame_paths]
        frames[0].save(
            args.out,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / fps),
            loop=0,
            optimize=True,
        )
        print(f"Saved {len(frame_paths)} frames ({len(frame_paths) / fps:.1f}s) -> {args.out}")

    finally:
        shutil.rmtree(frame_dir_rel, ignore_errors=True)


if __name__ == "__main__":
    main()