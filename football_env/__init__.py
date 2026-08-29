try:
    # Requires pettingzoo + gymnasium. Kept optional so that
    # football_env.constants / physics / entities / observations / rewards
    # can still be imported (and unit tested) in a minimal numpy-only
    # environment -- see tests/test_physics.py.
    from football_env.environment import FootballEnv
    __all__ = ["FootballEnv"]
except ImportError:
    __all__ = []
