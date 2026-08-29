"""raylib-based renderer for FootballEnv.

Deliberately decoupled from the simulation: this file only ever *reads*
players/ball/score and draws pixels. It never touches physics. That
separation is what lets training run headless at full speed while this
module is only imported when render_mode="human" (see FootballEnv.render).

Coordinate system: sim space is meters, origin at pitch center, +X right,
+Y "up" in football-pitch terms (towards the right touchline). Screen
space is pixels, origin top-left, +Y down. world_to_screen() does the
conversion + the margin needed to draw stands/run-off area around the
pitch.
"""

import math

import pyray as rl

from football_env import constants as C

MARGIN_M = 8.0          # meters of run-off/stand area drawn around the pitch
WORLD_W = C.PITCH_LENGTH + 2 * MARGIN_M
WORLD_H = C.PITCH_WIDTH + 2 * MARGIN_M


def _col(rgba):
    return rl.Color(rgba[0], rgba[1], rgba[2], rgba[3])


class Renderer:
    def __init__(self, team_size: int, width: int = 1280, height: int = 820):
        self.team_size = team_size
        rl.init_window(width, height, "MARL Football")
        rl.set_target_fps(60)
        self.width = width
        self.height = height
        self._recompute_scale()
        self._clock = 0.0

    def _recompute_scale(self):
        self.width = rl.get_screen_width()
        self.height = rl.get_screen_height()
        ui_h = 64  # top scoreboard bar reserved in pixels
        avail_h = self.height - ui_h
        self.scale = min(self.width / WORLD_W, avail_h / WORLD_H)
        self.offset_x = (self.width - WORLD_W * self.scale) / 2
        self.offset_y = ui_h + (avail_h - WORLD_H * self.scale) / 2

    def world_to_screen(self, pos):
        sx = self.offset_x + (pos[0] + WORLD_W / 2) * self.scale
        sy = self.offset_y + (WORLD_H / 2 - pos[1]) * self.scale
        return sx, sy

    def _len(self, meters):
        return meters * self.scale

    # -- main entry point, called once per frame --------------------------
    def draw(self, players, ball, score, step_count, max_steps):
        if rl.window_should_close():
            return False
        if rl.is_window_resized():
            self._recompute_scale()

        self._clock += rl.get_frame_time()

        rl.begin_drawing()
        rl.clear_background(_col((18, 24, 20, 255)))

        self._draw_stands()
        self._draw_pitch()
        self._draw_ball_shadow(ball)
        self._draw_players(players, ball)
        self._draw_ball(ball)
        self._draw_scoreboard(score, step_count, max_steps)

        rl.end_drawing()
        return True

    # -- background / stands ------------------------------------------------
    def _draw_stands(self):
        p0 = self.world_to_screen((-C.HALF_LENGTH - MARGIN_M, C.HALF_WIDTH + MARGIN_M))
        w = WORLD_W * self.scale
        h = WORLD_H * self.scale
        rl.draw_rectangle(int(p0[0]), int(p0[1]), int(w), int(h), _col((34, 40, 36, 255)))
        # Faint floodlight glow in the corners for a bit of atmosphere.
        for cx, cy in [(p0[0], p0[1]), (p0[0] + w, p0[1]), (p0[0], p0[1] + h), (p0[0] + w, p0[1] + h)]:
            rl.draw_circle_gradient(int(cx), int(cy), self._len(18), _col((255, 250, 220, 35)), _col((255, 250, 220, 0)))

    # -- pitch markings -------------------------------------------------------
    def _draw_pitch(self):
        stripe_w = C.PITCH_LENGTH / C.PITCH_STRIPE_COUNT
        for i in range(C.PITCH_STRIPE_COUNT):
            x0 = -C.HALF_LENGTH + i * stripe_w
            color = C.COLOR_PITCH_LIGHT if i % 2 == 0 else C.COLOR_PITCH_DARK
            p_tl = self.world_to_screen((x0, C.HALF_WIDTH))
            rl.draw_rectangle(int(p_tl[0]), int(p_tl[1]), math.ceil(self._len(stripe_w)) + 1,
                               math.ceil(self._len(C.PITCH_WIDTH)), _col(color))

        line = _col(C.COLOR_LINES)
        thick = max(1.5, self._len(0.12))

        self._draw_rect_outline((-C.HALF_LENGTH, C.HALF_WIDTH), (C.HALF_LENGTH, -C.HALF_WIDTH), line, thick)
        p1 = self.world_to_screen((0, C.HALF_WIDTH))
        p2 = self.world_to_screen((0, -C.HALF_WIDTH))
        rl.draw_line_ex(p1, p2, thick, line)
        center = self.world_to_screen((0, 0))
        rl.draw_circle_lines(int(center[0]), int(center[1]), self._len(9.15), line)
        rl.draw_circle(int(center[0]), int(center[1]), max(2.0, self._len(0.15)), line)

        for goal_x, sign in [(-C.HALF_LENGTH, 1), (C.HALF_LENGTH, -1)]:
            self._draw_rect_outline(
                (goal_x, C.PENALTY_AREA_WIDTH / 2),
                (goal_x + sign * C.PENALTY_AREA_LENGTH, -C.PENALTY_AREA_WIDTH / 2),
                line, thick,
            )
            self._draw_goal_frame(goal_x, sign)

    def _draw_goal_frame(self, goal_x, sign):
        top = self.world_to_screen((goal_x, C.GOAL_HALF_WIDTH))
        bot = self.world_to_screen((goal_x, -C.GOAL_HALF_WIDTH))
        back_x = goal_x - sign * C.GOAL_DEPTH
        top_back = self.world_to_screen((back_x, C.GOAL_HALF_WIDTH))
        bot_back = self.world_to_screen((back_x, -C.GOAL_HALF_WIDTH))
        net = _col((225, 225, 225, 160))
        post = _col((255, 255, 255, 255))
        rl.draw_line_ex(top, top_back, 2.0, net)
        rl.draw_line_ex(bot, bot_back, 2.0, net)
        rl.draw_line_ex(top_back, bot_back, 2.0, net)
        # simple net hatching for a bit of visual richness
        steps = 6
        for i in range(1, steps):
            t = i / steps
            a = (top[0] + (top_back[0] - top[0]) * t, top[1] + (top_back[1] - top[1]) * t)
            b = (bot[0] + (bot_back[0] - bot[0]) * t, bot[1] + (bot_back[1] - bot[1]) * t)
            rl.draw_line_ex(a, b, 1.0, net)
        rl.draw_line_ex(top, bot, max(2.5, self._len(0.12)), post)

    def _draw_rect_outline(self, corner_a, corner_b, color, thick):
        a = self.world_to_screen(corner_a)
        b = self.world_to_screen((corner_b[0], corner_a[1]))
        c = self.world_to_screen(corner_b)
        d = self.world_to_screen((corner_a[0], corner_b[1]))
        rl.draw_line_ex(a, b, thick, color)
        rl.draw_line_ex(b, c, thick, color)
        rl.draw_line_ex(c, d, thick, color)
        rl.draw_line_ex(d, a, thick, color)

    # -- entities ------------------------------------------------------------
    def _draw_players(self, players, ball):
        for agent_id, p in players.items():
            sx, sy = self.world_to_screen(p.position)
            radius = self._len(C.PLAYER_RADIUS)
            base_color = C.COLOR_TEAM_0 if p.team_id == 0 else C.COLOR_TEAM_1

            # ground shadow, offset slightly for a lightweight 3D feel
            rl.draw_ellipse(int(sx), int(sy + radius * 0.4), radius * 1.1, radius * 0.5, _col((0, 0, 0, 70)))

            if ball.owner_id == agent_id:
                pulse = 1.0 + 0.15 * math.sin(self._clock * 6.0)
                rl.draw_circle_lines(int(sx), int(sy), radius * 2.0 * pulse, _col(C.COLOR_POSSESSION_RING))

            rl.draw_circle(int(sx), int(sy), radius, _col(base_color))
            rl.draw_circle_lines(int(sx), int(sy), radius, _col((0, 0, 0, 140)))

            speed = math.hypot(*p.velocity)
            if speed > 0.4:
                heading = (p.velocity[0] / speed, p.velocity[1] / speed)
                tip = self.world_to_screen((p.position[0] + heading[0] * 0.7, p.position[1] + heading[1] * 0.7))
                rl.draw_line_ex((sx, sy), tip, max(1.5, radius * 0.25), _col((255, 255, 255, 200)))

            role = p.role
            rl.draw_text(role[0], int(sx - 4), int(sy - 6), max(8, int(radius)), _col((255, 255, 255, 230)))

    def _draw_ball_shadow(self, ball):
        sx, sy = self.world_to_screen(ball.position)
        r = self._len(C.BALL_RADIUS)
        rl.draw_ellipse(int(sx), int(sy + r * 1.2), r * 1.3, r * 0.6, _col(C.COLOR_BALL_SHADOW))

    def _draw_ball(self, ball):
        sx, sy = self.world_to_screen(ball.position)
        r = max(3.0, self._len(C.BALL_RADIUS))
        rl.draw_circle(int(sx), int(sy), r, _col(C.COLOR_BALL))
        rl.draw_circle_lines(int(sx), int(sy), r, _col((40, 40, 40, 255)))

    # -- HUD -------------------------------------------------------------------
    def _draw_scoreboard(self, score, step_count, max_steps):
        rl.draw_rectangle(0, 0, self.width, 64, _col(C.COLOR_UI_BG))
        seconds_left = max(0, (max_steps - step_count) // C.SIM_HZ)
        mins, secs = divmod(seconds_left, 60)
        text = f"TEAM A  {score[0]}  -  {score[1]}  TEAM B      {mins:02d}:{secs:02d}"
        font_size = 28
        text_w = rl.measure_text(text, font_size)
        rl.draw_text(text, int(self.width / 2 - text_w / 2), 18, font_size, _col(C.COLOR_UI_TEXT))

    def close(self):
        rl.close_window()
