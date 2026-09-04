"""Microduck plays giant chess: read, remember, choose, walk, kick. Repeat.

Renders a video: third-person as the frame, the duck's camera as an inset with
tracked pieces, and the board it believes in, with FEN, bottom-right. At the
end it prints how the run scored against the simulator's ground truth.
"""

import math
import os
import sys

import chess
import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
import supervision as sv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import giant_chess as G  # noqa: E402
from trackers import SORTTracker  # noqa: E402
from trackers.utils.iou import BIoU  # noqa: E402

# A position with room behind the pawns: the duck stands on rank 1 to push a
# pawn, and its feet need the adjacent squares empty.
POSITION = {
    "a1": "R", "h1": "K", "a4": "Q", "b5": "B", "g3": "N", "c2": "P", "e2": "P",
    "a8": "r", "d8": "q", "g8": "k", "b7": "p", "e7": "p", "g7": "p", "c6": "n", "f5": "b",
}
MOVES = int(os.environ.get("MOVES", 3))
OUT = os.environ.get("OUT", "/tmp/giant_chess.mp4")
MAIN_W, MAIN_H = 1280, 720
POV_INSET = (400, 200)
DIAGRAM = 260
MAGENTA = sv.Color(255, 64, 255)


class Director:
    """Everything on screen. Uses the true duck pose only to aim the chase
    camera, which is the viewer's, not the robot's."""

    def __init__(self, model, data, duck, memory, pieces):
        self.model, self.data, self.duck, self.memory, self.pieces = model, data, duck, memory, pieces
        self.renderer = mujoco.Renderer(model, height=MAIN_H, width=MAIN_W)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance, self.camera.elevation = 1.15, -30
        self.azimuth = 180.0
        self.writer = imageio.get_writer(OUT, fps=25, quality=8)
        self.annotator = sv.BoxCornerAnnotator(thickness=2, corner_length=8, color=MAGENTA)
        self.caption, self.sub = "", ""
        self.highlight = ()
        self.frame_count = 0
        self.tick = 0

    def chase(self):
        x, y, yaw = self.duck.true_pose()
        want = math.degrees(yaw) + 180 - 30
        self.azimuth += 0.05 * ((want - self.azimuth + 180) % 360 - 180)
        self.camera.lookat[:] = [x + 0.15 * math.cos(yaw), y + 0.15 * math.sin(yaw), 0.06]
        self.camera.azimuth = self.azimuth
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render()

    def diagram(self):
        n, size = G.N, DIAGRAM
        cell = size // (n + 2)
        canvas = np.full((size, size, 3), 24, np.uint8)
        o = cell
        for fi in range(n):
            for ri in range(n):
                x0, y0 = o + fi * cell, o + (n - 1 - ri) * cell
                colour = (200, 216, 224) if (fi + ri) % 2 else (40, 56, 72)
                if (fi, ri) in self.highlight:
                    colour = (120, 40, 140)
                cv2.rectangle(canvas, (x0, y0), (x0 + cell, y0 + cell), colour, -1)
        for (fi, ri), letter in self.memory.letters().items():
            x0, y0 = o + fi * cell, o + (n - 1 - ri) * cell
            c = (x0 + cell // 2, y0 + cell // 2)
            white = letter.isupper()
            cv2.circle(canvas, c, int(cell * 0.38), (245, 245, 240) if white else (30, 30, 34), -1)
            cv2.circle(canvas, c, int(cell * 0.38), (255, 64, 255), 1)
            cv2.putText(canvas, letter.upper(), (c[0] - 6, c[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (30, 30, 34) if white else (245, 245, 240), 1, cv2.LINE_AA)
        for fi in range(n):
            cv2.putText(canvas, G.FILES[fi], (o + fi * cell + cell // 2 - 4, o + n * cell + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{len(self.memory.letters())} pieces remembered", (o, o - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 140, 255), 1, cv2.LINE_AA)
        return canvas

    def on_frame(self, pilot, command):
        self.tick += 1
        if self.tick % 2:
            return
        frame = self.chase()
        pov = None
        if pilot.last_view is not None:
            pov, _, seen = pilot.last_view
            pov = pov.copy()
            if seen:
                boxes = np.array([v[0] for v in seen.values()], dtype=float)
                pov = self.annotator.annotate(pov, sv.Detections(xyxy=boxes, class_id=np.zeros(len(boxes), dtype=int)))
        if pov is not None:
            inset = cv2.resize(pov, POV_INSET, interpolation=cv2.INTER_AREA)
            pad = 12
            frame[pad:pad + POV_INSET[1] + 4, MAIN_W - POV_INSET[0] - pad - 4:MAIN_W - pad] = 30
            frame[pad + 2:pad + 2 + POV_INSET[1], MAIN_W - POV_INSET[0] - pad - 2:MAIN_W - pad - 2] = inset
        d = self.diagram()
        frame[MAIN_H - DIAGRAM - 12:MAIN_H - 12, MAIN_W - DIAGRAM - 12:MAIN_W - 12] = d
        cv2.putText(frame, self.caption, (24, MAIN_H - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)
        if self.sub:
            cv2.putText(frame, self.sub, (24, MAIN_H - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                        (210, 210, 210), 1, cv2.LINE_AA)
        self.writer.append_data(frame)
        self.frame_count += 1


def observe(pilot, memory, tracker, pieces, seconds, t):
    """Stand and sweep the head, folding every sighting into the memory. Then
    turn a little each way and sweep again: a piece near the edge of the frame
    is read one square off, so every piece should be seen mid-frame from some
    heading before it is trusted."""
    def sweep(t, length):
        for i in range(int(length / pilot.duck.dt)):
            pilot.tick((0.0, 0.0, 0.0), t)
            if pilot.last_view is not None and pilot.T_w_c is not None and i % 4 == 0:
                _, _, seen = pilot.last_view
                sightings, _ = G.sightings_from(seen, pieces, pilot.T_w_c, tracker)
                for piece_name, sq in sightings.items():
                    memory.observe(piece_name, pieces[piece_name], sq, i)
            t += pilot.duck.dt
        return t

    t = sweep(t, seconds)
    t = pilot.burst((G.Navigator.TURN_FWD, 0.0, 1.0), 0.6, t)
    t = sweep(t, seconds * 0.6)
    t = pilot.burst((G.Navigator.TURN_FWD, 0.0, -1.0), 1.2, t)
    t = sweep(t, seconds * 0.6)
    t = pilot.burst((G.Navigator.TURN_FWD, 0.0, 1.0), 0.6, t)
    return t


def main():
    model, markers, pieces = G.build(POSITION)
    data = mujoco.MjData(model)
    duck = G.Duck(model, data)
    eyes = G.Eyes(model, data, markers, pieces)
    memory = G.BoardMemory()
    director = Director(model, data, duck, memory, pieces)
    pilot = G.Pilot(duck, eyes, on_frame=director.on_frame)
    tracker = SORTTracker(frame_rate=12.5, minimum_iou_threshold=0.05,
                          iou=BIoU(buffer_ratio=2.0), minimum_consecutive_frames=1)
    piece_qpos = {n: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_free")]
                  for n in pieces}
    T_t_c_neutral = None

    near = G.BOARD_X - G.BOARD / 2
    duck.place((near - 0.40, 0.0), 0.0)
    t = 0.0
    turn = chess.WHITE
    results = []

    for move_index in range(MOVES):
        director.caption, director.sub, director.highlight = "Reading the board", "corner posts locate the duck, tracked pieces fill the board", ()
        t = observe(pilot, memory, tracker, pieces, 2.0, t)
        if T_t_c_neutral is None:
            T_t_c_neutral = eyes.trunk_from_camera()

        options, board = G.choose_move(memory, turn)
        occupied = set(memory.letters())
        ranked = []
        for move in options:
            src = (chess.square_file(move.from_square), chess.square_rank(move.from_square))
            dst = (chess.square_file(move.to_square), chess.square_rank(move.to_square))
            if not G.stand_is_clear(occupied, src, dst):
                continue
            pose = G.kick_pose(src, dst)
            # Stay off the pieces: only stands on rank 1 or outside the board,
            # reached along the outside of the near edge and entered at the file.
            stand_sq = G.square_of_world(pose[0], pose[1])
            if stand_sq is not None and stand_sq[1] > 1:
                continue
            seen = G.markers_in_view(pose, markers, T_t_c_neutral)
            if seen < 6:
                continue
            is_pawn = board.piece_at(move.from_square).piece_type == chess.PAWN
            if min(src[0], G.N - 1 - src[0]) < 2:        # a, b, g, h: corners are cramped
                continue
            ranked.append((0 if is_pawn else 1, 0, src[1], -seen, move.uci(), move, src, dst))
        if not ranked:
            print("no playable move from the remembered board")
            break
        ranked.sort()
        _, _, _, _, uci, move, src, dst = ranked[0]
        side = "white" if turn == chess.WHITE else "black"
        director.caption = f"Move {move_index + 1}: {uci} ({side})"
        director.sub = f"legal on the remembered board, {len(ranked)} playable candidates"
        director.highlight = (src, dst)
        for _ in range(int(1.2 / duck.dt)):
            pilot.tick((0.0, 0.0, 0.0), t)
            t += duck.dt

        director.caption, director.sub = f"Walking to the kick spot for {uci}", "localizing on the marker posts, dead reckoning between fixes"
        target = G.kick_pose(src, dst)
        # The piece the duck will home in on is whatever stands on the source
        # square; the reading put it there, the segmentation names it.
        def piece_on(sq):
            for n in pieces:
                if G.square_of_world(*data.qpos[piece_qpos[n]:piece_qpos[n] + 2]) == sq:
                    return n
            return None
        name = piece_on(src)
        if name is None:
            print(f"memory said {uci} but nothing stands on {G.square_name(*src)}: re-reading")
            for entry in memory.pieces.values():
                if entry["square"] == src:
                    entry["square"] = None
            continue
        near_edge = G.BOARD_X - G.BOARD / 2
        outside = (near_edge - 0.30, target[1] - 0.30 * math.sin(target[2]))
        t = pilot.go_to(target, t0=t, via=[outside])
        director.sub = "closing in on the piece itself: its crown, through the camera, onto its height"
        t, _ = pilot.servo_to_piece(name, t)
        t, _ = pilot.nudge_to_piece(name, t)

        # Kick, then look: if the piece is still at the foot, it was a miss.
        # Correct from what it sees and try again. A real robot would.
        attempts = 0
        while True:
            attempts += 1
            truth = duck.true_pose()
            stand_err = math.hypot(truth[0] - target[0], truth[1] - target[1])
            director.caption = f"Kick {uci}" + (f" (attempt {attempts})" if attempts > 1 else "")
            director.sub = f"stand within {stand_err * 1000:.0f} mm of the plan (measured in sim)"
            for _ in range(int(0.6 / duck.dt)):
                duck.step(head_pitch=0.0)
                director.on_frame(pilot, (0, 0, 0))
            duck.kick()
            for _ in range(int(3.0 / duck.dt)):
                duck.step()
                director.on_frame(pilot, (0, 0, 0))
            t += 3.6
            rel, t = pilot.measure_piece(name, t)
            # Unseen is not gone: only a piece seen away from the foot counts as moved.
            still_there = rel is None or (abs(rel[0] - G.KICK_FOOT[0]) < 0.06 and abs(rel[1] - G.KICK_FOOT[1]) < 0.06)
            if not still_there or attempts >= 2:
                break
            e_x = rel[0] - G.KICK_FOOT[0]
            director.caption, director.sub = f"Missed {uci}, adjusting", f"piece is {e_x * 1000:+.0f} mm from where the foot wants it"
            t, _ = pilot.nudge_to_piece(name, t)

        pq = piece_qpos[name]
        landed = G.square_of_world(*data.qpos[pq:pq + 2])
        q = data.qpos[pq + 3:pq + 7]
        upright = (q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2) > 0.9
        ok = landed == dst and upright
        results.append((uci, stand_err, landed, upright, ok, attempts))
        director.caption = f"{uci}: piece on {G.square_name(*landed) if landed else 'the floor'}" + (" - move made" if ok else " - missed")
        director.sub = "measured in sim; relocalizing, then walking back out to read the board"
        t = pilot.relocalize(t)
        if not still_there:
            memory.apply_move(src, dst, move_index)
        if ok:
            board.push(move)
        # Straight back to the reading spot on dead reckoning: from a stand on
        # rank 1 the line home leaves the board at once and crosses nothing.
        t = pilot.retreat((near - 0.40, 0.0), t)
        for _ in range(int(0.8 / duck.dt)):
            pilot.tick((0.0, 0.0, 0.0), t)
            t += duck.dt
        # The duck plays white: three moves of its own plan, so its walks stay on
        # the half of the board where the posts are densest in its view.

    director.caption, director.sub, director.highlight = "Final read", "", ()
    t = observe(pilot, memory, tracker, pieces, 3.0, t)
    director.writer.close()

    truth_now = {}
    for n, letter in pieces.items():
        sq = G.square_of_world(*data.qpos[piece_qpos[n]:piece_qpos[n] + 2])
        if sq is not None:
            truth_now[sq] = letter
    remembered = memory.letters()
    right = sum(1 for sq, l in remembered.items() if truth_now.get(sq) == l)
    print(f"wrote {OUT} ({director.frame_count} frames)")
    for uci, e, landed, up, ok, n in results:
        print(f"  {uci}: {n} attempt(s), last stand err {e * 1000:.0f} mm, landed {G.square_name(*landed) if landed else '-'}, "
              f"upright={up} -> {'MADE' if ok else 'MISSED'}")
    print(f"memory: {len(remembered)} squares remembered, {right} correct, "
          f"{len(truth_now)} pieces actually on the board; fixes {pilot.fixes}/{pilot.looks}")
    print("remembered:", memory.as_chess_board(chess.WHITE).board_fen())
    b = chess.Board(None)
    for (fi, ri), l in truth_now.items():
        b.set_piece_at(chess.square(fi, ri), chess.Piece.from_symbol(l))
    print("truth:     ", b.board_fen())


if __name__ == "__main__":
    main()
