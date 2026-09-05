"""Giant chess for Microduck: see the board, remember it, walk to a piece, kick it.

Nothing the duck decides on comes from simulator state. Border markers seen
through its own camera give both the board-plane homography (which square a
piece stands on) and, with the camera model, the duck's own pose on the board
(PnP). Pieces carry tracker ids and the board it believes in keeps each one on
its last-known square while it is out of view. python-chess picks a legal move,
the duck walks to the kick pose for that piece and kicks it one square.

Ground truth is read from the simulator only to score the run afterwards.
"""

import math
import os
import sys

import chess
import cv2
import mujoco
import numpy as np
import supervision as sv

RL = os.environ["MICRODUCK_RL"]
POLICIES = os.environ["MICRODUCK_POLICIES"]
sys.path.insert(0, os.path.join(RL, "scripts"))
os.chdir(RL)

from infer_policy import (  # noqa: E402
    BALL_OFFSET_ABS_Y,
    BALL_OFFSET_X,
    MICRODUCK_BALL_XML,
    PolicyInference,
)

# ---- The board ---------------------------------------------------------------
N = 8
SQUARE = 0.10
BOARD = N * SQUARE
BOARD_X = 0.90                    # board centre; the duck starts on the rank-1 side
MARGIN = 0.055                    # border markers sit outside the playing area
MARKERS_PER_EDGE = 7              # posts per edge, corners included
POST_H = (0.19, 0.13)             # alternating, so one edge of posts spans a plane, not a line
FILES = "abcdefgh"

# ---- The pieces ---------------------------------------------------------------
BASE_R, BASE_H, BASE_MASS = 0.045, 0.018, 0.08     # flat and heavy: launches one 10 cm square and stays up
HEIGHT = {"P": 0.10, "N": 0.12, "B": 0.13, "R": 0.12, "Q": 0.15, "K": 0.16}
WHITE = [0.92, 0.89, 0.82, 1]
BLACK = [0.10, 0.10, 0.12, 1]
FRICTION = [1.0, 0.02, 0.004]

# ---- The kick, measured -------------------------------------------------------
KICK_SCALE = 1.8                  # action scale during kick_right: launch, not push
KICK_YAW_OFFSET = math.radians(20.0)   # duck yaw = kick direction + this
# Where the foot expects the piece, 8 mm closer than the ball offset: a stand
# 12 mm short misses, 12 mm close still lands (measured).
# Measured on the 60 mm base: the sweet spot is 15 mm further out than the
# ball offset, and about 20 mm deep along the duck's facing.
KICK_FOOT = (BALL_OFFSET_X - 0.004, -BALL_OFFSET_ABS_Y)

# The head camera is rolled, so its vertical field is the narrow 53 degree one,
# and the stand pose pitches it 37 degrees down: the top of the frame is 10
# degrees below horizontal. Carrying the head this much up brings the marker
# posts and the floor into the same frame. Kicks still need a neutral head.
LOOK_PITCH = -0.30
STAND_HEIGHT = 0.116              # trunk above the floor when standing, measured
COAST = 0.02                      # metres the gait keeps moving after a stop command
CROWN_BIAS = 0.015                # crown-pixel range bias, measured

# ---- The camera --------------------------------------------------------------
POV_W, POV_H = 360, 720           # rendered portrait, rotated for display
FOVY = 90.0
FOCAL = (POV_H / 2) / math.tan(math.radians(FOVY) / 2)
K_MATRIX = np.array([[FOCAL, 0, POV_W / 2], [0, FOCAL, POV_H / 2], [0, 0, 1]], dtype=float)
CV_FROM_MJ = np.diag([1.0, -1.0, -1.0])


def square_name(fi, ri):
    return f"{FILES[fi]}{ri + 1}"


def square_index(name):
    return FILES.index(name[0]), int(name[1]) - 1


def board_xy(fi, ri):
    return ((fi + 0.5) * SQUARE, (ri + 0.5) * SQUARE)


def world_of(bx, by):
    """Board plane to world: ranks run along +x away from the duck's start,
    file a is on the duck's left (+y)."""
    return (BOARD_X - BOARD / 2 + by, BOARD / 2 - bx)


def board_of_world(wx, wy):
    return (BOARD / 2 - wy, wx - (BOARD_X - BOARD / 2))


def square_of_world(wx, wy):
    bx, by = board_of_world(wx, wy)
    fi, ri = int(math.floor(bx / SQUARE)), int(math.floor(by / SQUARE))
    return (fi, ri) if 0 <= fi < N and 0 <= ri < N else None


# ---- Piece meshes: turned on a lathe, the knight gets a head -------------------
PROFILES = {
    "P": [(0.045, 0.0), (0.045, 0.018), (0.030, 0.026), (0.018, 0.045), (0.016, 0.070),
          (0.024, 0.074), (0.020, 0.080), (0.026, 0.090), (0.014, 0.100), (0.001, 0.100)],
    "R": [(0.045, 0.0), (0.045, 0.018), (0.032, 0.028), (0.022, 0.050), (0.022, 0.095),
          (0.032, 0.100), (0.032, 0.120), (0.001, 0.120)],
    "B": [(0.045, 0.0), (0.045, 0.018), (0.030, 0.028), (0.018, 0.055), (0.016, 0.085),
          (0.026, 0.092), (0.021, 0.098), (0.012, 0.120), (0.006, 0.130), (0.001, 0.130)],
    "Q": [(0.045, 0.0), (0.045, 0.018), (0.032, 0.030), (0.020, 0.060), (0.017, 0.100),
          (0.028, 0.110), (0.024, 0.118), (0.030, 0.132), (0.016, 0.142), (0.008, 0.150),
          (0.001, 0.150)],
    "K": [(0.045, 0.0), (0.045, 0.018), (0.032, 0.030), (0.020, 0.062), (0.018, 0.105),
          (0.030, 0.116), (0.026, 0.124), (0.030, 0.138), (0.010, 0.146), (0.010, 0.160),
          (0.001, 0.160)],
    "N": [(0.045, 0.0), (0.045, 0.018), (0.032, 0.028), (0.024, 0.050), (0.024, 0.062),
          (0.001, 0.062)],
}
# Knight head silhouette in the (forward, up) plane, extruded sideways.
KNIGHT_HEAD = [(-0.020, 0.060), (0.014, 0.060), (0.020, 0.085), (0.032, 0.100),
               (0.026, 0.112), (0.010, 0.108), (0.000, 0.120), (-0.014, 0.118),
               (-0.022, 0.100), (-0.026, 0.080)]


def lathe(profile, segments=28):
    verts, faces = [], []
    rings = len(profile)
    for r, z in profile:
        for j in range(segments):
            a = 2 * math.pi * j / segments
            verts.append((r * math.cos(a), r * math.sin(a), z))
    for i in range(rings - 1):
        for j in range(segments):
            a, b = i * segments + j, i * segments + (j + 1) % segments
            c, d = a + segments, b + segments
            faces += [(a, b, d), (a, d, c)]
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)


def extrude(polygon, half_width):
    n = len(polygon)
    verts = [(x, -half_width, z) for x, z in polygon] + [(x, half_width, z) for x, z in polygon]
    faces = []
    for i in range(1, n - 1):
        faces += [(0, i + 1, i), (n, n + i, n + i + 1)]
    for i in range(n):
        j = (i + 1) % n
        faces += [(i, j, n + j), (i, n + j, n + i)]
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)


def add_meshes(spec):
    for kind, profile in PROFILES.items():
        v, f = lathe(profile)
        mesh = spec.add_mesh()
        mesh.name = f"piece_{kind}"
        mesh.uservert = v.flatten().tolist()
        mesh.userface = f.flatten().tolist()
    v, f = extrude(KNIGHT_HEAD, 0.016)
    mesh = spec.add_mesh()
    mesh.name = "knight_head"
    mesh.uservert = v.flatten().tolist()
    mesh.userface = f.flatten().tolist()


def add_piece(spec, name, letter, x, y):
    """Physics is two primitives: a heavy flat base and a light body, which is
    the combination that kicks one square and lands upright. The lathe mesh is
    what you see."""
    kind = letter.upper()
    colour = WHITE if letter.isupper() else BLACK
    body = spec.worldbody.add_body(name=name, pos=[x, y, 0.0])
    body.add_freejoint(name=f"{name}_free")
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[BASE_R, BASE_H / 2, 0],
                  pos=[0, 0, BASE_H / 2], mass=BASE_MASS, condim=6, friction=FRICTION,
                  rgba=colour, group=3)
    height = HEIGHT[kind]
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.018, (height - BASE_H) / 2 - 0.018, 0],
                  pos=[0, 0, BASE_H + (height - BASE_H) / 2], mass=0.006, condim=6,
                  friction=FRICTION, rgba=colour, group=3)
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"piece_{kind}", rgba=colour,
                  contype=0, conaffinity=0, mass=0.0, group=0)
    if kind == "N":
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="knight_head", rgba=colour,
                      contype=0, conaffinity=0, mass=0.0, group=0,
                      quat=[math.cos(-math.pi / 4), 0, 0, math.sin(-math.pi / 4)])


def marker_layout():
    """Marker posts around the board, in board-plane coordinates with the edge
    each sits on. Spheres on posts: a sphere projects as a circle whose
    centroid is its centre, and the posts clear the pieces. Heights alternate
    along an edge, so even a single visible edge gives PnP a plane to work
    with rather than a line."""
    points = []
    lo, hi = -MARGIN, BOARD + MARGIN
    ticks = np.linspace(0, BOARD, MARKERS_PER_EDGE)
    for i, t in enumerate(ticks):
        h = POST_H[i % 2]
        points.append(((t, lo, h), "near"))
        points.append(((t, hi, h), "far"))
    for i, t in enumerate(ticks[1:-1]):
        h = POST_H[(i + 1) % 2]
        points.append(((lo, t, h), "left"))
        points.append(((hi, t, h), "right"))
    return points


def build(position):
    spec = mujoco.MjSpec.from_file(MICRODUCK_BALL_XML)
    # The kick policy is ball-blind and the demo kicks pieces: hide the
    # practice ball and park it off the set so nothing meets it.
    for body in spec.bodies:
        if body.name == "ball":
            body.pos = [-3.0, -3.0, 0.05]
            for geom in body.geoms:
                geom.rgba = [0, 0, 0, 0]
                geom.group = 3
                geom.contype = 0
                geom.conaffinity = 0
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    add_meshes(spec)
    for fi in range(N):
        for ri in range(N):
            x, y = world_of(*board_xy(fi, ri))
            light = (fi + ri) % 2 == 1
            spec.worldbody.add_body(name=f"sq{fi}{ri}", pos=[x, y, 0.001]).add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX, size=[SQUARE / 2, SQUARE / 2, 0.001],
                rgba=[0.90, 0.87, 0.80, 1] if light else [0.34, 0.24, 0.17, 1],
                contype=0, conaffinity=0)
    markers = []
    for i, ((bx, by, bz), edge) in enumerate(marker_layout()):
        x, y = world_of(bx, by)
        hue = i / 32
        rgba = [0.5 + 0.5 * math.cos(2 * math.pi * hue), 0.5 + 0.5 * math.cos(2 * math.pi * (hue + 1 / 3)),
                0.5 + 0.5 * math.cos(2 * math.pi * (hue + 2 / 3)), 1]
        body = spec.worldbody.add_body(name=f"marker{i}", pos=[x, y, 0.0])
        if bz > 0.01:
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.004, bz / 2, 0], pos=[0, 0, bz / 2],
                          rgba=[0.75, 0.75, 0.75, 1], contype=0, conaffinity=0)
            body.add_geom(name=f"marker{i}", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.015, 0, 0],
                          pos=[0, 0, bz], rgba=rgba, contype=0, conaffinity=0)
        else:
            body.add_geom(name=f"marker{i}", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.012, 0.0012, 0],
                          pos=[0, 0, bz], rgba=rgba, contype=0, conaffinity=0)
        markers.append((f"marker{i}", (x, y, bz), edge))
    pieces = {}
    for square, letter in position.items():
        x, y = world_of(*board_xy(*square_index(square)))
        name = f"{square}_{letter}"
        add_piece(spec, name, letter, x, y)
        pieces[name] = letter
    for camera in spec.cameras:
        if camera.name == "head_camera":
            angle = math.radians(25)
            camera.quat = [math.cos(angle / 2), 0, math.sin(angle / 2), 0]
            camera.fovy = FOVY
    model = spec.compile()
    model.opt.timestep = 0.005
    return model, markers, pieces


# ---- Seeing --------------------------------------------------------------------
def geom_ids_of_body(model, body_name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid]


def mask_of(segmentation, geom_ids):
    is_geom = segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    return np.isin(segmentation[..., 0], geom_ids) & is_geom


def box_of(mask, minimum=6):
    if mask.sum() < minimum:
        return None
    return sv.mask_to_xyxy(mask[None], coordinate_convention="exclusive")[0]


def centroid_of(mask, minimum=4):
    if mask.sum() < minimum:
        return None
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()])


class Eyes:
    """Renders the head camera and pulls markers and pieces out of the frame."""

    def __init__(self, model, data, markers, pieces):
        self.model, self.data = model, data
        self.renderer = mujoco.Renderer(model, height=POV_H, width=POV_W)
        self.segmenter = mujoco.Renderer(model, height=POV_H, width=POV_W)
        self.segmenter.enable_segmentation_rendering()
        self.options = mujoco.MjvOption()
        self.options.geomgroup[2] = 0
        self.options.geomgroup[3] = 0
        self.markers = markers
        self.marker_ids = {name: [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)]
                           for name, _, _ in markers}
        self.piece_ids = {name: geom_ids_of_body(model, name) for name in pieces}
        self.piece_letters = dict(pieces)
        self.camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")

    def look(self):
        self.renderer.update_scene(self.data, camera="head_camera", scene_option=self.options)
        frame = np.ascontiguousarray(np.rot90(self.renderer.render()))
        self.segmenter.update_scene(self.data, camera="head_camera", scene_option=self.options)
        portrait = self.segmenter.render()
        seen_markers = {}
        for name, world, edge in self.markers:
            c = centroid_of(mask_of(portrait, self.marker_ids[name]))
            if c is not None:
                seen_markers[name] = (c, world, edge)
        rotated = np.rot90(portrait)
        seen_pieces = {}
        for name, ids in self.piece_ids.items():
            mask = mask_of(portrait, ids)
            b = box_of(mask_of(rotated, ids))
            if b is not None:
                # The camera is rolled: world-down is -x in the portrait
                # render, so the base of the piece is its leftmost column.
                # (Verified against ground truth; the other end is the top,
                # and a ray through the top lands 1.6x too far away.)
                ys, xs = np.nonzero(mask)
                x_base = xs.min()
                y_base = ys[xs <= x_base + 1].mean()
                x_top = xs.max()
                y_top = ys[xs >= x_top - 1].mean()
                seen_pieces[name] = (b, np.array([x_base, y_base], dtype=float),
                                     np.array([x_top, y_top], dtype=float))
        return frame, seen_markers, seen_pieces

    def trunk_from_camera(self):
        """Camera pose in the trunk frame: forward kinematics of the neck and
        head joints, which the real robot reads from its encoders."""
        trunk = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        T_w_t = np.eye(4)
        T_w_t[:3, :3] = self.data.xmat[trunk].reshape(3, 3)
        T_w_t[:3, 3] = self.data.xpos[trunk]
        T_w_c = np.eye(4)
        T_w_c[:3, :3] = self.data.cam_xmat[self.camera_id].reshape(3, 3)
        T_w_c[:3, 3] = self.data.cam_xpos[self.camera_id]
        return np.linalg.inv(T_w_t) @ T_w_c


def localize(seen_markers, T_t_c):
    """Duck pose (x, y, yaw) and camera pose in the world from the markers in
    view. Needs four markers not all on one edge. Returns (None, None)
    otherwise. PnP gives the camera; neck kinematics take it to the trunk."""
    if len(seen_markers) < 4:
        return None, None
    items = list(seen_markers.values())
    image = np.array([c for c, _, _ in items], dtype=np.float64)
    world = np.array([w for _, w, _ in items], dtype=np.float64)
    edges = [e.replace("_post", "") for _, _, e in items]

    def spread_ok(idx):
        centred = world[idx] - world[idx].mean(axis=0)
        return np.linalg.svd(centred, compute_uv=False)[1] > 0.04

    def solve(idx):
        ok, rv, tv = cv2.solvePnP(world[idx], image[idx], K_MATRIX, None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            return None
        proj, _ = cv2.projectPoints(world[idx], rv, tv, K_MATRIX, None)
        return rv, tv, np.linalg.norm(proj.reshape(-1, 2) - image[idx], axis=1)

    # A marker half-hidden behind a piece or cut by the frame has a biased
    # centroid. Fit, drop the worst, refit, until everything reprojects.
    idx = np.arange(len(image))
    while True:
        if len(idx) < 4 or not spread_ok(idx):
            return None, None
        result = solve(idx)
        if result is None:
            return None, None
        rvec, tvec, err = result
        if err.max() <= 6.0:
            break
        idx = np.delete(idx, int(np.argmax(err)))
    R_cv, _ = cv2.Rodrigues(rvec)
    T_mj_w = np.eye(4)
    T_mj_w[:3, :3] = CV_FROM_MJ @ R_cv
    T_mj_w[:3, 3] = CV_FROM_MJ @ tvec.ravel()
    T_w_c = np.linalg.inv(T_mj_w)
    if not 0.08 < T_w_c[2, 3] < 0.6:
        return None, None
    T_w_t = T_w_c @ np.linalg.inv(T_t_c)
    yaw = math.atan2(T_w_t[1, 0], T_w_t[0, 0])
    return np.array([T_w_t[0, 3], T_w_t[1, 3], yaw]), T_w_c


def ground_point(T_w_c, pixel):
    """Where the ray through a portrait pixel meets the floor."""
    d_cv = np.linalg.inv(K_MATRIX) @ np.array([pixel[0], pixel[1], 1.0])
    d_w = T_w_c[:3, :3] @ (CV_FROM_MJ @ d_cv)
    o = T_w_c[:3, 3]
    if d_w[2] >= -1e-6:
        return None
    s = -o[2] / d_w[2]
    return o + s * d_w


def project(T_w_c, point):
    """Portrait pixel of a world point, or None if behind the camera."""
    p_c = CV_FROM_MJ @ (np.linalg.inv(T_w_c) @ np.array([*point, 1.0]))[:3]
    if p_c[2] <= 1e-6:
        return None
    u = K_MATRIX @ (p_c / p_c[2])
    return u[:2]


# ---- Remembering -----------------------------------------------------------------
class BoardMemory:
    """The board the duck believes in, kept per piece rather than per square.

    A piece has an identity (the tracker's, standing in here for a detector
    that tells pieces apart), so it lives on exactly one square: the majority
    of its last few sightings. A piece out of view stays where it was last
    seen. That is the whole point of tracking for a board you cannot see all
    of at once, and it also kills the phantoms a per-square memory collects
    when a far piece's estimate jitters across a boundary as the head sweeps.
    """

    HISTORY = 7
    AGREE = 3

    def __init__(self):
        self.pieces = {}      # piece id -> dict(letter, history, square, step)

    def observe(self, piece_id, letter, square, step):
        if square is None:
            return            # seen but not placeable (ranged off the board): not evidence it is gone
        entry = self.pieces.setdefault(piece_id, {"letter": letter, "history": [], "square": None, "step": step})
        entry["history"] = (entry["history"] + [square])[-self.HISTORY:]
        best, count = max(((sq, entry["history"].count(sq)) for sq in set(entry["history"])), key=lambda x: x[1])
        if count >= self.AGREE:
            entry["square"], entry["step"] = best, step

    def apply_move(self, src, dst, step):
        """The duck saw the piece leave its foot: move it in memory, and give
        it a fresh, unanimous history there, so a few biased sightings on the
        walk home cannot put it back. The next reads still decide."""
        for name, entry in self.pieces.items():
            if entry["square"] == src:
                entry["square"] = dst
                entry["history"] = [dst] * self.HISTORY
                entry["step"] = step
                return

    def letters(self):
        """Square -> letter for every piece that has settled on a square."""
        out = {}
        for entry in self.pieces.values():
            if entry["square"] is not None:
                out[entry["square"]] = entry["letter"]
        return out

    def as_chess_board(self, turn):
        board = chess.Board(None)
        for (fi, ri), letter in self.letters().items():
            board.set_piece_at(chess.square(fi, ri), chess.Piece.from_symbol(letter))
        board.turn = turn
        return board


def in_frame(pixel, margin=25):
    """A pixel on the frame edge is a clipped piece, not a measurement."""
    return margin < pixel[0] < POV_W - margin and margin < pixel[1] < POV_H - margin


def piece_world(T_w_c, entry, kind):
    """A seen piece's centre on the floor, in the world. The crown is on the
    axis at a known height and is the reliable pixel; the base's front rim,
    one radius short of centre, is the fallback when the crown is cut off."""
    _, base, crown = entry
    if in_frame(crown):
        hit, d = ray_hit(T_w_c, crown, HEIGHT[kind])
        if hit is None:
            return None
        away = d[:2] / max(np.linalg.norm(d[:2]), 1e-9)
        return hit[:2] - CROWN_BIAS * away
    if in_frame(base):
        hit, d = ray_hit(T_w_c, base, 0.0)
        if hit is None:
            return None
        away = d[:2] / max(np.linalg.norm(d[:2]), 1e-9)
        return hit[:2] + BASE_R * away
    return None


def ray_hit(T_w_c, pixel, plane_z):
    d_cv = np.linalg.inv(K_MATRIX) @ np.array([pixel[0], pixel[1], 1.0])
    d_w = T_w_c[:3, :3] @ (CV_FROM_MJ @ d_cv)
    o = T_w_c[:3, 3]
    if d_w[2] >= -1e-6:
        return None, None
    s_ = (plane_z - o[2]) / d_w[2]
    return o + s_ * d_w, d_w


def sightings_from(seen_pieces, pieces, T_w_c, tracker):
    """Each seen piece to a square, through the camera onto its height.
    Returns {piece name: square} and the tracked detections for drawing."""
    if T_w_c is None or not seen_pieces:
        return {}, None
    names = list(seen_pieces)
    boxes = np.array([seen_pieces[n][0] for n in names], dtype=float)
    tracked = tracker.update(sv.Detections(xyxy=boxes, confidence=np.ones(len(boxes)),
                                           class_id=np.arange(len(boxes))))
    tracked = tracked[tracked.tracker_id != -1]
    sightings = {}
    for name in names:
        centre = piece_world(T_w_c, seen_pieces[name], pieces[name].upper())
        if centre is None:
            continue
        sq = square_of_world(centre[0], centre[1])
        if sq is not None:
            sightings[name] = sq
    return sightings, tracked


def visible_squares(T_w_c, occupied=(), margin=60, nearest=0.30, shadow=0.075):
    """Squares looked at squarely: centre well inside the frame, not so close
    that the duck's own body is in the way, and not behind another piece.
    A square in the shadow of a piece was not observed, whatever the frame
    says."""
    if T_w_c is None:
        return set()
    out = set()
    cam = np.array(T_w_c[:2, 3])
    blockers = [np.array(world_of(*board_xy(*sq))) for sq in occupied]
    for fi in range(N):
        for ri in range(N):
            centre = np.array(world_of(*board_xy(fi, ri)))
            ray = centre - cam
            dist = np.linalg.norm(ray)
            if dist < nearest:
                continue
            uv = project(T_w_c, (centre[0], centre[1], 0.0))
            if uv is None or not (margin < uv[0] < POV_W - margin and margin < uv[1] < POV_H - margin):
                continue
            unit = ray / dist
            shadowed = False
            for b in blockers:
                if np.allclose(b, centre):
                    continue
                along = float(np.dot(b - cam, unit))
                offset = b - cam
                across = abs(unit[0] * offset[1] - unit[1] * offset[0])
                if 0.05 < along < dist - 0.05 and across < shadow:
                    shadowed = True
                    break
            if not shadowed:
                out.add((fi, ri))
    return out


# ---- Deciding ----------------------------------------------------------------------
def choose_move(memory, turn):
    """A legal, non-capturing, one-square orthogonal move. One kick, one move."""
    board = memory.as_chess_board(turn)
    options = []
    for move in board.legal_moves:
        if board.is_capture(move) or move.promotion:
            continue
        df = chess.square_file(move.to_square) - chess.square_file(move.from_square)
        dr = chess.square_rank(move.to_square) - chess.square_rank(move.from_square)
        if (abs(df), abs(dr)) in ((1, 0), (0, 1)):
            options.append(move)
    return options, board


def kick_yaw_from(point, dst):
    """The heading to kick a piece standing at a world point into a square:
    aim at the square's centre from where the piece actually is, not from
    the centre of the square it is on. A piece 25 mm across at 100 mm range
    is a 14 degree difference, and consecutive kicks would otherwise carry
    the offset along."""
    dx, dy = world_of(*board_xy(*dst))
    return math.atan2(dy - point[1], dx - point[0]) + KICK_YAW_OFFSET


def kick_pose(src, dst):
    """Where the duck must stand, and face, to kick a piece from src to dst."""
    sx, sy = world_of(*board_xy(*src))
    dx, dy = world_of(*board_xy(*dst))
    direction = math.atan2(dy - sy, dx - sx)
    yaw = direction + KICK_YAW_OFFSET
    fx, fy = KICK_FOOT
    px = sx - (fx * math.cos(yaw) - fy * math.sin(yaw))
    py = sy - (fx * math.sin(yaw) + fy * math.cos(yaw))
    return np.array([px, py, yaw])


def markers_in_view(pose, markers, T_t_c_neutral):
    """How many posts the camera would see from a pose with the head neutral:
    the duck knows where the posts are, so it can prefer moves it can see."""
    T_w_t = np.eye(4)
    c, s_ = math.cos(pose[2]), math.sin(pose[2])
    T_w_t[:3, :3] = [[c, -s_, 0], [s_, c, 0], [0, 0, 1]]
    T_w_t[:3, 3] = [pose[0], pose[1], 0.125]
    T_w_c = T_w_t @ T_t_c_neutral
    n = 0
    for _, w, _ in markers:
        uv = project(T_w_c, w)
        if uv is not None and 10 < uv[0] < POV_W - 10 and 10 < uv[1] < POV_H - 10:
            n += 1
    return n


def stand_is_clear(occupied, src, dst, radius=0.15):
    """A move is only playable if the duck can stand behind the piece, and
    approach it, without standing in another piece."""
    pose = kick_pose(src, dst)
    approach = (pose[0] - 0.22 * math.cos(pose[2]), pose[1] - 0.22 * math.sin(pose[2]))
    for sq in occupied:
        if sq == src:
            continue
        px, py = world_of(*board_xy(*sq))
        for x, y in ((pose[0], pose[1]), approach):
            if math.hypot(px - x, py - y) < radius:
                return False
    return True


# ---- Walking ------------------------------------------------------------------------
class Navigator:
    """Walk to a pose the way the gait is comfortable: arcs.

    Measured on alpha_walking: wz=0.5 with 0.30 forward is a clean arc of
    about 0.28 m radius; wz=1.0 with no forward command turns in place about
    35 degrees in the first second and little after; small commands do
    nothing. So heading is corrected by proportional arcs, and only a large
    error gets an in-place burst.
    """

    WALK, CREEP, TURN_FWD = 0.30, 0.22, 0.15

    def __init__(self, target, standoff=0.22):
        self.target = np.array(target, dtype=float)
        self.standoff = standoff
        self.phase = "approach"
        self.done = False

    @staticmethod
    def wrap(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    def command(self, pose):
        x, y, yaw = pose
        tx, ty, tyaw = self.target
        if self.phase == "approach":
            gx = tx - self.standoff * math.cos(tyaw)
            gy = ty - self.standoff * math.sin(tyaw)
            if math.hypot(gx - x, gy - y) < 0.05:
                self.phase = "align"
            else:
                bearing = self.wrap(math.atan2(gy - y, gx - x) - yaw)
                if abs(bearing) > 1.0:
                    return 0.0, 0.0, float(np.sign(bearing)) * 1.0
                return self.WALK, 0.0, float(np.clip(1.2 * bearing, -0.5, 0.5))
        if self.phase == "align":
            err = self.wrap(tyaw - yaw)
            if abs(err) < 0.12:
                self.phase = "final"
            else:
                return 0.0, 0.0, float(np.sign(err)) * 1.0
        if self.phase == "final":
            along = (tx - x) * math.cos(tyaw) + (ty - y) * math.sin(tyaw)
            if along < 0.012:
                self.done = True
                return 0.0, 0.0, 0.0
            err = self.wrap(math.atan2(ty - y, tx - x) - yaw) if along > 0.06 else self.wrap(tyaw - yaw)
            return self.CREEP, 0.0, float(np.clip(1.2 * err, -0.5, 0.5))
        return 0.0, 0.0, 0.0


class DeadReckoning:
    """Between marker sightings, integrate the commanded velocity through the
    measured gait response (0.30 forward -> 0.12 m/s; wz 1.0 -> ~0.5 rad/s)."""

    def __init__(self):
        self.pose = None

    def observe(self, pose):
        self.pose = np.array(pose, dtype=float)

    @staticmethod
    def actual(command):
        vx, vy, wz = command
        if abs(wz) >= 0.8 and vx < 0.1:
            return 0.0, 0.0, 0.47 * wz          # turn bursts: about 27 degrees a second
        if abs(wz) >= 0.8:
            return 0.06, 0.0, 0.62 * wz         # forward turn: about 35 degrees and 60 mm a second
        if vx < 0.2:
            return 0.0, 0.0, 0.0
        return 0.40 * vx, 0.2 * vy, 0.6 * wz - 0.04

    def advance(self, command, dt):
        if self.pose is None:
            return None
        vx, vy, wz = self.actual(command)
        x, y, yaw = self.pose
        x += (vx * math.cos(yaw) - vy * math.sin(yaw)) * dt
        y += (vx * math.sin(yaw) + vy * math.cos(yaw)) * dt
        yaw = Navigator.wrap(yaw + wz * dt)
        self.pose = np.array([x, y, yaw])
        return self.pose


class Pilot:
    """Drives the duck to a kick pose on its own senses.

    Looks at the board centre while walking so markers stay in view, sweeps
    the head when it has lost them, and finishes by standing still to take
    clean fixes and creeping until the stand is within a centimetre.
    """

    def __init__(self, duck, eyes, on_frame=None):
        self.duck, self.eyes, self.on_frame = duck, eyes, on_frame
        self.dr = DeadReckoning()
        self.since_fix = 0.0
        self.fixes = 0
        self.looks = 0
        self.head_yaw = 0.0
        self.last_view = None
        self.T_w_c = None
        self.piece_kinds = {name: letter.upper() for name, letter in eyes.piece_letters.items()}

    def look(self, head_yaw):
        frame, seen_m, seen_p = self.eyes.look()
        self.looks += 1
        est, T_w_c = localize(seen_m, self.eyes.trunk_from_camera())
        if est is not None:
            self.dr.observe(est)
            self.fixes += 1
            self.since_fix = 0.0
            self.T_w_c = T_w_c
        self.last_view = (frame, seen_m, seen_p)
        return est

    def gaze(self, pose):
        """Head yaw that points the camera at the board centre."""
        if pose is None:
            return 0.0
        bearing = math.atan2(-pose[1], BOARD_X - pose[0]) - pose[2]
        return float(np.clip(Navigator.wrap(bearing), -0.7, 0.7))

    def tick(self, command, t, head=None, dr_command=None):
        # Sweep the head while walking: with pieces on the board, no single
        # gaze direction keeps markers in view from everywhere.
        pitch, self.head_yaw = head if head is not None else (LOOK_PITCH, 0.6 * math.sin(2.0 * t))
        if int(t / self.duck.dt) % 4 == 0:
            self.look(self.head_yaw)
        self.duck.step(command, head_yaw=self.head_yaw, head_pitch=pitch)
        self.dr.advance(command if dr_command is None else dr_command, self.duck.dt)
        self.since_fix += self.duck.dt
        if self.on_frame:
            self.on_frame(self, command)

    def settle_fix(self, t, looks=5):
        """Stand still and average fresh fixes: the best localization we get."""
        poses = []
        for i in range(looks * 4):
            cmd = (0.0, 0.0, 0.0)
            self.duck.step(cmd, head_yaw=self.head_yaw)
            if i % 4 == 0:
                est = self.look(self.head_yaw)
                if est is not None:
                    poses.append(est)
            if self.on_frame:
                self.on_frame(self, cmd)
        if not poses:
            return None
        p = np.mean(poses, axis=0)
        self.dr.observe(p)
        return p

    def burst(self, command, seconds, t):
        """A short open-loop move, then stand and take fresh fixes."""
        for _ in range(int(seconds / self.duck.dt)):
            self.tick(command, t)
            t += self.duck.dt
        for _ in range(int(0.5 / self.duck.dt)):
            self.tick((0.0, 0.0, 0.0), t)
            t += self.duck.dt
        self.settle_fix(t, looks=4)
        return t

    def turn_to(self, yaw, t, within=math.radians(4.0), limit=3.0):
        """Closed-loop turn on the spot: the open-loop response is not
        repeatable (the same 0.8 s command turned 2 to 30 degrees depending
        on what the gait did just before), but with the board in view the
        posts give a heading every few ticks, so keep turning until it is
        right. Head straight, so the posts ahead stay in view."""
        # Whether a turn command takes depends on what the gait did just
        # before (the same command turned 52 degrees one time and nothing
        # the next); stronger variants drift up to 10 cm, so a stalled turn
        # is left alone: the kick survives 15 degrees, a 10 cm shift not.
        t0 = t
        while t - t0 < limit:
            if self.dr.pose is None:
                break
            err = Navigator.wrap(yaw - self.dr.pose[2])
            if abs(err) < within:
                break
            command = (0.0, 0.0, 1.0) if err > 0 else (0.0, -0.1, -1.0)
            self.tick(command, t, head=(LOOK_PITCH, 0.0), dr_command=(0.0, 0.0, command[2]))
            t += self.duck.dt
        for _ in range(int(0.5 / self.duck.dt)):
            self.tick((0.0, 0.0, 0.0), t, head=(LOOK_PITCH, 0.0))
            t += self.duck.dt
        self.settle_fix(t, looks=4)
        return t

    def back_up(self, t, seconds=2.0):
        """Walk backward with the head straight: the gait walks backward
        only with the head neutral (measured: 20 cm in 2 s straight, nothing
        with the head sweeping), and the piece stays in view ahead."""
        for _ in range(int(seconds / self.duck.dt)):
            self.tick((-0.30, 0.0, 0.0), t, head=(LOOK_PITCH, 0.0))
            t += self.duck.dt
        for _ in range(int(0.5 / self.duck.dt)):
            self.tick((0.0, 0.0, 0.0), t)
            t += self.duck.dt
        self.settle_fix(t, looks=4)
        return t

    def turn_burst(self, sign, t, seconds=1.0):
        """Turn on the spot. Measured on alpha_walking: a positive yaw command
        alone turns left about 25 degrees a second, a negative one alone does
        nothing, and a negative one with 0.1 m/s of sideways command turns
        right about 28 degrees a second with 6 mm of drift. The dead
        reckoning is told about the turn, not the sideways command."""
        command = (0.0, 0.0, 1.0) if sign > 0 else (0.0, -0.1, -1.0)
        for _ in range(int(seconds / self.duck.dt)):
            self.tick(command, t, dr_command=(0.0, 0.0, command[2]))
            t += self.duck.dt
        for _ in range(int(0.5 / self.duck.dt)):
            self.tick((0.0, 0.0, 0.0), t)
            t += self.duck.dt
        self.settle_fix(t, looks=4)
        return t

    # Measured yaw per burst on alpha_walking (degrees, three trials each):
    # left  0.6 s -> 8, 0.8 s -> 29; right 0.25 s -> 7, 0.4 s -> 18, 0.8 s -> 26.
    # The response is a staircase, so a correction picks the nearest step.
    TURN_TABLE = {+1: ((0.6, 8.0), (0.8, 29.0)), -1: ((0.25, 7.0), (0.4, 18.0), (0.8, 26.0))}

    def align_yaw(self, yaw, t, within=math.radians(6.0), tries=4):
        """Turn to a heading with calibrated bursts, nearest step first."""
        for _ in range(tries):
            if self.dr.pose is None:
                break
            err = Navigator.wrap(yaw - self.dr.pose[2])
            if abs(err) < within:
                break
            sign = 1 if err > 0 else -1
            secs, _ = min(self.TURN_TABLE[sign], key=lambda st: abs(st[1] - abs(math.degrees(err))))
            t = self.turn_burst(sign, t, secs)
        return t

    def errors(self, target):
        pose = self.dr.pose
        along = (target[0] - pose[0]) * math.cos(target[2]) + (target[1] - pose[1]) * math.sin(target[2])
        across = -(target[0] - pose[0]) * math.sin(target[2]) + (target[1] - pose[1]) * math.cos(target[2])
        return along, across, Navigator.wrap(target[2] - pose[2])

    def piece_relative(self, piece_name):
        """The piece's centre in the trunk frame, from the top of the piece.

        A turned piece's crown sits on its axis at a known height, and it
        stays in frame when the base has already dropped below the bottom of
        the view. The ray through the crown, through the camera's known
        height and pitch (neck kinematics), meets the plane at the piece's
        height at its centre. No markers involved."""
        if self.last_view is None:
            return None
        _, _, seen = self.last_view
        if piece_name not in seen:
            return None
        _, base, crown = seen[piece_name]
        T_t_c = self.eyes.trunk_from_camera()

        def hit(pixel, plane_z):
            d_cv = np.linalg.inv(K_MATRIX) @ np.array([pixel[0], pixel[1], 1.0])
            d_t = T_t_c[:3, :3] @ (CV_FROM_MJ @ d_cv)
            o_t = T_t_c[:3, 3]
            if d_t[2] >= -1e-6:
                return None, None
            s_ = (plane_z - o_t[2]) / d_t[2]
            return (o_t + s_ * d_t)[:2], d_t

        if in_frame(crown):
            p, d_t = hit(crown, HEIGHT[self.piece_kinds[piece_name]] - STAND_HEIGHT)
            if p is None:
                return None
            # Calibrated against ground truth: the crown's extreme pixel reads
            # about 15 mm long at every range, so pull it back along the ray.
            away = np.array([d_t[0], d_t[1]])
            away /= max(np.linalg.norm(away), 1e-9)
            return p - CROWN_BIAS * away
        if in_frame(base):
            # Crown out of frame (it happens far away under a shallow gaze):
            # fall back to the base's front rim, one radius short of centre.
            p, d_t = hit(base, -STAND_HEIGHT)
            if p is None:
                return None
            away = np.array([d_t[0], d_t[1]])
            away /= max(np.linalg.norm(away), 1e-9)
            return p + BASE_R * away
        return None

    SERVO_HEAD = (0.55, -0.45)      # steep down, yaw right: the base rim stays in frame to 3 cm
    SERVO_HEAD_FAR = (0.30, -0.30)  # shallower while the piece is still more than 0.2 m away

    def measure_piece(self, piece_name, t, looks=4):
        """Stand, look at the piece, average what it sees."""
        readings = []
        for i in range(looks * 4):
            self.tick((0.0, 0.0, 0.0), t, head=self.SERVO_HEAD)
            t += self.duck.dt
            if i % 4 == 3:
                rel = self.piece_relative(piece_name)
                if rel is not None:
                    readings.append(rel)
        return (np.mean(readings, axis=0) if readings else None), t

    # The kick lands the piece with the foot spot anywhere from 15 mm past
    # to 8 mm short of the piece (measured), so the stop aims 4 mm past the
    # centre of that window to leave room for the coast.
    STOP_AHEAD = 0.003
    KICKSTART = float(os.environ.get("KICKSTART", 0.8))   # walking-speed seconds at the start of a servo

    def servo_to_piece(self, piece_name, t, limit=30.0, yaw=None):
        """Walk in on the piece continuously, watching it every few steps.

        Short open-loop bursts of this gait are dominated by start and stop
        transients (measured: a 0.6 s creep moves 14 mm one time and a 1 s creep
        56 mm plus 29 mm of coast the next), so the walk never stops until the
        piece is where the foot wants it, minus the coast. Heading is steered
        bang-bang on the piece's sideways offset, because the gait ignores
        small turn commands. The head looks shallow while the piece is far and
        steep once it is close, since the rolled camera's vertical field is
        narrow."""
        t0 = t
        want = np.array(KICK_FOOT)
        missing, backoffs = 0, 0
        distance = 0.35
        while t - t0 < limit:
            head = self.SERVO_HEAD if distance < 0.2 else self.SERVO_HEAD_FAR
            rel = self.piece_relative(piece_name)
            if rel is None:
                missing += 1
                command = (0.0, -0.1, -1.0) if missing > 15 else (Navigator.CREEP, 0.0, 0.0)
            else:
                missing = 0
                distance = float(rel[0])
                e_x, e_y = rel[0] - want[0], rel[1] - want[1]
                self.last_servo_error = (e_x, e_y)
                if e_x < self.STOP_AHEAD:
                    break
                # Steer by a gentle arc on the sideways offset, damped by the
                # heading error so the duck arrives aimed along the file, not
                # merely centred on the piece: the kick tolerates 12 degrees.
                yaw_err = 0.0 if (yaw is None or self.dr.pose is None) else Navigator.wrap(self.dr.pose[2] - yaw)
                turn = 3.0 * e_y / max(rel[0], 0.10) - 1.5 * yaw_err
                turn = float(np.clip(turn, -0.5, 0.5))
                if abs(turn) < 0.3 or rel[0] < 0.06:
                    turn = 0.0
                if e_x < 0.12 and t - t0 >= self.KICKSTART:
                    command = (Navigator.CREEP, 0.0, turn)
                else:
                    # From a standstill the creep command takes many seconds
                    # to get the gait going (measured: 12 s), so a short gap
                    # is opened at walking speed before easing off.
                    command = (Navigator.WALK, 0.0, turn)
            self.tick(command, t, head=head)
            t += self.duck.dt
        for i in range(int(1.0 / self.duck.dt)):
            self.tick((0.0, 0.0, 0.0), t, head=self.SERVO_HEAD)
            t += self.duck.dt
        rel, t = self.measure_piece(piece_name, t)
        if rel is not None:
            self.last_servo_error = (rel[0] - want[0], rel[1] - want[1])
        ok = rel is not None and abs(rel[0] - want[0]) < 0.03 and abs(rel[1] - want[1]) < 0.03
        return t, ok

    def piece_world_xy(self, piece_name, t):
        """Where the piece stands, from the foot's view and the duck's own
        pose. None if the piece is not seen."""
        rel, t = self.measure_piece(piece_name, t)
        if rel is None or self.dr.pose is None:
            return None, t
        x, y, yaw = self.dr.pose
        c, s = math.cos(yaw), math.sin(yaw)
        return (x + c * rel[0] - s * rel[1], y + s * rel[0] + c * rel[1]), t

    def square_up(self, piece_name, yaw, t, accept=math.radians(9.0), rounds=3):
        """Before the kick: fresh fixes with the head up, and if the heading
        has wandered (the creep in turns the duck up to 30 degrees, and a
        kick 30 degrees off the file lands the piece on the corner of the
        square), turn on the spot under closed loop, settle, and look again.
        One round overshoots by about 10 degrees; two or three converge.
        Then close the foot spot again."""
        turned = False
        for _ in range(rounds):
            self.settle_fix(t, looks=5)
            if self.dr.pose is None:
                break
            before = math.degrees(Navigator.wrap(self.dr.pose[2] - yaw))
            if abs(before) < math.degrees(accept):
                break
            t = self.turn_to(yaw, t, within=math.radians(6.0))
            turned = True
            after = None if self.dr.pose is None else math.degrees(Navigator.wrap(self.dr.pose[2] - yaw))
            print(f"  square_up: heading {before:+.0f} -> {after:+.0f} deg")
        if turned:
            t, _ = self.nudge_to_piece(piece_name, t)
        return t

    def relocalize(self, t):
        """Lost: stand still and sweep the head, and only if that finds no
        post, turn a little and try again. Walking blind is what drifts."""
        for _ in range(6):
            for i in range(int(1.5 / self.duck.dt)):
                self.tick((0.0, 0.0, 0.0), t, head=(LOOK_PITCH, 0.7 * math.sin(4.0 * i * self.duck.dt)))
                t += self.duck.dt
                if self.since_fix < 0.1:
                    return t
            t = self.burst((0.0, 0.0, 1.0), 1.0, t)
        return t

    def nudge_to_piece(self, piece_name, t, tries=5):
        """After a stop, close the last millimetres: tiny creeps forward, and
        a swing out and back in if it has gone past. Returns the last error."""
        want = np.array(KICK_FOOT)
        err = None
        for _ in range(tries):
            rel, t = self.measure_piece(piece_name, t)
            if rel is None:
                break
            err = (rel[0] - want[0], rel[1] - want[1])
            self.last_servo_error = err
            if abs(err[0]) <= 0.006 and abs(err[1]) <= 0.014:
                break
            if err[0] < -0.015 or abs(err[1]) > 0.03:
                break            # past it, or off line: a swing here knocks neighbours; kick as is
            # Calibrated steps from standstill (measured, three trials each):
            # walk 0.6 s -> 91 to 94 mm, creep 1.0 s -> 57 to 64 mm, creep
            # 0.7 s with a 0.5 turn command -> 44 mm and 2 degrees (a plain
            # 0.8 s creep turns the duck 10 degrees), creep 0.6 s -> 11 to
            # 15 mm with a 17 mm sideways shift. Shorter creeps do not get
            # the gait going.
            if err[0] > 0.075:
                t = self.burst((Navigator.WALK, 0.0, 0.0), 0.6, t)
            elif err[0] > 0.052:
                t = self.burst((Navigator.CREEP, 0.0, 0.0), 1.0, t)
            elif err[0] > 0.030:
                t = self.burst((Navigator.CREEP, 0.0, 0.5), 0.7, t)
            elif err[0] > 0.008:
                t = self.burst((Navigator.CREEP, 0.0, 0.0), 0.6, t)
            else:
                break
        return t, err

    def go_via(self, point, t, tolerance=0.08, limit=40.0, keep_off=()):
        """Walk to a point, heading free. Never sets off on a stale pose: a
        kick lurches the body with the head neutral and no post in view. If
        the estimate drifts into a remembered piece's reach, stop and look
        rather than walk through it."""
        if self.dr.pose is None or self.since_fix > 0.5:
            t = self.relocalize(t)
        # A piece the duck already stands beside cannot be avoided, only left.
        if self.dr.pose is not None:
            keep_off = [p for p in keep_off
                        if math.hypot(p[0] - self.dr.pose[0], p[1] - self.dr.pose[1]) > 0.16]
        t0 = t
        while t - t0 < limit:
            pose = self.dr.pose
            if pose is None or self.since_fix > 1.0:
                t = self.relocalize(t)
                continue
            def ahead(p):
                d = math.hypot(p[0] - pose[0], p[1] - pose[1])
                b = Navigator.wrap(math.atan2(p[1] - pose[1], p[0] - pose[0]) - pose[2])
                return d < 0.13 and abs(b) < 1.0
            if any(ahead(p) for p in keep_off):
                t = self.turn_burst(1.0, t)                 # a piece in the way: turn off it
                continue
            dist = math.hypot(point[0] - pose[0], point[1] - pose[1])
            if dist < tolerance:
                break
            bearing = Navigator.wrap(math.atan2(point[1] - pose[1], point[0] - pose[0]) - pose[2])
            if abs(bearing) > 0.6:
                # Turn on the spot in bursts (about 35 degrees each, then a
                # fresh fix) rather than sweep an arc through the pieces.
                t = self.turn_burst(float(np.sign(bearing)), t)
                continue
            command = (Navigator.WALK, 0.0, float(np.clip(2.0 * bearing, -0.5, 0.5)))
            self.tick(command, t)
            t += self.duck.dt
        else:
            pose = self.dr.pose
            print(f"go_via: gave up after {limit:.0f} s, "
                  f"{math.hypot(point[0] - pose[0], point[1] - pose[1]) * 1000:.0f} mm short of the waypoint")
        return t

    def retreat(self, point, t, face=None):
        """Leave the board on dead reckoning. Walking away from the board
        there is no post in view, so looking for one only stalls; the fix
        taken before turning is good to a few millimetres, and the walk is
        short. At the point, turn to face the board (or `face`) and take
        fresh fixes."""
        if self.dr.pose is None or self.since_fix > 0.5:
            t = self.relocalize(t)
        for _ in range(10):
            pose = self.dr.pose
            bearing = Navigator.wrap(math.atan2(point[1] - pose[1], point[0] - pose[0]) - pose[2])
            if abs(bearing) < 0.35:
                break
            t = self.turn_burst(float(np.sign(bearing)), t, 1.0 if abs(bearing) > 0.6 else 0.5)
        t0 = t
        while t - t0 < 12.0:
            pose = self.dr.pose
            dist = math.hypot(point[0] - pose[0], point[1] - pose[1])
            if dist < 0.06:
                break
            bearing = Navigator.wrap(math.atan2(point[1] - pose[1], point[0] - pose[0]) - pose[2])
            self.tick((Navigator.WALK, 0.0, float(np.clip(2.0 * bearing, -0.5, 0.5))), t)
            t += self.duck.dt
        for _ in range(int(0.5 / self.duck.dt)):
            self.tick((0.0, 0.0, 0.0), t)
            t += self.duck.dt
        if face is None:
            face = (BOARD_X, 0.0)
        for _ in range(10):
            pose = self.dr.pose
            bearing = Navigator.wrap(math.atan2(face[1] - pose[1], face[0] - pose[0]) - pose[2])
            if abs(bearing) < 0.3:
                break
            t = self.turn_burst(float(np.sign(bearing)), t, 1.0 if abs(bearing) > 0.6 else 0.5)
        t = self.relocalize(t)
        self.settle_fix(t)
        return t

    def go_to(self, target, t0=0.0, limit=60.0, via=()):
        """Optional waypoints first (a route that stays off the pieces), then
        a coarse approach to a point behind the target, then align the heading
        with the posts. The piece servo takes it from there."""
        t = t0
        for point in via:
            t = self.go_via(point, t)
        nav = Navigator(target, standoff=0.30)
        while nav.phase == "approach" and t - t0 < limit:
            pose = self.dr.pose
            if pose is None or self.since_fix > 2.0:
                t = self.relocalize(t)
                continue
            self.tick(nav.command(pose), t)
            t += self.duck.dt
        self.settle_fix(t)
        t = self.align_yaw(target[2], t)
        return t


# ---- The robot ------------------------------------------------------------------------
class Duck:
    def __init__(self, model, data):
        self.model, self.data = model, data
        self.policy = PolicyInference(
            model, data,
            walking_onnx_path=os.path.join(POLICIES, "alpha_walking.onnx"),
            standing_onnx_path=os.path.join(POLICIES, "alpha_stand.onnx"),
            kick_left_onnx_path=os.path.join(POLICIES, "ball_kick_left.onnx"),
            kick_right_onnx_path=os.path.join(POLICIES, "ball_kick_right.onnx"),
            ground_pick_onnx_path=os.path.join(POLICIES, "alpha_ground_pick.onnx"),
            new_cmd_obs=True, use_projected_gravity=True, kick_duration=2.0)
        self.policy._place_ball = lambda behavior: None
        self.base_scale = self.policy.action_scale
        self.dt = 4 * model.opt.timestep
        self.trunk = self.policy._trunk_qpos_adr

    def place(self, xy, yaw):
        t = self.trunk
        self.data.qpos[t:t + 3] = [xy[0], xy[1], 0.125]
        self.data.qpos[t + 3:t + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        self.data.qvel[:] = 0
        for i, qi in enumerate(self.policy.joint_qpos_indices):
            self.data.qpos[qi] = self.policy.default_pose[i]
        self.data.ctrl[:] = self.policy.default_pose
        mujoco.mj_forward(self.model, self.data)

    def true_pose(self):
        """Simulator ground truth, for scoring only."""
        t = self.trunk
        qw, qx, qy, qz = self.data.qpos[t + 3:t + 7]
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        return np.array([self.data.qpos[t], self.data.qpos[t + 1], yaw])

    def step(self, command=(0.0, 0.0, 0.0), head_yaw=0.0, head_pitch=LOOK_PITCH):
        p = self.policy
        p.update_behavior(self.dt)
        p.update_ground_pick_phase(self.dt)
        kicking = p.behavior_mode == "kick_right"
        p.action_scale = self.base_scale * KICK_SCALE if kicking else self.base_scale
        p.set_vel_cmd(*command)
        # The walk tolerates a turned head; the kick does not (measured: 0 mm
        # with the head turned, one square with it neutral).
        p.head_offset[:] = [0.0, 0.0 if kicking else head_pitch,
                            0.0 if kicking else float(np.clip(head_yaw, -0.7, 0.7)), 0.0]
        p.apply_action(p.infer())
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)

    def kick(self):
        self.policy.trigger_behavior("kick_right")

    @property
    def busy(self):
        return self.policy.behavior_mode is not None
