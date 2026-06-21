"""imgui_app.py — Knitting Reconstruction GUI (imgui_bundle + moderngl)

Stack:
  - imgui_bundle  : window + Dear ImGui UI (sliders, buttons, layout)
  - moderngl      : mesh rendering via OpenGL FBO → displayed as imgui image
  - scipy         : CubicSpline replaces vtk.vtkCardinalSpline
  - knitting_core : zero changes
"""

# %% PYOPENGL CONFIG FOR WAYLAND/LINUX
import os, sys
if sys.platform.startswith("linux"):
    os.environ["PYOPENGL_PLATFORM"] = "egl"

# %% IMPORTS
from knitting_core import (
    CONFIG,
    INITIAL_PARAMS,
    PARAM_INDEX,
    PARAM_NAMES,
    PARAM_RANGES,
    PROJECT_ROOT,
    REFERENCE_IMAGE_PATH,
    LOOP_HEIGHT_PARAM_INDICES,
    build_parametric_control_rows,
    build_display_meshes,
    compute_knitting_vertices, compute_knitting_faces,
    get_loop_color, save_combined_obj, KnittingOptimizer, run_optimization_loop,
)

import threading
import numpy as np
from PIL import Image
from scipy.interpolate import CubicSpline

import glfw
from imgui_bundle import imgui, imguizmo
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer
import moderngl

import mitsuba as mi
import jax.numpy as jnp

import tkinter as tk
from tkinter import filedialog as _filedialog
import json as _json

# %% CONFIG SHORTCUTS
PARAM_INIT   = list(INITIAL_PARAMS)
BITMAP_NP    = np.ones((CONFIG['geometry']['bitmap_rows'],
                        CONFIG['geometry']['bitmap_loops']))
BITMAP_JNP   = jnp.array(BITMAP_NP)

# %% GLSL SHADERS ─────────────────────────────────────────────────────────────

MESH_VERT = """
#version 330
in vec3 in_pos;
in vec3 in_norm;
uniform mat4 mvp;
uniform mat4 mv;
out vec3 v_norm;
void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
    v_norm = normalize(mat3(mv) * in_norm);
}
"""

MESH_FRAG = """
#version 330
in  vec3 v_norm;
uniform vec3  color;
uniform float model_alpha;
out vec4 f_color;
void main() {
    vec3  L    = normalize(vec3(0.5, 1.0, 0.8));
    float diff = clamp(dot(normalize(v_norm), L), 0.0, 1.0);
    f_color = vec4(color * (0.25 + 0.75 * diff), model_alpha);
}
"""

PT_VERT = """
#version 330
in  vec3 in_pos;
uniform mat4 mvp;
uniform int  hover_idx;
uniform int  selected_idx;
flat out int state;   // 0=normal 1=hover 2=selected
void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
    int vid = gl_VertexID;
    state = (vid == selected_idx) ? 2 : (vid == hover_idx ? 1 : 0);
    gl_PointSize = (state > 0) ? 16.0 : 10.0;
}
"""

PT_FRAG = """
#version 330
flat in int state;
out vec4 f_color;
void main() {
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    if (dot(c, c) > 1.0) discard;
    if      (state == 2) f_color = vec4(1.0, 1.0, 0.0, 1.0);  // drag  → yellow
    else if (state == 1) f_color = vec4(1.0, 0.5, 0.0, 1.0);  // hover → orange
    else                 f_color = vec4(1.0, 1.0, 1.0, 0.9);  // normal → white
}
"""

BG_VERT = """
#version 330
in vec2 in_pos;
out vec2 v_uv;
uniform float bg_scale_x;
uniform float bg_scale_y;
uniform float bg_rotation;
uniform float bg_offset_x;
uniform float bg_offset_y;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    vec2 c = v_uv - 0.5;
    float cr = cos(bg_rotation);
    float sr = sin(bg_rotation);
    vec2 rot = vec2(cr * c.x - sr * c.y, sr * c.x + cr * c.y);
    v_uv = vec2(rot.x / max(bg_scale_x, 0.01) - bg_offset_x,
                rot.y / max(bg_scale_y, 0.01) - bg_offset_y) + 0.5;
    gl_Position = vec4(in_pos, 0.9999, 1.0);
}
"""

BG_FRAG = """
#version 330
in vec2 v_uv;
uniform sampler2D bg_tex;
uniform float bg_alpha;
out vec4 f_color;
void main() {
    // clamp so areas outside the image show transparent (viewport crops naturally)
    vec2 clamped = clamp(v_uv, 0.0, 1.0);
    if (distance(clamped, v_uv) > 0.001) discard;
    vec4 col = texture(bg_tex, clamped);
    f_color = vec4(col.rgb, bg_alpha);
}
"""

# %% MATH HELPERS ─────────────────────────────────────────────────────────────

def look_at(eye, center, up=(0, 1, 0)):
    f = np.asarray(center, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.array([
        [r[0], r[1], r[2], -np.dot(r, eye)],
        [u[0], u[1], u[2], -np.dot(u, eye)],
        [-f[0],-f[1],-f[2],  np.dot(f, eye)],
        [0.,   0.,   0.,    1.             ],
    ], dtype=np.float32)

def perspective(fov_rad, aspect, near=0.01, far=500.0):
    f = 1.0 / np.tan(fov_rad / 2)
    return np.array([
        [f/aspect, 0,  0,                           0                          ],
        [0,        f,  0,                           0                          ],
        [0,        0,  (far+near)/(near-far),       2*far*near/(near-far)      ],
        [0,        0, -1,                           0                          ],
    ], dtype=np.float32)

def compute_normals(verts, tris):
    n = np.zeros_like(verts)
    e1 = verts[tris[:,1]] - verts[tris[:,0]]
    e2 = verts[tris[:,2]] - verts[tris[:,0]]
    fn = np.cross(e1, e2)
    np.add.at(n, tris[:,0], fn)
    np.add.at(n, tris[:,1], fn)
    np.add.at(n, tris[:,2], fn)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-8)

def rotation_matrix_xyz(rx, ry, rz):
    """4×4 rotation matrix from XYZ Euler angles (radians), applied as Rz @ Ry @ Rx."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1,0,0,0],[0,cx,-sx,0],[0,sx,cx,0],[0,0,0,1]], dtype=np.float32)
    Ry = np.array([[cy,0,sy,0],[0,1,0,0],[-sy,0,cy,0],[0,0,0,1]], dtype=np.float32)
    Rz = np.array([[cz,-sz,0,0],[sz,cz,0,0],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
    return Rz @ Ry @ Rx

def ray_sphere_hit(ro, rd, center, radius):
    """Returns t of nearest intersection, or np.inf on miss."""
    oc = ro - center
    b  = np.dot(rd, oc)
    c  = np.dot(oc, oc) - radius**2
    disc = b*b - c
    if disc < 0: return np.inf
    return -b - np.sqrt(disc)

def ray_plane_hit(ro, rd, plane_pt, plane_n):
    """Returns intersection point or None."""
    denom = np.dot(rd, plane_n)
    if abs(denom) < 1e-7: return None
    t = np.dot(plane_pt - ro, plane_n) / denom
    return (ro + t * rd) if t > 0 else None

# %% CAMERA ───────────────────────────────────────────────────────────────────

class Camera:
    def __init__(self):
        self.az     = 0.5
        self.el     = 0.4
        self.dist   = 8.0
        self.initial_dist = self.dist
        self.fov_deg = 45.0
        self.target = np.array([2.0, 2.0, 0.0], float)
        self._eye   = np.zeros(3, float)

    def _pos(self):
        x = self.dist * np.cos(self.el) * np.sin(self.az)
        y = self.dist * np.sin(self.el)
        z = self.dist * np.cos(self.el) * np.cos(self.az)
        return self.target + np.array([x, y, z])

    def view(self):
        self._eye = self._pos()
        return look_at(self._eye, self.target)

    def proj(self, w, h):
        return perspective(np.radians(self.fov_deg), w / max(h, 1))

    def mvp(self, w, h):
        return (self.proj(w, h) @ self.view()).astype(np.float32)

    def mv(self, w, h):
        return self.view().astype(np.float32)

    def unproject(self, px, py, vp_w, vp_h):
        """Cast a ray from camera through pixel (px, py) in viewport space."""
        # Refresh self._eye
        self.view()
        inv = np.linalg.inv(self.proj(vp_w, vp_h) @ self.view())
        ndc = np.array([2*px/vp_w - 1, 1 - 2*py/vp_h])
        def pt(z):
            p = inv @ np.array([*ndc, z, 1.0])
            return (p[:3] / p[3]).astype(float)
        d = pt(1) - pt(-1)
        return self._eye.copy(), d / np.linalg.norm(d)

    def orbit(self, dx, dy):
        self.az -= dx * 0.01
        self.el  = np.clip(self.el + dy * 0.01, -1.4, 1.4)

    def zoom(self, delta):
        self.dist = max(0.3, self.dist * (1 - delta * 0.1))

    def zoom_factor(self):
        return self.initial_dist / max(self.dist, 1e-6)

    def pan(self, dx, dy):
        v     = self.view()
        right = v[0, :3]
        up    = v[1, :3]
        s     = self.dist * 0.0015
        self.target -= right * dx * s - up * dy * s

# %% MESH RENDERER ────────────────────────────────────────────────────────────

class MeshRenderer:
    def __init__(self, ctx, vp_w, vp_h):
        self.ctx     = ctx
        self.prog    = ctx.program(vertex_shader=MESH_VERT, fragment_shader=MESH_FRAG)
        self.pt_prog = ctx.program(vertex_shader=PT_VERT,   fragment_shader=PT_FRAG)
        self.bg_prog = ctx.program(vertex_shader=BG_VERT,   fragment_shader=BG_FRAG)
        # Full-screen quad for background (triangle strip)
        quad = np.array([[-1,-1],[1,-1],[-1,1],[1,1]], dtype=np.float32)
        self.bg_vao = ctx.vertex_array(
            self.bg_prog,
            [(ctx.buffer(quad.tobytes()), '2f', 'in_pos')],
        )
        self.vp_w    = 1
        self.vp_h    = 1
        self.color_tex = None
        self.depth_rb = None
        self.fbo = None
        self.meshes = []      # list of (vao, n_indices, color)
        self.pt_vao = None
        self.n_pts  = 0
        self.resize(vp_w, vp_h)

    @property
    def texture_id(self):
        return self.fbo.color_attachments[0].glo

    def resize(self, vp_w, vp_h):
        vp_w = max(1, int(vp_w))
        vp_h = max(1, int(vp_h))
        if self.fbo is not None and vp_w == self.vp_w and vp_h == self.vp_h:
            return

        if self.fbo is not None:
            self.fbo.release()
            self.color_tex.release()
            self.depth_rb.release()

        self.vp_w = vp_w
        self.vp_h = vp_h
        self.color_tex = self.ctx.texture((vp_w, vp_h), 4)
        self.color_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.depth_rb = self.ctx.depth_renderbuffer((vp_w, vp_h))
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.color_tex],
            depth_attachment=self.depth_rb,
        )

    def set_meshes(self, verts_list, faces_list, row_indices=None, colors=None, meta=None):
        for vao, _, _ in self.meshes:
            vao.release()
        self.meshes.clear()
        for i, ((verts, _), faces) in enumerate(zip(verts_list, faces_list)):
            v  = np.array(verts, dtype=np.float32)
            f  = np.array(faces, dtype=np.int32)
            # quads → triangles
            tris = np.empty((len(f) * 2, 3), dtype=np.int32)
            tris[0::2] = f[:, [0, 1, 2]]
            tris[1::2] = f[:, [0, 2, 3]]
            nm = compute_normals(v, tris).astype(np.float32)
            vao = self.ctx.vertex_array(self.prog, [
                (self.ctx.buffer(v.tobytes()),  '3f', 'in_pos'),
                (self.ctx.buffer(nm.tobytes()), '3f', 'in_norm'),
            ], self.ctx.buffer(tris.astype(np.int32).tobytes()))
            if meta is not None:
                row_idx = meta[i].get('row', i)
            else:
                row_idx = row_indices[i] if row_indices is not None else i
            if colors is not None:
                color = colors[row_idx % len(colors)]
            else:
                color = get_loop_color(row_idx, 0)
            self.meshes.append((vao, len(tris) * 3, color))

    def set_ctrl_pts(self, flat_pts):
        if self.pt_vao:
            self.pt_vao.release()
            self.pt_vao = None
        self.n_pts = len(flat_pts)
        if self.n_pts == 0:
            return
        pts = np.asarray(flat_pts, dtype=np.float32)
        vbo = self.ctx.buffer(pts.tobytes())
        self.pt_vao = self.ctx.vertex_array(self.pt_prog, [(vbo, '3f', 'in_pos')])

    def render(self, mvp, mv, hover_idx=-1, selected_idx=-1,
               bg_tex=None, bg_alpha=0.5, bg_scale_x=1.0, bg_scale_y=1.0,
               bg_rotation=0.0, bg_offset_x=0.0, bg_offset_y=0.0,
               model_alpha=1.0):
        self.fbo.use()
        self.ctx.viewport = (0, 0, self.vp_w, self.vp_h)
        self.ctx.clear(0.12, 0.12, 0.12, 1.0)

        # ── Draw reference-image background quad ──────────────────────────────
        if bg_tex is not None:
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.ctx.disable(moderngl.CULL_FACE)
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            bg_tex.use(0)
            self.bg_prog['bg_tex'].value     = 0
            self.bg_prog['bg_alpha'].value    = float(bg_alpha)
            self.bg_prog['bg_scale_x'].value  = float(bg_scale_x)
            self.bg_prog['bg_scale_y'].value  = float(bg_scale_y)
            self.bg_prog['bg_rotation'].value = float(bg_rotation)
            self.bg_prog['bg_offset_x'].value = float(bg_offset_x)
            self.bg_prog['bg_offset_y'].value = float(bg_offset_y)
            self.bg_vao.render(moderngl.TRIANGLE_STRIP)
            self.ctx.disable(moderngl.BLEND)

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)

        # Enable blending for transparent model
        use_model_blend = model_alpha < 0.9999
        if use_model_blend:
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        for vao, n_idx, color in self.meshes:
            self.prog['mvp'].write(mvp.T.tobytes())
            self.prog['mv'].write(mv.T.tobytes())
            self.prog['color'].value = tuple(float(c) for c in color)
            self.prog['model_alpha'].value = float(model_alpha)
            vao.render(moderngl.TRIANGLES)

        if use_model_blend:
            self.ctx.disable(moderngl.BLEND)

        if self.pt_vao:
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.ctx.disable(moderngl.CULL_FACE)
            self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
            self.pt_prog['mvp'].write(mvp.T.tobytes())
            self.pt_prog['hover_idx'].value    = hover_idx
            self.pt_prog['selected_idx'].value = selected_idx
            self.pt_vao.render(moderngl.POINTS, vertices=self.n_pts)

        self.ctx.screen.use()

# %% SPLINE MANAGER ───────────────────────────────────────────────────────────

def _interp_spline(ctrl_pts, n_out):
    ctrl_pts = np.asarray(ctrl_pts, dtype=float)
    if len(ctrl_pts) <= 1:
        return np.repeat(ctrl_pts, n_out, axis=0)

    seg_len = np.linalg.norm(np.diff(ctrl_pts, axis=0), axis=1)
    t = np.concatenate(([0.0], np.cumsum(np.maximum(seg_len, 1e-6))))
    t_out = np.linspace(t[0], t[-1], n_out)
    if len(ctrl_pts) == 2:
        return np.column_stack([
            np.interp(t_out, t, ctrl_pts[:, i]) for i in range(3)
        ])
    return np.column_stack([
        CubicSpline(t, ctrl_pts[:, i], bc_type="natural")(t_out)
        for i in range(3)
    ])

class SplineManager:
    def __init__(self, bitmap, config, samples_per_loop=5):
        self.bitmap          = bitmap
        self.config          = config
        self.samples_per_loop = samples_per_loop
        self.ctrl_rows  = []                          # list of (N,3) arrays
        self.flat_pts   = np.empty((0, 3), np.float32)
        self._row_starts = [0]

    def init_from_params(self, params):
        self.ctrl_rows = build_parametric_control_rows(
            params, self.bitmap, self.samples_per_loop)
        self._rebuild()

    def _rebuild(self):
        self._row_starts = [0]
        for row in self.ctrl_rows:
            self._row_starts.append(self._row_starts[-1] + len(row))
        self.flat_pts = (
            np.concatenate(self.ctrl_rows).astype(np.float32)
            if self.ctrl_rows else np.empty((0, 3), np.float32)
        )

    def move(self, flat_idx, pos):
        for r in range(len(self.ctrl_rows)):
            s, e = self._row_starts[r], self._row_starts[r + 1]
            if s <= flat_idx < e:
                self.ctrl_rows[r][flat_idx - s] = pos
                break
        self._rebuild()

    def build_mesh(self, params):
        radius = params[PARAM_INDEX['radius']]
        ratio  = params[PARAM_INDEX['ellipse_ratio']]
        seg    = self.config['geometry']['segments']
        res    = self.config['geometry']['loop_res']
        n_out  = res * self.bitmap.shape[1] + 1
        verts_list = []
        for row in self.ctrl_rows:
            pts = _interp_spline(row, n_out)
            T   = np.gradient(pts, axis=0)
            T  /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-8
            U   = np.cross(T, [0, 0, 1])
            bad = np.linalg.norm(U, axis=1) < 1e-6
            U[bad] = np.cross(T[bad], [1, 0, 0])
            U  /= np.linalg.norm(U, axis=1, keepdims=True) + 1e-8
            V   = np.cross(T, U)
            angles  = np.linspace(0, 2*np.pi, seg, endpoint=False)
            offsets = (U[:,None,:] * np.cos(angles)[None,:,None] * radius * ratio
                     + V[:,None,:] * np.sin(angles)[None,:,None] * radius)
            verts_list.append(((pts[:,None,:] + offsets).reshape(-1, 3), n_out))
        return verts_list

# %% IMAGE TEXTURE HELPER ─────────────────────────────────────────────────────

def pil_to_texture(ctx, pil_img):
    """Upload a PIL image as a moderngl texture (RGBA, Y-flipped for GL)."""
    img  = pil_img.convert("RGBA").transpose(Image.FLIP_TOP_BOTTOM)
    w, h = img.size
    tex  = ctx.texture((w, h), 4, img.tobytes())
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    return tex


def draw_fitted_texture(texture_id, tex_w, tex_h, avail_w, avail_h, flip_y=False):
    """Draws a texture centered inside the available region while preserving aspect."""
    if tex_w <= 0 or tex_h <= 0 or avail_w <= 1 or avail_h <= 1:
        return

    scale = min(avail_w / tex_w, avail_h / tex_h)
    draw_w = max(1.0, tex_w * scale)
    draw_h = max(1.0, tex_h * scale)
    offset_x = max(0.0, (avail_w - draw_w) * 0.5)
    offset_y = max(0.0, (avail_h - draw_h) * 0.5)
    cursor = imgui.get_cursor_pos()
    imgui.set_cursor_pos((cursor.x + offset_x, cursor.y + offset_y))

    uv0 = imgui.ImVec2(0, 1) if flip_y else imgui.ImVec2(0, 0)
    uv1 = imgui.ImVec2(1, 0) if flip_y else imgui.ImVec2(1, 1)
    imgui.image(imgui.ImTextureRef(texture_id), imgui.ImVec2(draw_w, draw_h), uv0=uv0, uv1=uv1)

# %% MAIN ─────────────────────────────────────────────────────────────────────

def main():
    # ── Reference image ──────────────────────────────────────────────────────
    try:
        ref_pil = Image.open(REFERENCE_IMAGE_PATH).convert("RGB")
    except Exception:
        ref_pil = Image.new("RGB", (256, 256), (60, 40, 40))

    optimizer = KnittingOptimizer(ref_pil, BITMAP_JNP)

    # ── GLFW + OpenGL ─────────────────────────────────────────────────────────
    if not glfw.init():
        raise RuntimeError("GLFW init failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    window = glfw.create_window(1600, 900, "Knitting — imgui", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # ── Dear ImGui ────────────────────────────────────────────────────────────
    imgui.create_context()
    io = imgui.get_io()
    io.config_windows_move_from_title_bar_only = True
    io.config_flags |= imgui.ConfigFlags_.docking_enable
    io.config_flags |= imgui.ConfigFlags_.viewports_enable
    io.set_ini_filename(os.path.join(PROJECT_ROOT, "imgui_layout.ini"))
    impl = GlfwRenderer(window)

    style = imgui.get_style()
    style.window_menu_button_position = imgui.Dir_.none
    if io.config_flags & imgui.ConfigFlags_.viewports_enable:
        style.window_rounding = 0.0
        style.color_(imgui.Col_.window_bg).w = 1.0

    # ── moderngl (shares the existing GL context) ─────────────────────────────
    ctx = moderngl.create_context()

    # ── Scene objects ─────────────────────────────────────────────────────────
    camera   = Camera()
    renderer = MeshRenderer(ctx, 960, 720)
    spline   = SplineManager(BITMAP_NP, CONFIG)

    # ── Mutable app state ─────────────────────────────────────────────────────
    s = dict(
        params           = list(PARAM_INIT),
        bitmap           = np.ones((3, CONFIG['geometry']['bitmap_loops']), dtype=np.float32),
        display_copies_x = 0,
        display_copies_y = 0,
        mode             = 'parameter',     # 'parameter' | 'spline'
        hover_idx        = -1,
        selected_idx     = -1,
        is_rendering     = False,
        is_optimizing    = False,
        render_result    = None,            # PIL image set by bg thread
        pending_tex      = False,           # main thread needs to create texture
        render_tex       = None,            # moderngl Texture
        mouse_in_vp      = False,
        vp_origin        = (0.0, 0.0),      # screen pos of viewport image widget
        vp_scale         = 1.0,
        # Reference-image background overlay
        show_ref_bg      = False,
        ref_bg_alpha     = 0.5,
        ref_bg_scale_x   = 1.0,
        ref_bg_scale_y   = 1.0,
        ref_bg_lock_zoom = True,
        ref_bg_rotation  = 0.0,
        ref_bg_offset_x  = 0.0,
        ref_bg_offset_y  = 0.0,
        # Model opacity
        model_alpha      = 1.0,
        # Yarn / material colors — one per row, cycled from config palette
        single_model_color = [0.85, 0.12, 0.10],
        use_row_colors   = False,
        row_colors       = [
            list(CONFIG['ui']['yarn_colors'][i % len(CONFIG['ui']['yarn_colors'])])
            for i in range(3)
        ],
        row_visible      = np.ones(3, dtype=bool),
        # Mitsuba camera
        mi_cam_dist_mult = float(CONFIG['rendering']['camera_dist_mult']),
        mi_cam_fov       = float(CONFIG['rendering']['camera_fov']),
        view_fov         = 45.0,
        # Bitmap grid dimensions
        bitmap_rows      = 3,
        bitmap_cols      = CONFIG['geometry']['bitmap_loops'],
        # Spline control-point density
        samples_per_loop = 5,
        # Model rotation (Euler angles, radians)
        model_rot_x        = 0.0,
        model_rot_y        = 0.0,
        model_rot_z        = 0.0,
        model_rot_dragging = False,   # LMB held for free model rotation
        # Model translation
        model_tx           = 0.0,
        model_ty           = 0.0,
        model_tz           = 0.0,
        model_drag_mode    = 'rotate',   # 'rotate' | 'translate'
        # Save / load UI
        save_path        = os.path.join(PROJECT_ROOT, 'params.json'),
        load_path        = os.path.join(PROJECT_ROOT, 'params.json'),
        status_msg       = '',
        mesh_center      = np.zeros(3, dtype=np.float32),  # bounding-box center for pivot
        undo_stack       = [],
        max_undo         = 40,
        gizmo_edit_active = False,
        model_drag_undo_active = False,
    )
    camera.fov_deg = s['view_fov']

    def _sync_row_colors(n_rows):
        palette = CONFIG['ui']['yarn_colors']
        while len(s['row_colors']) < n_rows:
            idx = len(s['row_colors'])
            s['row_colors'].append(list(palette[idx % len(palette)]))
        if len(s['row_colors']) > n_rows:
            del s['row_colors'][n_rows:]

    def _sync_row_visibility(n_rows):
        old = np.asarray(s['row_visible'], dtype=bool)
        new_vis = np.ones(n_rows, dtype=bool)
        rows = min(old.shape[0], n_rows)
        new_vis[:rows] = old[:rows]
        s['row_visible'] = new_vis

    def build_display_meshes_precise(verts_list, faces_list, meta):
        if not verts_list:
            return [], [], []
        # Tile by the logical pattern period, not by the mesh bounding box.
        # The geometry can bulge outside its cell, but the repeated object
        # should still meet on the same knitting grid.
        x_period = max(float(s['bitmap_cols']), 1e-6)
        y_period = max(float(s['bitmap_rows']) * abs(float(s['params'][PARAM_INDEX['dy']])), 1e-6)
        display_vl, display_fl, display_meta = [], [], []
        for y_tile in range(-int(s['display_copies_y']), int(s['display_copies_y']) + 1):
            for x_tile in range(-int(s['display_copies_x']), int(s['display_copies_x']) + 1):
                translation = np.array([x_tile * x_period, y_tile * y_period, 0.0], dtype=np.float32)
                for (verts, n_points), faces, part_meta in zip(verts_list, faces_list, meta):
                    display_vl.append((np.asarray(verts, dtype=np.float32) + translation, n_points))
                    display_fl.append(faces)
                    display_meta.append(part_meta)
        return display_vl, display_fl, display_meta

    def prepare_display_meshes(vl, fl):
        row_vl, row_fl, meta = [], [], []
        for row_idx, (verts_item, faces_item) in enumerate(zip(vl, fl)):
            if not s['row_visible'][row_idx]:
                continue
            row_vl.append(verts_item)
            row_fl.append(faces_item)
            meta.append({'row': row_idx})
        return build_display_meshes_precise(row_vl, row_fl, meta)

    def active_colors():
        return s['row_colors'] if s['use_row_colors'] else [s['single_model_color']]

    # ── Initial mesh ──────────────────────────────────────────────────────────
    def rebuild_param_mesh():
        vl = compute_knitting_vertices(s['params'], s['bitmap'])
        fl = compute_knitting_faces(CONFIG['geometry']['segments'], vl)
        display_vl, display_fl, meta = prepare_display_meshes(vl, fl)
        renderer.set_meshes(display_vl, display_fl, colors=active_colors(), meta=meta)
        renderer.set_ctrl_pts([])
        _recompute_center(display_vl)

    def rebuild_spline_mesh():
        vl = spline.build_mesh(s['params'])
        fl = compute_knitting_faces(CONFIG['geometry']['segments'], vl)
        display_vl, display_fl, meta = prepare_display_meshes(vl, fl)
        renderer.set_meshes(display_vl, display_fl, colors=active_colors(), meta=meta)
        renderer.set_ctrl_pts(spline.flat_pts)
        _recompute_center(display_vl)

    def _recompute_center(display_vl):
        """Store the current display mesh center."""
        if not display_vl:
            return
        all_pts = np.concatenate([v for v, _ in display_vl], axis=0)
        s['mesh_center'] = ((all_pts.min(axis=0) + all_pts.max(axis=0)) * 0.5).astype(np.float32)

    def center_model_on_view():
        center = np.asarray(s['mesh_center'], dtype=np.float32)
        target = np.asarray(camera.target, dtype=np.float32)
        s['model_tx'], s['model_ty'], s['model_tz'] = (target - center).astype(float).tolist()

    def current_model_matrix():
        model_rot = rotation_matrix_xyz(
            s['model_rot_x'], s['model_rot_y'], s['model_rot_z'])
        cx, cy, cz = s['mesh_center']
        tneg = np.eye(4, dtype=np.float32)
        tneg[0, 3], tneg[1, 3], tneg[2, 3] = -cx, -cy, -cz
        tpos = np.eye(4, dtype=np.float32)
        tpos[0, 3], tpos[1, 3], tpos[2, 3] = cx, cy, cz
        tuser = np.eye(4, dtype=np.float32)
        tuser[0, 3], tuser[1, 3], tuser[2, 3] = s['model_tx'], s['model_ty'], s['model_tz']
        return (tuser @ tpos @ model_rot @ tneg).astype(np.float32)

    def transform_points(points, matrix):
        pts = np.asarray(points, dtype=np.float32)
        if len(pts) == 0:
            return pts
        homo = np.column_stack((pts, np.ones(len(pts), dtype=np.float32)))
        return (homo @ matrix.T)[:, :3]

    def snapshot_state():
        return {
            'params': list(s['params']),
            'bitmap': np.array(s['bitmap'], dtype=np.float32).copy(),
            'mode': s['mode'],
            'display_copies_x': s['display_copies_x'],
            'display_copies_y': s['display_copies_y'],
            'ref_bg_alpha': s['ref_bg_alpha'],
            'ref_bg_scale_x': s['ref_bg_scale_x'],
            'ref_bg_scale_y': s['ref_bg_scale_y'],
            'ref_bg_lock_zoom': s['ref_bg_lock_zoom'],
            'ref_bg_rotation': s['ref_bg_rotation'],
            'ref_bg_offset_x': s['ref_bg_offset_x'],
            'ref_bg_offset_y': s['ref_bg_offset_y'],
            'model_alpha': s['model_alpha'],
            'single_model_color': list(s['single_model_color']),
            'use_row_colors': s['use_row_colors'],
            'row_colors': [list(c) for c in s['row_colors']],
            'row_visible': np.array(s['row_visible'], dtype=bool).copy(),
            'view_fov': s['view_fov'],
            'camera_dist': camera.dist,
            'camera_az': camera.az,
            'camera_el': camera.el,
            'camera_target': camera.target.copy(),
            'samples_per_loop': s['samples_per_loop'],
            'bitmap_rows': s['bitmap_rows'],
            'bitmap_cols': s['bitmap_cols'],
            'model_rot_x': s['model_rot_x'],
            'model_rot_y': s['model_rot_y'],
            'model_rot_z': s['model_rot_z'],
            'model_tx': s['model_tx'],
            'model_ty': s['model_ty'],
            'model_tz': s['model_tz'],
            'model_drag_mode': s['model_drag_mode'],
            'ctrl_rows': [np.array(row, dtype=np.float32).copy() for row in spline.ctrl_rows],
            'selected_idx': s['selected_idx'],
        }

    def push_undo(label):
        snap = snapshot_state()
        snap['label'] = label
        s['undo_stack'].append(snap)
        if len(s['undo_stack']) > s['max_undo']:
            del s['undo_stack'][0]

    def restore_snapshot(snap):
        s['params'] = list(snap['params'])
        s['bitmap'] = np.array(snap['bitmap'], dtype=np.float32).copy()
        s['mode'] = snap['mode']
        for key in (
            'display_copies_x', 'display_copies_y', 'ref_bg_alpha',
            'ref_bg_scale_x', 'ref_bg_scale_y', 'ref_bg_lock_zoom',
            'ref_bg_rotation', 'ref_bg_offset_x', 'ref_bg_offset_y',
            'model_alpha', 'single_model_color', 'use_row_colors',
            'row_colors', 'view_fov',
            'samples_per_loop', 'bitmap_rows', 'bitmap_cols',
            'model_rot_x', 'model_rot_y', 'model_rot_z',
            'model_tx', 'model_ty', 'model_tz', 'model_drag_mode',
            'selected_idx',
        ):
            s[key] = snap[key]
        s['row_visible'] = np.array(snap['row_visible'], dtype=bool).copy()
        camera.dist = snap['camera_dist']
        camera.az = snap['camera_az']
        camera.el = snap['camera_el']
        camera.target = np.array(snap['camera_target'], dtype=float)
        camera.fov_deg = s['view_fov']
        spline.bitmap = s['bitmap']
        spline.samples_per_loop = s['samples_per_loop']
        spline.ctrl_rows = [np.array(row, dtype=np.float32).copy() for row in snap['ctrl_rows']]
        spline._rebuild()
        optimizer.bitmap = jnp.array(s['bitmap'])
        if s['mode'] == 'parameter':
            renderer.set_ctrl_pts([])
            rebuild_param_mesh()
        else:
            rebuild_spline_mesh()

    def undo_last():
        if not s['undo_stack']:
            return
        restore_snapshot(s['undo_stack'].pop())
        s['status_msg'] = 'Undid last change'

    rebuild_param_mesh()

    def on_bitmap_change():
        spline.bitmap     = s['bitmap']
        optimizer.bitmap  = jnp.array(s['bitmap'])
        if s['mode'] == 'parameter':
            rebuild_param_mesh()
        else:
            spline.init_from_params(s['params'])
            rebuild_spline_mesh()

    def on_bitmap_resize(new_rows, new_cols):
        """Resize the bitmap grid, preserving existing values where possible."""
        max_rows = len(LOOP_HEIGHT_PARAM_INDICES)
        new_rows = max(1, min(int(new_rows), max_rows))
        new_cols = max(1, min(int(new_cols), 16))
        old = s['bitmap']
        old_r, old_c = old.shape
        new_bm = np.ones((new_rows, new_cols), dtype=np.float32)
        cr = min(old_r, new_rows)
        cc = min(old_c, new_cols)
        new_bm[:cr, :cc] = old[:cr, :cc]
        s['bitmap']      = new_bm
        s['bitmap_rows'] = new_rows
        s['bitmap_cols'] = new_cols
        _sync_row_colors(new_rows)
        _sync_row_visibility(new_rows)
        on_bitmap_change()

    def loop_height_span(name):
        prefix = "loop_height_"
        if not name.startswith(prefix):
            return None
        try:
            return int(name[len(prefix):])
        except ValueError:
            return None

    def fit_loop_heights_to_rows():
        dy = float(s['params'][PARAM_INDEX['dy']])
        for span in range(1, s['bitmap_rows'] + 1):
            name = f"loop_height_{span}"
            if name in PARAM_INDEX:
                idx = PARAM_INDEX[name]
                lo, hi = PARAM_RANGES[idx]
                s['params'][idx] = float(np.clip(span * dy, lo, hi))
        on_bitmap_change()

    # ── Save / Load parameters ────────────────────────────────────────────────
    def _pick_file(mode):
        """Opens a native file dialog on the main thread; returns path string or ''."""
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        if mode == 'save':
            path = _filedialog.asksaveasfilename(
                parent=root,
                title='Save parameters',
                defaultextension='.json',
                filetypes=[('JSON files', '*.json'), ('All files', '*.*')],
                initialfile=os.path.basename(s['save_path']),
                initialdir=os.path.dirname(s['save_path']),
            )
        else:
            path = _filedialog.askopenfilename(
                parent=root,
                title='Load parameters',
                filetypes=[('JSON files', '*.json'), ('All files', '*.*')],
                initialdir=os.path.dirname(s['load_path']),
            )
        root.destroy()
        return path or ''

    def do_save_params():
        path = _pick_file('save')
        if not path:
            return
        data = {
            'params':           {PARAM_NAMES[i]: s['params'][i] for i in range(len(PARAM_NAMES))},
            'bitmap':           s['bitmap'].tolist(),
            'single_model_color': s['single_model_color'],
            'use_row_colors':   s['use_row_colors'],
            'row_colors':       s['row_colors'],
            'row_visible':      s['row_visible'].astype(int).tolist(),
            'mi_cam_dist_mult': s['mi_cam_dist_mult'],
            'mi_cam_fov':       s['mi_cam_fov'],
            'view_fov':         s['view_fov'],
            'samples_per_loop': s['samples_per_loop'],
            'display_copies_x': s['display_copies_x'],
            'display_copies_y': s['display_copies_y'],
            'ref_bg_alpha':     s['ref_bg_alpha'],
            'ref_bg_scale_x':   s['ref_bg_scale_x'],
            'ref_bg_scale_y':   s['ref_bg_scale_y'],
            'ref_bg_lock_zoom': s['ref_bg_lock_zoom'],
            'ref_bg_offset_x':  s['ref_bg_offset_x'],
            'ref_bg_offset_y':  s['ref_bg_offset_y'],
            'model_alpha':      s['model_alpha'],
            'ref_bg_rotation':  s['ref_bg_rotation'],
            'model_rot_x':      s['model_rot_x'],
            'model_rot_y':      s['model_rot_y'],
            'model_rot_z':      s['model_rot_z'],
            'model_tx':         s['model_tx'],
            'model_ty':         s['model_ty'],
            'model_tz':         s['model_tz'],
            'model_drag_mode':  s['model_drag_mode'],
        }
        try:
            with open(path, 'w') as f:
                _json.dump(data, f, indent=2)
            s['save_path']  = path
            s['status_msg'] = f'Saved → {os.path.basename(path)}'
        except Exception as e:
            s['status_msg'] = f'Save error: {e}'

    def do_load_params():
        path = _pick_file('load')
        if not path:
            return
        try:
            with open(path, 'r') as f:
                data = _json.load(f)
            # params
            p_dict = data.get('params', {})
            for i, name in enumerate(PARAM_NAMES):
                if name in p_dict:
                    lo, hi = PARAM_RANGES[i]
                    s['params'][i] = float(np.clip(p_dict[name], lo, hi))
            # bitmap
            if 'bitmap' in data:
                bm = np.array(data['bitmap'], dtype=np.float32)
                s['bitmap_rows'] = bm.shape[0]
                s['bitmap_cols'] = bm.shape[1]
                s['bitmap']      = bm
            # optional fields
            for key in ('single_model_color', 'use_row_colors', 'row_colors',
                        'mi_cam_dist_mult', 'mi_cam_fov',
                        'samples_per_loop', 'display_copies_x', 'display_copies_y',
                        'ref_bg_alpha', 'ref_bg_scale_x', 'ref_bg_scale_y',
                        'ref_bg_lock_zoom', 'ref_bg_rotation',
                        'ref_bg_offset_x', 'ref_bg_offset_y', 'model_alpha',
                        'view_fov', 'model_rot_x', 'model_rot_y', 'model_rot_z',
                        'model_tx', 'model_ty', 'model_tz', 'model_drag_mode'):
                if key in data:
                    s[key] = data[key]
            if 'row_visible' in data:
                s['row_visible'] = np.array(data['row_visible'], dtype=bool)
            elif 'loop_visible' in data:
                old_loop_visible = np.array(data['loop_visible'], dtype=bool)
                s['row_visible'] = np.any(old_loop_visible, axis=1)
            _sync_row_colors(s['bitmap_rows'])
            _sync_row_visibility(s['bitmap_rows'])
            camera.fov_deg = s['view_fov']
            s['load_path']  = path
            s['status_msg'] = f'Loaded ← {os.path.basename(path)}'
            spline.samples_per_loop = s['samples_per_loop']
            on_bitmap_change()
        except Exception as e:
            s['status_msg'] = f'Load error: {e}'

    # ── Static textures ───────────────────────────────────────────────────────
    ref_tex = pil_to_texture(ctx, ref_pil)

    # ── Background worker threads ─────────────────────────────────────────────
    def bg_render():
        try:
            vl = compute_knitting_vertices(s['params'], s['bitmap'])
            fl = compute_knitting_faces(CONFIG['geometry']['segments'], vl)
            mesh_data = [(v, [], f, n) for (v, n), f in zip(vl, fl)]
            path = os.path.join(CONFIG['rendering']['output_dir'], "meshes", "imgui_preview")
            save_combined_obj(mesh_data, path)
            cam_params = (s['mi_cam_dist_mult'], s['mi_cam_fov'])
            scene  = optimizer.get_scene_dict(path + "_combined.obj", s['params'],
                                              camera_params=cam_params)
            img    = mi.render(mi.load_dict(scene), spp=128)
            arr    = (np.clip(np.array(img), 0, 1) * 255).astype(np.uint8)
            s['render_result'] = Image.fromarray(arr)
            s['pending_tex']   = True
        except Exception as e:
            print(f"Render error: {e}")
        finally:
            s['is_rendering'] = False

    def bg_optimize():
        try:
            new_params, _ = run_optimization_loop(optimizer, s['params'])
            s['params'][:] = [float(v) for v in new_params]
            rebuild_param_mesh()
        except Exception as e:
            print(f"Optimize error: {e}")
        finally:
            s['is_optimizing'] = False

    # ── Scroll zoom — read from imgui's io each frame instead of glfw cb ─────
    # (GlfwRenderer already pipes glfw scroll → io.mouse_wheel)

    # ── Main loop ─────────────────────────────────────────────────────────────
    prev_mouse  = None

    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()
        imgui.new_frame()
        imguizmo.im_guizmo.begin_frame()
        # Scroll zoom via io (GlfwRenderer populates mouse_wheel each frame)
        if s['mouse_in_vp'] and imgui.get_io().mouse_wheel != 0:
            camera.zoom(imgui.get_io().mouse_wheel)

        # Upload pending render texture (must happen on GL thread)
        if s['pending_tex'] and s['render_result'] is not None:
            if s['render_tex']:
                s['render_tex'].release()
            s['render_tex']  = pil_to_texture(ctx, s['render_result'])
            s['pending_tex'] = False

        win_w, win_h = glfw.get_window_size(window)

        viewport = imgui.get_main_viewport()
        imgui.set_next_window_pos(viewport.pos)
        imgui.set_next_window_size(viewport.size)
        imgui.set_next_window_viewport(viewport.id_)
        host_flags = (
            imgui.WindowFlags_.no_docking |
            imgui.WindowFlags_.no_title_bar |
            imgui.WindowFlags_.no_collapse |
            imgui.WindowFlags_.no_resize |
            imgui.WindowFlags_.no_move |
            imgui.WindowFlags_.no_bring_to_front_on_focus |
            imgui.WindowFlags_.no_nav_focus |
            imgui.WindowFlags_.menu_bar
        )
        dockspace_flags = imgui.DockNodeFlags_.passthru_central_node
        imgui.push_style_var(imgui.StyleVar_.window_rounding, 0.0)
        imgui.push_style_var(imgui.StyleVar_.window_border_size, 0.0)
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0.0, 0.0))
        imgui.begin("MainDockSpace", flags=host_flags)
        imgui.pop_style_var(3)
        if imgui.begin_menu_bar():
            if imgui.begin_menu("Window"):
                clicked_reset, _ = imgui.menu_item("Reset Layout")
                if clicked_reset:
                    try:
                        os.remove(os.path.join(PROJECT_ROOT, "imgui_layout.ini"))
                    except FileNotFoundError:
                        pass
                imgui.end_menu()
            imgui.end_menu_bar()
        imgui.dock_space_over_viewport(flags=dockspace_flags)
        imgui.end()

        # ── Sidebar ───────────────────────────────────────────────────────────
        imgui.set_next_window_pos((20, 20), cond=imgui.Cond_.first_use_ever)
        imgui.set_next_window_size((320, 820), cond=imgui.Cond_.first_use_ever)
        imgui.begin("Controls")

        undo_disabled = not s['undo_stack']
        if undo_disabled:
            imgui.begin_disabled()
        if imgui.button("Undo##main", (120, 0)):
            undo_last()
        if undo_disabled:
            imgui.end_disabled()
        imgui.same_line()
        last_undo = s['undo_stack'][-1]['label'] if s['undo_stack'] else "No changes"
        imgui.text_disabled(last_undo)
        imgui.separator()

        imgui.text("Mode")
        imgui.same_line()
        if imgui.radio_button("Param",  s['mode'] == 'parameter') and s['mode'] != 'parameter':
            push_undo("Mode")
            s['mode'] = 'parameter'
            renderer.set_ctrl_pts([])
            rebuild_param_mesh()
        imgui.same_line()
        if imgui.radio_button("Spline", s['mode'] == 'spline') and s['mode'] != 'spline':
            push_undo("Mode")
            s['mode'] = 'spline'
            spline.init_from_params(s['params'])
            rebuild_spline_mesh()

        changed_x, new_copies_x = imgui.slider_int("Copies X", s['display_copies_x'], 0, 5)
        changed_y, new_copies_y = imgui.slider_int("Copies Y", s['display_copies_y'], 0, 5)
        if changed_x or changed_y:
            push_undo("Display copies")
            s['display_copies_x'] = new_copies_x
            s['display_copies_y'] = new_copies_y
            if s['mode'] == 'parameter':
                rebuild_param_mesh()
            else:
                rebuild_spline_mesh()

        imgui.separator()
        imgui.text("Parameters")
        imgui.spacing()

        params_changed = False
        for i, name in enumerate(PARAM_NAMES):
            span = loop_height_span(name)
            if span is not None and span > s['bitmap_rows']:
                continue
            lo, hi = PARAM_RANGES[i]
            changed, new_val = imgui.slider_float(
                f"##p{i}", s['params'][i], lo, hi,
                format=f"{name}: %.3f",
            )
            if imgui.is_item_activated():
                push_undo(name)
            if changed:
                s['params'][i] = new_val
                params_changed = True

        if imgui.small_button("Fit loop heights to rows##fit_loop_heights"):
            push_undo("Loop heights")
            fit_loop_heights_to_rows()
            params_changed = False

        if params_changed and s['mode'] == 'parameter':
            rebuild_param_mesh()

        imgui.separator()
        imgui.spacing()
        avail_controls_w = imgui.get_content_region_avail().x
        btn_w = max(120, (avail_controls_w - imgui.get_style().item_spacing.x) * 0.5)

        is_rendering = s['is_rendering']
        if is_rendering:
            imgui.begin_disabled()
        if imgui.button("Render##btn" if not is_rendering else "Rendering…", (btn_w, 0)):
            s['is_rendering'] = True
            threading.Thread(target=bg_render, daemon=True).start()
        if is_rendering:
            imgui.end_disabled()

        imgui.same_line()

        is_optimizing = s['is_optimizing']
        if is_optimizing:
            imgui.begin_disabled()
        if imgui.button("Optimize##btn" if not is_optimizing else "Running…", (btn_w, 0)):
            s['is_optimizing'] = True
            threading.Thread(target=bg_optimize, daemon=True).start()
        if is_optimizing:
            imgui.end_disabled()

        if s['mode'] == 'spline':
            imgui.spacing()
            imgui.text_colored(
                (0.9, 0.7, 0.3, 1.0),
                "LMB drag: rotate/move model\nRMB drag: orbit   Scroll: zoom\nLMB click: select point\nDrag gizmo arrows to move",
            )
            imgui.text(f"Points: {len(spline.flat_pts)}")
            if s['hover_idx'] >= 0:
                imgui.text(f"Hover: {s['hover_idx']}")
            if s['selected_idx'] >= 0:
                imgui.text(f"Selected: {s['selected_idx']}")
        else:
            imgui.spacing()
            imgui.text_colored(
                (0.6, 0.8, 0.6, 1.0),
                "LMB drag: rotate/move model\nRMB drag: orbit   Scroll: zoom\nMMB: pan",
            )

        # ── Viewport background image ─────────────────────────────────────────
        imgui.separator()
        if imgui.collapsing_header("Viewport Background"):
            _, s['show_ref_bg'] = imgui.checkbox("Show reference overlay", s['show_ref_bg'])
            if s['show_ref_bg']:
                old_lock_zoom = s['ref_bg_lock_zoom']
                changed_lock, new_lock_zoom = imgui.checkbox(
                    "Lock zoom with model##bg", s['ref_bg_lock_zoom'])
                if changed_lock:
                    s['ref_bg_lock_zoom'] = old_lock_zoom
                    push_undo("Background lock")
                    s['ref_bg_lock_zoom'] = new_lock_zoom
                ch, s['ref_bg_alpha'] = imgui.slider_float(
                    "Opacity##bg", s['ref_bg_alpha'], 0.0, 1.0, "%.2f")
                imgui.text("Width ")
                imgui.same_line()
                ch, s['ref_bg_scale_x'] = imgui.drag_float(
                    "##bgsx", s['ref_bg_scale_x'], 0.01, 0.01, 50.0, "W: %.2f")
                imgui.text("Height")
                imgui.same_line()
                ch, s['ref_bg_scale_y'] = imgui.drag_float(
                    "##bgsy", s['ref_bg_scale_y'], 0.01, 0.01, 50.0, "H: %.2f")
                if imgui.small_button("1:1##bg"):
                    s['ref_bg_scale_x'] = s['ref_bg_scale_y'] = 1.0
                imgui.same_line()
                if imgui.small_button("Link W=H##bg"):
                    s['ref_bg_scale_y'] = s['ref_bg_scale_x']
                imgui.text("Pan X  ")
                imgui.same_line()
                ch, s['ref_bg_offset_x'] = imgui.drag_float(
                    "##bgox", s['ref_bg_offset_x'], 0.001, -2.0, 2.0, "X: %.3f")
                imgui.text("Pan Y  ")
                imgui.same_line()
                ch, s['ref_bg_offset_y'] = imgui.drag_float(
                    "##bgoy", s['ref_bg_offset_y'], 0.001, -2.0, 2.0, "Y: %.3f")
                if imgui.small_button("Center##bg"):
                    s['ref_bg_offset_x'] = s['ref_bg_offset_y'] = 0.0
                ch, s['ref_bg_rotation'] = imgui.slider_float(
                    "Rotation##bg", s['ref_bg_rotation'],
                    -float(np.pi), float(np.pi), "%.2f rad")

        # ── Model opacity ─────────────────────────────────────────────────────
        imgui.separator()
        if imgui.collapsing_header("Model Opacity"):
            _, s['model_alpha'] = imgui.slider_float(
                "Opacity##mdl", s['model_alpha'], 0.0, 1.0, "%.2f")

        # ── Yarn colors ───────────────────────────────────────────────────────
        imgui.separator()
        if imgui.collapsing_header("Material"):
            changed_mode, use_row_colors = imgui.checkbox("Control colors per row##rowcolors", s['use_row_colors'])
            if changed_mode:
                push_undo("Color mode")
                s['use_row_colors'] = use_row_colors
                if s['mode'] == 'parameter':
                    rebuild_param_mesh()
                else:
                    rebuild_spline_mesh()

            if not s['use_row_colors']:
                changed_c, new_col = imgui.color_edit3(
                    "One color for all##single_color",
                    (float(s['single_model_color'][0]), float(s['single_model_color'][1]), float(s['single_model_color'][2])),
                )
                if imgui.is_item_activated():
                    push_undo("Single color")
                if changed_c:
                    s['single_model_color'] = list(new_col)
                    if s['mode'] == 'parameter':
                        rebuild_param_mesh()
                    else:
                        rebuild_spline_mesh()
            else:
                colors_changed = False
                for row_idx in range(s['bitmap_rows']):
                    col = s['row_colors'][row_idx]
                    changed_c, new_col = imgui.color_edit3(
                        f"Row {row_idx + 1}##row_color_{row_idx}",
                        (float(col[0]), float(col[1]), float(col[2])),
                    )
                    if imgui.is_item_activated():
                        push_undo("Row color")
                    if changed_c:
                        s['row_colors'][row_idx] = list(new_col)
                        colors_changed = True
                if colors_changed:
                    if s['mode'] == 'parameter':
                        rebuild_param_mesh()
                    else:
                        rebuild_spline_mesh()

        # ── Mitsuba camera ────────────────────────────────────────────────────
        imgui.separator()
        if imgui.collapsing_header("Mitsuba Camera"):
            _, s['mi_cam_dist_mult'] = imgui.slider_float(
                "Dist mult##mi", s['mi_cam_dist_mult'], 0.3, 3.0, "%.2f")
            _, s['mi_cam_fov'] = imgui.slider_float(
                "FoV (deg)##mi", s['mi_cam_fov'], 10.0, 120.0, "%.1f")

        imgui.separator()
        if imgui.collapsing_header("View Camera"):
            changed_view_fov, new_view_fov = imgui.slider_float(
                "FoV (deg)##view", s['view_fov'], 10.0, 120.0, "%.1f")
            if imgui.is_item_activated():
                push_undo("View FoV")
            if changed_view_fov:
                s['view_fov'] = new_view_fov
                camera.fov_deg = new_view_fov

        # ── Model rotation ────────────────────────────────────────────────────
        imgui.separator()
        if imgui.collapsing_header("Model Transform"):
            pi = float(np.pi)
            # ─ drag-mode radio buttons ─────────────────────────────────────
            imgui.text("LMB drag:")
            imgui.same_line()
            if imgui.radio_button("Rotate##dm",    s['model_drag_mode'] == 'rotate'):
                push_undo("Drag mode")
                s['model_drag_mode'] = 'rotate'
            imgui.same_line()
            if imgui.radio_button("Translate##dm", s['model_drag_mode'] == 'translate'):
                push_undo("Drag mode")
                s['model_drag_mode'] = 'translate'
            imgui.spacing()
            # ─ rotation sliders ──────────────────────────────────────────
            imgui.text_colored((0.8, 0.8, 0.4, 1.0), "Rotation")
            _, s['model_rot_x'] = imgui.slider_float(
                "X##mrot", s['model_rot_x'], -pi, pi, "X: %.2f rad")
            if imgui.is_item_activated():
                push_undo("Model rotation")
            _, s['model_rot_y'] = imgui.slider_float(
                "Y##mrot", s['model_rot_y'], -pi, pi, "Y: %.2f rad")
            if imgui.is_item_activated():
                push_undo("Model rotation")
            _, s['model_rot_z'] = imgui.slider_float(
                "Z##mrot", s['model_rot_z'], -pi, pi, "Z: %.2f rad")
            if imgui.is_item_activated():
                push_undo("Model rotation")
            if imgui.small_button("Reset rot##mrot"):
                push_undo("Reset rotation")
                s['model_rot_x'] = s['model_rot_y'] = s['model_rot_z'] = 0.0
            imgui.spacing()
            # ─ translation drag fields ────────────────────────────────
            imgui.text_colored((0.4, 0.8, 0.8, 1.0), "Position")
            _, s['model_tx'] = imgui.drag_float(
                "X##mpos", s['model_tx'], 0.01, -100.0, 100.0, "X: %.3f")
            if imgui.is_item_activated():
                push_undo("Model position")
            _, s['model_ty'] = imgui.drag_float(
                "Y##mpos", s['model_ty'], 0.01, -100.0, 100.0, "Y: %.3f")
            if imgui.is_item_activated():
                push_undo("Model position")
            _, s['model_tz'] = imgui.drag_float(
                "Z##mpos", s['model_tz'], 0.01, -100.0, 100.0, "Z: %.3f")
            if imgui.is_item_activated():
                push_undo("Model position")
            if imgui.small_button("Reset pos##mpos"):
                push_undo("Reset position")
                center_model_on_view()
            imgui.same_line()
            if imgui.small_button("Reset all##mall"):
                push_undo("Reset transform")
                s['model_rot_x'] = s['model_rot_y'] = s['model_rot_z'] = 0.0
                center_model_on_view()

        # ── Bitmap grid size ──────────────────────────────────────────────────
        imgui.separator()
        if imgui.collapsing_header("Bitmap Resolution"):
            max_rows = len(LOOP_HEIGHT_PARAM_INDICES)
            ch_r, new_rows = imgui.slider_int(
                "Rows##bres", s['bitmap_rows'], 1, max_rows)
            ch_c, new_cols = imgui.slider_int(
                "Columns##bres", s['bitmap_cols'], 1, 16)
            if ch_r or ch_c:
                push_undo("Bitmap size")
                on_bitmap_resize(new_rows, new_cols)

        imgui.separator()
        if imgui.collapsing_header("Row Visibility"):
            if imgui.small_button("Show all##rows"):
                push_undo("Show rows")
                s['row_visible'][:] = True
                if s['mode'] == 'parameter':
                    rebuild_param_mesh()
                else:
                    rebuild_spline_mesh()
            imgui.same_line()
            if imgui.small_button("Hide all##rows"):
                push_undo("Hide rows")
                s['row_visible'][:] = False
                if s['mode'] == 'parameter':
                    rebuild_param_mesh()
                else:
                    rebuild_spline_mesh()

            row_changed = False
            for r in range(s['bitmap_rows']):
                visible = bool(s['row_visible'][r])
                changed_row, new_visible = imgui.checkbox(f"Row {r + 1}##row_vis_{r}", visible)
                if changed_row:
                    if not row_changed:
                        push_undo("Row visibility")
                    s['row_visible'][r] = new_visible
                    row_changed = True
            if row_changed:
                if s['mode'] == 'parameter':
                    rebuild_param_mesh()
                else:
                    rebuild_spline_mesh()

        # ── Spline resolution ─────────────────────────────────────────────────
        imgui.separator()
        if imgui.collapsing_header("Spline Resolution"):
            ch_spl, new_spl = imgui.slider_int(
                "Samples/loop##spl", s['samples_per_loop'], 2, 20)
            if ch_spl:
                push_undo("Spline resolution")
                s['samples_per_loop']       = new_spl
                spline.samples_per_loop     = new_spl
                if s['mode'] == 'spline':
                    spline.init_from_params(s['params'])
                    rebuild_spline_mesh()

        # ── Save / Load parameters ────────────────────────────────────────────
        imgui.separator()
        if imgui.collapsing_header("Save / Load Parameters"):
            avail_sl_w = imgui.get_content_region_avail().x
            half_w = max(100, (avail_sl_w - imgui.get_style().item_spacing.x) * 0.5)
            if imgui.button("Save params…", (half_w, 0)):
                do_save_params()
            imgui.same_line()
            if imgui.button("Load params…", (half_w, 0)):
                do_load_params()
            if s['status_msg']:
                imgui.spacing()
                imgui.text_colored((0.4, 0.9, 0.4, 1.0), s['status_msg'])

        # ── Pattern ───────────────────────────────────────────────────────────
        imgui.separator()
        imgui.text("Pattern")
        imgui.same_line()
        if imgui.small_button("Reset##bmap"):
            push_undo("Pattern reset")
            s['bitmap'][:] = 1.0
            on_bitmap_change()

        nr, nc = s['bitmap'].shape
        CELL_W, CELL_H = 22, 16
        grid_w = nc * CELL_W + (nc - 1) * 2
        offset_x = max(0.0, (imgui.get_content_region_avail().x - grid_w) / 2)
        bmap_changed = False
        imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(2, 2))
        for r in range(nr):
            imgui.set_cursor_pos_x(imgui.get_cursor_pos().x + offset_x)
            for c in range(nc):
                val = s['bitmap'][r, c]
                if val > 0:
                    imgui.push_style_color(imgui.Col_.button,         (0.18, 0.62, 0.28, 1.0))
                    imgui.push_style_color(imgui.Col_.button_hovered, (0.28, 0.72, 0.38, 1.0))
                else:
                    imgui.push_style_color(imgui.Col_.button,         (0.22, 0.22, 0.22, 1.0))
                    imgui.push_style_color(imgui.Col_.button_hovered, (0.35, 0.35, 0.35, 1.0))
                if imgui.button(f"##bm_{r}_{c}", imgui.ImVec2(CELL_W, CELL_H)):
                    if not bmap_changed:
                        push_undo("Pattern")
                    s['bitmap'][r, c] = 0.0 if val > 0 else 1.0
                    bmap_changed = True
                imgui.pop_style_color(2)
                if c < nc - 1:
                    imgui.same_line()
        imgui.pop_style_var()
        if bmap_changed:
            on_bitmap_change()

        imgui.end()

        # ── Content panels ────────────────────────────────────────────────────
        # ── 3D viewport panel ─────────────────────────────────────────────────
        imgui.set_next_window_pos((360, 20), cond=imgui.Cond_.first_use_ever)
        imgui.set_next_window_size((840, 820), cond=imgui.Cond_.first_use_ever)
        imgui.begin("3D View", flags=imgui.WindowFlags_.no_scroll_with_mouse)
        imgui.text("3D Viewport")

        avail_x, avail_y = imgui.get_content_region_avail()
        disp_w = max(1, int(avail_x))
        disp_h = max(1, int(avail_y))
        renderer.resize(disp_w, disp_h)
        draw_pos  = imgui.get_cursor_screen_pos()
        s['vp_origin'] = (draw_pos.x, draw_pos.y)
        s['vp_scale']  = 1.0

        # Render mesh → FBO (optionally with background overlay)
        model_mat = current_model_matrix()
        mvp = (camera.mvp(disp_w, disp_h) @ model_mat).astype(np.float32)
        mv  = (camera.mv(disp_w, disp_h)  @ model_mat).astype(np.float32)
        bg_zoom = camera.zoom_factor() if s['ref_bg_lock_zoom'] else 1.0
        renderer.render(
            mvp, mv, s['hover_idx'], s['selected_idx'],
            bg_tex      = ref_tex if s['show_ref_bg'] else None,
            bg_alpha    = s['ref_bg_alpha'],
            bg_scale_x  = s['ref_bg_scale_x'] * bg_zoom,
            bg_scale_y  = s['ref_bg_scale_y'] * bg_zoom,
            bg_rotation = s['ref_bg_rotation'],
            bg_offset_x = s['ref_bg_offset_x'],
            bg_offset_y = s['ref_bg_offset_y'],
            model_alpha = s['model_alpha'],
        )

        # Display FBO (flip UV Y so OpenGL origin is correct)
        draw_fitted_texture(renderer.texture_id, disp_w, disp_h, avail_x, avail_y, flip_y=True)
        is_hovered   = imgui.is_item_hovered()
        s['mouse_in_vp'] = is_hovered

        # ── ImGuizmo translate gizmo for selected control point ───────────────
        if s['mode'] == 'spline' and s['selected_idx'] >= 0:
            local_pos = spline.flat_pts[s['selected_idx']].astype(np.float32)
            pos = transform_points([local_pos], model_mat)[0].astype(np.float32)
            M16 = imguizmo.im_guizmo.Matrix16

            # Build column-major Matrix16 objects via .values buffer
            view_m = M16(); view_m.values[:] = camera.view().T.flatten()
            proj_m = M16(); proj_m.values[:] = camera.proj(disp_w, disp_h).T.flatten()

            mat = np.eye(4, dtype=np.float32)
            mat[0, 3] = pos[0]; mat[1, 3] = pos[1]; mat[2, 3] = pos[2]
            obj_m = M16(); obj_m.values[:] = mat.T.flatten()  # tx,ty,tz → indices 12,13,14

            imguizmo.im_guizmo.set_orthographic(False)
            imguizmo.im_guizmo.set_drawlist()
            imguizmo.im_guizmo.set_rect(draw_pos.x, draw_pos.y, disp_w, disp_h)
            changed = imguizmo.im_guizmo.manipulate(
                view_m, proj_m,
                imguizmo.im_guizmo.OPERATION.translate,
                imguizmo.im_guizmo.MODE.world,
                obj_m,
            )
            if changed:
                if not s['gizmo_edit_active']:
                    push_undo("Spline point")
                    s['gizmo_edit_active'] = True
                new_world = np.array(obj_m.values[12:15], dtype=np.float32)
                new_local = transform_points([new_world], np.linalg.inv(model_mat))[0]
                spline.move(s['selected_idx'], new_local)
                rebuild_spline_mesh()
            elif s['gizmo_edit_active'] and not imguizmo.im_guizmo.is_using():
                s['gizmo_edit_active'] = False

        # ── Viewport mouse interaction ────────────────────────────────────────
        mx, my = imgui.get_mouse_pos()
        # Local coords in viewport image space (y=0 at top of displayed image)
        viewport_scale = max(float(s['vp_scale']), 1e-6)
        lx = (mx - s['vp_origin'][0]) / viewport_scale
        ly = (my - s['vp_origin'][1]) / viewport_scale

        if is_hovered:
            curr = (mx, my)

            # ── Camera orbit / pan (RMB / MMB) ───────────────────────────────
            if prev_mouse:
                dx = mx - prev_mouse[0]
                dy = my - prev_mouse[1]
                lmb = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT)  == glfw.PRESS
                rmb = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
                mmb = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE)== glfw.PRESS

                # Decide if LMB should rotate the model:
                # – always in parameter mode
                # – in spline mode only when gizmo isn't active and no point hovered
                if s['mode'] == 'spline' and s['selected_idx'] >= 0 and (
                        imguizmo.im_guizmo.is_using() or imguizmo.im_guizmo.is_over()):
                    lmb_rotates_model = False
                elif s['mode'] == 'spline' and s['hover_idx'] >= 0:
                    lmb_rotates_model = False
                else:
                    lmb_rotates_model = True

                if lmb and lmb_rotates_model and s['model_rot_dragging']:
                    if not s['model_drag_undo_active']:
                        push_undo("Model drag")
                        s['model_drag_undo_active'] = True
                    sens = 0.005
                    if s['model_drag_mode'] == 'rotate':
                        s['model_rot_y'] += dx * sens
                        s['model_rot_x'] += dy * sens
                    else:
                        # Translate in camera right/up plane
                        view = camera.view()
                        right = view[0, :3]
                        up    = view[1, :3]
                        t_sens = camera.dist * 0.003
                        s['model_tx'] += float(right[0]) * dx * t_sens - float(up[0]) * dy * t_sens
                        s['model_ty'] += float(right[1]) * dx * t_sens - float(up[1]) * dy * t_sens
                        s['model_tz'] += float(right[2]) * dx * t_sens - float(up[2]) * dy * t_sens
                elif rmb:
                    camera.orbit(dx, dy)
                elif mmb:
                    camera.pan(dx, dy)

            # Track LMB press/release for model drag
            if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
                s['model_rot_dragging'] = True
            else:
                s['model_rot_dragging'] = False
                s['model_drag_undo_active'] = False

            # ── Spline handle hover + select (LMB click) ─────────────────────
            if s['mode'] == 'spline' and len(spline.flat_pts) > 0:
                gizmo_active = s['selected_idx'] >= 0 and (
                    imguizmo.im_guizmo.is_using() or imguizmo.im_guizmo.is_over()
                )
                if not gizmo_active:
                    world_pts = transform_points(spline.flat_pts, model_mat)
                    homo = np.column_stack((world_pts, np.ones(len(world_pts), dtype=np.float32)))
                    view_proj = camera.proj(disp_w, disp_h) @ camera.view()
                    clip = homo @ view_proj.T
                    valid = clip[:, 3] > 1e-6
                    ndc = np.zeros((len(world_pts), 3), dtype=np.float32)
                    ndc[valid] = clip[valid, :3] / clip[valid, 3:4]
                    screen = np.column_stack((
                        (ndc[:, 0] * 0.5 + 0.5) * disp_w,
                        (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * disp_h,
                    ))
                    in_view = (
                        valid
                        & (ndc[:, 0] >= -1.0) & (ndc[:, 0] <= 1.0)
                        & (ndc[:, 1] >= -1.0) & (ndc[:, 1] <= 1.0)
                    )
                    d2 = np.sum((screen - np.array([lx, ly], dtype=np.float32)) ** 2, axis=1)
                    d2[~in_view] = np.inf
                    best_i = int(np.argmin(d2))
                    best_i = best_i if d2[best_i] <= 16.0 ** 2 else -1
                    s['hover_idx'] = best_i
                    # Click to select / deselect
                    if imgui.is_mouse_clicked(imgui.MouseButton_.left):
                        s['selected_idx'] = best_i   # -1 = deselect

            prev_mouse = curr

        else:
            prev_mouse = None

        imgui.end()

        # ── Render result panel ───────────────────────────────────────────────
        imgui.set_next_window_pos((1220, 20), cond=imgui.Cond_.first_use_ever)
        imgui.set_next_window_size((420, 360), cond=imgui.Cond_.first_use_ever)
        imgui.begin("Mitsuba Render")
        imgui.text("Mitsuba Render")
        if s['render_tex']:
            avail_x, avail_y = imgui.get_content_region_avail()
            draw_fitted_texture(
                s['render_tex'].glo,
                s['render_tex'].width,
                s['render_tex'].height,
                avail_x,
                avail_y,
            )
        else:
            imgui.spacing()
            imgui.text_disabled("Render output will appear here.")
            imgui.text_disabled("Use the button in the left rail.")
        imgui.end()

        # ── Reference image panel ─────────────────────────────────────────────
        imgui.set_next_window_pos((1220, 400), cond=imgui.Cond_.first_use_ever)
        imgui.set_next_window_size((420, 440), cond=imgui.Cond_.first_use_ever)
        imgui.begin("Reference Image")
        imgui.text("Reference Image")
        avail_x, avail_y = imgui.get_content_region_avail()
        draw_fitted_texture(
            ref_tex.glo,
            ref_tex.width,
            ref_tex.height,
            avail_x,
            avail_y,
        )
        imgui.end()

        # ── Final GL clear + imgui draw ───────────────────────────────────────
        imgui.render()
        fb_w, fb_h = glfw.get_framebuffer_size(window)
        ctx.viewport = (0, 0, fb_w, fb_h)
        ctx.screen.use()
        ctx.clear(0.08, 0.08, 0.08, 1.0)
        impl.render(imgui.get_draw_data())
        if io.config_flags & imgui.ConfigFlags_.viewports_enable:
            backup_window = glfw.get_current_context()
            imgui.update_platform_windows()
            imgui.render_platform_windows_default()
            glfw.make_context_current(backup_window)
        glfw.swap_buffers(window)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    impl.shutdown()
    imgui.destroy_context()
    glfw.terminate()


if __name__ == "__main__":
    main()
