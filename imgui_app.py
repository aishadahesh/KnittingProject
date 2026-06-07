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
import threading
import numpy as np
from PIL import Image
from scipy.interpolate import CubicSpline

import glfw
from imgui_bundle import imgui, imguizmo
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer
import moderngl

import mitsuba as mi
from knitting_core import (
    CONFIG,
    INITIAL_PARAMS,
    PARAM_INDEX,
    PARAM_NAMES,
    PARAM_RANGES,
    PROJECT_ROOT,
    REFERENCE_IMAGE_PATH,
    build_parametric_control_rows,
    build_display_meshes,
    compute_knitting_vertices, compute_knitting_faces,
    get_loop_color, save_combined_obj, KnittingOptimizer, run_optimization_loop,
)
import jax.numpy as jnp

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
uniform vec3 color;
out vec4 f_color;
void main() {
    vec3  L    = normalize(vec3(0.5, 1.0, 0.8));
    float diff = clamp(dot(normalize(v_norm), L), 0.0, 1.0);
    f_color = vec4(color * (0.25 + 0.75 * diff), 1.0);
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
        return perspective(np.radians(45), w / max(h, 1))

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

    def set_meshes(self, verts_list, faces_list, row_indices=None):
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
            color = get_loop_color(row_indices[i] if row_indices is not None else i, 0)
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

    def render(self, mvp, mv, hover_idx=-1, selected_idx=-1):
        self.fbo.use()
        self.ctx.viewport = (0, 0, self.vp_w, self.vp_h)
        self.ctx.clear(0.12, 0.12, 0.12, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)

        for vao, n_idx, color in self.meshes:
            self.prog['mvp'].write(mvp.T.tobytes())
            self.prog['mv'].write(mv.T.tobytes())
            self.prog['color'].value = tuple(float(c) for c in color)
            vao.render(moderngl.TRIANGLES)

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
    t     = np.arange(len(ctrl_pts), dtype=float)
    t_out = np.linspace(0, len(ctrl_pts) - 1, n_out)
    return np.column_stack(
        [CubicSpline(t, ctrl_pts[:, i])(t_out) for i in range(3)]
    )

class SplineManager:
    def __init__(self, bitmap, config):
        self.bitmap     = bitmap
        self.config     = config
        self.ctrl_rows  = []                          # list of (N,3) arrays
        self.flat_pts   = np.empty((0, 3), np.float32)
        self._row_starts = [0]

    def init_from_params(self, params):
        self.ctrl_rows = build_parametric_control_rows(params, self.bitmap)
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
        bitmap           = np.ones((CONFIG['geometry']['bitmap_rows'],
                                    CONFIG['geometry']['bitmap_loops']), dtype=np.float32),
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
    )

    # ── Initial mesh ──────────────────────────────────────────────────────────
    def rebuild_param_mesh():
        vl = compute_knitting_vertices(s['params'], s['bitmap'])
        fl = compute_knitting_faces(CONFIG['geometry']['segments'], vl)
        display_vl, display_fl, row_indices = build_display_meshes(
            vl,
            fl,
            s['params'],
            s['bitmap'],
            s['display_copies_x'],
            s['display_copies_y'],
        )
        renderer.set_meshes(display_vl, display_fl, row_indices)
        renderer.set_ctrl_pts([])

    def rebuild_spline_mesh():
        vl = spline.build_mesh(s['params'])
        fl = compute_knitting_faces(CONFIG['geometry']['segments'], vl)
        display_vl, display_fl, row_indices = build_display_meshes(
            vl,
            fl,
            s['params'],
            s['bitmap'],
            s['display_copies_x'],
            s['display_copies_y'],
        )
        renderer.set_meshes(display_vl, display_fl, row_indices)
        renderer.set_ctrl_pts(spline.flat_pts)

    rebuild_param_mesh()

    def on_bitmap_change():
        spline.bitmap     = s['bitmap']
        optimizer.bitmap  = jnp.array(s['bitmap'])
        if s['mode'] == 'parameter':
            rebuild_param_mesh()
        else:
            spline.init_from_params(s['params'])
            rebuild_spline_mesh()

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
            scene  = optimizer.get_scene_dict(path + "_combined.obj", s['params'])
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

        imgui.text("Mode")
        imgui.same_line()
        if imgui.radio_button("Param",  s['mode'] == 'parameter') and s['mode'] != 'parameter':
            s['mode'] = 'parameter'
            renderer.set_ctrl_pts([])
            rebuild_param_mesh()
        imgui.same_line()
        if imgui.radio_button("Spline", s['mode'] == 'spline') and s['mode'] != 'spline':
            s['mode'] = 'spline'
            spline.init_from_params(s['params'])
            rebuild_spline_mesh()

        changed_x, new_copies_x = imgui.slider_int("Copies X", s['display_copies_x'], 0, 5)
        changed_y, new_copies_y = imgui.slider_int("Copies Y", s['display_copies_y'], 0, 5)
        if changed_x or changed_y:
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
            lo, hi = PARAM_RANGES[i]
            changed, new_val = imgui.slider_float(
                f"##p{i}", s['params'][i], lo, hi,
                format=f"{name}: %.3f",
            )
            if changed:
                s['params'][i] = new_val
                params_changed = True

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
                "RMB drag: orbit   Scroll: zoom\nLMB click: select point\nDrag gizmo arrows to move",
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
                "RMB drag: orbit\nScroll: zoom   MMB: pan",
            )

        imgui.separator()
        imgui.text("Pattern")
        imgui.same_line()
        if imgui.small_button("Reset##bmap"):
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

        # Render mesh → FBO
        mvp = camera.mvp(disp_w, disp_h)
        mv  = camera.mv(disp_w, disp_h)
        renderer.render(mvp, mv, s['hover_idx'], s['selected_idx'])

        # Display FBO (flip UV Y so OpenGL origin is correct)
        draw_fitted_texture(renderer.texture_id, disp_w, disp_h, avail_x, avail_y, flip_y=True)
        is_hovered   = imgui.is_item_hovered()
        s['mouse_in_vp'] = is_hovered

        # ── ImGuizmo translate gizmo for selected control point ───────────────
        if s['mode'] == 'spline' and s['selected_idx'] >= 0:
            pos = spline.flat_pts[s['selected_idx']].astype(np.float32)
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
                spline.move(s['selected_idx'], obj_m.values[12:15])
                rebuild_spline_mesh()

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
                if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
                    camera.orbit(dx, dy)
                elif glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS:
                    camera.pan(dx, dy)

            # ── Spline handle hover + select (LMB click) ─────────────────────
            if s['mode'] == 'spline' and len(spline.flat_pts) > 0:
                gizmo_active = s['selected_idx'] >= 0 and (
                    imguizmo.im_guizmo.is_using() or imguizmo.im_guizmo.is_over()
                )
                if not gizmo_active:
                    ro, rd = camera.unproject(lx, ly, disp_w, disp_h)
                    # Hover
                    best_t, best_i = np.inf, -1
                    for i, pt in enumerate(spline.flat_pts):
                        t = ray_sphere_hit(ro, rd, pt.astype(float), 0.12)
                        if 0 < t < best_t:
                            best_t, best_i = t, i
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
