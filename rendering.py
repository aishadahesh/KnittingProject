import numpy as np
import moderngl
from PIL import Image
from imgui_bundle import imgui

# %% GLSL SHADERS ─────────────────────────────────────────────────────────────

MESH_VERT = """
#version 330
in vec3 in_pos;
in vec3 in_norm;
in vec2 in_uv;
uniform mat4 mvp;
uniform mat4 mv;
out vec3 v_norm;
out vec3 v_pos;
out vec3 v_obj_norm;
out vec2 v_uv;
void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
    v_norm = normalize(mat3(mv) * in_norm);
    v_pos = in_pos;
    v_obj_norm = normalize(in_norm);
    v_uv = in_uv;
}
"""

MESH_FRAG = """
#version 330
in  vec3 v_norm;
in  vec3 v_pos;
in  vec3 v_obj_norm;
in  vec2 v_uv;
uniform vec3  color;
uniform vec3  texture_tint;
uniform float texture_ridge_strength;
uniform float texture_ridge_scale;
uniform float texture_fiber_strength;
uniform float texture_fiber_scale;
uniform float texture_noise_strength;
uniform float texture_scale_x;
uniform float texture_scale_y;
uniform float texture_chevron_angle;
uniform float texture_fiber_sharpness;
uniform float texture_groove_darkness;
uniform float texture_center_shadow;
uniform float texture_gloss_strength;
uniform float texture_highlight_width;
uniform float texture_color_band_strength;
uniform float texture_color_band_scale;
uniform float texture_color_band_shift;
uniform float texture_saturation;
uniform float texture_contrast;
uniform float texture_uv_blend;
uniform float texture_fiber_rows;
uniform float texture_fiber_row_width;
uniform float texture_fiber_row_depth;
uniform float texture_fiber_layer_count;
uniform float texture_fiber_phase_jitter;
uniform float texture_fibers_per_row;
uniform float texture_sub_fiber_width;
uniform float texture_sub_fiber_depth;
uniform float texture_micro_fiber_strength;
uniform float texture_micro_fiber_scale;
uniform float texture_twist;
uniform vec3  light_color;
uniform float light_intensity;
uniform float model_alpha;

uniform sampler2D depth_tex;
uniform vec2  viewport_size;
uniform vec2  ao_uv_scale;
uniform float ao_strength;
uniform float ao_radius;

out vec4 f_color;

float hash31(vec3 p) {
    return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453);
}

vec3 ref_palette(float idx) {
    float i = mod(floor(idx), 4.0);
    if (i < 0.5) return vec3(0.08, 0.52, 1.0);
    if (i < 1.5) return vec3(1.0, 0.92, 0.08);
    if (i < 2.5) return vec3(0.95, 0.03, 0.05);
    return vec3(0.08, 0.045, 0.025);
}

vec3 apply_saturation(vec3 c, float saturation) {
    float gray = dot(c, vec3(0.299, 0.587, 0.114));
    return mix(vec3(gray), c, saturation);
}

void main() {
    vec3  L    = normalize(vec3(0.5, 1.0, 0.8));
    float diff = clamp(dot(normalize(v_norm), L), 0.0, 1.0);
    vec2 uv_p = vec2(
        (v_uv.x + texture_twist * (v_uv.y - 0.5)) * texture_scale_x,
        v_uv.y * texture_scale_y
    );
    vec2 pos_p = vec2(v_pos.x * texture_scale_x, v_pos.y * texture_scale_y);
    vec2 p = mix(pos_p, uv_p, texture_uv_blend);
    float ca = cos(texture_chevron_angle);
    float sa = sin(texture_chevron_angle);
    float diag_a = p.x * ca + p.y * sa;
    float diag_b = -p.x * ca + p.y * sa;
    float arm_blend = step(0.0, sin(p.x * texture_ridge_scale * 0.5));
    float fiber_axis = mix(diag_a, diag_b, arm_blend);
    float row_wave = sin((p.x * 0.65 + p.y * 1.15) * texture_ridge_scale);
    float cross_wave = sin((fiber_axis * 2.2 + v_pos.z * 3.7 + v_obj_norm.y * 1.8) * texture_fiber_scale);
    float ridge = mix(1.0, 0.72 + 0.42 * pow(0.5 + 0.5 * row_wave, 2.0), texture_ridge_strength);
    float fiber_shape = pow(abs(cross_wave), max(texture_fiber_sharpness, 0.25)) * sign(cross_wave);
    float fiber = mix(1.0, 0.86 + 0.26 * fiber_shape, texture_fiber_strength);
    fiber *= mix(1.0, 0.86 + 0.22 * sin(fiber_axis * texture_micro_fiber_scale + v_uv.y * 18.0), texture_micro_fiber_strength);
    float noise = mix(1.0, 0.82 + 0.36 * hash31(floor(v_pos * 18.0 + v_obj_norm * 9.0)), texture_noise_strength);
    float groove = 1.0 - texture_groove_darkness * pow(1.0 - abs(row_wave), 2.0);
    float center = 1.0 - texture_center_shadow * pow(1.0 - abs(sin(p.x * texture_ridge_scale)), 4.0);
    float band_idx = floor((p.x + p.y * 0.45) * texture_color_band_scale + texture_color_band_shift);
    vec3 band_color = ref_palette(band_idx);
    vec3 base_color = mix(color, band_color, texture_color_band_strength) * texture_tint;
    base_color = apply_saturation(base_color, texture_saturation);
    base_color = clamp((base_color - 0.5) * texture_contrast + 0.5, 0.0, 1.0);
    vec3 lit = base_color * ridge * fiber * noise * groove * center * light_color * light_intensity;
    float gloss_power = mix(96.0, 8.0, texture_highlight_width);
    float spec = pow(clamp(dot(normalize(v_norm), normalize(vec3(-0.25, 0.35, 1.0))), 0.0, 1.0), gloss_power);
    lit += vec3(spec * texture_gloss_strength) * light_color;
    
    float occlusion = 0.0;
    float current_depth = gl_FragCoord.z;
    vec2 screen_uv = gl_FragCoord.xy / viewport_size;
    for (int i = 0; i < 16; i++) {
        float angle = float(i) * 2.39996;
        float r = sqrt(float(i) + 0.5) / 4.0;
        vec2 offset = vec2(cos(angle), sin(angle)) * r * ao_uv_scale;
        float sampled_depth = texture(depth_tex, screen_uv + offset).r;
        float world_depth_diff = (current_depth - sampled_depth) * 500.0;
        if (world_depth_diff > 0.001 && world_depth_diff < ao_radius) {
            occlusion += (1.0 - world_depth_diff / ao_radius);
        }
    }
    occlusion = occlusion / 16.0;
    float ao_factor = clamp(1.0 - occlusion * ao_strength, 0.0, 1.0);

    f_color = vec4(clamp(lit * (0.25 + 0.75 * diff) * ao_factor, 0.0, 1.0), model_alpha);
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

OUTLINE_VERT = """
#version 330
in vec3 in_pos;
in vec3 in_norm;
uniform mat4 mvp;
uniform float outline_width;
void main() {
    vec3 p = in_pos + normalize(in_norm) * outline_width;
    gl_Position = mvp * vec4(p, 1.0);
}
"""

OUTLINE_FRAG = """
#version 330
uniform vec3 outline_color;
out vec4 f_color;
void main() {
    if (gl_FrontFacing) discard;
    f_color = vec4(outline_color, 1.0);
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
uniform float vp_aspect;   // viewport_w / viewport_h
uniform float img_aspect;  // image_w / image_h
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    vec2 c = v_uv - 0.5;
    // 1) Scale to isotropic physical space (units = viewport height).
    //    Without this, X spans vp_w pixels while Y spans vp_h pixels,
    //    so a naive rotation would mix incompatible axes.
    vec2 iso = vec2(c.x * vp_aspect, c.y);
    // 2) Rotate in this isotropic space — no distortion.
    float cr = cos(bg_rotation);
    float sr = sin(bg_rotation);
    vec2 rot = vec2(cr * iso.x - sr * iso.y, sr * iso.x + cr * iso.y);
    // 3) Convert from isotropic space to image UV space, then apply user scale/offset.
    //    Divide X by img_aspect so that one UV unit = one image width.
    v_uv = vec2(rot.x / (img_aspect * max(bg_scale_x, 0.01)) - bg_offset_x,
                rot.y /              max(bg_scale_y, 0.01)    - bg_offset_y) + 0.5;
    gl_Position = vec4(in_pos, 0.9999, 1.0);
}"""

BG_FRAG = """
#version 330
in vec2 v_uv;
uniform sampler2D bg_tex;
uniform float bg_alpha;
out vec4 f_color;
void main() {
    vec2 clamped = clamp(v_uv, 0.0, 1.0);
    if (distance(clamped, v_uv) > 0.001) discard;
    vec4 col = texture(bg_tex, clamped);
    f_color = vec4(col.rgb, bg_alpha);
}
"""

DEPTH_VERT = """
#version 330
in vec3 in_pos;
uniform mat4 mvp;
void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
}
"""

DEPTH_FRAG = """
#version 330
void main() {
}
"""

# %% MATH HELPERS ─────────────────────────────────────────────────────────────

def look_at(eye, center, up=(0, 1, 0)):
    f = np.asarray(center, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
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

def orthographic(left, right, bottom, top, near=-1000.0, far=1000.0):
    return np.array([
        [2.0 / (right - left), 0, 0, -(right + left) / (right - left)],
        [0, 2.0 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
        [0, 0, -2.0 / (far - near), -(far + near) / (far - near)],
        [0, 0, 0, 1.0],
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


def compute_tube_uvs(verts, n_points):
    """Create UVs for tube meshes: U follows the curve, V wraps the yarn cross-section."""
    verts = np.asarray(verts, dtype=np.float32)
    n_points = int(max(n_points, 1))
    if len(verts) == 0:
        return np.empty((0, 2), dtype=np.float32)
    ring_count = max(len(verts) // n_points, 1)
    u = np.repeat(
        np.linspace(0.0, 1.0, n_points, dtype=np.float32),
        ring_count,
    )[:len(verts)]
    v = np.tile(
        np.linspace(0.0, 1.0, ring_count, endpoint=False, dtype=np.float32),
        n_points,
    )[:len(verts)]
    return np.column_stack((u, v)).astype(np.float32)

def rotation_matrix_xyz(rx, ry, rz):
    """4×4 rotation matrix from XYZ Euler angles (radians), applied as Rz @ Ry @ Rx."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1,0,0,0],[0,cx,-sx,0],[0,sx,cx,0],[0,0,0,1]], dtype=np.float32)
    Ry = np.array([[cy,0,sy,0],[0,1,0,0],[-sy,0,cy,0],[0,0,0,1]], dtype=np.float32)
    Rz = np.array([[cz,-sz,0,0],[sz,cz,0,0],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
    return Rz @ Ry @ Rx

# %% CAMERA ───────────────────────────────────────────────────────────────────

class Camera:
    def __init__(self):
        self.target  = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.dist    = 15.0
        self.az      = 0.0   # Horizontal angle (rad)
        self.el      = 0.0   # Vertical angle (rad)
        self.fov_deg = 45.0

    def _pos(self):
        y = self.dist * np.sin(self.el)
        xz = self.dist * np.cos(self.el)
        x = xz * np.sin(self.az)
        z = xz * np.cos(self.az)
        return self.target + np.array([x, y, z], dtype=np.float32)

    def view(self):
        return look_at(self._pos(), self.target)

    def proj(self, w, h):
        aspect = max(1.0, w) / max(1.0, h)
        # Match the perspective framing at the current distance, but with
        # orthographic projection to remove perspective distortion.
        half_h = max(1e-4, self.dist * np.tan(np.radians(self.fov_deg) * 0.5))
        half_w = half_h * aspect
        return orthographic(-half_w, half_w, -half_h, half_h, near=-1000.0, far=1000.0)

    def mvp(self, w, h):
        return self.proj(w, h) @ self.view()

    def mv(self, w, h):
        return self.view()

    def unproject(self, px, py, vp_w, vp_h):
        """Constructs a world-space ray (origin, dir) matching the screen pixel coordinate."""
        x = (2.0 * px) / vp_w - 1.0
        y = 1.0 - (2.0 * py) / vp_h
        inv_proj = np.linalg.inv(self.proj(vp_w, vp_h))
        inv_view = np.linalg.inv(self.view())

        # ray point on near plane (z=-1 in NDC, but projection mapping varies)
        def pt(z):
            clip = np.array([x, y, z, 1.0], dtype=np.float32)
            eye = inv_proj @ clip
            eye /= eye[3]
            world = inv_view @ eye
            return world[:3]

        p0 = pt(-1.0)
        p1 = pt(1.0)
        rd = p1 - p0
        rd /= np.linalg.norm(rd) + 1e-8
        return p0, rd

    def orbit(self, dx, dy):
        self.az -= dx * 0.005
        self.el = np.clip(self.el + dy * 0.005, -np.pi/2 + 0.01, np.pi/2 - 0.01)

    def zoom(self, delta):
        self.dist = max(1.0, self.dist - delta)

    def zoom_factor(self):
        # 1.0 at reference distance of 15.0.
        # Lower distance means zoom-in, so this factor increases on zoom-in.
        return 15.0 / max(self.dist, 1e-6)

    def pan(self, dx, dy):
        """Translate model camera target in screen-plane dimensions."""
        sens = self.dist * 0.0012
        view = self.view()
        right = view[0, :3]
        up    = view[1, :3]
        self.target += (-right * dx + up * dy) * sens

# %% INTERACTION AND RAY MATH ──────────────────────────────────────────────────

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

def transform_points(points, matrix):
    pts = np.asarray(points)
    if not len(pts):
        return pts
    homo = np.column_stack((pts, np.ones(len(pts), dtype=np.float32)))
    res = homo @ matrix.T
    return res[:, :3] / res[:, 3:4]

# %% MESH RENDERER ────────────────────────────────────────────────────────────

class MeshRenderer:
    def __init__(self, ctx, vp_w, vp_h):
        self.ctx     = ctx
        self.prog    = ctx.program(vertex_shader=MESH_VERT, fragment_shader=MESH_FRAG)
        self.outline_prog = ctx.program(vertex_shader=OUTLINE_VERT, fragment_shader=OUTLINE_FRAG)
        self.pt_prog = ctx.program(vertex_shader=PT_VERT,   fragment_shader=PT_FRAG)
        self.bg_prog = ctx.program(vertex_shader=BG_VERT,   fragment_shader=BG_FRAG)
        self.depth_prog = ctx.program(vertex_shader=DEPTH_VERT, fragment_shader=DEPTH_FRAG)
        # Full-screen quad for background (triangle strip)
        quad = np.array([[-1,-1],[1,-1],[-1,1],[1,1]], dtype=np.float32)
        self.bg_vao = ctx.vertex_array(
            self.bg_prog,
            [(ctx.buffer(quad.tobytes()), '2f', 'in_pos')],
        )
        self.vp_w    = 1
        self.vp_h    = 1
        self.color_tex = None
        self.depth_tex = None
        self.fbo = None
        self.meshes = []      # list of (vao, outline_vao, depth_vao, n_indices, color, row_idx)
        self.mesh_pick_data = []
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
            if self.depth_tex is not None:
                self.depth_tex.release()

        self.vp_w = vp_w
        self.vp_h = vp_h
        self.color_tex = self.ctx.texture((vp_w, vp_h), 4)
        self.color_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.depth_tex = self.ctx.depth_texture((vp_w, vp_h))
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.color_tex],
            depth_attachment=self.depth_tex,
        )

    def set_meshes(self, verts_list, faces_list, row_indices=None, colors=None, meta=None):
        for vao, outline_vao, depth_vao, _, _, _ in self.meshes:
            vao.release()
            outline_vao.release()
            depth_vao.release()
        self.meshes.clear()
        self.mesh_pick_data.clear()
        for i, ((verts, n_points), faces) in enumerate(zip(verts_list, faces_list)):
            v  = np.array(verts, dtype=np.float32)
            f  = np.array(faces, dtype=np.int32)
            # quads → triangles
            tris = np.empty((len(f) * 2, 3), dtype=np.int32)
            tris[0::2] = f[:, [0, 1, 2]]
            tris[1::2] = f[:, [0, 2, 3]]
            nm = compute_normals(v, tris).astype(np.float32)
            pos_vbo = self.ctx.buffer(v.tobytes())
            norm_vbo = self.ctx.buffer(nm.tobytes())
            ibo = self.ctx.buffer(tris.astype(np.int32).tobytes())
            vao = self.ctx.vertex_array(self.prog, [
                (pos_vbo,  '3f', 'in_pos'),
                (norm_vbo, '3f', 'in_norm'),
            ], ibo)
            outline_vao = self.ctx.vertex_array(self.outline_prog, [
                (pos_vbo,  '3f', 'in_pos'),
                (norm_vbo, '3f', 'in_norm'),
            ], ibo)
            depth_vao = self.ctx.vertex_array(self.depth_prog, [
                (pos_vbo,  '3f', 'in_pos'),
            ], ibo)
            if meta is not None:
                row_idx = meta[i].get('row', i)
            else:
                row_idx = row_indices[i] if row_indices is not None else i
            if colors is not None:
                color = colors[row_idx % len(colors)]
            else:
                color = [0.8, 0.2, 0.2]
            self.meshes.append((vao, outline_vao, depth_vao, len(tris) * 3, color, row_idx))
            self.mesh_pick_data.append((v, row_idx))

    @staticmethod
    def _row_visible(row_idx, visible_rows):
        if visible_rows is None or len(visible_rows) == 0:
            return True
        base_idx = int(row_idx) % len(visible_rows)
        return bool(visible_rows[base_idx])

    def pick_mesh_index(self, model_mat, camera, vp_w, vp_h, mouse_x, mouse_y, visible_rows=None, max_distance_px=16.0):
        if not self.mesh_pick_data:
            return -1

        view_proj = camera.proj(vp_w, vp_h) @ camera.view()
        best_idx = -1
        best_d2 = float(max_distance_px * max_distance_px)

        for mesh_idx, (verts, row_idx) in enumerate(self.mesh_pick_data):
            if not self._row_visible(row_idx, visible_rows):
                continue
            if len(verts) == 0:
                continue
            stride = max(1, len(verts) // 400)
            sample = verts[::stride]
            world_pts = transform_points(sample, model_mat)
            homo = np.column_stack((world_pts, np.ones(len(world_pts), dtype=np.float32)))
            clip = homo @ view_proj.T
            valid = clip[:, 3] > 1e-6
            if not np.any(valid):
                continue

            ndc = np.zeros((len(world_pts), 3), dtype=np.float32)
            ndc[valid] = clip[valid, :3] / clip[valid, 3:4]
            in_view = (
                valid
                & (ndc[:, 0] >= -1.0) & (ndc[:, 0] <= 1.0)
                & (ndc[:, 1] >= -1.0) & (ndc[:, 1] <= 1.0)
            )
            if not np.any(in_view):
                continue

            screen = np.column_stack((
                (ndc[:, 0] * 0.5 + 0.5) * vp_w,
                (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * vp_h,
            ))
            d2 = np.sum((screen - np.array([mouse_x, mouse_y], dtype=np.float32)) ** 2, axis=1)
            d2[~in_view] = np.inf
            local_best = float(np.min(d2))
            if local_best < best_d2:
                best_d2 = local_best
                best_idx = mesh_idx

        return best_idx

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

    def get_row_for_mesh_index(self, mesh_idx):
        if mesh_idx < 0 or mesh_idx >= len(self.meshes):
            return None
        return int(self.meshes[mesh_idx][5])

    def sample_color(self, x, y):
        ix = int(np.clip(x, 0, self.vp_w - 1))
        iy = int(np.clip(y, 0, self.vp_h - 1))
        # FBO read viewport origin is bottom-left; UI coordinates are top-left.
        iy_flipped = self.vp_h - 1 - iy
        raw = self.fbo.read(viewport=(ix, iy_flipped, 1, 1), components=3, dtype='f1')
        if not raw or len(raw) < 3:
            return None
        rgb = np.frombuffer(raw, dtype=np.uint8)[:3].astype(np.float32) / 255.0
        return rgb

    def _set_program_uniforms(self, prog, uniforms_dict):
        """Data-driven dynamic uniform mapping."""
        for name, value in uniforms_dict.items():
            if name in prog:
                if name in ('mvp', 'mv'):
                    continue  # Matrices are written explicitly as bytes
                if isinstance(value, (list, tuple, np.ndarray)):
                    prog[name].value = tuple(float(x) for x in value)
                else:
                    prog[name].value = float(value)

    def render(self, mvp, mv, material_uniforms, hover_idx=-1, selected_idx=-1,
               hover_mesh_idx=-1, selected_mesh_idx=-1,
               visible_rows=None,
               bg_tex=None, bg_alpha=0.5, bg_uniforms=None,
               camera=None):
        self.fbo.use()
        self.ctx.viewport = (0, 0, self.vp_w, self.vp_h)
        
        # 1. Clear color and depth with depth writes enabled
        self.fbo.depth_mask = True
        self.ctx.clear(0.12, 0.12, 0.12, 1.0)

        # 2. Depth pre-pass
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.color_mask = (False, False, False, False)
        
        self.depth_prog['mvp'].write(mvp.T.tobytes())
        for mesh_idx, (_, _, depth_vao, n_idx, _, row_idx) in enumerate(self.meshes):
            if visible_rows is not None and row_idx < len(visible_rows) and not bool(visible_rows[row_idx]):
                continue
            depth_vao.render(moderngl.TRIANGLES)

        # 3. Restore color writes and disable depth writes
        self.ctx.color_mask = (True, True, True, True)
        self.fbo.depth_mask = False
        self.ctx.depth_func = '<='

        # ── Draw reference-image background quad ──────────────────────────────
        if bg_tex is not None:
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.ctx.disable(moderngl.CULL_FACE)
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            bg_tex.use(0)
            self.bg_prog['bg_tex'].value = 0
            self.bg_prog['bg_alpha'].value = float(bg_alpha)
            if bg_uniforms is not None:
                self._set_program_uniforms(self.bg_prog, bg_uniforms)
            self.bg_vao.render(moderngl.TRIANGLE_STRIP)
            self.ctx.disable(moderngl.BLEND)

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)

        # Enable blending for transparent model
        model_alpha = material_uniforms.get('model_alpha', 1.0)
        use_model_blend = model_alpha < 0.9999
        if use_model_blend:
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # Bind depth texture to unit 1
        self.depth_tex.use(location=1)
        if 'depth_tex' in self.prog:
            self.prog['depth_tex'].value = 1
        if 'viewport_size' in self.prog:
            self.prog['viewport_size'].value = (float(self.vp_w), float(self.vp_h))

        # Calculate ao_uv_scale
        if camera is not None:
            dist = camera.dist
            fov = camera.fov_deg
        else:
            dist = 15.0
            fov = 45.0
        half_h = max(1e-4, dist * np.tan(np.radians(fov) * 0.5))
        half_w = half_h * (self.vp_w / self.vp_h)
        ao_radius = material_uniforms.get('ao_radius', 0.15)
        ao_uv_scale = (
            float(ao_radius / (2.0 * half_w)),
            float(ao_radius / (2.0 * half_h))
        )
        if 'ao_uv_scale' in self.prog:
            self.prog['ao_uv_scale'].value = ao_uv_scale

        # Write matrices and common uniforms once per pass
        self.prog['mvp'].write(mvp.T.tobytes())
        self.prog['mv'].write(mv.T.tobytes())
        self._set_program_uniforms(self.prog, material_uniforms)

        for mesh_idx, (vao, outline_vao, depth_vao, n_idx, color, row_idx) in enumerate(self.meshes):
            if not self._row_visible(row_idx, visible_rows):
                continue
            base_color = np.asarray(color, dtype=np.float32)
            self.prog['color'].value = tuple(float(c) for c in base_color)
            vao.render(moderngl.TRIANGLES)

        # Restore default depth settings for outline & points
        self.ctx.depth_func = '<'
        self.fbo.depth_mask = True

        # Silhouette highlight pass: draw expanded backfaces for hover/selection.
        has_outline = selected_mesh_idx >= 0 or hover_mesh_idx >= 0
        if has_outline:
            self.outline_prog['mvp'].write(mvp.T.tobytes())
            self.outline_prog['outline_width'].value = 0.006
            self.ctx.disable(moderngl.CULL_FACE)

            def draw_outline(mesh_idx, color):
                if mesh_idx < 0 or mesh_idx >= len(self.meshes):
                    return
                _, outline_vao, _, _, _, row_idx = self.meshes[mesh_idx]
                if not self._row_visible(row_idx, visible_rows):
                    return
                self.outline_prog['outline_color'].value = color
                outline_vao.render(moderngl.TRIANGLES)

            # Selected wins if both states point to the same yarn.
            draw_outline(selected_mesh_idx, (0.15, 1.0, 0.25))
            if hover_mesh_idx != selected_mesh_idx:
                draw_outline(hover_mesh_idx, (1.0, 0.95, 0.2))

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

# %% IMAGE TEXTURE HELPER ─────────────────────────────────────────────────────

def pil_to_texture(ctx, pil_img):
    """Upload a PIL image as a moderngl texture (RGBA, Y-flipped for GL)."""
    img  = pil_img.convert("RGBA").transpose(Image.FLIP_TOP_BOTTOM)
    w, h = img.size
    tex  = ctx.texture((w, h), 4, img.tobytes())
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    return tex


def draw_fitted_texture(texture_id, tex_w, tex_h, avail_w, avail_h, flip_y=False, zoom=1.0, pan=(0.0, 0.0)):
    """Draws a texture centered inside the available region while preserving aspect."""
    if tex_w <= 0 or tex_h <= 0 or avail_w <= 1 or avail_h <= 1:
        return None

    scale = min(avail_w / tex_w, avail_h / tex_h)
    zoom = max(0.05, float(zoom))
    pan_x, pan_y = float(pan[0]), float(pan[1])
    draw_w = max(1.0, tex_w * scale * zoom)
    draw_h = max(1.0, tex_h * scale * zoom)
    offset_x = (avail_w - draw_w) * 0.5 + pan_x
    offset_y = (avail_h - draw_h) * 0.5 + pan_y
    cursor = imgui.get_cursor_pos()
    imgui.set_cursor_pos((cursor.x + offset_x, cursor.y + offset_y))
    draw_pos = imgui.get_cursor_screen_pos()

    uv0 = imgui.ImVec2(0, 1) if flip_y else imgui.ImVec2(0, 0)
    uv1 = imgui.ImVec2(1, 0) if flip_y else imgui.ImVec2(1, 1)
    imgui.image(imgui.ImTextureRef(texture_id), imgui.ImVec2(draw_w, draw_h), uv0=uv0, uv1=uv1)
    return (float(draw_pos.x), float(draw_pos.y), float(draw_w), float(draw_h))
