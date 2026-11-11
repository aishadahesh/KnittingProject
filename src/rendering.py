import logging
import math
import cv2
import numpy as np
import moderngl
from PIL import Image
from imgui_bundle import imgui

import charuco
import paths
import scan_geometry

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
uniform vec3  light_dir;
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
    vec3  L    = normalize(light_dir);
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
    float spec = pow(clamp(dot(normalize(v_norm), L), 0.0, 1.0), gloss_power);
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

# ── Yarn-simulation debug overlays ────────────────────────────────────────────
# Raw centreline segments and the contact points the collision barrier is acting
# on, drawn straight from the solver's own geometry rather than the display mesh.





# Ground plane for the orbit view: an analytic grid shaded in the fragment
# stage, so line density stays even regardless of how far the camera is.
GRID_VERT = """
#version 330
in vec3 in_pos;
uniform mat4 mvp;
out vec3 v_world_pos;
void main() {
    v_world_pos = in_pos;
    gl_Position = mvp * vec4(in_pos, 1.0);
}
"""

GRID_FRAG = """
#version 330
in vec3 v_world_pos;
out vec4 f_color;
void main() {
    vec2 coord = v_world_pos.xy;
    vec2 grid = abs(fract(coord - 0.5) - 0.5) / fwidth(coord);
    float line = min(grid.x, grid.y);
    float color_intensity = 1.0 - min(line, 1.0);

    vec2 major_grid = abs(fract(coord * 0.1 - 0.5) - 0.5) / fwidth(coord * 0.1);
    float major_line = min(major_grid.x, major_grid.y);
    float major_intensity = 1.0 - min(major_line, 1.0);

    vec4 final_color = vec4(0.32, 0.32, 0.32, 0.28) * color_intensity + vec4(0.48, 0.48, 0.48, 0.6) * major_intensity;

    float dist = length(coord);
    float fade = clamp(1.0 - dist / 80.0, 0.0, 1.0);
    final_color.a *= fade;

    if (final_color.a < 0.01) discard;
    f_color = final_color;
}
"""

PT_VERT = """
#version 330
in  vec3 in_pos;
uniform mat4 mvp;
uniform int  hover_idx;
uniform int  selected_idx;
uniform int  n_real;
flat out int state;   // 0=normal 1=hover 2=selected
flat out int is_virtual;
void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
    int vid = gl_VertexID;
    state = (vid == selected_idx) ? 2 : (vid == hover_idx ? 1 : 0);
    is_virtual = (vid >= n_real) ? 1 : 0;
    gl_PointSize = (state > 0) ? 16.0 : 10.0;
}
"""

PT_FRAG = """
#version 330
flat in int state;
flat in int is_virtual;
out vec4 f_color;
void main() {
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    if (dot(c, c) > 1.0) discard;
    if      (state == 2) f_color = vec4(1.0, 1.0, 0.0, 1.0);  // drag  → yellow
    else if (state == 1) f_color = vec4(1.0, 0.5, 0.0, 1.0);  // hover → orange
    else if (is_virtual == 1) f_color = vec4(0.0, 0.8, 1.0, 0.9); // virtual -> cyan
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

def orthographic(left, right, bottom, top, near=-1000.0, far=1000.0):
    return np.array([
        [2.0 / (right - left), 0, 0, -(right + left) / (right - left)],
        [0, 2.0 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
        [0, 0, -2.0 / (far - near), -(far + near) / (far - near)],
        [0, 0, 0, 1.0],
    ], dtype=np.float32)

def perspective(vfov_deg, aspect, near=0.05, far=500.0):
    f = 1.0 / math.tan(math.radians(max(5.0, min(170.0, float(vfov_deg)))) * 0.5)
    a = max(float(aspect), 1e-4)
    nf = 1.0 / max(float(near) - float(far), -1e6 if near < far else 1e-4)
    return np.array([
        [f / a, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near) * nf, 2.0 * far * near * nf],
        [0.0, 0.0, -1.0, 0.0],
    ], dtype=np.float32)

def compute_normals(verts, tris):
    # Area-weighted vertex normals: each triangle adds its unnormalised face
    # normal to all three of its corners.
    #
    # The scatter was three np.add.at calls. np.add.at takes numpy's unbuffered
    # path, which serialises every repeated-index update; np.bincount performs
    # the same accumulation as a buffered reduction. Measured on a real yarn
    # mesh (7200 verts, 14376 tris): 2.499 ms -> 1.058 ms, with a maximum
    # difference of 1.9e-7 -- bincount sums in float64 rather than float32, so
    # if anything it is the more accurate of the two.
    e1 = verts[tris[:,1]] - verts[tris[:,0]]
    e2 = verts[tris[:,2]] - verts[tris[:,0]]
    fn = np.cross(e1, e2)
    n = np.empty_like(verts)
    vertex_count = len(verts)
    for axis in range(3):
        weights = fn[:, axis]
        n[:, axis] = (
            np.bincount(tris[:, 0], weights=weights, minlength=vertex_count)
            + np.bincount(tris[:, 1], weights=weights, minlength=vertex_count)
            + np.bincount(tris[:, 2], weights=weights, minlength=vertex_count)
        )[:vertex_count]
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-8)


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
        span = max(1000.0, float(self.dist) * 8.0)
        return orthographic(-half_w, half_w, -half_h, half_h, near=-span, far=span)

    def mvp(self, w, h):
        return self.proj(w, h) @ self.view()

    def mv(self, w, h):
        return self.view()

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

def transform_points(points, matrix):
    pts = np.asarray(points)
    if not len(pts):
        return pts
    homo = np.column_stack((pts, np.ones(len(pts), dtype=np.float32)))
    res = homo @ matrix.T
    return res[:, :3] / res[:, 3:4]


SKY_VERT = """
#version 330 core
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

SKY_FRAG = """
#version 330 core
uniform float uHeight;
uniform float uHorizon;
uniform vec3 uStops[4];
out vec4 fragColor;
void main() {
    float row = uHeight - gl_FragCoord.y;
    float t = (row - uHorizon) / max(uHeight, 1.0);
    fragColor = vec4(row < uHorizon
        ? mix(uStops[0], uStops[1], clamp(t + 1.0, 0.0, 1.0))
        : mix(uStops[2], uStops[3], clamp(t, 0.0, 1.0)), 1.0);
}
"""

WORLD_VERT = """
#version 330 core
in vec3 in_pos;
in vec3 in_norm;
in vec3 in_col;
in vec2 in_uv;
uniform mat4 mvp;
uniform mat4 model;
out vec3 vPos;
out vec3 vNrm;
out vec3 vCol;
out vec2 vUv;
void main() {
    vec4 wp = model * vec4(in_pos, 1.0);
    vPos = wp.xyz;
    vNrm = mat3(model) * in_norm;
    vCol = in_col;
    vUv = in_uv;
    gl_Position = mvp * vec4(in_pos, 1.0);
}
"""

WORLD_FRAG = """
#version 330 core
in vec3 vPos;
in vec3 vNrm;
in vec3 vCol;
in vec2 vUv;
uniform vec3 base_color;
uniform vec3 uLight;
uniform vec3 uUp;
uniform vec3 uCamPos;
uniform float uShadow;
uniform float uShadowAlpha;
uniform float uUseTex;
uniform float uUseVertexCol;
uniform sampler2D uTex;
out vec4 fragColor;
void main() {
    if (uShadow > 0.5) {
        fragColor = vec4(0.10, 0.11, 0.13, uShadowAlpha);
        return;
    }
    vec3 col = (uUseVertexCol > 0.5) ? vCol : base_color;
    if (uUseTex > 0.5) {
        col = texture(uTex, vUv).rgb;
    }
    if (dot(vNrm, vNrm) < 1e-8) {
        fragColor = vec4(col, 1.0);
        return;
    }
    vec3 n = normalize(vNrm);
    if (!gl_FrontFacing) n = -n;
    vec3 l = normalize(uLight);
    vec3 v = normalize(uCamPos - vPos);
    vec3 h = normalize(l + v);
    float diff = max(dot(n, l), 0.0);
    float spec = pow(max(dot(n, h), 0.0), 48.0);
    vec3 amb = mix(vec3(0.34, 0.34, 0.32), vec3(0.55, 0.58, 0.62),
                   0.5 + 0.5 * dot(n, normalize(uUp)));
    vec3 c = col * (amb + 0.70 * diff) + vec3(0.9) * (0.20 * spec);
    fragColor = vec4(min(c, vec3(1.0)), 1.0);
}
"""

LINE_VERT = """
#version 330
in vec3 in_pos;
in vec3 in_color;
uniform mat4 mvp;
out vec3 v_color;
void main() {
    v_color = in_color;
    gl_Position = mvp * vec4(in_pos, 1.0);
}
"""

LINE_FRAG = """
#version 330
in vec3 v_color;
out vec4 frag_color;
void main() {
    frag_color = vec4(v_color, 1.0);
}
"""

_SKY_STOPS = np.array([
    (0.518, 0.690, 0.871),
    (0.878, 0.925, 0.965),
    (0.894, 0.886, 0.871),
    (0.776, 0.769, 0.753),
], dtype=np.float32)

_WORLD_LIGHT_DIR = (
    np.array([0.35, -0.25, 0.90], dtype=np.float32)
    / float(np.linalg.norm([0.35, -0.25, 0.90]))
)
_WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float32)


def _ground_projection_z(light, plane_z):
    M = np.eye(4, dtype=np.float32)
    rx = float(light[0]) / float(light[2])
    ry = float(light[1]) / float(light[2])
    pz = float(plane_z)
    M[0, 2], M[0, 3] = -rx, pz * rx
    M[1, 2], M[1, 3] = -ry, pz * ry
    M[2, 2], M[2, 3] = 0.0, pz
    return M


def _box_vertices(lo, hi, col, include_top=True):
    x0, y0, z0 = float(lo[0]), float(lo[1]), float(lo[2])
    x1, y1, z1 = float(hi[0]), float(hi[1]), float(hi[2])
    c = (float(col[0]), float(col[1]), float(col[2]))
    faces = [
        (((x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (x0, y0, z0)), (0.0, 0.0, -1.0)),
        (((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)), (1.0, 0.0, 0.0)),
        (((x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)), (-1.0, 0.0, 0.0)),
        (((x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)), (0.0, 1.0, 0.0)),
        (((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)), (0.0, -1.0, 0.0)),
    ]
    if include_top:
        faces.append((((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)), (0.0, 0.0, 1.0)))
    rows = []
    for pts, nrm in faces:
        for i in (0, 1, 2, 0, 2, 3):
            rows.append((*pts[i], *nrm, *c))
    return rows

_UR5E_LINK_FILES = (
    ("base", ("base_0.obj", "base_1.obj")),
    ("shoulder_link", ("shoulder_0.obj", "shoulder_1.obj", "shoulder_2.obj")),
    ("upper_arm_link", ("upperarm_0.obj", "upperarm_1.obj", "upperarm_2.obj", "upperarm_3.obj")),
    ("forearm_link", ("forearm_0.obj", "forearm_1.obj", "forearm_2.obj", "forearm_3.obj")),
    ("wrist_1_link", ("wrist1_0.obj", "wrist1_1.obj", "wrist1_2.obj")),
    ("wrist_2_link", ("wrist2_0.obj", "wrist2_1.obj", "wrist2_2.obj")),
    ("wrist_3_link", ("wrist3.obj",)),
)
_UR5E_VISUAL_CACHE = None


def _parse_obj_mesh(path):
    verts = []
    faces = []
    mtl = ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("usemtl "):
                mtl = line.split(None, 1)[1].strip()
            elif line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                parts = line.split()[1:]
                idx = [int(p.split("/")[0]) - 1 for p in parts]
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[k], idx[k + 1]))
    if not verts or not faces:
        return None
    v_arr = np.asarray(verts, dtype=np.float32)
    t_arr = np.asarray(faces, dtype=np.int32)
    n_arr = compute_normals(v_arr, t_arr).astype(np.float32)
    mtl_low = mtl.lower()
    if "blue" in mtl_low:
        color = (0.34, 0.64, 0.86)
    elif "black" in mtl_low or "dark" in mtl_low:
        color = (0.18, 0.19, 0.22)
    else:
        color = (0.76, 0.78, 0.81)
    return v_arr.tobytes(), n_arr.tobytes(), t_arr.tobytes(), int(len(t_arr) * 3), color


def _load_ur5e_visual_meshes():
    global _UR5E_VISUAL_CACHE
    if _UR5E_VISUAL_CACHE is not None:
        return _UR5E_VISUAL_CACHE
    meshes = []
    mesh_dir = paths.UR5E_MESHES_DIR
    for link_idx, filenames in _UR5E_LINK_FILES:
        for fname in filenames:
            fpath = mesh_dir / fname
            if not fpath.is_file():
                continue
            parsed = _parse_obj_mesh(fpath)
            if parsed is not None:
                meshes.append((link_idx, *parsed))
    _UR5E_VISUAL_CACHE = meshes
    return meshes

# %% MESH RENDERER ────────────────────────────────────────────────────────────

class MeshRenderer:
    def __init__(self, ctx, vp_w=640, vp_h=480):
        self.ctx     = ctx
        self.prog    = ctx.program(vertex_shader=MESH_VERT, fragment_shader=MESH_FRAG)
        self.outline_prog = ctx.program(vertex_shader=OUTLINE_VERT, fragment_shader=OUTLINE_FRAG)
        self.pt_prog = ctx.program(vertex_shader=PT_VERT,   fragment_shader=PT_FRAG)
        self.bg_prog = ctx.program(vertex_shader=BG_VERT,   fragment_shader=BG_FRAG)
        self.depth_prog = ctx.program(vertex_shader=DEPTH_VERT, fragment_shader=DEPTH_FRAG)
        # Full-screen quad for background (triangle strip)
        quad = np.array([[-1,-1],[1,-1],[-1,1],[1,1]], dtype=np.float32)
        self.bg_vbo = ctx.buffer(quad.tobytes())
        self.bg_vao = ctx.vertex_array(
            self.bg_prog,
            [(self.bg_vbo, '2f', 'in_pos')],
        )
        # Grid plane: one large quad, with the grid itself drawn analytically in
        # the fragment shader (see GRID_FRAG). Only drawn when show_grid is set,
        # which in practice means the orbit view.
        self.grid_prog = ctx.program(vertex_shader=GRID_VERT, fragment_shader=GRID_FRAG)
        grid_vertices = np.array([
            [-150.0, -150.0, 0.0],
            [150.0, -150.0, 0.0],
            [-150.0, 150.0, 0.0],
            [150.0, 150.0, 0.0],
        ], dtype=np.float32)
        grid_indices = np.array([0, 2, 1, 1, 2, 3], dtype=np.int32)
        self.grid_vbo = ctx.buffer(grid_vertices.tobytes())
        self.grid_ibo = ctx.buffer(grid_indices.tobytes())
        self.grid_vao = ctx.vertex_array(
            self.grid_prog,
            [(self.grid_vbo, '3f', 'in_pos')],
            self.grid_ibo,
        )
        self.sky_prog = ctx.program(vertex_shader=SKY_VERT, fragment_shader=SKY_FRAG)
        self.sky_vao = ctx.vertex_array(self.sky_prog, [])
        if 'uStops' in self.sky_prog:
            self.sky_prog['uStops'].write(_SKY_STOPS.tobytes())
        self.world_prog = ctx.program(vertex_shader=WORLD_VERT, fragment_shader=WORLD_FRAG)
        self.line_prog = ctx.program(vertex_shader=LINE_VERT, fragment_shader=LINE_FRAG)
        self.robot_vaos = []
        self.robot_buffers = []
        self.room_vao = None
        self.room_vbo = None
        self.room_vertex_count = 0
        self.board_plinth_vao = None
        self.board_plinth_vbo = None
        self.board_plinth_vertex_count = 0
        self.board_vao = None
        self.board_buffers = []
        self.charuco_tex = None
        self._env_cache_key = None
        self.wrist_fbo = None
        self.wrist_color_tex = None
        self.wrist_depth_tex = None
        self.wrist_size = (0, 0)
        self._tile_fbo = None
        self._tile_color_tex = None
        self._tile_depth_tex = None
        self._tile_fbo_size = (0, 0)
        self._swatch_tile_textures = []
        self._swatch_tiles_epoch = -1

        self.vp_w    = 1
        self.vp_h    = 1
        self.color_tex = None
        self.depth_tex = None
        self.fbo = None
        self.meshes = []      # list of (vao, outline_vao, depth_vao, n_indices, color, row_idx)
        self.mesh_buffers = []
        self.mesh_pick_data = []
        # Keep the semantic mesh metadata beside the GPU buffers.  Mode
        # isolation checks use this to reject a Scan grid before it can be
        # displayed by Puzzle Mode; row_idx alone cannot distinguish the two.
        self.mesh_meta = []
        self.pt_vao = None
        self.pt_buffer = None
        self.n_pts  = 0
        self.resize(vp_w, vp_h)

    @property
    def texture_id(self):
        return self.fbo.color_attachments[0].glo

    def close(self):
        self.set_meshes([], [])
        self.set_ctrl_pts([])
        for vao, _, _ in self.robot_vaos:
            vao.release()
        self.robot_vaos.clear()
        for buf in self.robot_buffers:
            buf.release()
        self.robot_buffers.clear()
        if self.board_vao is not None:
            self.board_vao.release()
            self.board_vao = None
        for buf in self.board_buffers:
            buf.release()
        self.board_buffers.clear()
        for tex in self._swatch_tile_textures:
            tex.release()
        self._swatch_tile_textures.clear()
        self._swatch_tiles_epoch = -1
        for name in ('room_vao', 'room_vbo', 'board_plinth_vao', 'board_plinth_vbo',
                     'charuco_tex', 'sky_vao', 'sky_prog',
                     'wrist_fbo', 'wrist_color_tex', 'wrist_depth_tex',
                     '_tile_fbo', '_tile_color_tex', '_tile_depth_tex',
                     'bg_vao', 'grid_vao', 'bg_vbo', 'grid_vbo', 'grid_ibo',
                     'fbo', 'color_tex', 'depth_tex', 'prog', 'outline_prog',
                     'pt_prog', 'bg_prog', 'depth_prog', 'grid_prog',
                     'world_prog', 'line_prog'):
            resource = getattr(self, name, None)
            if resource is not None:
                resource.release()
                setattr(self, name, None)

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

    @staticmethod
    def prepare_meshes(verts_list, faces_list):
        """Does set_meshes' CPU-side work once, so several renderers can share it.

        Triangulating the quads, computing normals and packing the byte buffers
        depends only on the geometry, not on which renderer receives it. The app
        uploads the same meshes to the main viewport and the orbit view on every
        rebuild, which meant doing all of that twice -- and compute_normals is
        the single most expensive thing in a rebuild.
        """
        prepared = []
        for (verts, _n_points), faces in zip(verts_list, faces_list):
            v = np.array(verts, dtype=np.float32)
            f = np.array(faces, dtype=np.int32)
            # quads → triangles
            tris = np.empty((len(f) * 2, 3), dtype=np.int32)
            tris[0::2] = f[:, [0, 1, 2]]
            tris[1::2] = f[:, [0, 2, 3]]
            nm = compute_normals(v, tris).astype(np.float32)
            prepared.append((v, tris, v.tobytes(), nm.tobytes(), tris.tobytes()))
        return prepared

    def set_meshes(self, verts_list, faces_list, row_indices=None, colors=None, meta=None, prepared=None):
        for vao, outline_vao, depth_vao, _, _, _ in self.meshes:
            vao.release()
            outline_vao.release()
            depth_vao.release()
        self.meshes.clear()
        for buffer in self.mesh_buffers:
            buffer.release()
        self.mesh_buffers.clear()
        self.mesh_pick_data.clear()
        self.mesh_meta = [dict(item) for item in meta] if meta is not None else []
        if prepared is None:
            prepared = self.prepare_meshes(verts_list, faces_list)
        for i, (v, tris, pos_bytes, norm_bytes, tri_bytes) in enumerate(prepared):
            pos_vbo = self.ctx.buffer(pos_bytes)
            norm_vbo = self.ctx.buffer(norm_bytes)
            ibo = self.ctx.buffer(tri_bytes)
            self.mesh_buffers.extend((pos_vbo, norm_vbo, ibo))
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
            if meta is None or not bool(meta[i].get('neighbor_swatch', False)):
                self.mesh_pick_data.append((v, row_idx))

    def set_prepared_meshes(self, prepared_meshes, colors=None, meta=None):
        if not prepared_meshes:
            self.set_meshes([], [], colors=colors, meta=meta, prepared=[])
            return
        if (
            isinstance(prepared_meshes, tuple)
            and len(prepared_meshes) >= 2
            and isinstance(prepared_meshes[0], (list, tuple))
            and (not prepared_meshes[0] or (isinstance(prepared_meshes[0][0], tuple) and len(prepared_meshes[0][0]) == 5))
            and not isinstance(prepared_meshes[0], np.ndarray)
        ):
            prep = prepared_meshes[0]
            if len(prepared_meshes) == 2 and meta is None and isinstance(prepared_meshes[1], list):
                if prepared_meshes[1] and isinstance(prepared_meshes[1][0], dict):
                    meta = prepared_meshes[1]
                elif colors is None:
                    colors = prepared_meshes[1]
            elif len(prepared_meshes) >= 3:
                if colors is None:
                    colors = prepared_meshes[1]
                if meta is None:
                    meta = prepared_meshes[2]
            self.set_meshes([], [], colors=colors, meta=meta, prepared=list(prep))
            return
        self.set_meshes([], [], colors=colors, meta=meta, prepared=list(prepared_meshes))

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
            world_pts = transform_points(verts[::stride], model_mat)
            screen, valid, ndc = project_to_screen(world_pts, view_proj, vp_w, vp_h)
            in_view = (
                valid
                & (ndc[:, 0] >= -1.0) & (ndc[:, 0] <= 1.0)
                & (ndc[:, 1] >= -1.0) & (ndc[:, 1] <= 1.0)
            )
            if not np.any(in_view):
                continue

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
        if self.pt_buffer is not None:
            self.pt_buffer.release()
            self.pt_buffer = None
        self.n_pts = len(flat_pts)
        if self.n_pts == 0:
            return
        pts = np.asarray(flat_pts, dtype=np.float32)
        self.pt_buffer = self.ctx.buffer(pts.tobytes())
        self.pt_vao = self.ctx.vertex_array(self.pt_prog, [(self.pt_buffer, '3f', 'in_pos')])

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
               camera=None, n_real_pts=-1, model_mat=None, show_grid=False,
               world_scene=None):
        self.fbo.use()
        self.ctx.viewport = (0, 0, self.vp_w, self.vp_h)
        
        # 1. Clear color and depth with depth writes enabled
        self.fbo.depth_mask = True
        self.ctx.clear(0.12, 0.12, 0.12, 1.0)

        if world_scene is not None and world_scene.get("show_sky", True) and bg_tex is None:
            view_sky = camera.view() if camera is not None else mv
            proj_sky = camera.proj(self.vp_w, self.vp_h) if camera is not None else np.eye(4, dtype=np.float32)
            self.fbo.depth_mask = False
            self._draw_sky(view_sky, proj_sky, self.vp_h)
            self.fbo.depth_mask = True

        hide_unit_cell = isinstance(world_scene, dict) and bool(world_scene.get("hide_unit_cell_mesh", False))

        if not hide_unit_cell:
            # 2. Depth pre-pass
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.enable(moderngl.CULL_FACE)
            self.ctx.color_mask = (False, False, False, False)

            self.depth_prog['mvp'].write(mvp.T.tobytes())
            for mesh_idx, (_, _, depth_vao, n_idx, _, row_idx) in enumerate(self.meshes):
                if not self._row_visible(row_idx, visible_rows):
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

        if not hide_unit_cell:
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
            if 'light_dir' in self.prog and 'light_dir' not in material_uniforms:
                self.prog['light_dir'].value = (0.5, 1.0, 0.8)
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
        else:
            self.ctx.color_mask = (True, True, True, True)
            self.ctx.depth_func = '<'
            self.fbo.depth_mask = True

        # ── Ground grid ──────────────────────────────────────────────────────
        # Depth-tested but not depth-written, so the model always reads clearly
        # against it and the grid never occludes geometry behind it.
        if show_grid and camera is not None and model_mat is not None:
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.fbo.depth_mask = False
            self.ctx.disable(moderngl.CULL_FACE)

            view_proj = camera.proj(self.vp_w, self.vp_h) @ camera.view()
            grid_mvp = view_proj @ model_mat
            self.grid_prog['mvp'].write(grid_mvp.astype(np.float32).T.tobytes())
            self.grid_vao.render(moderngl.TRIANGLES)

            self.fbo.depth_mask = True
            self.ctx.disable(moderngl.BLEND)

        if world_scene is not None and camera is not None:
            view = camera.view()
            view_proj = camera.proj(self.vp_w, self.vp_h) @ view
            self._draw_world_scene(view_proj, model_mat, world_scene, view=view, target_fbo=self.fbo)

        if self.pt_vao:
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.ctx.disable(moderngl.CULL_FACE)
            self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
            self.pt_prog['mvp'].write(mvp.T.tobytes())
            self.pt_prog['hover_idx'].value    = hover_idx
            self.pt_prog['selected_idx'].value = selected_idx
            if 'n_real' in self.pt_prog:
                self.pt_prog['n_real'].value   = n_real_pts if n_real_pts >= 0 else self.n_pts
            self.pt_vao.render(moderngl.POINTS, vertices=self.n_pts)

        if self.ctx.screen is not None:
            self.ctx.screen.use()

    def _draw_sky(self, view, proj, vp_h):
        if self.sky_vao is None or self.sky_prog is None:
            return
        fwd = -np.asarray(view[2, :3], dtype=np.float32)
        flat = np.array([fwd[0], fwd[1], 0.0], dtype=np.float32)
        n = float(np.linalg.norm(flat))
        d = view[:3, :3] @ (flat / n) if n > 1e-6 else np.zeros(3, dtype=np.float32)
        dz = -float(d[2])
        if dz > 1e-6:
            if abs(float(proj[3, 2])) > 1e-6:
                ndc_y = (float(proj[1, 1]) * float(d[1]) + float(proj[1, 2]) * float(d[2])) / dz
            else:
                ndc_y = 2.41421356 * (float(d[1]) / dz)
            horizon = (0.5 - 0.5 * ndc_y) * float(vp_h)
        else:
            horizon = 0.0 if float(fwd[2]) < 0.0 else float(vp_h)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.sky_prog['uHeight'].value = float(vp_h)
        self.sky_prog['uHorizon'].value = float(horizon)
        self.sky_vao.render(moderngl.TRIANGLES, vertices=3)

    def _ensure_environment_buffers(self, fabric_center, fabric_size=None):
        fc = np.asarray(fabric_center, dtype=np.float32).reshape(-1)[:3]
        spec = charuco.DEFAULT_BOARD
        if isinstance(fabric_size, (tuple, list, np.ndarray)) and len(fabric_size) >= 2:
            w_pat = float(fabric_size[0])
            h_pat = float(fabric_size[1])
        else:
            w_pat = float(spec["squares_x"]) * float(spec["square_mm"]) / 1000.0
            h_pat = float(spec["squares_y"]) * float(spec["square_mm"]) / 1000.0
        border = float(charuco.SHEET_BORDER_M)
        w_sheet = w_pat + 2.0 * border
        h_sheet = h_pat + 2.0 * border

        if self.charuco_tex is None:
            page = charuco.board_image(
                spec,
                px_per_mm=8.0,
                page_mm=(w_sheet * 1000.0, h_sheet * 1000.0),
            )
            paper = np.clip(24 + (page.astype(np.uint16) * (242 - 24)) // 255, 0, 255).astype(np.uint8)
            rgb = np.dstack((paper, paper, paper))
            self.charuco_tex = self.ctx.texture((rgb.shape[1], rgb.shape[0]), 3, rgb.tobytes())
            self.charuco_tex.build_mipmaps()
            self.charuco_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)

        if not self.robot_vaos:
            for link_idx, pos_b, norm_b, tri_b, _n_idx, color in _load_ur5e_visual_meshes():
                p_buf = self.ctx.buffer(pos_b)
                n_buf = self.ctx.buffer(norm_b)
                i_buf = self.ctx.buffer(tri_b)
                self.robot_buffers.extend((p_buf, n_buf, i_buf))
                vao = self.ctx.vertex_array(
                    self.world_prog,
                    [(p_buf, '3f', 'in_pos'), (n_buf, '3f', 'in_norm')],
                    i_buf,
                )
                self.robot_vaos.append((vao, link_idx, color))

        key = (
            round(float(fc[0]), 4),
            round(float(fc[1]), 4),
            round(float(fc[2]), 4),
            round(w_sheet, 4),
            round(h_sheet, 4),
        )
        if self._env_cache_key == key and self.room_vao is not None and self.board_vao is not None:
            return
        self._env_cache_key = key

        for name in ('room_vao', 'room_vbo', 'board_plinth_vao', 'board_plinth_vbo', 'board_vao'):
            res = getattr(self, name, None)
            if res is not None:
                res.release()
                setattr(self, name, None)
        for buf in self.board_buffers:
            buf.release()
        self.board_buffers.clear()

        bx, by, bz = float(fc[0]), float(fc[1]), float(fc[2])
        z_table = bz - 0.002
        z_floor = z_table - 0.80

        room_rows = []
        tile = 0.64
        half_floor = 3.20
        shades = ((0.792, 0.784, 0.769), (0.741, 0.729, 0.714))
        for i in range(10):
            x0 = -half_floor + i * tile
            x1 = x0 + tile
            for j in range(10):
                y0 = -half_floor + j * tile
                y1 = y0 + tile
                col = shades[(i + j) & 1]
                pts = ((x0, y0, z_floor), (x1, y0, z_floor), (x1, y1, z_floor), (x0, y1, z_floor))
                for idx in (0, 1, 2, 0, 2, 3):
                    room_rows.append((*pts[idx], 0.0, 0.0, 0.0, *col))

        tx0 = min(-0.20, bx - 0.35)
        tx1 = max(0.20, bx + 0.35)
        ty0, ty1 = -0.60, 0.60
        table_thick = 0.04
        leg = 0.06
        room_rows.extend(_box_vertices((tx0, ty0, z_table - table_thick), (tx1, ty1, z_table), (0.659, 0.620, 0.588)))
        for lx in (tx0, tx1 - leg):
            for ly in (ty0, ty1 - leg):
                room_rows.extend(_box_vertices((lx, ly, z_floor), (lx + leg, ly + leg, z_table - table_thick), (0.557, 0.518, 0.486)))

        room_arr = np.asarray(room_rows, dtype=np.float32)
        self.room_vertex_count = len(room_arr)
        self.room_vbo = self.ctx.buffer(room_arr.tobytes())
        self.room_vao = self.ctx.vertex_array(
            self.world_prog,
            [(self.room_vbo, '3f 3f 3f', 'in_pos', 'in_norm', 'in_col')],
        )

        hw, hh = 0.5 * w_sheet, 0.5 * h_sheet
        plinth_rows = _box_vertices(
            (bx - hw, by - hh, bz - 0.002),
            (bx + hw, by + hh, bz),
            (0.94, 0.94, 0.92),
            include_top=False,
        )
        plinth_arr = np.asarray(plinth_rows, dtype=np.float32)
        self.board_plinth_vertex_count = len(plinth_arr)
        self.board_plinth_vbo = self.ctx.buffer(plinth_arr.tobytes())
        self.board_plinth_vao = self.ctx.vertex_array(
            self.world_prog,
            [(self.board_plinth_vbo, '3f 3f 3f', 'in_pos', 'in_norm', 'in_col')],
        )

        top_verts = np.array([
            [bx - hw, by - hh, bz, 0.0, 0.0, 0.0, 0.0, 1.0],
            [bx + hw, by - hh, bz, 0.0, 0.0, 0.0, 1.0, 1.0],
            [bx + hw, by + hh, bz, 0.0, 0.0, 0.0, 1.0, 0.0],
            [bx - hw, by + hh, bz, 0.0, 0.0, 0.0, 0.0, 0.0],
        ], dtype=np.float32)
        top_idx = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
        v_buf = self.ctx.buffer(top_verts.tobytes())
        i_buf = self.ctx.buffer(top_idx.tobytes())
        self.board_buffers = [v_buf, i_buf]
        self.board_vao = self.ctx.vertex_array(
            self.world_prog,
            [(v_buf, '3f 3f 2f', 'in_pos', 'in_norm', 'in_uv')],
            i_buf,
        )

    def _ensure_world_vaos(self):
        self._ensure_environment_buffers((-0.45, -0.08, 0.0))

    def _scene_world_mapping(self, model_mat, fabric_center, phys_span, grid_span=None):
        if model_mat is not None:
            center = transform_points(np.zeros((1, 3), dtype=np.float32), model_mat)[0].astype(np.float32)
        else:
            center = np.zeros(3, dtype=np.float32)
        if grid_span is not None and float(grid_span) > 0.0:
            span = float(grid_span)
        elif self.mesh_pick_data and model_mat is not None:
            pts = np.vstack([v[::max(1, len(v) // 80)] for v, _ in self.mesh_pick_data if len(v)])
            world_pts = transform_points(pts, model_mat)
            min_v = world_pts.min(axis=0)
            max_v = world_pts.max(axis=0)
            span = max(float(max_v[0] - min_v[0]), float(max_v[1] - min_v[1]), 2.0)
        else:
            span = 80.0
        units_per_meter = span / max(float(phys_span), 0.05)
        fc = np.asarray(fabric_center, dtype=np.float32).reshape(-1)[:3]
        return center, fc, float(units_per_meter)

    def _ur_to_gl_matrix(self, ur_mat4, scene_center, fabric_center, units_per_meter):
        m = np.eye(4, dtype=np.float32)
        m[:3, :3] = np.asarray(ur_mat4[:3, :3], dtype=np.float32) * float(units_per_meter)
        ur_pos = np.asarray(ur_mat4[:3, 3], dtype=np.float32)
        m[:3, 3] = scene_center + (ur_pos - fabric_center) * float(units_per_meter)
        return m

    def _draw_world_scene(
        self,
        view_proj,
        model_mat,
        world_scene,
        draw_robot=True,
        draw_frusta=True,
        draw_shadows=True,
        view=None,
        target_fbo=None,
    ):
        fabric_center = world_scene.get("fabric_center", (-0.45, -0.08, 0.0))
        fabric_size = world_scene.get("fabric_size")
        raw_phys_span = world_scene.get("phys_span")
        if raw_phys_span is not None:
            phys_span = float(raw_phys_span)
        elif isinstance(fabric_size, (int, float, np.floating, np.integer)):
            phys_span = float(fabric_size)
        elif isinstance(fabric_size, (tuple, list, np.ndarray)) and len(fabric_size) >= 2:
            phys_span = max(float(fabric_size[0]), float(fabric_size[1]))
        else:
            phys_span = 0.18
        grid_span = world_scene.get("grid_span")
        self._ensure_environment_buffers(fabric_center, fabric_size)
        scene_center, fc, units_per_m = self._scene_world_mapping(model_mat, fabric_center, phys_span, grid_span=grid_span)

        fbo = target_fbo if target_fbo is not None else self.fbo
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        if fbo is not None:
            fbo.depth_mask = True

        if view is not None:
            cam_pos = (-view[:3, :3].T @ view[:3, 3]).astype(np.float32)
        else:
            cam_pos = np.array([0.0, -50.0, 80.0], dtype=np.float32)

        self.world_prog['uLight'].value = tuple(float(v) for v in _WORLD_LIGHT_DIR)
        self.world_prog['uUp'].value = tuple(float(v) for v in _WORLD_UP)
        self.world_prog['uCamPos'].value = tuple(float(v) for v in cam_pos)
        self.world_prog['uShadow'].value = 0.0
        self.world_prog['uShadowAlpha'].value = 0.20

        env_model = self._ur_to_gl_matrix(np.eye(4, dtype=np.float32), scene_center, fc, units_per_m)
        env_mvp = (view_proj @ env_model).astype(np.float32)

        show_table = bool(world_scene.get("show_table", world_scene.get("show_board", True)))
        show_board = bool(world_scene.get("show_board", show_table))

        if show_table and self.room_vao is not None:
            self.world_prog['mvp'].write(env_mvp.T.tobytes())
            self.world_prog['model'].write(env_model.T.tobytes())
            self.world_prog['uUseTex'].value = 0.0
            self.world_prog['uUseVertexCol'].value = 1.0
            self.room_vao.render(moderngl.TRIANGLES, vertices=self.room_vertex_count)

        if show_board and self.board_vao is not None:
            self.world_prog['mvp'].write(env_mvp.T.tobytes())
            self.world_prog['model'].write(env_model.T.tobytes())
            if self.board_plinth_vao is not None:
                self.world_prog['uUseTex'].value = 0.0
                self.world_prog['uUseVertexCol'].value = 1.0
                self.board_plinth_vao.render(moderngl.TRIANGLES, vertices=self.board_plinth_vertex_count)
            if self.charuco_tex is not None:
                self.charuco_tex.use(location=0)
                self.world_prog['uTex'].value = 0
                self.world_prog['uUseTex'].value = 1.0
                self.world_prog['uUseVertexCol'].value = 0.0
                self.world_prog['base_color'].value = (1.0, 1.0, 1.0)
                self.board_vao.render(moderngl.TRIANGLES)
            self.world_prog['uUseTex'].value = 0.0
            self._draw_macro_fabric_sheet(view_proj, model_mat, world_scene, units_per_m=units_per_m)

        joints = world_scene.get("joints") if "joints" in world_scene else world_scene.get("robot_joints")
        if joints is not None and self.robot_vaos and world_scene.get("show_robot", True):
            link_mats = scan_geometry.link_world_transforms(joints)
            if draw_shadows and (show_table or show_board):
                ground_proj = _ground_projection_z(_WORLD_LIGHT_DIR, float(fc[2]) + 0.0005)
                self.ctx.enable(moderngl.BLEND)
                self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
                if fbo is not None:
                    fbo.depth_mask = False
                self.world_prog['uShadow'].value = 1.0
                self.world_prog['uUseTex'].value = 0.0
                self.world_prog['uUseVertexCol'].value = 0.0
                for vao, link_name, _color in self.robot_vaos:
                    link_mat = link_mats.get(link_name)
                    if link_mat is None:
                        continue
                    gl_shadow = self._ur_to_gl_matrix(ground_proj @ link_mat, scene_center, fc, units_per_m)
                    mvp_shadow = (view_proj @ gl_shadow).astype(np.float32)
                    self.world_prog['mvp'].write(mvp_shadow.T.tobytes())
                    self.world_prog['model'].write(gl_shadow.T.tobytes())
                    vao.render(moderngl.TRIANGLES)
                self.world_prog['uShadow'].value = 0.0
                if fbo is not None:
                    fbo.depth_mask = True
                self.ctx.disable(moderngl.BLEND)

            if draw_robot:
                self.world_prog['uShadow'].value = 0.0
                self.world_prog['uUseTex'].value = 0.0
                self.world_prog['uUseVertexCol'].value = 0.0
                for vao, link_name, color in self.robot_vaos:
                    link_mat = link_mats.get(link_name)
                    if link_mat is None:
                        continue
                    gl_model = self._ur_to_gl_matrix(link_mat, scene_center, fc, units_per_m)
                    mvp_link = (view_proj @ gl_model).astype(np.float32)
                    self.world_prog['mvp'].write(mvp_link.T.tobytes())
                    self.world_prog['model'].write(gl_model.T.tobytes())
                    self.world_prog['base_color'].value = color
                    vao.render(moderngl.TRIANGLES)

        if draw_frusta and world_scene.get("show_frusta", True):
            line_verts = []
            plan_poses = world_scene.get("capture_plan_poses")
            if plan_poses is None:
                plan_poses = []
            prev_pt = None
            for p in plan_poses:
                pose_arr = np.asarray(p, dtype=np.float32).reshape(-1)
                if pose_arr.size < 6:
                    continue
                pos_gl = scene_center + (pose_arr[:3] - fc) * units_per_m
                if prev_pt is not None:
                    line_verts.extend((*prev_pt, 0.30, 0.78, 0.95, *pos_gl, 0.30, 0.78, 0.95))
                prev_pt = pos_gl
                self._append_frustum_lines(line_verts, pose_arr, scene_center, fc, units_per_m, 0.028, (0.35, 0.75, 0.95))
            cam_pose = world_scene.get("camera_pose")
            if cam_pose is not None:
                pose_arr = np.asarray(cam_pose, dtype=np.float32).reshape(-1)
                if pose_arr.size >= 6:
                    self._append_frustum_lines(line_verts, pose_arr, scene_center, fc, units_per_m, 0.055, (1.0, 0.82, 0.22))
            if line_verts:
                arr = np.asarray(line_verts, dtype=np.float32)
                vbo = self.ctx.buffer(arr.tobytes())
                vao = self.ctx.vertex_array(self.line_prog, [(vbo, '3f 3f', 'in_pos', 'in_color')])
                self.line_prog['mvp'].write(view_proj.astype(np.float32).T.tobytes())
                vao.render(moderngl.LINES)
                vao.release()
                vbo.release()

    @staticmethod
    def _append_frustum_lines(out, pose_arr, scene_center, fc, units_per_m, depth_m, color):
        origin = scene_center + (pose_arr[:3] - fc) * units_per_m
        rot = scan_geometry.rotvec_to_matrix(pose_arr[3:6]).astype(np.float32)
        d = float(depth_m) * units_per_m
        hw = d * 0.52
        hh = d * 0.36
        corners_cam = np.array([
            [-hw, -hh, d],
            [ hw, -hh, d],
            [ hw,  hh, d],
            [-hw,  hh, d],
        ], dtype=np.float32)
        corners = origin + corners_cam @ rot.T
        r, g, b = color
        for c in corners:
            out.extend((*origin, r, g, b, *c, r, g, b))
        for i in range(4):
            c0, c1 = corners[i], corners[(i + 1) % 4]
            out.extend((*c0, r, g, b, *c1, r, g, b))

    def render_unit_cell_tile(
        self,
        prepared_meshes,
        colors,
        center_xy,
        x_period,
        y_period,
        material_uniforms=None,
        tile_h=128,
    ) -> np.ndarray:
        cx, cy = float(center_xy[0]), float(center_xy[1])
        hx = max(0.5 * abs(float(x_period)), 1e-3)
        hy = max(0.5 * abs(float(y_period)), 1e-3)
        th = max(64, int(tile_h))
        tw = max(64, min(512, int(round(th * (hx / hy)))))

        if self._tile_fbo is None or self._tile_fbo_size != (tw, th):
            for name in ('_tile_fbo', '_tile_color_tex', '_tile_depth_tex'):
                res = getattr(self, name, None)
                if res is not None:
                    res.release()
                    setattr(self, name, None)
            self._tile_color_tex = self.ctx.texture((tw, th), 3)
            self._tile_depth_tex = self.ctx.depth_texture((tw, th))
            self._tile_fbo = self.ctx.framebuffer(
                color_attachments=[self._tile_color_tex],
                depth_attachment=self._tile_depth_tex,
            )
            self._tile_fbo_size = (tw, th)

        if (
            isinstance(prepared_meshes, tuple)
            and len(prepared_meshes) >= 2
            and isinstance(prepared_meshes[0], (list, tuple))
            and (not prepared_meshes[0] or (isinstance(prepared_meshes[0][0], tuple) and len(prepared_meshes[0][0]) == 5))
            and not isinstance(prepared_meshes[0], np.ndarray)
        ):
            if colors is None and len(prepared_meshes) >= 2:
                colors = prepared_meshes[1]
            prepared_meshes = prepared_meshes[0]

        proj = orthographic(cx - hx, cx + hx, cy - hy, cy + hy, near=-100.0, far=100.0)
        view = np.eye(4, dtype=np.float32)
        mvp = (proj @ view).astype(np.float32)
        mat_uniforms = dict(material_uniforms) if material_uniforms is not None else {}

        temp_vaos = []
        temp_buffers = []
        self._tile_fbo.use()
        self.ctx.viewport = (0, 0, tw, th)
        self._tile_fbo.depth_mask = True
        self.ctx.color_mask = (True, True, True, True)
        self.ctx.clear(0.10, 0.10, 0.12, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.depth_func = '<'
        try:
            for i, (_v, _tris, pos_bytes, norm_bytes, tri_bytes) in enumerate(prepared_meshes or ()):
                pos_vbo = self.ctx.buffer(pos_bytes)
                norm_vbo = self.ctx.buffer(norm_bytes)
                ibo = self.ctx.buffer(tri_bytes)
                temp_buffers.extend((pos_vbo, norm_vbo, ibo))
                vao = self.ctx.vertex_array(
                    self.prog,
                    [(pos_vbo, '3f', 'in_pos'), (norm_vbo, '3f', 'in_norm')],
                    ibo,
                )
                depth_vao = self.ctx.vertex_array(
                    self.depth_prog,
                    [(pos_vbo, '3f', 'in_pos')],
                    ibo,
                )
                col = colors[i % len(colors)] if colors else (0.8, 0.2, 0.2)
                temp_vaos.append((vao, depth_vao, col))

            if temp_vaos:
                self.ctx.color_mask = (False, False, False, False)
                self.depth_prog['mvp'].write(mvp.T.tobytes())
                for _vao, depth_vao, _col in temp_vaos:
                    depth_vao.render(moderngl.TRIANGLES)

                self.ctx.color_mask = (True, True, True, True)
                self._tile_fbo.depth_mask = False
                self.ctx.depth_func = '<='
                self._tile_depth_tex.use(location=1)
                if 'depth_tex' in self.prog:
                    self.prog['depth_tex'].value = 1
                if 'viewport_size' in self.prog:
                    self.prog['viewport_size'].value = (float(tw), float(th))
                ao_radius = float(mat_uniforms.get('ao_radius', 0.15))
                if 'ao_uv_scale' in self.prog:
                    self.prog['ao_uv_scale'].value = (
                        float(ao_radius / (2.0 * hx)),
                        float(ao_radius / (2.0 * hy)),
                    )
                self.prog['mvp'].write(mvp.T.tobytes())
                self.prog['mv'].write(view.T.tobytes())
                if 'light_dir' in self.prog and 'light_dir' not in mat_uniforms:
                    self.prog['light_dir'].value = (0.5, 1.0, 0.8)
                if 'model_alpha' in self.prog and 'model_alpha' not in mat_uniforms:
                    self.prog['model_alpha'].value = 1.0
                self._set_program_uniforms(self.prog, mat_uniforms)
                for vao, _depth_vao, col in temp_vaos:
                    base_col = np.asarray(col, dtype=np.float32).reshape(-1)[:3]
                    self.prog['color'].value = (float(base_col[0]), float(base_col[1]), float(base_col[2]))
                    vao.render(moderngl.TRIANGLES)

            raw = self._tile_fbo.read(components=3, dtype='f1')
            rgb = np.ascontiguousarray(np.flipud(np.frombuffer(raw, dtype=np.uint8).reshape((th, tw, 3))))
            return rgb
        finally:
            self.ctx.color_mask = (True, True, True, True)
            self.ctx.depth_func = '<'
            self._tile_fbo.depth_mask = True
            for vao, depth_vao, _col in temp_vaos:
                vao.release()
                depth_vao.release()
            for buf in temp_buffers:
                buf.release()
            if self.ctx.screen is not None:
                self.ctx.screen.use()

    def _draw_macro_fabric_sheet(self, view_proj, model_mat, world_scene, units_per_m=None):
        if not isinstance(world_scene, dict):
            return
        tile_images = world_scene.get("swatch_tile_images")
        swatch_slots = world_scene.get("swatch_slots")
        if not tile_images or not swatch_slots:
            return
        tiles_epoch = int(world_scene.get("swatch_tiles_epoch", 0))
        swatch_repeats = world_scene.get("swatch_repeats", (24, 24))
        rep_r = max(1, int(swatch_repeats[0]) if len(swatch_repeats) > 0 else 24)
        rep_c = max(1, int(swatch_repeats[1]) if len(swatch_repeats) > 1 else 24)

        if tiles_epoch != self._swatch_tiles_epoch or len(self._swatch_tile_textures) != len(tile_images):
            for tex in self._swatch_tile_textures:
                tex.release()
            self._swatch_tile_textures = []
            for rgb in tile_images:
                arr = np.ascontiguousarray(np.flipud(np.asarray(rgb, dtype=np.uint8)))
                if arr.ndim != 3 or arr.shape[2] < 3:
                    continue
                arr = np.ascontiguousarray(arr[:, :, :3])
                h, w = int(arr.shape[0]), int(arr.shape[1])
                tex = self.ctx.texture((w, h), 3, arr.tobytes())
                tex.repeat_x = True
                tex.repeat_y = True
                tex.build_mipmaps()
                tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
                self._swatch_tile_textures.append(tex)
            self._swatch_tiles_epoch = tiles_epoch

        if not self._swatch_tile_textures:
            return

        if units_per_m is None:
            fabric_center = world_scene.get("fabric_center", (-0.45, -0.08, 0.0))
            fabric_size = world_scene.get("fabric_size")
            raw_phys_span = world_scene.get("phys_span")
            if raw_phys_span is not None:
                phys_span = float(raw_phys_span)
            elif isinstance(fabric_size, (int, float, np.floating, np.integer)):
                phys_span = float(fabric_size)
            elif isinstance(fabric_size, (tuple, list, np.ndarray)) and len(fabric_size) >= 2:
                phys_span = max(float(fabric_size[0]), float(fabric_size[1]))
            else:
                phys_span = 0.18
            _, _, units_per_m = self._scene_world_mapping(
                model_mat, fabric_center, phys_span, grid_span=world_scene.get("grid_span")
            )

        mat = np.eye(4, dtype=np.float32) if model_mat is None else np.asarray(model_mat, dtype=np.float32)
        mvp = (view_proj @ mat).astype(np.float32)
        z_lift = 0.0015 * float(units_per_m)
        u_max = float(rep_c)
        v_max = float(rep_r)

        self.world_prog['mvp'].write(mvp.T.tobytes())
        self.world_prog['model'].write(mat.T.tobytes())
        self.world_prog['uTex'].value = 0
        self.world_prog['uUseTex'].value = 1.0
        self.world_prog['uUseVertexCol'].value = 0.0
        self.world_prog['base_color'].value = (1.0, 1.0, 1.0)

        quad_idx = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
        i_buf = self.ctx.buffer(quad_idx.tobytes())
        try:
            for idx, (x0, y0, x1, y1) in enumerate(swatch_slots):
                x0f, y0f, x1f, y1f = float(x0), float(y0), float(x1), float(y1)
                quad_verts = np.array([
                    [x0f, y0f, z_lift, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [x1f, y0f, z_lift, 0.0, 0.0, 1.0, u_max, 0.0],
                    [x1f, y1f, z_lift, 0.0, 0.0, 1.0, u_max, v_max],
                    [x0f, y1f, z_lift, 0.0, 0.0, 1.0, 0.0, v_max],
                ], dtype=np.float32)
                v_buf = self.ctx.buffer(quad_verts.tobytes())
                vao = self.ctx.vertex_array(
                    self.world_prog,
                    [(v_buf, '3f 3f 2f', 'in_pos', 'in_norm', 'in_uv')],
                    i_buf,
                )
                try:
                    tex = self._swatch_tile_textures[idx % len(self._swatch_tile_textures)]
                    tex.use(location=0)
                    vao.render(moderngl.TRIANGLES)
                finally:
                    vao.release()
                    v_buf.release()
        finally:
            i_buf.release()
            self.world_prog['uUseTex'].value = 0.0

    def render_wrist_camera(
        self,
        world_pose,
        joints_or_hfov=None,
        fabric_center=(-0.45, -0.08, 0.0),
        phys_span=None,
        out_w=None,
        out_h=None,
        material_uniforms=None,
        visible_rows=None,
        model_mat=None,
        intrinsics=None,
        *,
        width=None,
        height=None,
        fabric_size=(0.28, 0.19),
        dist_coeffs=None,
        grid_span=None,
        hfov_deg=55.0,
        world_scene=None,
    ):
        if isinstance(joints_or_hfov, (int, float, np.floating, np.integer)):
            hfov_deg = float(joints_or_hfov)
            joints = None
        else:
            joints = joints_or_hfov
        out_w = max(16, int(width if width is not None else (out_w if out_w is not None else 640)))
        out_h = max(16, int(height if height is not None else (out_h if out_h is not None else 480)))
        if phys_span is None:
            if isinstance(fabric_size, (int, float, np.floating, np.integer)):
                phys_span = float(fabric_size)
            elif isinstance(fabric_size, (tuple, list, np.ndarray)) and len(fabric_size) >= 2:
                phys_span = max(float(fabric_size[0]), float(fabric_size[1]))
            else:
                phys_span = 0.18
        if material_uniforms is None:
            material_uniforms = {}

        if self.wrist_fbo is None or self.wrist_size != (out_w, out_h):
            for name in ('wrist_fbo', 'wrist_color_tex', 'wrist_depth_tex'):
                res = getattr(self, name, None)
                if res is not None:
                    res.release()
            self.wrist_color_tex = self.ctx.texture((out_w, out_h), 3)
            self.wrist_depth_tex = self.ctx.depth_texture((out_w, out_h))
            self.wrist_fbo = self.ctx.framebuffer(
                color_attachments=[self.wrist_color_tex],
                depth_attachment=self.wrist_depth_tex,
            )
            self.wrist_size = (out_w, out_h)

        if model_mat is None:
            model_mat = np.eye(4, dtype=np.float32)
        scene_center, fc, units_per_m = self._scene_world_mapping(model_mat, fabric_center, phys_span, grid_span=grid_span)
        pose_arr = np.asarray(world_pose, dtype=np.float32).reshape(-1)[:6]
        eye_gl = scene_center + (pose_arr[:3] - fc) * units_per_m
        rot_wc = scan_geometry.rotvec_to_matrix(pose_arr[3:6]).astype(np.float32)
        right = rot_wc[:, 0]
        up = -rot_wc[:, 1]
        forward = rot_wc[:, 2]
        view = np.eye(4, dtype=np.float32)
        view[0, :3] = right
        view[1, :3] = up
        view[2, :3] = -forward
        view[0, 3] = -float(np.dot(right, eye_gl))
        view[1, 3] = -float(np.dot(up, eye_gl))
        view[2, 3] = float(np.dot(forward, eye_gl))

        if intrinsics is None:
            import hand_eye
            measured = hand_eye.load_intrinsics()
            if measured is not None:
                intrinsics = hand_eye.intrinsics_for(out_w, out_h, measured)
        elif isinstance(intrinsics, dict) and "fx" in intrinsics and "fl_x" not in intrinsics:
            import hand_eye
            intrinsics = hand_eye.intrinsics_for(out_w, out_h, intrinsics)

        K_mat, D_vec = _resolve_k_and_d(intrinsics, dist_coeffs, out_w, out_h, hfov_deg)
        has_distortion = bool(np.any(np.abs(D_vec) > 1e-7))
        if has_distortion:
            K_render, remap_xy = distortion_map(K_mat, D_vec, (out_w, out_h))
        else:
            K_render, remap_xy = K_mat, None

        fl_x = float(K_render[0, 0])
        fl_y = float(K_render[1, 1])
        cx = float(K_render[0, 2])
        cy = float(K_render[1, 2])
        near = max(0.05, 0.005 * float(units_per_m))
        far = max(2500.0, 12.0 * float(units_per_m))
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = 2.0 * fl_x / float(out_w)
        proj[0, 2] = 1.0 - 2.0 * cx / float(out_w)
        proj[1, 1] = 2.0 * fl_y / float(out_h)
        proj[1, 2] = 2.0 * cy / float(out_h) - 1.0
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = (2.0 * far * near) / (near - far)
        proj[3, 2] = -1.0

        view_proj = proj @ view
        mvp = (view_proj @ model_mat).astype(np.float32)
        mv = (view @ model_mat).astype(np.float32)

        self.wrist_fbo.use()
        self.ctx.viewport = (0, 0, out_w, out_h)
        self.wrist_fbo.depth_mask = True
        self.ctx.clear(0.18, 0.19, 0.22, 1.0)
        self.wrist_fbo.depth_mask = False
        self._draw_sky(view, proj, out_h)
        self.wrist_fbo.depth_mask = True
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_func = '<'

        ws = {
            "fabric_center": fabric_center,
            "fabric_size": fabric_size,
            "phys_span": phys_span,
            "grid_span": grid_span,
            "joints": joints,
            "show_board": True,
            "show_table": True,
            "show_robot": joints is not None,
        }
        for src in (material_uniforms, world_scene):
            if isinstance(src, dict):
                for k in (
                    "swatch_tile_images",
                    "swatch_tiles_epoch",
                    "swatch_slots",
                    "swatch_repeats",
                    "hide_unit_cell_mesh",
                ):
                    if k in src and src[k] is not None:
                        ws[k] = src[k]

        self._draw_world_scene(
            view_proj,
            model_mat,
            ws,
            draw_robot=False,
            draw_frusta=False,
            draw_shadows=joints is not None,
            view=view,
            target_fbo=self.wrist_fbo,
        )

        if not bool(ws.get("hide_unit_cell_mesh", False)):
            self.ctx.enable(moderngl.CULL_FACE)
            self.wrist_depth_tex.use(location=1)
            if 'depth_tex' in self.prog:
                self.prog['depth_tex'].value = 1
            if 'viewport_size' in self.prog:
                self.prog['viewport_size'].value = (float(out_w), float(out_h))
            if 'ao_uv_scale' in self.prog:
                self.prog['ao_uv_scale'].value = (0.01, 0.01)
            self.prog['mvp'].write(mvp.T.tobytes())
            self.prog['mv'].write(mv.T.tobytes())
            if 'light_dir' in self.prog and 'light_dir' not in material_uniforms:
                self.prog['light_dir'].value = (0.5, 1.0, 0.8)
            self._set_program_uniforms(self.prog, material_uniforms)
            for vao, _, _, _, color, row_idx in self.meshes:
                if not self._row_visible(row_idx, visible_rows):
                    continue
                self.prog['color'].value = tuple(float(c) for c in np.asarray(color, dtype=np.float32))
                vao.render(moderngl.TRIANGLES)

        raw = self.wrist_fbo.read(components=3, dtype='f1')
        if self.ctx.screen is not None:
            self.ctx.screen.use()
        img = np.ascontiguousarray(np.flipud(np.frombuffer(raw, dtype=np.uint8).reshape((out_h, out_w, 3))))
        if remap_xy is not None:
            img = cv2.remap(img, remap_xy[0], remap_xy[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return img

# %% IMAGE TEXTURE HELPER ─────────────────────────────────────────────────────

def distortion_map(
    K: np.ndarray,
    D: np.ndarray,
    size: tuple[int, int],
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    import cv2

    w, h = int(size[0]), int(size[1])
    K_arr = np.asarray(K, dtype=np.float64).reshape(3, 3)
    D_arr = np.asarray(D, dtype=np.float64).reshape(-1)
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    rays = cv2.undistortPoints(
        np.stack([u.ravel(), v.ravel()], axis=1).reshape(-1, 1, 2),
        K_arr,
        D_arr,
    ).reshape(-1, 2)
    focal = float(min(
        w * 0.5 / max(float(np.abs(rays[:, 0]).max()), 1e-6),
        h * 0.5 / max(float(np.abs(rays[:, 1]).max()), 1e-6),
    ))
    K_render = np.array(
        [[focal, 0.0, w * 0.5], [0.0, focal, h * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    xs = (rays[:, 0] * focal + w * 0.5).reshape(h, w).astype(np.float32)
    ys = (rays[:, 1] * focal + h * 0.5).reshape(h, w).astype(np.float32)
    return K_render, (xs, ys)


def _resolve_k_and_d(
    intrinsics: dict[str, float] | tuple[float, ...] | np.ndarray | None,
    dist_coeffs: tuple[float, ...] | list[float] | np.ndarray | None,
    width: int,
    height: int,
    hfov_deg: float = 55.0,
) -> tuple[np.ndarray, np.ndarray]:
    d_from_intr = None
    if isinstance(intrinsics, np.ndarray) and intrinsics.shape == (3, 3):
        K = intrinsics.astype(np.float64)
    elif isinstance(intrinsics, (tuple, list, np.ndarray)) and len(intrinsics) >= 4:
        fx, fy, cx, cy = (float(v) for v in intrinsics[:4])
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    elif isinstance(intrinsics, dict):
        default_fx = 0.5 * width / max(math.tan(math.radians(max(10.0, min(120.0, float(hfov_deg)))) * 0.5), 1e-4)
        fx = float(intrinsics.get("fl_x", intrinsics.get("fx", default_fx)))
        fy = float(intrinsics.get("fl_y", intrinsics.get("fy", fx)))
        cx = float(intrinsics.get("cx", width * 0.5))
        cy = float(intrinsics.get("cy", height * 0.5))
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        d_from_intr = np.array(
            [float(intrinsics.get(k, 0.0)) for k in ("k1", "k2", "p1", "p2", "k3")],
            dtype=np.float64,
        )
    else:
        fx = 0.5 * width / max(math.tan(math.radians(max(10.0, min(120.0, float(hfov_deg)))) * 0.5), 1e-4)
        K = np.array([[fx, 0.0, width * 0.5], [0.0, fx, height * 0.5], [0.0, 0.0, 1.0]], dtype=np.float64)

    if dist_coeffs is not None:
        D = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    elif d_from_intr is not None:
        D = d_from_intr
    else:
        D = np.zeros(5, dtype=np.float64)
    return K, D


def pil_to_texture(ctx, pil_img):
    """Upload a PIL image as a moderngl texture (RGBA, Y-flipped for GL)."""
    img  = pil_img.convert("RGBA").transpose(Image.FLIP_TOP_BOTTOM)
    w, h = img.size
    tex  = ctx.texture((w, h), 4, img.tobytes())
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    return tex


def project_to_screen(world_pts, view_proj, disp_w, disp_h):
    """World points to viewport pixels, with the behind-camera points flagged.

    Returns `(screen, valid, ndc)`. `valid` marks the points in front of the
    camera; the others hold zeros in `ndc` and meaningless values in `screen`,
    so every caller must mask with it. `ndc` comes back because hover testing
    needs it to reject points outside the frustum, not just behind it.
    """
    world_pts = np.asarray(world_pts, dtype=np.float32)
    homo = np.column_stack((world_pts, np.ones(len(world_pts), dtype=np.float32)))
    clip = homo @ view_proj.T
    valid = clip[:, 3] > 1e-6
    ndc = np.zeros((len(world_pts), 3), dtype=np.float32)
    ndc[valid] = clip[valid, :3] / clip[valid, 3:4]
    screen = np.column_stack((
        (ndc[:, 0] * 0.5 + 0.5) * disp_w,
        (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * disp_h,
    ))
    return screen, valid, ndc


def upload_rgb_texture(ctx, texture, rgb):
    """Upload an HxWx3 uint8 array into `texture`, reallocating if the size changed.

    Sibling to pil_to_texture rather than a replacement: that one takes a PIL
    image and always allocates, while the live views here re-upload a numpy
    frame many times a second and want to keep the texture they have.

    Returns the texture to assign back, which may be a new object. Callers each
    had their own copy of this and had drifted to different levels of
    correctness -- one never set the filter, another never handled a size change
    and would write new bytes into a stale-sized texture.
    """
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    height, width = rgb.shape[:2]
    rgba = np.dstack((rgb, np.full((height, width), 255, dtype=np.uint8)))
    # Flipped because GL's texture origin is bottom-left and image rows arrive
    # top-first.
    rgba = np.ascontiguousarray(np.flipud(rgba))
    if texture is None or tuple(texture.size) != (width, height):
        if texture is not None:
            try:
                texture.release()
            except Exception as exc:
                logging.warning("rendering.upload_rgb_texture: %s", exc)
        texture = ctx.texture((width, height), 4, rgba.tobytes())
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
    else:
        texture.write(rgba.tobytes())
    return texture


def draw_fitted_texture(texture_id, tex_w, tex_h, avail_w, avail_h, flip_y=True, zoom=1.0, pan=(0.0, 0.0)):
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
