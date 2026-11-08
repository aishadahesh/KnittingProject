"""Gaussian splats drawn on the GPU, fast enough to follow the mouse while a view is turned.

The same picture `gaussian_splatting.render_splat` draws on the CPU, but every frame: each Gaussian is
a screen-facing quad whose shape is its 3D covariance carried through the camera (EWA splatting, as
the trainer draws them), and the quads are blended back to front. The Gaussians' numbers live in a
float texture uploaded once per splat; a frame only sorts them by depth and uploads the order.

Styles (STYLES): "Soft splat" is the picture as trained; "Gaussian ellipses" draws every Gaussian
as a solid ellipse out to one standard deviation with a dark rim, so each one, its size, stretch and
turn can be seen; "Gaussian centres" draws only their middles, as dots - how many there are and where;
"Soft splat + Gaussian outlines" is the soft splat with every Gaussian's one-sigma outline drawn into
it, so the picture and the Gaussians that make it are seen at once. The outlines are part of each
Gaussian's own blend, so the ones behind are hidden by the cloth in front, as the Gaussians are.
"Gaussian axes" draws each Gaussian's longest axis as a 3D line segment (its middle +- AXIS_SIGMAS
standard deviations along that axis), carried through the camera and depth-tested like any 3D
geometry, and coloured by the way it points: red along the cloth's x, green along its y, blue
standing up out of it. Round Gaussians (the rebuilt yarn's) have no longest axis: their centres.
"Soft splat + Gaussian axes" draws the axes on the soft splat. The splat's solid cores (where a
Gaussian is more than half opaque) are first written into the depth buffer, so an axis behind the
cloth stays hidden and only the ones on the surface facing the camera show; each axis is drawn
AXIS_LIFT of the view's extent toward the eye, so its own Gaussian's core does not hide it.

Colour by height: each Gaussian coloured by how far it stands above or below the cloth's smooth bow
(blue low, red high) - a flat picture would come out one colour, a real 3D model shows its yarn's relief.
"""

from __future__ import annotations

import math

import moderngl
import numpy as np

import gaussian_splatting

STYLES = ("Soft splat", "Gaussian ellipses", "Gaussian centres", "Soft splat + Gaussian outlines", "Gaussian axes",
          "Soft splat + Gaussian axes")
SOFT, ELLIPSES, CENTRES, OUTLINED, AXES, SPLAT_AXES = range(len(STYLES))
DEPTH_ONLY = 9                            # the shader's style for the cores' depth pass (not a STYLES entry)
IDS = 10                                  # each pixel the number of the Gaussian in front there (gaussian_ids)
AXIS_LIFT = 0.015
AXIS_SIGMAS = 1.5
AXIS_MIN_OPACITY = 0.1
OUTLINE_COLOUR = (1.0, 0.55, 0.10)
TEXELS_PER_SPLAT = 4
DATA_WIDTH = 4096                     # texels per row of the data texture
NEAR = 1e-4
MAX_RADIUS_PX = 1024.0
# The CPU renderer's lens: 50 degrees across.
HALF_FOV_DEG = 25.0

_VERTEX = """
#version 330
uniform sampler2D data;
uniform mat3 world_to_cam;
uniform vec3 eye;
uniform float focal;
uniform vec2 size;
uniform int style;
uniform vec2 depth_range;
in vec2 corner;
in uint splat_index;
out vec3 v_colour;
out float v_opacity;
out vec2 v_local;

vec4 fetch(int k) {
    int texel = int(splat_index) * 4 + k;
    return texelFetch(data, ivec2(texel % 4096, texel / 4096), 0);
}

void main() {
    vec4 pos_opacity = fetch(0), colour = fetch(1), cov_a = fetch(2), cov_b = fetch(3);
    vec3 c = world_to_cam * (pos_opacity.xyz - eye);
    if (c.z < 1e-4) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);          // behind the eye: clipped away
        return;
    }
    mat3 sigma = mat3(cov_a.x, cov_a.y, cov_a.z,
                      cov_a.y, cov_b.x, cov_b.y,
                      cov_a.z, cov_b.y, cov_b.z);
    float f = focal / c.z;
    mat3 jac = mat3(f, 0.0, 0.0,
                    0.0, f, 0.0,
                    -f * c.x / c.z, -f * c.y / c.z, 0.0);
    mat3 t = jac * world_to_cam;
    mat3 cov = t * sigma * transpose(t);
    float a = cov[0][0] + 0.3, b = cov[0][1], d = cov[1][1] + 0.3;
    float mid = 0.5 * (a + d), spread = sqrt(max(0.25 * (a - d) * (a - d) + b * b, 0.0));
    float big = mid + spread, small = max(mid - spread, 0.1);
    vec2 major = abs(b) > 1e-6 ? normalize(vec2(b, big - a)) : (a >= d ? vec2(1.0, 0.0) : vec2(0.0, 1.0));
    vec2 minor = vec2(-major.y, major.x);
    vec2 reach = vec2(min(3.0 * sqrt(big), 1024.0), min(3.0 * sqrt(small), 1024.0));
    if (style == 2) {
        reach = vec2(2.0);                                // a centre: a dot a few pixels wide
        major = vec2(1.0, 0.0);
        minor = vec2(0.0, 1.0);
    }
    vec2 pixel = vec2(c.x, c.y) * f + 0.5 * size + corner.x * reach.x * major + corner.y * reach.y * minor;
    float z = (c.z - depth_range.x) / (depth_range.y - depth_range.x) * 2.0 - 1.0;
    gl_Position = vec4(2.0 * pixel.x / size.x - 1.0, 1.0 - 2.0 * pixel.y / size.y, z, 1.0);
    v_colour = colour.rgb;
    v_opacity = pos_opacity.w;
    v_local = 3.0 * corner;                               // in standard deviations along the axes
}
"""

_FRAGMENT = """
#version 330
in vec3 v_colour;
in float v_opacity;
in vec2 v_local;
uniform int style;
out vec4 frag;
void main() {
    float r2 = dot(v_local, v_local);
    if (style == 2) {                                     // a centre: a round opaque dot
        if (r2 > 9.0 || v_opacity < 0.1) discard;
        frag = vec4(v_colour, 1.0);
        return;
    }
    if (style == 10) {                                    // the Gaussian's number, exactly: its core, depth-tested
        if (r2 > 1.0 || v_opacity < 0.1) discard;
        frag = vec4(v_colour, 1.0);
        return;
    }
    if (style == 9) {                                     // depth only: the solid core of each Gaussian
        if (v_opacity * exp(-0.5 * r2) < 0.5) discard;
        frag = vec4(0.0);
        return;
    }
    if (style == 1) {                                     // an ellipse to one sigma, rimmed, lit like a lens
        if (r2 > 1.0 || v_opacity < 0.1) discard;
        float shade = r2 > 0.75 ? 0.3 : 0.7 + 0.3 * sqrt(1.0 - r2);
        frag = vec4(v_colour * shade, 1.0);
        return;
    }
    float alpha = min(v_opacity * exp(-0.5 * r2), 0.99);
    vec3 colour = v_colour;
    if (style == 3 && v_opacity >= 0.1) {                 // the one-sigma outline, about a pixel wide
        float r = sqrt(r2);
        float line = 1.0 - smoothstep(0.0, 1.2 * fwidth(r), abs(r - 1.0));
        line *= clamp((0.3 - fwidth(r)) / 0.2, 0.0, 1.0);    // faded on Gaussians a few pixels wide
        colour = mix(colour, vec3(1.0, 0.55, 0.10), line);
        alpha = max(alpha, 0.9 * line * v_opacity);
    }
    if (alpha < 1.0 / 255.0) discard;
    frag = vec4(colour * alpha, alpha);                   // premultiplied, blended back to front
}
"""

_LINE_VERTEX = """
#version 330
uniform mat3 world_to_cam;
uniform vec3 eye;
uniform float focal;
uniform vec2 size;
uniform vec2 depth_range;
uniform float lift;
in vec3 in_pos;
in vec3 in_colour;
out vec3 v_colour;
void main() {
    vec3 c = world_to_cam * (in_pos - eye);
    c.z -= lift;                                          // a little toward the eye, over its own Gaussian
    float z = (c.z - depth_range.x) / (depth_range.y - depth_range.x) * 2.0 - 1.0;
    // The same lens as the splat's quads, as clip coordinates (w = depth) so lines clip at the eye.
    gl_Position = vec4(2.0 * c.x * focal / size.x, -2.0 * c.y * focal / size.y, z * c.z, c.z);
    v_colour = in_colour;
}
"""

_LINE_FRAGMENT = """
#version 330
in vec3 v_colour;
out vec4 frag;
void main() { frag = vec4(v_colour, 1.0); }
"""

_COMPOSITE_VERTEX = """
#version 330
in vec2 corner;
out vec2 uv;
void main() { uv = 0.5 * corner + 0.5; gl_Position = vec4(corner, 0.0, 1.0); }
"""

_COMPOSITE_FRAGMENT = """
#version 330
uniform sampler2D layer;
uniform float strength;
in vec2 uv;
out vec4 frag;
void main() { frag = texture(layer, uv) * strength; }
"""


def _bow_terms(points, centre, axes):
    u, v = ((points - centre) @ axes[0]), ((points - centre) @ axes[1])
    return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], axis=1)


def cloth_plane(points: np.ndarray) -> tuple:
    """The cloth's lie: its middle, its in-plane axes and normal (the direction of least spread), the
    smooth bow it sags in (a quadratic over the plane), and the heights above that bow the colour
    ramp runs between (2nd to 98th percentile). Heights are measured from the bow, so the ramp shows
    the yarn's own relief, not the whole cloth's curl."""
    points = np.asarray(points, dtype=np.float64)
    centre = np.median(points, axis=0)
    sample = points[:: max(1, len(points) // 20000)]
    _, _, axes = np.linalg.svd(sample - centre, full_matrices=False)
    if axes[2][2] < 0:                    # laid-flat splats keep +z toward the cameras
        axes = -axes
    bow = np.linalg.lstsq(_bow_terms(sample, centre, axes), (sample - centre) @ axes[2], rcond=None)[0]
    heights = (points - centre) @ axes[2] - _bow_terms(points, centre, axes) @ bow
    low, high = np.percentile(heights, [2.0, 98.0])
    return centre, axes, bow, float(low), float(max(high, low + 1e-9))


def heights(points: np.ndarray, plane) -> np.ndarray:
    centre, axes, bow, _, _ = plane
    points = np.asarray(points, dtype=np.float64)
    return (points - centre) @ axes[2] - _bow_terms(points, centre, axes) @ bow


def height_colours(points: np.ndarray, plane) -> np.ndarray:
    """Blue (low) through cyan, green and yellow to red (high) by height above the cloth's bow."""
    low, high = plane[3], plane[4]
    t = np.clip((heights(points, plane) - low) / (high - low), 0.0, 1.0)
    stops = np.array([[0.10, 0.20, 0.90], [0.10, 0.80, 0.95], [0.20, 0.85, 0.30], [0.95, 0.85, 0.20], [0.90, 0.20, 0.15]])
    position = t * (len(stops) - 1)
    index = np.minimum(position.astype(int), len(stops) - 2)
    share = (position - index)[:, None]
    return (stops[index] * (1.0 - share) + stops[index + 1] * share).astype(np.float32)


def splat_covariances(splat: dict) -> np.ndarray:
    """Each Gaussian's 3D covariance as (n, 3, 3): from its scales and rotation, or round (sigma) when
    the splat has none (the rebuilt yarn tubes)."""
    count = len(splat["positions"])
    if "scales" in splat and len(splat["scales"]) == count and float(np.max(splat["scales"], initial=0.0)) > 0.0:
        return gaussian_splatting._covariances(splat["scales"].astype(np.float64), splat["quaternion"].astype(np.float64))
    sigma = np.asarray(splat.get("sigma", splat["radius"]), dtype=np.float64)
    return sigma[:, None, None] ** 2 * np.eye(3)[None]


def long_axes(splat: dict) -> np.ndarray:
    """Each Gaussian's longest axis as a segment: (n, 2, 6) float32, the two ends' positions and the
    direction colour (|x|, |y|, |z| of the axis). Empty for round Gaussians, which have none."""
    count = len(splat["positions"])
    if not ("scales" in splat and len(splat["scales"]) == count and float(np.max(splat["scales"], initial=0.0)) > 0.0):
        return np.zeros((0, 2, 6), dtype=np.float32)
    keep = np.asarray(splat["opacity"]) >= AXIS_MIN_OPACITY
    scales = np.asarray(splat["scales"], dtype=np.float64)[keep]
    q = np.asarray(splat["quaternion"], dtype=np.float64)[keep]
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
    w, x, y, z = q.T
    rotation = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1),
    ], axis=1)
    longest = np.argmax(scales, axis=1)
    rows = np.arange(len(scales))
    axis = rotation[rows, :, longest]                       # the rotation's column for that scale
    half = axis * (AXIS_SIGMAS * scales[rows, longest])[:, None]
    middle = np.asarray(splat["positions"], dtype=np.float64)[keep]
    colour = np.clip(np.abs(axis) * 1.1, 0.0, 1.0)
    segments = np.zeros((len(middle), 2, 6), dtype=np.float32)
    segments[:, 0, :3], segments[:, 1, :3] = middle - half, middle + half
    segments[:, :, 3:] = colour[:, None, :]
    return segments


def pack(splat: dict, colours: np.ndarray | None = None, min_opacity: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """The data texture's texels (TEXELS_PER_SPLAT RGBA per Gaussian, rows of DATA_WIDTH) and the kept
    Gaussians' positions (for the depth sort)."""
    keep = np.asarray(splat["opacity"]) >= min_opacity
    positions = np.asarray(splat["positions"], dtype=np.float32)[keep]
    colours = np.asarray(splat["colours"] if colours is None else colours, dtype=np.float32)[keep]
    cov = splat_covariances(splat)[keep]
    count = len(positions)
    texels = np.zeros((count, TEXELS_PER_SPLAT, 4), dtype=np.float32)
    texels[:, 0, :3], texels[:, 0, 3] = positions, np.asarray(splat["opacity"], dtype=np.float32)[keep]
    texels[:, 1, :3] = colours
    texels[:, 2, :3] = cov[:, 0, [0, 1, 2]]
    texels[:, 3, :3] = cov[:, [1, 1, 2], [1, 2, 2]]
    flat = texels.reshape(-1, 4)
    rows = max(1, math.ceil(len(flat) / DATA_WIDTH))
    padded = np.zeros((rows * DATA_WIDTH, 4), dtype=np.float32)
    padded[: len(flat)] = flat
    return padded.reshape(rows, DATA_WIDTH, 4), positions


class _Uploaded:
    def __init__(self, ctx, texels, positions, segments):
        self.texture = ctx.texture((DATA_WIDTH, texels.shape[0]), 4, texels.tobytes(), dtype="f4")
        self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.positions = positions.astype(np.float64)
        self.order = ctx.buffer(reserve=max(4, 4 * len(positions)))
        self.segments = len(segments)
        self.lines = ctx.buffer(segments.tobytes()) if len(segments) else None

    def release(self):
        self.texture.release()
        self.order.release()
        if self.lines is not None:
            self.lines.release()


class SplatRenderer:
    """Draws splats into an off-screen picture the GUI shows with draw_fitted_texture(..., flip_y=True)."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.program = ctx.program(vertex_shader=_VERTEX, fragment_shader=_FRAGMENT)
        self.line_program = ctx.program(vertex_shader=_LINE_VERTEX, fragment_shader=_LINE_FRAGMENT)
        self.composite = ctx.program(vertex_shader=_COMPOSITE_VERTEX, fragment_shader=_COMPOSITE_FRAGMENT)
        self.corners = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32).tobytes())
        self.quad = ctx.vertex_array(self.composite, [(self.corners, "2f", "corner")])
        self.uploads: dict = {}           # key -> (_Uploaded, quads' vertex array, axes' vertex array or None)
        self.targets: dict = {}           # picture name -> its framebuffer
        self.layer = None                 # a see-through layer's own picture, before it is laid over

    def _upload(self, key, splat, colours):
        if key not in self.uploads:
            texels, positions = pack(splat, colours() if callable(colours) else colours)
            data = _Uploaded(self.ctx, texels, positions, long_axes(splat))
            vao = self.ctx.vertex_array(self.program, [(self.corners, "2f", "corner"), (data.order, "u /i", "splat_index")])
            lines = (self.ctx.vertex_array(self.line_program, [(data.lines, "3f 3f", "in_pos", "in_colour")])
                     if data.lines is not None else None)
            self.uploads[key] = (data, vao, lines)
        return self.uploads[key]

    def forget(self, keep=()):
        """Frees every uploaded splat but those keyed in `keep`."""
        for key in [k for k in self.uploads if k not in keep]:
            data, vao, lines = self.uploads.pop(key)
            vao.release()
            if lines is not None:
                lines.release()
            data.release()

    def _framebuffer(self, old, size):
        if old is not None and old.size == size:
            return old
        if old is not None:
            old.color_attachments[0].release()
            old.depth_attachment.release()
            old.release()
        # The depth buffer is for the axes, which are drawn as 3D lines; the splats blend without it.
        fbo = self.ctx.framebuffer(color_attachments=[self.ctx.texture(size, 4)], depth_attachment=self.ctx.depth_renderbuffer(size))
        fbo.color_attachments[0].filter = (moderngl.LINEAR, moderngl.LINEAR)
        return fbo

    def _draw(self, key, splat, colours, eye, forward, right, up, size, style, depth_range, extent, focal):
        data, vao, lines = self._upload(key, splat, colours)
        if not len(data.positions):
            return
        world_to_cam = np.stack([right, -up, forward]).astype(np.float32)
        for program in (self.program, self.line_program):
            program["world_to_cam"].write(np.ascontiguousarray(world_to_cam.T).tobytes())
            program["eye"].value = tuple(float(v) for v in eye)
            program["focal"].value = focal
            program["size"].value = (float(size[0]), float(size[1]))
            program["depth_range"].value = depth_range

        def quads(style):
            self.program["style"].value = int(style)
            vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=len(data.positions))

        def axes(lift):
            self.line_program["lift"].value = float(lift)
            self.ctx.disable(moderngl.BLEND)
            self.ctx.enable(moderngl.DEPTH_TEST)
            lines.render(moderngl.LINES, vertices=2 * data.segments)
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.ctx.enable(moderngl.BLEND)

        if style == AXES and lines is not None:
            axes(0.0)
            return
        if style == IDS:                  # opaque cores, the nearest wins: no sort, no blend
            data.texture.use(0)
            self.program["data"].value = 0
            data.order.write(np.arange(len(data.positions), dtype=np.uint32).tobytes())
            self.ctx.disable(moderngl.BLEND)
            self.ctx.enable(moderngl.DEPTH_TEST)
            quads(IDS)
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.ctx.enable(moderngl.BLEND)
            return
        if style == AXES:                 # round Gaussians: no axis to draw, their centres instead
            style = CENTRES
        order = np.argsort(-((data.positions - eye) @ forward)).astype(np.uint32)       # far first
        data.order.write(order.tobytes())
        data.texture.use(0)
        self.program["data"].value = 0
        quads(SOFT if style == SPLAT_AXES else style)
        if style == SPLAT_AXES and lines is not None:
            # The cloth's solid cores into the depth buffer only (they blend in as nothing: no colour,
            # no cover), then the axes tested against them.
            self.ctx.enable(moderngl.DEPTH_TEST)
            quads(DEPTH_ONLY)
            self.ctx.disable(moderngl.DEPTH_TEST)
            axes(AXIS_LIFT * extent)

    def render(self, layers, camera, frame, size, background=(18, 18, 24), target="picture", style=SOFT, view=None):
        """Draws `layers` - [(key, splat, colours, strength)] - from the orbit `camera`
        (azimuth, elevation, zoom, pan) round `frame` (centre, extent) into a size (w, h) picture;
        each later layer is laid over the ones before at its strength (1 = as drawn). Returns the
        picture's texture. `key` names a splat's upload: a new key uploads, an old one reuses it;
        `colours` (None: the splat's own, or a function giving them) is only read on an upload.
        Each `target` name keeps its own picture, so several can be shown at once. `style` is one of
        STYLES' indices (module docstring). `view` - (eye, forward, right, up, focal px) - draws from that
        camera instead of the orbit one (a training camera, to lay the splat over its photo)."""
        size = (max(1, int(size[0])), max(1, int(size[1])))
        if view is not None:
            eye, forward, right, up = (np.asarray(v, dtype=np.float64) for v in view[:4])
            focal = float(view[4])
        else:
            azimuth, elevation, zoom, pan = camera
            eye, forward, right, up = gaussian_splatting.orbit_camera(None, azimuth, elevation, zoom, pan, frame)
            focal = 0.5 * size[0] / math.tan(math.radians(HALF_FOV_DEG))
        distance, extent = float(np.linalg.norm(np.asarray(frame[0]) - eye)), float(frame[1])
        depth_range = (max(distance - 4.0 * extent, 0.01 * distance), distance + 4.0 * extent)
        fbo = self.targets[target] = self._framebuffer(self.targets.get(target), size)
        ctx = self.ctx
        previous = ctx.fbo
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        ctx.enable(moderngl.BLEND)
        fbo.use()
        fbo.clear(*(c / 255.0 for c in background), 1.0)
        for index, (key, splat, colours, strength) in enumerate(layers):
            ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
            if index == 0 or strength >= 1.0:
                fbo.use()
                self._draw(key, splat, colours, eye, forward, right, up, size, style, depth_range, extent, focal)
                continue
            # A see-through layer: drawn on its own, then laid over the picture at its strength.
            self.layer = self._framebuffer(self.layer, size)
            self.layer.use()
            self.layer.clear(0.0, 0.0, 0.0, 0.0)
            self._draw(key, splat, colours, eye, forward, right, up, size, style, depth_range, extent, focal)
            fbo.use()
            self.layer.color_attachments[0].use(0)
            self.composite["layer"].value = 0
            self.composite["strength"].value = float(strength)
            self.quad.render(moderngl.TRIANGLE_STRIP)
        ctx.disable(moderngl.BLEND)
        target = previous if previous is not None else ctx.screen
        if target is not None:            # a standalone context has no screen
            target.use()
        return fbo.color_attachments[0]

    def gaussian_ids(self, key, splat, view, size, frame, min_opacity: float = 0.02) -> np.ndarray:
        """Which Gaussian of `splat` is in front at each pixel seen from `view` (render's): an (h, w) int
        array of indices into splat's own order, -1 where none is. Each Gaussian is drawn as its solid
        core (one standard deviation), depth-tested, its number written as its colour."""
        keep = np.flatnonzero(np.asarray(splat["opacity"]) >= min_opacity)
        code = keep + 1                                   # 0 is "no Gaussian"
        colours = np.stack([code & 255, (code >> 8) & 255, (code >> 16) & 255], axis=1).astype(np.float32) / 255.0
        full = np.zeros((len(splat["opacity"]), 3), dtype=np.float32)
        full[keep] = colours
        self.render([(key, splat, full, 1.0)], None, frame, size, background=(0, 0, 0), style=IDS, view=view)
        fbo = self.targets["picture"]
        w, h = fbo.size
        raw = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(h, w, 3)[::-1].astype(np.int64)
        return raw[..., 0] + (raw[..., 1] << 8) + (raw[..., 2] << 16) - 1

    def read(self, target="picture") -> np.ndarray:
        """A picture as an (h, w, 3) uint8 array, top row first."""
        fbo = self.targets[target]
        w, h = fbo.size
        rgb = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(h, w, 3)
        return np.ascontiguousarray(rgb[::-1])
