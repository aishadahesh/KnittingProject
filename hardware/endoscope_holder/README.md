# Endoscope mounts for the UR5e

Three designs, all for the 7 mm x 40 mm endoscope head, lens down on the tool axis:

| file | holds the camera | status |
|---|---|---|
| `endoscope_holder.scad/.stl` | in the OnRobot 2FG7 gripper's fingers | printed and in use until Oct 2026 - the camera ended up 5 cm beside the tool point and turned 49 deg (measured by the mount calibration) |
| `endoscope_flange_mount.scad/.stl` | bolted to the tool flange (5 Oct design) | never printed |
| `endoscope_flange_mount_v2.scad/.stl` | bolted to the tool flange, located by the flange pin | superseded by v3 (ears printed in mid-air) |
| `endoscope_flange_mount_v3.scad/.stl` | same plate; plain tube with a conical press-fit bore and a cable window, no riser/ears | superseded by v4 (cable had to be threaded through the bore; camera held by friction only) |
| **`endoscope_flange_mount_v4.scad/.stl`** | 4 mm plate, 4 bolt holes for the original M6 screws (7 mm thread, 9 mm head) in 2 mm-deep head pockets, no washers, no dowel; bore opens as an 11 mm funnel on the flange face and narrows steadily from 9 mm to a 5.6 mm lip at the tip, where the 6 mm camera (measured 12 Nov) stops; 4.2 mm cable slot up the tube and across the plate | **print this one**; camera goes in from the flange side before bolting; lens ~48 mm from flange (tool offset ~0.048 m, re-run the mount calibration). The .stl was built from the .scad's dimensions with manifold3d (OpenSCAD not installed) |

## v2: what it is

Remove the gripper. The plate bolts straight to the flange (ISO 9409-1-50-4-M6: four M6 x 12
socket screws on the 50 mm circle, counterbored) and the flange's 6 mm locating pin drops into
the blind hole between two of the bolts, so the mount can only go on one way and goes back the
same way after every remount. A 22 mm riser lifts a 30 mm clamp tube; the camera head sits in
the tube's 7.2 mm bore with its lens end 10 mm proud, the cable comes down the bore, turns out
through the pocket in the riser (+y side) and away. The 4.2 mm slot along the tube (-y side) is
where the cable is pressed in sideways before the head slides up into the bore; the two M3 ears
squeeze the slot shut so the head cannot tilt or slip.

The lens tip is **68 mm from the flange face**. In the app set *Tool offset* to 0.068 m
(All settings > 1 Setup > Calibration).

## Printing

Flange face down on the bed, no supports; 0.2 mm layers, 4 walls, 40 % infill, PETG or PLA.
Check the bolt circle on the print against the flange before fitting the camera. The .stl is a
0.2 mm voxel rendering of the .scad (OpenSCAD was not at hand); open the .scad in OpenSCAD and
export for a smooth surface, or print the voxel one - the holes are within 0.2 mm either way.

## After mounting

1. Scan one nut on white paper with the splat pattern and press *Next* under Reconstruct: the
   mount calibration should come out as a small offset along the axis only, with little turn.
2. Run a short sweep over a cloth: the export measures the image turn from the pictures and will
   say whether the picture is mirrored and which way its axes point.
3. Then the lens: `scripts/calibrate_lens.py` with a checkerboard.
