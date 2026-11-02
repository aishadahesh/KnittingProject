// UR5e tool-flange mount for the 7 mm x 40 mm endoscope, v2 - the camera on the tool axis, lens down.
//
// Replaces the gripper-held holder. Bolts straight onto the flange (ISO 9409-1-50-4-M6: 63 mm face,
// 4 x M6 on a 50 mm bolt circle, one 6 mm locating pin on the same circle between two bolts), so the
// camera sits on the tool axis and the pin fixes its turn: the hand-eye offset is then just a
// distance along the axis, and the image's axes are the tool's. Print flange face down, no supports.
// Units: mm. endoscope_flange_mount_v2.stl is this file at these values.
//
// v2 over the 5 Oct design: the locating-pin hole (repeatable orientation after every remount), a
// 30 mm tube with a full-length clamp slot and two M3 ears (the head cannot tilt in the bore), a
// pocket for the cable behind the head instead of two bare pillars, and a fillet where the tube
// meets the plate. After printing, set the app's tool offset to LENS_FROM_FLANGE below.

plate_r     = 31.5;   // the flange's own 63 mm face
plate_t     = 12;
hole_d      = 6.5;    // M6 clearance
cb_d        = 11;     // socket-head counterbore
cb_depth    = 6;      // 6 mm of plate under each head: M6 x 12 screws reach 6 mm into the flange
pcd         = 50;
pin_d       = 6.2;    // the flange's 6 mm locating pin, with print clearance
pin_depth   = 7;      // the pin stands 6 mm proud of the flange
pin_angle   = 0;      // bolts at 45/135/225/315, the pin between two of them

riser_h     = 16;     // plate top to tube bottom: room for the cable to turn out sideways
riser_w     = 22;     // the riser block's width (y) and depth (x) around the axis
cable_w     = 9;      // the cable's way out of the riser, towards +y
tube_r      = 9;
tube_len    = 30;     // 30 of the head's 40 mm held; the lens tip ends LENS_FROM_FLANGE below
bore_d      = 7.2;    // 7.0 mm head + 0.2 mm print clearance
slot        = 4.2;    // the cable (3.5 mm) presses in sideways through this, then the head slides into the bore; the ears' screw squeezes it shut
ear_t       = 6;      // each ear's thickness (x), length (y) and height (z)
ear_len     = 9;
screw_d     = 3.4;    // M3 clearance; a nut sits on the far ear
$fn         = 96;

tube_z0 = plate_t + riser_h;
tube_z1 = tube_z0 + tube_len;
LENS_FROM_FLANGE = tube_z1 + (40 - tube_len);   // = 68 mm: head bottom at tube_z1, lens 10 mm further
echo(str("lens tip is ", LENS_FROM_FLANGE, " mm from the flange face: set tool offset = ", LENS_FROM_FLANGE / 1000, " m"));

module plate()
    difference() {
        cylinder(r = plate_r, h = plate_t);
        for (a = [45 : 90 : 315]) rotate(a) translate([pcd / 2, 0, 0]) {
            translate([0, 0, -1]) cylinder(d = hole_d, h = plate_t + 2);
            translate([0, 0, plate_t - cb_depth]) cylinder(d = cb_d, h = cb_depth + 1);
        }
        rotate(pin_angle) translate([pcd / 2, 0, -1]) cylinder(d = pin_d, h = pin_depth + 1);
    }

module riser()
    difference() {
        translate([-riser_w / 2, -riser_w / 2, plate_t - 0.01]) cube([riser_w, riser_w, riser_h + 0.02]);
        // The cable's pocket: open towards +y and upwards into the tube's bore.
        translate([-cable_w / 2, -1, plate_t + 3]) cube([cable_w, riser_w, riser_h]);
    }

module tube()
    difference() {
        union() {
            translate([0, 0, tube_z0 - 0.01]) cylinder(r = tube_r, h = tube_len + 0.01);
            // Fillet-like skirt where the tube meets the riser.
            translate([0, 0, tube_z0 - 0.01]) cylinder(r1 = riser_w / 2 * 1.2, r2 = tube_r, h = 4);
        }
        translate([0, 0, tube_z0 - 1]) cylinder(d = bore_d, h = tube_len + 2);
        // Clamp slot along the whole tube, towards -y (away from the cable exit).
        translate([-slot / 2, -tube_r - 1, tube_z0 - 1]) cube([slot, tube_r + 1, tube_len + 2]);
    }

module ears()
    difference() {
        for (side = [-1, 1])
            translate([side > 0 ? slot / 2 : -slot / 2 - ear_t, -tube_r - ear_len + 2, tube_z0 + tube_len / 2 - ear_t / 2 + 4])
                cube([ear_t, ear_len, ear_t]);
        translate([-ear_t - slot, -tube_r - ear_len / 2, tube_z0 + tube_len / 2 + 4]) rotate([0, 90, 0]) cylinder(d = screw_d, h = 2 * ear_t + slot + 2);
    }

union() {
    plate();
    riser();
    tube();
    ears();
}
