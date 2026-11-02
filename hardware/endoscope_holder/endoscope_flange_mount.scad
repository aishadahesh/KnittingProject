// UR5e flange mount for the 7 mm x 40 mm endoscope - bolts straight onto the tool flange, no gripper.
// Flange: ISO 9409-1-50-4-M6 (63 mm face, 4 x M6 on a 50 mm bolt circle). Camera on the tool axis, lens away
// from the flange. Print flange face down on the bed - no supports. Units: mm.
// endoscope_flange_mount.stl is this file at these values.

plate_r    = 31.5;  // the flange's own 63 mm face - nothing overhangs it or the tool connector beside it
plate_t    = 12;
hole_d     = 6.5;   // M6 clearance
cb_d       = 11;    // socket-head counterbore
cb_depth   = 6;     // leaves 6 mm of plate under each head: M6 x 12 screws reach 6 mm into the flange
pcd        = 50;
pillar_gap = 9;     // the cable turns out sideways between the two pillars
pillar_h   = 14;
tube_r     = 9;
bore_d     = 7.2;   // 7.0 mm head + 0.2 mm print clearance
tube_len   = 25;    // 25 of the head's 40 mm held; the lens tip ends 66 mm from the flange face
slot       = 4.2;   // the cable presses in through this, then the head slides into the bore
ear_x      = 7.5;
screw_d    = 3.4;   // M3 screw and nut squeeze the slot shut
$fn        = 72;

tube_z = plate_t + pillar_h;

module plate()
    difference() {
        cylinder(r = plate_r, h = plate_t);
        for (a = [45 : 90 : 315]) rotate(a) translate([pcd / 2, 0, 0]) {
            translate([0, 0, -1]) cylinder(d = hole_d, h = plate_t + 2);
            translate([0, 0, plate_t - cb_depth]) cylinder(d = cb_d, h = cb_depth + 1);
        }
    }

module c_tube()
    difference() {
        translate([0, 0, tube_z]) cylinder(r = tube_r, h = tube_len);
        translate([0, 0, tube_z - 1]) cylinder(d = bore_d, h = tube_len + 2);
        translate([-slot / 2, 0, tube_z - 1]) cube([slot, tube_r + 1, tube_len + 2]);
    }

module ear(x0, x1)
    difference() {
        translate([x0, 7, tube_z + 4]) cube([x1 - x0, 8, 17]);
        translate([x0 - 1, 11, tube_z + 12.5]) rotate([0, 90, 0]) cylinder(d = screw_d, h = x1 - x0 + 2);
    }

union() {
    plate();
    for (side = [-1, 1])
        translate([side > 0 ? pillar_gap / 2 : -tube_r, -3, plate_t - 0.5]) cube([tube_r - pillar_gap / 2, 6, pillar_h + 2.5]);
    c_tube();
    ear(-ear_x, -slot / 2);
    ear(slot / 2, ear_x);
}
