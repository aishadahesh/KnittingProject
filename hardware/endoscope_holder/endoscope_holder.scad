// Endoscope holder for the OnRobot 2FG7 on the UR5e - first fit-check version.
// The 2FG7 grips the top block; the 7 mm x 40 mm endoscope head clamps in the tube, lens down, on the tool axis.
// Printed as drawn (grip block up). Units: mm. endoscope_holder.stl is this file at these values.

grip_w     = 50;    // between the fingers - inside the 2FG7's 35-73 mm outward range
grip_d     = 22;
grip_h     = 34;
shoulder_w = 80;    // lip under the fingertips, so the block cannot slide out downward
shoulder_d = 26;
shoulder_t = 5;
pillar_gap = 9;     // the cable turns out between the two pillars
pillar_h   = 14;
tube_r     = 9;
bore_d     = 7.2;   // 7.0 mm head + 0.2 mm print clearance
tube_len   = 25;    // 25 of the head's 40 mm held, 15 mm and the lens below
slot       = 4.2;   // the cable presses in through this, then the head is pulled up into the bore
ear_x      = 7.5;
screw_d    = 3.4;   // M3 screw and nut squeeze the slot shut
$fn        = 72;

tube_top = -9 + tube_len;
plate_z  = tube_top + pillar_h;
grip_z   = plate_z + shoulder_t;

module c_tube()
    difference() {
        translate([0, 0, -9]) cylinder(r = tube_r, h = tube_len);
        translate([0, 0, -10]) cylinder(d = bore_d, h = tube_len + 2);
        translate([-slot / 2, 0, -10]) cube([slot, tube_r + 1, tube_len + 2]);
    }

module ear(x0, x1)
    difference() {
        translate([x0, 7, -5]) cube([x1 - x0, 8, 17]);
        translate([x0 - 1, 11, 3.5]) rotate([0, 90, 0]) cylinder(d = screw_d, h = x1 - x0 + 2);
    }

union() {
    c_tube();
    for (side = [-1, 1])
        translate([side > 0 ? pillar_gap / 2 : -tube_r, -3, tube_top - 2]) cube([tube_r - pillar_gap / 2, 6, pillar_h + 2.5]);
    translate([-shoulder_w / 2, -shoulder_d / 2, plate_z]) cube([shoulder_w, shoulder_d, shoulder_t]);
    translate([-grip_w / 2, -grip_d / 2, grip_z - 0.5]) cube([grip_w, grip_d, grip_h + 0.5]);
    ear(-ear_x, -slot / 2);
    ear(slot / 2, ear_x);
}
