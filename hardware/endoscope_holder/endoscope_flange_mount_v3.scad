// UR5e tool-flange mount for the 7 mm x 40 mm endoscope, v3 - the camera on the tool axis, lens down.
//
// v3 over v2 (professor's feedback): no riser block, no clamp slot, no M3 ears (they printed in mid-air
// and were not needed). Just the flange plate and a plain tube standing on it, with a window in its
// side near the bottom for the cable. The bore is a cone: a bit wider at the top so the camera is easy to
// thread in from above, exactly the camera's thickness at the bottom so it is pushed in with force and
// held by friction with the plastic. If it is still loose, add a drop of glue.
// Print flange face down, no supports. Units: mm.
//
// Assembly: thread the cable down the bore and out through the side window, then push the head down
// until it stops on the plate; the lens then stands 10 mm proud of the tube top.
// After printing, set the app's tool offset to LENS_FROM_FLANGE below.

plate_r     = 31.5;   // the flange's own 63 mm face
plate_t     = 12;
hole_d      = 6.5;    // M6 clearance
cb_d        = 11;     // socket-head counterbore
cb_depth    = 6;      // 6 mm of plate under each head: M6 x 12 screws reach 6 mm into the flange
pcd         = 50;
pin_d       = 6.2;    // the flange's 6 mm locating pin, with print clearance
pin_depth   = 7;      // the pin stands 6 mm proud of the flange
pin_angle   = 0;      // bolts at 45/135/225/315, the pin between two of them

head_d      = 7.0;    // the camera head's diameter
head_len    = 40;
bore_bot_d  = head_d;         // exactly the camera's thickness at the bottom: press fit. If too tight to push in, use head_d + 0.1
bore_top_d  = head_d + 1.2;   // wider at the top, so the head enters easily
tube_r      = 9;
tube_len    = 30;     // 30 of the head's 40 mm held; the lens tip ends LENS_FROM_FLANGE above the flange face
skirt_h     = 4;      // small conical foot where the tube meets the plate
skirt_r     = 12;

win_w       = 5;      // the cable's window (cable is 3.5 mm), towards +y
win_h       = 8;      // its height above the plate top
$fn         = 96;

tube_z0 = plate_t;
tube_z1 = tube_z0 + tube_len;
LENS_FROM_FLANGE = plate_t + head_len;   // head sits on the plate top, lens at its far end = 52 mm
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

module tube()
    difference() {
        union() {
            translate([0, 0, tube_z0 - 0.01]) cylinder(r = tube_r, h = tube_len + 0.01);
            translate([0, 0, tube_z0 - 0.01]) cylinder(r1 = skirt_r, r2 = tube_r, h = skirt_h);
        }
        // Conical bore: head_d at the bottom, wider at the top. Blind: the plate is its floor.
        translate([0, 0, tube_z0]) cylinder(d1 = bore_bot_d, d2 = bore_top_d, h = tube_len + 0.01);
        // Cable window through the +y side, flush with the floor.
        translate([-win_w / 2, 0, tube_z0]) cube([win_w, skirt_r + 1, win_h]);
    }

union() {
    plate();
    tube();
}
