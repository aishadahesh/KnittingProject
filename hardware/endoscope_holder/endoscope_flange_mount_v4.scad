// UR5 tool-flange mount for the 6 mm x 40 mm endoscope, v4 - the camera on the tool axis, lens down.
//
// v4 over v3 (Nov 11 review):
//  - a cable slot along the whole tube and across the plate (+y side, between two bolts), so the
//    cable is pressed in sideways - it never has to be threaded through the bore, plug first;
//  - the tube is wide at the plate and narrows towards its tip, outside and inside. The bore ends in a
//    lip narrower than the camera, so with the arm pointing down the camera rests on the lip and
//    cannot fall out. The camera therefore goes in from the flange side, through a hole in the plate,
//    before the plate is bolted on. A full ring is left at the tip (the slot stops short of it) so the
//    lip cannot spread open.
//
// Assembly: press the cable into the slot, push the head up through the plate hole until it stops
// on the lip, lay the cable along the slot and out of the plate edge, then bolt the plate on.
// If the head slides back when the arm points up, wrap one turn of tape round it before inserting.
// Print flange face down, no supports, 100 % infill for the plate (or 6+ walls). Units: mm.

plate_r     = 31.5;   // the flange's 63 mm face (ISO 9409-1-50-4-M6)
plate_t     = 4;      // thin plate (Nov 12): screw heads sit on top, with a washer each
hole_d      = 6.5;    // M6 clearance
// Screws: the robot's original 4 x M6 with 7 mm of thread (9 mm heads), no washers. Each head sits in a
// pocket that leaves 2 mm of plate under it, so 5 mm of thread reaches into the flange.
// Tighten firmly, not hard - only 2 mm of plastic is under each head.
pocket_d     = 10.5;  // for the 9 mm head
under_head   = 2;     // plate left under each head
pcd         = 50;
// No dowel hole: the 4 bolts alone locate the plate, so re-run the mount calibration after each remount.

head_d      = 6.0;    // the camera head (measured Nov 12: 6 mm across, 40 mm long)
head_len    = 40;
back_gap    = 6;      // the head's back end stands this far above the plate, where the cable bends out

entry_d     = 11.0;   // funnel mouth on the flange face, so the head finds the hole easily
entry_h     = 1.5;    // the funnel's depth
bore_base_d = 9.0;    // wide at the bottom (3 mm bigger than the head): it slides in loosely...
bore_mid_d  = 6.3;    // ...narrowing steadily up the whole plate and tube to this, just before the lip
lip_d       = 5.6;    // narrower than the head: it stops here. The lens rim is all that is covered
lip_len     = 3;      // length of the cone from bore_mid_d to lip_d

lens_recess = 1.7;    // the head stops about this far inside the tip, where the lip cone reaches the head's 6.0 mm
tube_len    = back_gap + head_len;  // = 46 mm: the head's back end then sits ~4 mm above the plate top
tube_r0     = 10.5;   // outside radius at the plate
tube_r1     = 6.5;    // outside radius at the tip
skirt_h     = 6;      // wider cone where the tube meets the thin plate, for stiffness
skirt_r     = 16;     // stays clear of the screw heads and washers (their edges ~19 mm from centre)

slot_w      = 4.2;    // the cable is 3.5 mm
tip_ring    = 4;      // the slot stops this far short of the tip
$fn         = 96;

tube_z0 = plate_t;
tube_z1 = tube_z0 + tube_len;
LENS_FROM_FLANGE = tube_z1 - lens_recess;   // = 48.3 mm
echo(str("lens about ", LENS_FROM_FLANGE, " mm from the flange face: set tool offset = ", LENS_FROM_FLANGE / 1000, " m"));

module plate()
    difference() {
        cylinder(r = plate_r, h = plate_t);
        for (a = [45 : 90 : 315]) rotate(a) translate([pcd / 2, 0, 0]) {
            translate([0, 0, -1]) cylinder(d = hole_d, h = plate_t + 2);
            translate([0, 0, under_head]) cylinder(d = pocket_d, h = plate_t);
        }
    }

module body()
    union() {
        plate();
        translate([0, 0, tube_z0 - 0.01]) cylinder(r1 = tube_r0, r2 = tube_r1, h = tube_len + 0.01);
        translate([0, 0, tube_z0 - 0.01]) cylinder(r1 = skirt_r, r2 = tube_r0 - skirt_h * (tube_r0 - tube_r1) / tube_len, h = skirt_h);
    }

difference() {
    body();
    // bore: entry funnel, one long cone from the flange face to the lip, then the lip cone
    translate([0, 0, -0.01]) cylinder(d1 = entry_d, d2 = bore_base_d, h = entry_h + 0.01);
    translate([0, 0, -1]) cylinder(d = bore_base_d, h = 1.01);
    cylinder(d1 = bore_base_d, d2 = bore_mid_d, h = tube_z1 - lip_len + 0.01);
    translate([0, 0, tube_z1 - lip_len]) cylinder(d1 = bore_mid_d, d2 = lip_d, h = lip_len + 0.01);
    translate([0, 0, tube_z1]) cylinder(d = lip_d, h = 1);
    // cable slot: up the +y side of the tube, and across the plate to its edge
    translate([-slot_w / 2, 0, -1]) cube([slot_w, plate_r + 1, tube_len + plate_t - tip_ring + 1]);
}
