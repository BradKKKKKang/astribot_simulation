# Temporary fixed penholder layouts

Ten independent evaluation configs, in the order of the supplied photographs.
Run from the repository root (with the usual simulation environment loaded):

```bash
python astribot_simulation_integrate.py --config config/experiments/evaluation/penholder_fixed/01_20261103-154119.json --check-config
python astribot_simulation_integrate.py --config config/experiments/evaluation/penholder_fixed/01_20261103-154119.json
```

Select another JSON to change the layout. Each config defaults to one managed
evaluation episode, with a 2000-step budget; the interactive launcher runs until
closed. Results use a separate `outputs/evaluation/penholder_fixed/<config-stem>/`
directory for each layout.

These configs reuse the penholder scene, robot initial pose and success criteria.
`task.overrides.reset: null` disables the inherited empirical XY and yaw sampler:
creation and every reset use the configured poses. The inherited scene sampling
fields are inactive. Objects remain dynamic and can move during an episode.
Geometry-based grounding keeps both objects 1 mm above the tabletop initially.

## Photo estimates

Positions are approximate, not camera-calibrated measurements. A planar
perspective mapping was estimated independently for each photo from the white
table edges, extrapolating the cropped near corners, and assuming the existing
1.40 m by 0.70 m simulation tabletop. Pen endpoints determine its midpoint and
cap-facing direction; the holder uses its estimated base footprint center, not
its elevated rim. Lens distortion and height-induced parallax are not corrected.
In photo 01, most of the pen is occluded by the left gripper; its center is
estimated from the visible cap, approximate axis and the 15 cm model length.
Holder rotation retains the scene's upright default; its decorative orientation
has not been calibrated from the photos. No reference photos are copied into the
repository.

XY coordinates below are meters in `table.tabletop`: +X points left in the
robot's view, +Y toward the robot. World XY is `[0.6 - table_y, table_x]`.
Pen heading is the yaw offset from the forward-facing cap orientation: 0 degrees
points away from the robot, positive angles turn toward the robot's left.
The stored quaternion is WXYZ, with the cap at the asset's local -Z end.

| Config / photo | Pen XY (m) | Holder XY (m) | Pen heading (degrees) |
| --- | --- | --- | --- |
| [01_20261103-154119](01_20261103-154119.json) | [0.354, 0.188] | [0.509, -0.065] | +14.5 |
| [02_20261103-154116](02_20261103-154116.json) | [0.52, 0.203] | [0.468, 0.006] | +7.6 |
| [03_20261103-154113](03_20261103-154113.json) | [0.479, 0.178] | [0.087, 0.068] | -21.3 |
| [04_20261103-154110](04_20261103-154110.json) | [0.544, 0.087] | [0.075, 0.074] | +27.6 |
| [05_20261103-154108](05_20261103-154108.json) | [0.082, 0.143] | [0.476, 0.004] | +40.5 |
| [06_20261103-154105](06_20261103-154105.json) | [0.17, 0.111] | [0.451, 0.034] | -30.1 |
| [07_20261103-154101](07_20261103-154101.json) | [0.096, 0.184] | [0.186, -0.019] | +0.6 |
| [08_20261103-154055](08_20261103-154055.json) | [0.2, 0.181] | [0.138, 0.048] | +26.3 |
| [09_20261103-154050](09_20261103-154050.json) | [0.221, -0.223] | [0.404, 0.11] | +40.8 |
| [10_20261103-154047](10_20261103-154047.json) | [0.406, -0.102] | [0.064, 0.059] | -11.2 |
