# Temporary fixed penholder layouts

Ten independent evaluation configs, in the order of the supplied photographs.
Run from the repository root (with the usual simulation environment loaded):

```bash
python astribot_simulation_integrate.py --config config/experiments/evaluation/penholder_fixed/01_20261103-154119.json --check-config
python astribot_simulation_integrate.py --config config/experiments/evaluation/penholder_fixed/01_20261103-154119.json
```

Select another JSON to change the layout. Each config defaults to one managed
evaluation episode, with a 4000-step budget; the interactive launcher runs until
closed. Results use a separate `outputs/evaluation/penholder_fixed/<config-stem>/`
directory for each layout.

These configs reuse the penholder scene, robot initial pose and success criteria.
`task.overrides.reset: null` disables the inherited empirical XY and yaw sampler:
creation and every reset use the configured poses. The inherited scene sampling
fields are inactive. Objects remain dynamic and can move during an episode.
Geometry-based grounding keeps both objects 1 mm above the tabletop initially.

The current pen asset is **140 mm long × 20 mm maximum diameter**. Its radial
size was restored from 18 mm to 20 mm while retaining the 140 mm length.
The calibration captures below used a 160 mm pen; their image-space residuals
describe those archived captures, not the current shorter pen.

The current holder is **100 mm tall**, reduced from 120 mm with its horizontal
dimensions preserved. The archived calibration captures below use the original
120 mm holder; its initial ground clearance remains 1 mm with the new height.

## Remaining eight layouts after updated initial pose — 2026-11-04

Layouts **01, 03, 04, 05, 06, 08, 09 and 10** were refitted to their original
`20261103-*.jpg` photographs using the current `real_eval_init_pose.hdf5`
robot pose. **02 and 07 are byte-for-byte unchanged.** Only these eight
experiment JSONs' object placements and table instance overrides changed;
shared robot, task, camera, lighting and asset inputs are preserved.

Each layout now has its own table XY, yaw and XY scale, as in 02/07. Fitted
table dimensions range from 1.238–1.252 × 0.606–0.617 m; height remains
0.750 m. Pen XY/yaw aligns the visible endpoint centers with the existing
160 × 20 mm mesh. Holder XY fits the projected bounding-box center, and its
yaw follows the visible hanging tag's horizontal position relative to the
silhouette width. Tag placement is an approximate orientation cue; the
holder outline and decoration geometry still differ from the photographs.

Both objects retain geometry grounding and 1 mm initial tabletop clearance.
XY is expressed in **each layout's own `table.tabletop` frame**; the historical
world conversion `[0.6 - table_y, table_x]` below no longer applies to these
updated layouts. Reset sampling remains disabled.

| Layout | Table edge mean error, before → after (px) | Largest observed pen endpoint error, before → after (px) |
| --- | --- | --- |
| 01 | 66.8 → 2.6 | 23.3 → 0.0* |
| 03 | 65.9 → 2.6 | 34.3 → 5.2 |
| 04 | 65.7 → 2.7 | 29.9 → 2.0 |
| 05 | 67.7 → 2.7 | 20.6 → 7.8 |
| 06 | 65.9 → 2.7 | 26.6 → 7.8 |
| 08 | 65.5 → 2.5 | 21.9 → 5.6 |
| 09 | 65.3 → 2.7 | 23.8 → 2.9 |
| 10 | 64.5 → 2.6 | 28.8 → 5.8 |

Errors use the 1280 × 720 head-camera comparison. *01's rear endpoint is
hidden: only the cap and observed axis constrain its fit, so zero cap error
is by construction. 06/08 have partial occlusion. These are single-image
fitting residuals, not independent metric calibration measurements.

Evidence: [interactive opacity comparison](../../../../outputs/evaluation/penholder_fixed/remaining_alignment_20261104/index.html),
[01/03/04/05 comparison](../../../../outputs/evaluation/penholder_fixed/remaining_alignment_20261104/comparison_01_03_04_05.jpg),
[06/08/09/10 comparison](../../../../outputs/evaluation/penholder_fixed/remaining_alignment_20261104/comparison_06_08_09_10.jpg),
[annotations](../../../../outputs/evaluation/penholder_fixed/remaining_alignment_20261104/annotations.json),
and [validation](../../../../outputs/evaluation/penholder_fixed/remaining_alignment_20261104/validation.json).

All eight final configs were independently reloaded through the integrated
Isaac Lab launcher: normal reset, seed 42, 32 render updates, no post-reset
physics steps. Final file hashes, scene specifications, camera poses,
ground clearance and table footprint containment are checked by `report.py`.
The before capture uses the same current robot pose. All ten configs resolve.
Overlays are unmasked 50/50 blends of viewport-cropped/resized photographs and
actual head-camera output with its normal lens distortion. No extra image warp
or gripper alignment is used. This validates initial placement, not dynamics
or task success.

The evidence directory contains `capture.py`, `run_capture.sh`,
`run_after.sh`, `annotate.py`, `fit.py`, `report.py` and the original configs.
CPU report regeneration: `python outputs/evaluation/penholder_fixed/remaining_alignment_20261104/report.py`.
Capture requires the Isaac/ROS environment and a new output directory;
**use a separate process per final layout**, because their static tables now
differ. The historical shared-table capture helper below is not suitable.

## Fixed 07 reference fit — 2026-11-04

Layout 07 now uses `screenshot-20261104-162133.png`, with the current
`real_eval_init_pose.hdf5` initial robot pose. The experiment's own table
override is world XY `[0.63333058, 0.02824593]`, yaw 90.0251 degrees,
size 1.254007 × 0.608386 × 0.750 m. This is close to the current layout-02
reference fit; the shared table asset and other layout configurations are
unchanged. Initial robot pose, camera calibration, object meshes and lights
are preserved.

Pen world XY moved 16.49 mm and its yaw changed -0.7526 degrees. Holder
world XY moved 19.67 mm; it was turned 180 degrees to put the front
flower and hanging label toward the camera, matching their visibility in the
reference. Its projected box center is fitted, but the larger mesh outline
and decoration geometry retain visible differences. Pen dimensions remain
160 × 20 mm, producing residual endpoint differences of about 8.5/6.8 px.

The viewport crop `[13, 2, 920, 512]` is resized to 1280 × 720 and blended
50/50 with the actual head-camera output including the existing lens model.
No extra image warp, foreground masking or gripper alignment is applied. The
mean manually annotated table-boundary residual changes from
64.1 to 3.1 px. These are
single-image fitting residuals, not independent calibration measurements.

Evidence: [overlay](../../../../outputs/evaluation/penholder_fixed/fix07_alignment_20261104/overlay.png),
[before/after comparison](../../../../outputs/evaluation/penholder_fixed/fix07_alignment_20261104/comparison.jpg),
and [validation](../../../../outputs/evaluation/penholder_fixed/fix07_alignment_20261104/validation.json).
The saved configuration was reloaded in Isaac Lab: normal reset, seed 42,
32 rendering updates and no post-reset physics steps. This checks initial
placement, not dynamics/task success. The old shared-table batch helper is
not applicable to layouts 02/07; capture each with a fresh integrated launcher.

## Fixed 02 after updated initial pose — 2026-11-04

Refitted the same real photograph after the user updated `task.initial_pose`
from `real_eval_init_pose.hdf5`. Both captures in this pass use that new robot
pose. This pass preserves the initial-pose configuration, camera calibration,
object meshes and lighting; only layout 02's placement/instance overrides change.

Relative to the preceding fixed-02 fit: pen XY moved 26.05 mm and yaw changed
-1.7021 degrees; holder XY moved 28.15 mm. Table XY moved 16.88 mm, yaw
changed -1.2750 degrees, and its length/width changed by -1.98%/-3.37% within
the previously authorized per-layout size adjustment. The resulting tabletop
is 1.255178 × 0.608741 m, with height still 0.750 m. Its world XY is
`[0.62193763, 0.03237847]`, yaw 89.6333 degrees. Objects retain 1 mm initial
tabletop clearance. Other nine layout files and shared inputs are unchanged.

The table's manually selected edge-point mean residual is
13.6 → 3.3 px; the
pen's observed end-center residuals are 28.8/33.5 → 5.2/4.2 px. Holder
projected bounding-box center is fitted to its reference center; its mesh
outline remains different. These are single-image placement residuals, not
independent calibration or task-success results. The reference crop, normal
camera distortion and unmasked 50/50 overlay procedure are unchanged.

Evidence: [latest overlay](../../../../outputs/evaluation/penholder_fixed/fix02_alignment_20261104_pose2/overlay.png),
[before/after with new robot pose](../../../../outputs/evaluation/penholder_fixed/fix02_alignment_20261104_pose2/comparison.jpg),
and [validation](../../../../outputs/evaluation/penholder_fixed/fix02_alignment_20261104_pose2/validation.json).
The saved config was reloaded through the integrated Isaac Lab launcher; reset
seed 42, 32 render updates, no post-reset physics steps. Prior evidence below
remains archived and uses the previous initial pose.

## Fixed 02 update — 2026-11-04 (previous robot pose)

Layout 02 now uses `screenshot-20261104-152951.png` as its reference. Only its
experiment overrides changed: table world XY `[0.62165594, 0.04925887]`, yaw
90.9083 degrees, instance scale `[0.9146614754, 0.9, 1]` (1.280526 × 0.630 ×
0.750 m); the shared table asset and the other nine layouts retain their sizes.
The pen moved 3.72 mm in world XY and rotated +2.0855 degrees; the holder moved
1.57 mm. Object poses are expressed in the revised tabletop frame. Both objects
retain 1 mm initial clearance; the robot, cameras, object geometry and lighting
are unchanged.

The real viewport `[15, 9, 922, 519]` is resized to 1280 × 720. The overlay is
an unmasked 50/50 blend with the actual integrated Isaac Lab camera output,
including its existing lens distortion. No perspective fit or image warp is
applied. Table boundary sample residuals average about 3.2 px (maximum 9.7 px);
pen endpoint residuals are 3.3/2.6 px. These are fits to manually chosen image
landmarks, not independent physical calibration. Holder mesh outline, rounded
table corners/frame and grippers retain visible differences.

Evidence: [overlay](../../../../outputs/evaluation/penholder_fixed/fix02_alignment_20261104/overlay.png),
[before/after comparison](../../../../outputs/evaluation/penholder_fixed/fix02_alignment_20261104/comparison.jpg),
and [fit metadata](../../../../outputs/evaluation/penholder_fixed/fix02_alignment_20261104/fit_resized.json).
Capture uses normal reset, seed 42 and 32 render updates without post-reset
physics stepping; it validates placement, not dynamics or task completion.

The 2026-11-03 results below are historical, including their layout-02 poses
and metrics. The old multi-layout helper assumes identical static tables and
cannot capture the current layout 02 in the same scene as the other layouts;
use a fresh integrated-launcher process for this configuration.

## 2026-11-03 photo overlay review (historical)

Current pen dimensions are **160 mm long × 20 mm maximum diameter**. XY/yaw
remain from the earlier 150 × 16 mm photo fit; the current report uses new renders
and actual mesh endpoints. The original 150 × 16 mm final captures are retained
in `overlay_review/after_150x16/`. Photo occlusion annotations are unchanged.

The layouts were revised using the actual integrated Isaac head-camera render,
with **only object XY and pen heading changed**. The user chose to preserve the
robot initial pose and record remaining simulation occlusion differences.

The reference viewer viewport is cropped and resized to 1280 × 720. No homography,
local image warp, foreground replacement or camera adjustment is applied. For the original placement fit, the
15 cm pen endpoints were projected through the captured camera K/extrinsics and
fitted in tabletop XY/yaw. Photo 01 provides only a cap tip and its axis; the
hidden rear endpoint is inferred, not measured. Holder XY aligns the projected
mesh bounding-box center with the annotated photo silhouette center. Holder geometry and rotation were unchanged during fitting; the subsequent
pen-size update changes geometry only, without refitting XY/yaw. This is an image-space fit, not a
metric camera calibration or a guarantee of real-world position accuracy.

[Annotations](review/annotations.json) retain original JPEG pixel coordinates,
viewport crops, visible pen endpoints, holder boxes, approximate foreground
polygons, original poses, fit results and per-image occlusion differences.
Polygons describe photo occluders; they are not exact segmentation and are never
used to hide differences in the raw simulation overlays. Source images are read
from Downloads and are not modified.

Generated evidence: [interactive opacity comparison](../../../../outputs/evaluation/penholder_fixed/overlay_review/index.html),
[01–05 contact sheet](../../../../outputs/evaluation/penholder_fixed/overlay_review/contact_sheet_01_05.jpg),
[06–10 contact sheet](../../../../outputs/evaluation/penholder_fixed/overlay_review/contact_sheet_06_10.jpg),
and [projection metrics](../../../../outputs/evaluation/penholder_fixed/overlay_review/metrics.json).
Each layout has its own raw before/after render and unmasked 50% overlay.

### Reproduce capture and overlays

After loading the usual Isaac Python environment and ROS setup, source `env.sh`,
then explicitly isolate the review from the normal ROS domain. Use a **new**
output directory for capture (the script refuses to overwrite captures):

```bash
source env.sh
export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=198
python config/experiments/evaluation/penholder_fixed/review/verify.py capture \
  --output /tmp/penholder-review-new
```

The review helper calls `astribot_simulation_integrate.main`, creates the actual
Isaac scene and cameras, loads each resolved object's default pose, and calls the
normal reset path. The ten layouts share the same robot, task, table and assets.
Twelve rendering updates are captured at each reset without advancing physics;
this validates initial placement, not settling, collisions or task success.
ROS is localhost-only on domain 198. No SDK or hardware commands are sent.

The delivered evidence directory contains `before/` and `after/`. Rebuild its
report using CPU Python with NumPy/OpenCV:

```bash
python config/experiments/evaluation/penholder_fixed/review/verify.py report \
  --output outputs/evaluation/penholder_fixed/overlay_review \
  --photos /home/zihengkang/Downloads
```

Report generation rejects stale final scene/robot/task/placement fingerprints.
The report records each original JPEG's SHA-256. Removing the local evidence
requires regenerating the before capture from the original poses in annotations;
the report does not silently fabricate a baseline.

### Placement and residuals

XY coordinates are meters in `table.tabletop`: +X points left in the robot's
view, +Y toward the robot. World XY is `[0.6 - table_y, table_x]`. Pen heading
is the yaw offset from the forward-facing cap orientation; the stored quaternion
is WXYZ, with the cap at local -Z. All objects retain geometry grounding and
1 mm initial tabletop clearance. Random reset sampling remains disabled.

With the current 160 × 20 mm pen, the endpoint residual below is the largest distance to an **observed annotation**
at 1280 × 720. It is calculated from the actual captured object/camera poses,
not a claim of independent pixel-perfect segmentation. Holder projected box
centers align within 0.01 px by construction; their rendered outline remains
roughly 7–17 px larger in width/height than the annotated photo boxes.

| Config / photo | Pen XY (m) | Holder XY (m) | Pen heading (deg) | Max endpoint residual (px) |
| --- | --- | --- | --- | --- |
| [01_20261103-154119](01_20261103-154119.json) | [0.322776, 0.141315] | [0.447912, -0.0936] | +16.52 | 5.10 |
| [02_20261103-154116](02_20261103-154116.json) | [0.48696, 0.164218] | [0.422384, -0.024017] | +8.20 | 4.36 |
| [03_20261103-154113](03_20261103-154113.json) | [0.459037, 0.142624] | [0.090785, 0.016068] | -20.49 | 5.49 |
| [04_20261103-154110](04_20261103-154110.json) | [0.518131, 0.056657] | [0.080837, 0.023082] | +28.10 | 1.93 |
| [05_20261103-154108](05_20261103-154108.json) | [0.096815, 0.106639] | [0.436917, -0.023669] | +41.98 | 5.91 |
| [06_20261103-154105](06_20261103-154105.json) | [0.172372, 0.074585] | [0.41271, -0.007494] | -28.94 | 5.65 |
| [07_20261103-154101](07_20261103-154101.json) | [0.103509, 0.134735] | [0.176935, -0.053033] | +2.50 | 5.41 |
| [08_20261103-154055](08_20261103-154055.json) | [0.199376, 0.134086] | [0.132876, -0.000979] | +27.32 | 3.55 |
| [09_20261103-154050](09_20261103-154050.json) | [0.214511, -0.224908] | [0.355326, 0.05892] | +42.30 | 1.41 |
| [10_20261103-154047](10_20261103-154047.json) | [0.389046, -0.118958] | [0.062937, 0.008772] | -10.50 | 3.71 |

### Preserved occlusion information and remaining differences

| Photo | Reference and simulator comparison |
| --- | --- |
| 01 | Photo: barrel/rear end hidden. Simulation: barrel and rear end exposed; cap aligned. Hidden rear is not a measured landmark. |
| 02 | Both objects visible in photo and simulation. |
| 03 | Both objects visible; simulated holder is larger and its lower edge is closer to the left fingertip. |
| 04 | Both objects visible; simulated holder is larger and its lower edge is closer to the left fingertip. |
| 05 | Photo pen is clear of the gripper. Simulated left fingertip additionally occludes part of the cap. |
| 06 | Photo barrel/rear edge partly occluded. Simulation obscures more of the barrel/rear; boundaries differ. |
| 07 | Photo pen is clear of the gripper. Simulation places its left edge close to the finger silhouette, with a smaller visible gap. |
| 08 | Both have partial pen occlusion, but the simulated palm/linkage covers a larger/different barrel area. |
| 09 | Photo holder lower-right edge is partially occluded. Simulated holder remains visible with a different finger separation. |
| 10 | Both objects visible in photo and simulation. |

The holder shape/decoration, gripper pose and camera/table alignment retain their
existing differences. Per user instruction these were not fitted or modified.
