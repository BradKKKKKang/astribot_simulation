# Fold T-shirt evaluation

Select `configs/simulation/fold_tshirt.yaml` in Atlas, start Simulation, then Eval.
Simulation resolves `config/experiments/evaluation/fold_tshirt.json`; the task owns
its instruction, initial pose, reset and unconfigured success criteria. Atlas uses
the existing `simulation_s1_ros2` deployment and online dual-WebSocket inference
endpoint (`10.11.5.2:8000`). Load a suitable folding policy at that endpoint before
running policy-driven evaluation. No policy weights are selected by Simulation.

Assets are installed in `~/assets/fold_tshirt/`. `fold_tshirt_cloth.glb` is the
22,462-vertex, 44,568-triangle sewn midsurface, with UVs and the original jersey
material. It drives PhysX particle cloth and an identical render-only surface.
The full original visual GLB, canonical Blender source and physics NPZ are retained
alongside it for offline rendering. Real-time rendering does not use the unsafe
independent nearest-triangle skinning or the offline Blender Surface Deform modifier.
Garment width (70 cm), mass (180 g) and spring parameters remain uncalibrated.

## Initialization and regions

Robot joint targets are the first command frame in
`/home/zihengkang/ckpts/real_eval_init_pose.hdf5`. Named joint values and source hash
are embedded in the task; launch does not reopen that file. The fixed chassis uses
the experiment's robot placement. Do not substitute `real_eval_init_pose_2.hdf5`.

The table is approximately 1.269 by 0.611 m, at height 0.75 m. Its dimensions, XY and
yaw were fitted to the supplied head-camera screenshot with fixed camera calibration
and HDF5 joints. The near corners are cropped in the reference; the far corners and
visible near edge constrain the fit. This is a single-view approximation, not a
metric camera recalibration. See `outputs/fold_tshirt_setup/table_fit.json`.

The region image uses a 120 by 60 cm table and an origin at the near-left corner:

| Region | Image/table X (cm) | Image/table Y (cm) |
|---|---|---|
| R1 | 40–55 | 15–25 |
| R2 | 65–80 | 15–25 |
| R3 | 65–80 | 35–45 |
| R4 | 40–55 | 35–45 |

Coordinates scale with the fitted table, then convert to the centered tabletop
frame: `x_table = L/2 - x_real*L/1.2`, `y_table = W/2 - y_real*W/0.6`.
Each reset draws one region uniformly, then a supported release origin uniformly
within it. The same seed reproduces the same release pose and region selection;
bitwise deterministic GPU settling is not promised. Infeasible regions fail instead
of silently switching to a different region.

The garment starts tilted 75 degrees about the table's long axis, with its lowest
point 35 cm above the tabletop and zero velocity. It falls for 4 seconds while the
robot holds its task initialization pose. Only then is managed step 0 returned.
The response records the selected region, release pose and measured post-initialization cloth
bounds. Reset physics is not charged to the policy's step budget.

## Evaluation clock and result

Atlas still sends one batch per 0.01 s control step. This experiment uses
`physics_substeps: 4`, so each batch advances four 0.0025 s PhysX ticks. The reset's
400 settling control periods likewise advance 1,600 physics ticks. Other experiments
default to one physics tick per control step.

The production budget is 100 episodes, 6,500 control steps per episode, base seed 0.
Success predicates are deliberately empty, as requested. Evidence remains `unknown`;
budget completion is `evidence_insufficient` in Atlas, not a successful fold.
Rigid-body grasp-contact diagnostics do not measure particle-cloth contact pairs;
this task does not use those diagnostics to infer success or gripping force.

## Validation

`tests/unit/test_cloth_scene.py` checks region selection, infeasible regions,
material validation and the managed substep clock. The setup evidence directory
contains the fitted table/region mapping, image captures and a short isolated ROS
smoke batch. The smoke test uses actual Atlas `SimulationRPC`/`SimulationEpisode`
with the actual Isaac scene, but a hold-action adapter; it does not validate a VLA
policy, native SDK action conversion or real hardware. Existing initial-pose tests
referencing the independently deleted `press_cube.json` still fail for that reason.

Near-robot regions can contact the open grippers during the fall. Collisions remain
enabled; the four-region capture sheet records the actual outcomes, not guaranteed
contact-free tabletop poses.
