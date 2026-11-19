# Eval scene support

The integrated launcher and Atlas Console share the experiment resolver. Validate
a task before starting Isaac or ROS:

```bash
python astribot_simulation_integrate.py \
  --config config/experiments/evaluation/fold_tshirt.json \
  --managed-evaluation --check-config
```

`penholder`, `item_classification`, `press_button` and `fold_tshirt` use the same
launch path. Scene assets and their source digests must be available locally.
Robot settings come from the selected robot profile; tasks override only their
declared initial pose and runtime settings.

## Assets and observations

Assets declare rigid bodies, volumetric deformables or particle cloth through
`physics.body_type`. Deformables require a GPU physics device and explicit material
and meshing parameters. Cloth requires a single sewn midsurface and explicit
`particle_cloth` parameters. Rendered cloth vertices follow the physics state;
capturing a managed observation does not advance physics. Deformable pose reports
contain a fitted material frame and deformation residual, not a rigid-body pose.

Scenes can set `simulation.ambient_light_intensity`. Asset materials support a
vertex-color matrix and glTF texture wrap modes. Optional `plumb_bob` calibration
stored on USD cameras is applied to RGB output; cameras without calibration retain
their original pixels.

## Reset and time

Task initial poses can carry Parquet state-frame or HDF5 command-frame provenance.
The stored named joint targets are restored on reset; launch does not reopen the
source dataset. Joint filter history is cleared with the robot's initial state.

`uniform_in_regions` chooses a named tabletop region using the episode seed and
samples a supported placement within it. An infeasible selected region fails;
it is not silently replaced. `cloth_grasp` temporarily pins a central patch before
release, followed by `settle_steps` while holding the robot's initial pose.
These initialization periods finish before managed step zero and do not consume
the policy budget. The fold task's current settings use 250 hold periods and 400
settling periods. GPU settling is not guaranteed to be bitwise deterministic.

`runtime.simulation.physics_substeps` accepts integers from 1 to 16. With four
substeps and a 0.01 s control period, each PhysX tick is 0.0025 s; feedback and the
Eval step counter still advance once per control period. The default is one.

An explicit empty `success.predicates` list is supported. Evidence remains
`unknown`, budget completion is a timeout, and Atlas reports insufficient evidence
instead of a successful task. Rigid contact diagnostics do not establish cloth
grasp quality. Configuration and CPU tests do not validate GPU material behavior,
rendering, policy performance or real hardware.
