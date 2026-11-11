# Reference tabletop layout

Launch the scene in the configured Isaac Lab/ROS environment:

```bash
python astribot_simulation_integrate.py --config config/experiments/evaluation/item_classification.json
```

For configuration-only validation, add `--check-config`. This scene uses the
shared white table mesh with instance scale `[6/7, 6/7, 1]`: a 1.20 × 0.60 m
tabletop at 0.75 m height. Its world XY center is `[0.625, 0]`. The support
surface and collision geometry follow the same scale. The shared table asset,
penholder scene, black enclosure and lighting are unchanged.

The rear row contains the steamer on the left and purple plate on the right.
The front objects, left to right, are drumstick, bun, bread, bear and pig. The
bear lies face up. Positions and in-plane angles were refined against the
120 × 60 cm reference (`screenshot-20261105-161633.png`) using the head-camera
projection and rendered previews. These are image estimates, not calibrated
3D measurements. Every object is grounded from its actual mesh
with 1 mm clearance above the table. Edit the `placement` entries in
`config/scenes/tabletop_objects.json` to adjust the layout; the tabletop frame
has +X toward the image left and +Y toward the robot.

The objects reference metric pass-6 GLBs in `~/assets`; the bear and pig use the
fur-free derivatives described below. The purple plate references the existing `panzi`
pass-6 GLB at `~/assets/panzi.glb`; its diameter is assumed to be 20 cm, with uniform scaling (about
1.8 cm high). Source files are hash checked. The bear uses the fur-free model
with a narrowed blue torso and the preceding pink-head shape. The torso is
3.266 cm wide; the pink head remains 4.879 cm wide and 5.918 cm deep. Overall
bounds are approximately 6.693 × 9.923 × 6.463 cm in glTF XYZ, including ears
and nose.
The bear uses a PhysX volumetric deformable body;
the other four small objects use independent convex-hull rigid bodies, and the
two open containers use SDF collision. Masses, friction and elastic parameters
are simulation assumptions.

### Fur-free pig

The shared `config/assets/wanjuzhu.json` loads
`config/assets/meshes/wanjuzhu_smooth.glb` for the base scene and fixed layouts
01–10. `scripts/strip_wanjuzhu_fur.py` reuses the bear's GLB compactor to remove
14 separate terry-loop nodes from the body, snout, ears and tail. It preserves
every remaining vertex, face, normal, color, material and node transform,
including the padded body, facial details, satin back stitches and care label.
The unused original `~/assets/wanjuzhu.glb` was removed during the 2026-11-09
asset cleanup; the current smooth model remains installed in the repository.

Triangles decrease from 5,064,758 to 106,838 (97.89%), vertices from 2,645,631
to 53,991, and the GLB from 113,518,632 to 1,838,504 bytes (98.38%). The glTF
XYZ bounds change from 100 × 50 × 50 mm to approximately
99.445 × 50 × 49.713 mm solely because the surface loops are removed.
There is no compensating resize. Its rigid-body convex-hull mode, 15 g mass
and static/dynamic friction 2.0 remain unchanged. Layout XY and rotations are
preserved; grounding uses the new mesh with 1 mm tabletop clearance.

The runtime profile `config/assets/wanjuzhu.json` records the installed model's
path, digest, axis transform and physics. The separate generation sidecar
`meshes/wanjuzhu_smooth.json` was removed during cleanup; it was not used to
load the scene. To regenerate a derivative, first restore the original
metric source and pass its path explicitly:

```bash
python scripts/strip_wanjuzhu_fur.py \
  --source /path/to/restored/wanjuzhu.glb --output /tmp/wanjuzhu_smooth.glb
```

All ten fixed layouts and the base configuration pass CPU resolution and
grounding checks. Before source cleanup, the bear's stripping path reproduced
its archived smooth-asset hash exactly. No new GPU render or end-to-end FPS
comparison was run. Restart the simulation to load the new geometry;
Reset does not reload assets.

### Fur-free elastic bear

Both the un-numbered scene and fixed layouts 01–10 select
`config/assets/wanjuxiong.json`, which loads `~/assets/wanjuxiong_smooth_torso_w3p266.glb`.
The profile enables a PhysX FEM deformable body, tetrahedral collision, mass
15 g and dynamic friction 2.0. Each layout retains its XY/orientation and
recalculates grounding from the remaining vertices with 1 mm clearance.

The initial smooth GLB derives from the restored `~/assets/wanjuxiong_rigid.glb`.
`scripts/strip_wanjuxiong_fur.py` removes the eight `_ShortPlushPile` meshes
and compacts the binary buffer. Every retained vertex, face, normal, color,
texture and node transform is preserved; the smaller bounds reflect only the
removed fibers. Earlier revisions narrowed and thickened the pink head.
The active variant reduces only blue-torso width by 1 cm (4.266 to 3.266 cm)
about its center. Arms move inward about 0.573 cm each, and legs about
0.252 cm each, preserving their sizes and attachment to the torso. All head
vertices, head-to-feet length and part depths stay unchanged. The current
asset retains its provenance/dimension sidecar. Earlier variants and their
sidecars were removed during the 2026-11-09 cleanup. To repeat the narrowing
operation, restore the preceding variant and provide it explicitly:

```bash
python scripts/resize_wanjuxiong.py \
  --source /path/to/restored/wanjuxiong_smooth_head_w4p879_d5p918.glb \
  --torso-width-decrease 0.01 \
  --output /tmp/wanjuxiong_smooth_torso_w3p266.glb
```

Triangles drop from 2,093,280 to 113,280 (94.6%); vertices drop from 1,444,396
to 58,396, and the GLB shrinks from 56.36 MB to 3.27 MB. This reduces geometry
loading, rendering and deformable-skin update work. The FEM core already
excluded pile, so removing it does not eliminate elastic simulation cost.
No end-to-end FPS comparison has been measured.

Restart the simulation to load this asset; Reset does not reload it.
Only the active `wanjuxiong_smooth_torso_w3p266.glb` and its sidecar remain
in `~/assets` for the bear. The unused rigid profile was also removed from
`config/assets`. Artos provides static geometry, not measured elastic
material parameters.

### Elastic behavior and validation

Head, torso, ears and limbs form the remeshed tetrahedral core. The retained
colored surfaces, nose and hanging loop share one deforming
render mesh with material subsets. They follow the FEM motion. The nose is
treated as part of the soft skin, not a separately rigid plastic insert.
The initial dimensions and scene placement still come from the metric GLB.

The homogeneous elastic approximation uses Young's modulus 30 kPa, Poisson
ratio 0.35, elasticity damping 0.05 and the configured mass of 15 g.
Meshing resolution 20 and 255 position iterations are retained at the scene's
10 ms physics step. The 15 g/narrow-torso variant fails the deep-compression
element-orientation check: some tetrahedra invert. Resolution trials at 24
and 32 also failed, so the extra numerical cost was not retained. These
settings approximate a compressible plush toy; they
do not model a separate fabric shell, stuffing fibers or material hysteresis.
Self-collision is disabled; contacts with the table, press and robot remain
enabled. Strong folding and robot grasps have not been validated.

CUDA physics is required. Reset restores the complete nodal positions and
velocities and releases all nodal constraints; PhysX refreshes the collision
and render skin on the next physics tick. A paused camera capture immediately
after reset may therefore show the preceding skin until a step is taken.
Evaluation receives a best-fit material-frame pose and a deformation residual,
not a rigid-body root pose. Existing rigid-body grasp-force diagnostics exclude
the bear.

Run the isolated, ROS-free compression/release check with the Isaac Lab Python:

```bash
python scripts/validate_deformable_bear.py --captures \
  --output /tmp/wanjuxiong-soft-check
```

The report includes finite-state, element-inversion, support-contact, recovery
and reset checks. PNG checkpoints show the initial, settled, compressed,
released and reset shapes. See `verification.json` in the output directory
for actual results. At 15 g/friction 2.0, the isolated RTX 3090 check gives
38.11% compression, 99.48% recovered height and zero nodal reset error, but
minimum element Jacobian -0.11448 means the overall check **failed**. Recovery
alone does not validate strong squeezing. Evidence, including failed finer
mesh trials, is in `outputs/evaluation/wanjuxiong_torso_minus1cm_15g/`.
The previous head variant's passing report applies to 40 g and a wider torso.
No live robot pinch or full-scene performance benchmark was run.
The adapter follows the installed Isaac Lab 2.1.1 API and
the [Isaac Lab deformable-object workflow](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/tutorials/01_assets/run_deformable_object.html).

### Slip during a head pinch

At 15 g, gravity is 0.147 N instead of 0.392 N at 40 g, a 62.5% reduction.
For ideal opposing vertical contacts, friction must supply at least that load;
lifting acceleration increases the required support. This makes a gravity-led
slip easier to prevent, but does not establish a successful grasp.

Slipping after closing and being squeezed downward during closing are
different observations. For the latter, inspect contact height/orientation,
the curved head surface, closure amount/speed and deformation. A pinch on
sloped surfaces can exert downward force; excessive squeezing can move the
contact patches or destabilize a coarse FEM mesh. Increasing friction alone
does not correct those effects. Keep one variable fixed while comparing
grasp position, closure and material parameters.

The bear adapter authors dynamic friction 2.0. A read-only audit of the robot
source USD found four fingertip colliders with no bound physics material.
This does not determine the live engine defaults or effective contact-pair
friction. The existing grasp diagnostic filters only rigid scene objects,
so it provides no measured bear/finger forces. Audit evidence is in
`outputs/evaluation/wanjuxiong_torso_minus1cm_15g/gripper_material_audit.json`.
No recorded live slip trajectory or robot pinch was evaluated in this change.

`config/tasks/tabletop_objects.json` snapshots frame 0 of the user-selected
HDF5 command dataset, with its source SHA-256. It preserves the recorded arm,
torso, head and gripper targets; chassis placement follows the penholder
experiment's fixed base. Success criteria are temporarily explicit and empty
(`success.predicates: []`), so task evidence stays `unknown` and no episode is
automatically counted as successful. Managed evaluation uses 100 episodes and
seed 0, matching penholder, with a 6500-step limit per episode (65 simulated
seconds at 100 Hz; model waiting time does not consume steps). At that limit,
Simulation reports `timeout` with unknown evidence; Atlas records
`evidence_insufficient` and advances to the next episode.

Ten independent [fixed item-classification layouts](../config/experiments/evaluation/item_classification_fixed/README.md)
are available. Layout 01 snapshots the current placement above; 02–10 follow
the supplied numbered reference photographs. Each fixed config runs one managed
episode with a 6500-step limit and disables reset sampling. In Atlas select
`simulation/item_classification_fixed_01` through `simulation/item_classification_fixed_10`.

Select `simulation/item_classification` in Atlas, then Start Simulation and
Start Eval. Both repositories must include support for explicit empty success
criteria. CPU-only admission check:

```bash
python astribot_simulation_integrate.py --config config/experiments/evaluation/item_classification.json --managed-evaluation --check-config
```

Validation artifacts for the resized scene are in
`outputs/evaluation/tabletop_objects/aligned_120x60/`: three camera images,
pinhole images and `capture.json`. The isolated Isaac Lab check used localhost
ROS domain 197, reset seed 42 and 32 render updates without post-reset physics
steps. These images establish initial appearance, not grasp or settling
stability. GLB texture sampler wrap modes are forwarded to USD so repeating
wood textures do not acquire black borders outside the first UV tile.
