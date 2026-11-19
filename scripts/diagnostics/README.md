# Controlled T-shirt grasp sweep

`tshirt_grasp_sweep.py` runs right-hand pinch/lift diagnostics through the existing
managed ROS control service. It consumes a **previously recorded managed command
trace**, not raw dataset actions. Each condition owns a fresh Isaac process and
an empty loopback-only Fast DDS domain. Trials must run serially.

The default matrix contains eleven trials: 90%, 93%, 95.5%, 98%, and 100% closure;
world X/Y shirt translations of ±10 mm at 95.5%; and two more baseline repeats.
Translating the flat shirt changes the relative grasp location while preserving
the recorded robot approach. It also changes contact history during approach.
The gripper's actual gap is measured from cooked colliders, not inferred from
closure percentage. Cloth/contact/actuator settings stay unchanged.

Each trial executes the same 850-command approach, closes for 0.3 s, holds for
0.5 s, raises the wrist by 20 mm over 2 s with feedback FK/IK, and holds for 1 s.
The script records a state after every 100 Hz control step, particles at 20 Hz
from closure onward, and synchronized overview/closeup frames at 20 Hz. Extra
cameras are diagnostic-process instrumentation; onboard camera images are
reduced to 128×96 to limit resource use. The two diagnostic cameras are 640×480.
No pinning, attachment, or substitute force is introduced.

Use the installed Isaac Lab Python, the ROS2 Humble and message overlays, and
`PYTHONNOUSERSITE=1 PYTHONUTF8=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`.
The client also requires the neighboring Atlas checkout for its managed RPC and
DDS helpers, and a CPU USD installation for FK (`--usd-dependencies`). These
paths are explicit CLI options; this script does not install dependencies.

```bash
python scripts/diagnostics/tshirt_grasp_sweep.py prepare \
  --output outputs/diagnostics/my_sweep \
  --source-trace /path/to/managed/episode/steps.jsonl

python scripts/diagnostics/tshirt_grasp_sweep.py run-case \
  --output outputs/diagnostics/my_sweep --case baseline
# Run each remaining manifest condition serially after the previous process exits.
python scripts/diagnostics/analyze_tshirt_sweep.py outputs/diagnostics/my_sweep
python scripts/diagnostics/export_tshirt_sweep_videos.py outputs/diagnostics/my_sweep
```

A run refuses to overwrite an attempted trial. Keep failed attempts as evidence
and prepare a fresh output directory for a new complete sweep. `--timeout` bounds
the client's operations; cleanup signals only the child process group it owns.
Do not stop an existing user simulation without authorization.

Outputs include `manifest.json`, per-trial configuration, `run_status.json`,
`interaction.mp4`, `video.frames.jsonl`, raw frame JPEGs, particle snapshots,
managed traces, per-trial `analysis.json`, and aggregate `summary.json`/CSV.
The video exporter converts to H.264, checks 247 decoded frames and distinct,
moving camera panels, and creates closure/position comparison videos for
simulation times 7.5–12.3 s. Failed or incomplete runs must not enter comparisons.

Analysis tracks fixed particle IDs within each condition. Translated garments
may have different weld counts; particle IDs are not compared across placements.
Bilateral candidates use the 6 mm cloth contact band and are **not contact forces**.
Local patch statistics use particles within 15 mm of the material point nearest
the cooked finger hull centroid before changing closure (step 850); displacement is relative to the closed-hold boundary (step 930). Joint incoming
wrenches include gravity, inertia and other contacts. None of these measurements
proves a full-garment pickup or folding success. Missing cloth-to-articulation
reaction remains a separate engine limitation under investigation.

The managed episode ends at its configured 1,230-step budget. Its terminal reason
is therefore `timeout`; this expected experiment boundary is different from a
ROS RPC timeout or an initialization failure. Only complete command/state traces,
successful process cleanup and validated videos enter the final comparisons.

## Reaction-path experiment

`tshirt_reaction_probe.py --output /path/to/new/impact` first validates a native
reaction path without ROS: a collision shell is fixed to an undriven sliding
articulation link. The paired modes differ only in
`physics:excludeFromArticulation` on that fixed joint. Empty and rigid-payload
lanes are negative and positive controls for the cloth-impact lanes. Run with
the installed Isaac Python and a bounded external timeout.

`tshirt_grasp_sweep.py prepare --reaction-study ...` prepares eight trials at
95.5% closure and a −10 mm world-X shirt translation, selected from the prior
geometry sweep: native reference, articulation shells, ordinary rigid shells,
one repeat of each shell condition, a secondary pair with actual-gap feedback,
and a repeat of the reaction-enabled gap-feedback condition. The remaining CLI and loopback-only
execution protocol are unchanged. Run each manifest case serially.

The two right-finger colliders move to internally referenced collision shells;
each original link and shell receives half of the original mass and diagonal
inertia, with the same center of mass and principal axes. This preserves their
combined rigid inertia. Both modes have the same native fixed joints, collision
geometry, material and adjacent-body collision exclusions. Excluding a fixed
joint makes its shell an ordinary dynamic rigid body, allowing particle reaction
to propagate through that joint. No cloth particle is pinned or attached, and
no estimated or synthetic force is applied. The native reference checks changes
caused by introducing shells. This diagnostic intervention does **not** enable
a general engine-wide two-way interaction setting or modify production assets.

Run `analyze_tshirt_reaction.py /path/to/grasp --videos` after completion. It
measures the actual shell poses and gaps, audits shell-to-finger error, compares
initial particle identities and cooked geometry, and tracks one shared material
patch when initial particle arrays match exactly. It also reports divergence
before closure: reaction is active for the whole trajectory, including approach,
so this is not an intervention applied to an identical saved contact state.
The paired modes use different native rigid/articulation solver paths; their
results must not be interpreted as a pure contact-force gain sweep. Repeats
measure deterministic repeatability, not grasp success probability.
In the tested Isaac 4.5 environment the robot articulation requests 8 position
and 4 velocity iterations, while ordinary rigid bodies retain schema defaults
of 16 and 1. Changing solver participation can therefore also change effective
iteration scheduling. The probe establishes reaction transfer; it does not
remove all numerical differences between those solver paths.

The secondary pair uses the completed `art_shell` gap trace as a reference.
From step 851 onward, the same feedback controller adjusts closure percentage
in both arms (0.2 percent/mm per 10 ms, maximum 0.25 percentage-point correction
per step, with 0–100% limits and anti-windup). It records every observation and
command in `gap_feedback.json`. Analysis reports actual gap tracking error and
command saturation. The first 850 approach commands remain unchanged, so this
controls the later closure/lift gap rather than erasing prior contact-history
differences. Never describe a saturated or inaccurate run as gap-matched.

The physics-invariant tests in `tests/unit/test_tshirt_reaction.py` require CPU
USD for the scene-authoring test; the material-slip test needs only the ordinary
analysis dependencies. No ROS or GPU simulation starts during those tests.
