# Managed synchronous evaluation

Atlas owns the batch and episode schedule. Simulation owns robot/object reset,
one physics step, measured snapshots and config-defined task evaluation.
The default integrated launcher is continuous. `--managed-evaluation` selects
ROS2/Isaac Lab control and starts paused, with no free-running physics.

## Transport and identity

Use `astribot_msgs/srv/RawRequest` at `/simulation/evaluation/control`. Both string
fields contain JSON; no message submodule changes are needed. Requests use
`schema_version: 1`, `op`, `request_id`, and optionally `payload`.
`describe` and `heartbeat` require only these first three fields. All other
operations require `run_id`, `episode_id`, `step_id`, `config_fingerprint`.

Responses contain `schema_version`, `request_id`, `status` (`accepted`,
`completed`, or `error`) and either `result` or
`error: {code, message, execution_uncertain}`. An accepted response is only a
queue receipt. Repeat the identical request with the same ID to poll completion;
reusing an ID with different content is an error. Recent completed responses are
cached (128 receipts); an older ID is explicitly rejected without re-execution.
The ROS callback only queues work. The Isaac main thread performs reset, step,
camera reads and report writes. Heartbeat works independently of physics.

`describe.result` contains `managed: true`, the configuration fingerprint,
`dt: 0.01`, `frame_skip: 1`, `evaluation: {episodes, max_steps, seed}`,
`task: {name, instruction, success_configured}`, seven feedback `parts`, and
`state: {run_id, episode_id, step_id, phase}`. Part descriptions contain
`joint_names`, `dof`, `position_unit`: radians (`rad`) for torso/arms/head,
`percent` for the two grippers, and `m,m,rad` for the fixed chassis.
The fingerprint covers the resolved launch configuration, including output and
initial pose. A changed configuration requires a new connection/run.

## Operations

- `reset`: step 0, episode starting at 1, `payload: {seed}`. Seed must equal
  configured base seed plus episode ID minus one. The first reset starts a new
  run; subsequent resets require the preceding episode to be terminal. Robot
  root pose, joints, velocities, holding targets, objects and evaluator history
  reset together. Initial pose and task predicates come from the task config.
- `step`: next consecutive step number, `payload: {dt: 0.01, commands,
  capture_images: false}`. `commands` contains exactly the six non-chassis
  parts, each with ordered feedback `joint_names`, finite `position` and
  `velocity` arrays. Gripper positions use 0–100 percent; other positions use
  radians. Partial batches, stale identities and wrong dt are rejected before
  physics. Exactly one `sim.step()` runs. FPS prediction is disabled.
- `snapshot`: current completed step, empty payload. Captures all three RGB
  images without physics advancement. Repeated full snapshots at that identity
  are immutable. Request this between chunks; ordinary step receipts can omit
  images. Reset always captures images.
- `stop`: current or in-flight step; optional `payload: {reason: "aborted"}` or
  `"error"`. Cancels queued work, prevents subsequent steps and finalizes a
  running episode. An already executing physics step may complete, then stops.
- `heartbeat`: returns alive, configuration fingerprint, monotonic time and
  current state; no physics. An optionally supplied mismatching fingerprint is
  rejected. Clients pin the returned fingerprint and, once their run starts,
  verify the owner in state.
- `replay`: `payload: {enabled: true|false}`, episode 1, step 0. Acquire only
  while idle/stopped, before creating the SDK. Release with the same owner.
  Replay enables the ordinary continuous ROS command path. Pause retains the
  lease; release follows SDK teardown on completion/Stop/failure. Released owner
  IDs are retired and cannot acquire again under a new request ID. A repeated
  release of a retired owner (or repeated Stop of a retired Eval owner) returns
  completed without changing the current owner, even with a new request ID.
  Full identity/fingerprint and operation payload validation still apply. Eval reset is
  rejected while Replay owns control. Eval stop releases its lease; the
  simulation remains paused until a new explicit owner starts. Releasing Eval
  synchronizes legacy targets to measured joints with zero velocity, so Replay
  holds the completed boundary while its SDK initializes. Ownership changes
  discard queued legacy reset/object commands.

## Snapshots and outcomes

Reset/step/snapshot completion returns `result.snapshot` with `snapshot_id`,
`config_fingerprint`, run/episode/step IDs, `sim_time_s = step_id * 0.01`, seven
measured `joints` (names, position, velocity), actual initial object
`reset_layout`, evaluator `evaluation`, and `terminal: {done, reason}`.
`images` maps `cam_high`, `cam_left_wrist`, `cam_right_wrist` to
`{encoding: "jpeg", data_base64, width, height}`, or is empty for a step without
image capture. Images and joints belong to the same paused physics boundary.
Atlas retains its desired/WBC observation-state semantics and associates that
state with the matching acknowledged boundary. Repeated ROS feedback while
paused keeps SDK liveness; it is not a new snapshot or physics step.

Evaluator `failed` and `unknown` are nonterminal. `passed` latches success and
prevents further steps in that episode. Reaching `max_steps` yields timeout
unless that same step passed. A Stop accepted before completion takes priority
over a newly observed pass and reports aborted/error with done=true; an already
latched terminal result is retained. Reasons are `success`, `timeout`, `aborted`, or
`error`; nonterminal reason is null. Evidence status is recorded separately,
so timeout with unknown evidence cannot be counted as success. Native,
communication or physics errors are execution failures, not task failures.
An uncertain execution error requires stopping the faulted run.

Each terminal episode writes its own report and updates the batch summary
immediately. Reports include owner/episode/config identity, seed, actual reset
layout, environment steps, simulated duration, wall time, evidence and reason.
Simulation never starts the next episode. GUI/application exit finalizes an
active episode as aborted. Legacy Empty reset is ignored with a warning during
an Eval lease; object reset returns an error and direct joint commands are
ignored. The managed service is the authoritative completion interface.

## Validation limits

CPU tests cover queue/retry/conflict behavior, stale and partial requests,
one-step execution ordering, reset isolation, terminal latching, cancellation,
Replay exclusion and reports. Actual Isaac cameras/physics, ROS2 DDS transport,
SDK/WBC state alignment, and full model-driven batches require isolated runtime
validation. A GUI-paused/stopped Isaac timeline fails a requested step promptly
rather than entering Isaac's blocking timeline-resume path.
