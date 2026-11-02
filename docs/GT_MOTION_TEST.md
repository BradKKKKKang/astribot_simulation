# GT motion test

The `gt_motion` evaluation configuration uses the penholder S1 control settings and initial pose in an empty scene. It requires the Atlas GT dual-WebSocket source, 14 chunks of 32 waypoints at 30 Hz and 1498 acknowledged 0.01 s steps. Empty task predicates are allowed only with explicit `task.motion_test: true`; the task result remains unknown. Budget completion is not manipulation success.

Third-person recording runs on the simulation main thread after completed steps, at simulation-time deadlines. Rendering never advances physics. Use a fresh output path for each run; existing videos are rejected. The metadata and frame sidecar record timing. Hardware and robot-mounted observation cameras are unchanged.
