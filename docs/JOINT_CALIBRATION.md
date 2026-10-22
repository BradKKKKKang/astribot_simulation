# Isaac Lab 100 Hz Joint Response Calibration

本文记录当前使用 100 Hz 指令回放对齐 Isaac Lab 与真机关节响应的目标、方法、证据和未完成工作。它取代下方历史上 250 Hz SDK 回放计划作为当前执行说明。

## 项目目标

对三条真机 replay 使用同一条 100 Hz 关节位置指令时间轴，在 Isaac Lab 中按 10 ms 仿真步长执行，调整仿真关节控制与动力学参数，使仿真的关节位置和速度响应接近 Parquet 中对应时间的真机反馈。

拟合目标与运行时目标分开：

- **仿真时间响应**：比较相同 command frame / physics time 上的仿真与真机状态，用于评估关节动力学及控制参数。
- **墙钟实时性**：判断每个 10 ms 步是否能在 10 ms 墙钟期限内完成。当前回放记录了超时，但不保证硬实时。
- **SDK/ROS 链路**：命令发布、排队、传输和反馈订阅延迟尚未纳入当前拟合；需要单独验证。

旧计划中的角关节指标暂作为项目目标：三个 episode 的 20 个角关节位置 RMSE 小于 0.01 rad，速度 RMSE 小于 0.05 rad/s。所有结果仍须同时报告逐关节及逐 episode 数值，避免聚合指标掩盖单关节问题。夹爪使用百分比点和百分比点/秒计量，旧计划没有定义对应阈值；力矩相对误差目标暂停，直到真机与 Isaac actuator 力矩语义得到确认。

## 当前进度

| 阶段 | 状态 | 当前证据 |
| --- | --- | --- |
| 指令重采样 | 完成 | 三个 episode 都有共享 100 Hz command 网格；原始反馈行和来源元数据保留 |
| Isaac 物理基线 | 完成 | 三条轨迹均按 dt=0.01 s、frame_skip=1 完整执行，每帧对应一个物理步 |
| 墙钟时限检查 | 完成但有超时 | 三条轨迹分别记录 5、34、20 次完成期限超时；这不是硬实时保证 |
| 关节参数拟合 | 未完成 | 当前仍是统一 stiffness=800、damping=40、effort_limit=400，没有基于三条轨迹拟合的新参数 |
| 调参后交叉验证 | 未完成 | 尚无调参配置下的三轨迹结果 |
| SDK/ROS 传输验证 | 未完成 | 当前基线不经 SDK/ROS 命令通路，也不验证 ROS 反馈延迟 |

实现和结果：

- 重采样器： [scripts/resample_replay_commands.py](../scripts/resample_replay_commands.py)
- 同进程物理回放： [scripts/replay_artifacts_100hz.py](../scripts/replay_artifacts_100hz.py)
- 输入保留和裁边记录： [artifacts/resampled_100hz/manifest.json](../artifacts/resampled_100hz/manifest.json)
- 结果汇总： [results/alignment_100hz/summary.md](../results/alignment_100hz/summary.md)
- 离线结果审计： [results/alignment_100hz/verification.json](../results/alignment_100hz/verification.json)
- 运行来源、逐 episode 指标、deadline 和 hash： [results/alignment_100hz/run_manifest.json](../results/alignment_100hz/run_manifest.json)

## 数据与比较语义

三个原始 artifact 的 command 流约为 250 Hz，关节间时间戳略有偏差。重采样器从每个关节记录的 command 时间戳进行线性插值，取所有 command 流的共同覆盖区间并生成共享的 10 ms 网格。裁边和输出样本数记录在 manifest。原始 state 行不重采样、不覆盖；采集元数据另行保留。

当前回放使用 22 个关节的 position-mode command。虽然重采样 artifact 中保留 position、velocity、torque 和 acceleration 字段，物理基线只施加 position target，并将 velocity 和 torque command 设为零。真机 state 用于设置首帧的仿真位置/速度，并用于离线误差比较。

回放执行与比较方式：

1. 每个 command frame 调用一次完整 Isaac Lab env.step；每次物理步 dt=0.01 s，frame_skip=1。
2. 保存初始状态及每次 step 后的状态，因此 N 个 command frame 对应 N+1 个仿真状态。
3. 误差计算将真机反馈线性插值到仿真 physics time 对应的 artifact 时间；比较范围裁在 command 与真实 state 的共同覆盖区间内，末尾 post-step 状态不纳入误差。
4. 位置、速度和单位分别按 20 个角关节及 2 个夹爪关节统计。夹爪位置单位是百分比点，不是弧度。

基线状态采样直接读取 Isaac articulation tensor。它没有通过 SDK 发送 command，也没有等待 ROS command/state topic。因此，这些结果用于隔离评估 position-only 仿真响应，不代表 SDK/ROS 端到端结果。

## 仿真时间与硬实时

当前拟合按仿真时间而非实际墙钟时间配对。每个 command frame 仍使 PhysX 时间前进约 10 ms，且验证记录确认每条轨迹的 physics step 数与 command frame 数相同。脚本使用绝对墙钟 deadline 统计超时；落后 deadline 时不会丢弃 command frame，但后续 step 可能在墙钟上连续赶进度。

因此，在物理模型只依赖仿真时间、command frame 和固定物理步长的前提下，墙钟超时本身不会改变每步 10 ms 的仿真动力学时间轴，不应直接混进 position/velocity 参数拟合误差。它仍会影响实时运行能力：当前证据不能证明完整环境能硬实时达到 100 Hz。若目标包含真实 SDK/ROS 100 Hz 执行，必须另测每帧发布时间、接收时间、反馈延迟、丢帧和 deadline miss，并将观测到的时延与 actuator 参数区分开；当前没有显式估计或补偿 command-to-state latency。

## 100 Hz 物理基线

下表中的 RMSE 是跨 20 个角关节或 2 个夹爪关节的组汇总。完整逐关节数据位于各 episode 的 tracking_error.json。

| Episode | Command 帧 | 角关节位置 RMSE (rad) | 角关节速度 RMSE (rad/s) | 夹爪位置 RMSE (百分比点) | 夹爪速度 RMSE (百分比点/s) | 完成期限超时 | 单步耗时 >10 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| replay_train_20261020_213902 | 1,584 | 0.022087 | 0.062075 | 11.223891 | 63.478921 | 5 | 3 |
| replay_train_guess | 2,314 | 0.020888 | 0.056957 | 6.236967 | 33.184189 | 34 | 21 |
| replay_train_pour | 3,597 | 0.022584 | 0.058732 | 6.430199 | 30.658478 | 20 | 13 |

角关节基线尚未达到上述旧计划目标。三次回放均完整结束；完成轨迹不等于每个墙钟 deadline 都满足。首次受 GPU 并发影响的结果保留在 results/alignment_100hz_diagnostics/，不纳入正式汇总；第一个 episode 使用独占 GPU 重跑结果。

## 接下来的标定工作

1. 固定拟合契约：确认 position-only 是否是目标控制语义；确认 command 与 state 的 artifact 时间戳可直接配对，或先估计硬件 command-to-state 延迟。保持角关节、夹爪指标分开。
2. 将当前硬编码的统一 actuator 参数改为可审计的关节组或逐关节配置，并保留 800/40/400 作为基线。MuJoCo 参数只能作为候选参考，不应未经验证直接映射为 Isaac Lab ImplicitActuatorCfg 参数。
3. 基于基线逐关节误差曲线识别滞后、超调、稳态误差和 effort saturation，再对训练 episode 做有界参数搜索。每轮记录配置、模型、输入 artifact 的版本或 hash，以及原始逐关节指标。
4. 留出 episode 或时间窗口作验证，再用三条轨迹复核最终参数；汇总位置/速度误差、振荡和每个 episode 的 deadline 结果。
5. 若需要宣称 SDK/ROS 下的 100 Hz replay，再增加独立的端到端验证；不要用同进程物理基线替代该结论。

当前 checkout 中没有 scripts/calibrate_joints.py。2026-10-20 开发记录提到曾尝试独立 Isaac 标定入口，但 Isaac Kit 在写出结果前退出，端到端标定未验证；该尝试不计为已完成拟合工具。

## 变更记录

| 日期 | 内容 |
| --- | --- |
| 2026-10-22 | 用已验证的 100 Hz 重采样和同进程物理基线替换旧 250 Hz SDK 计划；区分仿真时间拟合、墙钟期限和 SDK/ROS 验证。 |
