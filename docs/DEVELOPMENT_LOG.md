

## 2026-10-22：SDK 回放逐关节对齐分析（含夹爪）

- **数据与比较方法**：使用选中的躯干第一关节配置，分析三条已保存的 SDK 回放记录，每条 22 个关节，共 66 组关节／轨迹对比。按仿真消息头时间戳插值真机反馈，保留启动瞬态，不拟合时间偏移。角关节位置误差采用度，夹爪采用 SDK 0–100 标度的百分比点；速度误差采用 rad/s 或百分比点/s。力矩等价性尚未验证，不纳入比较。
- **躯干持续偏差**：按 20261020_213902 / guess / pour 的顺序，躯干第二关节位置 RMSE 为 5.008 / 4.925 / 4.027 度，仿真减真机的平均偏差为 −4.910 / −4.837 / −3.883 度；第三关节位置 RMSE 为 1.650 / 1.114 / 1.728 度。这些偏差持续存在，并非只有孤立峰值；第一关节调参不代表全身已对齐。
- **夹爪差异**：左夹爪位置 RMSE 为 16.084 / 8.284 / 4.322 个百分比点，瞬时最大误差为 75.664 点；右夹爪为 0.131 / 1.502 / 6.947 点，瞬时最大误差为 51.155 点。曲线显示开合过渡阶段差异明显，其中 20261020_213902 的仿真左夹爪响应慢于真机。这些问题在第一关节调参前已存在，本轮未确定唯一根因，也未修改夹爪控制器。
- **其他偏差与后续顺序**：头部第二关节、左臂第一／第四关节和右臂第四关节也有剩余位置偏差。建议后续依次检查躯干第二关节、第三关节和夹爪瞬态响应；本轮分析未继续调参。
- **结果产物**：完整逐关节表、各轨迹的位置／速度误差、带符号偏差、p95、峰值时间、样本数及与基线的差异，见[分析报告](../results/torso_joint_1_tuning/joint_alignment/report.md)、[CSV](../results/torso_joint_1_tuning/joint_alignment/per_joint.csv) 和 [JSON](../results/torso_joint_1_tuning/joint_alignment/per_joint.json)。[对比曲线](../results/torso_joint_1_tuning/joint_alignment/largest_mismatches.png) 覆盖三条轨迹的躯干第二关节及双夹爪。
- **验证与边界**：重新计算全部 66 组位置 RMSE，并核对保存的 SDK 误差文件中的样本数；人工检查生成曲线。此前调参的最终专项回归共 30 项通过，覆盖夹爪、躯干控制、SDK 回放和仿真节拍。本轮仅分析已有记录，未新增仿真、SDK 控制、真实硬件操作或运行代码修改。

## 2026-10-22：躯干第一关节的独立 PD 与重力补偿调参

- **范围与配置**：新增可选 Isaac 配置 `torso_joint_1_control`，为该自由度配置独立隐式执行器，仅改变其增益与 PhysX 重力前馈。默认 YAML 采用 Kp=3200、Kd=320、力矩上限 400 N m、重力补偿系数 1.1；其他关节增益保持 800/40。控制语义、复现命令、曲线和耦合限制见 `docs/TORSO_JOINT_TUNING.md`。
- **物理回放筛选**：在 pour/guess 上测试 8 组初选参数和 4 组局部微调参数，再用未参与筛选的 20261020_213902 轨迹验证基线及两组候选，共完成 27 次物理回放。每帧指令推进一次 10 ms 物理步，每轮恢复记录中的初始位置／速度。该关节位置 RMSE 从 1.050/1.768/3.529 度降至 0.174/0.157/0.185 度。
- **SDK 链路验证**：使用已有 SDK 100 Hz 回放脚本，在仅限本机通信的 ROS 域 198 中，对三条轨迹分别重跑基线和选中参数。位置 RMSE 从 1.353/2.189/3.552 度降至 0.180/0.292/0.183 度；选中参数的状态观测频率为 96.31–97.55 Hz。两个本任务仿真器均以零退出码关闭，未控制真实硬件。
- **结果依据**：保存于 `results/torso_joint_1_tuning/`、其中引用的原始实验目录及 `results/torso_tuning_sdk_{baseline,selected}/`。未完成的调试目录不计入结果；隐式执行器力矩仅作诊断。
- **限制**：动力学耦合使其他 19 个角关节的综合位置 RMSE 略增，其中 pour 的躯干第二关节 RMSE 增加 0.01078 rad。SDK guess 在非零预定位残差下仍有早期 1.588 度峰值；未参与筛选轨迹的第一关节速度 RMSE 仍为 0.05244 rad/s。
- **验证**：控制专项测试共 22 项通过，覆盖既有夹爪和新增躯干用例；Python 语法、Shell 语法及空白检查通过。本结果不代表整机已标定或 SDK 反馈满足硬实时要求。

## 2026-10-22：无窗口 GPU 基准与 100 Hz 完整轨迹对齐

- **性能边界**：RTX 3090、4 ms 物理步长的 headless 长测中，完整 `env.step`（含 ROS executor、无相机）约 118.6 Hz，三相机约 29.9 Hz；physics-only 约 283.0 Hz，不能据此宣称完整回放稳定达到 250 Hz。
- **重采样与回放**：新增 `resample_replay_commands.py` 与 `replay_artifacts_100hz.py`；三组记录各生成 1,584 /2,314 /3,597 个 22 关节共用 100 Hz 指令帧，保留全部原始反馈与来源元数据。完整 `env.step` 采用 dt=0.01 s、frame_skip=1、headless、关闭相机，保存 N+1 状态及实际物理时间；共同覆盖区间裁边与最终网格取整记录在 manifest。
- **结果与验证**：三轨迹全部完成（进程退出码均 0），10 ms 完成期限超时分别 5 /34 /20 次；首组采用独占 GPU 重跑，原并发 GPU 诊断结果保留但排除。仅建立同进程 position-only 物理基线，未验证 SDK/ROS 传输，未调整 800/40/400 PD；力矩仅作诊断。汇总与哈希/状态/误差复算审计见 `results/alignment_100hz/`，离线测试 4 项通过。

## 2026-10-15: Isaac Lab gripper dataset units

- Reference: `cognition_press_three_buttons`, episode 000000 (326 frames).
  Metadata identifies joint gripper columns 14/22 and Cartesian gripper columns
  18/28. Both command representations use the 0–100 scale; the left command
  spans the full range and the right command stays at zero.
- Root cause: Isaac Lab treated these commands as radians and clamped positive
  values to zero in `[-0.5, 0]`. Convert the compensated command once from
  0–100 to 0–0.93 rad, then apply each USD/URDF follower joint's direction.
  This matches the existing MuJoCo actuator gain/bias ratio and Genesis /
  ManiSkill position-command conversion; those backends are unchanged.
- Convert Isaac position, velocity and acceleration feedback with the same
  signed scale. Preserve small out-of-range physical feedback instead of
  folding it through absolute value. Fix velocity-mode scalar indexing so
  either gripper can advance from its feedback position in dataset units.
- Validation: `astribot_simu_il` Python, PyTorch 2.7.0+cu128, CPU tensors;
  `python -m pytest tests/unit/test_isaaclab_gripper.py -o addopts='' -q`: 11
  passed. Tests execute actual control/feedback and base compensation methods
  with mocked ROS/Isaac startup dependencies. Both sides, six follower signs,
  endpoints, intermediate values, clamping, compensation and signed feedback
  are covered.
- Read the reference parquet with PyArrow in `atlas`, then checked all 326
  command pairs through the same CPU control methods. Maximum command-to-
  target-to-feedback error: 3.95e-6 scale units; left L1 targets span
  -0.93–0 rad, right L1 remains zero. No dataset files were modified.
- Scope: syntax and whitespace checks only beyond these focused checks; no
  Isaac physics/GUI, ROS transport, other-backend runtime or hardware run.
  Existing unrelated workspace changes are preserved.

## 2026-10-15：回放运行依赖与 SDK 控制节拍

- **依赖定位与修复**：在加载 `astribot_sdk_ros2/env.zsh` 的系统 Python SDK 环境中复现缺少 PyArrow；在仿真器 `astribot_simu_il` Python 环境中复现缺少 `cryptography`，原生 SDK 将后者掩盖为 `robotics_library_py ... is not a package` 导入错误。为本地系统 Python 安装 PyArrow 17，新增 `scripts/requirements-replay.txt` 以复现依赖安装。
- **启动检查**：等待仿真前先检查 `cryptography.fernet` 并导入 SDK；缺少依赖时明确指出解释器和安装命令。SDK 客户端仍在仿真及相机就绪检查后构造。
- **控制节拍**：初始位姿渐变和数据集目标均以 100 Hz 发送，与 SDK 频率一致，避免将 100/30 Hz 调用与 250 Hz SDK 混用。在相邻数据时间戳之间保持当前目标，跳过已延误的控制节拍。
- **文档**：恢复缺失的回放设置说明并从 README 链接，分别说明仿真器和 SDK 的 Python 环境，以及加载 SDK `env.zsh` 后需要设置的 ROS 环境变量。
- **离线验证**：18 项 pytest 测试、语法及空白检查通过。
- **实际回放验证**：使用真实 326 帧、10.833 秒轨迹，在 ROS2 Humble、`astribot_simu_il` 中的 Isaac Lab，以及加载 SDK `env.zsh` 的 `/usr/bin/python` 环境下回放。域 187 通过仅限本机发现和回环 UDP 传输隔离。六个可动部件各收到 1,386 条指令，最终指令与数据集的误差为零；每个部件产生 211–212 个包含运动的反馈样本，最终躯干／双臂／头部反馈误差不超过 0.076 rad，夹爪不超过 0.125 SDK 单位。观察器收到头部／左腕／右腕 RGB 图像分别为 64/94/106 帧。回放、查看器和本任务仿真器均退出，回放与仿真器退出码为零；本次未出现调用频率不匹配警告。
- **限制**：SDK 内部期望状态同步订阅仍记录 QoS 可靠性警告，但未阻止仿真器或观察器接收指令；本次没有修改外部 SDK。仅验证该 Isaac Lab/ROS2 环境，未验证真机或其他后端。
