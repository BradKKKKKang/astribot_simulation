# 开发日志

本日志供agnt记录工作内容。最新改动放在文档最前面。

## 2026-10-23：统一数据集回放、反馈同步与按压事件提取

合并当日 8 条记录及回放说明、同步分析；重复改动以原日志靠上的记录为准。

### 架构与入口

- **完整链路**：原始 25 DOF Parquet → Astribot SDK → ROS → 独立 Isaac Lab 进程 → ROS 关节反馈录制 → CPU 同步／USD FK／TCP 事件检测 → 静态 Web 查看器。事件位置来自实测关节反馈与模型运动学；ROS header 使用墙钟，不等同于 PhysX 时间。
- **入口收敛**：`scripts/replay_extract_event.py` 替代原回放入口；候选提取与模型导出分别移入 `src/placement_analysis/extraction.py`、`robot_export.py`。默认串联校验、回放、录制、SDK／相机窗口清理和事件包导出；`--extract-only`、`--export-robot-only` 复用 CPU 流程，`--extract-python` 分离 SDK 与 USD 依赖环境。修改检测参数可直接重提取已有录制。
- **数据与展示**：每个 episode 保存 `sim_states.parquet`、`api_commands.parquet`、`run_metadata.json`，批次由 `batch_manifest.json` 管理；导出包包含 `placement.json`、`events.parquet`、`metadata.json`、`robot.glb`。常驻网页 `http://10.11.12.73:4173/` 只读取静态数据，新包需前端 build 后加载；固定使用 S1 USD 导出的真实模型，缺失或 body 不匹配直接报错。SDK Cartesian 接口虽存在，但当前捕获未记录 endpoint 状态，离线提取继续使用 USD FK，tool／TCP 语义尚未数值对照。

### 主要代码改动

- **回放调度与运行参数**：`replay_extract_event.py` 默认 `direct`，原始帧、保持指令及预定位共用最小 10 ms SDK 调用间隔；保留全部源帧和实际调用时间，延迟后按节拍发送，避免突发追帧，因此高于 100 Hz 的源数据或 SDK 阻塞会延长回放。保留可选 `move` 点到点模式。`astribot_simulation.py` 新增 `--no-cameras`，仅覆盖当前进程的相机列表；配合 `--headless` 关闭渲染与 GUI，不修改默认 YAML、`frame_skip` 或其它参数。
- **反馈收尾与同步**：`replay_capture.py`、`ros_feedback.py` 及回放入口统一命令窗口和订阅生命周期：末尾保持订阅 drain 0.1 s，再最多等待 2 s 让各部件 header 覆盖末条命令；窗口终点仍取最后一次 dataset SDK 调用结束时间。同步到躯干第一关节的 header 时间，裁剪无法同步的首尾，记录命令窗口、同步窗口、裁剪量及缺口。最大插值跨度仍为 0.1 s，内部缺口保留并切断检测；缺失关节、非法时钟或采样不足仍失败。该策略取代早期“首尾覆盖不足即整段失败”的实现；实录曾出现命令窗口 `[0, 19.405]s`、有效窗口 `[0.159, 19.105]s` 和 8 个内部缺口。
- **失败留存与恢复**：同步失败仍保存原始 Parquet 和 `status: failed` 元数据。`--resume` 校验来源、设置及捕获哈希，跳过完整录制；匹配的失败 episode 先归档到 `_failed/` 再重试。回放前检查提取依赖、模型／TCP 配置和输出冲突；不完整批次不自动提取，非空事件包目录拒绝覆盖。
- **终端进度**：使用 `tqdm` 显示 episode／帧进度，固定单行、最多每 0.5 s 刷新；非 TTY 自动关闭，支持 `--no-progress`，完成清条、中断保留实际进度。`pyproject.toml` 增加 `tqdm>=4.66`。
- **按压检测**：`events.py`／`recorder.py` 新增可选 `motion_detector: press`，配置位于 `config/placement/placement_press_buttons.json`，通用模式仍为停留检测。连续反馈段内按 0.2 s 平滑高度，要求下压 ≥20 mm、抬起 ≥15 mm、横向偏移 ≤25 mm，两阶段各持续 0.08–1.5 s 且保持闭爪；以最低点定位、抬起后确认，保存该时刻原始 TCP／机器人快照，缺口及浅波动不拼接为重复候选。规则不硬编码手侧、时刻或候选数量。
- **按压数据与前端**：按压包升级为 schema 3，JSON／Parquet 增加 `press_depth_m`、`press_retract_m`、`press_lateral_m`；网页支持按压筛选、位移证据与事件姿态，默认显示运动候选。新包独立输出至 `public/data/press_buttons_press/`，保留原包。

### 验证结果与限制

- **反馈与入口**：当日边界裁剪／失败重试聚焦测试记录为 **87 passed、2 skipped**；统一入口此前完成 **82 项** CPU 测试，并以合成反馈串联独立提取解释器和完整 USD 导出，得到 33 个 S1 body、约 42.5 MB GLB。不同批次测试不累加；这些检查不代表重新执行了 SDK／GPU 轨迹。
- **按压参考与批量结果**：`press_buttons/episode_000003` 有 1,517 个同步采样、无缺口，原速度／加速度条件在三个标注窗口均为 0/77 命中。新检测得到左手 **5.650、8.794、12.351 s** 三个候选，右手为零；分别调整平滑、下压、抬起和横向阈值 ±20% 仍保留三个。基于已有录制完成 319 个 episode 的 CPU 重提取：共 1,587 个事件、949 个按压候选，其中 311 个 episode 各三个、8 个各两个；JSON／Parquet 事件 ID 与位移字段一致。
- **检测与网页验证**：带 CPU USD 依赖的检测／同步／模型导出测试 **53 passed**，前端单元测试 **11 passed**，新增按压浏览器用例 **1 passed**；4173 页面加载参考包显示三个按压点且无浏览器异常。以上为原记录证据，本次仅整理文档并检查结构、路径与空白，未重跑测试或仿真。
- **证据边界**：最初同步失败未留下完整录制，其异步收尾原因来自运行路径分析，后续实录补充了边界滞后证据。按压候选仅描述运动形态，未测接触力或按钮触发；整批数量不代表人工标注准确率，浅按压可能低于阈值，未证明硬实时 100 Hz 或真机等价性。

操作命令见 [DATASET_REPLAY.md](DATASET_REPLAY.md)，故障分析见 [REPLAY_FEEDBACK_SYNC_ANALYSIS_2026-10-23.md](REPLAY_FEEDBACK_SYNC_ANALYSIS_2026-10-23.md)。

## 2026-10-22：原始 Parquet 整目录回放与 Web 数据衔接

- **批量回放**：`replay_parquet.py` 直接读取原始 25 DOF 数据目录，经 SDK → ROS → Isaac Lab 逐 episode 回放，支持无查看器运行、反馈保存、断点续跑和校验报告。默认沿用 SDK 103 逐帧运动，保留原始时间间隔；反馈统计排除预定位，恢复时核对输入、设置和捕获文件。
- **Web 数据**：`extract_placement_events.py` 移除对逐关节对齐入口的依赖，按完整批次提取候选及配套模型；`export_placement_robot.py` 继续负责默认模型和参考姿态。运行说明见 [DATASET_REPLAY.md](DATASET_REPLAY.md)。
- **验证与限制**：press_buttons 全目录 dry-run 覆盖 **319 episodes、220,191 帧、约 2.04 小时**；缺少关节元数据，明确采用现有 Astribot 顺序。CPU 测试 **70 passed**，网页 `npm test`、语法及空白检查通过。实际 SDK／ROS／Isaac 重放尚未执行，由用户运行。


## 2026-10-22：Simulation 网页内网部署与开机自启

网页部署至 `http://10.11.12.73:4173/`，由用户级 systemd 服务 `astribot-placement-viewer` 后台运行。已启用开机自启和用户 linger，无需登录即可启动，异常退出后每 5 秒重试。

配置校验及服务重启验证通过，本机经内网地址访问页面和机器人模型均返回 HTTP 200；未进行整机重启验证。启动与管理见 [操作说明](PLACEMENT_ANALYSIS.md#内网访问)。

## 2026-10-22：示例停留点与 episode 热力场联动修正

- **原因**：示例数据将所有 episode 的独立按压停留点固定在 `[0.63, ±0.25, 0.79]`，导致“低速且低加速度”模式从 1–24 缩小到 23–24、仅 24 时，剩余点仍集中于同一位置。筛选链路原已按当前计入统计的点重算三维密度并重建纹理。
- **修正**：沿用固定随机种子，分别生成各 episode、各手的按压停留坐标；候选数量、关联去重和初始排除规则保留。功能说明同步明确当前点集归一化、视线最大密度与空结果清除的含义，真实数据的热点位置仍取决于剩余点位。
- **回归保护**：新增三个 episode 区间的密度峰位置检查；浏览器通过真实滑条输入检查 72 → 6 → 2 个样本及实际纹理和渲染颜色，确认热点移动、原热点降温、恢复区间结果一致和空结果清除。新增浏览器用例使用简化示意机器人，不依赖可选 USD 模型导出。
- **验证与边界**：Node 20.20.1 下 `npm test` 与 `npm run build` 成功；分别直接运行 `tests/data.test.js`、`tests/field.test.js` 共 **10 项通过**。Chrome SwiftShader 软件渲染下，区间热力场、局部密度颜色及文件导入导出共 **3 项浏览器用例通过**；完整 USD 模型的桌面／手机综合用例两次达到 120 秒上限，分别停在末尾的桌面截图和显示模式切换，该项未完成。验证服务使用独立本机 5187 端口，临时 Playwright 配置仅覆盖 `baseURL` 和测试目录；未更改原有测试超时与断言。本轮仅执行前端 CPU 验证。

## 2026-10-22：Torso 调参文档归档

为躯干三个关节加入独立 PD 与 PhysX 重力前馈：第一关节 Kp/Kd=3200/320、重力系数 1.1，第二、第三关节 3200/160、系数 1.0，驱动力矩上限均为 400 N m。
双夹爪采用 Kp/Kd=800/8、PhysX 速度上限 16 rad/s，并关闭额外 100 ms 速度预测，改善开合滞后与过冲。
此前三轨迹物理对照及 SDK → ROS → Isaac Lab 回放显示位置误差明显下降，专项回归 41 项通过；启动瞬态和夹爪速度偏差仍存在，未验证硬实时 100 Hz 或力矩等价性。
调参配置与原始证据保留在 `config/torso*_tuning*.json`、`results/torso*/`，两份独立调参文档删除；本轮仅核对文档引用与空白。

## 2026-10-22：Placement 改为 SDK → ROS → Isaac 反馈链路

- **项目规范**：在实际根协作文件 `agents.md` 的“项目边界”加入一句话：数据集轨迹统一经 SDK → ROS → Isaac Lab → ROS 状态反馈回放，placement 基于该链路反馈，不另建绕过 SDK／ROS 的回放路径。未创建重复的 `AGENT.md`。
- **实现变化**：复用现有 `replay_artifacts_sdk_100hz.py` 回放并保存反馈；`extract_placement_events.py` 不再初始化 Isaac 环境、直接设置关节或推进物理步，改为接受完成录制的 episode 目录，读取 `sim_states.parquet`、`api_commands.parquet` 和 `run_metadata.json`。提取仅在 CPU 上执行；调整检测阈值可复用录制，无需重跑轨迹。`--dry-run` 在 USD／ROS／SDK 导入和输出写入前返回。
- **坐标和同步契约**：`ros_feedback.py` 按 ROS header 时间将各关节插值到躯干第一关节采样时间，仅保留首个 API 调用至最后调用结束后 10 ms 的窗口；缺少括点、跨度大于检测器 `max_gap_s` 的样本不插值，大间隔保留为缺口。拒绝非 simulation、缺失关节、无效值、重复／倒退时间戳、首尾覆盖不足和不完整 API 捕获。质量统计及捕获文件哈希进入 metadata。`reference_pose.py` 编译 USD 关节树并重复计算反馈 FK；`recorder.py` 接受同步位姿样本，沿用两类候选、初始阶段标记及网页 schema v2。
- **解释边界**：当前 ROS header 使用墙钟，速度／加速度亦基于墙钟，不能声称使用 PhysX 时间。采用固定底座的 USD 世界位姿；TCP 从关节反馈和 USD link 坐标计算，其他指节由实测 L1 百分比推算。录制时模型必须与 `--usd` 一致，旧回放 metadata 无模型哈希，不能自动验证匹配。用户说明同步更新至 `PLACEMENT_ANALYSIS.md`，历史直接回放记录保留为历史证据。
- **CPU 验证**：系统 Python 3.10，关闭 pytest 插件自动加载；通过 Isaac 自带的 `pxr` 库在 CPU 加载 USD，未导入 SimulationApp。placement 检测、ROS 反馈同步／异常拒绝／无运行时 dry-run／CPU 转换、USD 导出／FK、SDK runner 离线测试共 **30 passed**。实际已有两条录制 dry-run 成功，分别同步得到 1,535／2,228 帧，均无大缺口；另一次尝试的第三条 pour 目录缺 `run_metadata.json`，明确失败后未纳入成功证据。
- **实际离线转换**：仅读取已有 `replay_train_20261020_213902` SDK 捕获，在 `/tmp/placement-sdk-cpu-review/` 完整生成 GLB、Parquet 和 JSON 候选包（1,535 帧）；这不是重新执行轨迹。完整转换用例另以合成 ROS 捕获验证事件及 schema，模型网格导出有独立测试。Python 语法、变更空白检查通过。网页代码未变，未重跑浏览器；本轮未启动 ROS 节点、SDK 控制连接、Isaac 仿真、GPU 或真实硬件。

CPU 聚焦测试（USD 库环境设置同功能说明）：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -o addopts='' \
  tests/unit/test_placement_analysis.py \
  tests/unit/test_placement_ros_feedback.py \
  tests/unit/test_placement_usd_export.py \
  tests/unit/test_replay_artifacts_sdk_100hz.py -q
```

## 2026-10-22：SDK 回放逐关节对齐分析（含夹爪）

- **数据与比较方法**：使用选中的躯干第一关节配置，分析三条已保存的 SDK 回放记录，每条 22 个关节，共 66 组关节／轨迹对比。按仿真消息头时间戳插值真机反馈，保留启动瞬态，不拟合时间偏移。角关节位置误差采用度，夹爪采用 SDK 0–100 标度的百分比点；速度误差采用 rad/s 或百分比点/s。力矩等价性尚未验证，不纳入比较。
- **躯干持续偏差**：按 20261020_213902 / guess / pour 的顺序，躯干第二关节位置 RMSE 为 5.008 / 4.925 / 4.027 度，仿真减真机的平均偏差为 −4.910 / −4.837 / −3.883 度；第三关节位置 RMSE 为 1.650 / 1.114 / 1.728 度。这些偏差持续存在，并非只有孤立峰值；第一关节调参不代表全身已对齐。
- **夹爪差异**：左夹爪位置 RMSE 为 16.084 / 8.284 / 4.322 个百分比点，瞬时最大误差为 75.664 点；右夹爪为 0.131 / 1.502 / 6.947 点，瞬时最大误差为 51.155 点。曲线显示开合过渡阶段差异明显，其中 20261020_213902 的仿真左夹爪响应慢于真机。这些问题在第一关节调参前已存在，本轮未确定唯一根因，也未修改夹爪控制器。
- **其他偏差与后续顺序**：头部第二关节、左臂第一／第四关节和右臂第四关节也有剩余位置偏差。建议后续依次检查躯干第二关节、第三关节和夹爪瞬态响应；本轮分析未继续调参。
- **结果产物**：完整逐关节表、各轨迹的位置／速度误差、带符号偏差、p95、峰值时间、样本数及与基线的差异，见[分析报告](../results/torso_joint_1_tuning/joint_alignment/report.md)、[CSV](../results/torso_joint_1_tuning/joint_alignment/per_joint.csv) 和 [JSON](../results/torso_joint_1_tuning/joint_alignment/per_joint.json)。[对比曲线](../results/torso_joint_1_tuning/joint_alignment/largest_mismatches.png) 覆盖三条轨迹的躯干第二关节及双夹爪。
- **验证与边界**：重新计算全部 66 组位置 RMSE，并核对保存的 SDK 误差文件中的样本数；人工检查生成曲线。此前调参的最终专项回归共 30 项通过，覆盖夹爪、躯干控制、SDK 回放和仿真节拍。本轮仅分析已有记录，未新增仿真、SDK 控制、真实硬件操作或运行代码修改。

## 2026-10-22：躯干 2、躯干 3 与夹爪 SDK 对齐修正

- **躯干控制**：为躯干第二、第三关节增加独立隐式执行器和逐关节 PhysX 重力前馈。默认均为 Kp=3200、Kd=160、力矩上限 400 N m、重力系数 1.0；每步只向对应 DOF 写入补偿，避免修改其他躯干或整机关节。
- **夹爪控制**：为左右夹爪增加独立六关节执行器组，默认 Kp=800、Kd=8、力矩上限 400 N m、PhysX 速度上限 16 rad/s。Isaac Lab 2.1 的隐式执行器会忽略旧 `velocity_limit` 字段，本轮改为 `velocity_limit_sim`，并保留 SDK 0–100 标度和原有六关节符号映射。
- **SDK 过冲修正**：SDK direct position 帧已附带由位置差分得到的速度；仿真之前又对夹爪启用了 100 ms 速度预测。关闭左右夹爪的 `vel_compensation` 后，避免快速开合时重复前推。配置顺序为 `[True, True, True, False, True, False]`，对应躯干、头、左臂、左夹爪、右臂、右夹爪。
- **物理结果**：三条完整轨迹上，躯干第二关节位置 RMSE 从 4.743/4.758/4.005° 降至 0.445/0.350/0.292°，第三关节从 1.533/0.959/1.726° 降至 0.514/0.387/0.511°。左夹爪从 15.896/8.339/4.500 降至 3.774/1.099/1.486 个百分比点；右夹爪从 0.124/2.824/7.628 降至 0.110/0.649/1.617。
- **SDK 结果**：基线和最终候选均由新脚本管理仿真器，在 ROS 域 198、`ROS_LOCALHOST_ONLY=1` 中完成三轨迹回放；SDK、仿真器退出码均为 0，每条轨迹各部件收到全部 command 帧（最终 pour 的一次 angular payload 审计误差为 0.000116 rad）。最终 SDK 位置 RMSE（213902/guess/pour）为：躯干 2 0.909/0.694/0.308°，躯干 3 0.698/0.562/0.555°，左夹爪 3.286/1.249/1.534，右夹爪 0.113/0.792/1.696 个百分比点。
- **结果产物**：参数筛选、最终物理基线／候选、SDK 基线／首次候选／最终候选和逐关节报告见 `results/torso_gripper_{screen,refinement,final_physics,sdk_baseline,sdk_selected,sdk_final}/` 及 `results/torso_gripper_tuning/`，试验参数见 `config/torso_gripper_tuning_{trials,refinement,selected}.json`。
- **验证**：专项回归 41 passed，覆盖逐组控制参数、三躯干补偿、六关节夹爪映射、关闭夹爪预测、SDK 回放与仿真节拍；Python、Shell 语法和空白检查通过。全仓 unit 测试未完成收集：未加载 ROS overlay 的 Isaac Python 缺少 `cv_bridge`，系统 Python 存在 pytest／anyio 插件版本冲突。
- **边界**：最终 SDK 状态观测约 91.28–95.85 Hz，未证明硬实时 100 Hz；预定位残差和启动瞬态保留在误差中。躯干 1 在 213902/guess 的位置 RMSE 从 0.169/0.288° 增至 0.318/0.389°；213902 右夹爪静止段速度 RMSE 从 3.378 增至 11.798 个百分点/s，左夹爪瞬时最大位置误差仍有 28.216 个百分点。力矩只作诊断，未宣称与真机等价；未操作真实硬件。

## 2026-10-22：躯干第一关节的独立 PD 与重力补偿调参

- **范围与配置**：新增可选 Isaac 配置 `torso_joint_1_control`，为该自由度配置独立隐式执行器，仅改变其增益与 PhysX 重力前馈。默认 YAML 采用 Kp=3200、Kd=320、力矩上限 400 N m、重力补偿系数 1.1；其他关节增益保持 800/40。试验参数见 `config/torso_tuning_{trials,refinement}.json`，曲线与原始记录见下述结果目录。
- **物理回放筛选**：在 pour/guess 上测试 8 组初选参数和 4 组局部微调参数，再用未参与筛选的 20261020_213902 轨迹验证基线及两组候选，共完成 27 次物理回放。每帧指令推进一次 10 ms 物理步，每轮恢复记录中的初始位置／速度。该关节位置 RMSE 从 1.050/1.768/3.529 度降至 0.174/0.157/0.185 度。
- **SDK 链路验证**：使用已有 SDK 100 Hz 回放脚本，在仅限本机通信的 ROS 域 198 中，对三条轨迹分别重跑基线和选中参数。位置 RMSE 从 1.353/2.189/3.552 度降至 0.180/0.292/0.183 度；选中参数的状态观测频率为 96.31–97.55 Hz。两个本任务仿真器均以零退出码关闭，未控制真实硬件。
- **结果依据**：保存于 `results/torso_joint_1_tuning/`、其中引用的原始实验目录及 `results/torso_tuning_sdk_{baseline,selected}/`。未完成的调试目录不计入结果；隐式执行器力矩仅作诊断。
- **限制**：动力学耦合使其他 19 个角关节的综合位置 RMSE 略增，其中 pour 的躯干第二关节 RMSE 增加 0.01078 rad。SDK guess 在非零预定位残差下仍有早期 1.588 度峰值；未参与筛选轨迹的第一关节速度 RMSE 仍为 0.05244 rad/s。
- **验证**：控制专项测试共 22 项通过，覆盖既有夹爪和新增躯干用例；Python 语法、Shell 语法及空白检查通过。本结果不代表整机已标定或 SDK 反馈满足硬实时要求。

## 2026-10-22：Simulation 操作候选点提取与 Three.js 查看器

- **范围与入口**：新增 `src/placement_analysis/`、`scripts/extract_placement_events.py` 和 `web/placement-viewer/`。Isaac Lab 离线提取操作候选；网页只读本地文件或同源静态数据，显示真实机器人、米制点位和三维体素密度，不提供 SDK、ROS 或仿真控制入口。提取脚本依赖 `scripts/replay_artifacts_data.py`，当前输入为 22 关节、完整 100 Hz 公共时间网格的 artifacts；不支持直接传入 25 DOF LeRobot 宽表。
- **两条检测路径**：左右手独立运行夹爪开合和 TCP 持续停留检测。开合采用滞回和持续确认，输出 `grasp / release`；停留要求速度与加速度同时低于阈值并持续至少 0.2 s，输出 `dwell`，每段停留只记录一个代表点。再次产生停留候选前需持续运动至少 0.08 s、位移至少 1 cm，防止噪声重复计数。导数使用实际物理时间，采样缺口切断检测窗口。
- **复杂操作关联与展示**：开合区间与同手停留区间按 0.15 s 默认容差关联，保留检测来源和 `related_event_ids`。网页提供“夹爪开合”“低速且低加速度”“任一满足”“同时满足”四种候选依据；任一满足对关联点去重，同时满足显示停留附近发生的开合点，不将另一只手的静止或单帧低速度视为证据。
- **初始位姿行为排除**：按 episode、按左右手，以首帧 TCP 的底座坐标为初始位置；持续离开 5 cm 半径至少 0.15 s 后解除初始阶段标记，后续返回原位不重新排除。默认同时忽略 episode 开头 1 s；网页允许取消初始阶段排除、调整开头忽略秒数，并显示灰色已排除候选供检查。原始候选全部保存，灰色检查点不计入事件统计、episode 数和密度热力图。
- **数据契约**：新增 `config/placement_analysis.json`。`events.parquet`、`placement.json`、`metadata.json` 保存两类检测候选、时间区间、世界／底座 TCP 位姿、运动量、初始阶段与排除依据，以及事件帧完整 body 位姿和关节状态；逐帧轨迹只在内存中用于检测。JSON 升级为 `schema_version: 2`，保留 v1 导入能力，旧文件缺少停留与初始阶段数据时禁用对应选项并提示重新提取。筛选导出保留 `metadata.view_filters`。
- **机器人模型、坐标与默认姿态**：`usd_export.py` 和 `scripts/export_placement_robot.py` 在 CPU 上将 Isaac 使用的同一份 USD 可视网格导出为 GLB，保留 body 局部几何及 Z-up、米制坐标；当前模型导出 33 个 body、1,158,401 个三角面，约 42.5 MB。`reference_pose.py` 使用 USD 关节局部坐标在 CPU 上计算参考姿态，`config/placement_reference_pose.json` 的头、躯干、双臂和 50% 夹爪目标逐项对应 SDK ROS2 `examples/103-move_to_joint_position.py`，没有执行该 SDK 示例。真实事件姿态仍从文件快照恢复。
- **网页最终交互**：品牌及页面标题从 Placement Atlas 改为 **Simulation**；支持左右手、episode、高度、坐标系、体素大小、模型透明度筛选和点击事件查看姿态。紫色方点以“停留点”复选框与“抓取点”“释放点”并排，控制显示及对应计数。“代表帧也需低速且低加速度”复选框及 `lowMotion` 网页筛选逻辑已移除；持续停留检测和数据中的运动量仍保留。排除区域标题统一为“初始位姿行为排除”。
- **验证证据与边界**：本功能的 CPU Python 检测／格式／坐标／USD 导出及参考姿态测试合计 **14 passed**；网页数据测试与生产构建通过；候选模式、去重、初始排除、导入导出和桌面／手机布局的浏览器用例 **2 passed**，使用禁用硬件 GPU 的 Chrome SwiftShader 软件渲染。后续停留点复选框、代表帧筛选移除和文案调整通过前端数据检查与构建；未把此前浏览器结果描述为这些后续调整的重新验证。输入 `--dry-run` 已验证 1,584 帧，不启动仿真。本功能未运行 Isaac/GPU 轨迹提取，页面分布为明确标注的合成示例；实际端到端提取仍待验证。TCP 的名义 15 cm 偏移尚未标定，候选点不等于物体中心、按压接触或最终落点。
- **文档收敛**：使用说明、检测语义、配置及输出契约保留在 [PLACEMENT_ANALYSIS.md](PLACEMENT_ANALYSIS.md)，同步最终筛选项名称；删除 `web/placement-viewer/README.md` 中与该说明重复的启动／测试文档，并移除功能说明中重复的 CPU 测试章节。测试代码继续保留，复现命令统一记录如下。本轮只修改文档，校对引用与格式，不重跑测试或轨迹。

CPU 测试复现（USD 用例需能导入 `pxr`；现有 Isaac USD 库的 CPU 加载方式见功能说明）：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -o addopts='' \
  tests/unit/test_placement_analysis.py \
  tests/unit/test_placement_usd_export.py -q

cd web/placement-viewer
npm test
npm run build
# 另一个终端先启动 npm run dev；浏览器使用 CPU 软件渲染。
npx playwright test
```

浏览器默认使用 `/usr/bin/google-chrome`，可通过 `PLACEMENT_CHROME` 指定其他可执行文件。没有导出的机器人模型时，先按功能说明在 CPU 上导出，才能运行检查真实 USD 模型的浏览器用例。

## 2026-10-22：无窗口 GPU 基准与 100 Hz 完整轨迹对齐

- **性能边界**：RTX 3090、4 ms 物理步长的 headless 长测中，完整 `env.step`（含 ROS executor、无相机）约 118.6 Hz，三相机约 29.9 Hz；physics-only 约 283.0 Hz，不能据此宣称完整回放稳定达到 250 Hz。
- **100 Hz 数据与唯一回放入口**：`resample_replay_commands.py` 将三条原始 command 线性采样到共同 10 ms 网格，输出 1,584 / 2,314 / 3,597 帧、22 关节；原始 state 行和采集元数据不变，交集裁边与 synthetic command 来源见 `artifacts/resampled_100hz/manifest.json`。当前唯一回放脚本是 `scripts/replay_artifacts_sdk_100hz.py`：每帧调用 `Astribot.set_joints_position(control_way="direct")`，经 ROS2 command topic 到 Isaac，再从 ROS2 state topic 录反馈。SDK command、仿真 state、API 调用时间、header/接收时间和逐关节误差均保存在 `results/alignment_sdk_100hz/`。
- **旧入口清理**：移除过时的 250 Hz `replay_artifacts_to_sim.py`、同进程无 SDK 的 `replay_artifacts_100hz.py`、其专属单元测试和 `README_replay_artifacts.md`。公共 100 Hz 数据校验、关节映射和仿真初态设置迁至 `scripts/replay_artifacts_data.py`，`extract_placement_events.py` 改为依赖该共享模块，不再依赖无 SDK replay runner。
- **SDK 链路结果**：三条轨迹均完整执行；每个部件收到全部 SDK command 帧，command payload RMSE 为 0 或极小。仿真 state topic 观测约 96.01–96.94 Hz。按 simulator header time 配对时，20 个角关节 position RMSE 为 0.022861–0.024468 rad，velocity RMSE 为 0.060635–0.067402 rad/s；2 个夹爪 position RMSE 为 5.816284–11.320950 percentage points。SDK receive-time 指标相近。角关节目标 0.01 rad / 0.05 rad/s 尚未达到；当前 actuator 仍为统一 800/40/400，torque 仅诊断。完整汇总见 `results/alignment_sdk_100hz/consolidated_summary.md`。
- **100 Hz 墙钟 pacing**：`ASTRIBOT_ISAAC_CONTROL_HZ=100` 为可选环境循环节拍，默认不启用；以绝对 monotonic deadline 调度，step 耗时不再额外叠加一个 period，超期时跳过过期 deadline。100 Hz 指 physics step 语义；实际 SDK 回放 state 观测约 96 Hz，因此未证明硬实时。
- **第三人称视频尝试**：本地保留 `results/video_100hz/replay_train_pour_third_person.mp4`，1280×720、30 fps、470 帧、15.67 s。相机在 SDK 预定位稳定后才启动，动作期间持续运行、没有逐帧暂停；但该视频只覆盖约 15.64 s 仿真时间，而 `pour` command span 为 35.96 s。该次运行同时只观察到约 43.53 Hz state 频率，说明相机负载下未实现完整 100 Hz 回放；视频是部分片段，不应作为完整轨迹交付。本轮不重录。
- **验证**：SDK replay 离线测试 5 项通过；`python -m py_compile` 覆盖 SDK runner、Isaac env 和 simulation factory。删除旧测试后未再运行被删的无 SDK 测试。placement 的 Isaac/GPU 提取流程未重跑。
- **清理后验证**：SDK runner、simulation timing、placement analysis 和 USD export 离线测试共 16 passed、1 skipped；`extract_placement_events.py --dry-run` 验证 100 Hz 输入 1,584 帧成功；未启动 Isaac/GPU 或 SDK 回放。
- **SDK 回放运行契约**：仿真终端使用 `astribot_simu_il` + ROS2 Humble，SDK 终端 source `astribot_sdk_ros2/env.zsh`；两端必须使用同一个隔离 `ROS_DOMAIN_ID`，测试使用 `ROS_LOCALHOST_ONLY=1`。仿真配置 `ASTRIBOT_FRAME_SKIP=1` 和可选 `ASTRIBOT_ISAAC_CONTROL_HZ=100`；SDK runner 输出 API command、ROS command capture、ROS state capture、payload audit、tracking error 和 run metadata。
- **旧结果的证据边界**：`results/alignment_100hz/` 是已保留的历史同进程物理基线，metadata 中引用的旧 runner 文件已清理；当前可复用回放只认 `scripts/replay_artifacts_sdk_100hz.py`。`results/alignment_sdk_100hz/` 是 SDK-to-Isaac 三轨迹结果，`artifacts/resampled_100hz/manifest.json` 是输入来源与裁边依据。

## 2026-10-20：标定入口可用性状态

- **保留实现**：保留独立 `scripts/calibrate_joints.py`、显式零增益执行器、PD／滤波／时延回放及确定性检查实现。
- **实际状态**：当前 Isaac Kit 串行 GPU 检查可初始化 USD、32 个关节系统自由度和 CUDA PhysX，但在结果 Parquet 写出前自行退出，尚无可复验的端到端输出；因此仍需定位该入口，才能用于 `fit` 闭环。
- **验证边界**：Atlas 侧 21 项共享控制核心无硬件测试通过，不能替代 Isaac 物理验证。未启动 ROS1 或真实硬件。

## 2026-10-20：离线关节响应标定

- **范围与入口**：新增 `scripts/calibrate_joints.py` 和 `docs/JOINT_CALIBRATION.md`。入口直接读取 Atlas 异步记录契约和逐关节 YAML，不依赖 ROS、SDK、相机或已有环境工厂。
- **控制与数据语义**：采用固定底座、显式零增益执行器，由 Atlas 的 PD／滤波／时延控制驱动，使用固定力矩限制和物理时间指令回放；反馈只用于初始状态。双夹爪使用已有的带符号六关节映射，SDK 夹爪力矩仅用于诊断。
- **检查能力**：运行时检查隐式驱动已关闭、执行器力矩已提交；`--verify` 检查重复性、渲染无关性和半物理步长收敛。帮助入口和 Python 3.8 语法检查通过。Atlas 测试使用测试对象验证共享回放／控制器，不代表 Isaac 物理验证。
- **GPU 尝试与未完成项**：Isaac Sim 可见本机 RTX 3090，驱动版本 580.178.04。串行无窗口运行已到达 USD 加载、关节系统创建、零增益执行器设置和 CUDA PhysX 初始化，但 Isaac Kit 在结果文件写出前退出，未输出 Python 异常或结果产物。重置路径已缩减为一次关节状态写入，半步长检查改用独立 Kit 进程，但端到端 `--verify` 仍未验证。
- **范围边界**：未运行 ROS1 真机采集；保留用户已有的 ROS 回放、环境、配置、资源和开发日志修改。

## 2026-10-16：采用 SDK 103 接口的 Parquet 回放模式

- **范围与接口**：为 `scripts/replay_parquet.py` 新增 `--control-mode move`。初始位姿和每个数据帧均调用 SDK 示例 103 使用的 `move_joints_position`，以相邻帧时间戳间隔作为阻塞运动时长；原有指令流通过 `--control-mode direct` 保留。
- **对比结果**：在隔离 Isaac Lab 环境中回放 326 帧样本，move 模式的数据集阶段耗时 11.770 s，原始记录为 10.833 s。SDK 生成的手臂指令位置单步变化最大为 0.00904 rad，direct 模式为 0.05069 rad；左夹爪观测反馈的最大单步变化分别为 13.46 和 47.24 SDK 单位。move 模式最终目标误差低于 6e-6。
- **运行行为与限制**：三路相机持续输出；move 与 direct 两次运行及各自的本任务仿真器均正常退出。move 模式不与仿真时间同步，不保证帧边界速度连续，也不消除渲染负载或数据本身的不连续。
- **验证**：`tests/unit/test_parquet_replay.py -o addopts='' -q` 共 20 项通过，语法和空白检查通过；在隔离 ROS2/Isaac Lab 环境中完成样本的 move/direct 全轨迹回放。未使用真实硬件。

## 2026-10-15：USD 外观与 Isaac Lab 相机标定

- **USD 外观**：新增 `src/sim_assets_tools/adapt_s1_usd_appearance.py`，从 MuJoCo 资源生成材质、贴图及轮子视觉层；`astribot_descriptions/usd/astribot_s1_usd/` 新增 `astribot_s1_appearance.usdc`、`astribot_s1_textured.usda`，覆盖 42 个视觉网格（4 个贴图），保留原物理结构。
- **配置与输出**：`config/astribot_s1/simulation_isaaclab_param.yaml` 默认加载 textured USD，头部设为 1280×720、双腕 640×360；`src/astribot_envs/astribot_isaaclab_env.py` 支持逐路分辨率，移除输出端翻转，直接发布传感器原图。
- **相机外参**：修改 `astribot_s1.usda` 三个 Camera prim；头部采用 OpenCV 位姿并转换为 USD/OpenGL，消除旧光心被头壳遮挡的黑屏；双腕绕各自光轴旋转 180°，位置和视线不变。垂直 FOV 为头部 51.4175°、双腕 57°，方向修正在外参中完成。
- **核查与文档**：新增 `scripts/verify_s1_usd_appearance.py`、`scripts/render_s1_usd_appearance.py`、`scripts/diagnose_isaaclab_head_camera.py`，扩展 `scripts/verify_isaaclab_rgb.py`；三路实际渲染、内外参及 ROS/SDK 检查通过，API 与传感器原始像素一致。

## 2026-10-15：重构仿真仓库协作规范

- **范围与依据**：基于分区阅读的源码、配置、测试、安装脚本、ROS 消息和模型引用结构，重写根 agents.md 与 AGENT.md；保留中文正文及通用协作、安全、Git 和证据规则，用实际仿真架构替换导入的外仓机制与路径。
- **结构与内容**：agents.md 保留七节结构及实现偏好，明确文档职责和中文规范例外；AGENT.md 保留主标题及七节结构，其余二级标题原样保留，按用户确认使用“垃圾回收”中文标题，并将该节合并为一条要求。配置、后端能力、ROS、相机、资产和测试描述区分现有行为与工程要求。
- **接口与运行**：N/A；仅调整协作规范，未修改运行代码、配置、依赖、模型或消息子模块，也未修复审阅中发现的其它问题。
- **验证**：使用 Python 标准库核对标题结构、本地链接、外仓机制残留、文件格式、日志历史字节及变更前文件哈希；35 个链接有效，80 个受保护文件内容未变，两个子模块状态未变。git diff --check 与两份未跟踪规范相对修改前快照的 git diff --no-index --check 未报告空白错误；本次未启动安装器、仿真、ROS、GPU 渲染或真实硬件，不据此推断运行能力。

## 2026-10-15：回放运行依赖与 SDK 控制节拍

- **依赖定位与修复**：在加载 `astribot_sdk_ros2/env.zsh` 的系统 Python SDK 环境中复现缺少 PyArrow；在仿真器 `astribot_simu_il` Python 环境中复现缺少 `cryptography`，原生 SDK 将后者掩盖为 `robotics_library_py ... is not a package` 导入错误。为本地系统 Python 安装 PyArrow 17，新增 `scripts/requirements-replay.txt` 以复现依赖安装。
- **启动检查**：等待仿真前先检查 `cryptography.fernet` 并导入 SDK；缺少依赖时明确指出解释器和安装命令。SDK 客户端仍在仿真及相机就绪检查后构造。
- **控制节拍**：初始位姿渐变和数据集目标均以 100 Hz 发送，与 SDK 频率一致，避免将 100/30 Hz 调用与 250 Hz SDK 混用。在相邻数据时间戳之间保持当前目标，跳过已延误的控制节拍。
- **文档**：恢复缺失的回放设置说明并从 README 链接，分别说明仿真器和 SDK 的 Python 环境，以及加载 SDK `env.zsh` 后需要设置的 ROS 环境变量。
- **离线验证**：18 项 pytest 测试、语法及空白检查通过。
- **实际回放验证**：使用真实 326 帧、10.833 秒轨迹，在 ROS2 Humble、`astribot_simu_il` 中的 Isaac Lab，以及加载 SDK `env.zsh` 的 `/usr/bin/python` 环境下回放。域 187 通过仅限本机发现和回环 UDP 传输隔离。六个可动部件各收到 1,386 条指令，最终指令与数据集的误差为零；每个部件产生 211–212 个包含运动的反馈样本，最终躯干／双臂／头部反馈误差不超过 0.076 rad，夹爪不超过 0.125 SDK 单位。观察器收到头部／左腕／右腕 RGB 图像分别为 64/94/106 帧。回放、查看器和本任务仿真器均退出，回放与仿真器退出码为零；本次未出现调用频率不匹配警告。
- **限制**：SDK 内部期望状态同步订阅仍记录 QoS 可靠性警告，但未阻止仿真器或观察器接收指令；本次没有修改外部 SDK。仅验证该 Isaac Lab/ROS2 环境，未验证真机或其他后端。

## 2026-10-15：修复回放探针的 ROS2 上下文

- **根因与修复**：就绪探针改用绑定自身 `Context` 的 `SingleThreadedExecutor`，避免在默认上下文尚未初始化时调用全局执行器。正常和异常路径均按执行器、节点、上下文的顺序关闭资源。
- **反馈兼容**：接受仿真器合成的固定底盘反馈中已有的空 `frame_id`；可动部件反馈仍须包含仿真坐标系标记。
- **离线验证**：回放测试共 15 项通过，覆盖就绪、超时、拒绝非仿真反馈及资源清理。
- **ROS2 验证与边界**：在 ROS2 Humble、域 197、本机回环通信中，收到七路合成关节状态话题及三路相机话题；两次无发布者探针均超时并正常清理，默认 ROS 上下文始终未初始化。未构造 SDK 客户端，未运行运动、GUI 或 GPU。

## 2026-10-15：Parquet 片段的 SDK 回放工具

- **范围与入口**：新增 `scripts/replay_parquet.py`，通过 ROS2 SDK 临时回放 Parquet 中的 Astribot 25 自由度关节位置片段，支持帧范围、时间戳／FPS、速度和初始位姿设置。固定底盘不参与回放，夹爪沿用 SDK 的 0–100 标度。
- **数据与启动检查**：校验指令和时间戳的有限性、可选 LeRobot 元数据及轨迹边界；确认反馈来自仿真且三路相机都有数据后，启动已有的头部、左腕和右腕查看器。
- **文档与测试**：新增 `docs/PARQUET_REPLAY.md`，说明仅限本机通信的环境设置和清理行为，并在 `tests/unit/test_parquet_replay.py` 中补充离线测试。
- **验证**：`python -m py_compile scripts/replay_parquet.py` 通过；使用真实 326 帧 Parquet 和 60 帧半速片段完成 dry-run；隔离离线测试 `tests/unit/test_parquet_replay.py -o addopts='' -q` 共 12 项通过。未运行 ROS/GPU 或真机回放。

## 2026-10-14：统一 Isaac Lab GPU 仿真、相机输出与 SDK 接口

完成 SDK 103 运动显示修复、三路 RGB 相机接入、SDK 300 图像接口适配及终端环境脚本修复。以下按当日最终实现合并记录。

### 运行版本与最终配置

- 修正 editable 安装路径至 Isaac Lab 2.1.1，`scripts/lite_install/install_isaaclab.sh` 默认版本同步更新；使用 Isaac Sim 4.5。
- `astribot_simulation.py` 切换为加载 `config/astribot_s1/simulation_isaaclab_param.yaml`。最终配置为 `physics_device: "cuda:0"`、`use_fabric: True`，启用头部及左右腕三路 640×480 RGB 相机；默认 `mode: "human"` 显示场景窗口并发布图像，`mode: "rgb_array"` 用于无窗口采集。

### GPU 运动显示与仿真生命周期

- 问题表现为 SDK 控制后关节状态及 Fabric 刚体姿态变化，但界面停留在初始姿态，定位到渲染同步链路。
- `src/astribot_envs/astribot_isaaclab_env.py` 将物理设备与 Fabric 配置传入 `SimulationCfg`，增加状态日志；启动时显式设置 `/app/asyncRendering=false`，并将普通及持久化的 `rtx/hydra/readTransformsFromFabricInRenderDelegate` 均设为 `false`，使用 Hydra 场景委托读取 PhysX Fabric 变换。
- 启动时仅重定向标准输出/错误的文件描述符，避免关闭 Kit 持有的 Python 流；退出前清理回调和仿真单例，避免关闭时卡住。

### 三路 RGB 输出与 SDK 300 接口

- `astribot_descriptions/usd/astribot_s1_usd/astribot_s1.usda`：为从 MuJoCo 迁移的三个 `Camera` prim 补齐单位 `xformOp:scale` 及 translate/orient/scale 操作顺序，满足 Isaac Lab 变换校验。
- `src/astribot_envs/astribot_isaaclab_env.py`：通过 `CameraCfg(spawn=None)` 绑定模型已有的 `head_rgbd`、`left_wrist_rgbd`、`right_wrist_rgbd`，保留安装位置、朝向和光学参数；启动时启用 `--enable_cameras`，在仿真主线程读取 RGB，并增加首帧 `RGB output started` 日志。
- `src/astribot_envs/astribot_base_env.py`：通过统一 RGB 发布链路输出原始图像并接入 SDK 压缩图像发布。
- `src/simu_utils/robot_ros_interface.py`：新增 `/astribot_camera_control`（`SetBool`）服务及三路 SDK 压缩图像话题；JPEG 编码前执行 RGB→BGR 转换，仅在服务启用且有订阅者时编码，无相机配置时拒绝启用。压缩图像 QoS 为 BEST_EFFORT、depth 1。

| 用途 | ROS2 话题 |
| --- | --- |
| 原始 RGB（`sensor_msgs/Image`，`rgb8`） | `/astribot_whole_body/camera/{head_rgbd,left_wrist_rgbd,right_wrist_rgbd}/image_raw` |
| SDK JPEG（`sensor_msgs/CompressedImage`） | `/astribot_camera/{head_rgbd,left_wrist_rgbd,right_wrist_rgbd}/color_compress/compressed` |

Python 内可通过 `env.get_camera_image("head_rgbd")["rgb_img"]` 直接取得 `(480, 640, 3)` 的 `uint8` RGB 数组。主窗口显示场景视口；独立 Camera Viewer 在另一个终端订阅三路原始图像，需与仿真使用相同 ROS 域（`ROS_DOMAIN_ID=25`）。

### 终端环境脚本

- `env.sh`：修复未加载 ROS 时执行 `exit 1` 导致被 `source` 的终端退出的问题，错误分支改为提示后 `return 1`；使用兼容的 `=` 判断 ROS 版本，根据 Bash/Zsh 加载对应消息工作区 setup，并检查文件及加载结果。
- 修正 Zsh 脚本路径解析和路径引用，显式导出 `PYTHONPATH` 并避免重复追加项目路径。每个新终端先加载 ROS；Bash 使用 `setup.bash`，Zsh 使用 `setup.zsh`。

在仓库根目录的新 Zsh 终端中预览（原终端保持仿真运行）：

```zsh
conda activate astribot_simu_il
source /opt/ros/humble/setup.zsh
source env.sh
python tests/unit/test_camera_viewer.py \
  --cameras head_rgbd left_wrist_rgbd right_wrist_rgbd
```

### 当日验证结果与已知限制

- **SDK 103**：隔离 ROS 域中运行原始 `103-move_to_joint_position.py`，GPU 物理与 Fabric 开启时退出码为 `0`，双臂、躯干和头部运动可见，视口及相机画面随运动更新。
- **RGB 输出**：`scripts/verify_isaaclab_rgb.py` 检查三路图像尺寸、类型、非空画面、ROS 消息编码及负载大小，并保存 PNG，三路 GPU 验证均通过。窗口模式主入口运行 60 步，三路各收到 60 帧；另一次对运行中仿真订阅 8 秒，收到头部 46 帧、左右腕各 45 帧，确认日志静默时仍持续输出。
- **SDK 300**：成功激活三路 RGB，8 秒显示 118 帧头部图像（640×480）；新增 `tests/unit/test_sdk_camera_ros2.py`，启停与 JPEG、无相机拒绝启用、无订阅者跳过编码共 3 项协议测试通过。
- **环境脚本**：Bash 与 Zsh 下，ROS 版本未设置、无效或消息 setup 缺失时均返回失败且保留 shell；加载 ROS2 Humble 后，项目路径、消息包路径及 `ROS_DOMAIN_ID=25` 设置正确。
- **验证范围**：当前仅单环境（`num_envs=1`），多环境批量控制与状态读取尚未实现，大规模 GPU 并行吞吐量未评估；仅输出 RGB，深度和点云字段仍为 `None`；SDK 300 原例退出路径中的未定义 `robot_shutdown()` 未修改或验证。
- **故障归因限制**：Isaac Sim 4.5 GUI experience 原已关闭异步渲染，本项目显式设置用于固定启动配置，不能单独据此解释修复原因。Hydra 开关控制变换读取路径；尚未完成异步渲染与 Hydra 设置的单变量对照，底层原生插件失效原因未确认。

## 2026-10-23：回放文档分层与网页更新流程

- `DATASET_REPLAY.md` 整理为操作手册，区分 validate/dry-run、候选提取、静态构建和浏览器刷新；补充输出目录与 dataset URL 的对应关系、已有录制重提取、新数据回放、输出目录不覆盖和 preview 服务使用方式。
- `REPLAY_FEEDBACK_SYNC_ANALYSIS_2026-10-23.md` 集中保存同步故障的历史证据、当前同步契约、episode 3 按压特征和整批验证结果。将最初未保留反馈的故障归因明确标记为假设，区分边界裁剪与缺失运动重建，历史测试结果不冒充本次重跑。
- 文档显式使用当前存在的 `config/placement/placement_press_buttons.json`，说明脚本中仍指向已缺失文件的检测／参考姿态默认路径；本次未修改配置移动或恢复已删除文件。
- 验证：在线 bundle 为 schema 3、press、complete，319 episodes／949 press candidates；现路径下 episode 3 extract-only dry-run 通过（1,517 samples，0 gaps），CPU extraction preflight 通过。两份文档的本地链接、代码围栏、Bash/Zsh 命令语法、空白检查和 git diff --check 通过。本次未重新导出、构建网页、调用 SDK/ROS 或启动仿真；在线结果来自此前已完成的构建。两份文档原为已有未跟踪文件，未整体纳入提交。

## 2026-10-23：Web 任务选择与提取后自动发布

- **任务选择**：统计上方的来源灰框改为任务下拉框，支持已发布任务、示例和当前导入文件。页面每 5 秒读取运行时任务索引；初次打开跟随最新任务，手动选择后只跟踪该任务的新版本。显式 dataset URL 与本地导入不被轮询切换；切换失败保留原数据与选择。
- **自动发布**：`replay_extract_event.py` 的正常回放／`--extract-only` 在完整提取成功后发布 `placement.json` 及配套 `robot.glb`，同步 `public/data` 与已有 `dist/data`，无需再次构建或重启 preview。`--task-name` 默认使用输出目录名，同名更新；`--no-web-publish` 可关闭发布。dry-run、部分／失败提取不发布，原有输出目录不覆盖规则保持不变。
- **文件一致性**：发布使用进程间锁、不可变内容版本目录及原子索引替换，旧浏览器请求仍能读取旧版本；历史版本不会自动删除。只同步 Web 展示所需 JSON／GLB，完整原始导出仍留在 `--output` 目录。
- **本地部署**：已将现有 `press_buttons`、`press_buttons_press` 完整包登记为任务，并构建本次前端更新。在线 preview 根页面验证默认载入 `press_buttons_press`，范围为 319 episodes，筛选后 949 个有效候选，页面脚本错误为空。未启动仿真、SDK／ROS 或真实硬件。
- **Python／数据测试**：自动发布及 CLI 接线聚焦测试 9 passed；前端 Node 测试 11 passed。回放与发布的完整 Python 回归为 53 passed、6 failed；6 项失败均遇到仓库已有的默认 `config/placement_analysis.json` 缺失，本轮未调整该配置迁移。按压任务须显式传入现有 `config/placement/placement_press_buttons.json`。
- **用户整理数据后的调整**：验证期间 `public/data/demo` 被外部移除，按压数据目录被改名为 `press_cube_olddemo`。用户明确要求任务列表只保留 `press_cube`，因此将现有完整按压包以该名称发布并收敛索引，移除网页示例入口；不删除原始数据或已发布历史版本。浏览器测试改为从已发布任务读取配套真实 GLB，并仅在测试路由中生成合成点，不再依赖已删除的 demo 资产。
- **最终浏览器验证**：任务发布／切换／后台更新／失败保留、按压证据、全部筛选与桌面／移动端布局、本地导入及导出共 4 passed。完整筛选用例在 CPU SwiftShader 下约 2.1 分钟，超时由 120 秒调整为 180 秒，断言未放宽。此前一次重跑遇到源码热更新干扰，另一次遇到外部删除 demo 资产；最终使用发布任务模型的固定测试路由完整通过。语法与空白检查通过。本次未提交混有既有改动的工作树；完整 Python 回归中的既有配置缺失仍未解决。
- **最终上线核对**：重新构建后，常驻 preview 页面只有 `press_cube` 一个任务选项、无“查看示例”入口；实际加载 319 episodes／949 有效候选，页面脚本错误为空。用于测试的本地 5173 dev 服务已停止，常驻 4173 preview 服务保持运行。

## 2026-10-23：任务选择直接使用 data 子目录

- **任务来源**：按用户要求取消 `tasks.json`，任务名称直接取 `public/data` 的一级子目录名；目录内需要 `placement.json` 和 `robot.glb` 两个普通文件，隐藏目录、符号链接和缺少配套文件的目录不列入。现有目录显示为 `press_cube_olddemo`。
- **服务与更新**：Vite dev／preview 共享只读目录扫描与文件服务，`/api/tasks` 每次直接扫描文件系统，`/data/` 始终读取 `public/data`，缺失时不回退到过时的 dist 副本。浏览器每 5 秒检查新增、改名、删除及数据修改；当前目录消失后切换到现存任务，全部移除时清空点位和机器人。文件版本用于模型请求校验，避免任务替换时把旧 JSON 与新 GLB 混用。
- **导出行为**：完整提取后复制 JSON／GLB 到 `public/data/<输出目录名>`，`--task-name` 仅覆盖实际目标目录名，不再写显示名称索引；输出已经在该目录时无需复制。替换已有任务时保留其它文件，并在目录切换失败时恢复原目录。build 只构建前端资产；数据新增／改名／更新无需 build 或服务重启，首次采用服务插件需要重启现有 preview。
- **测试**：Node 数据／目录扫描／HTTP 动态读取与版本校验共 13 passed；Python 发布及 CLI 接线聚焦测试 17 passed。检查覆盖目录改名／删除、不读取伪索引、排除符号链接、目录名限制、配套模型缺失、原目录输出不自复制和同目录其它文件保留。未运行 SDK、ROS、仿真或硬件。
- **浏览器与上线**：任务切换／改名／删除／空场景及本地导入回归 2 passed；构建与语法／空白检查通过。已重启原有 `astribot-placement-viewer` 用户服务。在线 API 只返回 `press_cube_olddemo`，旧索引地址返回 404，dist 无数据副本；页面直接加载 public 原目录，显示 319 episodes／949 有效候选，无页面脚本错误。测试 dev 服务已停止。

## 2026-10-26: Decision 3.0 image recognition CLI

- Filled the existing empty `scripts/jev_test.py` with a standard-library client for `POST https://decision.vllm-sr.ai/v1/systemone`, using `vllm-sr/d3-turbo` by default. The original PNG/JPEG/WebP image is sent as a base64 data URL with the requested Chinese question and red/blue/yellow block choices. No expected answer is included in the request.
- Usage: set `DECISION_API_KEY`, then run `python3 scripts/jev_test.py --image /path/to/photo.png`. The script prints the original JSON response, selected label, available option probabilities and request duration. `--dry-run` validates local input and previews the request with image data omitted, without requiring a key or making a network request. `--timeout` defaults to 60 seconds.
- Validation: Python syntax, actual supplied-image dry-run, base64 round-trip, mocked request/response handling, HTTP/network errors and missing credentials passed. No live inference, ROS, simulator or hardware calls were made; model accuracy remains for the user to evaluate.
