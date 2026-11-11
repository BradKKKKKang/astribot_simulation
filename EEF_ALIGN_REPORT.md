**EEF 数据与回放方案 · 2026-11-11**

目标：对照原视频检查校正后的 EEF，优先使用离线 SDK IK、CPU USD FK 和 Web 播放，避免逐帧仿真渲染。本报告记录只读调查结果与待实现方案。

**数据现状。** 样本位于 `ssh aliyun:/mnt/oss/cognition/QW_Data/ce_data_astribot/eef_sample`。导出快照记录 8 个数据集、25 类映射、2,040 个 root、10,158 条轨迹、3,587,605 帧和 33,401 路视频。本轮实际读取每类一条，共 25 条、13,722 帧，未重新全量验收。

- 每条轨迹含 `eef.parquet`、`source.json`、`videos/`；所属 root 的 `MAPPING.json` 保存映射合同。样本直接复制已校正值，不应再次套用转换旋转或夹爪归一化。
- `observation.state/action/eef_mask` 为 34 维：躯干 `[0:9]`、左臂 `[9:18]`、左夹爪 `18`、右臂 `[19:28]`、右夹爪 `28`、头部 `[29:31]`、底盘 `[31:34]`。姿态编码为 `xyz + 旋转矩阵前两行`。
- 完整手臂姿态分别检查 `mask[9:18].all()`、`mask[19:28].all()`。抽查轨迹的躯干、头部均无有效数据；不能把零占位送入 IK。该样本集合的夹爪约定为 `0=开、100=闭`。
- 使用 `sample_timestamp = timestamp - timestamp[0]` 对齐零起点视频；抽查 FPS 为 7–30，不能固定按 30 FPS。部分 action 是下一帧 state，视频对照默认使用 state，并单独显示 action。
- 抽查 parquet 未含实际相机位姿/内参列；像素级叠加需另追溯标定。视频存在 HEVC，浏览器兼容性需处理。
- 当前正式 RoboMINDv2 已恢复 45 个 root 的缺失 EEF，其说明明确指出 `eef_sample` 尚未刷新；样本与正式数据须分别绑定版本。

**坐标结论。** 34 维格式一致，不代表已统一到 Astribot chassis。下表针对标准 EEF；保留的原始辅助字段可能使用其他参考系。

| 数据范围 | 转换后的参考系 |
|---|---|
| A1：Franka / Genie1 / Lift2 / Split Aloha | 各源机器人的整机 robot-base；位置保留，末端局部轴向校正 |
| M1 Agilex、RoboMINDv2 Franka | 左右各自 arm-base；共同基座变换未知 |
| M1 Franka（Panda） | `panda_link0`；源末端 `panda_hand`，单臂放入右臂槽 |
| DROID、PrimeBot、RoboMIND v1、RW-RL R1Lite | 保留配置声明的源 robot 参考系，另做末端局部旋转 |
| RW-RL G2 | 源参考系旋转：位置 `(x,y,z) → (x,z,-y)`，另做末端局部旋转 |
| XDOF | 保留源参考系；转换记录未进一步确定物理原点 |
| ABC-130K（不在当前 eef_sample） | 原始 world，位置不变，末端轴向校正 |
| HABIT（不在当前 eef_sample） | 每个 episode、每只手臂各自首帧 state 坐标系 |
| AgiBotWorld2026、GenieSim3.0（不在当前 eef_sample） | 源位置旋转为 `(x,z,-y)`；源物理原点仍需确认 |
| Challenge Real/WorldModel、Galaxea、两类 agibot_data（不在当前 eef_sample） | 保留源/既有已验收参考系；现有说明不足以统一认定物理原点 |

普通转换满足 `p_out = Q p_source + t`、`R_out = Q R_source L`；右乘 `L` 只改末端轴向。HABIT 则为 `p'=R0ᵀ(p-p0)`、`R'=R0ᵀRQ`，一个 parquet 内不同 episode 可以有不同原点。具体以正式 root 的 `meta/conversion.json → plan.coordinates` 或对应 alignment 记录为准；样本以 `MAPPING.json → contract.coordinates` 为准。

**Astribot 对照基准。** `/home/zihengkang/Datasets/real_eval/cap_pen_merged/data/chunk-000/episode_000000.parquet` 共 1,030 帧。离线 FK 数值支持其 EEF 使用 SDK chassis 坐标：关节 command 对 EEF command 的两臂平均位置残差约 0.50 mm；改成 world 假设则约 25–33 mm。state 对同帧实测关节 FK 仍有约 10 mm 残差，原因未确认。

- 当前 S1 模型：`p_torso_base = p_chassis - [0,0,0.097]`；场景 world 还需应用机器人放置变换。
- SDK 手臂 tool 偏移为末端连杆局部 `[0,-0.15,0] m`；不能混用 placement 的抓取 TCP 定义。
- 正确变换参考系可保持同一物理目标；跨本体还需处理工作空间、限位和工具差异。没有证据表明某一种参考系天然具有更高任务成功率。

**建议链路。**

```text
样本 state/action + mask + 映射 + 时间
    → 显式基座/工具适配、缺失部位初姿
    → SDK 连续 IK → 关节与误差缓存 → CPU USD FK
    → Web 机器人 + 原始 EEF 目标 + 原视频，共用时间轴
```

- 默认 state 对照视频，action 作为目标层；原始 EEF 与 IK 达成 EEF 同时显示，区分校正问题和求解残差。
- 明确配置初始关节，随后以上一帧 IK 解连续求解；逐帧保存失败、位置/角度误差、关节跳变。未知基座关系可使用显式“相对运动”展示，但不能据此验证双臂绝对关系。
- 按 episode 预计算、按需加载；缓存绑定源版本、映射、SDK/模型、工具定义、初姿和采样策略。视频支持 Range；暂停、拖动和倍速不重复求解。
- 当前 `web/placement-viewer` 主要展示事件快照，需补完整时间轴与视频联播；复用 [scene.pose](web/placement-viewer/src/scene.js) 和 [UsdKinematics](src/placement_analysis/reference_pose.py)。
- 已有原型在 `outputs/eef_ik_usd_fk/`：两条历史轨迹的公共 SDK IK 全帧返回成功，但最大 EEF 残差仍约 16 / 34 mm；相同关节下 USD FK 与 PhysX 所检连杆最大差约 0.0011 mm。属于历史证据，本轮未重跑。
- 公共 IK 预览不等于完整 action-chunk 执行：后者还有插值、WBC、限幅与反馈。需要物理跟踪/接触验证时，增加 `SDK → 无相机渲染的 managed physics → 实际关节反馈 → Web` 对照。

实施顺序：25 类各一条完成数据/坐标审查与视频同步 → 接入 IK/FK 和误差展示 → 少量无渲染物理对照 → 批量缓存。尚未验证本批跨本体数据的 Astribot IK 成功率与整体速度。
