# 物品分类固定摆放 01–10

`01.json` 完整保存本次修改前的当前摆放；`02.json`–`10.json` 分别对应
`~/Downloads/02.png`–`10.png`。与 penholder 固定摆放一样，每组有独立的
实验配置和结果目录，默认评估 1 个 episode、6500 steps、seed 0。
基础入口 `../item_classification.json` 仍为 100 个 episode，步数上限也已改为 6500。
100 Hz 下每个 episode 最多推进 65 秒仿真时间，模型等待时间不计入步数。

从仓库根目录启动，替换 `01` 即可切换摆放：

```bash
python astribot_simulation_integrate.py --config config/experiments/evaluation/item_classification_fixed/01.json --check-config
python astribot_simulation_integrate.py --config config/experiments/evaluation/item_classification_fixed/01.json
```

Atlas 中选择 `simulation/item_classification_fixed_01` 至
`simulation/item_classification_fixed_10`，然后使用 Start Simulation / Start Eval。
对应入口位于相邻 `astribot_atlas` 仓库的 `configs/simulation/`。
结果分别保存到 `outputs/evaluation/item_classification_fixed/01/` 至 `10/`。
交互仿真持续运行到关闭窗口；6500 是 managed evaluation 的 episode 上限。

01–10 的玩具熊均显式引用 `config/assets/wanjuxiong.json`，加载
`~/assets/wanjuxiong_smooth_torso_w3p266.glb`，使用 FEM 弹性体和四面体碰撞。
蓝色躯干宽度减少 1 cm，从约 4.266 cm 变为 3.266 cm；手脚向内平移保持连接，
自身尺寸不变。粉色头部仍为宽约 4.879 cm、厚约 5.918 cm，头部所有几何不变。
实际总宽约 6.693 cm（由耳朵决定）、总长约 9.923 cm、含鼻子厚度约 6.463 cm。
整只熊质量 15 g、杨氏模量 30 kPa、动摩擦系数 2.0。
各组保留自己的 XY 和朝向，加载时按新几何重新计算落桌高度和 1 mm 间隙。
已运行的仿真需要先 Stop Simulation，再 Start Simulation；只执行 Reset 不会重新加载资产。
当前资产路径与材料参数见[玩具熊配置](../../../assets/wanjuxiong.json)；历史修改与深压缩验证限制保留在[开发日志](../../../../docs/DEVELOPMENT_LOG.md)的 2026-11-06 条目中。

01–10 的玩具猪也使用无绒毛模型：共享 `wanjuzhu.json` 引用仓库中的
`config/assets/meshes/wanjuzhu_smooth.glb`。移除 14 组独立毛圈后，三角面从
5,064,758 减至 106,838；身体、五官、刺绣、标签及其材质保持原样。
玩具猪仍是 15 g 动态刚体，静摩擦和动摩擦系数均为 2.0；各组 XY 和朝向保留，
初始落桌间隙仍为 1 mm。需重启仿真加载新模型，尚未实测 FPS 提升幅度。
当前资产路径、来源哈希及物理参数见[玩具猪配置](../../../assets/wanjuzhu.json)。

## 摆放约定

每份配置显式保存桌子和七个物体的 placement。01 与当前场景的全部位置、
四元数和桌子缩放逐项一致。02–10 使用现有头部相机外参、内参、镜头畸变
和实际 GLB 几何，将照片中物体轮廓中心及长轴朝向拟合到桌面。
蒸笼朝向参考底部竹条方向，圆盘和包子的朝向沿用当前值。
桌子保持 1.20 × 0.60 m、高 0.75 m、世界 XY `[0.625, 0]`。
玩具熊采用上述新尺寸，玩具猪采用上述去绒毛几何；其余物体尺寸、机器人初始姿态、相机和灯光沿用当前场景。

坐标为 `table.tabletop`，单位米；+X 朝画面左侧，+Y 朝机器人。
四元数采用 WXYZ。所有物体用实际几何底面落桌，初始间隙 1 mm。
`task.overrides.reset: null` 关闭随机重置；每次 reset 恢复同一组摆放。
玩具熊使用弹性体，其余可移动物体使用动态刚体。当前 15 g、窄躯干版本的强压测试
出现局部单元翻转，整体物理检查未通过；此前通过的是 40 g、较宽躯干版本。
真实夹爪捏合与滑落尚未复现，不能把减重视为已解决抓取问题。
成功条件沿用当前显式空列表，不会凭空判定成功；达到 6500 steps 后结束为 timeout。

## 遮挡与精度

这是单张照片的摆放估计，并非实测三维标定。照片中被夹爪遮挡的轮廓需要推测；
模型外形、材质、机器人夹爪外观及遮挡边界仍可能与实物不同。

- 02：小猪大部分被右夹爪遮挡，中心和朝向为估计值。
- 06：包子位于左手腕下方，仅小部分可见，中心为估计值。
- 10：初始照片中的面包被右机械臂完全遮挡。现已结合补充视角
  `10-modify.png`，以蒸笼、盘子、鸡腿、包子和小猪为参照重新估计位置。
  桌面 XY 更新为 `[-0.20138608, 0.18274419]` m：中心位于盘子朝机器人
  方向约 23.7 cm、略偏左 1.2 cm，距桌子前沿约 11.7 cm。
  相比原配置，向桌后移 2.8 cm、向机器人右侧移 1.3 cm；朝向保持不变。
  估计以两个时刻之间物体未移动为前提，不再仅为模拟机械臂遮挡而移动面包。
  当前初始仿真视角可见一小部分面包边缘，尚未完全复现实拍中的遮挡。
  [补充视角投影核对](../../../../outputs/evaluation/item_classification_fixed/layout10_alternate_view/alternate_view_projection.png)
  和[拟合记录](../../../../outputs/evaluation/item_classification_fixed/layout10_alternate_view/fit.json)
  保存五个参照物、相机视角拟合和移除单个参照物后的敏感性检查。

[标注记录](review/annotations.json) 保存原图 SHA-256、头部画面裁剪框、
物体轮廓框/长轴、遮挡标记、01 原始 placement 及拟合配置。
照片只裁去查看器边框并缩放到 1280 × 720，不使用额外透视变换或遮挡蒙版。

## 验证

CPU 检查十组配置、managed evaluation 准入、01 摆放一致性、独立输出路径、
熊的弹性模式/无绒毛几何/尺寸/摩擦系数、猪的无绒毛几何/刚体参数，以及实际网格的桌面边界和 1 mm 间隙：

```bash
OPENBLAS_NUM_THREADS=1 python config/experiments/evaluation/item_classification_fixed/review/verify.py check
```

在正常 Isaac Lab / ROS 环境中采集全部摆放，输出目录必须尚不存在：

```bash
source env.sh
export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=197
python config/experiments/evaluation/item_classification_fixed/review/verify.py capture --output /tmp/item-fixed-new
python config/experiments/evaluation/item_classification_fixed/review/verify.py report --output /tmp/item-fixed-new
```

采集经过 integrated launcher 和正常 reset，每组 seed 42，随后由共享的刚体/弹性体
重置接口应用各组姿态，并在物理推进前核对 reset pose。弹性蒙皮需推进 1 个物理步
刷新，再执行 32 次渲染更新；记录同时保存 reset 姿态和实际采集姿态。
所有组共享同一桌子、资产和机器人。报告校验配置指纹和原图 SHA-256。
这些检查验证初始摆放，不代表抓取、长期物理稳定性或任务成功率测试。

首次十组验证结果位于 `outputs/evaluation/item_classification_fixed/review_20261105/`：
[01–05 对照](../../../../outputs/evaluation/item_classification_fixed/review_20261105/comparison_01_05.jpg)、
[06–10 对照](../../../../outputs/evaluation/item_classification_fixed/review_20261105/comparison_06_10.jpg)、
[验证记录](../../../../outputs/evaluation/item_classification_fixed/review_20261105/validation.json)。
每组目录保存头部、左右手腕相机画面；02–10 另有真实照片裁剪和 50% 原样叠加图。
其中 10 的面包位置早于本次补充视角修正；当前 10 的重置渲染另存于
`outputs/evaluation/item_classification_fixed/layout10_alternate_view/render/`。
