# Repository Guidelines

本文档记录 Astribot Simulation 仓库级、长期有效的工程边界。通用协作方法见 [agents.md](agents.md)，用户操作说明见 [README.md](README.md)，设计与变更记录放在 [docs/](docs/)。两份根协作规范使用中文，其它代码和文档沿用原有语言约定。不要在本文件记录日期、commit、分支状态、验收快照或临时 workaround。

## 目录与入口

- [astribot_simulation.py](astribot_simulation.py)：命令行入口，接收 --config 与 --headless，负责退出信号和运行资源清理。
- [astribot_simulation_integrate.py](astribot_simulation_integrate.py)：评测 JSON 入口，--config 默认加载 penholder 评测配置，通过共享解析器创建机器人、场景资产和任务。
- [src/astribot_envs/](src/astribot_envs/)：Gymnasium 基类、工厂及 MuJoCo、Genesis、ManiSkill、Isaac Lab 后端；工厂按 simulator_type 选择实现。
- [src/simu_utils/](src/simu_utils/)：ROS1/ROS2 接口、运动学和通用工具；[src/sim_assets_tools/](src/sim_assets_tools/)：模型、网格和纹理转换。
- [config/astribot_s1/](config/astribot_s1/)：各后端与机器人配置；[astribot_descriptions/](astribot_descriptions/) 和 [astribot_msgs/](astribot_msgs/) 是模型资源和 ROS 消息的 Git 子模块。
- [env.sh](env.sh)：加载对应 ROS 消息工作区并设置项目路径和 ROS 环境；[pyproject.toml](pyproject.toml)：Python 依赖声明与开发工具配置；[scripts/](scripts/)：安装与构建入口。
- [tests/unit/](tests/unit/)、[tests/integration/](tests/integration/)：导入检查、手动工具及仿真集成测试；共享 fixture 位于 [tests/conftest.py](tests/conftest.py)。

## 稳定契约

- 修改公共环境接口时，同步检查基类、工厂注册、四个后端、ROS 接口与测试。reset 返回 (observation, info)，step 返回 (observation, reward, terminated, truncated, info)；统一接口不代表各后端传感器、控制模式或观测能力完全一致。当前运行循环采样 action，但实际关节控制主要来自 ROS 命令，不能将其描述为完整机器人 action 控制协议。
- 旧版 --config YAML 通过 yaml.safe_load 加载为可变字典，model_path 与 ASTRIBOT_SIMU_ROOT 拼接。--experiment 使用 src/simulation_config 的严格 JSON 解析与配置指纹；placement_test 与 evaluation 分别校验，不能将其保证外推到旧版 YAML 或所有后端。修改配置时核对 robot_list、joint_names_list、vel_compensation_list 的位置对应及模型中的关节、相机、传感器名称，不把未知字段、重复键或路径逃逸拒绝写成现有保证。
- ROS_VERSION 决定 ROS1/ROS2 分支；运行前先加载 ROS 及对应构建的消息工作区。env.sh 使用 ASTRIBOT_SIMU_ROOT、PYTHONPATH 和 ROS_DOMAIN_ID，诊断时应确认实际值与模块来源。消息变化须同步生产者、消费者和两套消息构建定义，不能假定 ROS1/ROS2 生成范围一致。
- 关节名称、顺序、自由度、控制模式和状态消息必须一致；位姿工具采用 [x, y, z, qw, qx, qy, qz]，与 ROS 字段或后端坐标约定转换时显式处理顺序、单位和坐标系。固定底盘反馈属于仿真兼容行为，不能等同于控制权授权或真实底盘能力。
- 相机接口采用 rgb_img、depth_img、point_cloud 字段；RGB 与深度发布编码分别为 rgb8、32FC1，实际可用字段取决于后端。相机名称、尺寸、颜色顺序、坐标约定、话题或 QoS 变化必须联动配置、模型、发布端、订阅端和验证；ROS2 SDK JPEG 发布需保持开关及订阅者条件，并在编码前正确转换 RGB/BGR。
- Isaac Lab 必须保持 AppLauncher 与后续仿真模块的初始化顺序，应用循环与相机读取在仿真主线程执行；其它后端的循环及通用相机线程按各自实现处理，不强行统一线程模型。入口负责先停止 ROS executor 和 spin 线程，再关闭环境、节点及 ROS；生命周期变化须核对线程、渲染资源与应用的归属和关闭结果。
- 模型转换须维护 MJCF、URDF、USD 的引用、名称、比例、坐标和纹理关系；视觉修改不得无意改变关节、惯量、质量或碰撞。资源子模块与消息子模块的修改及父仓库指针更新应明确纳入任务范围。

## 安全边界

- 未取得明确硬件授权时，仅做静态、mock、只读检查或已确认隔离的仿真/loopback 验证。仿真会创建 ROS 发布者、订阅者和服务，不能凭项目名称或 headless 模式推断通信隔离；运行前确认 ROS master 或 domain、发现范围与目标设备。
- 控制命令、重置、相机服务调用和资产转换都具有副作用，必须符合任务授权与已确认的目标范围；仿真状态标记或服务回复不证明真实硬件安全隔离。
- 停止和清理仅作用于确认归属的进程、节点、线程、渲染资源与文件。归属不明、并发使用或关闭失败时保留现场并报告，不做全局 pkill、资源删除或批量子模块清理。
- 外部命令、消息等待与关闭操作应有明确边界、超时和错误原因；凭据、私钥、用户私有路径及实验数据不得泄露。安装、消息构建和模型转换不是只读 preflight，不得作为普通文档核查自动执行。

## Git 与合并审查

- 遵守 agents.md 的工作树保护和 Git 安全规则；先确认根仓库与子模块各自的 staged、unstaged、untracked、ignored 变更，不覆盖或吸收他人修改。
- 合并前核对 merge-base、双方变更及关键规则；合并后逐路径检查语义。禁止以整文件 ours/theirs 替代冲突分析。
- README、协作规范、配置、依赖声明、安装运行脚本和消息构建定义变更，须执行对应的文档、静态或聚焦验证；不得把文档审查扩展为未经确认的安装或设备操作。
- 区分子模块源码、生成的消息链接与 build/install/log 目录、模型输出和父仓库指针；不因工作树 dirty 要求重置、清理或拒绝研究工作。

## 内部研究模式

- 当前源码、配置、pyproject.toml、所选安装脚本及子模块资源是工作依据；依赖声明含版本范围和部分固定版本，不宣称已有完整依赖锁或 native ABI 认证，也不额外建立发行认证链。
- 通用安装入口是 [scripts/install.sh](scripts/install.sh)，消息构建入口是 [scripts/build.sh](scripts/build.sh)；轻量安装器位于 [scripts/lite_install/](scripts/lite_install/)。根据实际后端、ROS 版本、解释器与模块来源确认环境，不指定全仓唯一 Conda 环境。
- 安装器可能改动系统包、Python 环境、shell 配置、子模块和构建输出；不能把 dirty 状态作为安装、启动或回滚门禁，也不能为安装抹去已有修改。
- 本仓库使用 pytest 和已有静态工具，没有单一统一无硬件验证脚本。导入、仿真、GPU 渲染、ROS 通信和真实硬件须分别报告，不能从内部测试推断未执行的现场能力。

## 变更与验证

1. 先阅读适用规范、相关实现、配置、依赖、测试和工作树，明确输入、输出、资源归属、副作用和失败边界；沿现有接口与 helper 最小修复，不为过时 fixture 增加兼容层。
2. 公共 API、配置、消息、相机、资产或生命周期变化，应同步关联实现、测试和文档；用户行为或协作规则变化时追加 [docs/DEVELOPMENT_LOG.md](docs/DEVELOPMENT_LOG.md)，历史记录只追加、不回写。
3. 导入检查使用 python -m pytest tests/unit/test_simulator_imports.py，先确认对应环境；缺少可选后端会 skip，Isaac 项检查的是 Isaac Sim 导入，不代表后端运行通过。pytest 默认覆盖率选项来自 pyproject.toml，会要求相应开发依赖并产生报告。
4. tests/unit/test_camera_viewer.py 是需要 ROS、cv_bridge 和 GUI 的手动订阅工具，收集整个 tests/unit/ 也可能因其顶层导入失败，不能称为纯无 ROS 单元测试。tests/integration/test_simulation_pipeline.py 会创建仿真并发布 ROS；仅在确认隔离和依赖后运行，ROS_VERSION 非 2 或可选后端未安装时会跳过，当前没有 Isaac Lab pipeline 用例。
5. 文档修改检查标题、链接、路径、事实和空白；脚本或代码修改按需运行无写入的语法检查与聚焦测试，再按风险扩大。只有确认实现会提前返回时，才执行 --help 或 dry-run；安装器和转换工具不得按参数名称推定无副作用。
6. 分别记录静态/mock、导入、隔离仿真/loopback 和真实硬件覆盖，列明失败、skip 与未执行项。软件 PASS 不能代替硬件或全后端验证，不为取得绿灯降低检查要求。

## 垃圾回收

控制新增文件和维护成本：优先复用现有实现与测试，临时产物放在临时目录，仅保留与交付直接相关且可复用的内容；收尾时检查并清理本任务产生的无用文件和代码，不删除或覆盖用户已有改动、模型资源及归属不明的产物，不借清理扩大重构范围。
