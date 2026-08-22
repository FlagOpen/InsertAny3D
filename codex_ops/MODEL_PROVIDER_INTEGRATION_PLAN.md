# 多模型 3DGS 集成调用中心落地方案

更新时间：2026-08-21

## 1. 目标和硬约束

目标是在现有服务器流水线中增加一个可配置的模型提供方（provider）入口，首批支持
`trellis`、`sam3d`、`hunyuan`。调用方只选择模型和少量模型参数，后续阶段统一消费
标准 3DGS、统一渲染视图、GIM 和 pose 结果。

硬约束：

1. 最终交付物必须是 Gaussian PLY/3DGS，不能因为 Hunyuan 原生输出 mesh 就改变
   Unity 的导入契约。
2. 所有 provider 都必须经过坐标契约转换和服务器渲染；坐标轴、原点、归一化尺度、
   相机 near/far 和朝向参数由 provider 自己声明，不能在编排器里硬编码一套参数。
3. 输入分割必须可管理、可复现、可审计。模型自带的 rembg 只能作为该模型的兜底预处理，
   不能替代流程需要的目标 mask、锚点 mask 或生成后 SAGS mask。
4. SAM 3D Objects 使用现有 `third_party/TRELLIS/.venv`，不替换其 CUDA、PyTorch、
   torchvision 或已编译扩展配置。
5. 第三方源码和权重放在 `third_party/<MODEL>`，新写的 provider、转换器、契约和测试
   放在独立目录（建议 `model_center/`），不把适配逻辑提交进第三方目录。
6. 一次只运行一个 `Task_*`；每次任务保留完整 manifest、日志和中间产物。

## 2. 现状和调研结论

### 2.1 现有流水线

当前入口是 `codex_remote_tools/run_insert_pipeline.py`，TRELLIS 生成、
`render_trellis_views.py` 渲染、GIM、联合 pose 和 SAGS 都直接由 TRELLIS 路径驱动。
任务目录已经具备可复用的阶段边界：

```text
01_segmentation/  02_trellis/  03_rendered_3dgs/
04_gim/            05_pose/    06_sags/  logs/  manifest.json
```

渲染器消费的是带位置、颜色、尺度、旋转、opacity 的 Gaussian PLY，并输出 RGB、
径向深度、COLMAP 相机和 3DGS model。pose 当前使用旧 ModelScope 风格的点级跨视角
一致性筛选和一次联合相似变换，这个规则应保持不变。

### 2.2 TRELLIS

官方 TRELLIS image pipeline 原生同时提供 Gaussian、radiance field 和 mesh，
`outputs['gaussian'][0].save_ply()` 可以直接形成标准入口；现有 renderer 已针对
TRELLIS 的归一化坐标和 SH/深度兼容性做过处理。

### 2.3 SAM 3D Objects

官方输入是图片加 object mask，示例直接调用 `output["gs"].save_ply()`，因此它天然
满足 3DGS 产物要求，不需要 mesh-to-Gaussian。官方 setup 以 Linux、至少 32GB
显存和 Python 3.11 / PyTorch 2.5.1 + CUDA 12.1 为支持口径；当前远端 TRELLIS
环境正是 Python 3.11.13、PyTorch 2.5.1+cu121。适配层在不改变精度、采样步数和
标准 Gaussian decoder 的前提下，已于 23.6 GiB RTX 3090 完成真实生成和渲染。
因此 capability report 保留 32 GiB 为上游推荐值，把 23.5 GiB 设为当前经过实测的
硬门槛；不能据此推断未测试的更小显存也受支持。

当前环境已有 kaolin、hydra、numpy、opencv、trimesh、xatlas 等主要依赖，并已补齐
`pytorch3d`、`gsplat`、MoGe 和 SAM3D inference extra。安装过程只补非核心包，没有
升级或降级 CUDA、PyTorch、torchvision 及 TRELLIS 已编译扩展。

SAM 3D 的官方代码/权重需要受控模型仓库访问；权重不能提交到仓库。当前规划优先
使用用户指定的 ModelScope 镜像，避免额外引入一套 Hugging Face 凭据。它的 license
是 Meta SAM License，使用和再分发需保留协议及贸易管制约束。

### 2.4 Hunyuan3D

官方 Hunyuan3D 2/2.1 image-to-shape pipeline 返回 `trimesh` mesh，纹理阶段返回
带材质的 mesh/GLB；官方并不把 3DGS 作为原生输出。服务器上目前已有独立
`third_party/Hunyuan3D-2/.venv`，其 PyTorch 2.7.1+cu118 与 TRELLIS 的
2.5.1+cu121 不同，不能把它的依赖强行装进 TRELLIS 环境。provider 中应调用它已有
环境，再在共享适配层执行 mesh-to-Gaussian。

Hunyuan 受 Tencent Hunyuan 3D 2.0 Community License 约束，协议对地域、分发、
服务披露和可接受用途有额外要求；在启用 provider 前应由项目负责人确认使用地域和
产品形态合规。

## 3. 推荐架构

新增独立目录（不放入第三方源码）：

```text
model_center/
├── __init__.py
├── cli.py                         # 统一命令入口
├── config.py                      # provider/profile 配置校验
├── contracts.py                   # 输入、Gaussian、坐标、渲染契约
├── registry.py                    # provider 注册和能力发现
├── orchestrator.py                # 阶段编排，复用现有 GIM/pose/SAGS
├── providers/
│   ├── base.py                    # Provider 接口
│   ├── trellis.py
│   ├── sam3d.py
│   └── hunyuan.py
├── segmentation/
│   ├── manager.py                 # 统一 mask 管理
│   └── engines.py                 # external/alpha/rembg/grounded_sam
├── transforms/
│   ├── base.py
│   ├── trellis.py
│   ├── sam3d.py
│   └── hunyuan.py
├── converters/
│   └── mesh_to_gaussian.py        # Hunyuan mesh -> 标准 Gaussian PLY
├── renderers/
│   └── provider_render.py         # 统一调用现有渲染核心
└── tests/
    ├── test_contracts.py
    ├── test_transforms.py
    └── fixtures/
```

`codex_remote_tools/run_insert_pipeline.py` 仍是兼容入口，但 provider/profile 解析、
权重身份、坐标转换和 renderer 命令已经由独立 `model_center/` 适配层提供；后续可继续
把剩余阶段编排迁入 `model_center.orchestrator`。`tools/run_local_edit_remote_pipeline.py`
只负责 `model_provider`、profile 和 provider 参数透传，不复制模型分支逻辑。

### 3.1 Provider 接口

```python
class ModelProvider(Protocol):
    name: str
    runtime_python: Path

    def check_environment(self) -> EnvironmentReport: ...
    def prepare_input(self, request: GenerationRequest) -> PreparedInput: ...
    def generate(self, prepared: PreparedInput, output_dir: Path) -> GeneratedAsset: ...
    def coordinate_contract(self) -> CoordinateContract: ...
```

`GeneratedAsset` 至少包含：`gaussian_ply`、可选 `source_mesh/glb`、输入 mask 引用、
provider/model/version、随机种子、权重标识、坐标契约、转换记录和 SHA-256。

### 3.2 配置示例

建议在 `codex_ops/insert_workflow.defaults.json` 增加：

```json
{
  "model_provider": "trellis",
  "model_profile": "default",
  "provider_options": {},
  "segmentation": {
    "engine": "external_or_grounded_sam",
    "input_mask_policy": "required_for_sam3d",
    "fallback": "rembg"
  }
}
```

profile 只描述默认值，不允许任务偷偷修改算法默认值。每次运行的
`workflow_defaults.json` 和 `manifest.json` 写入解析后的完整 profile，保证续跑和
审计可复现。

日常切换只改两个字段；统一入口不会因为某个 provider 不可用而静默回退：

| 目的 | `model_provider` | `provider_options` |
| --- | --- | --- |
| 现有生产路线 | `trellis` | `{}` |
| SAM3D 标准质量 | `sam3d` | `{}`（解析为标准 decoder、dist=1、native spconv） |
| SAM3D 低密度显式模式 | `sam3d` | `{"decoder":"gaussian_4"}` |
| SAM3D 高 voxel 应急 | `sam3d` | `{"downsample_ss_dist":2}`，仍失败才改为 4 |
| Hunyuan shape-only | `hunyuan` | `{}` |

SAM3D 还必须提供 `model_input_mask`（或可生成该 mask 的 prompt）；Hunyuan 是否启用
paint/PBR 由独立选项控制。所有 options 都会写入 manifest 和 candidate id。

## 4. 分割策略

把分割拆成三个明确用途：

| 用途 | 输入 | 默认策略 | 产物 |
| --- | --- | --- | --- |
| 生成前目标 mask | 编辑后的中心图 | 优先人工/Unity mask；否则 GroundingDINO + SAM（现有 legacy/auto） | `00_model_input/mask.png`、boxes、detections |
| 生成前 anchor/composite mask | 编辑后的中心图 | 多 prompt 取并集；仅 composite profile 启用 | `00_model_input/composite.png`、`mask_manifest.json` |
| 生成后 SAGS mask | 生成的 `center.png` | 现有 LangSAM/legacy；用于从组合 Gaussian 提取插入物体 | `01_segmentation/mask.png`、`points.json` |

规则：

- `sam3d` 必须有单物体 mask；没有 mask 时失败，不默认对整张图调用。
- `trellis` 延续当前 `composite` 默认；`cutout` 仍可兼容。
- `hunyuan` 使用 RGBA/cutout 作为输入。Hunyuan 的 rembg 只在没有外部 alpha 时
  作为 provider-local fallback，必须把 `engine=rembg`、模型版本和输出 alpha 写入
  manifest。
- 中心图上的外部 mask 与生成后 mask 不复用。前者约束生成输入，后者负责 SAGS 点
  对齐；两者尺寸、语义和坐标系分别记录。
- 统一的 `MaskArtifact` 包含原图 SHA、mask SHA、尺寸、来源引擎、prompt、阈值、
  检测框、实例索引和是否人工确认。

推荐优先级：`provided_mask` > `grounded_sam`（现有 GroundingDINO + SAM） >
`langsam` > provider-local `rembg`。`rembg` 不能用于 anchor 语义筛选，也不能替代
生成后的 SAGS segmentation。

## 5. 3DGS、坐标和渲染契约

### 5.1 统一中间格式

所有 provider 在离开 `generate()` 前必须写出标准 Gaussian PLY，字段至少兼容现有
`gaussian_model.py`：`x/y/z`、`f_dc_*`、`opacity`、`scale_*`、`rot_*`；SH degree
不足时补零。可选保留 `source_mesh` 供诊断，但后续阶段只读标准 Gaussian。

### 5.2 Provider 坐标契约

每个 provider 返回：

```json
{
  "sourceFrame": "provider_native",
  "targetFrame": "generated_world",
  "axisMatrix": [[...], [...], [...]],
  "handedness": "right",
  "upAxis": "y",
  "forwardAxis": "z",
  "origin": "object_center",
  "normalization": {"mode": "aabb_max_extent", "scale": 1.0},
  "renderDefaults": {"distance": 1.5, "near": 0.8, "far": 1.6},
  "unityImport": {"generatedAxis": "legacy-flip-z"}
}
```

转换层必须同时变换 Gaussian 的位置、旋转和尺度；不能只翻转 xyz。转换前后写
`coordinate_transform.json`，并用非对称合成模型验证轴向、手性、相机投影和单位尺度。
最终 Unity pose 仍由 GIM + 联合相似变换求出，不要为不同模型恢复 left/center/right
独立互相否决的旧逻辑。

### 5.3 渲染

第一版直接复用现有 `render_trellis_views.py` 的渲染核心，但把入口改成读取
`CoordinateContract` 和 `GeneratedAsset`，文件名仍为 `left/center/right`，输出仍为
RGB、absdepth、COLMAP 相机、`model/`。provider 可以覆盖：canonical 相机半径、
near/far、yaw/pitch、front-axis 和是否需要轴变换。

所有 provider 都必须经过：

1. canonical 三视图 smoke render；
2. 按 Unity 粗 pose 的真实外参重渲染；
3. GIM、联合 pose 和（组合任务）SAGS。

如果 Hunyuan 的 mesh-to-Gaussian 只产生表面 splats，深度仍使用同一 Gaussian
renderer 输出；需要在诊断中标记其深度性质与 TRELLIS/SAM3D 原生 Gaussian 不同。

## 6. SAM 3D 接入方案

### 6.1 third_party 和环境

服务器执行一次性准备（不纳入适配层提交）：

```text
third_party/SAM3D-Objects/       # 官方源码 checkout/tag
third_party/SAM3D-Objects/checkpoints/modelscope/  # ModelScope 权重，禁止提交
```

使用 `third_party/TRELLIS/.venv/bin/python`。先记录当前核心版本和 CUDA 扩展清单，
再以 `--no-deps` editable 安装官方源码，逐项补齐缺少的非核心依赖。SAM3D 官方
requirements 中 `torch==2.5.1+cu121`、`torchvision==0.20.1+cu121` 与现环境匹配；
不得执行会升级/降级它们的普通 `pip install -e '.[inference]'`。

执行前检查显示 TRELLIS venv 尚未安装 `MoGe`、`gsplat`、`pytorch3d`；这三项是
SAM3D 最小推理验证的明确前置项。接入后这三项已经按固定 commit/版本补齐；另有官方 inference extra
对 `kaolin` 的版本要求与现有 `kaolin==0.18.0` 存在分歧，必须通过最小导入/推理
测试确认是否真的需要降级，默认不降级。

需重点验证/补装：

- SAM3D 自身 package、`hydra-core`、MoGe；
- `pytorch3d` 对当前 torch 的可编译性；
- 官方 inference extra 的 `gsplat`；
- `kaolin`/`xatlas` 版本冲突。优先保留现有 TRELLIS 版本，适配代码绕开不需要的
  mesh 后处理，不为 SAM3D 改动 TRELLIS renderer 依赖。

若 `pytorch3d` 或 `gsplat` 无法在当前环境无损安装，停止安装并报告，不创建第二个
SAM3D 环境；先采用官方 `rendering_engine=pytorch3d` 的最小推理路径评估缺失项。

### 6.2 权重来源、认证和落盘约定

#### SAM3D（首选 ModelScope）

模型来源固定记录为：

```text
https://www.modelscope.cn/models/facebook/sam-3d-objects
model_id: facebook/sam-3d-objects
revision: 执行时解析并固定，不能长期依赖可变 master
```

ModelScope 页面当前包含官方 `checkpoints/` 目录（pipeline、encoder、generator、
Gaussian/mesh decoder 等文件），总存储量约 13GB。执行时应下载完整目录，不能只取
单个 `slat_generator` 文件；具体 pipeline profile 再决定实际加载的 encoder/decoder。

认证沿用服务器已有的 ModelScope access token：

- 优先读取服务器现有的 ModelScope CLI/SDK 登录凭据；
- 若执行脚本需要显式变量，使用运行时注入的 `MODELSCOPE_API_TOKEN`；
- token 只存在于 secret store、受限环境变量或 ModelScope 本地凭据文件中；禁止写入
  Git、此文档、命令行参数、任务 JSON、`workflow_defaults.json`、manifest 和日志；
- 下载日志只能记录 `model_id`、固定 revision、文件数量、大小和 SHA-256，不能打印
  Authorization header 或 token 内容。

建议缓存与第三方源码分离：

```text
/opt/data/private/ljn/.cache/modelscope/       # SDK 下载缓存
/opt/data/private/ljn/InsertAny3D/third_party/SAM3D-Objects/
└── checkpoints/modelscope/                    # 可选 materialized 路径，禁止提交
```

执行时在 `weights_manifest.json` 记录 `source=modelscope`、model id、revision、
download timestamp、文件清单、SHA-256、pipeline 配置路径和 license acknowledgement。
本次实际落盘路径为
`third_party/SAM3D-Objects/checkpoints/modelscope/`，清单为
`third_party/SAM3D-Objects/checkpoints/modelscope_weights_manifest.json`；ModelScope
固定 revision 为
`b7f2543810586d96ce6185e549f4d695e51560a8`，源码 checkout 为
`f91db411c50efee93d8db7aeb323885650f6f722`。清单包含 32 个远端 blob、总大小
13,167,432,538 bytes，weight fingerprint 为
`a8ac13404ae72f1cbedbe8228719d35c9007782ec34b2da4bb76b30fc503967c`；pipeline 配置为
`checkpoints/modelscope/checkpoints/pipeline.yaml`。`checkpoints/ss_encoder.safetensors`
是 ModelScope API 报告的零字节占位文件，清单将它列入
`omittedZeroByteRemoteFiles`，实际推理使用同目录的 `ss_encoder.ckpt`。

#### 备用来源和其他模型

- **SAM3D 备用来源**：官方 Hugging Face `facebook/sam-3d-objects`。只有 ModelScope
  不可用或文件不完整时才启用，仍使用同一 `weights_manifest.json`，并明确记录
  `source=huggingface`；不同时维护两套可变副本。
- **TRELLIS**：继续使用现有 `microsoft/TRELLIS-image-large` 缓存/模型目录；不因
  provider 中心重复下载，执行时只校验 revision 和权重哈希。
- **Hunyuan3D**：优先复用服务器现有 `third_party/Hunyuan3D-2/.venv` 和已有模型
  缓存；缺失时再从 Tencent 官方 Hunyuan3D-2 模型页或可验证的 ModelScope 镜像补齐。
  本次服务器已存在 Hugging Face cache snapshot
  `9cd649ba6913f7a852e3286bad86bfa9a2d83dcf`（缓存目录
  `/opt/data/private/ljn/.cache/huggingface/hub/models--tencent--Hunyuan3D-2/`），包含
  `hunyuan3d-dit-v2-0` shape、`hunyuan3d-paint-v2-0` paint 和对应 VAE 子目录；本次不
  重复下载。shape、paint/PBR 权重分别记录，不能把 mesh 权重误标为 3DGS 权重。
- **MoGe**：SAM3D ModelScope snapshot 不包含独立的 MoGe depth checkpoint。本次从
  官方 Hugging Face 仓库 `Ruicheng/moge-vitl` 固定 revision
  `ad326bfb61facd6c52b5a825bc1e34d7c97d9672` 获取 `model.pt`，落盘为
  `third_party/SAM3D-Objects/checkpoints/modelscope/moge-vitl-model.pt`（禁止提交）。
  文件大小 1,256,823,446 bytes，SHA-256 为
  `da96b09a0485a3c45a5aa455e67743c8b4efc4dd8437c1f2aa93c2b4303d957f`；可提交的元数据
  位于 `codex_ops/sam3d_moge_weights_manifest.json`，不包含 access token。运行时优先
  使用该本地路径，不隐式访问网络。
- **GroundingDINO/SAM/SAM2/LangSAM/GIM**：沿用现有本地或 third_party 权重，后续
  由统一 segmentation manager 做能力检查；缺失时只报告，不隐式联网下载。
- `pytorch3d`、`gsplat`、Kaolin 等属于代码/编译依赖，不进入权重清单；其版本和
  安装来源单独写入环境报告。

首次接入只实现单物体：读取 `image.png + mask.png`，调用官方
`Inference(config_path, compile=False)`，取 `output["gs"]`，调用其 `save_ply()`，
再由统一坐标适配器写成标准 PLY。多物体 `make_scene()` 留到第二阶段，避免一开始
把 layout/pose 语义与 InsertAny3D 的 anchor pose 混在一起。

### 6.3 SAM3D 坐标验收

使用一个非对称盒、带明显颜色标记的合成输入，验收：输出 PLY 的 up/forward 方向、
旋转四元数、尺度、三视图正面以及 Unity 粗 pose 重投影误差。所有转换参数落到
`providers/sam3d.py` 的 profile，不写进第三方源码。

当前 24GB 节点已完成标准 Gaussian 的 GPU 生成、canonical 三视图 RGB/depth 渲染，
以及自洽三视图的 GIM/联合 pose smoke。provider manifest 将该契约标为
`gpu_render_smoke_complete_axis_pending`：这证明 identity 适配能被现有 renderer 和 pose
链路正确消费，但严格的非对称左右手性夹具与真实 Unity 外参基线仍未完成，不能把该
状态解释为最终轴向验收。

### 6.4 24GB 显存结论和配置优先级

同一张 256x256 图片、同一 mask、seed=1、25+25 官方采样步数，在 RTX 3090
（23.5685 GiB）、torch 2.5.1+cu121 上的分阶段实测如下。峰值为 PyTorch
`max_memory_reserved`，外部 `nvidia-smi` 观测峰值约 13,347 MiB。

| 配置 | stage-1 voxels（前 -> 后） | Gaussian 数 | 峰值 reserved | PLY 大小 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| `gaussian`, dist=1 | 5,920 -> 5,920 | 189,440 | 13,631,488,000 B（12.695 GiB） | 12,882,336 B | 默认质量配置，通过 |
| `gaussian_4`, dist=1 | 5,920 -> 5,920 | 23,680 | 13,631,488,000 B（12.695 GiB） | 1,610,655 B | 峰值不降，输出密度降 8 倍 |
| `gaussian`, dist=2 | 5,920 -> 5,920 | 189,440 | 13,631,488,000 B（12.695 GiB） | 12,882,336 B | 本样本无收益 |
| `gaussian`, dist=4 | 5,920 -> 5,920 | 189,440 | 13,631,488,000 B（12.695 GiB） | 12,882,336 B | 本样本无收益 |

定位结果和采用顺序：

1. **先做不降低效果的修复。** `spconv==2.3.6` 的 auto 路径在 stage 2 首个 sparse
   convolution 选择 implicit-GEMM 后触发 native `SIGFPE`，不是 CUDA OOM，也不是
   decoder OOM。适配层默认进程内设置 `SPCONV_ALGO=native`；模型权重、浮点精度、
   采样步数和输出 decoder 均不变。再只加载 3DGS 契约实际使用的标准 decoder，并在
   每个阶段结束后释放已经不会再访问的模块。实测分阶段 reserved 峰值为：模型加载
   11.084 GiB、pointmap 12.695 GiB、stage 1 10.990 GiB、stage 2 4.211 GiB、decoder
   0.574 GiB。瓶颈是 pointmap/模型驻留，不是 decoder。
2. **`gaussian_4` 不是 24GB 峰值优化。** 它与标准 decoder 的 transformer 结构和
   checkpoint 大小接近，只把每个 voxel 的 Gaussian 数从 32 改为 4；因此主要减少
   PLY 大小和后续渲染负载，不能降低发生在 pointmap/stage 1 之前的峰值。它仅作为用户
   明确选择的低密度/吞吐 profile，不能自动回退，也不作为默认质量配置。
3. **最后才启用 `downsample_ss_dist`。** 上游实现按邻域距离裁掉 stage-1 内部 voxel，
   不是对所有坐标做固定比例抽样；当本样本只有 5,920 个表面 voxel 时，dist=2/4 均未
   删点，显存和输出完全不变。只有诊断报告显示 stage-1 voxel 已进入高密度区（issue
   建议重点关注 >10k）且标准配置确实 OOM 时，才依次尝试 2、4，并把删点前后数量写入
   `sam3d_memory_report.json`，接受其可能造成的几何质量损失。

默认 profile 因而固定为 `decoder=gaussian`、`downsample_ss_dist=1`、
`spconv_algo=native`、`load_unused_decoders=false`、`sequential_offload=true`。调用者可用
`provider_options` 显式覆盖；完整解析后的 options 已进入 candidate cache key，配置切换
不会误复用旧 PLY。

## 7. Hunyuan 接入和 mesh-to-Gaussian

Hunyuan provider 使用当前 `third_party/Hunyuan3D-2/.venv`，调用官方 shape pipeline；
如启用纹理则再调用 paint pipeline，先保存原始 GLB/mesh。

第一版转换器建议纯 Python/Trimesh、确定性、可测试：

1. 读取 GLB/mesh 的三角面、顶点法线、UV 和材质/纹理；
2. 按三角形面积和配置密度进行表面采样，保留 barycentric UV 颜色；
3. 每个采样点生成一个 Gaussian，切向尺度由局部三角形/采样间距决定，法向尺度为
   可配置薄层厚度，opacity 做有限值裁剪；
4. 由法线构造稳定切向 frame 和 quaternion；
5. 应用 Hunyuan native -> generated_world 的轴、中心化和归一化；
6. 写标准 PLY，同时输出 `mesh_to_gaussian.json`（输入 hash、采样数、密度、厚度、
   材质策略、转换矩阵）。

不要在第一版把 mesh 当作 Unity 最终产物，也不要直接复用 TRELLIS 的内部
`Gaussian` 类写入第三方目录。若表面 splat 质量不足，再评估 Mesh2Splat 或
“mesh 多视图渲染 + gsplat 优化”作为可插拔 converter；这属于质量增强阶段，不阻塞
调用中心的接口落地。

## 8. 编排改造顺序

### Phase 0：契约和基线

- [x] 固化 `GenerationRequest`、`GeneratedAsset`、`MaskArtifact`、`CoordinateContract`、
  `RenderRequest` 的 JSON schema。
- [x] 为现有 TRELLIS 建立 `TrellisProvider`，行为保持不变；旧 CLI 通过兼容 wrapper 调用。
- [x] 将 provider/version/contract/converter、输入 mask/hash、seed 和权重身份写入 manifest
  和 candidate id。

### Phase 1：统一分割和 provider registry

- [x] 抽出 `segmentation/manager.py`，接入现有 provided/legacy/LangSAM 路径。
- [x] 实现 `registry.py` 和 `check_environment`，启动前输出可读的能力报告。
- [x] 加入 profile 校验、权重路径检查、显存要求和 license acknowledgement 字段。

### Phase 2：SAM3D（首个新模型）

- [x] 在用户提供 ModelScope 权重后，于 TRELLIS venv 做无核心版本变更的安装和
  `py_compile`；官方 inference 模块已完成导入验证。
- [x] 实现单物体 SAM3D provider、坐标契约、标准/gaussian_4 decoder 选择、分阶段显存
  报告、PLY 边界和三视图渲染命令；标准质量配置已在 24GB RTX 3090 真实运行。
- [x] 彩色合成单物体 fixture、canonical RGB/depth、GIM/pose smoke 已通过。
- [ ] 严格非对称左右手性 fixture 与真实 Unity 单物体场景质量基线仍待补充；这不阻塞
  provider 调用链，但阻塞把 SAM3D 坐标状态标成最终验收。

### Phase 3：Hunyuan

- [x] 保留其独立 venv，provider 只负责生成 GLB/mesh。
- [x] 实现确定性 mesh-to-Gaussian、材质采样和 Hunyuan 坐标契约。
- [x] 已用同一 renderer 生成 RGB/depth，并完成 provider-output SAGS smoke；自洽的 Unity
  三视图夹具上 GIM 得到 3871/3935/4198 个匹配，联合 pose 得到 6421 个一致点，
  `hunyuan_pose/05_pose/pose.json` 为 `status=ready`（scale=0.999473276）。真实项目
  场景的质量基线仍作为后续模型对比工作。

### Phase 4：切换和生产化

- [x] 在 `run_insert_batch.py` jobs/defaults 增加 `model_provider` 和 profile；未知
  provider 会在 preflight/profile 阶段拒绝，不会静默回退到 TRELLIS。
- [x] 为 provider、转换、渲染、GIM、pose、SAGS 阶段写 provider-neutral manifest、
  debug bundle 和状态码。
- [x] 缓存 key 使用输入 hash + mask hash + provider + model revision + weights
  fingerprint + profile + seed；模型切换不能复用错误的 PLY。
- [x] 加入显存串行门禁、失败阶段日志和 `--input-ply` 兼容；续跑清理策略仍沿用现有
  task ownership 规则。

## 9. 验证和验收门槛

### 环境

```bash
third_party/TRELLIS/.venv/bin/python -m py_compile model_center/**/*.py
third_party/TRELLIS/.venv/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda)'
git diff --check
```

SAM3D 接入前后上述 torch/CUDA 输出必须完全一致；不得覆盖 TRELLIS 已编译扩展。

### Provider 合成测试

- [x] registry 选择、未知 provider 拒绝、profile 合并和 candidate id 稳定；
- [x] mask 尺寸/hash/来源记录，多实例并集和空 mask 失败；
- [x] 轴矩阵、四元数、尺度同时变换，非对称 fixture 的 round-trip 误差小于阈值；
- [x] PLY 字段有限值、点数大于零、旋转单位长度、opacity/scale 合法；
- [x] Hunyuan mesh-to-Gaussian 在无 GPU 的 fixture 上可重复；当前 `model_center`
  测试共 8 项通过，并覆盖 SAM3D 显存参数透传及 options 参与 cache key。

### 真实 smoke

验收结果（服务器 `runs/model_center_smoke_20260821/`）：

- TRELLIS：真实 `microsoft/TRELLIS-image-large` 生成完成，12+12 steps 产出
  126,208 点标准 Gaussian PLY；canonical left/center/right RGB/depth 非空。
- TRELLIS downstream：三视图 GIM 得到 3575/3529/3549 个匹配，联合 pose 得到 7403
  个一致点，`trellis_pose/05_pose/pose.json` 为 `status=ready`（scale=0.985770561）。
- Hunyuan：真实 shape pipeline 产出 GLB，确定性转换为 12,000 点 `surface_splats`，
  同一 renderer 产出三视图 RGB/depth，`06_sags/inserted_object.ply` 已生成；GIM
  3871/3935/4198、pose 6421 inliers 且 `status=ready`。其 manifest 明确记录 Hunyuan
  mesh -> Gaussian 的 representation、revision 和转换参数。
- SAM3D：`sam3d_full_quality_24g/` 在 23.6 GiB RTX 3090 上以标准 32-Gaussian decoder、
  `downsample_ss_dist=1`、25+25 steps 真实生成 189,440 点 PLY，并输出非空、未裁切的
  left/center/right RGB 和 radial-depth；pipeline status 为 `ready`。分阶段峰值、voxel
  数和输出点数保存在 `02_trellis/sam3d_memory_report.json`。
- SAM3D downstream：`sam3d_pose/` 的自洽三视图 GIM 得到 4265/3974/4285 个匹配，
  旧 ModelScope 风格的点级跨视角一致性筛选后，用 8161 点执行一次联合相似变换；
  `05_pose/pose.json` 为 `status=ready`（scale=0.981386638，三视图 pose inlier ratio
  均为 1.0）。这是接口/坐标/深度 smoke，不替代真实 Unity 场景精度基线。
- SAM3D 显存对比：四组原始报告和最终 capability 快照归档在
  `runs/model_center_smoke_20260821/sam3d_memory_probe/`，不依赖易清理的 `/tmp`。

早期门禁材料仍保存在 `runs/model_center_smoke_20260821/sam3d_preflight/`，仅作为定位
历史，不能代表当前 capability。当前环境报告为 `available=true`，记录 tested minimum
23.5 GiB 和 upstream recommended 32 GiB；ModelScope revision/fingerprint、完整权重
清单和 `tokenRecorded=false` 均保留。正式任务仍需按自己的输入图片和人工/自动 mask
产生独立 debug bundle。

## 10. 风险和暂不做事项

- SAM3D 上游按 32GB 给出支持口径；本适配层只对当前 23.6 GiB RTX 3090 + 固定软件栈
  做过真实验证。低于 23.5 GiB 继续由 capability check 拒绝；遇到高 voxel 输入时必须
  依据分阶段报告判断，不能把单个 5,920-voxel 样本外推为所有输入都不会 OOM。
- `SPCONV_ALGO=native` 是当前 RTX 3090/CUDA 12.1 的稳定路径；升级 spconv、cumm、GPU
  架构或 CUDA 后需要重新比较 native/auto，不能永久假设上游 auto 都有同一故障。
- `gaussian_4` 和 `downsample_ss_dist>1` 都可能降低表示密度或几何质量，只允许显式选择，
  不做隐藏式 OOM 自动回退。
- SAM3D 依赖的 Pytorch3D/gsplat 可能触发编译或版本冲突；“不改变 TRELLIS 核心环境”
  高于快速安装成功。
- Hunyuan mesh 转 Gaussian 的视觉质量与原生 Gaussian 不等价，首版必须在 manifest 和
  debug bundle 标注 `representation=surface_splats`。
- 不在本任务中改 Unity 导入器、SAGS 算法或 pose 数学；它们只消费标准化产物。
- 不在第三方目录提交源码修改、锁文件或权重；若必须打补丁，放在
  `model_center/patches/` 并记录上游 commit 和补丁原因。

## 11. 当前状态和需要用户协助的事项

本方案已记录 ModelScope 作为 SAM3D 首选权重来源及现有 access token 的复用规则。本轮
已执行 provider 适配层、profile/分割管理、坐标/渲染契约、三种 provider 的真实 smoke、
源码 checkout、TRELLIS venv 的最小依赖接入、ModelScope 权重和 MoGe depth 权重物化：

- TRELLIS 核心版本前后保持 `torch==2.5.1+cu121`、`torchvision==0.20.1+cu121`、
  Python `3.11.13`；没有执行任何 torch/CUDA 升降级。
- 已接入 `sam3d_objects` editable、MoGe commit
  `a8c37341bc0325ca99b9d57981cc3bb2bd3e255b`、PyTorch3D commit
  `75ebeeaea0908c5527e7b1e305fbc7681382db47`、gsplat commit
  `2323de5905d5e90e035f792fe65bad0fedd413e7`，并补齐 Lightning 运行时依赖；官方
  `notebook/inference.py` 已成功导入。
- 服务器当前单卡为 24GB RTX 3090；capability report 测得 23.5685 GiB，当前状态为
  `available=true`，同时明确展示 tested minimum 23.5 GiB 与 upstream recommended
  32 GiB。标准 decoder 在该卡完成真实生成，峰值约 13GB；原始 auto-spconv 失败是
  stage 2 的 native `SIGFPE`，不是 decoder 或 CUDA OOM。
- 适配层默认只加载所需标准 Gaussian decoder、阶段结束释放已完成模块、使用 native
  spconv；不改变采样精度或输出密度。`gaussian_4` 实测不降低峰值，downsample=2/4 在
  5,920-voxel 样本上不删点，因此都不作为默认显存方案。
- ModelScope SDK 使用服务器已有凭据目录，token 没有写入命令行、manifest、日志或
  本文档。
- 统一适配层位于 `codex_remote_tools/model_center/`，没有向 TRELLIS、SAM3D 或
  Hunyuan 第三方源码目录写入适配实现；第三方源码、模型缓存和权重清单保持可替换。
- Hunyuan 使用现有独立 Python 3.12 / torch 2.7.1+cu118 环境，TRELLIS/SAM3D 继续使用
  Python 3.11 / torch 2.5.1+cu121，未混装两套核心依赖。

1. 提供一个严格非对称、带人工 mask 的测试图片以及一个真实 Unity 单物体任务，用于
   完成 SAM3D 左右手性、Unity 外参重渲染和真实质量基线；当前调用链已可运行，但坐标
   manifest 仍明确标记 axis pending。
2. 确认项目允许使用 SAM License 和 Hunyuan Community License（尤其 Hunyuan 的
   地域限制、服务披露和商业规模条款）。
3. 确认 Hunyuan 是否只要求 shape，还是同时启用 paint/PBR；本轮已完成 shape-only，
   paint/PBR 仍未默认启用，这会直接影响显存和耗时。

## 12. 官方资料

- [Microsoft TRELLIS README](https://github.com/microsoft/TRELLIS)
- [Meta SAM 3D Objects README](https://github.com/facebookresearch/sam-3d-objects)
- [SAM 3D setup](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md)
- [SAM 3D Objects on ModelScope](https://www.modelscope.cn/models/facebook/sam-3d-objects)
- [Tencent Hunyuan3D-2 README](https://github.com/Tencent/Hunyuan3D-2)
- [Mesh2Splat reference implementation](https://github.com/electronicarts/mesh2splat)
- [SAM License](https://github.com/facebookresearch/sam-3d-objects/blob/main/LICENSE)
- [Hunyuan 3D Community License](https://github.com/Tencent/Hunyuan3D-2/blob/main/LICENSE)
