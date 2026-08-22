# InsertAny3D

InsertAny3D 是一个“Unity 场景 + 生成模型 + 多视角几何对齐”的物体插入流水线。
它接收 Unity 场景的多视角渲染结果和一张图片编辑结果，在服务器上生成可导入
Unity 的 Gaussian Splatting 物体，并估计该物体在 Unity 世界坐标中的位置、旋转和缩放。

仓库保存的是可复现的主项目代码、环境配置、第三方源码版本记录和必要补丁；模型权重、
Python 虚拟环境、Unity 工程、场景数据和运行结果不放入 Git。

## 工作方式

一次完整任务由 Unity 和服务器两部分协作完成：

```text
Unity 场景
  ├─ 渲染原场景 left / center / right（RGB、深度、相机参数）
  ├─ 保存任务 manifest 和提示词
  └─ 接收服务器返回的 inserted_object.ply 与 pose.json
              │
              ▼
服务器 InsertAny3D
  ├─ 图片编辑结果检查或生成
  ├─ 3D provider 生成组合物体
  ├─ Gaussian 渲染、GIM 匹配和多视角 pose 估计
  ├─ SAGS 从组合物体中提取新物体
  └─ 输出 PLY、pose、manifest 和诊断结果
              │
              ▼
Unity 导入新物体，应用 pose，渲染 original / inserted 评估视图
```

Unity 负责场景、相机、任务管理、图片和深度渲染，以及最终 Gaussian 资产导入和
位姿应用。本仓库只提供服务器端代码和文件协议，不包含 Unity 脚本源码、Unity
编辑器安装方式或 Unity 工程本身。Unity 侧只需要按本文约定提供输入，并消费服务器输出。

## Unity 与服务器流程

### 1. Unity 准备任务

在 Unity 场景中为每个插入位置建立稳定的任务 ID，例如 `Task_001`。任务应记录：

- 插入参考点和用于渲染的相机参数；
- 原场景的 `left`、`center`、`right` 三视图；
- 每张图对应的 float32 深度和相机参数；
- 任务描述、编辑提示词和锚点提示词。

Unity 将这些内容写入一个任务目录。服务端最少需要：

```text
<task>/
├── step1/
│   ├── left/image.png       image.raw       image.camera.json
│   ├── center/image.png     image.raw       image.camera.json
│   └── right/image.png      image.raw       image.camera.json
└── task_manifest.json
```

如果已经在 Unity 或其他图片编辑工具中得到编辑后的中心图，还应放在：

```text
<task>/edited/center.png
```

三组 `image.png`、`image.raw`、`image.camera.json` 必须保持同一视角和同一顺序。
深度和相机数据是 pose 估计的输入，不能只上传 RGB。

### 2. 服务器处理任务

服务器入口是：

```bash
third_party/TRELLIS/.venv/bin/python tools/run_insert_pipeline.py \
  --run-root <run-root> \
  --task-id Task_001 \
  --input-image <task>/edited/center.png \
  --task-prompt '在场景中插入一个红色邮箱' \
  --scene-image <task>/step1/left/image.png \
  --scene-image <task>/step1/center/image.png \
  --scene-image <task>/step1/right/image.png \
  --scene-depth <task>/step1/left/image.raw \
  --scene-depth <task>/step1/center/image.raw \
  --scene-depth <task>/step1/right/image.raw \
  --scene-camera <task>/step1/left/image.camera.json \
  --scene-camera <task>/step1/center/image.camera.json \
  --scene-camera <task>/step1/right/image.camera.json \
  --run-sags
```

批量任务使用 `tools/run_insert_batch.py`，每个任务拥有独立目录和日志：

```bash
third_party/TRELLIS/.venv/bin/python tools/run_insert_batch.py \
  --jobs <scene>/insert_jobs.json \
  --skip-ready
```

服务端阶段如下：

1. **输入和提示词**：读取 Unity manifest、编辑图和任务参数。
2. **3D provider**：默认使用 TRELLIS；也支持 SAM3D Objects 和 Hunyuan3D。
3. **Gaussian 渲染**：生成统一的 RGB、深度、COLMAP 相机和 Gaussian 模型目录。
4. **GIM 匹配**：将生成视图与 Unity 原场景视图匹配。
5. **Pose 估计**：深度反投影后计算 generated world 到 Unity world 的相似变换。
6. **SAGS 提取**：从“锚点 + 新物体”的组合 Gaussian 中提取新物体。
7. **证据记录**：每个阶段写入 manifest、日志和诊断文件。

### 3. Unity 回收结果

服务器任务目录的关键输出是：

```text
<run-root>/<task-id>/
├── 05_pose/pose.json
└── 06_sags/inserted_object.ply
```

Unity 只应导入 `06_sags/inserted_object.ply`。`02_trellis/sample.ply` 仍包含锚点，
直接导入会造成原场景物体重复。

Unity 导入 PLY 后，将 `05_pose/pose.json` 中的 position、xyzw rotation 和统一 scale
应用到生成物，再渲染原场景和插入后场景进行目视检查或基准评测。

## 服务器输出目录

```text
<run-root>/<task-id>/
├── 01_segmentation/          初始 mask、points 和分割诊断
├── 02_trellis/               provider 输入、sample.ply、生成 manifest
├── 03_rendered_3dgs/         RGB、深度、相机和 Gaussian model
├── 03_sags_views/             SAGS 使用的多视角输入（启用 ring6 时）
├── 04_gim/                   匹配图、matches.json 和几何诊断
├── 05_pose/pose.json          Unity 世界坐标位姿
├── 06_sags/inserted_object.ply 最终回传 Unity 的新物体
├── logs/                     阶段日志
└── manifest.json             参数、权重和阶段状态
```

失败任务也会保留 manifest 和日志，便于定位失败阶段。不要让多个任务共用同一个
`--output-dir`；批处理优先使用 `--run-root` 与 `--task-id`。

## 支持的 provider

| provider | 输出 | 运行环境 | 说明 |
| --- | --- | --- | --- |
| `trellis` | Gaussian PLY | `third_party/TRELLIS/.venv` | 默认路线，先生成组合物体再由 SAGS 提取 |
| `sam3d` | Gaussian PLY | `third_party/TRELLIS/.venv` | 需要单物体输入 mask |
| `hunyuan` | mesh，再转换为 Gaussian PLY | `third_party/Hunyuan3D-2/.venv` | 使用独立 CUDA/PyTorch 环境 |

provider、版本、权重 revision、坐标契约和转换信息都会写入任务 manifest。未知
provider 会在 preflight 阶段失败，不会静默回退到 TRELLIS。

## 安装和验证

从 GitHub 获取源码：

```bash
git clone https://github.com/Junnan-bjtu/InsertAny3D.git
cd InsertAny3D
```

只运行而不需要 Git 时，也可以使用 ModelScope 快照：

```bash
modelscope download --model KHUU0424/InsertAny3D_v1
```

完整安装顺序见 [`install.md`](install.md)：

```bash
bash tools/bootstrap_third_party.sh
bash tools/install_environments.sh all
bash tools/download_models.sh
CUDA_VISIBLE_DEVICES=0 bash tools/verify_environments.sh
```

成功标志为：

```text
INSERTANY3D_THREE_ENVIRONMENTS_READY
```

不需要模型推理的基础自测：

```bash
third_party/TRELLIS/.venv/bin/python tools/test_estimate_similarity_pose.py
third_party/TRELLIS/.venv/bin/python tools/test_insert_batch.py
PYTHONPATH=tools third_party/TRELLIS/.venv/bin/python tools/model_center/tests/test_model_center.py
```

第三方源码不会直接提交到主仓库。固定上游 commit、补丁和新增文件的重建方式见
[`code/third_party/THIRD_PARTY_REPOS.md`](code/third_party/THIRD_PARTY_REPOS.md)。
HPSv2 等外部指标仓库的处理方式见 [`metrics/HPSV2_SOURCE.md`](metrics/HPSV2_SOURCE.md)。

## 常用服务器工具

| 工具 | 作用 |
| --- | --- |
| `tools/run_insert_pipeline.py` | 单任务完整编排 |
| `tools/run_insert_batch.py` | 从 jobs JSON 串行运行任务 |
| `tools/auto_segment.py` | 生成分割 mask 和 SAGS points |
| `tools/segment_anchor_views.py` | 生成多视角锚点 mask |
| `tools/render_trellis_views.py` | 按 yaw/pitch 生成定向视图 |
| `tools/run_gim_match.py` | 两张图片的 GIM 匹配 |
| `tools/estimate_similarity_pose.py` | 多视角相似变换估计 |
| `tools/run_sags_text.py` | 无 UI 执行 SAGS 分割 |
| `metrics/run_metric.py` | 指标统一入口，目前支持 HPSv2 |

详细文件协议和阶段参数见 [`codex_ops/WORKFLOW.md`](codex_ops/WORKFLOW.md)，
Unity 侧完整任务说明见 [`codex_ops/UNITY_INSERT_WORKFLOW_TUTORIAL_ZH.md`](codex_ops/UNITY_INSERT_WORKFLOW_TUTORIAL_ZH.md)。

## 复现边界

- Unity 工程和 Unity 脚本不在本仓库中，需要由调用方提供。
- 模型权重和缓存由安装脚本下载，不进入 Git。
- 第三方源码由固定 commit、patch 和 overlay 重建。
- 输入图片、深度、相机文件和运行结果属于任务数据，不进入 Git。
- 完整端到端结果仍受 GPU、CUDA、上游模型服务和输入数据质量影响。

## 致谢

本项目使用或参考了 [TRELLIS](https://github.com/microsoft/TRELLIS)、
[GIM](https://github.com/xuelunshen/gim)、
[SAGS](https://github.com/XuHu0529/SAGS)、
[Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) 和相关开源项目。
