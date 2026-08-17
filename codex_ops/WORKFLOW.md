# InsertAny3D 批处理工作流

更新时间：2026-08-16

本文记录服务器端图片编辑、分割、TRELLIS、3DGS 渲染、GIM 和 SAGS
适配器，以及本地 Unity 的步骤 1、5、6 如何组织输出。当前服务器项目根目录为：

```text
/opt/data/private/ljn/InsertAny3D
```

## 1. 任务目录约定

是的，渲染结果按任务隔离保存。Unity 批处理器接收一个输出根目录，
每个 `InsertTask.taskId` 自动成为一级目录；建议每个场景再使用一个独立的
输出根目录，避免不同场景出现同名任务：

```text
<render-root>/<scene-id>/<task-id>/
├── step1/
│   ├── left/       image.png  image.raw  image.txt  image.camera.json
│   ├── center/     image.png  image.raw  image.txt  image.camera.json
│   └── right/      image.png  image.raw  image.txt  image.camera.json
├── task_manifest.json
└── step6/
    ├── original/
    │   ├── pitch_10/view_000/ ... view_029/
    │   ├── pitch_20/view_000/ ... view_029/
    │   ├── pitch_30/view_000/ ... view_029/
    │   └── pitch_40/view_000/ ... view_029/
    └── inserted/
        └── 同样的四组俯视角和 30 个横向视角
```

输出根目录另有 `run_manifest.json`。每个 `view_NNN` 目录由 `RendererCore`
写入 RGB、float32 径向深度、旧兼容文本和结构化相机 JSON。默认
配置为 4 个俯视角、每个 30 个横向视角，因此每个任务步骤 6 有
`4 * 30 * 2 = 240` 个视角目录。任务之间不会共享 `Generated` 物体：渲染
某个任务时，其他任务的生成物会被禁用，锚点标记也会被禁用。

服务器端可用 `--run-root <server-root>/<scene-id> --task-id <task-id>` 自动
建立同名任务目录；原有 `--output-dir` 仍兼容：

```text
<server-root>/<scene-id>/<task-id>/
├── 01_segmentation/
├── 02_trellis/
├── 03_rendered_3dgs/
│   ├── source/images/       RGB 渲染（sphere 或 left/center/right）
│   ├── source/depths/       invdepth PNG 和 absdepth RAW
│   └── model/               3DGS 模型、相机和 cfg_args
├── 04_gim/pair_00/          match.png、warp.png、matches.json
├── 05_pose/pose.json         generated world -> Unity world 的 position/rotation/scale
├── 06_sags/inserted_object.ply  从组合 Gaussian 中提取的新物体
├── logs/                    每个阶段的完整 stdout/stderr
└── manifest.json             阶段状态、渲染/pose 参数和输入输出路径
```

不要让多个任务共用同一个 `--output-dir`。批量运行优先使用 `--run-root` 和
`--task-id`；脚本会拒绝包含目录分隔符或路径穿越的任务 ID。

## 2. 本地 Unity（步骤 1、5、6）

在 Unity 项目 `MyProjects/Farm` 中，每个插入任务是一个带 `InsertTask`
组件的对象，建议使用 `Task_001` 到 `Task_005` 这样的稳定 ID。通过
`工具 > 插入流程 > 任务管理`从选中锚点建立模板，并填写插入位置、
相机参数。prompt 分为默认约束和任务补充两层；manifest 同时保存两层文本及其
合并后的 effective prompt。首批建议用 `legacy`，由人检查/修正生成物 mask。
窗口中的`三视图预览 > 刷新预览`使用正式渲染路径快速显示左、中、右视角，
最长边限制为 512 像素；正常宽度横向铺满，仅在约 360 像素以下改为纵排。
全部数值标签仍可左右拖动，低频参数收在`高级设置`中。

批量入口可重复指定 task；脚本会按参数顺序串行启动独立 Unity 进程：

```bat
cd MyProjects\Farm
python Tools\render_insert_workflow.py ^
  --stage anchor ^
  --scene Assets/GSTestScene.unity ^
  --output F:\InsertRuns\Farm_001 ^
  --task Task_001 --task Task_002 --task Task_003
```

省略 `--task` 时保留旧的一进程处理场景内全部任务模式。独立进程会增加 Unity
启动时间，但任务的日志、失败状态和图形状态互不影响。

服务端完成后，在“任务管理”选择
`<server-task>/06_sags/inserted_object.ply`，点击
`导入生成物（PLY）`。工具复用已安装的 Gaussian Splatting 1.1.1
插件，将 PLY 转为 Float32 `GaussianSplatAsset`，自动创建 renderer 并挂到
该 task 的 `Generated`。随后自动应用 pose 并执行步骤 6：

```bat
python Tools\render_insert_workflow.py ^
  --stage benchmark ^
  --scene Assets/GSTestScene.unity ^
  --output F:\InsertRuns\Farm_001 ^
  --pose-root F:\ServerRuns\Farm_001 ^
  --task Task_001 --task Task_002 --task Task_003
```

脚本按 `<pose-root>/<task-id>/05_pose/pose.json` 查找位姿，应用世界位置、xyzw
四元数和统一缩放。`--stage anchor` 只执行步骤 1，`--stage benchmark` 执行
步骤 5 的 pose 应用和步骤 6。默认使用 `-force-d3d12`，不要加
`-nographics`；Gaussian Splatting 需要图形设备。

步骤 6 的 original/inserted 两组都写 RGB 和 float32 径向深度。深度翻转已避免
循环内重复复制整张 NativeArray。`高级设置 > 使用相机灯`默认关闭；开启时才会
禁用场景灯并加入临时相机灯。也可在“任务管理”手动选择同一份 pose JSON。

## 3. APIYi Gemini 图片编辑

脚本不会把密钥写入命令行日志、manifest 或错误信息。推荐只在当前 shell
导出密钥：

```bash
export GEMINI_API_KEY='***'
export GEMINI_BASE_URL='https://api.apiyi.com/v1'
export GEMINI_IMAGE_URL='https://api.apiyi.com/v1beta/models/gemini-3.1-flash-image-preview:generateContent'
export GEMINI_MODEL='gemini-3.1-pro-preview'

third_party/TRELLIS/.venv/bin/python tools/gemini_edit.py \
  --input-image <task>/step1/center/image.png \
  --output-image <task>/edited/center.png \
  --prompt-file <task>/edit_prompt.txt \
  --response-json <task>/edited/response.json
```

不需要联网时可以先检查配置而不发送请求：

```bash
third_party/TRELLIS/.venv/bin/python tools/gemini_edit.py \
  --input-image input.png --output-image output.png \
  --prompt 'test' --dry-run
```

已用本地 HTTP mock 验证 task prompt + inline image 请求、APIYi `/v1` 到原生
`/v1beta/models/...:generateContent` 路由推导、Gemini `inlineData` 响应解析、
图片/响应落盘和密钥不出现在日志中，见 `codex_ops/gemini_mock.log`。没有使用
真实 key 请求 APIYi，因此不宣称外部服务可用性已验收。

## 4. 分割、TRELLIS、渲染和 GIM

服务器默认使用缓存目录，避免重复下载已有 Hugging Face 权重：

```bash
export HF_HOME=/opt/data/private/ljn/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0
```

### 生成视图分割

组合路线不会在 TRELLIS 前先分割编辑图。TRELLIS 先重建“锚点 + 新物体”，再在
它自己渲染的中心图上定位新物体，从而让 2D 点和 SAGS 的 3DGS 相机对齐：

```bash
third_party/TRELLIS/.venv/bin/python tools/auto_segment.py \
  --input <task>/03_rendered_3dgs/source/images/center.png \
  --output-dir <task>/01_segmentation \
  --prompt 'red mailbox' \
  --engine legacy
```

这里的 LangSAM/legacy 只生成 SAGS 的初始 2D mask 和点，不属于 TRELLIS，也
不替换 SAGS 内部用于多视角传播的 SAM ViT-H。`legacy` 固定使用已有
GroundingDINO + SAM；`langsam` 固定使用 LangSAM；`auto` 先 LangSAM、失败后
回退 legacy。开发期可用 `auto`，正式批次建议固定一个引擎，避免任务间混用。

输出包括
`mask.png`、`cutout.png`、`annotated.png`、`detections.json`、`points.json`
和 `manifest.json`。

首批允许人工修正。若在生成的 `center.png` 上由人工工具得到新的正点击点，
可跳过自动分割并直接驱动 SAGS：

```bash
third_party/TRELLIS/.venv/bin/python tools/run_insert_pipeline.py \
  --input-image <task>/edited/center.png \
  --input-ply <task>/02_trellis/sample.ply \
  --run-root <server-root>/<scene-id> --task-id Task_001 \
  --skip-segmentation --sags-points-json <task>/manual/points.json \
  --run-sags --render-mode anchor --render-resolution 1024
```

`points.json` 可以是 `[{"x": 512, "y": 480, "label": 1}]`，也可以是包含
`points` 数组的对象；坐标必须对应生成的 `center.png`，不是原 Unity 图片。

### TRELLIS 组合资产

```bash
third_party/TRELLIS/.venv/bin/python tools/generate_trellis_asset.py \
  --input-image <task>/edited/center.png \
  --output-dir <task>/02_trellis
```

输入是完整编辑图，输出的 `sample.ply` 包含锚点与新物体的组合。它用于生成
匹配视图和估计 pose；最终导入 Unity 的是 SAGS 从该组合中提取的
`06_sags/inserted_object.ply`。`sample.glb` 为可选产物，GLB 提取失败不会阻断
PLY 主链路，除非加 `--require-glb`。

### 3DGS 渲染

默认的环绕视图入口适合生成 NVS/3DGS 数据集：

```bash
third_party/TRELLIS/.venv/bin/python tools/render_trellis_3dgs.py \
  --input-ply <task>/02_trellis/sample.ply \
  --output-dir <task>/03_rendered_3dgs \
  --resolution 1024 --radius 1.5 \
  --latitudes 10,20,30,40 --views-per-latitude 30
```

这样会生成 120 张 RGB、逆深度 PNG、绝对深度 RAW、COLMAP 相机文件和
可供 SAGS 加载的 `model/`。环绕入口需要 40 度时显式传
`--render-latitudes 10,20,30,40`。插入主流程默认使用下述 anchor 三视图入口，
而不是 sphere 环绕入口。

要严格复用 Unity 的中心相机规则（`yaw`、`pitch`、`distance` 和
`-24/0/+24` 横向偏移），使用定向入口：

```bash
third_party/TRELLIS/.venv/bin/python tools/render_trellis_views.py \
  --input-ply <task>/02_trellis/sample.ply \
  --output-dir <task>/03_rendered_3dgs \
  --yaw-degrees 0 --pitch-degrees 12 --distance 1.5 \
  --side-angle-degrees 24 --resolution 1024
```

它写入 `source/images/left.png`、`center.png`、`right.png`，对应的深度、
COLMAP 相机和 `views.json`。TRELLIS 生成物采用归一化坐标，`distance` 是
生成物坐标系中的距离，不要求与 Unity distance 数值一致。两边应保持相同 FOV，
再分别选择能完整、清楚包含锚点和新物体的 distance。

### GIM

```bash
third_party/gim/.venv/bin/python tools/run_gim_match.py \
  --image0 <task>/step1/left/image.png \
  --image1 <task>/03_rendered_3dgs/source/images/left.png \
  --output-dir <task>/04_gim/pair_00 --model gim_roma
```

可重复传入 `--gim-pair IMAGE0 IMAGE1`，每一对写入 `pair_00`、`pair_01` 等，
原图像素坐标、几何内点、逐点 confidence、原图/预处理尺寸保存在
`matches.json`。这些坐标已还原到原图分辨率，可直接与深度图配合。

### 深度反投影和 pose

`RendererCore` 的 `image.raw` 是相机到表面的径向距离，TRELLIS 的
`absdepth/*.raw` 是相机 z-depth，不能用同一个反投影公式。新入口会分别
反投影，再把多个视角的 3D 对应点合并，以 RANSAC + 加权 Umeyama 求
generated world 到 Unity world 的相似变换：

```bash
third_party/gim/.venv/bin/python tools/estimate_similarity_pose.py \
  --view <task>/04_gim/pair_00/matches.json \
         <unity-task>/step1/left/image.raw \
         <unity-task>/step1/left/image.camera.json \
         <task>/03_rendered_3dgs/source/depths/absdepth/left.raw \
  --view <task>/04_gim/pair_02/matches.json \
         <unity-task>/step1/right/image.raw \
         <unity-task>/step1/right/image.camera.json \
         <task>/03_rendered_3dgs/source/depths/absdepth/right.raw \
  --generated-cameras <task>/03_rendered_3dgs/source/sparse/0/cameras.txt \
  --generated-images <task>/03_rendered_3dgs/source/sparse/0/images.txt \
  --output <task>/05_pose/pose.json
```

默认坐标约定为当前导出 `point_cloud.ply` 对应的 `identity`；若要复现旧
`my_pipeline.py` 的手工 z 翻转，可显式传 `--generated-axis legacy-flip-z`。
输出以四元数为主，同时包含 Euler、统一 scale、4x4 矩阵、每视角有效点和
残差统计。

### 单任务编排器

```bash
third_party/TRELLIS/.venv/bin/python tools/run_insert_pipeline.py \
  --input-image <task>/edited/center.png \
  --run-root <server-root>/<scene-id> --task-id Task_001 \
  --task-prompt '在墙边插入一个红色邮箱' \
  --scene-image <task>/step1/left/image.png \
  --scene-image <task>/step1/center/image.png \
  --scene-image <task>/step1/right/image.png \
  --scene-depth <task>/step1/left/image.raw \
  --scene-depth <task>/step1/center/image.raw \
  --scene-depth <task>/step1/right/image.raw \
  --scene-camera <task>/step1/left/image.camera.json \
  --scene-camera <task>/step1/center/image.camera.json \
  --scene-camera <task>/step1/right/image.camera.json \
  --trellis-input composite --seg-engine legacy --render-mode anchor \
  --render-resolution 1024 --render-fov 53.1301023542 \
  --render-yaw-degrees 0 --render-pitch-degrees 12 \
  --render-distance 1.5 --render-side-angle-degrees 24 \
  --run-sags
```

`composite`、`anchor` 和 1024 均为当前默认值。组合模式下 segmentation 阶段会
先标为 deferred：TRELLIS 和三视图渲染完成后，编排器才在生成的 center 图上
分割，再运行 SAGS。若任务描述无法被确定性词表提取为物体短语，改用
`--prompt '具体物体名'`，它与 `--task-prompt` 互斥。

当传入多张 `--scene-image` 时，顺序应为 left、center、right；若文件名中
包含这些词，编排器也会按文件名选择对应的生成视图。只传一张场景图时会
默认匹配生成的 `center.png`。如果要使用环绕模式，显式传
`--render-mode sphere` 和 `--render-latitudes`；组合路线的 SAGS 点对齐要求
anchor/center，因此不要同时传 `--run-sags`。

三组 `--scene-depth` 和 `--scene-camera` 必须与 `--scene-image` 同序。提供后
自动执行 pose 阶段；默认 GIM 三张都运行，但 pose 只联合 `left,right`，可用
`--pose-view-names all` 或其他逗号列表调整。没有深度/相机参数时维持旧行为，
停在 GIM；也可显式 `--skip-pose`。

已有组合 PLY 时可用 `--input-ply` 跳过 TRELLIS；已有阶段可用
`--skip-segmentation`、`--skip-render` 或 `--skip-gim`。每个阶段的成功标志
会写入对应日志和 `manifest.json`。

## 5. 场景级串行调度

单任务编排器一次只拥有一个 task 目录。需要批量运行时，用 JSON job 文件让
调度器逐个启动它；默认某个任务失败后继续后面的任务，`--fail-fast` 才会在
首个失败处停止：

```bash
third_party/TRELLIS/.venv/bin/python tools/run_insert_batch.py \
  --jobs <scene>/insert_jobs.json --skip-ready
```

最小 job 文件如下。`prompts` 的默认层和任务层会合并写入每个任务的
`prompts.json`；`object_effective` 会作为 `--prompt` 传给生成视图的
`legacy`/LangSAM 分割。`pipeline_args` 可覆盖不常用参数。

```json
{
  "run_root": "/runs/Farm_001",
  "defaults": {
    "trellis_input": "composite",
    "seg_engine": "legacy",
    "render_resolution": 1024,
    "render_mode": "anchor",
    "run_sags": true,
    "prompts": {
      "edit_default": "Keep the anchor unchanged.",
      "object_default": ""
    }
  },
  "tasks": [
    {
      "task_id": "Task_001",
      "input_image": "/runs/Farm_001/Task_001/edited/center.png",
      "unity_manifest": "/runs/Farm_001/Task_001/task_manifest.json",
      "object_prompt": "red mailbox",
      "scene_images": [".../left.png", ".../center.png", ".../right.png"],
      "scene_depths": [".../left/image.raw", ".../center/image.raw", ".../right/image.raw"],
      "scene_cameras": [".../left/image.camera.json", ".../center/image.camera.json", ".../right/image.camera.json"]
    }
  ]
}
```

调度器写 `<run-root>/batch_manifest.json`，每个任务写独立的
`logs/batch.log`、`prompts.json` 和服务端 `manifest.json`，因此可以用
`--skip-ready` 从中断处继续。它不改变单任务命令，也不并行占用 GPU。
`unity_manifest` 是可选的；提供后会自动读取“任务管理”写入的默认/任务
prompt，再由 job 中显式字段覆盖。

## 6. SAGS 文本适配器

```bash
third_party/TRELLIS/.venv/bin/python tools/run_sags_text.py \
  --model-dir <task>/03_rendered_3dgs/model \
  --points-json <task>/01_segmentation/points.json \
  --output-ply <task>/06_sags/inserted_object.ply \
  --view-name center \
  --gd-interval 20
```

`--diagnose-only` 可先查看每个视角的投票数量。当前默认
`--force-seed-radius 2`，会把 3D prompt 投影点回填到每视角 SAM mask，避免
原始 SAGS 在烟测样本中出现 SAM mask 非空但 Gaussian 投票为 0 的情况。
这是兼容策略，不等同于已完成的生产精度验证；要复现上游行为可加
`--no-force-seed`。

## 7. 已验证结果和兼容补丁

以下结果已在服务器 RTX 3090 上通过，日志均在 `codex_ops/`：

| 阶段 | 标志/结果 |
| --- | --- |
| legacy 分割回退 | `AUTO_SEGMENT_READY legacy 1`，`codex_ops/legacy_mailbox.log` |
| TRELLIS | `TRELLIS_ASSET_READY`，生成 `sample.ply`，`codex_ops/trellis_asset.log` |
| 3DGS 渲染 | `TRELLIS_3DGS_RENDER_READY 6`，`codex_ops/render_fixed.log` |
| 定向三视图渲染 | `TRELLIS_POSE_RENDER_READY 3`，`codex_ops/pose_render_smoke_v2.log` |
| GIM | `GIM_MATCH_READY 4999 3940`，`codex_ops/gim_wrapper.log` |
| 编排器 | `INSERT_PIPELINE_READY`，`reproduction_outputs/codex/orchestrated_fixed` |
| 三视图编排器 | 3 个 GIM pair 均成功，`codex_ops/anchor_pipeline_smoke.log` |
| SAGS（带 seed 回填） | `SAGS_TEXT_READY`，791 vertices，`codex_ops/sags_text_forced.log` |
| Gemini 请求/响应契约 | 本地 mock 成功且无密钥泄漏，`codex_ops/gemini_mock.log` |
| GIM 元数据 | 4831 matches/confidence/原图坐标，`codex_ops/gim_metadata_smoke.log` |
| 双视角 pose 数学与 I/O | 20/20 内点，scale 1.65，矩阵最大误差 8.32e-08，`codex_ops/final_pose_pipeline_synthetic.log` |
| task prompt -> SAGS | 中文任务改写为 `mailbox`，命名视角输出 2433 vertices，`codex_ops/task_prompt_segment.log`、`codex_ops/sags_pose_named.log` |
| Unity 任务分类 | step1 实际 3/3，`MyProjects/Farm/InsertTaskClassificationSmoke` |
| Unity pose -> step6 | pose 已应用，original/inserted 各 1 且图片不同，`MyProjects/Farm/InsertPoseApplicationSmoke` |
| Unity Gaussian 资产导入 | 255232 splats 的真实 PLY 转换成功并清理烟测资产，`MyProjects/Farm/gaussian-import-smoke.log` |
| Unity 最新脚本编译/渲染 | `CAMERA_RENDERER_CLI_OK`，`MyProjects/Farm/post-decision-compile.log` |
| 场景级串行调度器 | `INSERT_BATCH_TEST_READY`，本地与远端均通过 |
| SAGS Gradio 环境 | `GRADIO_READY 5.27.0 0.0.1`、`SAGS_APP_IMPORT_READY`、`SAGS_GRADIO_UI_READY Blocks` |

为兼容当前安装版本，已保留以下本地补丁：

- TRELLIS Gaussian renderer 强制 Python SH 转 RGB；否则 absdepth 扩展会产生
  全黑 RGB，但深度仍非零。
- Gaussian opacity 做有限值裁剪，并补充 `max_sh_degree` 兼容别名。
- SAGS rasterizer 适配新版 `kernel_size`/`subpixel_offset` 参数及 2/4 返回值
  ABI 差异。
- `run_sags_text.py` 注入无 UI 的 Gradio stub，并修正上游全局 `predictor`
  闭包问题。

## 8. 待定项和首轮标定

- LangSAM 的合法模型键已修正为 `sam2.1_hiera_small`，但权重尚未完成下载和
  端到端验收。首批固定 `legacy`；后续是否转为固定 `langsam`，应在相同任务上
  A/B 后决定。长期 `auto` 会让数据集混用不同分割引擎。
- SAGS venv 已安装 Gradio 5.27.0 和 `gradio_litmodel3d` 0.0.1；`app_text`
  导入及完整 `Blocks` 构建均已通过。`gradio_litmodel3d` 元数据声明 Gradio `<5`
  且 LangSAM 锁定 Gradio 5.0.2，因此 `pip check` 仍报告版本声明冲突；当前 legacy
  UI 实测可构建，CLI 主链路继续使用 headless 入口。启动 UI 时应设置
  `GRADIO_ANALYTICS_ENABLED=False`，避免服务器访问外部分析端点时等待超时。
- 原始 SAGS mask 点不一致时，seed 回填半径是否固定为 2 像素，标记为
  `待定`，生产批处理前应在目标类别上重新评估。
- 深度反投影、双视角 pose 和 Unity 自动应用已接通。按当前决定暂不设置自动
  拒绝门槛；真实首任务仍需人工确认`导入时翻转 Z 轴`开/关哪一个与
  pose 的 `generated-axis=identity` 约定一致。
- 组合图的新物体在原 Unity 侧不存在；当前按决定不额外处理这类匹配外点，先
  以人工结果为准。
- task prompt 改写目前是确定性中英文关键词规则，稳定且无额外 API 成本，但
  长尾类别需要显式 `--prompt` 或扩充词表；是否改用 LLM 结构化改写标记为
  `待定`。
- 三类 prompt 已进入 Unity manifest；锚点 prompt 如何参与服务端编辑/质量检查
  尚未定协议。当前模型入口只使用完整 task prompt 或明确的 object prompt。

## 9. tmux 与日志

正确的远端入口和项目目录是：

```bash
ssh -p 25367 root@10.126.56.69
cd /opt/data/private/ljn/InsertAny3D
```

手工 legacy 标注界面从 SAGS venv 启动；按串行调度约定，使用前先确认当前没有
正在运行的模型任务：

```bash
cd /opt/data/private/ljn/InsertAny3D/third_party/SAGS
GRADIO_ANALYTICS_ENABLED=False CUDA_VISIBLE_DEVICES=0 .venv/bin/python app_text.py
```

长任务使用已有会话 `codex_insert`：

```bash
tmux attach -t codex_insert
tmux capture-pane -pt codex_insert:0 -S -120
tail -f codex_ops/<stage>.log
```

任务完成后不要留下正在占用 GPU 的后台进程；日志和验收产物可以保留用于复核。
本次命令、标志和产物索引另见 `codex_ops/COMMANDS_20260816.md`。
