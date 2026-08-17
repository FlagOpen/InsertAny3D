# Unity 场景物体插入完整操作教程

这份教程用于完成一整轮 InsertAny3D 插入任务：在 Unity 场景中选择插入点，渲染左、中、右三张图，在服务器上完成图片编辑、组合物体重建、SAGS 提取、双视角匹配和位姿估计，再把生成物放回 Unity，最后渲染原场景与插入后场景的240 组评估视图。

教程可以独立使用，不要求先阅读项目中的其他说明文件。

## 1. 现在能不能开始

可以开始制作场景和建立插入任务。当前已经具备以下能力：

- Unity 中创建独立的 `InsertTask`，记录插入坐标、相机参数和提示词。
- 每个任务独立渲染 `left/center/right` 三个视角，以及 RGB、深度和相机参数。
- 服务器按 JSON 任务表串行运行 TRELLIS、legacy 分割、SAGS、GIM 和 pose。
- Unity 将 SAGS 结果转换成 `GaussianSplatAsset`，并应用服务器返回的 pose。
- 每个任务渲染 120 张原场景图和 120 张插入后场景图，二者都带深度。

但第一个真实任务应当视为“标定任务”，先只做一个任务，不要一开始就运行全部五个任务的最终 240 图。第一个任务必须人工确认：

- 插入物是否完整，是否误带了锚点物体。
- pose 的位置、旋转和缩放是否合理。
- `高级设置 > 导入时翻转 Z 轴` 应该开启还是关闭。
- TRELLIS 相机的 `render_distance` 是否能完整拍到组合物体。
- legacy 分割是否只选中了新物体。

首个任务正确后，再把相同约定用于该场景的其他任务。

## 2. 本教程使用的目录

下面用 `Farm_A_001` 作为场景或批次名称。实际使用时请换成自己的稳定名称，不要在同一批任务运行到一半时改名。

| 内容 | 示例位置 |
| --- | --- |
| Windows Unity 项目 | `F:\Shared\Unity_Projects\MyProjects\Farm` |
| Unity 本地渲染 | `F:\InsertRuns\Farm_A_001` |
| 下载回来的服务器结果 | `F:\ServerRuns\Farm_A_001` |
| 服务器登录 | `ssh -p 25367 root@10.126.56.69` |
| 服务器项目 | `/opt/data/private/ljn/InsertAny3D` |
| 服务器本批结果 | `/opt/data/private/ljn/InsertAny3D/runs/Farm_A_001` |

本文中标有“Windows PowerShell”的命令在本机 PowerShell 中执行；标有“远端Linux”的命令必须先 SSH 登录服务器后执行。

## 3. 制作和保存 Unity 场景

### 3.1 打开项目

用 Unity Hub 打开：

```text
F:\Shared\Unity_Projects\MyProjects\Farm
```

Unity 版本使用已经安装的 `2022.3.55f1c1`。等待右下角的资源导入和脚本编译结束，确认 Console 中没有红色编译错误。

Unity Hub 可以开着，也可以关闭。真正运行命令行渲染时，不能同时用 Unity Editor 打开同一个项目，否则 Unity 会报告项目已经被另一个进程占用。

### 3.2 新建场景

1. 在 Unity 中建立或打开原始场景。
2. 使用 `File > Save As`，把场景保存到 `Assets` 目录内，例如：

   ```text
   Assets/Scenes/Farm_A_001.unity
   ```

3. 完成环境模型、材质、灯光和原有 Gaussian Splat 的摆放。
4. 确认场景单位尽量统一。推荐把 1 个 Unity 单位理解为约 1 米。
5. 原场景物体不要放到 `__InsertAny3D/Tasks` 或某个任务的 `Generated` 下面。

`Generated` 只属于返回的新物体。渲染某个任务时，工具会自动隐藏其他任务的`Generated`，并在拍照时隐藏全部黄色锚点标记。

### 3.3 选择合适的锚点和插入区域

服务器会依靠左右视图中的共同内容估计位置，所以插入点附近应当有可匹配的原场景物体。好的锚点通常具有以下特点：

- 三个视角里都能看到。
- 表面有纹理、边缘或明显形状，不是一整块纯色平面。
- 不会被新物体完全遮住。
- 离插入位置较近，和新物体处于相近深度范围。
- 不是会移动、摆动或在不同渲染中变化的物体。

如果原物体的 Pivot 不在你想插入的位置，推荐先创建一个空物体：

1. 在 Hierarchy 中右键，选择 `Create Empty`。
2. 命名为 `InsertPoint_Task001`。
3. 用移动工具把它放到新物体应当落下的位置。
4. 选中这个空物体，再用“任务管理”创建任务。

“任务管理”会复制它的世界坐标作为 `AnchorMarker`。创建任务后，这个临时空物体可以保留，也可以删除；真正的任务坐标已经保存在 `AnchorMarker` 中。

## 4. 创建 InsertTask

### 4.1 打开任务窗口

在 Unity 顶部菜单选择：

```text
工具 > 插入流程 > 任务管理
```

选中刚才准备的锚点物体或空物体，点击：

```text
新建任务
```

Unity 会自动建立类似下面的结构：

```text
__InsertAny3D
└── Tasks
    └── Task_001
        ├── AnchorMarker
        └── Generated
```

每个插入位置建立一个任务。一个场景约五个任务时，建议使用：

```text
Task_001
Task_002
Task_003
Task_004
Task_005
```

`任务编号` 必须唯一。开始服务器处理后不要再改编号，因为本地与服务器都使用它作为目录名和结果对应关系。

### 4.2 设置插入坐标

有两种调整方法：

- 在“任务管理”中直接填写 `插入坐标` 的 X、Y、Z。
- 点击 `选中锚点`，然后在 Scene 视图中使用移动手柄调整黄色球体。

这里的坐标既是相机的 LookAt 点，也是任务的插入参考点。它不一定等于最终Gaussian 的中心；最终位姿由服务器匹配结果决定。

### 4.3 设置相机

初次使用建议保持以下默认值：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `水平角` | `0` | 中间相机绕 Unity Y 轴的水平角度 |
| `俯视角` | `12` | 相机俯视角；正值通常表示相机位于锚点上方 |
| `距离` | `3` | Unity 相机到锚点的距离 |
| `左右夹角` | `24` | 左右相机相对中间相机的水平偏移角 |
| `宽` | `1024` | 输出宽度 |
| `高` | `1024` | 输出高度 |
| `焦距（px）` | `1024` | 像素焦距，对应约 53.13 度垂直 FOV |

选择任务时，Scene 视图会显示三条从锚点连向相机位置的 Gizmo 线。白线是中间相机，青色线是左右相机。

调相机时的目标不是让锚点占满画面，而是同时满足：

- 中间图中有足够空间插入新物体。
- 左右图仍能看到锚点和相邻环境。
- 三张图之间有较大的共同可见区域。
- 新物体预期位置不要贴着画面边缘。
- 相机不要进入墙体、地面或已有模型内部。

`水平角`、`俯视角` 和 `距离` 直接显示在当前任务下；不常修改的其余参数收在`高级设置`中。所有数值都可以直接输入，也可以在参数文字上按住鼠标左键左右拖动。

高级选项：

- `遵循场景可见性` 默认关闭。关闭时，Unity Scene 视图中临时点掉的“眼睛”不一定影响最终渲染；要严格按 Scene Visibility 渲染时再打开。
- `使用相机灯` 默认关闭。关闭表示保留场景原灯光。打开后，拍照期间会临时禁用场景灯并使用相机灯，一般只用于原场景过暗的特殊情况。
- `导入时翻转 Z 轴` 只影响以后导入的生成物，不影响步骤 1 的三视图。首个任务先保留默认值，等返回 PLY 后做一次开关对比。

如果修改了分辨率或焦距，服务器端也必须使用同样的 FOV。计算公式是：

```text
垂直 FOV = 2 × atan(Height / (2 × FocalLength))
```

结果需要换算成角度。保持 `1024 × 1024` 和焦距 `1024` 时，不需要自行计算，服务器直接使用 `53.1301023542`。

### 4.4 填写提示词

任务窗口里有三组提示词。每组都有默认文本和任务补充文本，服务器记录的是二者拼接后的完整文本。

#### 编辑要求

这是给图片编辑模型的要求，应说明“添加什么、放在哪里、原图哪些内容不能改”。例如：

```text
在木箱右侧的地面上添加一个红色金属邮箱。邮箱竖直放置，高度约为木箱的一半。
不要移动、删除或改变木箱、地面和背景。
```

#### 物体名称

这是 legacy 分割寻找新物体时使用的短语。只写物体本身，不写位置关系和长句。Grounding-DINO 通常更适合简短、具体的英文名词，例如：

```text
red mailbox
```

不要写成：

```text
the new object next to the wooden box that should be extracted
```

这个字段不能为空，否则自动流程不知道应当从组合 Gaussian 中提取哪一个物体。

#### 锚点说明

这是锚点的补充说明，例如：

```text
the wooden crate beside the insertion point
```

当前它主要被记录在 manifest 中，便于人工检查和后续扩展，不直接驱动当前的SAGS 提取。

`默认提示词` 是每个任务各自保存的一份默认约束。一般不要删除已有默认约束，只在上方`任务提示词`中填写本任务内容。

### 4.5 保存场景

所有任务建立完成后按 `Ctrl+S`。确认 Hierarchy 中约有五个独立任务，每个任务都有自己的 `AnchorMarker` 和空的 `Generated`。

## 5. 渲染步骤 1 的左、中、右三视图

### 5.1 在窗口中预览三视图

在任务窗口的`三视图预览`区域点击`刷新预览`，会显示左、中、右三个实际渲染视角。预览与正式输出使用同一套 RendererCore、相机 FOV 和灯光设置，只把最长边限制为 512 像素以提高交互速度。

默认预览原场景并隐藏所有任务的生成物，这与步骤 1 一致。导入 PLY 后需要检查插入效果时，可以勾选`显示生成物`再刷新。修改坐标或相机参数后状态会变为`待刷新`，再次点击按钮即可更新。

正常宽度下三张图横向铺满预览区，外侧不留额外边距；只有窗口窄于约 360 像素时才改为纵向铺满。窗口不提供横向滚动，因此缩放时左边距不会漂移。

窗口预览用于快速检查构图；命令行试拍仍是最终依据，因为它会输出完整的 1024 × 1024 RGB、深度和相机参数文件。建议先把一个任务的预览调好，再正式渲染该任务。

运行命令前：

1. 在 Unity 中保存场景。
2. 关闭 Unity Editor。Unity Hub 不必关闭。
3. 打开 Windows PowerShell。

### 5.2 单任务试拍

在 Windows PowerShell 中执行：

```powershell
Set-Location "F:\Shared\Unity_Projects\MyProjects\Farm"

python .\Tools\render_insert_workflow.py `
  --stage anchor `
  --scene Assets/Scenes/Farm_A_001.unity `
  --output F:\InsertRuns\Farm_A_001 `
  --task Task_001
```

脚本会自己启动 Unity 2022.3.55f1c1，不需要提前打开 Unity Editor。它默认使用D3D12，并且没有使用 `-nographics`，因此 Gaussian Splatting 可以使用 GPU。

如果 Unity 安装位置不是脚本默认位置，可显式指定：

```powershell
python .\Tools\render_insert_workflow.py `
  --unity "D:\YourUnityPath\Editor\Unity.exe" `
  --stage anchor `
  --scene Assets/Scenes/Farm_A_001.unity `
  --output F:\InsertRuns\Farm_A_001 `
  --task Task_001
```

### 5.3 检查输出

成功后会得到：

```text
F:\InsertRuns\Farm_A_001\
├── logs\insert-workflow.Task_001.log
├── run_manifest.Task_001.json
├── run_manifest.json
└── Task_001\
    ├── task_manifest.json
    └── step1\
        ├── left\
        │   ├── image.png
        │   ├── image.raw
        │   ├── image.txt
        │   └── image.camera.json
        ├── center\...
        └── right\...
```

逐张打开三个 `image.png`。合格标准是：

- `center` 的插入区域清楚，周围有足够空间容纳新物体。
- `left`、`right` 和 `center` 都能看到同一个锚点。
- 左右图不是大面积被墙、模型或透明物遮挡。
- 三张图亮度和材质正常，没有全黑、全透明或 Gaussian 缺失。
- 画面没有黄色 AnchorMarker；它只用于编辑器定位，渲染时会自动隐藏。

每个 1024 × 1024 的 `image.raw` 应为 `4,194,304` 字节，也就是每像素一个float32 径向距离。不要把它当作 PNG 打开，也不要对它做图片缩放。

如果构图不合格，重新打开 Unity，修改该任务的`水平角`、`俯视角`或`距离`，刷新预览，保存并关闭 Unity，再执行同一条命令。相同输出目录会更新这个任务的确定性目录。

### 5.4 渲染该场景的全部任务

单任务构图确认后，串行渲染其余任务：

```powershell
Set-Location "F:\Shared\Unity_Projects\MyProjects\Farm"

python .\Tools\render_insert_workflow.py `
  --stage anchor `
  --scene Assets/Scenes/Farm_A_001.unity `
  --output F:\InsertRuns\Farm_A_001 `
  --task Task_001 `
  --task Task_002 `
  --task Task_003 `
  --task Task_004 `
  --task Task_005
```

这里会为每个 task 单独启动一次 Unity，并按照参数顺序串行运行。速度比一个Unity 进程处理全部任务慢一些，但某个任务失败时不会污染其他任务的图形状态和
日志。

## 6. 把 Unity 输入传到服务器

### 6.1 建立远端批次目录

Windows PowerShell：

```powershell
ssh -p 25367 root@10.126.56.69
```

登录后，在远端 Linux 执行：

```bash
cd /opt/data/private/ljn/InsertAny3D
mkdir -p runs/Farm_A_001
```

输入 `exit` 返回本机 PowerShell。

### 6.2 上传任务目录

Windows PowerShell：

```powershell
Set-Location "F:\InsertRuns\Farm_A_001"

scp -P 25367 -r `
  .\Task_001 .\Task_002 .\Task_003 .\Task_004 .\Task_005 `
  root@10.126.56.69:/opt/data/private/ljn/InsertAny3D/runs/Farm_A_001/
```

上传后，远端每个任务应当同时具有 `step1` 和 `task_manifest.json`。

## 7. 编辑中间图片

每个任务只编辑：

```text
Task_001/step1/center/image.png
```

把“任务管理”中完整的`编辑要求`一起提交给现有图片编辑服务。编辑结果
统一保存为：

```text
/opt/data/private/ljn/InsertAny3D/runs/Farm_A_001/Task_001/edited/center.png
```

每个任务都使用相同的目录结构。

编辑后的图片必须满足：

- 相机视角、裁剪范围和透视关系不变。
- 锚点、地面、墙面和背景尽量逐像素保持不变。
- 只添加要求的新物体，不删除或替换原物体。
- 新物体完整可见，不要紧贴画面边缘。
- 尺寸、接触位置、阴影和光照合理。
- 最终文件仍建议保存为 1024 × 1024 PNG。

如果编辑模型内部只能接收较低分辨率，可以在送入模型前临时下采样；服务输出仍应尽量恢复到约定尺寸，并人工确认没有明显模糊或整体几何漂移。

若编辑结果在本机生成，可这样上传：

```powershell
ssh -p 25367 root@10.126.56.69 `
  "mkdir -p /opt/data/private/ljn/InsertAny3D/runs/Farm_A_001/Task_001/edited"

scp -P 25367 `
  "F:\Edited\Farm_A_001\Task_001\center.png" `
  root@10.126.56.69:/opt/data/private/ljn/InsertAny3D/runs/Farm_A_001/Task_001/edited/center.png
```

对 Task_002 到 Task_005 重复执行。

项目也提供 `tools/gemini_edit.py`，但只有在服务器已经安全配置`GEMINI_API_KEY` 和对应接口时才使用。不要把 API key 写入 job JSON、教程文件或命令日志。

## 8. 编写服务器批任务 JSON

在本机文本编辑器中新建 `insert_jobs.Farm_A_001.json`。下面是一个完整的一任务示例；五任务时复制 `tasks` 中的对象，并替换任务编号、路径、相机角度和物体名。

```json
{
  "run_root": "/opt/data/private/ljn/InsertAny3D/runs/Farm_A_001",
  "defaults": {
    "trellis_input": "composite",
    "seg_engine": "legacy",
    "render_resolution": 1024,
    "render_fov": 53.1301023542,
    "render_mode": "anchor",
    "run_sags": true,
    "pose_view_names": "left,right"
  },
  "tasks": [
    {
      "task_id": "Task_001",
      "input_image": "/opt/data/private/ljn/InsertAny3D/runs/Farm_A_001/Task_001/edited/center.png",
      "unity_manifest": "/opt/data/private/ljn/InsertAny3D/runs/Farm_A_001/Task_001/task_manifest.json",
      "object_prompt": "red mailbox",
      "render_yaw_degrees": 0,
      "render_pitch_degrees": 12,
      "render_distance": 1.5,
      "render_side_angle_degrees": 24,
      "scene_images": [
        "/opt/data/private/ljn/InsertAny3D/runs/Farm_A_001/Task_001/step1/left/image.png",
        "/opt/data/private/ljn/InsertAny3D/runs/Farm_A_001/Task_001/step1/center/image.png",
        "/opt/data/private/ljn/InsertAny3D/runs/Farm_A_001/Task_001/step1/right/image.png"
      ],
      "scene_depths": [
        "/opt/data/private/ljn/InsertAny3D/runs/Farm_A_001/Task_001/step1/left/image.raw",
        "/opt/data/private/ljn/InsertAny3D/runs/Farm_A_001/Task_001/step1/center/image.raw",
        "/opt/data/private/ljn/InsertAny3D/runs/Farm_A_001/Task_001/step1/right/image.raw"
      ],
      "scene_cameras": [
        "/opt/data/private/ljn/InsertAny3D/runs/Farm_A_001/Task_001/step1/left/image.camera.json",
        "/opt/data/private/ljn/InsertAny3D/runs/Farm_A_001/Task_001/step1/center/image.camera.json",
        "/opt/data/private/ljn/InsertAny3D/runs/Farm_A_001/Task_001/step1/right/image.camera.json"
      ]
    }
  ]
}
```

填写时注意：

- `scene_images`、`scene_depths`、`scene_cameras` 的顺序必须都是`left, center, right`。
- `object_prompt` 必须准确指向新物体，不能同时描述锚点。
- `render_yaw_degrees`、`render_pitch_degrees` 和`render_side_angle_degrees` 应与 Unity task 一致。
- Unity 的 `Distance=3` 和 TRELLIS 的 `render_distance=1.5` 不需要数值一致，因为二者坐标尺度不同。TRELLIS distance 只需让生成的锚点和新物体完整入镜。
- `unity_manifest` 会把 Unity 中的默认 prompt 和任务补充 prompt 带入服务器记录；job 中显式写的 `object_prompt` 优先。
- 五个任务可以使用不同 yaw、pitch 和 TRELLIS distance。

把 JSON 上传到服务器：

```powershell
scp -P 25367 `
  ".\insert_jobs.Farm_A_001.json" `
  root@10.126.56.69:/opt/data/private/ljn/InsertAny3D/runs/Farm_A_001/insert_jobs.json
```

## 9. 在服务器串行运行任务

### 9.1 先做 dry-run

登录服务器：

```powershell
ssh -p 25367 root@10.126.56.69
```

远端 Linux：

```bash
cd /opt/data/private/ljn/InsertAny3D

third_party/TRELLIS/.venv/bin/python tools/run_insert_batch.py \
  --jobs runs/Farm_A_001/insert_jobs.json \
  --cuda-device 0 \
  --dry-run
```

dry-run 不加载模型，也不占用 GPU。它会打印每个 task 最终要运行的命令。先确认：

- 没有“文件不存在”。
- 每个 task 的输入和输出目录都使用自己的任务编号。
- 命令含有 `--trellis-input composite`、`--seg-engine legacy` 和`--run-sags`。
- 三组场景图片、深度和相机文件顺序正确。

### 9.2 选择 GPU

执行：

```bash
nvidia-smi
```

选择显存比较空闲、且没有别人正在运行重任务的 GPU。下面仍以 GPU 0 为例；实际使用时把 `--cuda-device 0` 换成选定编号。

### 9.3 在 tmux 中正式运行

先进入一个可断线继续运行的 tmux 会话：

```bash
tmux new-session -A -s insert_work
```

然后执行：

```bash
cd /opt/data/private/ljn/InsertAny3D

third_party/TRELLIS/.venv/bin/python tools/run_insert_batch.py \
  --jobs runs/Farm_A_001/insert_jobs.json \
  --cuda-device 0 \
  --skip-ready
```

调度器会严格串行运行 Task_001、Task_002……，不会让多个任务同时占用 GPU。按 `Ctrl+B`，松开后再按 `D`，可以退出 tmux 而不中止任务。以后执行下面命令重新进入：

```bash
tmux attach -t insert_work
```

某个任务失败时，默认会记录失败并继续后面的任务。最终输出：

```text
INSERT_BATCH_READY ...
```

表示任务表中的任务均已完成。`--skip-ready` 会跳过已有 `status=ready` 的任务，适合断点续跑。若你修改了一个已经 ready 的任务并希望强制重跑，不要给该次命令加 `--skip-ready`，最好另建一个只包含该任务的临时 job JSON。

## 10. 检查服务器结果

每个任务的主要目录是：

```text
Task_001
├── 01_segmentation
│   ├── mask.png
│   ├── annotated.png
│   └── points.json
├── 02_trellis
│   └── sample.ply
├── 03_rendered_3dgs
│   ├── source/images/left.png
│   ├── source/images/center.png
│   ├── source/images/right.png
│   ├── source/depths
│   └── model
├── 04_gim
│   ├── pair_00
│   ├── pair_01
│   └── pair_02
├── 05_pose
│   └── pose.json
├── 06_sags
│   └── inserted_object.ply
├── logs
└── manifest.json
```

不要只看最后有没有 PLY。每个任务至少检查以下内容：

1. `03_rendered_3dgs/source/images/center.png`应当能看到 TRELLIS 重建的“锚点 + 新物体”组合，而且二者都完整入镜。如果被裁切，修改该 task 的 `render_distance` 后重跑。这个 distance 不必等于 Unity。

2. `01_segmentation/annotated.png` 和 `mask.png`mask 应当只覆盖新物体，不应把锚点或大片背景一起选中。

3. `04_gim/pair_00/match.png` 和 `pair_02/match.png`匹配线应主要落在锚点和不变的环境区域。大量线落在无关位置通常表示编辑图改动过大、视角不一致，或生成的组合物体质量不足。

4. `05_pose/pose.json`文件应存在，位置、旋转、scale 都应是有限数值。当前没有自动 pose 质量拒绝门槛，最终仍要回 Unity 目视检查。

5. `06_sags/inserted_object.ply`文件应存在且不为空。它才是需要导回 Unity 的新物体；不要导入`02_trellis/sample.ply`，后者还包含锚点，会在 Unity 中造成锚点重复。

如果只因 TRELLIS 三视图裁切而要调整 distance，不必重新生成组合物体。建立一个只包含该任务的临时 job，把以下字段加入这个 task：

```json
{
  "input_ply": "/opt/data/private/ljn/InsertAny3D/runs/Farm_A_001/Task_001/02_trellis/sample.ply",
  "render_distance": 2.0
}
```

然后不带 `--skip-ready` 运行这个临时 job。编排器会复用 `sample.ply`，重新渲染三视图，并更新 segmentation、SAGS、GIM 和 pose。确认新 distance 后，再把数值写回正式 job 文件。

可以查看批次状态：

```bash
cd /opt/data/private/ljn/InsertAny3D
python3 -m json.tool runs/Farm_A_001/batch_manifest.json
python3 -m json.tool runs/Farm_A_001/Task_001/manifest.json
```

任务日志位于：

```text
runs/Farm_A_001/Task_001/logs/batch.log
runs/Farm_A_001/Task_001/logs/trellis.log
runs/Farm_A_001/Task_001/logs/segmentation.log
runs/Farm_A_001/Task_001/logs/gim_pair_00.log
runs/Farm_A_001/Task_001/logs/pose.log
runs/Farm_A_001/Task_001/logs/sags.log
```

## 11. legacy 分割错误时如何人工修正

首批任务使用 `legacy`。自动结果不准确时，不要把包含锚点的错误 PLY直接导入Unity。可以用 SAGS Gradio 页面人工点击新物体。

### 11.1 启动远端 UI

先在服务器选择一个空闲 GPU，然后执行：

```bash
cd /opt/data/private/ljn/InsertAny3D/third_party/SAGS

GRADIO_ANALYTICS_ENABLED=False \
GRADIO_SERVER_PORT=7860 \
CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python app_text.py
```

保持该窗口运行。在本机另开一个 Windows PowerShell，建立 SSH 隧道：

```powershell
ssh -p 25367 -N -L 7860:127.0.0.1:7860 root@10.126.56.69
```

浏览器打开：

```text
http://127.0.0.1:7860
```

### 11.2 人工标注

1. 在模型路径框输入绝对路径：

   ```text
   /opt/data/private/ljn/InsertAny3D/runs/Farm_A_001/Task_001/03_rendered_3dgs/model
   ```

2. 点击 `打开3D模型`，等待图片和 SAM 特征加载完成。
3. 在 Gallery 选择中心视图。
4. 在 `Origin` 图上点击新物体内部，不要点击锚点。
5. 在 `S / M / L` 中选择边界最合适的 SAM mask。
6. 点击 `分割为3D`，检查“临时结果”。
7. 结果正确时点击 `加入最终结果`。
8. 若背面或遮挡部分缺失，切换到其他视图，再点击缺失部分、分割并加入最终结果。
9. 在对象名输入框填写 `inserted_object` 并提交保存。

当前页面的 `Segment Prompt` 文本框还没有接入实际回调，人工修正时应使用图片点击，不要依赖这个文本框。

保存结果位于：

```text
.../03_rendered_3dgs/model/objects/inserted_object/fg.ply
```

把它替换为标准输出：

```bash
mkdir -p /opt/data/private/ljn/InsertAny3D/runs/Farm_A_001/Task_001/06_sags

cp \
  /opt/data/private/ljn/InsertAny3D/runs/Farm_A_001/Task_001/03_rendered_3dgs/model/objects/inserted_object/fg.ply \
  /opt/data/private/ljn/InsertAny3D/runs/Farm_A_001/Task_001/06_sags/inserted_object.ply
```

人工修正 SAGS 不需要重新计算 pose，因为 pose 使用的是组合物体与 Unity 原场景的图像匹配，而不是最终提取 PLY 的 mask。

完成后在服务器 UI 终端按 `Ctrl+C`，本机 SSH 隧道也按 `Ctrl+C`。不要让 UI长期占用 GPU。

## 12. 下载 PLY 和 pose 到 Windows

在 Windows PowerShell 中执行。下面的循环会为五个任务建立标准目录，只下载Unity 真正需要的两个文件：

```powershell
$scene = "Farm_A_001"
$tasks = "Task_001", "Task_002", "Task_003", "Task_004", "Task_005"
$localRoot = "F:\ServerRuns\$scene"
$remoteRoot = "/opt/data/private/ljn/InsertAny3D/runs/$scene"

foreach ($task in $tasks) {
  New-Item -ItemType Directory -Force -Path "$localRoot\$task\05_pose" | Out-Null
  New-Item -ItemType Directory -Force -Path "$localRoot\$task\06_sags" | Out-Null

  scp -P 25367 `
    "root@10.126.56.69:$remoteRoot/$task/05_pose/pose.json" `
    "$localRoot\$task\05_pose\pose.json"

  scp -P 25367 `
    "root@10.126.56.69:$remoteRoot/$task/06_sags/inserted_object.ply" `
    "$localRoot\$task\06_sags\inserted_object.ply"
}
```

先确认文件都下载成功，再打开 Unity。

## 13. 在 Unity 中导入新物体并应用 pose

### 13.1 导入 PLY

1. 用 Unity Hub 再次打开 Farm 项目和 `Farm_A_001.unity`。
2. 打开`工具 > 插入流程 > 任务管理`。
3. 点击 `Task_001`。
4. 展开`高级设置`，先确认`导入时翻转 Z 轴`当前值。
5. 点击`导入生成物（PLY）`。
6. 选择：

   ```text
   F:\ServerRuns\Farm_A_001\Task_001\06_sags\inserted_object.ply
   ```

7. 等待 Unity 完成转换。

工具会创建无损 Float32 `GaussianSplatAsset`：

```text
Assets/GaussianAssets/InsertTasks/Task_001/
```

并在任务下建立：

```text
Generated/GaussianSplat
```

如果该 task 已经导入过 PLY，Unity 会询问是否替换。确认文件正确后再选择`替换`。

`导入时翻转 Z 轴`是在导入时应用到 `GaussianSplat` 子物体上的。修改复选框不会自动改变已经导入的对象；切换后必须重新导入，或明确调整子物体的localScale Z。

### 13.2 应用 pose

仍在“任务管理”中，点击：

```text
应用位姿（JSON）
```

选择：

```text
F:\ServerRuns\Farm_A_001\Task_001\05_pose\pose.json
```

工具会把 position、xyzw quaternion rotation 和 scale 应用到该任务的`Generated` 根物体。

### 13.3 首个任务的人工标定

对 Task_001 做以下检查：

- 新物体位于预期插入区域附近。
- 新物体尺寸与锚点和环境相符。
- 上下方向正确，没有横躺、镜像或前后颠倒。
- 新物体和地面接触合理，没有明显悬空或深埋。
- 场景中没有第二份锚点；如果有，通常是误导入了 `sample.ply`。
- 从多个 Scene 视角观察时，Gaussian 没有只在单一角度看起来正确。

第一次应分别比较：

1. `导入时翻转 Z 轴`开启，重新导入 PLY，再应用 pose。
2. `导入时翻转 Z 轴`关闭，重新导入 PLY，再应用同一 pose。

选择方向正确的一种，并在同一批场景中保持一致。当前 pose 默认按`generated-axis=identity` 生成，不要一边随意旋转 `Generated`，一边继续使用原 pose 文件，否则命令行 benchmark 会再次应用 pose，覆盖手工拖动结果。

若 pose 本身明显错误，先检查服务器 GIM 和 pose 输入，不要直接渲染 240 图。

### 13.4 导入其余任务并保存场景

Task_001 验证正确后，对其余任务重复“导入 PLY”和“应用 pose”。完成后按`Ctrl+S` 保存场景。

保存非常重要。最终 benchmark 是一个新的 Unity 进程；如果没有保存场景，它看不到刚才导入到 Hierarchy 中的 `Generated/GaussianSplat`。

## 14. 渲染步骤 6 的 240 个评估视图

运行前再次确认：

- 每个任务的 `Generated` 下都有 `GaussianSplat`。
- 场景已经保存。
- 每个 task 都有对应的 `pose.json`。
- Unity Editor 已关闭。
- `F:\InsertRuns\Farm_A_001` 有足够磁盘空间。

Windows PowerShell：

```powershell
Set-Location "F:\Shared\Unity_Projects\MyProjects\Farm"

python .\Tools\render_insert_workflow.py `
  --stage benchmark `
  --scene Assets/Scenes/Farm_A_001.unity `
  --output F:\InsertRuns\Farm_A_001 `
  --pose-root F:\ServerRuns\Farm_A_001 `
  --task Task_001 `
  --task Task_002 `
  --task Task_003 `
  --task Task_004 `
  --task Task_005
```

命令会在每个独立 Unity 进程中重新读取对应的 `pose.json`，然后串行渲染：

- pitch = 10、20、30、40 度。
- 每个 pitch 有 30 个 yaw，间隔 12 度，完整环绕一圈。
- `original` 120 个视图，隐藏所有新物体。
- `inserted` 120 个视图，只显示当前 task 的新物体。
- 每个视图同时写 `image.png`、`image.raw`、`image.txt` 和`image.camera.json`。

输出结构：

```text
F:\InsertRuns\Farm_A_001\Task_001\step6\
├── original\
│   ├── pitch_10\view_000 ... view_029
│   ├── pitch_20\view_000 ... view_029
│   ├── pitch_30\view_000 ... view_029
│   └── pitch_40\view_000 ... view_029
└── inserted\
    └── 同样的 4 × 30 目录
```

1024 × 1024 float32 深度每张约 4 MiB。每个任务仅 240 张深度就接近 0.94 GiB，再加 RGB 和元数据，五个任务通常需要超过 5 GiB。正式运行前应预留更充足空间。

## 15. 核对最终输出

Windows PowerShell：

```powershell
$root = "F:\InsertRuns\Farm_A_001\Task_001\step6"

(Get-ChildItem "$root\original" -Recurse -Filter image.png).Count
(Get-ChildItem "$root\inserted" -Recurse -Filter image.png).Count
(Get-ChildItem "$root\original" -Recurse -Filter image.raw).Count
(Get-ChildItem "$root\inserted" -Recurse -Filter image.raw).Count
```

四行都应输出：

```text
120
```

然后检查：

- `task_manifest.json` 中 `actualStep6OriginalViews=120`。
- `task_manifest.json` 中 `actualStep6InsertedViews=120`。
- 同一个 pitch/view 下，original 与 inserted 相机完全一致。
- original 中没有生成物，inserted 中有当前 task 的生成物。
- inserted 中没有其他 task 的生成物。
- 两组深度文件都存在且大小正确。
- 随机抽查前、后、左、右多个角度，新物体没有突然消失或严重错位。

## 16. 常见问题

| 现象 | 最常见原因 | 处理方法 |
| --- | --- | --- |
| 命令行 Unity 报项目已打开 | Unity Editor 正在打开同一项目 | 保存并关闭 Editor 后重试；Hub 可保持开启 |
| RGB 中没有原有 Gaussian | 使用了 `-nographics` 或图形后端异常 | 使用现有脚本；必要时加 `--api d3d11` 重试，不要用 `-nographics` |
| 三视图中插入区域太小 | Unity `距离`太大 | 回“任务管理”减小`距离`后重拍 |
| 左右图看不到共同锚点 | 水平角、距离或左右夹角不合适 | 调整`水平角`/`距离`；首轮尽量保留 ±24 度 |
| TRELLIS center 裁掉了组合物体 | 服务器 `render_distance` 太小 | 增大该 task 的 render_distance，使用已有 sample.ply 重跑后续阶段 |
| SAGS PLY 带着锚点 | legacy mask 把锚点也选中，或导错了文件 | 人工修正 SAGS；Unity 只导入 `06_sags/inserted_object.ply` |
| 新物体镜像或前后反了 | Z 轴约定不匹配 | 切换`导入时翻转 Z 轴`后重新导入并比较 |
| 新物体完全不出现 | 没有保存导入后的场景，或 `Generated` 为空 | 检查 Hierarchy、重新导入 PLY、保存场景 |
| benchmark 后手工位置被覆盖 | 命令重新应用了 pose.json | 修正 pose 或 pose 协议，不要仅拖动 Generated 根物体 |
| 没有生成 pose.json | depth/camera 缺失或三组列表顺序错误 | 检查 left/center/right 的 PNG、RAW、camera JSON 是否一一对应 |
| 批任务修改后没有重跑 | 使用了 `--skip-ready` | 对修改的任务去掉 `--skip-ready`，最好使用单任务临时 job |
| UI 启动长时间等待 | Gradio 在尝试外部 analytics 请求 | 设置 `GRADIO_ANALYTICS_ENABLED=False` |
| 深度图不能用普通看图软件打开 | `image.raw` 是 float32 径向距离 | 保持原文件，交给反投影和指标脚本读取 |

## 17. 推荐的首批执行顺序

不要一开始就连续完成五个任务。推荐顺序是：

1. 建好完整 Unity 场景。
2. 建立五个 InsertTask，但先只处理 Task_001。
3. 反复调整 Task_001 三视图，直到构图稳定。
4. 编辑 Task_001 中间图。
5. 服务器只跑 Task_001。
6. 检查组合重建、mask、GIM、pose 和 SAGS PLY。
7. 回 Unity 比较`导入时翻转 Z 轴`开与关。
8. 手动应用 pose，从多个角度目视验收。
9. 只为 Task_001 跑一次 benchmark，检查 120 + 120 输出。
10. 约定固定后，再编辑和串行处理 Task_002 到 Task_005。

这能把坐标、方向或分割约定的问题限制在一个任务内，避免一次生成大量无效结果。

## 18. 每个任务的完成清单

一个任务只有同时满足下面各项，才算完成：

- [ ] Unity `left/center/right` 三张 RGB 构图合格。
- [ ] 三张 `image.raw` 和 `image.camera.json` 都存在。
- [ ] 编辑图只增加了目标物体，没有改变相机和锚点。
- [ ] TRELLIS center 中锚点与新物体都完整。
- [ ] legacy/SAGS mask 只包含新物体；必要时已人工修正。
- [ ] 左右 GIM 匹配主要落在合理的共同区域。
- [ ] `05_pose/pose.json` 存在。
- [ ] `06_sags/inserted_object.ply` 存在且没有重复锚点。
- [ ] Unity 中已确认 Z 翻转、位置、旋转和比例。
- [ ] 导入后的 Unity 场景已经保存。
- [ ] original 120 张、inserted 120 张 RGB 都存在。
- [ ] original 120 张、inserted 120 张深度都存在。
- [ ] 随机抽查多个环绕角度，没有明显错位或消失。

完成首个真实任务后，再根据实际错误决定是否调整 Z 轴约定、pose 质量门槛、
SAGS seed 半径或 LangSAM。当前阶段不建议同时改动这些变量。
