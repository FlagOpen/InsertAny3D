# InsertAny3D
代码仓库正在完善中

## DDLs:

#### 2025/08/07
- 创建仓库，确定仓库框架和大致日程

#### 2025/08/09
- 完成Unity脚本
- 完成通信模块
  1. 实现任意自由渲染接口
  2. 实现物体位姿控制

#### 2025/08/14
- 实现服务器反向连接
* 说明：
  1. 服务器运行python tools/ipc_server.py
  2. 端口转发12345到本地
  3. Unity连接本地转发端口
  4. 服务器运行python ipc_api_demo.py


#### 2025/08/13
- 建立主src
  1. 确认主src的文件目录格式
  2. 需要包含submodules和各个submodules的调用方式

#### 2025/08/15
- 调试Generate and Insert
- 将Unity包装成一个下载即可使用的工程

#### 2025/08/18
- 建立Agent模块和通信模块
  1. 实现自动Tasks
  2. 实现自动CLIP Prompts
  3. 实现自动CLIP选图
- 建立CLIP选图的采样策略
  1. 完全随机模式
  2. LookAt模式
    1. 对LookAt Object的位置的搜索（3D: xyz）
    2. 对视角的搜索（3D: E,A,D）

#### 2025/08/20
- 测试选图采样策略


## 代码目录结构
```txt
InsertAny3D
- Unity Proj # 创建新的Unity时直接复制该目录即可
- src
  - agent
    - view_select_tools
    - docs
    - api_key.conf
  - assets
    - date
      - name
        - source
          - images
          - sparse
        - output
          - point_cloud
          - object_{obj_name}
        - unity
        - gim
  - submodules
    - TRELLIS
      - 3DGS Renderer
    - GIM
    - SAGS
    - LangSAM
  - server_tools
  - main.py
```


## 代码逻辑结构：
### 1. Generate and Insert:

#### 输入：
> 1. 正面视角位姿$V1$，渲染深度图$D_{V1}$，渲染rgb图$I_{V1}$；
> 2. 侧面视角位姿$V2$，渲染深度图$D_{V2}$，渲染rgb图$I_{V2}$；
> 3. 正面视角渲染rgb图的编辑后结果$I_{V1}'$
> 位姿保存格式类似cameras.txt文件，深度图为raw格式，rgb图为png格式

#### 操作：
> 1. 对$I_{V1}'$进行分割，获取主物体+Anchor的组合物体分割结果$I''_{V1}$
>    - 这一步可能会使用GPT-4o or RemBG进行，之前是直接拉框or打点
> 2. TRELLIS：$I''_{V1}$ -> 3D Asset $A_c$
> 3. 对$A_c$进行环绕渲染，获取3DGS格式的source文件夹{images, sparse}
> 4. 从images中选取最接近$I_{V1}$和$I_{V2}$的图片$R_{V1}$和$R_{V2}$
> 5. 使用Multiview Depth Matching，获取位姿矩阵$S$,$V$,$T$
> 5. 用SAGS进行物体分割，获取前景3D Asset $A_s$

#### 输出：
> 1. 插入物体$A_s$
> 2. 插入位姿$S$,$V$,$T$

#### Submodules:
1. (18G VRAM) TRELLIS
2. ( 4G VRAM) 3DGS Renderer
3. (20G VRAM) GIM
4. (20G VRAM) SAGS+LangSAM

### 2. InsertAny3D

包含Unity脚本、Agent模块、通信模块、图片编辑模块
#### 1. Unity C#脚本
给定位姿渲染图片，并将渲染图片保存为指定格式：

    1. 单图，渲染rgb、深度图，并保存单个位姿文件
    2. 多图序列，3dgs格式，即source{images, sparse}，方便重建为3DGS场景
    3. 相机序列，存储为视频输出
    4. 给定$A_s$、$S$,$V$,$T$，将指定物体插入到指定位姿
#### 2. Agent模块
    1. gpt-4o：查看场景，生成tasks，clip prompt，edit prompt
    2. clip & gpt-4o：从Unity中获取图片，选出$V1$、$V2$
    3. 调用图片编辑模块编辑，并选取最佳编辑$I_{V1}'$
    4. 任务要求文档
#### 3. 通信模块
    1. 调用Generate and Insert
    2. Python IPC：获取Agent要求，传递给Unity
    3. Unity IPC：获取渲染需求，渲染出文件保存到指定路径
    4. Python loader：读取渲染文件
#### 4. 图片编辑模块
    1. gpt-4o-image
    2. FLUX-kontext
    3. Fooocus
    4. ...

## 致谢

感谢以下优秀开源项目：[TRELLIS](https://github.com/microsoft/TRELLIS)、[Fooocus](https://github.com/lllyasviel/Fooocus)、[SAGS](https://github.com/XuHu0529/SAGS)、[UnityGaussianSplatting](https://github.com/aras-p/UnityGaussianSplatting) 和 [GIM](https://github.com/xuelunshen/gim)。

## 实验运行环境

本仓库中的实验在以下服务器环境中运行：

- 操作系统：Ubuntu 20.04.6 LTS（Focal Fossa），x86_64 架构
- Linux 内核：5.15.0-139-generic
- 显卡：8 张 NVIDIA GeForce RTX 3090
- NVIDIA 驱动版本：550.163.01
- 系统 CUDA 工具包：12.4
- NVCC 版本：V12.4.131

项目统一使用三套运行环境，不再为每个第三方仓库分别创建环境：

- 主环境：`third_party/TRELLIS/.venv`，供 TRELLIS、TRELLIS-old、SAGS、
  LangSAM 和 3DGS 渲染代码共同使用；
- Hunyuan 环境：`third_party/Hunyuan3D-2/.venv`；
- GIM 环境：`third_party/gim/.venv`。

`third_party/SAGS/.venv` 和 `third_party/TRELLIS-old/.venv` 只是指向主环境的
符号链接。MVInpainter 是对比实验模型，不属于 InsertAny3D 主流程，也不创建
主流程运行环境。

环境、权重和运行结果不会上传到模型仓库。从新 clone 开始的完整安装方法见
[`install.md`](install.md)，三套环境的详细说明见
[`环境安装说明.md`](环境安装说明.md)。可以使用
`tools/install_environments.sh` 分别安装三套环境，再使用
`tools/verify_environments.sh` 验证环境和 CUDA 扩展。历史环境的完整软件包
版本仍保存在 `code/environment` 和各第三方仓库的
`ENVIRONMENT_VERSIONS.txt` 中，供排查版本差异时参考。

## 第一阶段可运行入口

项目已经提供三个非交互命令：

- `tools/segment_image.py`：使用 GroundingDINO 和 SAM 生成文本提示对应的前景掩码；
- `third_party/gim/demo.py`：执行 GIM 图片匹配并输出匹配图和校正图；
- `tools/render_trellis_3dgs.py`：将 TRELLIS Gaussian PLY 渲染为 RGB、深度、COLMAP 相机文件和 3DGS 模型目录。

具体命令、目录结构和服务器 2 验收记录见
[`第一阶段运行说明.md`](第一阶段运行说明.md)。

### MVInpainter 数据说明

`third_party/MVInpainter/data` 中的 `mvimagenet` 和 `masks` 原本包含数百万个
小文件，超过模型仓库单次上传的文件数量限制。因此发布时将它们完整打包为：

- `third_party/MVInpainter/data/mvimagenet.tar`
- `third_party/MVInpainter/data/masks.tar`

下载后进入 `third_party/MVInpainter/data` 目录，执行以下命令即可恢复原目录：

```bash
tar -xf mvimagenet.tar
tar -xf masks.tar
```
