# InsertAny3D 安装说明

本文用于把刚克隆的仓库安装到可运行状态，也可直接交给 AI 按顺序执行。

当前仓库已有组合流水线脚本，但没有脱离 Unity 输入和任务参数的“一键运行”方式。安装完成的标准是：六个第三方源码仓库重建完成，三套 Python 环境通过 GPU 验证，SAGS、GIM、TRELLIS 和 Hunyuan3D-2 所需权重就位。MVInpainter 是对比实验，不属于主流程。

## 1. 准备机器

推荐使用 Linux x86_64、NVIDIA GPU 和 Conda。服务器 2 已验证的配置是 Ubuntu 20.04、RTX 3090、NVIDIA 驱动 570.207、CUDA Toolkit 12.4 和 11.8。

安装前确认：

```bash
nvidia-smi
git --version
conda --version
/usr/local/cuda-12.4/bin/nvcc --version
/usr/local/cuda-11.8/bin/nvcc --version
```

还需要 `git-lfs`、C/C++ 编译工具和能访问 GitHub、PyPI、PyTorch、Hugging Face 的网络。建议至少预留 120 GB；若还要下载全部模型、缓存和 MVInpainter 数据，建议预留 200 GB。

完整模型流程建议使用 24 GB 显存。显存较小的机器可以先完成环境验证，再单独测试模块。

## 2. 克隆并检查仓库

需要修改或提交代码时，使用 Git 地址克隆：

```bash
git clone https://www.modelscope.cn/models/KHUU0424/InsertAny3D_v1.git InsertAny3D
cd InsertAny3D
```

`modelscope download --model KHUU0424/InsertAny3D_v1` 下载的是不含 `.git` 的运行快照，适合只运行、不提交代码的情况。

确认以下内容存在：

```bash
test -f tools/install_environments.sh
test -f tools/verify_environments.sh
test -f tools/bootstrap_third_party.sh
test -d environment
test -d code/third_party/patches
test -d code/third_party/overlays
```

任一命令报错都说明仓库内容没有下载完整。不要继续安装，也不要自行把第三方仓库升级到最新版。

`third_party`、模型权重和虚拟环境不会存入 Git。先按固定提交重建第三方源码：

```bash
bash tools/bootstrap_third_party.sh
```

安装脚本发现 `third_party` 不存在时也会自动执行这一步。

重建后的子仓库会显示本地修改，这是正常现象：脚本先检出固定的上游提交，
再应用本项目保存的源码补丁和新增文件。版本清单见
`code/third_party/THIRD_PARTY_REPOS.md`。不要在子仓库中执行清理或升级。

## 3. 指定 Conda 和 CUDA

安装脚本需要同时找到 CUDA 12.4 和 11.8。系统路径不同时，先设置环境变量：

```bash
export CONDA_BIN="$(command -v conda)"
export CUDA_12_HOME=/usr/local/cuda-12.4
export CUDA_11_HOME=/usr/local/cuda-11.8
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=8
```

`8.6` 适用于 RTX 3090。其他显卡应改成对应的 CUDA Compute Capability。内存不足或编译进程被杀时，把 `MAX_JOBS` 调成 `2` 或 `4`。

服务器 2 使用的是自定义安装位置，对应配置为：

```bash
export CONDA_BIN=/opt/data/private/ljn/miniconda3/bin/conda
export CUDA_12_HOME=/opt/data/private/ljn/cuda-12.4
export CUDA_11_HOME=/opt/data/private/ljn/cuda-11.8
```

## 4. 安装三套环境

从仓库根目录执行：

```bash
bash tools/install_environments.sh all 2>&1 | tee install.log
```

脚本会创建：

| 环境 | Python 路径 | 用途 |
| --- | --- | --- |
| 主环境 | `third_party/TRELLIS/.venv/bin/python` | TRELLIS、SAGS、LangSAM、3DGS 渲染 |
| Hunyuan 环境 | `third_party/Hunyuan3D-2/.venv/bin/python` | Hunyuan3D-2 |
| GIM 环境 | `third_party/gim/.venv/bin/python` | GIM、RoMa、Depth Anything |

只需要某个模块时，也可以分别安装：

```bash
bash tools/install_environments.sh main
bash tools/install_environments.sh hunyuan
bash tools/install_environments.sh gim
```

不要混用三套环境，也不要在系统 Python 中补包。SAGS 的 .venv 会指向主环境；TRELLIS-old 仅保留历史文件，不参与安装或主流程。

安装脚本会把部分源码放到 `.build/environment_sources`，其中包含 editable 安装依赖。环境仍在使用时不要删除这个目录。

## 5. 下载模型权重

建议先把 Hugging Face 缓存放到空间充足的位置，再执行统一下载脚本：

```bash
export HF_HOME=/path/to/large-disk/huggingface
mkdir -p "$HF_HOME"
bash tools/download_models.sh
```

脚本下载主流程需要的 SAGS、GIM、TRELLIS 和 Hunyuan3D-2 权重；直接下载的
三个权重会校验 SHA-256，Hugging Face 模型固定到已验证的 revision。重复执行
不会重复下载已通过校验的文件。

网络环境需要镜像时，可以额外设置 `HF_ENDPOINT`。后续运行也应保持相同的
`HF_HOME`，否则程序会认为权重尚未下载。GIM 的上游权重托管在 Google Drive；
网络无法访问时，脚本会明确失败，不会用错误文件继续安装。

## 6. 验证安装

先执行统一环境验证：

```bash
CUDA_VISIBLE_DEVICES=0 bash tools/verify_environments.sh
```

最后一行出现下面内容，才表示三套环境和 CUDA 扩展均可用：

```text
INSERTANY3D_THREE_ENVIRONMENTS_READY
```

再运行两个不需要模型推理的仓库自测：

```bash
third_party/TRELLIS/.venv/bin/python tools/test_estimate_similarity_pose.py
third_party/TRELLIS/.venv/bin/python tools/test_insert_batch.py
```

成功标志分别是 `INSERT_PIPELINE_POSE_SYNTHETIC_OK` 和 `INSERT_BATCH_TEST_READY`。

需要继续验证分割、GIM 和 TRELLIS 渲染时，按 `第一阶段运行说明.md` 操作；需要了解组合流水线参数时，查看 `tools/README_INSERT_PIPELINE.md`。

## 7. 可选：MVInpainter

MVInpainter 是对比实验，不参与默认安装。如果确实需要它，再按其上游 README 单独建立环境和下载权重。

如果仓库中提供了打包数据，可在 `third_party/MVInpainter/data` 下解包：

```bash
cd third_party/MVInpainter/data
tar -xf mvimagenet.tar
tar -xf masks.tar
```

数据很大，主流程用户可以跳过整节。

## 8. 常见问题

- `未找到 conda`：重新设置 `CONDA_BIN`，确保它指向真实的 `conda` 可执行文件。
- `nvcc not found`：检查 `CUDA_12_HOME`、`CUDA_11_HOME`，不要只看 `nvidia-smi` 显示的 CUDA 版本。
- CUDA 扩展编译失败：先确认 CUDA 路径和显卡架构，再降低 `MAX_JOBS` 后重跑对应环境。
- `No space left on device`：优先清理 Conda 和 pip 下载缓存，不要删除 `.venv`、`.build/environment_sources` 或 Hugging Face 模型缓存。
- 模型再次下载：运行时的 `HF_HOME` 与下载权重时不一致。
- 某个模块能导入但完整流水线不能运行：这是可能的。当前仓库仍需要 Unity 输出、图片编辑结果和任务参数，不应把环境验证成功理解成端到端流程已经自动打通。

## 给 AI 的执行约束

1. 所有命令从仓库根目录执行。
2. 每完成一节就检查退出码；失败时停止，不要用升级依赖来绕过错误。
3. 以 `environment/requirements-*.txt` 和安装脚本锁定的版本为准。
4. 不删除 `.venv`、`.build/environment_sources`、模型权重和 Hugging Face 缓存。
5. 最终必须报告统一验证脚本的最后一行，以及两个仓库自测的成功标志。
