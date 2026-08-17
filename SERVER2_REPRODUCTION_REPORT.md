# 服务器 2 复现报告

复现位置：`/opt/data/private/ljn/InsertAny3D`

复现日期：2026 年 8 月 14 日

## 结论

ModelScope 中的代码与数据已经完整下载到服务器 2，六个第三方仓库已经恢复到记录的提交。运行环境已经整理为一套主环境和两套独立环境，两套 CUDA 工具包、CUDA 扩展和模型权重均已安装。所有环境都通过了 Python、PyTorch、CUDA 和主要扩展的导入及显卡计算测试。

目前已实际跑通 Hunyuan3D-2、GIM 和 TRELLIS 三条代表性模型链路。仓库尚未提供从图片生成、匹配、分割一直贯通到 Unity 插入的统一可执行入口，因此本次不能宣称整个 InsertAny3D 端到端流程已经跑通。

## 服务器环境

- 操作系统：Ubuntu 20.04.5 LTS
- Linux 内核：5.11.0-46-generic
- 显卡：NVIDIA GeForce RTX 3090
- NVIDIA 驱动：570.207
- 本地 CUDA 工具包：11.8 和 12.4
- NVCC：11.8.89 和 12.4.131

根目录 README 中的 Ubuntu 20.04.6、内核 5.15 和驱动 550.163.01 是原始服务器的环境记录，不是服务器 2 的配置。各模块实际使用的软件版本如下：

| 模块 | Python | PyTorch | PyTorch CUDA |
| --- | --- | --- | --- |
| Hunyuan3D-2 | 3.12.11 | 2.7.1+cu118 | 11.8 |
| SAGS | 3.11.13 | 2.5.1+cu121 | 12.1 |
| TRELLIS | 3.11.13 | 2.5.1+cu121 | 12.1 |
| TRELLIS-old | 3.11.13 | 2.5.1+cu121 | 12.1 |
| GIM | 3.9.23 | 1.12.1+cu113 | 11.3 |

SAGS 和 TRELLIS-old 共用 `third_party/TRELLIS/.venv` 主环境；Hunyuan3D-2 和 GIM 各自保留独立环境。MVInpainter 仅作为对比实验代码保留，不再创建运行环境。

## 代码与数据

- ModelScope 数据清单：11346/11346 个文件，合计 52840957268 字节。
- MVInpainter 的 `mvimagenet.tar` 和 `masks.tar` 已解包到项目所需位置。
- 六个上游仓库的提交如下：

| 仓库 | 提交 |
| --- | --- |
| Hunyuan3D-2 | `b173994017b1ab9559792fbdfa6194952e2ae2e0` |
| MVInpainter | `323d7f6ce3f73b0f263eb7f07dc48aefa6f27f34` |
| SAGS | `4c020b3290072a26b2b8ce9b023b7e553741b884` |
| TRELLIS | `1c4ab02e359f991d949cc527b81f065f2f266b92` |
| TRELLIS-old | `eb83038919f6e1feb63accf3a97a377a608c497d` |
| GIM | `89e9cddbf1f013f50587a0198b0382b657cf0f05` |

环境目录 `.venv` 没有来自模型仓库，而是根据各模块的环境版本清单重新创建。模型权重也在服务器 2 直接从模型托管站下载，没有经过本机与服务器之间传输。

## 验证结果

- 三套实际环境全部通过 CUDA 张量运算和主要原生扩展导入测试；主环境已经分别从 TRELLIS、SAGS 和 TRELLIS-old 的工作目录完成验证。
- Hunyuan3D-2：完成 5 步扩散、体素解码和网格导出，输出 `reproduction_outputs/hunyuan_smoke.glb`。
- GIM：完成真实图片对的密集匹配，得到形状为 `(1344, 2688, 4)` 的匹配结果，并采样 5000 个稀疏对应点。
- TRELLIS：从本地权重加载全部 6 个子模型，完成两阶段各 4 步采样和高斯解码，输出 `reproduction_outputs/trellis_smoke.ply`。

第一阶段功能入口也已完成验收：

- SAGS 文本分割入口生成了非空 `mask.png`、透明前景图、检测框预览和检测信息；
- 修复后的 GIM 官方演示生成了匹配图和透视校正图；
- TRELLIS 3DGS 渲染入口从已有 PLY 生成了 RGB、深度、COLMAP 相机文件和 3DGS 模型目录。

具体命令和产物见 `第一阶段运行说明.md`。

主要日志位于项目外的 `/opt/data/private/ljn/`：

- `insertany3d_hunyuan_model_test.log`
- `insertany3d_gim_model_test3.log`
- `insertany3d_trellis_model_test2.log`
- `insertany3d_trellis_old_smoke.log`
- `insertany3d_download_weights.log`

## 已知限制

1. 根目录 README 描述的是设计中的完整流程，但当前代码没有一个可以直接运行的端到端入口，SAGS、Unity 和 IPC 的完整串联仍需按具体实验参数验证。MVInpainter 已确定为对比模型，不纳入主流程。
2. GIM 原始 `demo.py` 已完成模型匹配，但在最终绘图时调用 `fast_make_matching_figure` 少传两个参数；本次使用不改变模型逻辑的测试脚本验证了匹配和采样本身。
3. 原始服务器当前 SSH 仍为 TCP 连接超时。TRELLIS 两个 flexicubes 子模块在原环境中被记录为 `dirty`，但发布内容没有包含二进制差异；服务器 2 已恢复其基础提交，待原服务器恢复后还需核对差异。
4. SAGS 的发布补丁只能通过上下文模糊匹配应用。当前代码可以安装和导入，但仍建议在原服务器恢复后再核对一次补丁内容。
