# 第三方仓库版本记录

根仓库不直接保存这六个第三方工作树。`tools/bootstrap_third_party.sh`
会克隆下面的固定提交，再应用 `patches/` 中的已跟踪源码差异和
`overlays/` 中新增的源码、配置文件。权重、虚拟环境、缓存、输入数据和
运行结果不在补丁或 overlay 中。

因此 `third_party/*` 在安装完成后出现本地修改是预期状态，不代表版本漂移。
若上游地址不可访问，则无法仅凭根仓库重建第三方源码。

## Hunyuan3D-2
- 仓库：https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git
- 提交：b173994017b1ab9559792fbdfa6194952e2ae2e0
- 分支：main

## MVInpainter
- 仓库：https://github.com/ewrfcas/MVInpainter.git
- 提交：323d7f6ce3f73b0f263eb7f07dc48aefa6f27f34
- 分支：main

## SAGS
- 仓库：https://github.com/MrHandsomeljn/SAGS.git
- 提交：4c020b3290072a26b2b8ce9b023b7e553741b884
- 分支：Gradio-by-MrHandsomeljn

## TRELLIS
- 仓库：https://github.com/MrHandsomeljn/TRELLIS
- 提交：1c4ab02e359f991d949cc527b81f065f2f266b92
- 分支：main

## TRELLIS-old
- 仓库：https://github.com/microsoft/TRELLIS.git
- 提交：eb83038919f6e1feb63accf3a97a377a608c497d
- 分支：main

## gim
- 仓库：https://github.com/xuelunshen/gim.git
- 提交：89e9cddbf1f013f50587a0198b0382b657cf0f05
- 分支：main
