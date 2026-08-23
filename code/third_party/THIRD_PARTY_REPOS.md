# 第三方仓库版本记录

SAGS 已迁移为 Git submodule，由根仓库固定提交版本。其余第三方工作树暂由
`tools/bootstrap_third_party.sh` 克隆固定提交，再应用 `patches/` 中的已跟踪源码差异和
`overlays/` 中新增的源码、配置文件。权重、虚拟环境、缓存、输入数据和运行结果不进入 Git。

SAGS 工作树应保持干净；其余尚未迁移的 `third_party/*` 在 bootstrap 后出现预期补丁差异不代表版本漂移。
若上游或 fork 地址不可访问，对应第三方源码将无法仅凭根仓库重建。

## Hunyuan3D-2
- 仓库：https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git
- 提交：b173994017b1ab9559792fbdfa6194952e2ae2e0
- 分支：main

## MVInpainter
- 仓库：https://github.com/ewrfcas/MVInpainter.git
- 提交：323d7f6ce3f73b0f263eb7f07dc48aefa6f27f34
- 分支：main

## SAGS
- 仓库：https://github.com/Junnan-bjtu/SAGS.git
- 提交：cb905c26178b9ff1cf2de51cf0051d509192f159
- 分支：insertany3d

## TRELLIS
- 仓库：git@github.com:Junnan-bjtu/TRELLIS.git
- 提交：50599ef1b32bcc43924b19449f9c45689f660e96
- 分支：insertany3d

## TRELLIS-old
- 仓库：https://github.com/microsoft/TRELLIS.git
- 提交：eb83038919f6e1feb63accf3a97a377a608c497d
- 分支：main

## gim
- 仓库：https://github.com/Junnan-bjtu/gim.git
- 提交：e126052d86aa99292e41d289f6fb0b0f37dafe87
- 分支：insertany3d
