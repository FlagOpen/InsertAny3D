# InsertAny3D 可复现代码

本目录只保存 InsertAny3D 自身代码、文档、环境配置，以及第三方仓库的精确来源记录，不包含模型权重、虚拟环境或实验数据。

- `project/`：InsertAny3D 自身代码和文档。
- `third_party/THIRD_PARTY_REPOS.md`：第三方仓库地址、分支和提交版本。
- `third_party/patches/`：第三方仓库中已跟踪文件的本地修改。
- `third_party/overlays/`：第三方仓库中未跟踪的手写源码和配置文件。
- `environment/`：各模块环境版本记录。

复现时先按记录克隆第三方仓库并切换到指定提交，再应用 patch 和 overlay，最后根据环境版本文件安装依赖。模型权重需按各上游项目说明另行下载。
