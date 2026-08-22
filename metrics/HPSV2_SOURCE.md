# HPSv2 外部依赖

`metrics/evaluate_hpsv2.py` 使用 HPSv2 作为无监督图文一致性指标。HPSv2
源码不复制进主仓库，也不把它的 `.git` 目录提交到主仓库。

当前服务器上的 `metrics/HPSv2` 是独立 Git 仓库，来源为其自身的 `origin`，
并保留本地修改。部署时应固定外部仓库 commit，再应用本地补丁：

```bash
git clone <HPSv2-上游或项目 fork> metrics/HPSv2
git -C metrics/HPSv2 checkout <固定 commit>
git -C metrics/HPSv2 apply metrics/patches/HPSv2.patch
```

如果不需要本地修改，直接使用外部仓库固定 commit 即可。HPSv2 权重和 CLIP
权重通过运行参数或环境变量提供，不进入 Git；评测输出写入运行目录的
`metrics/hpsv2.json`，也不进入 Git。
