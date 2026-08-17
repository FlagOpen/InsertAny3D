# InsertAny3D 决策记录

更新时间：2026-08-16

## 已采用

- 采用“组合物体优先”：图片编辑后的完整画面直接输入 TRELLIS，使锚点和新物体
  处于同一个生成坐标系。不会在 TRELLIS 前先把新物体抠出来。
- TRELLIS 三视图渲染后，再在生成的 `center.png` 上定位新物体。这样分割点与
  SAGS 使用的 3DGS 相机严格对齐；SAGS 输出 `06_sags/inserted_object.ply`，
  供 Unity 插入，避免把组合物体中的锚点重复插入原场景。
- Unity 使用已安装的 `org.nesnausk.gaussian-splatting` 1.1.1，把 SAGS PLY
  转成 `GaussianSplatAsset` 并挂到对应任务的 `Generated`。第一轮采用 Float32
  无损格式，不提前引入压缩误差。
- Unity 和 TRELLIS 都以 1024 x 1024 作为流程基准。GIM 或其他模型若限制输入
  尺寸，只在该模型入口下采样，并保留原始 1024 图和坐标缩放信息。
- Unity distance 和 TRELLIS distance 分属不同坐标系，不要求数值一致；要求 FOV
  一致，并分别选择能完整、清楚包含锚点和新物体的距离。
- 步骤 6 同时输出 RGB 和 float32 径向深度。深度竖直翻转已改为只复制一次
  NativeArray，避免 1024 分辨率、240 视图时的重复大数组分配。
- 当前继续使用 Unity `image.raw` 的现有径向距离编码进行反投影；不在首批前
  强行改成 z-depth，后续只在指标明确要求时再增加转换。
- 每个任务独立运行，任务之间串行调度。Unity CLI 可重复传入 `--task`；每个
  task 启动独立 Unity 进程、独立日志和独立 manifest。无 `--task` 时保留旧的
  一进程全任务入口。
- 暂不增加自动 pose 接受/拒绝门槛。首轮全流程由人工查看结果；manifest 仍保留
  内点数、残差等诊断数据，便于后续制定阈值。
- 首批分割采用 `legacy`，允许人工标注/修正原物体与生成物体的区域；这一步暂不
  追求全自动。LangSAM 留作后续固定引擎对照实验。
- prompt 采用“默认约束 + 任务补充”两层。默认文本用于保持锚点、场景和透视，
  用户文本描述本次物体/风格；Unity manifest 和服务端 job 都保存两层及合并后的
  effective 文本。
- 服务端场景批量采用 JSON job 文件串行调度，每个任务独立启动单任务编排器；
  默认失败后继续，支持 `--fail-fast` 和 `--skip-ready`。
- Task Manager 负责原场景搭建、插入点选择和任务配置。它可从选中锚点生成
  `InsertTask` 模板，并直接编辑位置、相机参数、编辑 prompt、物体 prompt、
  锚点 prompt、分辨率和光照选项。
- 光照选项已按真实语义改名为 `Replace Scene Lights`：默认关闭；打开时临时禁用
  场景灯并使用相机灯，不再使用容易误解的 `Use Scene Lighting` 名称。
- 服务器单任务编排器按 `01_segmentation`、`02_trellis`、
  `03_rendered_3dgs`、`04_gim`、`05_pose`、`06_sags` 保存；失败阶段会让
  manifest 总状态为 `failed`，即使使用 `--continue-on-error` 也不会误报 ready。

## LangSAM 在流程中的位置

LangSAM 不属于 TRELLIS，也不是替换 SAGS 内部的 SAM。它只是“生成后的中心
渲染图 + 文本 -> 2D mask/点击点”的入口适配器；这些点随后交给 SAGS，SAGS
仍使用自己的 SAM ViT-H 在多个 3DGS 视角上传播并提取 Gaussian。

- `legacy`：现有 GroundingDINO + SAM 生成初始 mask/点，当前已实测可用。
- `langsam`：固定使用 LangSAM；失败就让任务失败，结果来源最一致，但需要先把
  对应权重完整缓存并完成验收。
- `auto`：先尝试 LangSAM，失败时回退 legacy，适合开发期跑通；大批量时可能让
  不同任务混用不同分割引擎，数据一致性较差。

## 待确认或待首轮标定

- 后续生产批次是否改为固定 `langsam`，等权重和人工标注质量对比后再决定；当前
  不使用 `auto` 作为长期数据集策略。
- SAGS 的 2 像素 seed 回填是否保留。它解决了烟测中“有 SAM mask 但 Gaussian
  票数为 0”的问题，但需要用真实类别比较边界质量。
- `Flip Imported Gaussian Z` 默认开启只是兼容当前样本。首个真实任务需要同时看
  开/关两个结果，之后固定全批次约定；这不需要现在增加自动门槛。
- 新物体与原图的 GIM 外点暂不单独处理，按当前决定先人工观察整体结果。
- `anchorPrompt` 目前只作为默认/任务文本记录在 manifest 和 job 中，不直接参与
  模型调用；后续若需要约束编辑服务，再扩展 API 协议。
- 真实 Gemini 编辑、组合 TRELLIS、三视图 GIM、pose、SAGS、Unity 导入与 240 图
  的完整首任务尚需端到端验收。合成 pose 和各阶段烟测不能替代这一步。
- Gradio 5.27.0 与 `gradio_litmodel3d` 0.0.1 已安装到 SAGS venv；`app_text`
  导入和完整 `Blocks` 构建已通过。两者的包元数据版本范围不一致，LangSAM 也锁定
  其他依赖版本，因此保留为已知声明冲突，不把 Gradio UI 纳入无人值守 CLI 主链路。
  手工启动时固定 `GRADIO_ANALYTICS_ENABLED=False`，避免外部分析请求阻塞。

## 安全

`GEMINI_API_KEY` 只从环境变量或显式参数读取，不写入脚本、日志或 manifest。
目前没有使用真实密钥发送外部请求。
