# Server-side InsertAny3D tools

这些脚本已同步到服务器项目 `/opt/data/private/ljn/InsertAny3D/tools`。

| 脚本 | 用途 |
| --- | --- |
| `gemini_edit.py` | APIYi Gemini 图片编辑，输入图片和 prompt，输出图片 |
| `auto_segment.py` | LangSAM/legacy 文本分割并生成 SAGS `points.json` |
| `generate_trellis_asset.py` | 编辑图生成 TRELLIS Gaussian PLY/可选 GLB |
| `render_trellis_3dgs.py` | PLY 生成 RGB、深度、COLMAP 和 3DGS model |
| `render_trellis_views.py` | 按 yaw/pitch/distance 生成 left/center/right 定向视图 |
| `run_gim_match.py` | 任意两张图片的 GIM 匹配及 `matches.json` |
| `estimate_similarity_pose.py` | 多视角双深度反投影并输出 Unity pose JSON |
| `run_sags_text.py` | 无 UI 调用已有 SAGS 多视角分割 |
| `run_insert_pipeline.py` | 单任务串联组合 TRELLIS、三视图、GIM、pose、SAGS，并写任务 manifest/`05_pose`/`06_sags` |
| `run_insert_batch.py` | 从 JSON job 文件串行启动多个独立单任务编排器，写 batch manifest |
| `test_gemini_edit.py` | 不联网的 Gemini API 请求/响应契约测试 |
| `test_estimate_similarity_pose.py` | 双视角 pose 和编排器的合成真值测试 |
| `test_insert_batch.py` | 无第三方依赖的 job/prompt 串行命令构建测试 |

每个任务使用独立的
`--run-root <server-root>/<scene-id> --task-id Task_001` 调用；脚本只拥有一个
任务目录，适合由外部串行调度器逐任务启动。`--output-dir` 保留兼容。
当前默认是 `--trellis-input composite`、`--render-mode anchor`、1024 分辨率；
组合流程在生成 `center.png` 后再分割并可用 `--run-sags` 产出
`06_sags/inserted_object.ply`。完整参数、Unity 文件协议和输出树见
`codex_ops/WORKFLOW.md`。

场景级批量可使用：

```bash
third_party/TRELLIS/.venv/bin/python tools/run_insert_batch.py \
  --jobs <scene>/insert_jobs.json --skip-ready
```

job 的 `prompts` 支持 `edit_default/edit_user`、`object_default/object_user`、
`anchor_default/anchor_user`；调度器会把每一对合并并写入任务目录的
`prompts.json`。首批建议 `seg_engine` 固定为 `legacy`，人工修正 mask 后再继续。
若已经得到人工 `points.json`，在任务的 `options` 中加入
`sags_points_json` 和 `skip_segmentation: true` 即可跳过自动分割。
任务也可提供 `unity_manifest`，自动读取 Unity 输出的默认/用户 prompt；job 中
显式 prompt 始终优先。

远端入口为 `ssh -p 25367 root@10.126.56.69`。需要人工修正 legacy 标注时，使用：

```bash
cd /opt/data/private/ljn/InsertAny3D/third_party/SAGS
GRADIO_ANALYTICS_ENABLED=False CUDA_VISIBLE_DEVICES=0 .venv/bin/python app_text.py
```

Gradio UI 已完成模块导入和 `Blocks` 构建测试；CLI 批处理仍使用
`run_sags_text.py`，不会启动 Web 服务。
