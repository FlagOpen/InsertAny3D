"""Select the canonical six Step 6 views without rendering anything."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


_VIEW_RE = re.compile(r"^view_(?P<index>\d{3})$")


def _pitch_dirs(root: Path) -> list[tuple[float, Path]]:
    values: list[tuple[float, Path]] = []
    for path in root.glob("pitch_*"):
        if not path.is_dir():
            continue
        try:
            values.append((float(path.name.removeprefix("pitch_")), path))
        except ValueError:
            continue
    return sorted(values)


def _view_dirs(pitch_root: Path) -> list[tuple[int, Path]]:
    values: list[tuple[int, Path]] = []
    for path in pitch_root.glob("view_*"):
        if not path.is_dir():
            continue
        match = _VIEW_RE.fullmatch(path.name)
        if match:
            values.append((int(match.group("index")), path))
    return sorted(values)


def _image_file(view_dir: Path) -> Path:
    image = view_dir / "image.png"
    if not image.is_file():
        raise FileNotFoundError(f"缺少步骤6图像: {image}")
    return image


def select_six_views(run_root: str | Path, task_id: str, *, pitches: Iterable[float] | None = None,
                     views: int | None = None) -> list[dict[str, object]]:
    """Return left/center/right at low/high pitch, validated from filenames.

    The center is view index 000. Left/right are the immediately adjacent
    circular views (last and first respectively), so this remains valid for
    arbitrary benchmark view counts. No renderer or image transformation is
    involved.
    """
    root = Path(run_root).expanduser().resolve() / task_id / "step6" / "inserted"
    available = _pitch_dirs(root)
    if len(available) < 2:
        raise ValueError(f"步骤6至少需要两个俯视角目录: {root}")
    wanted = sorted({float(value) for value in pitches}) if pitches is not None else [available[0][0], available[-1][0]]
    if len(wanted) != 2:
        raise ValueError("必须选择恰好两个俯视角（低、高）")
    by_pitch = {pitch: path for pitch, path in available}
    missing = [pitch for pitch in wanted if pitch not in by_pitch]
    if missing:
        raise ValueError(f"指定俯视角不存在: {missing}; 可用={sorted(by_pitch)}")
    selected: list[dict[str, object]] = []
    for pitch in wanted:
        pitch_root = by_pitch[pitch]
        view_list = _view_dirs(pitch_root)
        count = views if views is not None else len(view_list)
        if count < 3:
            raise ValueError(f"{pitch_root} 至少需要三个 view 目录")
        indices = {index for index, _ in view_list}
        if views is not None and indices != set(range(count)):
            raise ValueError(f"{pitch_root} 的 view 文件名必须完整覆盖 view_000 到 view_{count - 1:03d}")
        if 0 not in indices:
            raise ValueError(f"{pitch_root} 缺少中心视角 view_000")
        right = 1 if 1 in indices else min(indices - {0})
        # A remote metric staging directory may intentionally contain only
        # the selected six files. Preserve the canonical view_029 label when
        # that sparse subset is present.
        left = 29 if 29 in indices else count - 1
        if left not in indices:
            left = max(indices - {0, right})
        for label, index in (("left", left), ("center", 0), ("right", right)):
            path = next(path for value, path in view_list if value == index)
            selected.append({"label": label, "pitch": pitch, "viewIndex": index,
                             "path": str(_image_file(path))})
    return selected
