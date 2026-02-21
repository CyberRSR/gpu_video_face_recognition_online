"""
Monitor recordings_playwright, remux ready videos into in_video, enqueue recgn.py,
and maintain an HTML index for found fragments.
Python: 3.13+
"""

from __future__ import annotations

import argparse
import fnmatch
import html
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


INPUT_VIDEO_EXTENSIONS = {".mp4", ".mkv"}
FRAGMENT_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov"}
DATE_PATTERN = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})")
DISTANCE_PATTERN = re.compile(r"_d(?P<distance>\d+(?:\.\d+)?)$", re.IGNORECASE)


@dataclass(slots=True)
class Config:
    base_dir: Path
    source_dir: Path
    in_video_dir: Path
    output_dir: Path
    mask_file: Path
    queue_file: Path
    sent_file: Path
    imported_file: Path
    processed_file: Path
    html_file: Path
    stable_seconds: int
    poll_seconds: int
    process_mkv: bool
    once: bool


@dataclass(slots=True)
class Tools:
    mkvmerge: str | None
    ffmpeg: str | None


def log(message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def ensure_text_file(path: Path, initial_text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(initial_text, encoding="utf-8")


def read_non_empty_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as file:
        for raw in file:
            line = raw.strip()
            if line:
                lines.append(line)
    return lines


def append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(line.rstrip() + "\n")


def ensure_mask_file(mask_file: Path) -> None:
    if mask_file.exists():
        return
    default_content = (
        "# Include masks (fnmatch). One mask per line.\n"
        "# Example: *589518489*\n"
        "# Exclude mask: prefix with !\n"
        "# Example: !*test*\n"
        "\n"
        "*\n"
    )
    ensure_text_file(mask_file, default_content)


def load_masks(mask_file: Path) -> tuple[list[str], list[str]]:
    includes: list[str] = []
    excludes: list[str] = []
    for line in read_non_empty_lines(mask_file):
        if line.startswith("#"):
            continue
        if line.startswith("!"):
            pattern = line[1:].strip()
            if pattern:
                excludes.append(pattern)
        else:
            includes.append(line.strip())
    return includes, excludes


def matches_masks(name: str, relative_path: str, includes: list[str], excludes: list[str]) -> bool:
    targets = [name, relative_path.replace("\\", "/")]

    if includes:
        include_match = any(
            fnmatch.fnmatchcase(target, pattern)
            for target in targets
            for pattern in includes
        )
        if not include_match:
            return False

    excluded = any(
        fnmatch.fnmatchcase(target, pattern)
        for target in targets
        for pattern in excludes
    )
    return not excluded


def find_executable(candidates: Iterable[str | Path | None]) -> str | None:
    for candidate in candidates:
        if not candidate:
            continue
        candidate_str = str(candidate)
        if os.path.isabs(candidate_str):
            if os.path.exists(candidate_str):
                return candidate_str
            continue
        resolved = shutil.which(candidate_str)
        if resolved:
            return resolved
    return None


def detect_tools(base_dir: Path) -> Tools:
    mkvmerge = find_executable(
        [
            os.environ.get("MKVMERGE_PATH"),
            base_dir / "mkvtoolnix" / "mkvmerge.exe",
            r"C:\Program Files\MKVToolNix\mkvmerge.exe",
            "mkvmerge.exe",
            "mkvmerge",
        ]
    )
    ffmpeg = find_executable(
        [
            os.environ.get("FFMPEG_PATH"),
            base_dir / "ffmpeg" / "ffmpeg.exe",
            base_dir / "ffmpeg.exe",
            r"c:\inpoutp\sv_cam\ffmpeg.exe",
            "ffmpeg.exe",
            "ffmpeg",
        ]
    )
    return Tools(mkvmerge=mkvmerge, ffmpeg=ffmpeg)


def normalize_dir_names(values: Iterable[str]) -> set[str]:
    return {value.strip().lower() for value in values if value and value.strip()}


def iter_files_safe(root_dir: Path, ignored_dir_names: Iterable[str] = ()) -> Iterable[Path]:
    ignored = normalize_dir_names(ignored_dir_names)

    def on_walk_error(exc: OSError) -> None:
        bad_path = getattr(exc, "filename", "") or str(exc)
        log(f"Skip directory due to access error: {bad_path}")

    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=True, onerror=on_walk_error):
        if ignored:
            dirnames[:] = [d for d in dirnames if d.lower() not in ignored]
        for filename in filenames:
            yield Path(dirpath) / filename


def is_stable(path: Path, stable_seconds: int) -> bool:
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return False
    return age >= stable_seconds


def discover_candidates(source_dir: Path, process_mkv: bool) -> list[Path]:
    if not source_dir.exists():
        return []
    candidates: list[Path] = []
    try:
        entries = list(source_dir.iterdir())
    except OSError as exc:
        log(f"Cannot scan source dir {source_dir}: {exc}")
        return []

    for path in entries:
        try:
            if not path.is_file():
                continue
        except OSError as exc:
            log(f"Skip inaccessible path: {path} ({exc})")
            continue
        suffix = path.suffix.lower()
        if suffix not in INPUT_VIDEO_EXTENSIONS:
            continue
        if suffix == ".mkv" and not process_mkv:
            continue
        candidates.append(path)
    candidates.sort(key=lambda p: str(p).lower())
    return candidates


def choose_destination_path(source_file: Path, in_video_dir: Path) -> Path:
    in_video_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{source_file.stem}.mkv"
    candidate = in_video_dir / base_name
    if not candidate.exists():
        return candidate

    index = 1
    while True:
        candidate = in_video_dir / f"{source_file.stem}__{index}.mkv"
        if not candidate.exists():
            return candidate
        index += 1


def run_command(command: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as exc:  # pylint: disable=broad-except
        return False, str(exc)

    if result.returncode == 0:
        return True, ""
    error = (result.stderr or result.stdout or "").strip()
    if not error:
        error = f"exit code {result.returncode}"
    return False, error


def remux_to_mkv(source_file: Path, target_file: Path, tools: Tools) -> tuple[bool, str, str]:
    if not tools.mkvmerge:
        return False, "", "mkvmerge was not found. Install MKVToolNix."

    # MKVToolNix GUI uses mkvmerge under the hood; plain remux with defaults.
    mkvmerge_command = [tools.mkvmerge, "--output", str(target_file), str(source_file)]
    if target_file.exists():
        target_file.unlink(missing_ok=True)

    ok, error = run_command(mkvmerge_command)
    if ok and target_file.exists() and target_file.stat().st_size > 0:
        return True, "mkvmerge-defaults", ""

    # Optional fallback for troubleshooting only; main path must stay mkvmerge.
    if tools.ffmpeg:
        ffmpeg_command = [
            tools.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-fflags",
            "+genpts",
            "-i",
            str(source_file),
            "-map",
            "0",
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            "-y",
            str(target_file),
        ]
        if target_file.exists():
            target_file.unlink(missing_ok=True)
        ff_ok, ff_error = run_command(ffmpeg_command)
        if ff_ok and target_file.exists() and target_file.stat().st_size > 0:
            return True, "ffmpeg-fallback", ""
        return False, "", f"mkvmerge: {error or 'output not created'} | ffmpeg: {ff_error or 'output not created'}"

    return False, "", f"mkvmerge: {error or 'output not created'}"


def import_to_in_video(source_file: Path, target_file: Path, tools: Tools) -> tuple[bool, str, str]:
    suffix = source_file.suffix.lower()
    if suffix in {".mp4", ".mkv"}:
        return remux_to_mkv(source_file, target_file, tools)

    return False, "", f"Unsupported extension: {suffix}"


def load_imported_sources(imported_file: Path) -> set[str]:
    imported: set[str] = set()
    for line in read_non_empty_lines(imported_file):
        parts = line.split("\t")
        if len(parts) >= 2:
            imported.add(parts[1])
        else:
            imported.add(parts[0])
    return imported


def process_recordings_cycle(config: Config, tools: Tools) -> tuple[int, int]:
    includes, excludes = load_masks(config.mask_file)
    imported_sources = load_imported_sources(config.imported_file)
    known_targets = set(read_non_empty_lines(config.queue_file)) | set(
        read_non_empty_lines(config.processed_file)
    )

    imported_count = 0
    queued_count = 0
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for source_file in discover_candidates(config.source_dir, config.process_mkv):
        source_key = str(source_file.resolve())
        if source_key in imported_sources:
            continue
        if not is_stable(source_file, config.stable_seconds):
            continue

        try:
            relative_path = source_file.relative_to(config.source_dir).as_posix()
        except ValueError:
            relative_path = source_file.name

        if not matches_masks(source_file.name, relative_path, includes, excludes):
            continue

        target_file = choose_destination_path(source_file, config.in_video_dir)
        ok, method, error = import_to_in_video(source_file, target_file, tools)
        if not ok:
            log(f"Import failed: {source_file} -> {target_file} | {error}")
            continue

        imported_count += 1
        imported_sources.add(source_key)
        append_line(
            config.imported_file,
            f"{now_str}\t{source_key}\t{target_file.resolve()}\t{method}",
        )

        target_key = str(target_file.resolve())
        if target_key in known_targets:
            continue

        append_line(config.queue_file, target_key)
        append_line(config.sent_file, f"{now_str}\t{target_key}\t{source_key}")
        known_targets.add(target_key)
        queued_count += 1
        log(f"Queued: {target_file.name} (from {source_file.name}, {method})")

    return imported_count, queued_count


def extract_fragment_distance(file_stem: str) -> float | None:
    match = DISTANCE_PATTERN.search(file_stem)
    if match:
        return float(match.group("distance"))
    return None


def extract_fragment_datetime(path: Path) -> datetime:
    candidates = [path.parent.name, path.stem, path.name]
    for candidate in candidates:
        match = DATE_PATTERN.search(candidate)
        if match:
            value = f"{match.group('date')}_{match.group('time')}"
            try:
                return datetime.strptime(value, "%Y-%m-%d_%H-%M-%S")
            except ValueError:
                pass
    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        return datetime.now()


def collect_fragments(output_dir: Path) -> tuple[list[dict[str, object]], tuple[int, int]]:
    rows: list[dict[str, object]] = []
    count = 0
    newest_mtime = 0

    if not output_dir.exists():
        return rows, (count, newest_mtime)

    for path in iter_files_safe(output_dir):
        try:
            if not path.is_file():
                continue
        except OSError:
            continue
        if path.suffix.lower() not in FRAGMENT_VIDEO_EXTENSIONS:
            continue
        if "img_jpg" in path.parts:
            continue

        try:
            stat = path.stat()
            resolved = path.resolve()
        except OSError:
            continue

        count += 1
        newest_mtime = max(newest_mtime, int(stat.st_mtime))
        dt = extract_fragment_datetime(path)
        distance = extract_fragment_distance(path.stem)

        rows.append(
            {
                "name": path.name,
                "name_lower": path.name.lower(),
                "distance": distance,
                "distance_text": f"{distance:.2f}" if distance is not None else "",
                "date": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "date_ts": int(dt.timestamp()),
                "source": path.parent.name,
                "source_lower": path.parent.name.lower(),
                "size_mb": stat.st_size / (1024 * 1024),
                "uri": resolved.as_uri(),
                "absolute_path": str(resolved),
            }
        )

    rows.sort(key=lambda row: int(row["date_ts"]), reverse=True)
    return rows, (count, newest_mtime)


def render_html(rows: list[dict[str, object]], html_file: Path) -> None:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row_html: list[str] = []

    for row in rows:
        name = html.escape(str(row["name"]))
        name_lower = html.escape(str(row["name_lower"]), quote=True)
        source = html.escape(str(row["source"]))
        source_lower = html.escape(str(row["source_lower"]), quote=True)
        distance = html.escape(str(row["distance_text"]))
        distance_attr = "" if row["distance"] is None else f"{float(row['distance']):.6f}"
        date_text = html.escape(str(row["date"]))
        date_ts = int(row["date_ts"])
        size_text = f"{float(row['size_mb']):.2f}"
        uri = html.escape(str(row["uri"]), quote=True)
        abs_path = html.escape(str(row["absolute_path"]), quote=True)

        row_html.append(
            f"<tr data-name=\"{name_lower}\" data-source=\"{source_lower}\" "
            f"data-distance=\"{distance_attr}\" data-date=\"{date_ts}\">"
            f"<td><a href=\"{uri}\" title=\"{abs_path}\">{name}</a></td>"
            f"<td>{distance}</td>"
            f"<td>{date_text}</td>"
            f"<td>{source}</td>"
            f"<td>{size_text}</td>"
            "</tr>"
        )

    html_text = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>found_fragments_colored_ index</title>
  <style>
    :root {{
      --bg: #f2f4f8;
      --panel: #ffffff;
      --text: #0f172a;
      --accent: #0f766e;
      --line: #d7dce5;
    }}
    body {{
      margin: 0;
      padding: 20px;
      font-family: "Segoe UI", Tahoma, sans-serif;
      background: linear-gradient(140deg, #e8eef8 0%, #f7fafc 55%, #e7f5ef 100%);
      color: var(--text);
    }}
    .panel {{
      max-width: 1400px;
      margin: 0 auto;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: 0 10px 26px rgba(15, 23, 42, 0.08);
      overflow: hidden;
    }}
    .header {{
      padding: 16px 20px 10px;
      border-bottom: 1px solid var(--line);
      background: #f8fafb;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 20px;
    }}
    .meta {{
      font-size: 13px;
      color: #475569;
    }}
    .filters {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      padding: 12px 20px;
      border-bottom: 1px solid var(--line);
      background: #ffffff;
    }}
    label {{
      display: block;
      font-size: 12px;
      color: #475569;
      margin-bottom: 4px;
    }}
    input {{
      width: 100%;
      box-sizing: border-box;
      border: 1px solid #bcc5d3;
      border-radius: 8px;
      padding: 8px;
      font-size: 13px;
    }}
    button {{
      border: 1px solid #0d9488;
      background: var(--accent);
      color: #fff;
      border-radius: 8px;
      padding: 8px 12px;
      cursor: pointer;
      margin-top: 18px;
      font-size: 13px;
    }}
    .table-wrap {{
      overflow: auto;
      max-height: calc(100vh - 250px);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 900px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      text-align: left;
      padding: 9px 10px;
      font-size: 13px;
      white-space: nowrap;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #f1f5f9;
      z-index: 1;
      user-select: none;
    }}
    th.sortable {{
      cursor: pointer;
    }}
    th.sortable::after {{
      content: " ^v";
      color: #64748b;
    }}
    tr:hover {{
      background: #f8fafc;
    }}
    a {{
      color: #0369a1;
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
  </style>
</head>
<body>
  <div class="panel">
    <div class="header">
      <h1>found_fragments_colored_</h1>
      <div class="meta">Rows: <span id="visibleCount">{len(rows)}</span> / {len(rows)} | Generated: {generated_at}</div>
    </div>
    <div class="filters">
      <div>
        <label for="filterName">File name contains</label>
        <input id="filterName" type="text" placeholder="part of file name">
      </div>
      <div>
        <label for="filterSource">Source folder contains</label>
        <input id="filterSource" type="text" placeholder="camera or source folder">
      </div>
      <div>
        <label for="filterDistMin">Distance min</label>
        <input id="filterDistMin" type="number" step="0.01" placeholder="0.00">
      </div>
      <div>
        <label for="filterDistMax">Distance max</label>
        <input id="filterDistMax" type="number" step="0.01" placeholder="1.00">
      </div>
      <div>
        <label for="filterDateFrom">Date from</label>
        <input id="filterDateFrom" type="datetime-local">
      </div>
      <div>
        <label for="filterDateTo">Date to</label>
        <input id="filterDateTo" type="datetime-local">
      </div>
      <div>
        <button id="resetFilters" type="button">Reset filters</button>
      </div>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th class="sortable" data-key="name">File</th>
            <th class="sortable" data-key="distance">Distance</th>
            <th class="sortable" data-key="date">Date</th>
            <th class="sortable" data-key="source">Source</th>
            <th class="sortable" data-key="size">Size MB</th>
          </tr>
        </thead>
        <tbody id="rows">
          {''.join(row_html)}
        </tbody>
      </table>
    </div>
  </div>
<script>
  const rows = Array.from(document.querySelectorAll('#rows tr'));
  let sortKey = '';
  let sortAsc = true;

  function numberOrNull(value) {{
    const n = parseFloat(value);
    return Number.isFinite(n) ? n : null;
  }}

  function updateVisibleCount() {{
    const visible = rows.filter(row => row.style.display !== 'none').length;
    document.getElementById('visibleCount').textContent = visible.toString();
  }}

  function applyFilters() {{
    const nameFilter = document.getElementById('filterName').value.trim().toLowerCase();
    const sourceFilter = document.getElementById('filterSource').value.trim().toLowerCase();
    const distMin = numberOrNull(document.getElementById('filterDistMin').value);
    const distMax = numberOrNull(document.getElementById('filterDistMax').value);
    const dateFromRaw = Date.parse(document.getElementById('filterDateFrom').value);
    const dateToRaw = Date.parse(document.getElementById('filterDateTo').value);
    const dateFrom = Number.isFinite(dateFromRaw) ? Math.floor(dateFromRaw / 1000) : null;
    const dateTo = Number.isFinite(dateToRaw) ? Math.floor(dateToRaw / 1000) : null;

    for (const row of rows) {{
      const rowName = row.dataset.name || '';
      const rowSource = row.dataset.source || '';
      const rowDist = numberOrNull(row.dataset.distance || '');
      const rowDate = parseInt(row.dataset.date || '0', 10);
      let visible = true;

      if (nameFilter && !rowName.includes(nameFilter)) visible = false;
      if (sourceFilter && !rowSource.includes(sourceFilter)) visible = false;
      if (distMin !== null && (rowDist === null || rowDist < distMin)) visible = false;
      if (distMax !== null && (rowDist === null || rowDist > distMax)) visible = false;
      if (dateFrom !== null && rowDate < dateFrom) visible = false;
      if (dateTo !== null && rowDate > dateTo) visible = false;

      row.style.display = visible ? '' : 'none';
    }}
    updateVisibleCount();
  }}

  function compareRows(a, b, key) {{
    if (key === 'name') {{
      return (a.dataset.name || '').localeCompare(b.dataset.name || '');
    }}
    if (key === 'source') {{
      return (a.dataset.source || '').localeCompare(b.dataset.source || '');
    }}
    if (key === 'distance') {{
      const da = numberOrNull(a.dataset.distance || '');
      const db = numberOrNull(b.dataset.distance || '');
      if (da === null && db === null) return 0;
      if (da === null) return 1;
      if (db === null) return -1;
      return da - db;
    }}
    if (key === 'size') {{
      const sa = parseFloat(a.cells[4].textContent || '0');
      const sb = parseFloat(b.cells[4].textContent || '0');
      return sa - sb;
    }}
    const ta = parseInt(a.dataset.date || '0', 10);
    const tb = parseInt(b.dataset.date || '0', 10);
    return ta - tb;
  }}

  function sortBy(key) {{
    if (sortKey === key) {{
      sortAsc = !sortAsc;
    }} else {{
      sortKey = key;
      sortAsc = key === 'distance' || key === 'name' || key === 'source';
    }}

    rows.sort((a, b) => {{
      const result = compareRows(a, b, key);
      return sortAsc ? result : -result;
    }});

    const tbody = document.getElementById('rows');
    for (const row of rows) {{
      tbody.appendChild(row);
    }}
    applyFilters();
  }}

  document.querySelectorAll('th.sortable').forEach((th) => {{
    th.addEventListener('click', () => sortBy(th.dataset.key));
  }});

  ['filterName', 'filterSource', 'filterDistMin', 'filterDistMax', 'filterDateFrom', 'filterDateTo']
    .forEach((id) => {{
      document.getElementById(id).addEventListener('input', applyFilters);
    }});

  document.getElementById('resetFilters').addEventListener('click', () => {{
    ['filterName', 'filterSource', 'filterDistMin', 'filterDistMax', 'filterDateFrom', 'filterDateTo']
      .forEach((id) => {{
        document.getElementById(id).value = '';
      }});
    applyFilters();
  }});

  sortBy('distance');
</script>
</body>
</html>
"""
    html_file.write_text(html_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Watch recordings_playwright, remux queue inputs, build fragment HTML index."
    )
    parser.add_argument("--base-dir", default=str(script_dir), help="Base working directory.")
    parser.add_argument(
        "--stable-seconds",
        type=int,
        default=600,
        help="Minimal unchanged age for source file before processing.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=30,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one cycle and exit.",
    )
    parser.add_argument(
        "--no-mkv-source",
        action="store_true",
        help="Disable direct import of source .mkv files (only process .mp4).",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    base_dir = Path(args.base_dir).resolve()
    return Config(
        base_dir=base_dir,
        source_dir=base_dir / "recordings_playwright",
        in_video_dir=base_dir / "in_video",
        output_dir=base_dir / "found_fragments_colored_",
        mask_file=base_dir / "recording_name_masks.txt",
        queue_file=base_dir / "recgn_queue.txt",
        sent_file=base_dir / "recgn_sent_files.txt",
        imported_file=base_dir / "watcher_imported_sources.txt",
        processed_file=base_dir / "recgn_processed_files.txt",
        html_file=base_dir / "found_fragments_colored_index.html",
        stable_seconds=max(0, int(args.stable_seconds)),
        poll_seconds=max(1, int(args.poll_seconds)),
        process_mkv=not bool(args.no_mkv_source),
        once=bool(args.once),
    )


def bootstrap_files(config: Config) -> None:
    config.in_video_dir.mkdir(parents=True, exist_ok=True)
    ensure_mask_file(config.mask_file)
    ensure_text_file(config.queue_file)
    ensure_text_file(config.sent_file)
    ensure_text_file(config.imported_file)
    ensure_text_file(config.processed_file)


def main() -> int:
    args = parse_args()
    config = build_config(args)
    bootstrap_files(config)
    tools = detect_tools(config.base_dir)

    log(f"source_dir={config.source_dir}")
    log(f"in_video_dir={config.in_video_dir}")
    log(f"queue_file={config.queue_file}")
    log(f"processed_file={config.processed_file}")
    log(f"stable_seconds={config.stable_seconds}, poll_seconds={config.poll_seconds}")
    log(f"mkvmerge={tools.mkvmerge or 'not found'}")
    log(f"ffmpeg={tools.ffmpeg or 'not found'}")

    last_signature: tuple[int, int] | None = None

    while True:
        imported_count, queued_count = process_recordings_cycle(config, tools)
        if imported_count or queued_count:
            log(f"cycle: imported={imported_count}, queued={queued_count}")

        rows, signature = collect_fragments(config.output_dir)
        if signature != last_signature:
            render_html(rows, config.html_file)
            last_signature = signature
            log(f"HTML updated: {config.html_file} (rows={len(rows)})")

        if config.once:
            return 0
        time.sleep(config.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
