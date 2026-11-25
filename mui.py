"""MUI：跟 LLM 對話來分析 / 標註 / 繪製圖片的 Gradio 介面（目前只有前端與操作流程，後端為 stub）。

操作流程（範例）：
    1. 左邊資料夾選圖
    2. 「幫我把圓心都找出來」 → LLM 產生程式碼 → 執行 → 產生圖層 circles（x, y, r）
    3. 「把圓心連線並測量距離」 → 產生圖層 distances（kind=line, x1..y2, length_px, label）
    4. 「用綠色大小 8 標記出來」 → LLM 產生繪圖程式碼 → 圖片更新
    5. 「匯出」分頁輸入檔名 → 輸出到固定資料夾：<名稱>.png + <名稱>_<圖層>.csv
    程式碼執行失敗時，會把錯誤丟回給 LLM 修正後重跑（次數可設定）。

用法：
    python mui.py --host 0.0.0.0 --port 7860 --root ./data --output ./data/output
"""

import argparse
import contextlib
import csv
import io
import os
import re
import tempfile
import traceback

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image, ImageColor, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff"}
ANNOT_EXTS = {".txt", ".csv"}
LABEL_COLUMNS = ("label", "class", "name", "category", "cls")
PALETTE = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
           "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff"]

FORMATS = [
    "自動判斷",
    "點：x y [label]",
    "圓：x y r [label]",
    "框：x1 y1 x2 y2 [label]",
    "框：x y w h [label]",
    "線：x1 y1 x2 y2 [label]",
    "YOLO 框：cls cx cy w h（正規化）",
    "多邊形 / YOLO-seg：cls x1 y1 x2 y2 ...",
    "Mask 圖片",
]
F_AUTO, F_POINT, F_CIRCLE, F_XYXY, F_XYWH, F_LINE, F_YOLO, F_POLY, F_MASK = FORMATS
COLOR_MODES = ["依圖層自動上色", "單一顏色", "依 label 自動上色"]

ROOT_DIR = "."      # --root：瀏覽圖片的根目錄
OUTPUT_DIR = "."    # --output：匯出的固定資料夾
WORK_DIR = tempfile.mkdtemp(prefix="mui_")  # 程式碼產生的中間標註檔


# ---------------------------------------------------------------------------
# 路徑工具
# ---------------------------------------------------------------------------

def _inside(path, root):
    return os.path.commonpath([os.path.abspath(path), root]) == root


def resolve_in_root(path):
    path = os.path.abspath(os.path.join(ROOT_DIR, os.path.expanduser(path or "")))
    if not _inside(path, ROOT_DIR):
        raise gr.Error(f"路徑必須在根目錄底下：{ROOT_DIR}")
    return path


def safe_name(name, default="untitled"):
    name = re.sub(r"[^\w\-.]+", "_", str(name or "").strip()).strip("._")
    return name or default


def find_annotations(image_path):
    """找跟圖片同名的標註檔：同資料夾的 stem.txt / stem.csv / stem_mask.*，以及 YOLO 的 ../labels/stem.txt。"""
    folder, name = os.path.split(image_path)
    stem = os.path.splitext(name)[0]
    candidates = [os.path.join(folder, stem + ext) for ext in sorted(ANNOT_EXTS)]
    candidates += [os.path.join(folder, f"{stem}_mask{ext}") for ext in sorted(IMAGE_EXTS)]
    candidates.append(os.path.join(os.path.dirname(folder), "labels", stem + ".txt"))
    return [os.path.abspath(p) for p in candidates if os.path.isfile(p) and _inside(p, ROOT_DIR)]


# ---------------------------------------------------------------------------
# 標註解析
#   shape  = {"kind": point|circle|box|line|polygon|mask, "label": str|None, "coords": [...] 或 "mask": ndarray}
#   layer  = {"df": 表格(DataFrame), "shapes": [shape], "path": csv 路徑或 None}
#   annots = {"layers": {名稱: layer}, "current": 目前圖層名稱}
# ---------------------------------------------------------------------------

def _is_num(s):
    try:
        return not np.isnan(float(s))
    except (TypeError, ValueError):
        return False


def _fmt_num(v):
    return str(int(v)) if float(v).is_integer() else str(v)


def _read_rows(path):
    with open(path, encoding="utf-8-sig") as f:
        text = f.read()
    if path.lower().endswith(".csv"):
        rows = list(csv.reader(io.StringIO(text)))
    else:
        rows = [line.replace(",", " ").split() for line in text.splitlines()]
    rows = [[c.strip() for c in r] for r in rows]
    rows = [r for r in rows if any(r) and not r[0].startswith("#")]
    header = None
    if rows and not any(_is_num(c) for c in rows[0]):
        header = [c.lower() for c in rows[0]]
        rows = rows[1:]
    return header, rows


def _record_to_shape(rec):
    """一筆有欄位名稱的資料 → shape。若有 kind 欄位就依它決定，否則依欄位推斷。"""
    def has(*keys):
        return all(_is_num(rec.get(k)) for k in keys)

    def first(*keys):
        return next((k for k in keys if _is_num(rec.get(k))), None)

    label = next((rec[c] for c in LABEL_COLUMNS if c in rec and rec[c] not in (None, "") and str(rec[c]) != "nan"), None)
    label = None if label is None else (_fmt_num(label) if _is_num(label) else str(label))
    kind = str(rec.get("kind") or "").strip().lower() or None
    if kind == "nan":
        kind = None

    k1, k2 = first("x1", "xmin", "left"), first("y1", "ymin", "top")
    k3, k4 = first("x2", "xmax", "right"), first("y2", "ymax", "bottom")
    if kind == "polygon" or (kind is None and rec.get("points")):
        nums = [float(v) for v in str(rec.get("points", "")).replace(",", " ").split()]
        return {"kind": "polygon", "label": label, "coords": nums}
    if None not in (k1, k2, k3, k4) and kind in (None, "box", "line"):
        return {"kind": kind or "box", "label": label, "coords": [float(rec[k]) for k in (k1, k2, k3, k4)]}
    x, y = first("x", "cx", "x_center"), first("y", "cy", "y_center")
    if x and y:
        X, Y = float(rec[x]), float(rec[y])
        w, h, r = first("w", "width"), first("h", "height"), first("r", "radius")
        if w and h and kind in (None, "box"):
            return {"kind": "box", "label": label, "coords": [X, Y, X + float(rec[w]), Y + float(rec[h])]}
        if r and kind in (None, "circle"):
            return {"kind": "circle", "label": label, "coords": [X, Y, float(rec[r])]}
        return {"kind": "point", "label": label, "coords": [X, Y]}
    raise gr.Error(f"看不懂這筆資料：{rec}（支援 x,y / x,y,r / x1,y1,x2,y2 / x,y,w,h / kind 欄位）")


def records_to_shapes(header, rows):
    header = [str(h).strip().lower() for h in header]
    return [_record_to_shape(dict(zip(header, row))) for row in rows]


def df_to_shapes(df):
    return [] if df is None or df.empty else records_to_shapes(list(df.columns), df.values.tolist())


def _parse_row(tokens, fmt):
    """解析沒有表頭的一列。"""
    tokens = [t for t in tokens if t]
    label = None
    if not _is_num(tokens[0]):
        label, tokens = tokens[0], tokens[1:]
    elif not _is_num(tokens[-1]):
        label, tokens = tokens[-1], tokens[:-1]
    nums = [float(t) for t in tokens]
    n = len(nums)

    if fmt == F_AUTO:
        normalized = n > 1 and all(0 <= v <= 1 for v in nums[1:])
        if label is None and n == 5 and normalized:
            fmt = F_YOLO
        elif label is None and n >= 7 and n % 2 == 1:
            fmt = F_POLY
        elif n in (2, 3):
            fmt = F_POINT
        elif n in (4, 5):
            fmt = F_XYXY
        elif n >= 6:
            fmt = F_POLY
        else:
            raise gr.Error(f"無法判斷格式：{tokens}")

    if fmt == F_POINT:
        if label is None and n >= 3:
            label = _fmt_num(nums[2])
        return {"kind": "point", "label": label, "coords": nums[:2]}
    if fmt == F_CIRCLE:
        if label is None and n >= 4:
            label = _fmt_num(nums[3])
        return {"kind": "circle", "label": label, "coords": nums[:3]}
    if fmt in (F_XYXY, F_XYWH, F_LINE):
        if label is None and n == 5:
            label, nums = _fmt_num(nums[0]), nums[1:]
        a, b, c, d = nums[:4]
        coords = [a, b, a + c, b + d] if fmt == F_XYWH else [a, b, c, d]
        return {"kind": "line" if fmt == F_LINE else "box", "label": label, "coords": coords}
    if fmt == F_YOLO:
        lab = label or _fmt_num(nums[0])
        cx, cy, w, h = nums[1:5] if label is None else nums[:4]
        return {"kind": "box", "label": lab, "coords": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]}
    if fmt == F_POLY:
        if label is None and n % 2 == 1:
            label, nums = _fmt_num(nums[0]), nums[1:]
        return {"kind": "polygon", "label": label, "coords": nums}
    raise gr.Error(f"格式「{fmt}」不適用於文字檔")


def _scale_if_normalized(shapes, w, h):
    """若所有座標都在 0~1，就視為正規化座標並換算成像素。"""
    vals = [v for s in shapes if s["kind"] != "mask" for v in s["coords"]]
    if vals and all(0 <= v <= 1 for v in vals):
        for s in shapes:
            if s["kind"] != "mask":
                s["coords"] = [v * (w if i % 2 == 0 else h) for i, v in enumerate(s["coords"])]
    return shapes


def parse_annotation(path, fmt, size):
    w, h = size
    ext = os.path.splitext(path)[1].lower()
    if fmt == F_MASK or ext in IMAGE_EXTS:
        mask = np.array(Image.open(path).convert("L").resize((w, h), Image.NEAREST))
        return [{"kind": "mask", "label": str(v), "mask": mask == v} for v in np.unique(mask) if v != 0]
    header, rows = _read_rows(path)
    shapes = records_to_shapes(header, rows) if header else [_parse_row(r, fmt) for r in rows]
    return _scale_if_normalized(shapes, w, h)


def shapes_to_df(shapes):
    rows = []
    for s in shapes:
        row, c = {"kind": s["kind"], "label": s.get("label")}, s.get("coords", [])
        if s["kind"] == "point":
            row.update(x=c[0], y=c[1])
        elif s["kind"] == "circle":
            row.update(x=c[0], y=c[1], r=c[2])
        elif s["kind"] in ("box", "line"):
            row.update(x1=c[0], y1=c[1], x2=c[2], y2=c[3])
        elif s["kind"] == "polygon":
            row.update(points=" ".join(f"{v:.1f}" for v in c))
        elif s["kind"] == "mask":
            row.update(pixels=int(s["mask"].sum()))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 圖層
# ---------------------------------------------------------------------------

def empty_annots():
    return {"layers": {}, "current": None}


def add_layer(annots, name, df=None, shapes=None, path=None):
    name = safe_name(name, "layer")
    if df is None:
        df = shapes_to_df(shapes)
    if shapes is None:
        shapes = df_to_shapes(df)
    layers = {**annots["layers"], name: {"df": df, "shapes": shapes, "path": path}}
    return {"layers": layers, "current": name}


def current_layer(annots):
    return annots["layers"].get(annots["current"]) if annots["current"] else None


def layer_summary(name, layer):
    kinds = pd.Series([s["kind"] for s in layer["shapes"]]).value_counts().to_dict()
    return f"**{name}**（{len(layer['shapes'])} 筆：" + "、".join(f"{k} {v}" for k, v in kinds.items()) + "）"


def annots_summary(annots):
    if not annots["layers"]:
        return "目前沒有標註圖層"
    return "圖層：" + "；".join(layer_summary(n, l) for n, l in annots["layers"].items())


def load_annotation_files(annots, paths, fmt, size):
    msgs = []
    for p in paths:
        try:
            shapes = parse_annotation(p, fmt, size)
        except gr.Error:
            raise
        except Exception as e:  # noqa: BLE001
            raise gr.Error(f"解析 {os.path.basename(p)} 失敗：{e}")
        name = os.path.splitext(os.path.basename(p))[0]
        is_table = p.lower().endswith((".csv", ".txt"))
        annots = add_layer(annots, name, shapes=shapes, path=p if is_table else None)
        msgs.append(f"{name}：{len(shapes)} 筆")
    return annots, "；".join(msgs)


# ---------------------------------------------------------------------------
# 繪製
# ---------------------------------------------------------------------------

def _rgb(color, fallback="#ff0000"):
    try:
        return ImageColor.getrgb(color)[:3]
    except (ValueError, TypeError, AttributeError):
        m = re.findall(r"[\d.]+", str(color))
        return tuple(int(float(v)) for v in m[:3]) if len(m) >= 3 else ImageColor.getrgb(fallback)[:3]


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def draw(img, shapes, color="#ff0000", width=3, point_size=5, opacity=1.0, by_label=False,
         show_label=False, font_size=None):
    """把標註畫到圖片上並回傳新圖。shapes 可以是 shapes list 或 DataFrame。

    程式碼裡的用法：
        img = draw(img, layers["circles"], color="lime", point_size=8)
        img = draw(img, layers["distances"], color="yellow", width=2, show_label=True)
    """
    if isinstance(shapes, pd.DataFrame):
        shapes = df_to_shapes(shapes)
    base = img.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    width, point_size = int(width), int(point_size)
    font = _font(int(font_size or max(14, min(base.size) // 40)))
    labels = sorted({str(s["label"]) for s in shapes if s.get("label") is not None})
    a_line, a_fill = int(255 * opacity), int(90 * opacity)

    for s in shapes:
        use_palette = by_label and s.get("label") is not None
        rgb = _rgb(PALETTE[labels.index(str(s["label"])) % len(PALETTE)]) if use_palette else _rgb(color)
        anchor, text_anchor = None, "lb"
        if s["kind"] in ("point", "circle"):
            x, y = s["coords"][:2]
            p = point_size
            d.ellipse([x - p, y - p, x + p, y + p], fill=rgb + (a_line,))
            anchor = (x + p, y - p)
            if s["kind"] == "circle":
                r = s["coords"][2]
                d.ellipse([x - r, y - r, x + r, y + r], outline=rgb + (a_line,), width=width)
                anchor = (x - r, y - r)
        elif s["kind"] == "box":
            x1, y1, x2, y2 = s["coords"]
            d.rectangle([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], outline=rgb + (a_line,), width=width)
            anchor = (min(x1, x2), min(y1, y2))
        elif s["kind"] == "line":
            x1, y1, x2, y2 = s["coords"]
            d.line([(x1, y1), (x2, y2)], fill=rgb + (a_line,), width=width)
            anchor, text_anchor = ((x1 + x2) / 2, (y1 + y2) / 2), "mm"
        elif s["kind"] == "polygon":
            pts = list(zip(s["coords"][0::2], s["coords"][1::2]))
            if len(pts) >= 2:
                d.polygon(pts, fill=rgb + (a_fill,), outline=rgb + (a_line,), width=width)
                anchor = pts[0]
        elif s["kind"] == "mask":
            m = Image.fromarray((s["mask"] * int(160 * opacity)).astype(np.uint8))
            layer.paste(Image.new("RGBA", base.size, rgb + (255,)), (0, 0), m)
            ys, xs = np.nonzero(s["mask"])
            anchor = (int(xs.min()), int(ys.min())) if len(xs) else None
        if show_label and anchor and s.get("label") is not None:
            text = str(s["label"])
            l, t, r_, b = d.textbbox(anchor, text, font=font, anchor=text_anchor)
            d.rectangle([l - 3, t - 2, r_ + 3, b + 2], fill=(0, 0, 0, int(170 * opacity)))
            d.text(anchor, text, fill=rgb + (255,), font=font, anchor=text_anchor)

    return Image.alpha_composite(base, layer).convert("RGB")


# ---------------------------------------------------------------------------
# 編輯步驟：{"path": 原圖路徑, "steps": [(名稱, PIL.Image), ...], "idx": 目前步驟}
# ---------------------------------------------------------------------------

def empty_state():
    return {"path": None, "steps": [], "idx": -1}


def current_image(state):
    if not state or state["idx"] < 0:
        return None
    return state["steps"][state["idx"]][1]


def push_step(state, name, img):
    steps = state["steps"][: state["idx"] + 1] + [(name, img)]
    return {**state, "steps": steps, "idx": len(steps) - 1}


def require_image(state):
    img = current_image(state)
    if img is None:
        raise gr.Error("請先從左邊資料夾選一張圖片")
    return img


# ---------------------------------------------------------------------------
# 畫面更新（每個 render_* 回傳的順序要對應 build_ui 裡的 outputs 清單）
# ---------------------------------------------------------------------------

def render(state):
    """→ viewer, steps"""
    choices = [f"{i}. {name}" for i, (name, _) in enumerate(state["steps"])]
    return current_image(state), gr.update(choices=choices, value=choices[state["idx"]] if choices else None)


def render_annots(annots):
    """→ annot_info, layer_dd, annot_table, annot_file, draw_layers, export_layers"""
    names = list(annots["layers"])
    layer = current_layer(annots)
    return (
        annots_summary(annots),
        gr.update(choices=names, value=annots["current"]),
        layer["df"] if layer else pd.DataFrame(),
        layer["path"] if layer else None,
        gr.update(choices=names, value=names),
        gr.update(choices=names, value=names),
    )


# ---------------------------------------------------------------------------
# 事件：資料夾
# ---------------------------------------------------------------------------

def open_folder(path):
    folder = resolve_in_root(path)
    if not os.path.isdir(folder):
        raise gr.Error(f"資料夾不存在：{folder}")
    entries = sorted(os.listdir(folder))
    subdirs = [e for e in entries if os.path.isdir(os.path.join(folder, e)) and not e.startswith(".")]
    images = [os.path.join(folder, e) for e in entries if os.path.splitext(e)[1].lower() in IMAGE_EXTS]
    rel = os.path.relpath(folder, ROOT_DIR)
    return (
        folder,
        gr.update(value=[(p, os.path.basename(p)) for p in images], label=f"圖片（{len(images)} 張）· {rel}"),
        images,
        gr.update(choices=subdirs, value=None),
    )


def go_up(path):
    folder = resolve_in_root(path)
    return open_folder(folder if folder == ROOT_DIR else os.path.dirname(folder))


def enter_subdir(path, sub):
    return open_folder(os.path.join(resolve_in_root(path), sub) if sub else path)


def select_image(paths, auto_annot, evt: gr.SelectData):
    path = paths[evt.index]
    img = Image.open(path).convert("RGB")
    state = push_step({"path": path, "steps": [], "idx": -1}, "原圖", img)
    info = f"**{os.path.relpath(path, ROOT_DIR)}** · {img.width}×{img.height}"
    annots, files = empty_annots(), find_annotations(path) if auto_annot else []
    if files:
        try:
            annots, msg = load_annotation_files(annots, files, F_AUTO, img.size)
            info += f" · 已載入同名標註（{msg}）"
        except gr.Error as e:
            info += f" · 同名標註解析失敗：{e}"
    stem = os.path.splitext(os.path.basename(path))[0]
    return (state, *render(state), annots, *render_annots(annots), info, f"{stem}_result", files or None)


# ---------------------------------------------------------------------------
# 事件：步驟
# ---------------------------------------------------------------------------

def jump_step(state, choice):
    if choice:
        state = {**state, "idx": int(choice.split(".", 1)[0])}
    return (state, *render(state))


def undo(state):
    if state["idx"] > 0:
        state = {**state, "idx": state["idx"] - 1}
    return (state, *render(state))


def redo(state):
    if state["idx"] < len(state["steps"]) - 1:
        state = {**state, "idx": state["idx"] + 1}
    return (state, *render(state))


def reset(state):
    if state["steps"]:
        state = push_step(state, "還原原圖", state["steps"][0][1])
    return (state, *render(state))


# ---------------------------------------------------------------------------
# 事件：圖層 / 疊加
# ---------------------------------------------------------------------------

def _file_paths(files):
    if not files:
        return []
    files = files if isinstance(files, list) else [files]
    return [f if isinstance(f, str) else getattr(f, "name", str(f)) for f in files]


def select_layer(annots, name):
    if name in annots["layers"]:
        annots = {**annots, "current": name}
    return (annots, *render_annots(annots))


def delete_layer(annots, name):
    layers = {k: v for k, v in annots["layers"].items() if k != name}
    annots = {"layers": layers, "current": next(reversed(layers), None) if layers else None}
    return (annots, *render_annots(annots))


def load_files(state, annots, files, fmt):
    img = require_image(state)
    paths = _file_paths(files)
    if not paths:
        raise gr.Error("請先上傳標註檔（txt / csv / mask 圖片）")
    annots, msg = load_annotation_files(annots, paths, fmt, img.size)
    return (annots, *render_annots(annots), "已載入 — " + msg)


def apply_overlay(state, annots, selected, mode, color, width, point_size, opacity, show_label):
    img = require_image(state)
    selected = [n for n in (selected or []) if n in annots["layers"]]
    if not selected:
        raise gr.Error("請勾選要疊加的圖層（先請 LLM 產生，或在「標註資料」分頁載入檔案）")
    names = list(annots["layers"])
    for name in selected:
        c = PALETTE[names.index(name) % len(PALETTE)] if mode == COLOR_MODES[0] else color
        img = draw(img, annots["layers"][name]["shapes"], c, width, point_size, float(opacity),
                   by_label=mode == COLOR_MODES[2], show_label=show_label)
    state = push_step(state, "疊加 " + "+".join(selected), img)
    return (state, *render(state))


# ---------------------------------------------------------------------------
# 執行程式碼
# ---------------------------------------------------------------------------

CODE_HELP = """**可用變數**：`img` 目前圖片（PIL.Image）、`layers` 所有圖層 `{名稱: DataFrame}`、`table` / `overlays` 目前圖層的表格 / shapes、`image_path`
**可用函式**：`draw(img, layers["名稱"], color=, width=, point_size=, opacity=, by_label=, show_label=, font_size=)`；模組 `np`, `pd`, `Image`, `ImageDraw`, `ImageFilter`, `ImageOps`, `ImageEnhance`
**產出**：改圖 → 把結果指定回 `img`；新增/更新圖層 → `annotations = DataFrame` 加上 `annotation_name = "名稱"`，或 `annotations = {"名稱": DataFrame, ...}`
**欄位**：點 `x,y`、圓 `x,y,r`、框 `x1,y1,x2,y2` 或 `x,y,w,h`、線 `kind=line,x1,y1,x2,y2`，可加 `label` 和任何額外欄位（例如 `length_px`）
⚠️ 程式碼直接在伺服器上執行，只適合本機 / 信任環境。"""

DEFAULT_CODE = """# 範例：把所有圖層畫出來，線段顯示 label
for i, (name, df) in enumerate(layers.items()):
    is_line = "kind" in df and (df["kind"] == "line").any()
    img = draw(img, df, color=["lime", "yellow", "cyan"][i % 3], width=3, point_size=6, show_label=is_line)
"""


def execute_code(code, state, annots, name="執行程式碼"):
    """執行程式碼 → (state, annots, log, 結果摘要, 是否成功)。圖片有變才新增步驟；有設定 annotations 才更新圖層。"""
    img = require_image(state)
    layer = current_layer(annots)
    ns = {
        "img": img.copy(), "image_path": state["path"],
        "layers": {k: v["df"].copy() for k, v in annots["layers"].items()},
        "table": layer["df"].copy() if layer else pd.DataFrame(),
        "overlays": layer["shapes"] if layer else [],
        "draw": draw, "np": np, "pd": pd, "Image": Image, "ImageDraw": ImageDraw, "ImageFont": ImageFont,
        "ImageFilter": ImageFilter, "ImageOps": ImageOps, "ImageEnhance": ImageEnhance,
    }
    buf, notes = io.StringIO(), []
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, ns)  # noqa: S102 —— 原型階段刻意允許

        out = ns.get("img")
        if isinstance(out, np.ndarray):
            out = Image.fromarray(out.astype(np.uint8))
        if not isinstance(out, Image.Image):
            raise TypeError("程式碼結束時 `img` 必須是 PIL.Image 或 numpy 陣列")

        data = ns.get("annotations")
        if data is not None:
            if not isinstance(data, dict):
                data = {ns.get("annotation_name") or "annotations": data}
            for lname, df in data.items():
                df = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
                lname = safe_name(lname, "annotations")
                shapes = df_to_shapes(df)  # 先確認能解析
                path = os.path.join(WORK_DIR, f"{lname}_{len(os.listdir(WORK_DIR)) + 1}.csv")
                df.to_csv(path, index=False)
                annots = add_layer(annots, lname, df=df, shapes=shapes, path=path)
                notes.append(f"圖層「{lname}」{len(df)} 筆")

        out = out.convert("RGB")
        if out.size != img.size or not np.array_equal(np.asarray(out), np.asarray(img)):
            state = push_step(state, name, out)
            notes.append("圖片已更新")
        log, ok = buf.getvalue() or "執行成功", True
    except Exception:  # noqa: BLE001
        log, ok = buf.getvalue() + traceback.format_exc(limit=-3), False
        notes = ["執行失敗"]
    return state, annots, log, "，".join(notes) or "執行完成，沒有變更", ok


def run_code(code, state, annots):
    state, annots, log, note, _ = execute_code(code, state, annots)
    return (state, *render(state), annots, *render_annots(annots), f"{note}\n\n{log}")


# ---------------------------------------------------------------------------
# 匯出（固定資料夾 OUTPUT_DIR，檔名由使用者決定）
# ---------------------------------------------------------------------------

def export(state, annots, name, save_img, layer_names):
    img = require_image(state)
    if not name or not name.strip():
        raise gr.Error("請輸入檔名")
    name = safe_name(os.path.splitext(name.strip())[0])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    saved, overwritten = [], []

    def target(filename):
        p = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(p):
            overwritten.append(filename)
        saved.append(p)
        return p

    if save_img:
        img.save(target(f"{name}.png"))
    for lname in layer_names or []:
        layer = annots["layers"].get(lname)
        if layer:
            df = layer["df"] if not layer["df"].empty else shapes_to_df(layer["shapes"])
            df.to_csv(target(f"{name}_{lname}.csv"), index=False)
    if not saved:
        raise gr.Error("沒有勾選要匯出的內容")
    msg = f"已匯出到 `{OUTPUT_DIR}`：" + "、".join(os.path.basename(p) for p in saved)
    if overwritten:
        msg += f"\n\n⚠️ 覆蓋了既有檔案：{'、'.join(overwritten)}"
    return msg, saved


# ---------------------------------------------------------------------------
# 對話：後端 stub
# ---------------------------------------------------------------------------

COLOR_WORDS = {"紅": "red", "綠": "lime", "藍": "blue", "黃": "yellow", "白": "white", "黑": "black",
               "橘": "orange", "紫": "purple", "青": "cyan", "粉": "pink"}

FIND_CIRCLES_CODE = '''# 找圓心：有 OpenCV 用 HoughCircles；沒有的話用亮度門檻 + 連通區域（縮圖後做，再換算回原尺寸）
gray = np.asarray(img.convert("L"))
rows = []
try:
    import cv2
    found = cv2.HoughCircles(cv2.medianBlur(gray, 5), cv2.HOUGH_GRADIENT, dp=1.2,
                             minDist=max(10, min(gray.shape) // 20), param1=100, param2=30)
    if found is not None:
        rows = [{"x": float(x), "y": float(y), "r": float(r)} for x, y, r in found[0]]
except ImportError:
    scale = max(1, max(gray.shape) // 400)
    small = gray[::scale, ::scale]
    fg = small < small.mean() if small.mean() > 127 else small > small.mean()
    seen = np.zeros_like(fg)
    H, W = fg.shape
    for sy, sx in zip(*np.nonzero(fg)):
        if seen[sy, sx]:
            continue
        stack, pix = [(sy, sx)], []
        seen[sy, sx] = True
        while stack:
            y, x = stack.pop()
            pix.append((y, x))
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < H and 0 <= nx < W and fg[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if len(pix) < 20:
            continue
        ys, xs = np.array(pix).T
        rows.append({"x": xs.mean() * scale, "y": ys.mean() * scale, "r": float(np.sqrt(len(pix) / np.pi)) * scale})
annotations = pd.DataFrame(rows, columns=["x", "y", "r"]).round(1)
annotations["label"] = [f"C{i + 1}" for i in range(len(annotations))]
annotation_name = "circles"
print(annotations.to_string(index=False))
'''

CONNECT_CODE = '''# 依 x 排序，把相鄰圓心連線並計算 pixel 距離
c = layers.get("circles")
if c is None or c.empty:
    raise ValueError("找不到 circles 圖層，請先找圓心")
c = c.sort_values("x").reset_index(drop=True)
rows = []
for a, b in zip(c.itertuples(), c.iloc[1:].itertuples()):
    d = float(np.hypot(b.x - a.x, b.y - a.y))
    rows.append({"kind": "line", "x1": a.x, "y1": a.y, "x2": b.x, "y2": b.y,
                 "from": a.label, "to": b.label, "length_px": round(d, 1), "label": f"{d:.1f}px"})
annotations = pd.DataFrame(rows)
annotation_name = "distances"
print(annotations[["from", "to", "length_px"]].to_string(index=False))
'''


def chat_backend(message, history, context):
    """之後換成真正的 LLM。

    參數：
        message: 使用者這次輸入的文字
        history: 對話紀錄（messages 格式；含折疊的程式碼與執行結果訊息）
        context: {
            "image_path", "size", "image"(PIL), "steps"(編輯步驟名稱),
            "layers"({名稱: DataFrame}), "current_layer", "layers_summary",
            "code"(編輯器目前內容),
            "attempt"(第幾次嘗試，從 0 開始), "error"(上次執行的錯誤訊息或 None), "failed_code"(上次失敗的程式碼),
        }
    回傳：
        {"reply": 回覆文字, "code": 要執行的 Python 程式碼或 None}
        有回傳 code 時，前端會執行它；失敗就帶著 error 再呼叫一次（直到成功或達到重試上限）。
    """
    # --- 修錯重試（stub 只示範固定一種修法；真的 LLM 會看 error 自己改）---
    if context["error"]:
        if "rotate" in (context["failed_code"] or ""):
            return {"reply": "（stub）錯誤是 rotate 的角度給了字串，改成數字再試一次。",
                    "code": "img = img.rotate(90, expand=True)\n"}
        return {"reply": f"（stub）看到錯誤，但 stub 不會修：\n```\n{context['error'].strip().splitlines()[-1]}\n```",
                "code": None}

    if context["image_path"] is None:
        return {"reply": "（stub）目前沒有選圖片，先從左邊選一張吧。", "code": None}

    layers = context["layers"]
    if any(k in message for k in ("連線", "距離", "測量")):
        if "circles" not in layers:
            code = (FIND_CIRCLES_CODE
                    + '\nlayers["circles"] = annotations\n'
                    + CONNECT_CODE
                    + 'annotations = {"circles": layers["circles"], "distances": annotations}\n')
            return {"reply": "（stub）還沒有圓心資料，我先找圓心再連線。", "code": code}
        return {"reply": "（stub）把相鄰的圓心連線，並計算 pixel 距離。", "code": CONNECT_CODE}

    if "圓" in message and not any(k in message for k in ("畫", "標記")):
        return {"reply": "（stub）我用霍夫圓偵測來找圓心。", "code": FIND_CIRCLES_CODE}

    if any(k in message for k in ("畫", "標記")):
        if not layers:
            return {"reply": "（stub）目前沒有標註資料可以畫，要不要先讓我找圓心？", "code": None}
        color = next((v for k, v in COLOR_WORDS.items() if k in message), "lime")
        m = re.search(r"(\d+)", message)
        size = int(m.group(1)) if m else 6
        code = f"# 依照指示：主色 {color}、大小 {size}\n"
        if "circles" in layers:
            code += f'img = draw(img, layers["circles"], color={color!r}, width={max(2, size // 3)}, point_size={size}, show_label=True)\n'
        if "distances" in layers:
            code += f'img = draw(img, layers["distances"], color="yellow", width={max(2, size // 3)}, show_label=True)\n'
        for name in layers:
            if name not in ("circles", "distances"):
                code += f'img = draw(img, layers[{name!r}], color={color!r}, point_size={size})\n'
        return {"reply": f"（stub）用 {color}、大小 {size} 標記 {', '.join(layers)}。", "code": code}

    if "旋轉" in message:  # 故意寫錯，示範自動修錯
        return {"reply": "（stub）把圖片旋轉 90 度。", "code": 'img = img.rotate("90")\n'}

    for k, code in {"灰階": "img = ImageOps.grayscale(img).convert('RGB')\n",
                    "邊緣": "img = img.convert('L').filter(ImageFilter.FIND_EDGES).convert('RGB')\n",
                    "模糊": "img = img.filter(ImageFilter.GaussianBlur(radius=4))\n"}.items():
        if k in message:
            return {"reply": f"（stub）幫你做{k}處理。", "code": code}

    return {
        "reply": (f"（stub）目前圖片 {os.path.basename(context['image_path'])}（{context['size'][0]}×{context['size'][1]}），"
                  f"{context['layers_summary']}。\n\n可以依序試試：\n1. 幫我把圓心都找出來\n2. 把圓心連線並測量距離\n"
                  f"3. 用綠色大小 8 標記出來\n4. 到「匯出」分頁匯出\n\n（輸入「旋轉」可以看自動修錯的示範）"),
        "code": None,
    }


def _code_message(code, title):
    return {"role": "assistant", "content": f"```python\n{code.rstrip()}\n```", "metadata": {"title": title}}


def respond(message, history, state, annots, code, auto_run, max_retries):
    history = list(history or [])
    log = gr.update()

    def outputs():
        return ("", history, state, *render(state), annots, *render_annots(annots), code, log)

    if not message or not message.strip():
        yield outputs()
        return
    history.append({"role": "user", "content": message})
    yield outputs()

    error = failed_code = None
    for attempt in range(int(max_retries) + 1):
        img = current_image(state)
        context = {
            "image_path": state["path"], "size": img.size if img else None, "image": img,
            "steps": [n for n, _ in state["steps"][: state["idx"] + 1]],
            "layers": {k: v["df"] for k, v in annots["layers"].items()},
            "current_layer": annots["current"], "layers_summary": annots_summary(annots),
            "code": code, "attempt": attempt, "error": error, "failed_code": failed_code,
        }
        result = chat_backend(message, history, context)
        history.append({"role": "assistant", "content": result["reply"]})
        new_code = result.get("code")
        if not new_code:
            break
        code = new_code
        history.append(_code_message(code, "🛠 程式碼" if attempt == 0 else f"🛠 修正後的程式碼（第 {attempt} 次重試）"))
        if not auto_run:
            history.append({"role": "assistant", "content": "程式碼已放到「程式碼」分頁，確認後按「執行」。"})
            break
        yield outputs()

        state, annots, log, note, ok = execute_code(code, state, annots, name="LLM：" + message[:20])
        if ok:
            history.append({"role": "assistant", "content": f"✅ {note}"})
            break
        error, failed_code = log, code
        last_line = log.strip().splitlines()[-1] if log.strip() else "未知錯誤"
        if attempt < int(max_retries):
            history.append({"role": "assistant", "content": f"⚠️ 執行失敗：`{last_line}`，交給 LLM 修正…"})
        else:
            history.append({"role": "assistant", "content": f"❌ 執行失敗：`{last_line}`（已達重試上限，詳細錯誤在「程式碼」分頁）"})
        yield outputs()
    yield outputs()


# ---------------------------------------------------------------------------
# 介面
# ---------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(title="MUI") as demo:
        state = gr.State(empty_state())
        annots = gr.State(empty_annots())
        image_paths = gr.State([])

        with gr.Row(equal_height=False):
            # ---- 左：資料夾 ----
            with gr.Column(scale=2, min_width=260):
                gr.Markdown("### 資料夾")
                with gr.Row():
                    folder_box = gr.Textbox(value=ROOT_DIR, show_label=False, scale=5)
                    up_btn = gr.Button("⬆", scale=1, min_width=40)
                subdir_dd = gr.Dropdown(label="子資料夾", choices=[], interactive=True)
                auto_annot = gr.Checkbox(value=True, label="選圖時自動載入同名標註檔")
                gallery = gr.Gallery(label="圖片", columns=3, height=560, allow_preview=False, object_fit="cover")

            # ---- 中：圖片檢視 + 工具 ----
            with gr.Column(scale=4, min_width=420):
                image_info = gr.Markdown("尚未選擇圖片")
                viewer = gr.Image(label="圖片檢視", type="pil", interactive=False, height=520)
                with gr.Row():
                    undo_btn = gr.Button("↶ 復原", size="sm")
                    redo_btn = gr.Button("↷ 重做", size="sm")
                    reset_btn = gr.Button("還原原圖", size="sm")
                steps = gr.Radio(label="編輯步驟（點選可跳回）", choices=[])

                with gr.Tabs():
                    with gr.Tab("標註資料"):
                        annot_info = gr.Markdown(annots_summary(empty_annots()))
                        with gr.Row():
                            layer_dd = gr.Dropdown(label="檢視圖層", choices=[], interactive=True, scale=4)
                            del_layer_btn = gr.Button("刪除此圖層", size="sm", scale=1)
                        annot_table = gr.Dataframe(interactive=False, max_height=260, wrap=True)
                        annot_file = gr.File(label="此圖層 csv（下載）", interactive=False, height=80)
                        with gr.Accordion("載入既有標註檔為新圖層", open=False):
                            overlay_files = gr.File(label="txt / csv / mask 圖片（可多個，每個檔案一個圖層）",
                                                    file_count="multiple", height=110)
                            overlay_fmt = gr.Dropdown(FORMATS, value=F_AUTO, label="格式（無表頭的 txt 用）")
                            load_btn = gr.Button("載入")
                            load_status = gr.Markdown()

                    with gr.Tab("疊加樣式"):
                        gr.Markdown("手動把圖層畫到圖片上（也可以直接在對話裡請 LLM 畫）。")
                        draw_layers = gr.CheckboxGroup(label="要疊加的圖層", choices=[])
                        with gr.Row():
                            ov_mode = gr.Radio(COLOR_MODES, value=COLOR_MODES[0], label="上色方式")
                            ov_color = gr.ColorPicker(value="#ff0000", label="單一顏色")
                            ov_show_label = gr.Checkbox(value=True, label="顯示 label")
                        with gr.Row():
                            ov_width = gr.Slider(1, 12, value=3, step=1, label="線寬")
                            ov_point = gr.Slider(1, 30, value=5, step=1, label="點大小")
                            ov_opacity = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="不透明度")
                        overlay_btn = gr.Button("疊加到圖片上", variant="primary")

                    with gr.Tab("程式碼"):
                        gr.Markdown(CODE_HELP)
                        code_box = gr.Code(value=DEFAULT_CODE, language="python", lines=14, interactive=True)
                        with gr.Row():
                            run_btn = gr.Button("執行", variant="primary")
                            auto_run = gr.Checkbox(value=True, label="LLM 回傳程式碼時自動執行")
                            max_retries = gr.Slider(0, 5, value=2, step=1, label="失敗自動重試次數")
                        code_log = gr.Textbox(label="輸出 / 錯誤", lines=6, interactive=False)

                    with gr.Tab("匯出"):
                        gr.Markdown(f"輸出資料夾：`{OUTPUT_DIR}`")
                        export_name = gr.Textbox(label="檔名（不含副檔名）")
                        export_img = gr.Checkbox(value=True, label="目前圖片 → <檔名>.png")
                        export_layers = gr.CheckboxGroup(label="圖層 → <檔名>_<圖層>.csv", choices=[])
                        export_btn = gr.Button("匯出", variant="primary")
                        export_status = gr.Markdown()
                        export_files = gr.File(label="下載", file_count="multiple", interactive=False)

            # ---- 右：對話 ----
            with gr.Column(scale=3, min_width=320):
                gr.Markdown("### 對話")
                chatbot = gr.Chatbot(label="對話", height=660)
                with gr.Row():
                    user_input = gr.Textbox(placeholder="例如：幫我把圓心都找出來（Enter 送出）",
                                            show_label=False, scale=6)
                    send_btn = gr.Button("送出", variant="primary", scale=1, min_width=60)
                clear_btn = gr.Button("清除對話", size="sm")

        # ---- 事件 ----
        view = [state, viewer, steps]
        annot_view = [annots, annot_info, layer_dd, annot_table, annot_file, draw_layers, export_layers]

        folder_outputs = [folder_box, gallery, image_paths, subdir_dd]
        demo.load(open_folder, folder_box, folder_outputs)
        folder_box.submit(open_folder, folder_box, folder_outputs)
        up_btn.click(go_up, folder_box, folder_outputs)
        subdir_dd.input(enter_subdir, [folder_box, subdir_dd], folder_outputs)
        gallery.select(select_image, [image_paths, auto_annot],
                       view + annot_view + [image_info, export_name, overlay_files])

        steps.input(jump_step, [state, steps], view)
        undo_btn.click(undo, state, view)
        redo_btn.click(redo, state, view)
        reset_btn.click(reset, state, view)

        layer_dd.input(select_layer, [annots, layer_dd], annot_view)
        del_layer_btn.click(delete_layer, [annots, layer_dd], annot_view)
        load_btn.click(load_files, [state, annots, overlay_files, overlay_fmt], annot_view + [load_status])
        overlay_btn.click(apply_overlay, [state, annots, draw_layers, ov_mode, ov_color, ov_width, ov_point,
                                          ov_opacity, ov_show_label], view)
        run_btn.click(run_code, [code_box, state, annots], view + annot_view + [code_log])
        export_btn.click(export, [state, annots, export_name, export_img, export_layers],
                         [export_status, export_files])

        chat_inputs = [user_input, chatbot, state, annots, code_box, auto_run, max_retries]
        chat_outputs = [user_input, chatbot] + view + annot_view + [code_box, code_log]
        user_input.submit(respond, chat_inputs, chat_outputs)
        send_btn.click(respond, chat_inputs, chat_outputs)
        clear_btn.click(lambda: [], None, chatbot)

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="MUI Gradio app")
    parser.add_argument("--host", default="127.0.0.1", help="綁定的 host（對外開放用 0.0.0.0）")
    parser.add_argument("--port", type=int, default=7860, help="埠號")
    parser.add_argument("--root", default=".", help="圖片根目錄；瀏覽限制在此目錄內")
    parser.add_argument("--output", default=None, help="匯出的固定資料夾（預設 <root>/output）")
    parser.add_argument("--share", action="store_true", help="建立 gradio 公開分享連結")
    return parser.parse_args()


def main():
    global ROOT_DIR, OUTPUT_DIR
    args = parse_args()
    ROOT_DIR = os.path.abspath(os.path.expanduser(args.root))
    OUTPUT_DIR = os.path.abspath(os.path.expanduser(args.output or os.path.join(ROOT_DIR, "output")))
    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share,
                allowed_paths=[ROOT_DIR, OUTPUT_DIR, WORK_DIR])


if __name__ == "__main__":
    main()
