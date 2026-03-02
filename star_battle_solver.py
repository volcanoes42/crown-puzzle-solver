import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pulp
from collections import defaultdict
import os

st.set_page_config(page_title="Crown Puzzle Solver", layout="wide")
st.title("👑 Crown Puzzle (Star Battle) ソルバー")
st.markdown("スマホゲームのパズル画像から領域と王冠の初期配置を自動読み取りし、PuLP で解を求めます。")

# ─── ヘルパー関数 ───
def rgb_to_hex(r, g, b):
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"

def text_color_for_bg(r, g, b):
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "#000000" if lum > 140 else "#ffffff"

def read_grid_colors(img_rgb, gx1, gy1, gx2, gy2, n):
    """各セルの色を枠近くからサンプリング（マーク回避）"""
    grid_img = img_rgb[gy1:gy2, gx1:gx2]
    h, w = grid_img.shape[:2]
    cell_w, cell_h = w / n, h / n
    colors = np.zeros((n, n, 3), dtype=int)
    for r in range(n):
        for c in range(n):
            sx = int(c * cell_w + cell_w * 0.15)
            sy = int(r * cell_h + cell_h * 0.15)
            patch = grid_img[max(0, sy - 3):sy + 4, max(0, sx - 3):sx + 4]
            colors[r, c] = patch.mean(axis=(0, 1)).astype(int)
    return colors

def detect_marks(img_rgb, gx1, gy1, gx2, gy2, n):
    """各セル中央を解析し、王冠・×・空を判定する。

    判定ロジック:
      1. セル中央パッチの標準偏差 > 15 → マークあり
      2. マークありセルで金色ピクセル(R>200, G>150, B<120)比率 > 0.3 → 王冠
      3. それ以外のマーク → ×

    Returns:
        marks: dict (r,c) -> 'crown' | 'x' | 'empty'
    """
    grid_img = img_rgb[gy1:gy2, gx1:gx2]
    h, w = grid_img.shape[:2]
    cell_w, cell_h = w / n, h / n
    marks = {}
    for r in range(n):
        for c in range(n):
            cx = int((c + 0.5) * cell_w)
            cy = int((r + 0.5) * cell_h)
            patch = grid_img[max(0, cy - 10):cy + 11, max(0, cx - 10):cx + 11]
            std_val = patch.std(axis=(0, 1)).mean()
            if std_val > 15:
                gold = (
                    (patch[:, :, 0] > 200)
                    & (patch[:, :, 1] > 150)
                    & (patch[:, :, 2] < 120)
                )
                gold_ratio = gold.sum() / (patch.shape[0] * patch.shape[1])
                marks[(r, c)] = "crown" if gold_ratio > 0.3 else "x"
            else:
                marks[(r, c)] = "empty"
    return marks

def cluster_colors(cell_colors, n_clusters):
    """K-means で色を n_clusters 個のクラスタに分類"""
    flat = cell_colors.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.5)
    _, labels, centers = cv2.kmeans(
        flat, n_clusters, None, criteria, 20, cv2.KMEANS_PP_CENTERS
    )
    return labels.reshape(cell_colors.shape[0], cell_colors.shape[1]).astype(int), centers.astype(int)

def render_grid_html(n, color_func, content_func, cell_size=42):
    """グリッドを HTML テーブルで描画"""
    html = '<table style="border-collapse:collapse;margin:auto;">'
    for r in range(n):
        html += "<tr>"
        for c in range(n):
            bg = color_func(r, c)
            hex_bg = rgb_to_hex(*bg)
            txt_col = text_color_for_bg(*bg)
            content = content_func(r, c)
            html += (
                f'<td style="width:{cell_size}px;height:{cell_size}px;text-align:center;'
                f"vertical-align:middle;background:{hex_bg};color:{txt_col};"
                f'border:1px solid #666;font-size:{cell_size // 2}px;font-weight:bold;">'
                f"{content}</td>"
            )
        html += "</tr>"
    html += "</table>"
    return html

def solve_star_battle(n, stars_per, region_map, initial_crowns=None):
    """PuLP で Star Battle (Crown) を解く"""
    prob = pulp.LpProblem("CrownPuzzle", pulp.LpMaximize)
    x = {
        (r, c): pulp.LpVariable(f"x_{r}_{c}", cat="Binary")
        for r in range(n)
        for c in range(n)
    }
    prob += 0

    for r in range(n):
        prob += pulp.lpSum(x[r, c] for c in range(n)) == stars_per
    for c in range(n):
        prob += pulp.lpSum(x[r, c] for r in range(n)) == stars_per

    region_cells = defaultdict(list)
    for r in range(n):
        for c in range(n):
            region_cells[region_map[r, c]].append((r, c))
    for rid, cells in region_cells.items():
        prob += pulp.lpSum(x[r, c] for r, c in cells) == stars_per

    for r in range(n - 1):
        for c in range(n - 1):
            prob += x[r, c] + x[r, c + 1] + x[r + 1, c] + x[r + 1, c + 1] <= 1

    if initial_crowns:
        for ir, ic in initial_crowns:
            if 0 <= ir < n and 0 <= ic < n:
                prob += x[ir, ic] == 1

    solver = pulp.PULP_CBC_CMD(msg=0)
    status = prob.solve(solver)

    if pulp.LpStatus[status] == "Optimal":
        sol = np.zeros((n, n), dtype=int)
        for r in range(n):
            for c in range(n):
                if pulp.value(x[r, c]) > 0.5:
                    sol[r, c] = 1
        return sol
    return None


# ═══════════════════════════════════════════
# サイドバー
# ═══════════════════════════════════════════
st.sidebar.header("⚙️ パラメータ設定")
n = st.sidebar.number_input("グリッドサイズ (n×n)", 4, 15, 10)
stars_per = st.sidebar.number_input("各行/列/領域の王冠の数", 1, 3, 2)

st.sidebar.header("📐 グリッド領域座標")
c1, c2 = st.sidebar.columns(2)
gx1 = c1.number_input("左上 X", value=84)
gy1 = c1.number_input("左上 Y", value=320)
gx2 = c2.number_input("右下 X", value=665)
gy2 = c2.number_input("右下 Y", value=900)

# ═══════════════════════════════════════════
# 画像読み込み
# ═══════════════════════════════════════════
uploaded = st.file_uploader("パズル画像をアップロード", type=["png", "jpg", "jpeg"])

img_rgb = None
sample_path = "/mnt/user-data/uploads/IMG_5728.PNG"

if uploaded is not None:
    buf = np.frombuffer(uploaded.read(), dtype=np.uint8)
    img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
elif os.path.exists(sample_path):
    if st.button("📷 サンプル画像 (IMG_5728.PNG) を使用"):
        st.session_state["use_sample"] = True
    if st.session_state.get("use_sample"):
        img_bgr = cv2.imread(sample_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

if img_rgb is None:
    st.info("パズル画像をアップロードするか、サンプル画像を使用してください。")
    st.stop()

st.image(img_rgb, caption="入力画像", width=380)

# ═══════════════════════════════════════════
# 画像処理: 色読み取り + マーク検出
# ═══════════════════════════════════════════
cell_colors = read_grid_colors(img_rgb, gx1, gy1, gx2, gy2, n)
region_map, centers = cluster_colors(cell_colors, n)
region_color_map = {i: tuple(centers[i]) for i in range(n)}

marks = detect_marks(img_rgb, gx1, gy1, gx2, gy2, n)
auto_crowns = sorted([(r, c) for (r, c), v in marks.items() if v == "crown"])
auto_xs = sorted([(r, c) for (r, c), v in marks.items() if v == "x"])

# ═══════════════════════════════════════════
# 読み取り結果表示
# ═══════════════════════════════════════════
st.subheader("📊 読み取り結果")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("**領域マップ** (色→領域ID)")
    html_region = render_grid_html(
        n,
        color_func=lambda r, c: region_color_map[region_map[r, c]],
        content_func=lambda r, c: str(region_map[r, c]),
        cell_size=38,
    )
    st.markdown(html_region, unsafe_allow_html=True)

with col_right:
    st.markdown("**マーク検出** (👑=王冠, ✕=×, ・=空)")
    def mark_content(r, c):
        m = marks.get((r, c), "empty")
        if m == "crown":
            return "👑"
        elif m == "x":
            return "✕"
        return ""
    html_marks = render_grid_html(
        n,
        color_func=lambda r, c: region_color_map[region_map[r, c]],
        content_func=mark_content,
        cell_size=38,
    )
    st.markdown(html_marks, unsafe_allow_html=True)

st.info(
    f"👑 王冠 **{len(auto_crowns)}** 個検出: "
    f"{', '.join(f'({r+1}行{c+1}列)' for r,c in auto_crowns) if auto_crowns else 'なし'}　｜　"
    f"✕ ×マーク **{len(auto_xs)}** 個検出"
)

# ═══════════════════════════════════════════
# 手動修正
# ═══════════════════════════════════════════
with st.expander("✏️ 領域マップの手動修正", expanded=False):
    st.markdown("画像認識が正しくない場合、各行をカンマ区切りの領域ID (0～n-1) で修正できます。")
    default_text = "\n".join(
        ",".join(str(region_map[r, c]) for c in range(n)) for r in range(n)
    )
    map_text = st.text_area("領域マップ", value=default_text, height=250)

with st.expander("✏️ 王冠の初期配置を手動修正", expanded=False):
    st.markdown("自動検出結果を修正できます。行,列 (0始まり) を1行に1組ずつ入力。空にすると初期配置なし。")
    default_crowns = "\n".join(f"{r},{c}" for r, c in auto_crowns)
    crowns_text = st.text_area("王冠の座標", value=default_crowns, height=100)

# パース: 領域マップ
try:
    edited = []
    for line in map_text.strip().split("\n"):
        edited.append([int(x.strip()) for x in line.split(",")])
    edited_map = np.array(edited)
    assert edited_map.shape == (n, n)
except Exception:
    st.error("領域マップの形式が不正です。各行にn個のカンマ区切り整数を入力してください。")
    st.stop()

# パース: 王冠
initial_crowns = set()
if crowns_text.strip():
    for line in crowns_text.strip().split("\n"):
        parts = line.strip().split(",")
        if len(parts) == 2:
            initial_crowns.add((int(parts[0].strip()), int(parts[1].strip())))

# ═══════════════════════════════════════════
# 求解
# ═══════════════════════════════════════════
if st.button("🚀 解を求める", type="primary"):
    with st.spinner("PuLP で求解中..."):
        solution = solve_star_battle(n, stars_per, edited_map, initial_crowns)

    if solution is not None:
        st.success(f"✅ 解が見つかりました！ (王冠の総数: {solution.sum()})")

        region_cells = defaultdict(list)
        for r in range(n):
            for c in range(n):
                region_cells[edited_map[r, c]].append((r, c))
        final_colors = {}
        for rid, cells in region_cells.items():
            avg = np.mean([cell_colors[r, c] for r, c in cells], axis=0).astype(int)
            final_colors[rid] = tuple(avg)

        st.subheader("👑 解答")
        html_sol = render_grid_html(
            n,
            color_func=lambda r, c: final_colors.get(edited_map[r, c], (200, 200, 200)),
            content_func=lambda r, c: "👑" if solution[r, c] == 1 else "",
            cell_size=48,
        )
        st.markdown(html_sol, unsafe_allow_html=True)

        st.subheader("📝 テキスト表示")
        lines = []
        for r in range(n):
            lines.append(" ".join("👑" if solution[r, c] else "・" for c in range(n)))
        st.code("\n".join(lines))
    else:
        st.error("❌ 解が見つかりませんでした。")
        st.warning("領域マップやパラメータを再確認してください。")
