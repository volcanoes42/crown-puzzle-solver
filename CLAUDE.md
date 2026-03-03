# CLAUDE.md — Crown Puzzle Solver

This file provides context for AI assistants working in this repository.

---

## Project Overview

**Crown Puzzle (Star Battle) Solver** is a single-file Streamlit web application that:

1. Accepts a smartphone screenshot of a Crown / Star Battle puzzle
2. Auto-detects the grid size, colored regions, and any pre-placed crowns via computer vision
3. Solves the puzzle using integer linear programming (ILP)
4. Displays the solution as a color-coded HTML grid and plain-text representation

The UI is written in Japanese. The puzzle rules: place exactly `stars_per` crowns in each row, column, and colored region, with no two crowns in adjacent cells (including diagonals).

---

## Repository Structure

```
crown-puzzle-solver/
├── star_battle_solver.py   # Entire application (341 lines)
├── requirements.txt        # Python package dependencies
├── packages.txt            # System-level dependencies (for deployment)
└── CLAUDE.md               # This file
```

This is intentionally a minimal, single-file project. Do not split it into multiple modules unless there is a strong reason.

---

## Tech Stack

| Layer | Library | Version |
|---|---|---|
| Web UI | streamlit | >=1.30.0 |
| Image processing | opencv-python-headless | >=4.8.0 |
| Numerical arrays | numpy | >=1.24.0 |
| Image I/O | Pillow | >=10.0.0 |
| ILP solver | PuLP (CBC backend) | >=2.7.0 |
| Signal processing | scipy | >=1.10.0 |

System packages (`packages.txt`) provide OpenGL/GLib support for headless OpenCV:
- `libgl1-mesa-glx`
- `libglib2.0-0`

---

## Running the App

```bash
pip install -r requirements.txt
streamlit run star_battle_solver.py
```

There is no build step, no Docker file, and no dev server separate from Streamlit.

---

## Code Architecture

All code lives in `star_battle_solver.py` in this order:

### 1. Helper functions (lines 15–20)
- `rgb_to_hex(r, g, b)` — converts RGB tuple to CSS hex string
- `text_color_for_bg(r, g, b)` — returns `#000000` or `#ffffff` based on luminance (threshold 140)

### 2. Image processing functions (lines 22–123)

| Function | Purpose |
|---|---|
| `detect_grid_size(img_gray, gx1, gy1, gx2, gy2)` | Canny edge detection + `scipy.signal.find_peaks` on row/column profiles to locate grid lines and infer grid dimensions. Returns `(n_rows, n_cols, h_lines, v_lines)`. |
| `read_grid_colors(img_rgb, gx1, gy1, gx2, gy2, n)` | Samples a 7×7 pixel patch at 15% offset from each cell's top-left corner (to avoid mark overlap). Returns `(n, n, 3)` NumPy int array. |
| `detect_marks(img_rgb, gx1, gy1, gx2, gy2, n)` | Samples a 21×21 patch at each cell centre. If `std > 15` → mark present. Gold pixel ratio `(R>200, G>150, B<120) > 0.3` → crown; otherwise X. Returns `{(r, c): "crown"|"x"|"empty"}`. |
| `cluster_colors(cell_colors, n_clusters)` | K-means (OpenCV, PP_CENTERS init, 200 iterations) on flattened color array. Returns `(labels_2d, centers)`. |
| `render_grid_html(n, color_func, content_func, cell_size=42)` | Builds an HTML `<table>` string. Callers pass lambdas for background color and cell content. Used with `st.markdown(..., unsafe_allow_html=True)`. |

### 3. Solver (lines 125–166)
`solve_star_battle(n, stars_per, region_map, initial_crowns=None)`

Uses **PuLP** with binary variables `x[r, c]` and the CBC solver. Constraints:
- Each row sums to `stars_per`
- Each column sums to `stars_per`
- Each region (identified by `region_map[r, c]` integer label) sums to `stars_per`
- No 2×2 block contains more than one crown: `x[r,c] + x[r,c+1] + x[r+1,c] + x[r+1,c+1] <= 1`
- Pre-placed crowns (from image detection or manual input) are fixed: `x[ir,ic] == 1`

Returns a `(n, n)` NumPy int array (1 = crown) or `None` if infeasible.

### 4. Streamlit UI (lines 169–342)

**Sidebar:**
- Grid ROI coordinates: `gx1`, `gy1`, `gx2`, `gy2` (default: 84, 320, 665, 900 — tuned for a specific phone screenshot)
- Grid size `n` (4–15): pre-filled from auto-detection, user-editable
- `stars_per`: crowns per row/column/region (default 2)

**Main area flow:**
1. File uploader (or sample image button if `/mnt/user-data/uploads/IMG_5728.PNG` exists)
2. Display uploaded image
3. Auto-detect grid → show region map and mark detection side-by-side in two columns
4. Expandable text areas for manual region map correction (CSV rows) and initial crown positions (`row,col` per line, 0-indexed)
5. "Solve" button → run ILP → display solution grid + plain-text output

---

## Key Conventions

### Coordinate system
- Image pixels: standard `(x, y)` with `x` = column, `y` = row
- Grid cells: `(r, c)` where `r` is row (0-indexed from top), `c` is column (0-indexed from left)
- ROI extraction: `img[gy1:gy2, gx1:gx2]` — note numpy row-first ordering

### Color handling
- Internal color storage: RGB (not BGR). OpenCV reads as BGR and is converted immediately: `cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)`
- Colors stored as NumPy `int` arrays of shape `(3,)` or `(n, n, 3)`

### Region map
- Integer labels `0` to `n-1` stored in a `(n, n)` NumPy int array
- Region IDs are assigned by K-means cluster index, not by any semantic ordering
- Manual correction format: n lines of n comma-separated integers

### Marks dictionary
- Keys: `(row, col)` tuples
- Values: `"crown"`, `"x"`, or `"empty"`

### HTML rendering
- `render_grid_html` always uses inline styles; no external CSS
- Text color auto-selected for contrast via luminance formula: `0.299R + 0.587G + 0.114B`

### Streamlit state
- Only `st.session_state["use_sample"]` is used for persistent state across reruns
- All other state is derived fresh from the uploaded image on each rerun

---

## Testing

There are **no automated tests**. Manual testing is done by:
1. Uploading a real puzzle screenshot
2. Verifying region detection visually
3. Correcting the region map if needed
4. Running the solver and checking the solution satisfies all constraints

If adding tests, use `pytest`. Test the pure functions (`detect_grid_size`, `cluster_colors`, `solve_star_battle`) with synthetic inputs rather than requiring real images.

---

## Common Development Tasks

### Adjusting detection thresholds
- Grid line detection: `height=np.max(h_profile) * 0.3` and `distance=max(15, h // 20)` in `detect_grid_size`
- Mark detection: `std_val > 15` threshold and gold pixel thresholds `(R>200, G>150, B<120)` in `detect_marks`
- Color sampling offset: `cell_w * 0.15` in `read_grid_colors`

### Changing solver constraints
All ILP constraints are in `solve_star_battle`. Add new constraints using `prob += <pulp_expression>`.

### Adding support for a new puzzle variant
1. Extend `solve_star_battle` with new constraint parameters
2. Add corresponding sidebar inputs
3. Pass the new parameters through from the UI to the solver call at line 311

### Modifying the HTML grid display
Edit `render_grid_html`. The `color_func` and `content_func` lambdas are passed at each call site — update those call sites for different display modes.

---

## Deployment Notes

This app is designed for deployment on **Streamlit Community Cloud** (or similar):
- `requirements.txt` handles Python deps
- `packages.txt` handles system deps (parsed by Streamlit Cloud's `apt-get`)
- The sample image path `/mnt/user-data/uploads/IMG_5728.PNG` is a Streamlit Cloud upload path; it is optional (the button only appears if the file exists)

---

## Git Workflow

- Primary branch: `master`
- Feature/task branches: `claude/<description>-<session-id>` (managed by AI assistants)
- Remote: `origin` (internal proxy at `http://local_proxy@127.0.0.1:22543/git/volcanoes42/crown-puzzle-solver`)
- Always push with: `git push -u origin <branch-name>`
- Commit messages are short and imperative (e.g., "Update star_battle_solver.py")
