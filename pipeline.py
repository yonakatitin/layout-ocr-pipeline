import math
import cv2
import pytesseract
from pytesseract import Output
import os
import html as htmllib

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ============================================================
# LAYOUT-AWARE TEXT EXTRACTION PIPELINE
# End-to-end: OCR -> clustering -> style detection -> inpainting -> HTML
# ============================================================

TESS_LANG = "ind+eng"
TESS_CONFIGS = ["--psm 3", "--psm 11"]

# ============================================================
# 1. TILED OCR EXTRACTION
# Potong gambar jadi tile overlap, OCR tiap tile (psm3+psm11),
# gabung + dedup + filter noise.
# ============================================================

def make_tiles(img_w, img_h, target_tile_px=3200, overlap=0.35):
    """
    Bikin grid tile yang saling overlap.
    - target_tile_px: ukuran ideal 1 sisi tile (pixel)
    - overlap: seberapa besar tile bertumpuk (0.25 = 25%)
    Minimal grid 2x2 (1 pass OCR full-image gampang miss teks kontras rendah).
    """
    n_cols = max(2, math.ceil(img_w / target_tile_px))
    n_rows = max(2, math.ceil(img_h / target_tile_px))

    tw = img_w / (n_cols - (n_cols - 1) * overlap)
    th = img_h / (n_rows - (n_rows - 1) * overlap)

    xs = [int(i * tw * (1 - overlap)) for i in range(n_cols)]
    ys = [int(i * th * (1 - overlap)) for i in range(n_rows)]

    tiles = []
    for y in ys:
        for x in xs:
            x1 = min(img_w, int(x + tw))
            y1 = min(img_h, int(y + th))
            tiles.append((x, y, x1, y1))
    return tiles

def iou(a, b):
    ax0, ay0, ax1, ay1 = a["left"], a["top"], a["left"] + a["width"], a["top"] + a["height"]
    bx0, by0, bx1, by1 = b["left"], b["top"], b["left"] + b["width"], b["top"] + b["height"]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    return inter / (area_a + area_b - inter + 1e-6)

def contained_ratio(a, b):
    """Seberapa besar persentase box a yang 'ketelan' di dalam box b (0..1)."""
    ax0, ay0, ax1, ay1 = a["left"], a["top"], a["left"] + a["width"], a["top"] + a["height"]
    bx0, by0, bx1, by1 = b["left"], b["top"], b["left"] + b["width"], b["top"] + b["height"]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0) or 1
    return inter / area_a

def dedup_words(words, iou_thresh=0.4, containment_thresh=0.75):
    """
    Buang duplikat 2 cara:
    1. IoU tinggi -> box yang hampir sama persis (dari tile overlap)
    2. Containment tinggi -> box kecil yang 'ketelan' di dalam box lain
       yang lebih besar (fragmen/pecahan kata salah baca, biasanya
       muncul dari psm11 yang lebih 'berani' nebak-nebak)
    """
    words = sorted(words, key=lambda w: (-w["conf"], -len(w["text"])))
    kept = []
    for w in words:
        is_dup = False
        for k in kept:
            if iou(w, k) > iou_thresh:
                is_dup = True
                break
            if contained_ratio(w, k) > containment_thresh and w["width"] * w["height"] <= k["width"] * k["height"]:
                is_dup = True
                break
        if not is_dup:
            kept.append(w)
    return kept

def ocr_words_tiled(image_bgr, lang=TESS_LANG, min_conf=45, target_tile_px=3200, overlap=0.35):
    """
    OCR seluruh gambar lewat tiling (psm3 + psm11) + dedup + filter noise.
    Pakai 2 ukuran grid tile berbeda (default + lebih kecil/granular)
    supaya area yang gagal terdeteksi di grid besar (karena tile-nya
    terlalu 'rame' buat Tesseract) masih punya kesempatan ketangkep
    lewat grid yang lebih kecil.
    """
    h, w = image_bgr.shape[:2]
    tile_sets = [
        make_tiles(w, h, target_tile_px=target_tile_px, overlap=overlap),
        make_tiles(w, h, target_tile_px=int(target_tile_px * 0.55), overlap=overlap),
    ]

    all_words = []
    for tiles in tile_sets:
        for (x0, y0, x1, y1) in tiles:
            crop = image_bgr[y0:y1, x0:x1]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            for config in TESS_CONFIGS:
                data = pytesseract.image_to_data(gray, lang=lang, config=config, output_type=Output.DICT)
                for i in range(len(data["text"])):
                    text = data["text"][i]
                    conf = float(data["conf"][i])
                    if not text.strip() or conf < min_conf:
                        continue
                    ww, wh = data["width"][i], data["height"][i]
                    if ww <= 0 or wh <= 0:
                        continue
                    all_words.append({
                        "text": text,
                        "conf": conf,
                        "left": x0 + data["left"][i],
                        "top": y0 + data["top"][i],
                        "width": ww,
                        "height": wh,
                    })

    words = dedup_words(all_words)
    words = [w for w in words if looks_like_real_word(w["text"], w["conf"])]
    words = filter_outliers(words)
    return words

def looks_like_real_word(text, conf, min_high_conf=85, min_low_conf=60):
    """
    Filter buat buang noise OCR (biasanya dari ikon/ilustrasi yang
    kebaca sebagai simbol/huruf random).
    - Token pendek (<=2 karakter) sering salah baca ikon/garis grafis
      dengan confidence lumayan tinggi (85-88) -> butuh threshold
      lebih ketat (92) supaya nggak lolos, sementara kata pendek asli
      ("PT", "di", "IT,") biasanya confidence-nya >=93.
    - Token lebih panjang (>2 karakter) pakai threshold normal (85).
    - Confidence sedang (>=60) -> harus terlihat kayak kata beneran
      (minimal 4 karakter alfanumerik, bukan simbol doang).
    """
    t = text.strip()
    if not t:
        return False
    alnum = sum(ch.isalnum() for ch in t)
    required_conf = min_high_conf if len(t) > 2 else 92
    if conf >= required_conf and alnum >= 1:
        return True
    if conf >= min_low_conf and alnum >= 4 and alnum / len(t) > 0.6:
        return True
    return False


def filter_outliers(words, max_height_ratio=3.2, short_symbol_max_h=260):
    """
    Buang deteksi yang jelas aneh:
    - token pendek (<=2 char) tanpa huruf/angka tapi bounding box-nya
      raksasa (biasanya salah baca elemen grafis/ikon)
    - token pendek yang tinggi bounding box-nya jauh di atas rata-rata
      tinggi teks lain di gambar (outlier ukuran)
    """
    if not words:
        return words
    heights = sorted(w["height"] for w in words)
    median_h = heights[len(heights) // 2]

    cleaned = []
    for w in words:
        text = w["text"].strip()
        has_alnum = any(c.isalnum() for c in text)
        if not has_alnum and len(text) <= 2 and w["height"] > short_symbol_max_h:
            continue
        if median_h > 0 and w["height"] > median_h * max_height_ratio and len(text) <= 3:
            continue
        cleaned.append(w)
    return cleaned

import numpy as np

# ============================================================
# 2. LAYOUT CLUSTERING
# Kata -> baris -> paragraf, pakai union-find biar robust untuk
# layout multi-kolom (card bersebelahan, hexagon, dll).
# ============================================================

def cluster_lines(words, v_tol_factor=0.45, h_gap_factor=2.0):
    """
    Gabungkan kata jadi baris pakai union-find: 2 kata dianggap 1 baris
    HANYA KALAU deket secara vertikal (center Y mirip) DAN deket secara
    horizontal (gap kecil) sekaligus. Ini lebih robust dibanding pisah
    row-dulu-baru-kolom, karena nggak ada 'drift' dari average yang
    terus membesar.
    """
    n = len(words)
    if n == 0:
        return []
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(n):
        for j in range(i + 1, n):
            wi, wj = words[i], words[j]
            avg_h = (wi["height"] + wj["height"]) / 2.0
            ci = wi["top"] + wi["height"] / 2.0
            cj = wj["top"] + wj["height"] / 2.0
            v_dist = abs(ci - cj)

            i_right = wi["left"] + wi["width"]
            j_right = wj["left"] + wj["width"]
            if wi["left"] <= wj["left"]:
                h_gap = max(0, wj["left"] - i_right)
            else:
                h_gap = max(0, wi["left"] - j_right)

            if v_dist < avg_h * v_tol_factor and h_gap < avg_h * h_gap_factor:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(words[i])

    lines = []
    for group in groups.values():
        group.sort(key=lambda w: w["left"])
        top = min(w["top"] for w in group)
        bottom = max(w["top"] + w["height"] for w in group)
        lines.append({"top": top, "bottom": bottom, "words": group})

    lines.sort(key=lambda l: (l["top"], min(w["left"] for w in l["words"])))
    return lines


def cluster_paragraphs(lines, gap_factor=0.7, min_x_overlap=0.35, max_height_ratio=1.7):
    """
    Gabungkan baris jadi paragraf pakai union-find (bukan cuma cek baris
    yang bersebelahan di list, karena urutan list bisa interleaved antar
    kolom). Syarat gabung: jarak vertikal kecil + overlap horizontal +
    ukuran font mirip (biar judul besar ga ke-gabung sama body text kecil).
    """
    if not lines:
        return []

    def line_bounds(line):
        l = min(w["left"] for w in line["words"])
        r = max(w["left"] + w["width"] for w in line["words"])
        return l, r

    n = len(lines)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    bounds = [line_bounds(l) for l in lines]
    for i in range(n):
        for j in range(i + 1, n):
            li, lj = lines[i], lines[j]
            h_i, h_j = li["bottom"] - li["top"], lj["bottom"] - lj["top"]
            avg_h = (h_i + h_j) / 2.0
            gap = (lj["top"] - li["bottom"]) if li["top"] <= lj["top"] else (li["top"] - lj["bottom"])

            l_i, r_i = bounds[i]
            l_j, r_j = bounds[j]
            overlap = max(0, min(r_i, r_j) - max(l_i, l_j))
            min_w = min(r_i - l_i, r_j - l_j) or 1
            x_overlap_ratio = overlap / min_w

            height_ratio = max(h_i, h_j) / max(min(h_i, h_j), 1)

            if gap < avg_h * gap_factor and x_overlap_ratio > min_x_overlap and height_ratio < max_height_ratio:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(lines[i])
    paragraphs = list(groups.values())
    for p in paragraphs:
        p.sort(key=lambda l: l["top"])
    paragraphs.sort(key=lambda p: (p[0]["top"], min(w["left"] for w in p[0]["words"])))
    return paragraphs

def is_likely_logo(para_lines, max_word_count=1, max_chars=4):
    """
    Deteksi paragraf yang kemungkinan besar logo/watermark, bukan
    konten yang perlu diedit: 1 baris, 1 kata pendek (<=4 karakter).
    Paragraf seperti ini kita biarin di background asli (nggak
    di-inpaint, nggak dijadikan elemen teks HTML).
    """
    if len(para_lines) > 1:
        return False
    words = para_lines[0]["words"]
    if len(words) > max_word_count:
        return False
    text = " ".join(w["text"] for w in words).strip()
    return len(text) <= max_chars

# ============================================================
# 3. STYLE ESTIMATION
# Pisahkan pixel teks dari background (Otsu), ambil warna dominan
# + estimasi ukuran font per paragraf.
# ============================================================

def glyph_mask_for_box(gray, left, top, w, h, pad=3):
    """
    Ambil crop di sekitar 1 kata (+ sedikit padding), lalu pisahkan
    pixel 'teks' dari 'background' pakai Otsu threshold.
    Asumsi: pixel teks itu MINORITAS di dalam box (background lebih
    dominan), jadi kalau hasil threshold malah mayoritas putih,
    kita balik (invert).
    """
    H, W = gray.shape[:2]
    x0 = max(0, left - pad)
    y0 = max(0, top - pad)
    x1 = min(W, left + w + pad)
    y1 = min(H, top + h + pad)
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return None, x0, y0
    _, th = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = np.count_nonzero(th) / th.size
    if white_ratio > 0.5:
        th = cv2.bitwise_not(th)
    return th, x0, y0


def estimate_word_color_and_mask(image_bgr, gray, word):
    """Kembalikan (warna_bgr, mask_full_size) untuk 1 kata."""
    H, W = gray.shape[:2]
    mask, x0, y0 = glyph_mask_for_box(gray, word["left"], word["top"], word["width"], word["height"])
    full_mask = np.zeros((H, W), dtype=np.uint8)
    color = (30, 30, 30)  # default abu gelap kalau gagal
    if mask is not None and mask.any():
        full_mask[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]] = mask
        ys, xs = np.where(mask > 0)
        pixels = image_bgr[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]][ys, xs]
        if len(pixels):
            color = tuple(int(c) for c in np.median(pixels, axis=0))
    return color, full_mask


def bgr_to_hex(bgr):
    b, g, r = bgr
    return "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))


def build_paragraph_record(image_bgr, gray, para_lines, img_w, img_h):
    """
    Hitung bounding box gabungan + style (warna, ukuran font, alignment,
    dst) untuk 1 paragraf (list of lines), sekaligus kumpulin glyph mask.

    KNOWN LIMITATION: posisi & lebar box diestimasi murni dari bounding
    box kata hasil OCR, bukan dari elemen desain asli (card/kolom).
    Pada body-text yang wrap ke banyak baris secara tidak simetris,
    titik tengah hasil estimasi bisa meleset beberapa px dari desain
    aslinya -- terlihat pada header 1-baris yang di-recenter mengikuti
    body-text di bawahnya (lihat harmonize_column_headers).
    """
    line_texts = []
    line_lefts, line_rights = [], []
    all_colors, all_heights = [], []
    glyph_areas, box_areas = [], []
    combined_mask = np.zeros(gray.shape[:2], dtype=np.uint8)

    for line in para_lines:
        words = line["words"]
        line_texts.append(" ".join(w["text"] for w in words))
        line_lefts.append(min(w["left"] for w in words))
        line_rights.append(max(w["left"] + w["width"] for w in words))
        for w in words:
            color, mask = estimate_word_color_and_mask(image_bgr, gray, w)
            all_colors.append(color)
            all_heights.append(w["height"])
            glyph_areas.append(int(mask.sum() / 255))
            box_areas.append(w["width"] * w["height"])
            combined_mask = cv2.bitwise_or(combined_mask, mask)

    all_words = [w for line in para_lines for w in line["words"]]
    left = min(w["left"] for w in all_words)
    top = min(w["top"] for w in all_words)
    right = max(w["left"] + w["width"] for w in all_words)
    bottom = max(w["top"] + w["height"] for w in all_words)

    median_h = float(np.percentile(all_heights, 85)) if all_heights else 20.0
    font_size_px = median_h * 0.92

    if len(para_lines) > 1:
        tops = [line["top"] for line in para_lines]
        gaps = [b - a for a, b in zip(tops, tops[1:])]
        line_height_px = float(np.median(gaps))
    else:
        line_height_px = font_size_px * 1.25

    color_bgr = tuple(int(c) for c in np.median(np.array(all_colors), axis=0)) if all_colors else (30, 30, 30)

    stroke_density = (sum(glyph_areas) / sum(box_areas)) if box_areas else 0.18
    font_weight = 700 if stroke_density > 0.35 else 400

    # deteksi alignment: kalau titik tengah tiap baris relatif konsisten
    # (variasinya kecil dibanding variasi posisi kiri tiap baris),
    # berarti teks itu center-aligned di desain aslinya.
    para_center = (left + right) / 2.0
    line_centers = [(l + r) / 2.0 for l, r in zip(line_lefts, line_rights)]
    if len(line_centers) > 1:
        center_dev = float(np.std([c - para_center for c in line_centers]))
        left_dev = float(np.std([l - left for l in line_lefts]))
        text_align = "center" if center_dev < left_dev * 0.6 else "left"
    else:
        text_align = "left"

    pad_x = (right - left) * 0.06
    pad_y = (bottom - top) * 0.15

    record = {
        "text": "\n".join(line_texts),
        "left": left - pad_x, "top": top - pad_y,
        "width": (right - left) + pad_x * 2, "height": (bottom - top) + pad_y * 2,
        "font_size_px": round(font_size_px, 1),
        "line_height_px": round(line_height_px, 1),
        "color": bgr_to_hex(color_bgr),
        "font_weight": font_weight,
        "text_align": text_align,
    }
    return record, combined_mask

def harmonize_column_headers(records, img_w, width_ratio_thresh=0.6, title_width_ratio=0.4):
    """
    (penjelasan sama seperti sebelumnya)
    Elemen yang lebarnya > 40% lebar gambar dianggap judul/elemen
    full-width (bukan bagian dari 1 kolom card), jadi di-exclude total
    dari proses union supaya nggak "menjembatani" 2 kolom berbeda
    jadi 1 grup yang salah.
    """
    n = len(records)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    is_title_like = [r["width"] > img_w * title_width_ratio for r in records]

    for i in range(n):
        if is_title_like[i]:
            continue
        for j in range(i + 1, n):
            if is_title_like[j]:
                continue
            a, b = records[i], records[j]
            a_l, a_r = a["left"], a["left"] + a["width"]
            b_l, b_r = b["left"], b["left"] + b["width"]
            overlap = max(0, min(a_r, b_r) - max(a_l, b_l))
            min_w = min(a_r - a_l, b_r - b_l) or 1
            if overlap / min_w > 0.4:
                union(i, j)

    groups = {}
    for i in range(n):
        if is_title_like[i]:
            continue
        groups.setdefault(find(i), []).append(i)

    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        # pilih body-text (multi-baris) sebagai acuan titik tengah kalau
        # ada; kalau semua 1-baris, pakai yang paling lebar sebagai acuan
        multiline_idxs = [i for i in idxs if "\n" in records[i]["text"]]
        ref_idx = (
            max(multiline_idxs, key=lambda i: records[i]["width"])
            if multiline_idxs
            else max(idxs, key=lambda i: records[i]["width"])
        )
        ref = records[ref_idx]
        ref_center = ref["left"] + ref["width"] / 2.0

        for i in idxs:
            if i == ref_idx:
                continue
            r = records[i]
            if "\n" not in r["text"]:
                r["left"] = ref_center - r["width"] / 2.0
    return records

# ============================================================
# 4. INPAINTING
# Hapus teks asli dari background biar HTML overlay nggak dobel.
# ============================================================

def inpaint_background(image_bgr, full_text_mask, dilate=15, radius=20):
    """Hapus area teks (dari mask) di background pakai inpainting."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1))
    mask = cv2.dilate(full_text_mask, kernel, iterations=1)
    return cv2.inpaint(image_bgr, mask, radius, cv2.INPAINT_TELEA)

# ============================================================
# 5. HTML GENERATION
# Render paragraf jadi <div> contenteditable, posisi absolute,
# di atas background hasil inpainting.
# ============================================================

STAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  html, body {{
    margin: 0; padding: 0; background: #e9edf1;
    font-family: 'Segoe UI', Arial, Helvetica, sans-serif;
  }}
  .slide-viewport {{
    position: relative;
    width: 100%;
    max-width: {img_w}px;
    margin: 0 auto;
    aspect-ratio: {img_w} / {img_h};
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(0,0,0,0.18);
  }}
  .slide-stage {{
    position: absolute;
    top: 0; left: 0;
    width: {img_w}px;
    height: {img_h}px;
    transform-origin: top left;
    background: #ffffff;
  }}
  .slide-stage img.bg-layer {{
    position: absolute;
    top: 0; left: 0;
    width: {img_w}px;
    height: {img_h}px;
    display: block;
    user-select: none;
    pointer-events: none;
  }}
  .text-el {{
    position: absolute;
    box-sizing: border-box;
    overflow: hidden;
    white-space: pre-wrap;
    outline: none;
    cursor: text;
    padding: 0 2px;
  }}
  .text-el:hover {{
    outline: 1.5px dashed rgba(37, 99, 235, 0.55);
    outline-offset: 2px;
  }}
  .text-el:focus {{
    outline: 1.5px solid #2563eb;
    outline-offset: 2px;
    background: rgba(255,255,255,0.35);
  }}
</style>
</head>
<body>
<div class="slide-viewport" id="viewport">
  <div class="slide-stage" id="stage">
    <img class="bg-layer" src="{bg_src}" alt="background">
    {text_nodes}
  </div>
</div>
<script>
  const viewport = document.getElementById('viewport');
  const stage = document.getElementById('stage');
  const STAGE_W = {img_w};
  function rescale() {{
    const scale = viewport.clientWidth / STAGE_W;
    stage.style.transform = `scale(${{scale}})`;
  }}
  window.addEventListener('resize', rescale);
  window.addEventListener('load', rescale);
  new ResizeObserver(rescale).observe(viewport);
  rescale();
</script>
</body>
</html>
"""

TEXT_NODE_TEMPLATE = (
    '<div class="text-el" contenteditable="true" spellcheck="false" '
    'style="left:{left}px; top:{top}px; width:{width}px; height:{height}px; '
    'font-size:{font_size}px; line-height:{line_height}px; color:{color}; '
    'font-weight:{font_weight}; text-align:{text_align};">{text}</div>'
)


def render_html(img_w, img_h, bg_src, paragraphs, title="Slide"):
    nodes = []
    for p in paragraphs:
        text = htmllib.escape(p["text"]).replace("\n", "<br>")
        nodes.append(
            TEXT_NODE_TEMPLATE.format(
                left=p["left"], top=p["top"], width=p["width"], height=p["height"],
                font_size=p["font_size_px"], line_height=p["line_height_px"],
                color=p["color"], font_weight=p["font_weight"],
                text_align=p["text_align"], text=text,
            )
        )
    return STAGE_TEMPLATE.format(
        title=htmllib.escape(title), img_w=img_w, img_h=img_h,
        bg_src=bg_src, text_nodes="\n    ".join(nodes),
    )

def process_slide(image_path, out_dir="output"):
    """Proses 1 gambar slide end-to-end: OCR -> clustering -> style ->
    inpainting -> HTML. Return dict berisi path file yang dihasilkan."""
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(image_path))[0]

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = img.shape[:2]

    words = ocr_words_tiled(img)
    lines = cluster_lines(words)
    paragraphs = cluster_paragraphs(lines)
    paragraphs = [p for p in paragraphs if not is_likely_logo(p)]

    records = []
    full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for p in paragraphs:
        record, mask = build_paragraph_record(img, gray, p, img_w, img_h)
        records.append(record)
        full_mask = cv2.bitwise_or(full_mask, mask)

    records = harmonize_column_headers(records, img_w)

    clean_bg = inpaint_background(img, full_mask)
    bg_path = os.path.join(out_dir, f"{name}_bg.jpg")
    cv2.imwrite(bg_path, clean_bg, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    html_str = render_html(img_w, img_h, f"{name}_bg.jpg", records, title=name)
    html_path = os.path.join(out_dir, f"{name}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_str)

    return {"html": html_path, "background": bg_path, "n_paragraphs": len(records)}


def process_folder(folder="images", out_dir="output", pattern="slide*.jpg"):
    """Proses semua slide di 1 folder sekaligus."""
    import glob
    results = {}
    for path in sorted(glob.glob(os.path.join(folder, pattern))):
        print(f"Memproses {path} ...")
        res = process_slide(path, out_dir=out_dir)
        results[path] = res
        print(f"  -> {res['n_paragraphs']} blok teks -> {res['html']}")
    return results

# ============================================================
# 6. DRIVER
# ============================================================

if __name__ == "__main__":
    process_folder("images", "output")