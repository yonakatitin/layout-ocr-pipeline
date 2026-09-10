import math
import cv2
import pytesseract
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

TESS_LANG = "ind+eng"
TESS_CONFIGS = ["--psm 3", "--psm 11"]


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
    """OCR seluruh gambar lewat tiling (psm3 + psm11) + dedup + filter noise."""
    h, w = image_bgr.shape[:2]
    tiles = make_tiles(w, h, target_tile_px=target_tile_px, overlap=overlap)

    all_words = []
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
    Hitung bounding box gabungan + style (warna, ukuran font, dst)
    untuk 1 paragraf (list of lines), sekaligus kumpulin glyph mask-nya
    (nanti dipakai buat inpainting di langkah berikutnya).
    """
    line_texts = []
    all_colors, all_heights = [], []
    glyph_areas, box_areas = [], []
    combined_mask = np.zeros(gray.shape[:2], dtype=np.uint8)

    for line in para_lines:
        words = line["words"]
        line_texts.append(" ".join(w["text"] for w in words))
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

    median_h = float(np.median(all_heights)) if all_heights else 20.0
    font_size_px = median_h / 0.70  # perkiraan cap-height -> font-size CSS

    if len(para_lines) > 1:
        tops = [line["top"] for line in para_lines]
        gaps = [b - a for a, b in zip(tops, tops[1:])]
        line_height_px = float(np.median(gaps))
    else:
        line_height_px = font_size_px * 1.25

    color_bgr = tuple(int(c) for c in np.median(np.array(all_colors), axis=0)) if all_colors else (30, 30, 30)

    stroke_density = (sum(glyph_areas) / sum(box_areas)) if box_areas else 0.18
    font_weight = 700 if stroke_density > 0.30 else 400

    record = {
        "text": "\n".join(line_texts),
        "left": left, "top": top,
        "width": right - left, "height": bottom - top,
        "font_size_px": round(font_size_px, 1),
        "line_height_px": round(line_height_px, 1),
        "color": bgr_to_hex(color_bgr),
        "font_weight": font_weight,
    }
    return record, combined_mask

if __name__ == "__main__":
    img = cv2.imread("images/slide3.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    words = ocr_words_tiled(img)
    lines = cluster_lines(words)
    paragraphs = cluster_paragraphs(lines)

    img_h, img_w = img.shape[:2]
    for p in paragraphs:
        record, mask = build_paragraph_record(img, gray, p, img_w, img_h)
        print(f"text={record['text'][:40]!r:42} color={record['color']} font_size={record['font_size_px']} weight={record['font_weight']}")