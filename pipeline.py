import math
import cv2
import pytesseract
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

TESS_LANG = "ind+eng"


def make_tiles(img_w, img_h, target_tile_px=3200, overlap=0.25):
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


def dedup_words(words, iou_thresh=0.4):
    """Buang deteksi duplikat akibat overlap antar tile.
    Diurutkan by confidence tinggi dulu -> panjang teks -> yang menang disimpan."""
    words = sorted(words, key=lambda w: (-w["conf"], -len(w["text"])))
    kept = []
    for w in words:
        if any(iou(w, k) > iou_thresh for k in kept):
            continue
        kept.append(w)
    return kept


def ocr_words_tiled(image_bgr, lang=TESS_LANG, min_conf=45, target_tile_px=3200, overlap=0.25):
    """OCR seluruh gambar lewat tiling + dedup, hasilnya list of dict per kata."""
    h, w = image_bgr.shape[:2]
    tiles = make_tiles(w, h, target_tile_px=target_tile_px, overlap=overlap)

    all_words = []
    for (x0, y0, x1, y1) in tiles:
        crop = image_bgr[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        data = pytesseract.image_to_data(gray, lang=lang, output_type=Output.DICT)
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
    - Confidence tinggi (>=85) + minimal 1 huruf/angka -> dipercaya.
    - Confidence sedang (>=60) -> harus terlihat kayak kata beneran
      (minimal 4 karakter alfanumerik, bukan simbol doang).
    """
    t = text.strip()
    if not t:
        return False
    alnum = sum(ch.isalnum() for ch in t)  # huruf DAN angka dihitung
    if conf >= min_high_conf and alnum >= 1:
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

if __name__ == "__main__":
    for n in range(1, 6):
        img = cv2.imread(f"images/slide{n}.jpg")
        words = ocr_words_tiled(img)
        print(f"\n=== slide{n}.jpg -> {len(words)} kata terdeteksi ===")
        for w in sorted(words, key=lambda w: w["top"]):
            print(f"  {w['text']!r:20} conf={w['conf']:.0f} left={w['left']} top={w['top']}")