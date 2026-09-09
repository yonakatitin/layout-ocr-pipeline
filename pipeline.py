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


if __name__ == "__main__":
    img = cv2.imread("images/slide1.jpg")
    h, w = img.shape[:2]
    tiles = make_tiles(w, h)
    print(f"Ukuran gambar: {w}x{h}")
    print(f"Jumlah tile: {len(tiles)}")
    for t in tiles:
        print(t, "-> lebar:", t[2] - t[0], "tinggi:", t[3] - t[1])

    print("\n--- OCR per tile, cari 'Profile Image Studio' ---")
    for (x0, y0, x1, y1) in tiles:
        crop = img[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        data = pytesseract.image_to_data(gray, lang=TESS_LANG, output_type=Output.DICT)
        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            conf = float(data["conf"][i])
            if text and conf > 45:
                print(f"tile{(x0,y0,x1,y1)} -> {text!r:20} conf={conf:.0f} left={x0+data['left'][i]} top={y0+data['top'][i]}")