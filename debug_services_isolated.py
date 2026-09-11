import cv2
import pytesseract
from pytesseract import Output
import pipeline

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img = cv2.imread("images/slide4.jpg")
h, w = img.shape[:2]
tiles = pipeline.make_tiles(w, h)
print("Tiles:", tiles)

# cari tile mana yang mengandung area 'Services' (sekitar x=3300-4700, y=2050-2450)
for (x0, y0, x1, y1) in tiles:
    if x0 <= 3700 <= x1 and y0 <= 2250 <= y1:
        print(f"\nTile yang relevan: {(x0,y0,x1,y1)}, ukuran: {x1-x0}x{y1-y0}")
        crop = img[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        for psm in [3, 11]:
            data = pytesseract.image_to_data(gray, lang="ind+eng", config=f"--psm {psm}", output_type=Output.DICT)
            words = [(data['text'][i], data['conf'][i]) for i in range(len(data['text'])) if data['text'][i].strip() and "erv" in data['text'][i].lower()]
            print(f"  psm {psm}: cari 'Serv' -> {words if words else 'TIDAK KETEMU'}")