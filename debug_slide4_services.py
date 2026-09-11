import cv2
import pipeline

img = cv2.imread("images/slide4.jpg")
words = pipeline.ocr_words_tiled(img)

print("--- cari kata mengandung 'ervic' ---")
found = False
for w in words:
    if "ervic" in w["text"].lower():
        print(f"FOUND: {w}")
        found = True
if not found:
    print("Nggak ketemu sama sekali di hasil ocr_words_tiled!")

print("\n--- semua kata di area hexagon kanan-bawah (kira2 left>3000, top>2200) ---")
for w in words:
    if w["left"] > 3000 and w["top"] > 2200:
        print(f"{w['text']!r:20} conf={w['conf']:.0f} left={w['left']} top={w['top']}")