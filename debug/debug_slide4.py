import cv2
import pipeline

img = cv2.imread("images/slide4.jpg")
words = pipeline.ocr_words_tiled(img)
for w in sorted(words, key=lambda w: w["top"]):
    if w["text"] in ("Human", "Resource", "Filosofi", "Sarang", "Lebah", "Services", "Research", "Center"):
        print(f"{w['text']!r:12} left={w['left']:5} top={w['top']:5} width={w['width']:5} height={w['height']:5}")