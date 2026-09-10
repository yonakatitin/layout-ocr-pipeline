import cv2
import pipeline

for n in [1, 4]:
    img = cv2.imread(f"images/slide{n}.jpg")
    words = pipeline.ocr_words_tiled(img)
    print(f"\n=== slide{n} ===")
    for w in words:
        if w["text"].strip() in ("N", "4", "A", "IS"):
            print(f"{w['text']!r:6} conf={w['conf']:.0f} left={w['left']} top={w['top']} width={w['width']} height={w['height']}")