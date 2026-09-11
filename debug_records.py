import cv2
import numpy as np
import pipeline

img = cv2.imread("images/slide3.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_h, img_w = img.shape[:2]

words = pipeline.ocr_words_tiled(img)
lines = pipeline.cluster_lines(words)
paragraphs = pipeline.cluster_paragraphs(lines)
paragraphs = [p for p in paragraphs if not pipeline.is_likely_logo(p)]

for p in paragraphs:
    record, _ = pipeline.build_paragraph_record(img, gray, p, img_w, img_h)
    print(f"text={record['text'][:30]!r:32} left={record['left']:.0f} width={record['width']:.0f} font_size={record['font_size_px']} align={record['text_align']}")

print("\n--- detail kata di paragraf Cloud Computing (body text) ---")
for p in paragraphs:
    if "Solusi" in p[0]["words"][0]["text"] or any("Solusi" in w["text"] for line in p for w in line["words"]):
        for line in p:
            for w in line["words"]:
                print(f"{w['text']!r:15} height={w['height']} conf={w['conf']:.0f}")