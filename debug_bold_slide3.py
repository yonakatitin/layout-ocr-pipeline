import cv2
import pipeline

img = cv2.imread("images/slide3.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
words = pipeline.ocr_words_tiled(img)
lines = pipeline.cluster_lines(words)
paragraphs = pipeline.cluster_paragraphs(lines)
paragraphs = [p for p in paragraphs if not pipeline.is_likely_logo(p)]

for p in paragraphs:
    glyph_areas, box_areas = [], []
    for line in p:
        for w in line["words"]:
            _, mask = pipeline.estimate_word_color_and_mask(img, gray, w)
            glyph_areas.append(int(mask.sum() / 255))
            box_areas.append(w["width"] * w["height"])
    density = sum(glyph_areas) / sum(box_areas)
    text_preview = " ".join(w["text"] for line in p for w in line["words"])[:35]
    print(f"{text_preview!r:38} stroke_density={density:.3f}")