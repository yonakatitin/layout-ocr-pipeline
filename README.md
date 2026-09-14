# Layout-Aware Text Extraction Pipeline

A pipeline that extracts text from presentation slide images (along with its
position, color, and font size), then re-renders the result as an HTML
overlay that can be clicked and edited directly on top of the original
background image.

## Features

- Automatic text detection from slide images (Tesseract-based OCR)
- Estimation of position (bounding box), color, font size, and bold/normal
  weight per text block
- Removal of original text from the background (inpainting) so extracted
  text elements don't visually double up with the source text
- HTML output with directly clickable & editable text (`contenteditable`)
- Robust against multi-column layouts (side-by-side cards, hexagons, etc.)

## Requirements

- Python 3.10+
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (Windows
  build) with the Indonesian language pack (`ind`) installed
- Python packages: `opencv-python`, `numpy`, `pytesseract`, `pillow`

## Installation

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install opencv-python numpy pytesseract pillow
```

Install Tesseract OCR from the [UB-Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki),
making sure to check **Indonesian language data** during setup. Update the
`tesseract_cmd` path at the top of `pipeline.py` to match your installation
location:

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## Usage

1. Place slide images (`.jpg`) in the `images/` folder
2. Run:

```bash
python pipeline.py
```

3. Results are written to the `output/` folder: for each `images/slideN.jpg`,
   it produces `output/slideN.html` (the main file, open it in a browser)
   and `output/slideN_bg.jpg` (the inpainted background)

To process a single file or call it from another script:

```python
from pipeline import process_slide, process_folder

process_slide("images/slide1.jpg", out_dir="output")   # single file
process_folder("images", "output")                       # all files
```

## Pipeline Architecture

The end-to-end flow lives in `pipeline.py`, organized into 6 stages:

1. **Tiled OCR Extraction** — the image is split into an overlapping tile
   grid, each tile is OCR'd (psm3 + psm11), and results are merged and
   cleaned of duplicates/noise. Tiling is used because running OCR
   directly on the full-resolution image often fails to detect large,
   low-contrast text on visually busy backgrounds.
2. **Layout Clustering** — OCR'd words are grouped into lines, then lines
   are grouped into paragraphs, purely based on bounding-box geometry
   (union-find), making it robust against multi-column layouts.
3. **Style Estimation** — text color, font size, bold/normal weight, and
   text alignment are estimated per paragraph from pixel analysis (Otsu
   thresholding to separate glyphs from the background).
4. **Inpainting** — the original text is removed from the background
   (`cv2.inpaint`) based on the accumulated glyph mask, so the HTML
   overlay doesn't visually duplicate the original text.
5. **HTML Generation** — each paragraph is rendered as a
   `<div contenteditable>`, absolutely positioned on top of the inpainted
   background, with responsive auto-scaling via JavaScript.
6. **Driver** — `process_slide()` for a single file, `process_folder()`
   for batch-processing every file in a folder.

## Generalizability to Other Images

This pipeline is **not hardcoded** to the content or design of the 5 sample
slides — every stage works purely from pixel analysis and geometry. A few
parameters were tuned based on the visual characteristics of the sample
images (~5700px resolution, flat-design sans-serif fonts, Indonesian/English
text):

| Parameter | Location | Tuned based on |
|---|---|---|
| `font_weight` threshold (`0.32`) | `build_paragraph_record` | The stroke-density gap between bold and normal text in this font |
| `dilate`, `radius` for inpainting (`15`, `20`) | `inpaint_background` | Font size & image resolution |
| Confidence thresholds (`85`, `92`, `60`) | `looks_like_real_word` | OCR noise characteristics on these images |
| `lang="ind+eng"` | `TESS_LANG` | Content language |

For other slide images with similar visual characteristics (flat design,
high resolution, Indonesian/English text), the pipeline should work out of
the box without modification. For images that differ significantly
(photographic backgrounds, very low resolution, extremely decorative fonts,
other languages), these parameters may need adjustment, but the pipeline's
structure and logic remain applicable as-is.

## Known Limitations

- **Very wide words that fall exactly on a tile boundary** can occasionally
  get split (i.e. one word ends up as two incomplete detections). This is
  mitigated by using a fairly large tile overlap, but it doesn't eliminate
  the issue entirely for words wider than the overlap zone.
- **1-2 words on very busy backgrounds** (lots of graphic elements and
  lines) can sometimes go completely undetected by Tesseract, even though
  the text is clearly legible when isolated in a small crop. Adding a
  second, finer-grained tiling pass to catch these cases was attempted but
  caused regressions (duplicated text) elsewhere, so that approach was not
  kept.
- **Horizontal position of single-line headers** (e.g. short card titles)
  can be off by a few pixels from the column's true center, because the
  position is estimated from the body-text bounding box below it, which
  doesn't always accurately represent the true column width when the
  body-text wraps asymmetrically.
- **Logos/watermarks** (single-word, single-line paragraphs of ≤4
  characters) are excluded from text extraction (left untouched in the
  original background), since their bounding box often captures
  surrounding icon elements, making font-size estimation inaccurate if
  treated as regular text.

## Folder Structure

```
.
├── pipeline.py          # main pipeline code
├── images/              # input: slide images (.jpg)
├── output/               # output: .html files + inpainted backgrounds
├── requirements.txt
└── README.md
```