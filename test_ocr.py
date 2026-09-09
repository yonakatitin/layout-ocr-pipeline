import cv2
import pytesseract
from pytesseract import Output

# karena tesseract belum di PATH windows
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# baca salah satu gambar slide
img = cv2.imread("images/slide1.jpg")
print("Ukuran gambar:", img.shape)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# jalankan OCR, ambil data per-kata (bounding box + teks + confidence)
data = pytesseract.image_to_data(gray, lang="ind+eng", output_type=Output.DICT)

count = 0
for i in range(len(data["text"])):
    text = data["text"][i].strip()
    conf = float(data["conf"][i])
    if text and conf > 50:
        print(f"text={text!r:25} conf={conf:.0f}  left={data['left'][i]:5} top={data['top'][i]:5} w={data['width'][i]:4} h={data['height'][i]:4}")
        count += 1
    if count >= 15:
        break