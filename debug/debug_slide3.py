import cv2
import pytesseract
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img = cv2.imread("images/slide3.jpg")
crop = img[0:1828, 0:3276]
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

data = pytesseract.image_to_data(gray, lang="ind+eng", config="--psm 11", output_type=Output.DICT)
for i in range(len(data["text"])):
    t = data["text"][i].strip()
    if t:
        print(f"{t!r:20} conf={data['conf'][i]:.0f} left={data['left'][i]} top={data['top'][i]}")