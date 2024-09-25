import cv2
import pytesseract

from appbots.core.utils import get_path

# img_path = get_path("assets/test.png")
img_path = get_path("assets/test2.jpg")

# 读取图像
img = cv2.imread(img_path)

# 预处理
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# 字符识别
text = pytesseract.image_to_string(thresh, config='--psm 6', lang='chi_sim')  # psm 6表示单行文本

if __name__ == '__main__':
    print(text)
