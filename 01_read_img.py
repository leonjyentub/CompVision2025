import cv2
import numpy as np
from PIL import Image

'''
OpenCV 和 Pillow 讀取的顏色通道順序不同：

OpenCV: BGR
Pillow: RGB

OpenCV 讀取的圖片是以 BGR 格式存儲的，而 Pillow 讀取的圖片是以 RGB 格式存儲的。

# BGR 轉 RGB
rgb_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

# RGB 轉 BGR
bgr_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

# Pillow 轉 OpenCV
img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# OpenCV 轉 Pillow
img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

# 全彩轉灰階
gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

# Pillow 轉灰階
gray_img = img_pil.convert('L')
'''

fileName = '108501594_p0_master1200.jpg'
# 使用 OpenCV 讀取圖片
img_cv = cv2.imread(fileName)
print("OpenCV 讀取的圖片陣列:")
print('type:', type(img_cv))
print("Shape:", img_cv.shape)  # 輸出維度 (height, width, channels)
print("Data type:", img_cv.dtype)  # 輸出數據類型
print("Array data:")
print(img_cv)
print("\n")

# 使用 Pillow 讀取圖片
img_pil = Image.open(fileName)
# 轉換為 numpy 陣列
img_array = np.array(img_pil)
print("Pillow 讀取的圖片陣列:")
print('type:', type(img_array))
print("Shape:", img_array.shape)
print("Data type:", img_array.dtype)
print("Array data:")
print(img_array)

# 顯示一些基本資訊
print("\n基本資訊:")
print("圖片大小:", img_pil.size)  # (width, height)
print("圖片模式:", img_pil.mode)  # RGB, RGBA, L 等

# 注意：OpenCV 讀取圖片是 BGR 格式
# Pillow 讀取圖片是 RGB 格式
# 使用 Pillow 顯示
img_pil.show()

# 使用 OpenCV 顯示
cv2.imshow('OpenCV Image', img_cv)
cv2.waitKey(5000)
cv2.destroyAllWindows()

import matplotlib.pyplot as plt
plt.imshow(img_array)
plt.axis('off')  # 不顯示座標軸
plt.show()