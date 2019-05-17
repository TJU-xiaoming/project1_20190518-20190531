"""基本操作任务
3. 读取一张图片
4. 显示图像 (cv2.imshow)
5. 保存图像 (cv2.imwrite)
6. 打印图片的尺寸 (shape)
7. 在图片上绘制一条直线， 直线坐标（0，0）（100,100）
8. 绘制矩形
9. 绘制圆形
10. 绘制椭圆
11. 绘制多边形
12. 向图像中添加文字（中文）
13. 将图片的颜色通道变换（BRG -> RGB）
14. 图像旋转(顺时针旋转30度)
15. 图像缩放至200×200×3
16. 图像向右平移20px， 向下平移50px
17. 透视变换
18. 仿射变换"""

import cv2 as cv
import numpy as np

img = cv.imread(r'.\image.jpg')
print('图片的尺寸：', img.shape)
height, width, channel = img.shape
print('height:', height, 'width:', width, 'channel:', channel)

"在图片上绘制一条红色直线， 直线坐标（0，0）（100,100）"
cv.line(img, (0, 0), (100, 100), (0, 0, 255))

"在图片上绘制一个蓝色矩形， 自定义坐标（0，0）（200,0）（200,100）（0,100）"
cv.line(img, (0, 0), (200, 0), (255, 0, 0))
cv.line(img, (200, 0), (200, 100), (255, 0, 0))
cv.line(img, (200, 100), (0, 100), (255, 0, 0))
cv.line(img, (0, 100), (0, 0), (255, 0, 0))
'画矩形方法2'
cv.rectangle(img, (0, 0), (200, 100), (255, 0, 0), 2)

'在图片上绘制一个绿色圆形，自定义原点（100,100），半径20'
cv.circle(img, (100, 100), 20, (0, 255, 0), 2)

'在图片上绘制一个绿色椭圆，自定义中心点（256,256），长轴100，短轴50，逆时针角度0，顺时针起始角度0和120°，图形填充'
# 画椭圆——需要输入中心点位置，长轴和短轴的长度，椭圆沿逆时针选择角度，椭圆沿顺时针方向起始角度和结束角度
cv.ellipse(img, (256, 256), (100, 50), 0, 0, 120, (0, 255, 0), -1)

'绘制一个简单多边形'
data = np.array([[10, 3], [48, 19], [60, 3], [98, 19]], np.int32)  # 注：数据类型必须是int32
data = data.reshape((-1, 1, 2))
cv.polylines(img, [data], True, (0, 0, 255), 1)# 如果第三个参数是 False，我们得到的多边形是不闭合的（首尾不相连）。参数：图像，点集，是否闭合，颜色，线条粗细
cv.imshow('picture show', img)
cv.imwrite('.\image_copy.jpg', img, [int(cv.IMWRITE_JPEG_QUALITY), 100])  # 高像素保存
cv.waitKey(0)


'添加英文'
import cv2
img = cv2.imread(r'.\image.jpg')
font = cv.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'A beautiful place', (100, 200), font, 1, (255, 0, 0), 2)
cv2.imshow('display_English',img)
cv2.waitKey(0)


'添加中文'
import cv2
import numpy
from PIL import Image, ImageDraw, ImageFont
img = cv2.imread(r'.\image.jpg')

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, numpy.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)

img = cv2ImgAddText(img, "风景真美丽！", 140, 140, (0, 0, 255), 50)
cv2.imshow('display_Chinese',img)
cv2.waitKey(0)

'将图片的颜色通道变换（BRG -> RGB）'
import cv2 as cv
img1 = cv.imread(r'.\image.jpg')
B, G, R = cv.split(img1)
img2 = cv.merge([R, G, B])
cv.imshow('1', B)
cv.imshow('2', G)
cv.imshow('3', R)
cv.imshow("BRG -> RGB", img2)
cv.waitKey(0)

' 图像旋转(顺时针旋转90度)'
import cv2
from math import *

img = cv2.imread(r".\image.jpg")
height, width = img.shape[:2]
degree = -30
# 旋转后的尺寸
heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
matRotation[0, 2] += (widthNew - width) / 2  # 因为旋转之后,坐标系原点是新图像的左上角,所以需要根据原图做转化
matRotation[1, 2] += (heightNew - height) / 2
imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
cv2.imshow("img_original", img)
cv2.imshow("img_Rotation", imgRotation)
cv2.waitKey(0)

'图像缩放至200×200×3'
import cv2 as cv

img = cv.imread(r'.\image.jpg')
print('original shape:',img.shape[:2])
# height, width = img.shape[:2]
# reSize = cv.resize(img, (int(width/2), int(height/2)), interpolation=cv2.INTER_CUBIC) 按比例缩放
reSize = cv.resize(img, (int(200), int(200)), interpolation=cv2.INTER_CUBIC)
cv.imshow('img_original', img)
cv.imshow('img_reSize', reSize)
cv.waitKey(0)

'将原图像向右平移20px,再向下平移50px'
import cv2 as cv
import numpy as np

img = cv.imread(r'.\image.jpg')
height, width = img.shape[:2]
matShift = np.float32([[1, 0, 20], [0, 1, 50]]) # 转换矩阵
dst = cv.warpAffine(img, matShift, (height, width))
cv.imshow('original', img)
cv.imshow('move', dst)
cv.waitKey(0)

'透视变换（空间变换）'
import cv2 as cv
import numpy as np

img = cv.imread(r'.\image.jpg')
pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
M = cv.getPerspectiveTransform(pts1, pts2)  # 找到变换矩阵
dst = cv.warpPerspective(img, M, (300, 300))
cv.imshow('original', img)
cv.imshow('PerspectiveTransform', dst)
cv.waitKey(0)


'仿射变换（平面变换）'
import cv2 as cv
import numpy as np

img = cv.imread(r'.\image.jpg')
height, width = img.shape[:2]
pts1 = np.float32([[50, 65], [150, 65], [210, 210]]) # 变换前3个点
pts2 = np.float32([[50, 100], [150, 65], [100, 250]]) # 变换后3个点
M = cv.getAffineTransform(pts1, pts2)
dst = cv.warpAffine(img, M, (height, width))
cv.imshow('original', img)
cv.imshow('AffineTransform', dst)
cv.waitKey(0)

