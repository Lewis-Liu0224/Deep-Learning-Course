from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("lena.jpg")
img_r, img_g, img_b= img.split()
plt.rcParams['font.sans-serif']="SimHei"
plt.figure(figsize=(10, 10))


plt.subplot(221)
plt.axis("off")
img_small = img_r.resize((50, 50))
plt.imshow(img_small)
plt.title("R-缩放", fontsize=14)

'''
)将G通道的图像先水平镜像，
再顺时针旋转90度，显示在子图2中，
子标题为:“G-镜像+旋转”，字体大小为14，
并显示坐标轴；
'''
plt.subplot(222)
img_flr = img_g.transpose(Image.FLIP_LEFT_RIGHT)
img_r270 = img_flr.transpose(Image.ROTATE_270)
plt.imshow(img_r270)
plt.axis("on")
plt.title("G-镜像+旋转", fontsize=14)

'''
(3)对B通道的图像进行裁剪，裁剪位置：左上角(0, 0) 右下角(150, 150)，显示在子图3中，子标题为:“B-裁剪”，字体大小为14；
'''
plt.subplot(223)
img_cut = img_b.crop((0, 0, 150, 150))
plt.imshow(img_cut)
plt.axis("off")
plt.title("B-裁剪", fontsize=14)

'''
将原始的R、G、B通道的图像合并，显示在子图4中，子标题显示图像的色彩模式，字体大小为14；
'''
plt.subplot(224)
img_rgb = Image.merge("RGB",[img_r,img_g,img_b])
plt.axis("off")
plt.imshow(img_rgb)
plt.title("RGB", fontsize=14)

# (5)将要求(4)的处理结果保存为PNG格式的图片，路径为当前工作目录，文件名为“test.png”，如图2所示；
# (6)将以上生成的4幅图像显示在2×2的画布中，全局标题为“图像基本操作”，标题字体大小为20，颜色为蓝色。

img_rgb.save("test.png")

plt.suptitle("图像基本操作", fontsize=20, y=1, color="blue")

plt.show()
