import cv2
import numpy as np
from collections import Counter
from tqdm import tqdm
from numba import jit

def hex_to_BGR(hex_color):
    # 去掉 '#' 符号并解析RGB值
    hex_color = hex_color.lstrip('#')
    bgr_color = np.array([int(hex_color[i:i + 2], 16) for i in (4, 2, 0)])
    return bgr_color

def load_image(filename):
    # 读取图像
    image = cv2.imread(filename)
    if image is None:
        raise FileNotFoundError(f"Cannot open image file: {filename}")
    return image

def simplifyAndSeparate(image):
    global colors
    global colornames
    global scale_factor
    global imgname
    global masks_images
    # 设置缩放倍数
    # scale_factor = 1
    # 计算目标图像的宽度和高度
    target_width = int(image.shape[1] * scale_factor)
    target_height = int(image.shape[0] * scale_factor)
    # 缩放图像
    myimage = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    # 遍历图像的每个像素
    # 使用tqdm


    print(f"开始简化分离{imgname}")

    # 初始化每种颜色的遮罩图像
    for color in colors:
        masks_images.append(np.zeros_like(myimage))

    for i in tqdm(range(myimage.shape[0]), desc=f"简化{imgname}"):
            j_s(myimage,masks_images,i,colors) # 将像素设置为对应的颜色


    return myimage

#@jit
def j_s(myimage,masks_images, i,colors):
    for j in range(myimage.shape[1]):
        pixel = myimage[i, j]#获取[i,j]像素值

            # 计算像素颜色与定义的颜色值的欧氏距离
            #distances = {}
            #for color, value in colors.items():
            #    distances[color] = np.linalg.norm(pixel - value)
            #min_distance_color = min(distances, key=distances.get)
            #j_s(myimage,i,j,colors[min_distance_color])

            # 计算像素颜色与定义的颜色值的欧氏距离，不使用字典

        # 找到最小距离对应的颜色，并将像素设置为该颜色
        distances = np.linalg.norm(pixel - colors, axis=1) # 计算每种颜色的欧氏距离，返回一个数组，数组的索引对应颜色索引
        min_distance_color_index = np.argmin(distances) # 找到最小距离对应的颜色索引

        masks_images[min_distance_color_index][i,j] = myimage[i, j] = colors[min_distance_color_index]


def simplify(image):
    global colors
    global scale_factor
    global imgname
    # 设置缩放倍数
    # scale_factor = 1
    # 计算目标图像的宽度和高度
    target_width = int(image.shape[1] * scale_factor)
    target_height = int(image.shape[0] * scale_factor)
    # 缩放图像
    myimage = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    # 遍历图像的每个像素
    # 使用tqdm


    print(f"开始简化{imgname}")

    for i in tqdm(range(myimage.shape[0]), desc=f"简化{imgname}"):
        for j in range(myimage.shape[1]):
            pixel = myimage[i, j]

            # 计算像素颜色与定义的颜色值的欧氏距离
            distances = {}
            for color, value in colors.items():
                distances[color] = np.linalg.norm(pixel - value)
            min_distance_color = min(distances, key=distances.get)

            j_s(myimage,i,j,colors[min_distance_color])

    return myimage


def separate(image):
    global colors
    # 将图像转换为RGB格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 保存每种颜色的遮罩图像
    masks = []
    for color in colors:
        # 创建一个与输入图像尺寸相同的 黑色背景图像
        # 遍历colors中的每个颜色
        for i in tqdm(range(image_rgb.shape[0]), desc=f"分离{color}"):
            mask = np.zeros_like(image_rgb)
            masks.append(j_se(mask,image_rgb,i,color))

    return masks

def j_se(mask,image_rgb,i,color):
    for j in range(image_rgb.shape[1]):
    # 如果像素颜色与当前颜色匹配，则将其复制到遮罩图像中
        if np.all(image_rgb[i, j] == color):
            mask[i, j] = np.array([255, 255, 255])
    return mask


def combine_2image(image1, image2,combine_name):
    # 确保两张图片具有相同的尺寸
    global comb_index
    assert image1.shape == image2.shape, "图片尺寸不匹配"

    # 创建一个与输入图像尺寸相同的空白图像
    output_image = np.zeros_like(image1)

    # 遍历输入图像的每个像素
    for y in tqdm(range(image1.shape[0]), desc=f"合并{combine_name}"):
        j_c(image1,y,0,image2,output_image)

    return output_image

@jit
def j_c(image1,y,x,image2,output_image):
    for x in range(image1.shape[1]):
                # 如果任意一个像素不是黑色，则将其复制到输出图像中
                if (not np.all(image1[y, x] == 0)
                        or not np.all(image2[y, x] == 0)):
                    output_image[y, x] = np.array([255, 255, 255])

@jit
def blackelsewhite(image):
    #所有不是黑色的像素都变成白色
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if not np.all(image[i, j] == 0):
                image[i, j] = np.array([255, 255, 255])
    return image



def smooth(image):
    # 定义膨胀核大小和迭代次数
    kernel_size = 5 
    iterations = 1 
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    image = cv2.dilate(image, kernel, iterations=iterations)
    image = cv2.erode(image, kernel, iterations=iterations)
    image = cv2.erode(image, kernel, iterations=iterations)
    image = cv2.dilate(image, kernel, iterations=iterations)

    return image

def save_image(image, filename):
    # 保存图像
    cv2.imwrite(filename, image)


# 配置
hexColors = {  # BGR表示
    "DBlue": "#161F7D", #"#161F7D", [0]
    "Blue": "#5DA7E3",  #"#5DA7E3", [1]
    "DGreen": "#193522",#"#193522", [2]
    "Green": "#F9E195", #"#F9E195", [3]
    "Black": "#061008", #"#061008", [4]
    "White": "#E6EAEB", #"#E6EAEB", [5]
}
scale_factor = 1 # 设置缩放倍数, 0.25表示缩小到原来的四分之一



#imgpath = "origin.jpg" # 设置待处理图片路径
imgpath = input("请输入待处理图片路径> ")
if imgpath == "":
    imgpath = "origin.jpg"

# 初始化
#colors = {color: hex_to_BGR(value) for color, value in hexColors.items()}
#列表colors
colornames = list(hexColors.keys())
colors = [hex_to_BGR(value) for value in hexColors.values()]
imgname = imgpath.split('/')[-1].split('.')[0]

output_directory = f"{imgname}_output"
import os
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 工作流开始
# 读取图像
origin_image = load_image(imgpath)

# 简化图像
masks_images=[]
simplified_image = simplifyAndSeparate(origin_image)

# 保存简化后的图像

save_image(simplified_image, f"{imgname}_output/simplified.png")

# 分离颜色
#masks_images = separate(simplified_image)

# 保存每种颜色的遮罩图像
for mask, color in zip(masks_images, colornames):
    #save_image(mask, f"{imgname}_output/mask_{colors[idx]}.png")
    save_image(mask, f"{imgname}_output/mask_{color}.png")
# 合并遮罩图像

# 正面阻焊层：深绿+浅绿+白+黑 （即浅蓝+深蓝后取反）
#layer_front_mask = combine_2image(masks_images[1], masks_images[0], "正面阻焊层")
#layer_front_mask = cv2.bitwise_not(layer_front_mask)
# 浅绿+深绿+黑(因为嘉立创EDA中黑色遮罩的位置会去除阻焊层，所以需要透明的地方只有浅绿和深绿，所以合并这两个颜色即可)
layer_front_mask = combine_2image(masks_images[3], masks_images[2], "正面阻焊层1/2")
layer_front_mask = combine_2image(layer_front_mask, masks_images[5], "正面阻焊层2/2")
layer_front_mask = smooth(layer_front_mask)
save_image(layer_front_mask, f"{imgname}_output/{imgname}_layer_front_mask.png")

# 正面铜皮层：浅蓝+黑 （铜皮裸露即没有阻焊层但有铜皮层，所以类似焊盘）
layer_front_copper = combine_2image(masks_images[1], masks_images[4], "正面铜皮层")
#layer_front_copper = combine_2image(layer_front_copper, masks_images[2], "正面铜皮层2/2")
layer_front_copper = cv2.bitwise_not(layer_front_copper)
layer_front_copper = smooth(layer_front_copper)
save_image(layer_front_copper, f"{imgname}_output/{imgname}_layer_front_copper.png")

# 正面丝印层：白
layer_front_silk = blackelsewhite(masks_images[5])
layer_front_silk = smooth(layer_front_silk)
save_image(layer_front_silk, f"{imgname}_output/{imgname}_layer_front_silk.png")

# 背面阻焊层：深绿是背面有阻焊层，浅绿是背面没阻焊层,左右镜像处理
layer_back_mask = blackelsewhite(masks_images[3])
#layer_back_mask = cv2.flip(layer_back_mask, 1)
layer_back_mask = smooth(layer_back_mask)
save_image(layer_back_mask, f"{imgname}_output/{imgname}_layer_back_mask.png")

# 背面铜皮层：深绿有没有无所谓，有了更有金属的冷峻感，没有就显得死板。
layer_back_copper = blackelsewhite(masks_images[1])
#layer_back_copper = cv2.flip(layer_back_copper, 1)
layer_back_copper = smooth(layer_back_copper)
save_image(layer_back_copper, f"{imgname}_output/{imgname}_layer_back_copper.png")


