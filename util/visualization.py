import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

def draw_rectangles_with_labels_from_npz(entity, label, in_the_image, vis, npz_path, image_path, output_path):
    """
    在图像上绘制矩形框及其对应的实体标签和名字。

    :param entity: 实体名字
    :param label: 实体的标签
    :param in_the_image: 实体是否在图像中
    :param vis: 包含要绘制的 bounding_boxes 索引的数组
    :param npz_path: 输入 .npz 文件路径
    :param image_path: 输入图像路径
    :param output_path: 输出图像路径
    """
    # 加载 .npz 文件
    data = np.load(npz_path)

    bounding_boxes = data['bounding_boxes']

    # 打开图像
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # 加载字体
    try:
        font = ImageFont.truetype("arial.ttf", 10)
    except IOError:
        font = ImageFont.load_default()

    if in_the_image:
        for v in vis:
            if v < len(bounding_boxes):
                # 获取指定的 bounding box
                box = bounding_boxes[v]
                x1, y1, x2, y2 = box

                # 绘制矩形框
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

                # 拼接文本标签
                text = f"{entity} ({label})"

                # 获取文本尺寸
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

                # 确定文本位置，文本框略大于文本
                text_background = (x1, y1 - text_height - 4, x1 + text_width + 4, y1)
                draw.rectangle(text_background, fill="red")
                draw.text((x1 + 2, y1 - text_height - 4), text, fill="white", font=font)

    # 将图像模式转换为 RGB，以便保存为 JPEG 格式
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # 保存输出图像
    image.save(output_path)
# 示例调用
entity = "Kevin Durant"
label = "person"
in_the_image = True
vis = [2, ]
npz_path = "../data/img_vinvl/O_4154.jpg.npz"
image_path = "../data/img/O_4154.jpg"
output_path = "output.jpg"

draw_rectangles_with_labels_from_npz(entity, label, in_the_image, vis, npz_path, image_path, output_path)
