from PIL import Image, ImageDraw, ImageFont
import numpy as np


def draw_rectangles_with_labels_from_triplets(triplets, coarse_dict, npz_path, image_path, output_path):
    """
    在图像上绘制矩形框及其对应的实体标签和名字。
    :param triplets: 包含细粒度标签和是否在图像中的三元组
    :param coarse_dict: 包含粗粒度标签和是否在图像中的三元组
    :param npz_path: 输入 .npz 文件路径，包含 bounding boxes
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

    # 遍历 triplets 和 coarse_dict
    for triplet_key, vis in triplets.items():
        coarse_key = [key for key in coarse_dict if triplet_key.split()[0] in key][0]

        entity, fine_label, in_the_image_str = triplet_key.rsplit(' ', 2)
        _, coarse_label, _ = coarse_key.rsplit(' ', 2)

        in_the_image = True if in_the_image_str == 'True' else False

        if in_the_image and vis is not None and vis != [None]:
            vis_idx = int(vis)
            if vis_idx < len(bounding_boxes):
                # 获取 bounding box
                box = bounding_boxes[vis_idx]
                x1, y1, x2, y2 = box

                # 绘制矩形框
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

                # 拼接文本标签，显示为：实体名字（粗粒度：细粒度）
                text = f"{entity} ({coarse_label}: {fine_label})"

                # 获取文本尺寸
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

                # 确定文本背景框的大小，略大于文本
                text_background = (x1, y1, x1 + text_width + 4, y1 + text_height + 4)
                draw.rectangle(text_background, fill="red")
                # 在背景框上绘制文本
                draw.text((x1 + 2, y1 + text_height - 8), text, fill="white", font=font)

    # 将图像模式转换为 RGB，以便保存为 JPEG 格式
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # 保存输出图像
    image.save(output_path)


# # 示例调用
# triplets = {'Jolie actor True': 8, 'Donald Trump politician False': [None]}
# coarse_dict = {'Jolie person True': 8, 'Donald Trump person False': [None]}
#
# npz_path = "../data/img_vinvl/O_2371.jpg.npz"
# image_path = "../data/img/O_2371.jpg"
# output_path = "output2.jpg"
#
# draw_rectangles_with_labels_from_triplets(triplets, coarse_dict, npz_path, image_path, output_path)
