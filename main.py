import streamlit as st
import os
import re
import torch
import time
from PIL import Image
from transformers import T5Tokenizer, T5Config
from model.VisionT5 import VisionT5
from model.T5FineTuner import T5FineTuner
from util.inference import run_inference
from util.visualization import draw_rectangles_with_labels_from_triplets
from annotated_text import annotated_text


# Streamlit 页面标题
st.title("FMNER 可视化系统")

# 创建模型配置
def create_config():
    config = T5Config.from_pretrained('model/t5-base')
    config.feat_dim = 2048
    config.pos_dim = 36
    config.dropout_rate = 0.1
    config.dropout = 0.1
    config.attention_dropout = 0.1
    config.activation_dropout = 0.1
    config.vinvl_region_number = 36
    return config


# 加载模型和分词器
@st.cache_resource
def load_model_and_tokenizer(model_path, checkpoint_path):
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    config = create_config()
    tfm_model = VisionT5(config)
    model = T5FineTuner(tfm_model=tfm_model, tokenizer=tokenizer)

    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    model.model.eval()

    return model, tokenizer


# 加载模型
model_path = 'model/t5-base'
checkpoint_path = 'model/test.ckpt'
model, tokenizer = load_model_and_tokenizer(model_path, checkpoint_path)

# 推文输入
sentence = st.text_area("请输入英文推文",
                        "News Update Angelina Jolie slams Donald Trump 's stance on religious freedom and immigration")

# 图片上传
uploaded_image = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # 保存上传的图片
    image_path = os.path.join("data/img", uploaded_image.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())

    # 获取对应的 .npz 文件路径（假设 npz 文件和图片文件名相关）
    npz_filename = os.path.splitext(uploaded_image.name)[0] + '.jpg.npz'
    img_path_vinvl = os.path.join('data/img_vinvl', npz_filename)

    if os.path.exists(img_path_vinvl):
        # 运行推理
        with st.spinner('模型推理中...'):
            generated_text, triplets, coarse_dict = run_inference(model, tokenizer, sentence, img_path_vinvl)

        # 生成输出图片
        output_image_path = os.path.join("output", "output.jpg")
        draw_rectangles_with_labels_from_triplets(triplets, coarse_dict, img_path_vinvl, image_path, output_image_path)

        # 显示识别结果
        st.subheader("模型输出结果")
        

        st.success('Model Inferencing Succeeds!!!', icon="✅")

        lines = generated_text.split('[SSEP]')

        container = st.container()
        
        for line in lines:
            container.write(line.strip())
            

        # 提取并美观展示命名实体
        st.subheader("命名实体识别结果")
        entity, coarse_label, fine_label = None, None, None
        for line in lines:
            # 匹配格式：<命名实体> is a [粗粒度标签] and a [细粒度标签]
            match = re.search(r"(.+?) is a (\w+) and a (\w+), which is (\w+)", line)
            if match:
                entity = match.group(1)
                coarse_label = match.group(2)
                fine_label = match.group(3)
            annotated_text(
                (entity, "<ENTITY>"),
                " is a ",
                (coarse_label, "<COARSE_LABEL>"),
                " and a ",
                (fine_label, "<FINE_LABEL>"),
            )
            st.write("\n")


        # 显示处理后的图片
        st.subheader("视觉对象定位")
        st.image(output_image_path, caption="命名实体识别与对象定位", use_column_width=True)
    else:
        st.error(f"找不到与 {uploaded_image.name} 对应的 .npz 文件。请检查文件是否正确。")
