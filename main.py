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

print(time)

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
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        config = create_config()
        tfm_model = VisionT5(config)
        model = T5FineTuner(tfm_model=tfm_model, tokenizer=tokenizer)

        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])

        model.eval()
        model.model.eval()

        return model, tokenizer
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        return None, None

# 加载模型
model_path = 'model/t5-base'
checkpoint_path = 'model/test2.ckpt'
model, tokenizer = load_model_and_tokenizer(model_path, checkpoint_path)


# 创建三个列来排放按钮，并设置不同颜色
col1, col2, col3 = st.columns(3)

# 初始化 session_state 以存储按钮状态和路径
if "sentence" not in st.session_state:
    st.session_state.sentence = "News Update Angelina Jolie slams Donald Trump 's stance on religious freedom and immigration"
if "image_path" not in st.session_state:
    st.session_state.image_path = None
if "img_path_vinvl" not in st.session_state:
    st.session_state.img_path_vinvl = None

if col1.button('Test 1'):
    st.session_state.sentence = "RT @ hahnsmith : Minnie enjoying his first morning at the lake ."
    st.session_state.image_path = "data/img/1311544.jpg"
    st.session_state.img_path_vinvl = "data/img_vinvl/1311544.jpg.npz"

if col2.button('Test 2'):
    st.session_state.sentence = "And here he is as Clark Kent . . . ."
    st.session_state.image_path = "data/img/153305.jpg"
    st.session_state.img_path_vinvl = "data/img_vinvl/153305.jpg.npz"

if col3.button('Test 3'):
    st.session_state.sentence = "Chicago nights in the sky . Summer forever"
    st.session_state.image_path = "data/img/70560.jpg"
    st.session_state.img_path_vinvl = "data/img_vinvl/70560.jpg.npz"

# 文本输入框，显示点击按钮后的句子或默认句子
sentence = st.text_area("请输入英文推文", value=st.session_state.sentence, height=100)

# 图片上传
uploaded_image = st.file_uploader("或者上传图片", type=["jpg", "jpeg", "png"])

# 显示上传的图片或按钮设置的图片
if uploaded_image is not None:
    # 如果上传了图片，显示上传的图片并生成 img_path_vinvl
    st.session_state.image_path = os.path.join("data/img", uploaded_image.name)
    with open(st.session_state.image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())
    st.image(st.session_state.image_path, caption="上传的图片", use_column_width=True)

    # 根据上传的图片名称生成 img_path_vinvl
    st.session_state.img_path_vinvl = os.path.splitext(uploaded_image.name)[0] + '.jpg.npz'
    st.session_state.img_path_vinvl = os.path.join('data/img_vinvl', st.session_state.img_path_vinvl)
else:
    # 如果没有上传图片，显示按钮点击后的图片
    if st.session_state.image_path:
        st.image(st.session_state.image_path, caption="测试图片", use_column_width=True)

# 使用 st.columns 来创建一个占满全行的按钮
col_full = st.columns(1)
with col_full[0]:
    # 让按钮占满整行
    st.markdown("""
    <style>
    div.stButton > button:first-child {
        width: 100%;
        height: 50px;
        font-size: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    confirm_clicked = st.button("确定")


# 只有在点击“确定”按钮时才开始推理
if confirm_clicked:
    try:
        img_path_vinvl = st.session_state.img_path_vinvl
        if img_path_vinvl and os.path.exists(img_path_vinvl):
            # 运行推理
            with st.spinner('模型推理中...'):
                generated_text, triplets, coarse_dict = run_inference(model, tokenizer, sentence, img_path_vinvl)

            # 生成输出图片
            output_image_path = os.path.join("output", "output.jpg")
            draw_rectangles_with_labels_from_triplets(triplets, coarse_dict, img_path_vinvl, st.session_state.image_path, output_image_path)

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
            st.error(f"找不到 {img_path_vinvl} 对应的 .npz 文件。请检查文件是否正确。")
    except Exception as e:
        st.error(f"推理过程中出现错误: {str(e)}")