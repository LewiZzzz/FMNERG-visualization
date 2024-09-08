import os
import torch

from PIL import Image
from transformers import T5Tokenizer, T5Config
from model.VisionT5 import VisionT5
from model.T5Inference import T5Inference
from model.T5FineTuner import T5FineTuner
from util.inference import run_inference

# 示例：加载模型和分词器
model_path = 'model/t5-base'
checkpoint_path = 'model/test.ckpt'
def create_config():
    config = T5Config.from_pretrained(model_path)
    config.feat_dim = 2048
    config.pos_dim = 36
    config.dropout_rate = 0.1
    config.dropout = 0.1
    config.attention_dropout = 0.1
    config.activation_dropout = 0.1
    config.vinvl_region_number = 36

    return config

# 加载模型和分词器
def load_model_and_tokenizer(model_path, checkpoint_path):
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    config = create_config()
    tfm_model = VisionT5(config)
    # 先加载模型结构
    model = T5FineTuner(tfm_model=tfm_model, tokenizer=tokenizer)

    # 然后加载模型权重
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    # 将模型设置为 eval 模式
    model.eval()
    model.model.eval()

    return model, tokenizer


#
# def extract_spans_para_quad_single(seq, vis_label, img_id):
#     triplets = {}
#     coarse_dict = {}
#
#     sents = [s.strip() for s in seq.split('[SSEP]')]
#     idx = 0
#     for s in sents:
#         try:
#             part_one, part_two = s.split(', which')
#             entity, labels = part_one.split(' is a ')
#             coarse_label, fine_label = labels.split(' and a ')
#             in_the_image = 'in the image' in part_two
#             vis = vis_label[idx] if in_the_image else [None]
#             idx += 1
#
#         except Exception as e:
#             print(f'Error in parsing sequence: {s}. Error: {e}')
#             entity, fine_label, coarse_label, in_the_image, vis = '', '', '', '', []
#         triplets[f"{entity} {fine_label} {in_the_image}"] = vis
#         coarse_dict[f"{entity} {coarse_label} {in_the_image}"] = vis
#     return triplets, coarse_dict
#
#
# def parse_and_visualize_output(generated_text, img_id, img_path_vinvl):
#     # 解析生成的文本
#     triplets, coarse_dict = extract_spans_para_quad_single(generated_text, vis_pred=None, img_id=img_id)
#
#     # 可视化逻辑
#     visualize_entities_on_image(triplets, img_id, img_path_vinvl)


if __name__ == "__main__":
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(model_path, checkpoint_path)

    # sentence = "Humbling to attend Gold Star families event in Peoria. May God bless those who gave all and their families."
    sentence = "2 years ago today you were crying your eyes out watching Kevin Durant’s emotional MVP speech !"
    # img_path_vinvl = './data/img_vinvl/819987.jpg.npz'
    img_path_vinvl = './data/img_vinvl/O_4154.jpg.npz'

    # 运行推理
    generated_text = run_inference(model, tokenizer, sentence, img_path_vinvl)
