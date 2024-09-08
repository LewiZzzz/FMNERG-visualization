import numpy as np
import torch

def prepare_single_example(sentence, tokenizer, img_path_vinvl, vinvl_region_number=36):
    # 将文本转换为模型输入格式
    tokenized_input = tokenizer.encode_plus(
        sentence,
        max_length=200,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )

    image_boxes = np.zeros((vinvl_region_number, 4), dtype=np.float32)
    image_feature = np.zeros((vinvl_region_number, 2048), dtype=np.float32)

    img = np.load(img_path_vinvl)
    image_num = img['num_boxes']
    image_feature_ = img['box_features']

    # normalize
    image_feature_ = (image_feature_ / np.sqrt((image_feature_ ** 2).sum()))

    final_num = min(image_num, vinvl_region_number)
    image_feature[:final_num] = image_feature_[:final_num]
    image_boxes[:final_num] = img['bounding_boxes'][:final_num]

    vis_attention_mask = [1] * int(final_num)
    vis_attention_mask.extend([0] * int(vinvl_region_number - final_num))

    # 将 vis_feat 和 vis_attention_mask 转换为 PyTorch 张量
    vis_feat = torch.tensor(image_feature, dtype=torch.float32).unsqueeze(0)
    vis_attention_mask = torch.tensor(vis_attention_mask, dtype=torch.float32).unsqueeze(0)

    return tokenized_input, vis_feat, vis_attention_mask


def run_inference(model, tokenizer, sentence, img_path_vinvl, vinvl_region_number=36):
    # 准备输入
    tokenized_input, img_feat, vis_attention_mask = prepare_single_example(
        sentence,  tokenizer, img_path_vinvl, vinvl_region_number
    )

    # 将输入送入模型
    with torch.no_grad():
        # 生成文本，使用自定义的 generate 方法
        output, vis_similarities = model.model.generate_VisionT5(
            input_ids=tokenized_input['input_ids'].to(model.device),
            attention_mask=tokenized_input['attention_mask'].to(model.device),
            vis_feats=img_feat.to(model.device),
            vis_attention_mask=vis_attention_mask.to(model.device),
            max_length=200,
            num_beams=1,
            vinvl_region_number=36
        )


    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print("dec:", generated_text)

    return generated_text