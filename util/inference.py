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

class EmptyResultError(Exception):
    """自定义异常类，当返回结果为空时抛出"""
    pass

# 定义自定义异常类
class EmptyGeneratedTextError(Exception):
    """自定义异常类，当生成的文本为空时抛出"""
    pass

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
            min_length=10,
            num_beams=1,
            vinvl_region_number=36
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # 检查生成文本是否为空
    if not generated_text.strip():
        print("Generated text is empty.")
        raise EmptyGeneratedTextError("The generated text is empty. Check the input data or model configuration.")

    print("generate_text:", generated_text)

    vis_prediction = turn_vis_similarities_to_vis_pred(vis_similarities, output, vinvl_region_number)

    print("vis_pred:", vis_prediction)


    triplets, coarse_dict = extract_spans_para_quad_single(generated_text, vis_prediction)

    print("triplets:", triplets)
    print("coarse_dict", coarse_dict)

    return generated_text, triplets, coarse_dict



def turn_vis_similarities_to_vis_pred(vis_similarities, outs, vinvl_region_number):
    """
    vis_similarities: [bts, max_length, vis_box_num]
    outs: [bts, max_length]
    """
    mask_for_classifier_index = []
    for pred in outs:
        _list_total = 0
        this_batch = []
        flag = True
        for token in pred:
            if token == 59:
                this_batch.append(False)
                this_batch.extend([False] * _list_total)
                _list_total = 0
                flag = False
                continue
            if token not in [16, 8, 1023]:  # the ids of "in the image"
                this_batch.append(False)
                this_batch.extend([False] * _list_total)
                _list_total = 0
            elif token == [16, 8, 1023][_list_total]:
                if not flag:
                    this_batch.append(False)
                    this_batch.extend([False] * _list_total)
                    _list_total = 0
                    flag = True
                    continue
                _list_total += 1
                if _list_total == 3:
                    this_batch.extend([True, True, True])
                    _list_total = 0
            else:
                this_batch.append(False)
                this_batch.extend([False] * _list_total)
                _list_total = 0
        this_batch.extend([False] * _list_total)
        mask_for_classifier_index.append(this_batch)

    mask_for_classifier_index = torch.tensor(mask_for_classifier_index)
    vis_similarities = vis_similarities[mask_for_classifier_index]
    if len(vis_similarities) == 0:
        return vis_similarities
    vis_pred = torch.argmax(vis_similarities.view(-1, 3, vinvl_region_number).mean(dim=1), dim=1)

    return vis_pred


def extract_spans_para_quad_single(seq, vis_pred):
    triplets = {}
    coarse_dict = {}

    sents = [s.strip() for s in seq.split('[SSEP]')]
    idx = 0
    for s in sents:
        try:
            part_one, part_two = s.split(', which')
            entity, labels = part_one.split(' is a ')
            coarse_label, fine_label = labels.split(' and a ')

            # 处理 "in the image" 和 "not in the image" 两种情况
            if 'not in the image' in part_two:
                in_the_image = False
            elif 'in the image' in part_two:
                in_the_image = True
            else:
                raise ValueError(f"Invalid sentence structure in part_two: {part_two}")

            # 如果是 in_the_image 且 vis_pred 有足够的数据，则访问 vis_pred
            if in_the_image and idx < len(vis_pred):
                vis = vis_pred[idx]
            else:
                vis = [None]
            idx += 1

        except ValueError as ve:
            print(f"ValueError occurred while parsing sentence: {s}. Error: {ve}")
            entity, fine_label, coarse_label, in_the_image, vis = '', '', '', '', []
        except IndexError as ie:
            print(f"IndexError: vis_pred index out of range at idx {idx}. Error: {ie}")
            entity, fine_label, coarse_label, in_the_image, vis = '', '', '', '', []
        except Exception as e:
            print(f"Unexpected error in parsing sequence: {s}. Error: {e}")
            entity, fine_label, coarse_label, in_the_image, vis = '', '', '', '', []

        # 检查 entity 和标签是否为空，如果为空，抛出自定义异常
        if not entity or not fine_label or not coarse_label:
            print(f"Parsed values are empty for sentence: {s}")
            continue  # 跳过这一项，但不终止整个处理

        triplets[f'{entity} {fine_label} {in_the_image}'] = vis
        coarse_dict[f'{entity} {coarse_label} {in_the_image}'] = vis

        # 在返回之前检查 triplets 或 coarse_dict 是否为空
        if not triplets or not coarse_dict:
            raise EmptyResultError("The extracted triplets or coarse dictionary is empty.")

    return triplets, coarse_dict

