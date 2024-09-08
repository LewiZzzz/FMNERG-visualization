# FMNER 可视化系统

该项目是一个基于 `T5` 模型和视觉推理的命名实体识别 (NER) 系统，支持从输入文本和图片中抽取三元组信息，并将识别出的实体和视觉对象进行可视化展示。系统使用 `PyTorch Lightning` 框架进行模型训练与推理。

## 目录
- [环境配置](#环境配置)
- [安装教程](#安装教程)
- [模型下载](#模型下载)
- [项目结构](#项目结构)
- [使用方法](#使用方法)
- [参考文献](#参考文献)

## 环境配置

本项目推荐使用 `Python 3.7.x` 版本，主要依赖的库包括 `Transformers`, `PyTorch`, 和 `PyTorch Lightning`。为了确保版本一致性，请按照以下步骤进行安装。

## 安装教程

### 国内镜像源安装指南

由于网络原因，国内用户推荐使用华为云的镜像源进行安装，以提高安装速度。可以通过以下步骤安装所需的依赖库。

1. **安装 `Transformers` 库**（用于处理 T5 模型）：

    ```bash
    pip install transformers==4.0.0 -i https://mirrors.huaweicloud.com/repository/pypi/simple/
    ```

2. **安装 `sentencepiece`**（用于处理 T5 模型的分词）：

    ```bash
    pip install sentencepiece==0.1.91 -i https://mirrors.huaweicloud.com/repository/pypi/simple/
    ```

3. **安装 `PyTorch Lightning`**（用于模型训练和推理）：

    ```bash
    pip install pytorch_lightning==0.8.1 -i https://mirrors.huaweicloud.com/repository/pypi/simple/
    ```

4. **安装 `lmdb`**（用于数据加载和存储）：

    ```bash
    pip install lmdb -i https://mirrors.huaweicloud.com/repository/pypi/simple/
    ```

5. **安装 `setuptools`**（确保安装正确的版本来避免依赖问题）：

    ```bash
    pip install setuptools==59.5.0 -i https://mirrors.huaweicloud.com/repository/pypi/simple/
    ```

6. **安装 `torchvision`**（用于处理视觉数据）：

    ```bash
    pip install torchvision==0.7.0 -i https://mirrors.huaweicloud.com/repository/pypi/simple/
    ```

7. **安装 `boto3`**（用于处理 AWS 相关服务，如果不需要可以省略）：

    ```bash
    pip install boto3 -i https://mirrors.huaweicloud.com/repository/pypi/simple/
    ```

8. **安装**`streamlit`（用于产生前端页面）

    ```bash
    pip install streamlit -i https://mirrors.huaweicloud.com/repository/pypi/simple/
    pip install st-annotated-text
    ```

### GPU 加速 (可选)

如果需要使用 GPU 进行加速，可以按照 [PyTorch 官方文档](https://pytorch.org/get-started/locally/) 安装支持 CUDA 的 `torch` 和 `torchvision` 版本。

## 模型下载

本项目需要使用以下模型及预训练权重文件：

1. **T5-base** 模型：这是一个经过广泛使用的文本生成模型，适合处理 NER 和生成任务。你可以通过 `Transformers` 库直接下载该模型：
   
    ```bash
    from transformers import T5Tokenizer, T5ForConditionalGeneration
   
    model = T5ForConditionalGeneration.from_pretrained('t5-base')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    ```

2. **TIGER 模型权重**：该模型是经过自定义训练的 NER 和视觉推理模型。请下载模型权重文件并将其放置在 `model/` 目录下。权重文件可以联系项目负责人获取，或根据实际需求通过以下命令加载：

    ```bash
    # 示例加载权重的方式
    model.load_state_dict(torch.load('model/TIGER_weights.ckpt'))
    ```

## 项目结构

```
├── data/                          # 数据文件夹
│   ├── img/                       # 上传的图片
│   └── img_vinvl/                 # 存放图片对应的 .npz 文件
├── model/                         # 存放模型文件和权重
│   ├── t5-base/                   # T5 模型配置文件
│   └── TIGER_weights.ckpt         # TIGER 训练模型权重
├── output/                        # 可视化输出图片文件夹
├── util/                          # 实用函数
│   ├── inference.py               # 推理函数
│   └── visualization.py           # 可视化函数
├── main.py                        # 主程序
├── README.md                      # 项目说明文件
└── requirements.txt               # 依赖库
```

## 使用方法

### 1. 运行系统

运行 `streamlit` 以启动前端可视化页面：

```bash
streamlit run main.py
```

### 2. 输入数据

- 在页面中，输入一段英文推文，并上传一张图片文件。
- 系统会自动匹配 `data/img_vinvl/` 目录下与上传图片文件名对应的 `.npz` 文件，并进行推理和可视化展示。

### 3. 输出结果

- 系统会在前端页面中显示推理结果和生成的图片。生成的图片会标注文本中提到的实体和视觉对象。
- 所有生成的图片都会保存到 `output/` 目录中。

## 示例

### 输入

推文: 
```
News Update Angelina Jolie slams Donald Trump ' s stance on religious freedom and immigration
```

上传的图片：`O_2371.jpg`

### 输出

系统将展示标注了视觉对象（例如 Angelina Jolie 和 Donald Trump）的图片，以及与文本对应的实体识别结果。

---

## 参考文献

1. [Hugging Face Transformers 文档](https://huggingface.co/docs/transformers/)
2. [PyTorch 官方文档](https://pytorch.org/docs/)
3. [Streamlit 官方文档](https://docs.streamlit.io/)
4. [Jieming Wang, Ziyan Li, Jianfei Yu, Li Yang, and Rui Xia. 2023. Fine-Grained Multimodal Named Entity Recognition and Grounding with a Generative Framework. In Proceedings of the 31st ACM International Conference on Multimedia (MM '23). Association for Computing Machinery, New York, NY, USA, 3934–3943.](https://doi.org/10.1145/3581783.3612322)
5. https://github.com/NUSTM/FMNERG
