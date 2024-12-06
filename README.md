# 2024 iFLYTEK A.I. 开发者大赛：多模态图文检索挑战赛方案

## 1. 项目简介

[多模态图文检索挑战赛](https://challenge.xfyun.cn/topic/info?type=graphic-retrieval-challenge&option=ssgy)旨在解决多模态图文检索问题，即根据查询文本从图像库中检索相关图像，我们使用了 **Chinese-CLIP** 模型，结合 **LoRA (Low-Rank Adaptation)** 技术，进行图像与文本的嵌入学习，并通过余弦相似度进行检索。在本次比赛中，团队**小老正**获得了第二名的好成绩。

## 2. 数据说明

### 数据源

数据来自于 iFLYTEK A.I. 开发者大赛的多模态图文检索挑战赛，具体可以从以下链接下载：

- [数据下载链接](https://challenge.xfyun.cn/topic/info?type=graphic-retrieval-challenge&option=stsj)
- 数据存放路径：`./xfdata`

### 数据格式

- `train.csv`: 训练集文件，包含图像的路径和对应的文本标题。
- `test_query.csv`: 测试集文件，包含查询文本。
- `image`: 存放图像文件的目录。
- `sample_submit.csv`: 提交格式的样本文件。

## 3. 方案概述

在本方案中，我们使用了 **Chinese-CLIP** 模型进行图像和文本的共同嵌入学习。具体实现过程中，我们将文本和图像输入模型，获取它们的嵌入向量（embedding），然后通过余弦相似度计算文本和图像之间的匹配度，最终进行检索。

### 方案步骤

1. **数据读取与预处理**

   首先，我们加载训练数据和测试数据，并对图像进行预处理，使其适配模型输入的要求。

   ```python
   df_train = pd.read_csv('../xfdata/train.csv', sep="\t")
   df_test_text = pd.read_csv('../xfdata/test_query.csv')

   for image_name in df_train['path'][: 1]:
       img_path = os.path.join('../xfdata/image', image_name)
       text = df_train.loc[df_train['path'] == image_name, 'title'].values[0]
       print(text)
       img = Image.open(img_path)
       plt.imshow(img)
       plt.show()
   ```

   以上代码首先读取 `train.csv` 文件和测试集 `test_query.csv` 文件，加载图像路径和标题。接着，通过 `PIL` 库加载并展示图像。为了让模型适应不同的图像输入，图像需要进行翻转处理。

2. **模型加载**

   我们使用了 **Chinese-CLIP** 预训练模型，该模型支持中文文本和图像的多模态输入。通过 `transformers` 库，我们加载预训练模型和处理器。

   ```python
   from transformers import ChineseCLIPProcessor, ChineseCLIPModel
   import torch

   device = "cuda" if torch.cuda.is_available() else "cpu"
   model_path = "/root/onethingai-tmp/models--OFA-Sys--chinese-clip-vit-huge-patch14/snapshots/503e16b560aff94c1922f13a86a7693d36957a4f"
   model = ChineseCLIPModel.from_pretrained(model_path).to(device)
   processor = ChineseCLIPProcessor.from_pretrained(model_path)
   ```

   在此部分代码中，我们使用 `ChineseCLIPProcessor` 和 `ChineseCLIPModel` 来处理输入的图像和文本，并通过 `.to(device)` 将模型加载到GPU（如果可用）进行训练。

3. **LoRA 微调**

   在训练过程中，我们使用了 **LoRA** 技术来减少计算量并提高训练效率。LoRA的核心思想是只调整Transformer网络中的低秩矩阵，而不调整整个网络的参数，这样可以在保证性能的同时，减少训练时的计算开销。

   ```python
   from peft import get_peft_model, LoraConfig

   target_modules = []
   for i in range(24):
       target_modules.append(f"text_model.encoder.layer.{i}.attention.self.query")
       target_modules.append(f"text_model.encoder.layer.{i}.attention.self.key")
       target_modules.append(f"text_model.encoder.layer.{i}.attention.self.value")

   for i in range(32):
       target_modules.append(f"vision_model.emcoder.layers.{i}.self_attn.q_proj")
       target_modules.append(f"vision_model.emcoder.layers.{i}.self_attn.k_proj")
       target_modules.append(f"vision_model.emcoder.layers.{i}.self_attn.v_proj")

   lora_config = LoraConfig(
       r=64,
       lora_alpha=96,
       target_modules=target_modules
   )

   lora_model = get_peft_model(model, lora_config)
   ```

   我们定义了 `LoRAConfig` 来指定哪些模块的低秩矩阵需要进行微调。在这里，我们微调了文本模型和视觉模型中的注意力模块（QKV部分）。

4. **训练模型**

   在训练过程中，我们采用了标准的训练循环，并使用 `AdamW` 优化器进行优化。每个epoch都会计算训练损失和准确率，并更新模型的参数。

   ```python
   def train_epoch(model, dataloader, optimizer, device):
       model.train()
       total_loss = 0
       all_preds = []
       all_labels = []
       progress_bar = tqdm(dataloader, desc="Training", leave=False, dynamic_ncols=True)

       for batch in progress_bar:
           images = [Image.open(image_path).transpose(Image.FLIP_LEFT_RIGHT).convert("RGB") for image_path in batch["image_path"]]
           inputs = processor(text=batch["title"], images=images, return_tensors="pt", padding=True, truncation=True, max_length=52).to(device)
           outputs = model(**inputs, return_loss=True)
           loss = outputs.loss
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           total_loss += loss.item()
           preds = torch.argmax(outputs["logits_per_text"], dim=1).detach().cpu().numpy()
           all_preds.extend(preds)
           all_labels.extend(np.arange(outputs["logits_per_text"].shape[0]))
           progress_bar.set_postfix(loss=loss.item())

       avg_loss = total_loss / len(dataloader)
       acc = accuracy_score(all_labels, all_preds)
       return avg_loss, acc
   ```

   这里的 `train_epoch` 函数实现了模型的训练过程。我们首先将图像和文本输入模型，计算损失并进行反向传播以更新权重。

5. **预测与检索**

   训练完成后，我们使用训练好的模型进行图像和文本的嵌入提取，并计算查询文本与图像之间的相似度。

   ```python
   model.eval()
   dataset_test_image = Dataset.from_pandas(df_test_image)

   def get_image_embed(batch):
       with torch.no_grad():
           image_paths = [os.path.join('../xfdata/image', image_name) for image_name in batch["path"]]
           images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
           pixel_values = processor(text=None, images=images, return_tensors="pt")["pixel_values"].to(device)
           image_embeds = model.get_image_features(pixel_values)
           batch["image_embeds"] = image_embeds
           return batch

   dataset_test_image = dataset_test_image.map(get_image_embed, batched=True, batch_size=256)
   ```

   `get_image_embed` 函数处理图像数据，提取每张图像的特征向量，并存储在 `image_embeds` 中。同样的过程也用于文本数据。通过计算文本和图像嵌入的余弦相似度，我们可以从图像库中检索与查询文本最相关的图像。

6. **提交结果**

   最后，我们根据相似度矩阵生成预测结果，并保存为 `submit.csv` 文件，提交给比赛平台。

   ```python
   df_submit = pd.read_csv("../xfdata/sample_submit.csv")
   df_submit["path"] = df_test_image.loc[index_not_train[similar_matrix.argmax(axis=1)], "path"].values
   df_submit.to_csv("../prediction_result/submit.csv", index=False)
   ```

## 4. 算力平台

为了高效训练模型，我们使用了 [onethingai](https://onethingai.com/invitation?code=wGZHFckZ) 提供的算力平台。该平台提供了强大的GPU资源，使我们能够在较短的时间内完成模型训练和微调。

## 5. 贡献者

- **团队名称**：小老正
- **成员**：[孟子正]