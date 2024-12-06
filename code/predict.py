import pandas as pd 
from PIL import Image 
import matplotlib.pyplot as plt 
import os 


# 读取数据
# 读取数据
df_train = pd.read_csv('../xfdata/train.csv', sep="\t")
df_test_text = pd.read_csv('../xfdata/test_query.csv')

for image_name in df_train['path'][: 1]:
    img_path = os.path.join('../xfdata/image', image_name)
    # 需要将图片左右翻转，这样才能调正
    text = df_train.loc[df_train['path']==image_name, 'title'].values[0]
    print(text)
    img = Image.open(img_path)
    plt.imshow(img)
    plt.show()


from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import os
import torch 


device = "cuda" if torch.cuda.is_available() else "cpu"
# 设置镜像端点
os.environ["HF_ENDPOINT"] = "https://huggingface.co"
os.environ["TRANSFORMERS_CACHE"] = "hf-mirror"
model_path = "/root/onethingai-tmp/models--OFA-Sys--chinese-clip-vit-huge-patch14/snapshots/503e16b560aff94c1922f13a86a7693d36957a4f"
# model = ChineseCLIPModel.from_pretrained(model_path).to(device)
model = torch.load("../user_data/LoRA.pth")
processor = ChineseCLIPProcessor.from_pretrained(model_path)
print("Loading Done!")


from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from tqdm import tqdm 
import torch 
import numpy as np 

df_train = pd.read_csv('../xfdata/train.csv', sep="\t")
img_path = "../xfdata/image"
df_train["image_path"] = df_train["path"].apply(lambda x: os.path.join(img_path, x))
train_dataset = Dataset.from_pandas(df_train)
train_dataset


df_test_image = pd.DataFrame(os.listdir("../xfdata/image"))
df_test_image.columns = ["path"]
df_test_text = pd.read_csv('../xfdata/test_query.csv')
display(df_test_image.head(2))
display(df_test_text.head(2))


from datasets import Dataset


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

dataset_test_text = Dataset.from_pandas(df_test_text)
def get_text_embed(batch):
    with torch.no_grad():
        inputs = processor(text=batch["title"], images=None, return_tensors="pt", padding=True, truncation=True, max_length=52).to(device)
        text_embeds = model.get_text_features(**inputs)
        batch["text_embeds"] = text_embeds
        return batch
dataset_test_text = dataset_test_text.map(get_text_embed, batched=True, batch_size=512)

dataset_test_text.set_format("torch", columns=["text_embeds"])
dataset_test_image.set_format("torch", columns=["image_embeds"])
image_embeddings = dataset_test_image["image_embeds"].to(device)
text_embeddings = dataset_test_text["text_embeds"].to(device)
image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
np.save("../user_data/image_embeddings.npy", image_embeddings.detach().cpu().numpy())
np.save("../user_data/text_embeddings.npy", text_embeddings.detach().cpu().numpy())

# 在非训练集的图片中检索
index_not_train = df_test_image.loc[~df_test_image["path"].isin(df_train["path"])].index
similar_matrix = torch.mm(text_embeddings, image_embeddings[index_not_train].T).detach().cpu().numpy()

# 提交结果
df_submit = pd.read_csv("../xfdata/sample_submit.csv")
df_submit["path"] = df_test_image.loc[index_not_train[similar_matrix.argmax(axis=1)], "path"].values
df_submit.to_csv("../prediction_result/submit.csv", index=False)