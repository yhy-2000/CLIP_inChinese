from utils import load_and_save_image
import os
import clip
import torch
from PIL import Image
import json
from utils import get_id

# 前期图片编码的准备
def prepare():

    # 首先加载数据库中的图片并保存对应文件夹
    # load_and_save_image()

    # 对每个图片进行编码，保存进database.txt
    VISIBLE_CUDA_DEVICES = '2'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    home_path = "./coco_matched/"
    # Load the model
    model, preprocess = clip.load('ViT-B/32', device)

    li = os.listdir(home_path)

    cnt = 0

    dict = {}
    for img_path in li:
        cnt += 1
        path = home_path + img_path
        img = Image.open(path)
        with torch.no_grad():
            image_input = preprocess(img).unsqueeze(0).to(device)
            image_features = model.encode_image(image_input)

            image_features/=image_features.norm(dim=-1,keepdim=True)

            id = get_id(img_path)

            dict[id] = {}
            dict[id]['feature'] = image_features.numpy().tolist()
            dict[id]['path'] = path

            if cnt % 20 == 0:
                with open("database.json", "w") as f:
                    json.dump(dict, f)

    with open("database.json", "w") as f:
        json.dump(dict, f)


# 将数据库中的image_embedding加载进内存,提高速度
def get_list(path='./database.json'):
    # with open(path,"r") as j:
    #     dict=json.load(j)
    # li=[]
    # for d in dict:
    #     li.append([dict[d]['feature'],dict[d]['path']])
    name=[]
    li=[]
    with open("database.txt","r") as f:
        lines=f.readlines()
        for i in range(len(lines)):
            lines[i]=eval(lines[i])
            name.append(lines[i][1])
            li.append(lines[i][0])

    li=torch.Tensor(li)
    li=torch.squeeze(li)
    return li,name
