# S0: 先拿英文的建立一个完整框架
# 捕获用户输入的text,调用clip的model.encode_text函数对其进行编码，和数据库中的embedding一一匹配
import clip
import torch
from prepare import *


if __name__=='__main__':
    # 第一次要调用prepare
    # prepare()

    # 将image_embedding加载进入内存
    image_features, get_name=get_list()

    VISIBLE_CUDA_DEVICES='2'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    text="a pig"

    # 将下面这一部分换成自己训练的部分
    text_input=clip.tokenize(text)

    text_features=model.encode_text(text_input)

    text_features/=text_features.norm(dim=-1,keepdim=True)

    # 逐一匹配image和text，计算相似度
    similarity = (100*text_features@image_features.T).softmax(dim=-1)

    # S4: 返回topk的图片，并展示
    value,indices = similarity[0].topk(5)

    print('Top 5 predictions\n')

    for value,index in zip(value,indices):
        print(get_name[index])
        # import cv2
        # img=cv2.imread(get_name[index])
        # cv2.imshow("image",img)




