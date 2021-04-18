import os
import shutil

def copy(li,path,save_path='coco_matched/'):
    imgs = os.listdir(path)
    # img_id=[]
    # for img in imgs:
    #     id = img.split('_')[2][:-4]
    #     num = int(id)
    #     id=str(num)
    #     img_id.append(id)
    #
    # for id in li:
    #     if id not in img_id:
    #         pass

    cnt = 0
    for img in imgs:
        id = img.split('_')[2][:-4]
        num = int(id)
        id = str(num)

        if id in li:
            shutil.copy(path + img, save_path + img)
            cnt+=1
            print(cnt)
        # else:
        #     print("image not found！ id=",id)

def load_and_save_image(path='/home/origin/yszhu/coco/coco/images/'):
    with open("matched_total.txt","r") as f:
        lines=f.readlines()
        li=[]
        for line in lines:
            if line=='\n':
                continue
            li.append(line.split('$$')[1].strip())
            li.append(line.split('$$')[2].strip())
    copy(li, path + 'train2014/')
    copy(li, path + 'val2014/')

# 根据图片名称裁剪出string类型的id

def get_id(img):
    id = img.split('_')[2][:-4]
    num = int(id)
    id = str(num)
    return id

# load_and_save_image()
import clip

print(clip.available_models())