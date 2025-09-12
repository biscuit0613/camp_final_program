import os
import random
import shutil

# 配置参数
val_percent = 0.1  # 验证集比例
img_dir = 'dataset/images'
label_dir = 'dataset/labels'
train_img_dir = 'dataset/images/train'
val_img_dir = 'dataset/images/val'
train_label_dir = 'dataset/labels/train'
val_label_dir = 'dataset/labels/val'

os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
random.shuffle(img_files)
val_num = int(len(img_files) * val_percent)
val_imgs = set(img_files[:val_num])
train_imgs = set(img_files[val_num:])

for img_set, img_dst, label_dst in [
    (train_imgs, train_img_dir, train_label_dir),
    (val_imgs, val_img_dir, val_label_dir)
]:
    for img_name in img_set:
        # 复制图片
        shutil.copy(os.path.join(img_dir, img_name), os.path.join(img_dst, img_name))
        # 复制label
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_src_path = os.path.join(label_dir, label_name)
        if os.path.exists(label_src_path):
            shutil.copy(label_src_path, os.path.join(label_dst, label_name))

print(f"数据集划分完成，验证集数量: {len(val_imgs)}，训练集数量: {len(train_imgs)}")