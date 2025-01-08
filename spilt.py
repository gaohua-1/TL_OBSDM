import shutil
import random
import os

# 原始路径
image_original_path = r"D:/BaiduNetdiskDownload/yolov7-main/dataset1/images/"
label_original_path = r"D:/BaiduNetdiskDownload/yolov7-main/dataset1/labels/"

cur_path = os.getcwd()
# 训练集路径
train_image_path = os.path.join(cur_path, r"D:/BaiduNetdiskDownload/yolov7-main/dataset2/images/train/")
train_label_path = os.path.join(cur_path, r"D:/BaiduNetdiskDownload/yolov7-main/dataset2/labels/train/")

# 验证集路径
val_image_path = os.path.join(cur_path, r"D:/BaiduNetdiskDownload/yolov7-main/dataset2/images/val/")
val_label_path = os.path.join(cur_path, r"D:/BaiduNetdiskDownload/yolov7-main/dataset2/labels/val/")

# 训练集目录
list_train = os.path.join(cur_path, r"D:/BaiduNetdiskDownload/yolov7-main/dataset2/train.txt")
list_val = os.path.join(cur_path, r"D:/BaiduNetdiskDownload/yolov7-main/dataset2/val.txt")

train_percent = 0.9
val_percent = 0.1

def del_file(path):
    for i in os.listdir(path):
        file_data = path + "\\" + i
        os.remove(file_data)

def mkdir():
    if not os.path.exists(train_image_path):
        os.makedirs(train_image_path)
    else:
        del_file(train_image_path)
    if not os.path.exists(train_label_path):
        os.makedirs(train_label_path)
    else:
        del_file(train_label_path)

    if not os.path.exists(val_image_path):
        os.makedirs(val_image_path)
    else:
        del_file(val_image_path)
    if not os.path.exists(val_label_path):
        os.makedirs(val_label_path)
    else:
        del_file(val_label_path)

def clearfile():
    if os.path.exists(list_train):
        os.remove(list_train)
    if os.path.exists(list_val):
        os.remove(list_val)

def main():
    mkdir()
    clearfile()

    file_train = open(list_train, 'w')
    file_val = open(list_val, 'w')

    total_txt = os.listdir(label_original_path)
    num_txt = len(total_txt)
    list_all_txt = range(num_txt)

    num_train = int(num_txt * train_percent)
    num_val = int(num_txt * val_percent)

    train = random.sample(list_all_txt, num_train)
    val = [i for i in list_all_txt if i not in train]

    print("训练集数目：{}, 验证集数目：{}".format(len(train), len(val)))
    for i in list_all_txt:
        name = total_txt[i][:-4]

        srcImage = image_original_path + name + '.jpg'
        srcLabel = label_original_path + name + ".txt"

        if i in train:
            dst_train_Image = train_image_path + name + '.jpg'
            dst_train_Label = train_label_path + name + '.txt'
            shutil.copyfile(srcImage, dst_train_Image)
            shutil.copyfile(srcLabel, dst_train_Label)
            file_train.write(dst_train_Image + '\n')
        elif i in val:
            dst_val_Image = val_image_path + name + '.jpg'
            dst_val_Label = val_label_path + name + '.txt'
            shutil.copyfile(srcImage, dst_val_Image)
            shutil.copyfile(srcLabel, dst_val_Label)
            file_val.write(dst_val_Image + '\n')

    file_train.close()
    file_val.close()

if __name__ == "__main__":
    main()
