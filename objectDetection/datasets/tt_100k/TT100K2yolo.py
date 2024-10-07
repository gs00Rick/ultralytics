import os
from random import random
import cv2
import shutil
import json
import xml.dom.minidom
from tqdm import tqdm
from objectDetection.utils import FilePathUtils


class TT100K2COCO:
    def __init__(self):
        self.original_datasets = 'tt_100k'
        self.to_datasets = 'coco'
        self.tt100k_path = os.path.join(FilePathUtils.get_project_root(), "objectDetection", "datasets", "tt_100k")

    def get_images_dir(self):
        """
        获取图像文件夹的路径

        该方法将TT100K数据集的根路径与"images"目录名结合，构造出图像文件夹的完整路径并返回

        :return: 图像文件夹的完整路径
        """
        images_path = os.path.join(self.tt100k_path, "images")
        return images_path

    def get_labels_dir(self):
        """
        获取标签文件夹的路径

        该方法将TT100K数据集的根路径与"labels"目录名结合，构造出图像文件夹的完整路径并返回

        :return: 标签文件夹的完整路径
        """
        labels_path = os.path.join(self.tt100k_path, "labels")
        return labels_path

    def create_tt100k_path_structures(self):
        """
        创建TT100K数据集的目录结构。

        此方法旨在为TT100K数据集创建必要的目录结构，包括图片和标签的训练、验证和测试子目录。
        使用os.makedirs函数，并设置exist_ok=True，以确保目录如果已存在不会引发错误。
        """
        try:
            # 尝试创建文件夹
            os.makedirs(self.tt100k_path, exist_ok=True)
            os.makedirs(os.path.join(self.tt100k_path, "images"), exist_ok=True)
            os.makedirs(os.path.join(self.tt100k_path, "labels"), exist_ok=True)
            os.makedirs(os.path.join(self.tt100k_path, "images", "test"), exist_ok=True)
            os.makedirs(os.path.join(self.tt100k_path, "images", "train"), exist_ok=True)
            os.makedirs(os.path.join(self.tt100k_path, "images", "val"), exist_ok=True)
            os.makedirs(os.path.join(self.tt100k_path, "labels", "test"), exist_ok=True)
            os.makedirs(os.path.join(self.tt100k_path, "labels", "train"), exist_ok=True)
            os.makedirs(os.path.join(self.tt100k_path, "labels", "val"), exist_ok=True)
        except Exception as e:
            # 处理创建文件夹过程中可能出现的错误
            print(f"创建文件夹出错: {e}")

    def class_statistics(self):
        # os.makedirs('annotations', exist_ok=True)
        # 存放数据的父路径，（要改成自己的）
        parent_path = FilePathUtils.get_origin_datasets_paths(self.original_datasets)

        # 读TT100K原始数据集标注文件
        with open(os.path.join(parent_path, 'annotations_all.json')) as origin_json:
            origin_dict = json.load(origin_json)
            classes = origin_dict['types']
        # 建立统计每个类别包含的图片的字典
        sta = {}
        for i in classes:
            sta[i] = []

        images_dic = origin_dict['imgs']

        # 记录所有保留的图片
        saved_images = []
        # 遍历TT100K的imgs
        for image_id in images_dic:
            image_element = images_dic[image_id]
            image_path = image_element['path']

            # 添加图像的信息到dataset中
            image_path = image_path.split('/')[-1]
            obj_list = image_element['objects']

            # 遍历每张图片的标注信息
            for anno_dic in obj_list:
                label_key = anno_dic['category']
                # 防止一个图片多次加入一个标签类别
                if image_path not in sta[label_key]:
                    sta[label_key].append(image_path)

        # 只保留包含图片数超过100的类别（重新划分，阈值100可根据需求修改）
        result = {k: v for k, v in sta.items() if len(v) >= 100}

        for i in result:
            print("the type of {} includes {} images".format(i, len(result[i])))
            saved_images.extend(result[i])

        saved_images = list(set(saved_images))
        print("total types is {}".format(len(result)))

        type_list = list(result.keys())
        result = {"type": type_list, "details": result, "images": saved_images}
        print(type_list)
        # 保存结果
        json_name = os.path.join(parent_path, 'statistics.json')
        with open(json_name, 'w', encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)

    def original_datasets2object_datasets_re(self):
        '''
        重新划分数据集
        :return:
        '''
        # os.makedirs('annotations2', exist_ok=True)
        # 存放数据的父路径，（要改成自己的）
        parent_path = FilePathUtils.get_origin_datasets_paths(self.original_datasets)

        # 读TT100K原始数据集标注文件
        with open(os.path.join(parent_path, 'annotations_all.json')) as origin_json:
            origin_dict = json.load(origin_json)

        with open(os.path.join(parent_path, 'statistics.json')) as select_json:
            select_dict = json.load(select_json)
            classes = select_dict['type']

        train_dataset = {'info': {}, 'licenses': [], 'categories': [], 'images': [], 'annotations': []}
        val_dataset = {'info': {}, 'licenses': [], 'categories': [], 'images': [], 'annotations': []}
        test_dataset = {'info': {}, 'licenses': [], 'categories': [], 'images': [], 'annotations': []}
        label = {}  # 记录每个标志类别的id
        count = {}  # 记录每个类别的图片数
        owntype_sum = {}

        info = {
            "year": 2021,  # 年份
            "version": '1.0',  # 版本
            "description": "TT100k_to_coco",  # 数据集描述
            "contributor": "Tecent&Tsinghua",  # 提供者
            "url": 'https://cg.cs.tsinghua.edu.cn/traffic-sign/',  # 下载地址
            "date_created": 2021 - 1 - 15
        }
        licenses = {
            "id": 1,
            "name": "null",
            "url": "null",
        }

        train_dataset['info'] = info
        val_dataset['info'] = info
        test_dataset['info'] = info
        train_dataset['licenses'] = licenses
        val_dataset['licenses'] = licenses
        test_dataset['licenses'] = licenses

        # 建立类别和id的关系
        for i, cls in enumerate(classes):
            train_dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'traffic_sign'})
            val_dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'traffic_sign'})
            test_dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'traffic_sign'})
            label[cls] = i
            count[cls] = 0
            owntype_sum[cls] = 0

        images_dic = origin_dict['imgs']

        obj_id = 1

        # 计算出每个类别共‘包含’的图片数
        for image_id in images_dic:

            image_element = images_dic[image_id]
            image_path = image_element['path']
            image_name = image_path.split('/')[-1]
            # 在所选的类别图片中
            if image_name not in select_dict['images']:
                continue

            # 处理TT100K中的标注信息
            obj_list = image_element['objects']
            # 记录图片中包含最多的实例所属的type
            includes_type = {}
            for anno_dic in obj_list:
                if anno_dic["category"] not in select_dict["type"]:
                    continue
                # print(anno_dic["category"])
                if anno_dic["category"] in includes_type:
                    includes_type[anno_dic["category"]] += 1
                else:
                    includes_type[anno_dic["category"]] = 1
            # print(includes_type)
            own_type = max(includes_type, key=includes_type.get)
            owntype_sum[own_type] += 1

        # TT100K的annotation转换成coco的
        for image_id in images_dic:

            image_element = images_dic[image_id]
            image_path = image_element['path']
            image_name = image_path.split('/')[-1]
            # 在所选的类别图片中
            if image_name not in select_dict['images']:
                continue
            print("dealing with {} image".format(image_path))
            # shutil.copy(os.path.join(parent_path,image_path),os.path.join(parent_path,"dataset/JPEGImages"))

            # 处理TT100K中的标注信息
            obj_list = image_element['objects']
            # 记录图片中包含最多的实例所属的type
            includes_type = {}
            for anno_dic in obj_list:
                if anno_dic["category"] not in select_dict["type"]:
                    continue
                # print(anno_dic["category"])
                if anno_dic["category"] in includes_type:
                    includes_type[anno_dic["category"]] += 1
                else:
                    includes_type[anno_dic["category"]] = 1
            # print(includes_type)
            own_type = max(includes_type, key=includes_type.get)
            count[own_type] += 1
            num_rate = count[own_type] / owntype_sum[own_type]

            # 切换dataset的引用对象，从而划分数据集根据每个类别类别的总数量按7：2：1分为了train_set,val_set,test_set。
            # 其中每个图片所属类别根据该图片包含的类别的数量决定（归属为含有类别最多的类别）
            # target是符合条件的图片路径，也就是实例数大于100的图片（要改成自己的）
            target = self.get_images_dir()
            if num_rate < 0.7:
                dataset = train_dataset
                shutil.copyfile(parent_path + image_path, target + '/train/' + image_path.split("/")[1])
            elif num_rate < 0.9:
                dataset = val_dataset
                shutil.copyfile(parent_path + image_path, target + '/val/' + image_path.split("/")[1])
            else:
                print("dataset=test_dataset")
                dataset = test_dataset
                shutil.copyfile(parent_path + image_path, target + '/test/' + image_path.split("/")[1])

            for anno_dic in obj_list:
                if anno_dic["category"] not in select_dict["type"]:
                    continue
                x = anno_dic['bbox']['xmin']
                y = anno_dic['bbox']['ymin']
                width = anno_dic['bbox']['xmax'] - anno_dic['bbox']['xmin']
                height = anno_dic['bbox']['ymax'] - anno_dic['bbox']['ymin']
                label_key = anno_dic['category']

                dataset['annotations'].append({
                    'area': width * height,
                    'bbox': [x, y, width, height],
                    'category_id': label[label_key],
                    'id': obj_id,
                    'image_id': image_id,
                    'iscrowd': 0,
                    # mask, 矩形是从左上角点按顺时针的四个顶点
                    'segmentation': [[x, y, x + width, y, x + width, y + height, x, y + height]]
                })
                # 每个标注的对象id唯一
                obj_id += 1

            # 用opencv读取图片，得到图像的宽和高
            im = cv2.imread(os.path.join(parent_path, image_path))
            # print(image_path)
            H, W, _ = im.shape
            # 添加图像的信息到dataset中
            dataset['images'].append({'file_name': image_name,
                                      'id': image_id,
                                      'width': W,
                                      'height': H})

        jsonPath = self.get_labels_dir()
        # 保存结果。（下面这个 dataset/annotations/ 依旧要改成自己的实际路径，可以自己设置）
        for phase in ['train', 'val', 'test']:
            json_name = os.path.join(jsonPath, '{}.json'.format(phase))
            with open(json_name, 'w', encoding="utf-8") as f:
                if phase == 'train':
                    json.dump(train_dataset, f, ensure_ascii=False, indent=1)
                if phase == 'val':
                    json.dump(val_dataset, f, ensure_ascii=False, indent=1)
                if phase == 'test':
                    json.dump(test_dataset, f, ensure_ascii=False, indent=1)

    def coco_json2yolo_txt(self, class_json):
        # COCO 格式的数据集转化为 YOLO 格式的数据集
        # --json_path 输入的json文件路径
        # --save_path 保存的文件夹名字，默认为当前目录下的labels。

        '''
        parser = argparse.ArgumentParser()
        # 这里根据自己的json文件位置，换成自己的就行
        parser.add_argument('--json_path',
                            default='D:/yolodate/tt100k_2021/dateset/annotations/train.json',
                            type=str, help="input: coco format(json)")
        # 这里设置.txt文件保存位置
        parser.add_argument('--save_path', default='D:/yolodate/tt100k_2021/dateset/annotations/', type=str,
                            help="specify where to save the output dir of labels")
        arg = parser.parse_args()
        '''

        def convert(size, box):
            dw = 1. / (size[0])
            dh = 1. / (size[1])
            x = box[0] + box[2] / 2.0
            y = box[1] + box[3] / 2.0
            w = box[2]
            h = box[3]
            # round函数确定(xmin, ymin, xmax, ymax)的小数位数
            x = round(x * dw, 6)
            w = round(w * dw, 6)
            y = round(y * dh, 6)
            h = round(h * dh, 6)
            return (x, y, w, h)

        # class_json = 'train'
        json_file = os.path.join(self.get_labels_dir(), '%s.json' % class_json)  # COCO Object Instance 类型的标注
        # ana_txt_save_path = 'D:/jinxData/TT100K/data/dataset/annotations/train'  # 保存的路径
        ana_txt_save_path = os.path.join(self.get_labels_dir(), class_json)  # 保存的路径

        data = json.load(open(json_file, 'r'))
        if not os.path.exists(ana_txt_save_path):
            os.makedirs(ana_txt_save_path)

        id_map = {}  # coco数据集的id不连续！重新映射一下再输出！
        with open(os.path.join(ana_txt_save_path, 'classes.txt'), 'w') as f:
            # 写入classes.txt
            for i, category in enumerate(data['categories']):
                f.write(f"{category['name']}\n")
                id_map[category['id']] = i
        # print(id_map)
        list_file = open(os.path.join(ana_txt_save_path, '%s.txt' % class_json.format()), 'w')
        for img in tqdm(data['images']):
            filename = img["file_name"]
            img_width = img["width"]
            img_height = img["height"]
            img_id = img["id"]
            head, tail = os.path.splitext(filename)
            ana_txt_name = head + ".txt"  # 对应的txt名字，与jpg一致
            f_txt = open(os.path.join(ana_txt_save_path, ana_txt_name), 'w')
            for ann in data['annotations']:
                if ann['image_id'] == img_id:
                    box = convert((img_width, img_height), ann["bbox"])
                    f_txt.write("%s %s %s %s %s\n" % (id_map[ann["category_id"]], box[0], box[1], box[2], box[3]))
            f_txt.close()
            # 这里的相对路径，是你yolo模型里数据集的路径，请根据自己的实际情况修改
            list_file.write(
                '/%s/%s/%s.jpg\n' % (self.get_images_dir(), class_json.format(), head))
        list_file.close()


if __name__ == '__main__':
    tt100k = TT100K2COCO()
    tt100k.create_tt100k_path_structures()
    tt100k.class_statistics()
    tt100k.original_datasets2object_datasets_re()
    tt100k.coco_json2yolo_txt('val')
    tt100k.coco_json2yolo_txt('train')
    tt100k.coco_json2yolo_txt('test')
