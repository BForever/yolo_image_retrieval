import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import time
import cv2
import os
import subprocess


def maxpooling1d(lists):
    if len(lists) < 1:
        return lists
    index = 1
    result = lists[0]
    while index < len(lists):
        result = np.maximum(result, lists[index])
        index = index + 1
    return result


def maxpooling2d(lists):
    if len(lists) < 1:
        return lists
    index = 1
    result = maxpooling1d(lists[0])
    while index < len(lists):
        temp = maxpooling1d(lists[index])
        result = np.maximum(result, temp)
        index = index + 1
    return result


if __name__ == '__main__':
    from argparse import ArgumentParser

    # 参数处理
    parser = ArgumentParser()
    parser.add_argument('--data', dest='data_dir', type=str,
                        default='imgs',
                        help='data base image directory')
    parser.add_argument('--query', dest='query_file', type=str,
                        default='imgs/mouse1.jpg',
                        help='query file path')

    parser.add_argument('--features_t', dest='feature_t_file', type=str,
                        default='train.npz',
                        help='yolo 1024 train features file path')
    parser.add_argument('--objects_t', dest='object_t_file', type=str,
                        default='object_t_list.csv',
                        help='train object list file path')
    parser.add_argument('--features_e', dest='feature_e_file', type=str,
                        default='eval.npz',
                        help='yolo 1024 eval features file path')
    parser.add_argument('--objects_e', dest='object_e_file', type=str,
                        default='object_e_list.csv',
                        help='eval object list file path')
    parser.add_argument('--num', dest='obj_no', type=int,
                        default=0,
                        help='eval object no. in list')
    parser.add_argument('--pca', dest='pca', type=float,
                        default=0.9,
                        help='pca ratio')
    parser.add_argument('--dis', dest='dis_type', type=str,
                        default='cos',
                        help='distance type: l2 or cos')
    parser.add_argument('--pooling', dest='pooling', type=str,
                        default='max',
                        help='support 3 pooling types: single, average, max')
    args = parser.parse_args()


    # 距离计算函数：支持欧式距离L2和余弦距离
    def distance(vector1, vector2):
        dist = None
        if args.dis_type == 'l2':
            dist = np.sqrt(np.sum(np.square(vector1 - vector2)))
        elif args.dis_type == 'cos':
            dist = np.dot(vector1, vector2) / (
                    np.linalg.norm(vector1) * (np.linalg.norm(vector2)))
        else:
            print("unsupported distance type, will use l2 distance")
            dist = np.sqrt(np.sum(np.square(vector1 - vector2)))
        return dist


    # 从文件中获取特征图
    features_t = np.load(args.feature_t_file)
    features_t = features_t['arr_0']
    print("train features shape: ", features_t.shape)
    print("max:{} min:{} avg:{}".format(features_t.max(), features_t.min(),
                                        features_t.mean()))
    features_e = np.load(args.feature_e_file)
    features_e = features_e['arr_0']
    print("eval features shape: ", features_e.shape)

    # 从文件中获取物体列表
    objects_t = pd.read_csv(args.object_t_file)
    print("objects_t: ")
    print(objects_t)
    print("")

    objects_e = pd.read_csv(args.object_e_file)
    print("objects_e: ")
    print(objects_e)
    print("")


    def display(results):
        for j in range(3):
            res = results[j]
            index = res[2]
            filename = objects_t.iloc[index, 10]
            img = cv2.imread(filename)
            lx = objects_t.iloc[index, 3]
            ly = objects_t.iloc[index, 4]
            rx = objects_t.iloc[index, 5]
            ry = objects_t.iloc[index, 6]
            obj = img[ly:ry, lx:rx]
            # cv2.imshow("r{}:{}".format(j, filename), img)
            cv2.imshow("{}:{}".format(j,filename), obj)
            cv2.waitKey(0)



    # 待查询物体信息
    object_no = args.obj_no
    ex = objects_e.iloc[object_no, 1]
    ey = objects_e.iloc[object_no, 2]
    blocks = objects_e.iloc[object_no, 7]
    w = objects_e.iloc[object_no, 11]
    h = objects_e.iloc[object_no, 12]
    elx = int(objects_e.iloc[object_no, 3] / (w / blocks))
    ely = int(objects_e.iloc[object_no, 4] / (h / blocks))
    erx = int(objects_e.iloc[object_no, 5] / (w / blocks))
    ery = int(objects_e.iloc[object_no, 6] / (h / blocks))

    # 待查询物体特征预处理：将二维的特征向量数组处理为单个特征向量
    if args.pooling == 'average':
        features_e = features_e[0, :, elx:erx + 1, ely:ery + 1].mean(1).mean(1)
    elif args.pooling == 'max':
        features_e = maxpooling2d(
            features_e[0, :, elx:erx + 1, ely:ery + 1].transpose(1, 2, 0))
    elif args.pooling == 'single':
        features_e = features_e[0, :, ex, ey]
    else:
        print("unsupported pooling type: {}".format(args.pooling))
        exit(1)
    print(features_e.shape)

    # 直接计算待处理物体与数据库中所有物体的距离
    min_dist = 10000000
    extracted_features = []
    results = []
    start = time.time()
    for i in range(objects_t.shape[0]):
        image_index = objects_t.iloc[i, 0]
        x = objects_t.iloc[i, 1]
        y = objects_t.iloc[i, 2]
        blocks = objects_t.iloc[i, 7]
        w = objects_t.iloc[i, 11]
        h = objects_t.iloc[i, 12]
        lx = int(objects_t.iloc[i, 3] / (w / blocks))
        ly = int(objects_t.iloc[i, 4] / (h / blocks))
        rx = int(objects_t.iloc[i, 5] / (w / blocks))
        ry = int(objects_t.iloc[i, 6] / (h / blocks))
        cls = objects_t.iloc[i, 8]
        label = objects_t.iloc[i, 9]
        image_filename = objects_t.iloc[i, 10].split('/')[-1]

        features = None
        if args.pooling == 'average':
            features = features_t[image_index, :, lx:rx + 1, ly:ry + 1].mean(
                1).mean(1)
        elif args.pooling == 'max':
            features = maxpooling2d(
                features_t[image_index, :, lx:rx + 1, ly:ry + 1].transpose(1, 2,
                                                                           0))
        elif args.pooling == 'single':
            features = features_t[image_index, :, x, y]
        else:
            print("unsupported pooling type: {}".format(args.pooling))
            exit(1)

        extracted_features.append(features)
        dist = distance(features, features_e)
        results.append([float(dist), label, i, image_filename])

    results.sort(
        key=lambda item: float(item[0]) * (-1 if args.dis_type == 'cos' else 1))
    print("results: ")
    print(np.array(results)[:15])
    print("elapsed: {} s".format(time.time() - start))
    # display(results)

    # 将数据库物体特征进行主成分分析降维后再一一比较距离
    pca = PCA(args.pca)
    print(pca.fit(extracted_features))
    features_e_pca = pca.transform([features_e])[0]
    features_t_pca = pca.transform(extracted_features)
    print(pca.explained_variance_ratio_)
    print("ratio: ", sum(pca.explained_variance_ratio_))

    start = time.time()
    results = []
    for i in range(objects_t.shape[0]):
        image_index = objects_t.iloc[i, 0]
        x = objects_t.iloc[i, 1]
        y = objects_t.iloc[i, 2]
        blocks = objects_t.iloc[i, 7]
        w = objects_t.iloc[i, 11]
        h = objects_t.iloc[i, 12]
        lx = int(objects_t.iloc[i, 3] / (w / blocks))
        ly = int(objects_t.iloc[i, 4] / (h / blocks))
        rx = int(objects_t.iloc[i, 5] / (w / blocks))
        ry = int(objects_t.iloc[i, 6] / (h / blocks))
        cls = objects_t.iloc[i, 8]
        label = objects_t.iloc[i, 9]
        image_filename = objects_t.iloc[i, 10].split('/')[-1]

        features = features_t_pca[i]
        dist = distance(features, features_e_pca)
        results.append([float(dist), label, i, image_filename])

    results.sort(
        key=lambda item: float(item[0]) * (-1 if args.dis_type == 'cos' else 1))
    print("results: ")
    print(np.array(results)[:15])
    print("elapsed: {} s".format(time.time() - start))
    # display(results)
