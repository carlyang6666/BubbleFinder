import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from math import sqrt
from skimage.feature import blob_log
from sklearn.cluster import DBSCAN

# 定义 BubbleFinder 函数
def BubbleFinder(pred2D, min_sigma, max_sigma, threshold, overlap, param1, param2, minRadius, maxRadius, eps, min_samples):
    """
    检测气泡的函数，结合 LoG、Hough Circle 和 DBSCAN，同时计算气泡数量和总面积。
    """
    from math import pi

    # LoG 检测
    pred2D_i = pred2D * 255
    blobs_log = blob_log(pred2D_i, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold * 255, overlap=overlap)
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    # Hough Circle Transform
    cimg = cv2.GaussianBlur(pred2D, (5, 5), 0, 0)
    for i in range(cimg.shape[0]):
        for j in range(cimg.shape[1]):
            if cimg[i, j] < threshold:
                cimg[i, j] = 0
    cimg = np.uint8(cimg * 255)
    circles = cv2.HoughCircles(cimg, cv2.HOUGH_GRADIENT, 1, 1, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    circles = circles[0] if circles is not None else np.empty((0, 3))

    # 将 LoG 和 Hough Circle 的结果合并
    blobs_log_xyr = np.transpose(np.array([blobs_log[:, 1], blobs_log[:, 0], blobs_log[:, 2]]))
    coords_r = np.vstack((blobs_log_xyr, circles))
    coords = coords_r[:, 0:2]

    # 使用 DBSCAN 进行聚类
    db = DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree').fit(coords)
    labels = db.labels_

    # 将聚类标签附加到结果中
    coords_r = np.column_stack((coords_r, labels))

    # 聚合结果
    coords_s = np.empty((0, 4))
    for i in range(0, labels.max() + 1):
        cluster = coords_r[coords_r[:, 3] == i]
        mean_cluster = np.mean(cluster, axis=0, dtype=np.float64)
        coords_s = np.append(coords_s, [mean_cluster], axis=0)

    # 添加离群点
    outliers = coords_r[coords_r[:, 3] == -1]
    coords_a = np.vstack((coords_s, outliers))

    # **计算气泡信息**
    num_bubbles = coords_a.shape[0]  # 气泡数量
    bubble_sizes = np.array([pi * (r ** 2) for _, _, r, _ in coords_a])  # 每个气泡面积
    total_area = np.sum(bubble_sizes)  # 气泡总面积

    return coords_a, num_bubbles, bubble_sizes, total_area


# 加载模型
model = 'best.model.keras'
prediction_model = tf.keras.models.load_model(model)

# 图片文件夹路径
annotation_folder = 'data/cropped/annotations'
image_folder = 'data/cropped/image'
output_folder = 'output'

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历两个文件夹，检查图片名是否匹配
annotation_files = sorted(os.listdir(annotation_folder))
image_files = sorted(os.listdir(image_folder))

for annotation_file, image_file in zip(annotation_files, image_files):
    if annotation_file.split('.')[0] == image_file.split('.')[0]:  # 检查文件名是否匹配
        print(f"Processing pair: {annotation_file} and {image_file}")

        # 加载训练图片和测试图片
        annotation_path = os.path.join(annotation_folder, annotation_file)
        image_path = os.path.join(image_folder, image_file)
        annotation_img = Image.open(annotation_path).convert("RGB")
        test_images = np.expand_dims(np.array(annotation_img), axis=0)
        train_img = Image.open(image_path).convert("RGB")
        train_images = np.expand_dims(np.array(train_img), axis=0)

        # 使用训练图像进行预测
        pred_img = train_images[0]
        pred_img = tf.convert_to_tensor(pred_img, dtype=tf.float32)
        pred_img = (pred_img - tf.reduce_min(pred_img)) / (tf.reduce_max(pred_img) - tf.reduce_min(pred_img))
        pred_img = tf.expand_dims(pred_img, axis=0)

        # 预测
        prediction = prediction_model.predict(pred_img)
        pred_results2D = prediction[0, :, :, 0]

        # 将预测结果叠加到原始图像
        pred_img2D = train_images[0].astype(np.float32) / 255.0
        superim = pred_img2D.copy()
        threshold = 0.25
        superim[pred_results2D > threshold] = 1.0

        # 使用 BubbleFinder 获取气泡信息
        coords, num_bubbles, bubble_sizes, total_area = BubbleFinder(
            pred_results2D, min_sigma=1, max_sigma=9, threshold=0.25, overlap=0.9,
            param1=0.1, param2=15, minRadius=1, maxRadius=17, eps=5, min_samples=2)

        # **计算面积直方图数据**
        area = coords[:, 2] ** 2 * np.pi  # 每个气泡的面积（像素平方）
        no_bubbles = len(coords[:, 2])    # 气泡总数
        total_area = np.sum(area)         # 气泡总面积

        # 绘图
        fig, axes = plt.subplots(1, 3, figsize=(25, 20))  # 增大图像尺寸
        ax = axes.ravel()

        # 原始图片
        ax[0].imshow(pred_img2D, cmap='gist_gray')
        ax[0].title.set_text('Original Image')

        # 叠加气泡标注的预测结果
        ax[1].imshow(superim, cmap='gist_gray')
        for blob in coords:
            x, y, r, i = blob
            c = plt.Circle((x, y), r, color='lime', linewidth=1, fill=False)
            ax[1].add_patch(c)
        ax[1].title.set_text('Superimposed Prediction')

        # 带圆圈标注的预测结果
        ax[2].imshow(pred_img2D, cmap='gist_gray')
        for blob in coords:
            x, y, r, i = blob
            c = plt.Circle((x, y), r, color='lime', linewidth=1, fill=False)
            ax[2].add_patch(c)
        ax[2].title.set_text('Labeled Image')

        # **添加气泡统计信息到第一个子图**
        ax[2].text(0.5, -0.15, f"Total bubbles: {no_bubbles}", color='red', fontsize=15, ha='center', transform=ax[0].transAxes)
        ax[2].text(0.5, -0.25, f"Total area: {total_area:.2f} px^2", color='blue', fontsize=15, ha='center', transform=ax[0].transAxes)

        # 手动调整边距
        plt.subplots_adjust(top=0.9, bottom=0.3, left=0.1, right=0.9, hspace=0.3, wspace=0.3)

        # 保存图像到输出文件夹
        output_path = os.path.join(output_folder, f"{annotation_file.split('.')[0]}_result.png")
        plt.savefig(output_path)
        plt.close(fig)

