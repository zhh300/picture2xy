import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.spatial import distance as scidist


def enhance_details(img):
    """图像细节增强函数：通过CLAHE、双边滤波和多尺度细节融合增强图像局部细节"""
    # === 自适应直方图均衡化（增强局部对比度） ===
    # 创建CLAHE对象，限制对比度2.0，分块大小8x8
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img) # 应用CLAHE到输入图像

    # === 边缘保持滤波（降噪同时保留边缘） ===
    # 双边滤波参数：d=9（邻域直径）, sigmaColor=75（颜色空间标准差）, sigmaSpace=75（坐标空间标准差）
    # 转换结果为浮点型以便后续浮点运算
    blurred = cv2.bilateralFilter(enhanced, 9, 75, 75).astype(np.float32)  # 转换为浮点型

    # === 多尺度细节增强 ===
    kernel_sizes = [3, 5, 7]# 定义不同尺度的高斯核尺寸
    detail_layers = []
    for k in kernel_sizes:
        # 高斯模糊层：使用当前核尺寸，标准差自动计算（sigmaX=0）
        blurred_layer = cv2.GaussianBlur(blurred, (k, k), 0)
        # 计算细节层：原图 - 模糊层（显式指定输出为CV_32F类型）
        detail_layer = cv2.subtract(blurred, blurred_layer, dtype=cv2.CV_32F)
        detail_layers.append(detail_layer)

    # === 细节融合与增强 ===
    # 合并策略：原图与平均细节层各占50%权重
    # 公式：combined = blurred*0.5 + avg_detail*0.5 + 0（偏移量）
    combined = cv2.addWeighted(blurred, 0.7,
                               sum(detail_layers) / len(detail_layers), 0.3, 0,
                               dtype=cv2.CV_32F)

    # === 后处理 ===
    # 截断超界值：将像素值限制在0-255范围内
    # 转换回uint8类型（OpenCV标准图像格式）
    return np.clip(combined, 0, 255).astype(np.uint8)


def sort_contours(contours):
    """基于旅行商问题的近似路径优化
    功能：对轮廓进行排序，最小化绘图设备的移动距离
    输入：OpenCV检测到的轮廓列表（每个轮廓为Nx1x2的numpy数组）
    输出：优化排序后的轮廓列表（包含可能的翻转操作）
    """
    if not contours:
        return []

    # === 步骤1：提取所有轮廓的端点 ===
    endpoints = []
    for c in contours:
        if len(c) > 0:
            # 每个轮廓存储起点和终点（格式：(x,y)）
            endpoints.append(c[0][0])# 起点：轮廓的第一个点
            endpoints.append(c[-1][0])# 终点：轮廓的最后一个点

    # === 步骤2：构建TSP问题模型 ===
    # 将端点转换为numpy数组（形状：2N x 2）
    points = np.array(endpoints)
    # 计算欧氏距离矩阵（形状：2N x 2N）
    dist_matrix = scidist.cdist(points, points, 'euclidean')

    # === 步骤3：最近邻贪心算法求解TSP ===
    sorted_indices = [0]
    unvisited = set(range(1, len(points)))
    # 逐步选择最近邻节点
    while unvisited:
        last = sorted_indices[-1]
        # 找出最近的未访问节点（lambda函数实现距离比较）
        next_idx = min(unvisited, key=lambda x: dist_matrix[last][x])
        sorted_indices.append(next_idx)
        unvisited.remove(next_idx)

    # === 步骤4：重构轮廓顺序 ===
    sorted_contours = []
    used = set() #  记录已处理的原始轮廓索引
    for idx in sorted_indices:
        # 计算对应的原始轮廓索引（每个轮廓贡献2个端点）
        contour_idx = idx // 2
        if contour_idx not in used:
            # 获取原始轮廓（注意OpenCV轮廓格式）
            contour = contours[contour_idx]
            # 判断是否需要翻转轮廓方向
            # 当idx为奇数时，说明当前端点对应轮廓的终点
            if idx % 2 == 1:
                # 翻转轮廓点顺序（使终点变为起点）
                contour = np.flip(contour, axis=0)

            sorted_contours.append(contour)
            used.add(contour_idx)

    return sorted_contours


def image_to_trajectory(image_path, canny_threshold=50, epsilon=0.005, min_contour_length=10):
    """主转换函数"""
    # 读取并预处理图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("无法读取图像，请检查文件路径")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = enhance_details(gray)

    # 边缘检测与细化
    edges = cv2.Canny(enhanced, canny_threshold, canny_threshold * 3)
    skeleton = skeletonize(edges.astype(bool)).astype(np.uint8) * 255  # 修改骨架化方式

    # 提取轮廓
    contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 过滤短轮廓
    contours = [c for c in contours if cv2.arcLength(c, False) > min_contour_length]

    # 路径优化排序
    sorted_contours = sort_contours(contours)

    # 生成轨迹数据
    trajectory = []
    last_point = np.array([0, 0])  # 初始位置

    for contour in sorted_contours:
        if len(contour) < 2:
            continue

        # 轮廓简化
        epsilon_val = epsilon * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon_val, True).squeeze()

        if approx.ndim < 2 or len(approx) < 2:
            continue

        points = [tuple(p) for p in approx] if approx.ndim == 2 else [tuple(approx)]

        # 添加抬笔移动
        current_start = np.array(points[0])
        if np.linalg.norm(last_point - current_start) > 5:
            trajectory.append((tuple(last_point), points[0], 'red'))

        # 添加落笔路径
        for i in range(len(points) - 1):
            trajectory.append((points[i], points[i + 1], 'blue'))

        last_point = np.array(points[-1])

    return trajectory


def plot_trajectory(trajectory, line_width=1):
    """可视化函数"""
    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    # 绘制所有线段
    for start, end, color in trajectory:
        x = [start[0], end[0]]
        y = [start[1], end[1]]
        ax.plot(x, y, color=color, linewidth=line_width)

    # 设置坐标系属性
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 示例用法
    try:
        # 参数设置
        IMAGE_PATH = 'lnt.png'  # 输入图像路径
        CANNY_THRESH = 60  # 边缘检测阈值(30-100)
        EPSILON = 0.001  # 轮廓简化程度(0.001-0.01)
        LINE_WIDTH = 0.3  # 可视化线宽

        # 执行转换
        trajectory = image_to_trajectory(IMAGE_PATH,
                                         canny_threshold=CANNY_THRESH,
                                         epsilon=EPSILON)

        # 可视化结果
        plot_trajectory(trajectory, line_width=LINE_WIDTH)

    except Exception as e:
        print(f"错误发生: {str(e)}")
