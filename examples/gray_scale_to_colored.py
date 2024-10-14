from appbots.core.images.builder import get_image_from_path
from appbots.core.images.transforms import gray_transform
from appbots.core.nn_modules.conv_sobel import SobelConv2d
from appbots.core.plots.plot import plot_images
from appbots.core.utils import get_path
import torch


def grayscale_to_colored(img: torch.Tensor, n=3):
    # 读取灰度图像并归一化
    img = img / 255.0  # 添加batch

    start = torch.min(img)
    end = torch.max(img)

    img = img.unsqueeze(0)

    # 颜色等分
    colors = torch.linspace(start, end, n)
    colors = colors.view(n, 1, 1)  # 扩展维度

    # 计算每个像素点与各颜色的距离，并选择最近的
    distances = torch.abs(img - colors)
    _, indices = torch.min(distances, dim=1)

    # 将索引转换为颜色
    _colored_img = colors[indices].squeeze(0)
    _colored_img = _colored_img.permute(2, 3, 0, 1)

    # 将彩色图像转换为BGR格式
    # _colored_img = cv2.cvtColor(_colored_img, cv2.COLOR_GRAY2BGR)

    return _colored_img * 255


def connected_components(binary_image, connectivity=8):
    """
    使用两遍扫描法实现连通域标记

    Args:
        binary_image: 二值图像，形状为(H, W)
        connectivity: 连通性，4或8

    Returns:
        labels: 连通域标签图，形状为(H, W)
    """

    height, width = binary_image.shape
    labels = torch.zeros(height, width, dtype=torch.int64)  # 初始化标签图

    # 第一遍扫描
    label = 1
    for i in range(height):
        for j in range(width):
            if binary_image[i, j] == 1:
                neighbors = []
                if i > 0:
                    neighbors.append(labels[i-1, j])
                if j > 0:
                    neighbors.append(labels[i, j-1])
                if connectivity == 8 and i > 0 and j > 0:
                    neighbors.append(labels[i-1, j-1])

                # 获取最小非零标签
                min_label = min(neighbors)
                if min_label > 0:
                    labels[i, j] = min_label
                else:
                    labels[i, j] = label
                    label += 1

    # 第二遍扫描，合并等价标签
    equivalences = {}
    for i in range(height):
        for j in range(width):
            if binary_image[i, j] == 1:
                current_label = labels[i, j]
                if current_label in equivalences:
                    current_label = equivalences[current_label]

                # 更新邻居的标签
                neighbors = []
                if i > 0:
                    neighbors.append(labels[i-1, j])
                if j > 0:
                    neighbors.append(labels[i, j-1])
                if connectivity == 8 and i > 0 and j > 0:
                    neighbors.append(labels[i-1, j-1])

                for neighbor in neighbors:
                    if neighbor > 0 and neighbor != current_label:
                        equivalences[neighbor] = min(current_label, neighbor.item())

    # 重新标记
    for i in range(height):
        for j in range(width):
            if binary_image[i, j] == 1:
                current_label = labels[i, j]
                while current_label in equivalences:
                    current_label = equivalences[current_label]
                labels[i, j] = current_label

    return labels


def get_bounding_boxes(labels: torch.Tensor):
    # 输入：连通域标签图
    # 输出：边界框列表，每个元素为[左上角x, 左上角y, 宽, 高]

    _boxes = []
    for label in torch.unique(labels)[1:]:  # 跳过背景
        mask = labels == label
        rows, cols = torch.where(mask)
        min_row, max_row = torch.min(rows), torch.max(rows)
        min_col, max_col = torch.min(cols), torch.max(cols)
        _boxes.append([min_col, min_row, max_col - min_col + 1, max_row - min_row + 1])
    return _boxes


if __name__ == '__main__':
    # 示例用法
    img_tensor, _ = get_image_from_path(get_path("assets/test.png"), size=(600, 300))
    gray_tensor = gray_transform(img_tensor)

    print("gray_tensor:", gray_tensor.shape)

    # colored_img = grayscale_to_colored(gray_tensor)

    sobel_model = SobelConv2d()
    sobel_edges: torch.Tensor = sobel_model(gray_tensor.unsqueeze(0) / 255.0)

    sobel_edges = sobel_edges.unsqueeze(0)

    print("sobel_edges:", sobel_edges.shape)

    # b_img = sobel_edges.squeeze(0).to(dtype=torch.int32) / 255
    # boxes = get_bounding_boxes(connected_components(b_img))
    # print("boxes:", len(boxes))
    # plot_boxes(gray_tensor, boxes=boxes)

    plot_images([img_tensor, gray_tensor, sobel_edges])
