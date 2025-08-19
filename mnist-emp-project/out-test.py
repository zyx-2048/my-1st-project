import torch
import torch.nn as nn
from PIL import Image, ImageOps
import os
import argparse
import time
import matplotlib.pyplot as plt
from torchvision import transforms
from model import MNIST_EMP_Model
import numpy as np
import shutil
import cv2

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="使用训练好的模型预测图片分类")
    parser.add_argument("--input_dir", type=str, default="D:\\WAIT-before-to-git\\my-1st-project\\mnist-emp-project\\input_images",
                        help="包含待预测图片的目录路径")
    parser.add_argument("--model_path", type=str, default="D:\\WAIT-before-to-git\\my-1st-project\\mnist-emp-project\\mnist_emp_model.pth",
                        help="训练好的模型路径")
    parser.add_argument("--cache_dir", type=str, default="D:\\WAIT-before-to-git\\my-1st-project\\mnist-emp-project\\preprocessed_images",
                        help="预处理后图片的缓存目录")
    parser.add_argument("--output_file", type=str, default="D:\\WAIT-before-to-git\\my-1st-project\\mnist-emp-project\\prediction_results.txt",
                        help="预测结果输出文件路径")
    parser.add_argument("--show_images", action="store_true",
                        help="是否显示预测图片")
    parser.add_argument("--invert_colors", action="store_true",
                        help="是否反转图像颜色（不反转适用于白底黑字图像）")
    return parser.parse_args()

def preprocess_image(image_path, output_dir, size=(28, 28), invert=False):
    """
    预处理图片：转换为灰度、调整大小、二值化，并确保黑底白字
    返回预处理后的图片和保存路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取文件名
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"preprocessed_{filename}")
    
    # 打开原始图片并转换为灰度
    img = Image.open(image_path).convert("L")
    
    # 如果指定了反转颜色，则进行反转
    if invert:
        img = ImageOps.invert(img)
    
    # 调整大小
    img = img.resize(size, Image.LANCZOS)
    
    # 转换为numpy数组进行处理
    img_np = np.array(img)
    
    # 自适应阈值二值化（使用OpenCV以获得更好的效果）
    # 首先确保图像是uint8类型
    if img_np.dtype != np.uint8:
        img_np = img_np.astype(np.uint8)
    
    # 使用自适应阈值处理
    thresh = cv2.adaptiveThreshold(
        img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # 形态学操作去除噪声
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # 确保背景为黑色，前景为白色
    # 计算黑色像素比例
    black_pixels = np.sum(thresh == 0)
    white_pixels = np.sum(thresh == 255)
    
    # 如果白色像素多于黑色像素，则反转
    if white_pixels > black_pixels:
        thresh = cv2.bitwise_not(thresh)
    
    # 转换为PIL图像
    preprocessed_img = Image.fromarray(thresh)
    
    # 保存预处理后的图片
    preprocessed_img.save(output_path)
    
    return preprocessed_img, output_path

def predict_image(model, image, device):
    """使用模型预测单张图片的分类"""
    # 确保图片是PIL Image对象
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # 应用与训练相同的转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 预处理图片并添加批次维度
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        output = model(input_tensor)
        
        # 检查输出是否有效
        if output.size(0) == 0 or output.size(1) == 0:
            raise RuntimeError(f"模型输出无效: {output.size()}")
        
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        _, predicted = torch.max(output, 1)
    
    # 获取预测结果和置信度
    class_idx = predicted.item()
    
    # 确保索引在有效范围内
    if class_idx < 0 or class_idx >= probabilities.size(0):
        raise RuntimeError(f"预测索引 {class_idx} 超出范围 [0, {probabilities.size(0)}-1]")
    
    confidence = probabilities[class_idx].item()
    
    # 类别映射
    class_names = {
        0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
        5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
        10: "EMP"
    }
    
    # 确保类别存在
    if class_idx not in class_names:
        raise RuntimeError(f"未知类别索引: {class_idx}")
    
    return class_names[class_idx], confidence, probabilities.cpu().numpy()

def display_results(image_path, prediction, confidence, class_probabilities, preprocessed_img=None):
    """显示预测结果"""
    plt.figure(figsize=(12, 6))
    
    # 显示原始图片
    plt.subplot(1, 3, 1)
    orig_img = Image.open(image_path)
    plt.imshow(orig_img)
    plt.title(f"Original Image\n{os.path.basename(image_path)}")
    plt.axis("off")
    
    # 显示预处理后的图片
    if preprocessed_img:
        plt.subplot(1, 3, 2)
        plt.imshow(preprocessed_img, cmap="gray")
        plt.title("Preprocessed Image")
        plt.axis("off")
    
    # 显示概率分布
    plt.subplot(1, 3, 3)
    classes = list(range(len(class_probabilities)))
    class_labels = [str(i) for i in range(len(class_probabilities)-1)] + ["EMP"]
    colors = plt.cm.viridis(np.linspace(0, 1, len(class_probabilities)))
    
    bars = plt.bar(classes, class_probabilities, color=colors)
    plt.xticks(classes, class_labels)
    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.ylim(0, 1.1)
    plt.title(f"Prediction: {prediction} ({confidence:.2%})")
    
    # 添加概率值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f"{height:.2f}", ha="center", va="bottom")
    
    plt.tight_layout()
    plt.show()

def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    # 打印配置信息
    print("="*60)
    print("图片分类预测配置:")
    print(f"输入目录: {args.input_dir}")
    print(f"模型路径: {args.model_path}")
    print(f"缓存目录: {args.cache_dir}")
    print(f"结果文件: {args.output_file}")
    print(f"显示图片: {'是' if args.show_images else '否'}")
    print(f"反转颜色: {'是' if args.invert_colors else '否'}")
    print("="*60)
    
    # 检查输入目录是否存在
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录不存在 - {args.input_dir}")
        return
    
    # 获取图片文件列表
    image_files = []
    valid_extensions = (".png", ".jpg", ".jpeg")
    for f in os.listdir(args.input_dir):
        full_path = os.path.join(args.input_dir, f)
        if os.path.isfile(full_path) and f.lower().endswith(valid_extensions):
            image_files.append(full_path)
    
    if not image_files:
        print(f"错误: 在 {args.input_dir} 中没有找到图片文件 (支持.png, .jpg, .jpeg)")
        return
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    model = MNIST_EMP_Model().to(device)
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在 - {args.model_path}")
        return
    
    # 加载模型权重
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print("模型加载成功")
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        return
    
    # 清空缓存目录
    if os.path.exists(args.cache_dir):
        shutil.rmtree(args.cache_dir)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # 创建结果文件
    with open(args.output_file, "w", encoding="utf-8") as result_file:
        result_file.write("图片预测结果\n")
        result_file.write("="*60 + "\n")
        result_file.write(f"模型: {args.model_path}\n")
        result_file.write(f"输入目录: {args.input_dir}\n")
        result_file.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        result_file.write(f"反转颜色: {'是' if args.invert_colors else '否'}\n")
        result_file.write("="*60 + "\n\n")
        
        result_file.write("{:<30} {:<10} {:<15} {}\n".format(
            "图片名称", "预测结果", "置信度", "预处理图片路径"))
        result_file.write("-"*80 + "\n")
    
    # 处理每张图片
    print("\n开始预测...")
    start_time = time.time()
    processed_count = 0
    
    for i, img_path in enumerate(image_files):
        try:
            print(f"处理图片 {i+1}/{len(image_files)}: {os.path.basename(img_path)}")
            
            # 预处理图片
            preprocessed_img, cache_path = preprocess_image(
                img_path, args.cache_dir, invert=args.invert_colors)
            
            # 预测分类
            prediction, confidence, probabilities = predict_image(
                model, preprocessed_img, device)
            
            # 打印结果
            filename = os.path.basename(img_path)
            print(f"  -> 预测结果: {prediction} (置信度: {confidence:.2%})")
            
            # 写入结果文件
            with open(args.output_file, "a", encoding="utf-8") as result_file:
                result_file.write("{:<30} {:<10} {:<15.2%} {}\n".format(
                    filename, prediction, confidence, cache_path))
            
            # 显示结果
            if args.show_images:
                display_results(img_path, prediction, confidence, 
                               probabilities, preprocessed_img)
            
            processed_count += 1
        
        except Exception as e:
            print(f"处理图片 {img_path} 时出错: {str(e)}")
            with open(args.output_file, "a", encoding="utf-8") as result_file:
                result_file.write(f"{os.path.basename(img_path)} 处理失败: {str(e)}\n")
    
    total_time = time.time() - start_time
    print(f"\n预测完成! 成功处理 {processed_count}/{len(image_files)} 张图片, 耗时: {total_time:.2f}秒")
    print(f"预测结果已保存到: {args.output_file}")
    print(f"预处理图片已保存到: {args.cache_dir}")
    print(f"提示: 如果预测结果不准确，请尝试使用 --invert_colors 参数反转图像颜色")

if __name__ == "__main__":
    main()
