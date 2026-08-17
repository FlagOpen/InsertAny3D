import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ["http_proxy"] ="http://127.0.0.1:9084"
os.environ["https_proxy"]="http://127.0.0.1:9084"
import json
import torch
import pyiqa
import hpsv2
import argparse
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from torchvision import transforms

niqe_metric = pyiqa.create_metric('niqe').cuda()
qalign = pyiqa.create_metric('qalign').cuda()

def convert_rgba_to_rgb_with_black_bg(img):
    """将RGBA图像转换为RGB，以黑色为底色"""
    background = Image.new('RGB', img.size, (0, 0, 0))
    background.paste(img, mask=img.split()[-1])  # 使用alpha通道作为掩码
    return background

def calculate_niqe_for_image(image_path, prompt=""):
    """计算单张图片的NIQE分数"""
    try:
        # 加载图像
        img = Image.open(image_path)
        
        # 处理不同图像模式
        if img.mode == 'RGBA':
            img = convert_rgba_to_rgb_with_black_bg(img)
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 转换为张量
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        img_tensor = transform(img).unsqueeze(0).float()  # 形状: (1, 3, H, W)
        
        # 可选：使用GPU加速
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
        
        # 计算NIQE分数（使用pytorch_msssim的实现）
        niqe_score = niqe_metric(img_tensor)
        qalign_score_a = qalign(img_tensor,task_="aesthetic")
        qalign_score_q = qalign(img_tensor,task_="quality")
        if prompt != "":
            hpsv2_score = hpsv2.score(img, prompt, hps_version="v2.0")[0]
        else:
            hpsv2_score = None
        return niqe_score.item(), qalign_score_a.item(), qalign_score_q.item(), float(hpsv2_score)
    
    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {str(e)}")
        return None,None,None,None

def process_image_directory(directory, output_json="niqe_scores.json"):
    """处理目录下所有图片，计算NIQE分数并保存为JSON"""
    # 支持的图片文件扩展名
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    
    # 获取目录下所有图片文件
    image_files = []
    for root, _, files in os.walk(directory):
        for f in files:
            if Path(f).suffix.lower() in image_extensions:
                image_files.append(os.path.join(root, f))
    
    if not image_files:
        print(f"在目录 {directory} 中未找到任何图片文件")
        return
    try:
        with open(os.path.join(directory, "_prompt.json"), "r") as f:
            prompt_dict = json.load(f)
    except: prompt_dict = None
    # 计算每张图片的NIQE分数
    results = {}
    total = len(image_files)
    ignore_names = ["-ours","-ge","-gg","-sam+mv","-mv","-sam","-wo"]
    for i, img_path in tqdm(enumerate(image_files, 1)):
        rel_path = os.path.relpath(img_path, directory) # 相对路径
        key = rel_path.replace(".png","")
        for n in ignore_names: key = key.replace(n,"")
        prompt = prompt_dict[key] if prompt_dict is not None else ""
        niqe_score, qalign_score_a, qalign_score_q, hpsv2 = calculate_niqe_for_image(img_path, prompt)
        if niqe_score is not None and qalign_score_a is not None and qalign_score_q is not None:
            results[rel_path] = {
                "niqe":round(niqe_score, 4),
                "hpsv2":round(hpsv2,4),
                "qalign_a":round(qalign_score_a, 4),
                "qalign_q":round(qalign_score_q, 4),
            }
            print(f"{rel_path:35s} niqe: {niqe_score:12.8f}, qalign_a: {qalign_score_a:12.8f}, qalign_q: {qalign_score_q:12.8f}, hpsv2: {hpsv2:12.8f}")
    
    # 保存结果到JSON文件
    save_path = os.path.abspath(os.path.join(directory,output_json))
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"处理完成，结果已保存到 {save_path}")
    print(f"成功处理 {len(results)}/{total} 张图片")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='计算图像目录中所有图片的NIQE和QAlign分数')
    parser.add_argument('--dir', type=str, required=True, help='包含图像的目录路径')
    parser.add_argument('--output', type=str, default='niqe_scores.json', help='输出JSON文件的路径 (默认: niqe_scores.json)')
    return parser.parse_args()

if __name__ == "__main__":
    # args = parse_args()
    # process_image_directory("/home/xk/data/xk/GaussianEditor/Images_Metrics/editor"    , "_score.json")
    # process_image_directory("/home/xk/data/xk/GaussianEditor/Images_Metrics/editor"    , "_score.json")
    # process_image_directory("/home/xk/data/xk/GaussianEditor/Images_Metrics/ours"      , "_score.json")
    # process_image_directory("/home/xk/data/xk/GaussianEditor/Images_Metrics/ablation/metrics"  , "_score.json")
    process_image_directory("/home/xk/data/xk/GaussianEditor/Images_Metrics/ablation/Sunflower"  , "_score.json")
    # process_image_directory(args.dir, args.output)