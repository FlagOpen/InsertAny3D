import os
import cv2
import json
import imageio
import subprocess
import torch
import numpy as np
from typing import Literal
from argparse import Namespace
from trellis.representations import Gaussian
from trellis.utils.render_utils import render_sphere, save_colmap_image_camera, render_video
from trellis.utils.insertany3d_gs_utils import transform_gaussian, read_extrinsics_text, read_intrinsics_text, readColmapCameras, camera_to_JSON

import datetime
current_time = lambda : datetime.datetime.now().strftime("%y_%m_%d-%H_%M_%S") #2025_08_05-10:00:00

def load_gaussian(ply_path):
    trellis_gs = Gaussian([-0.5,-0.5,-0.5,0.5,0.5,0.5])
    # Older TRELLIS renderer code refers to max_sh_degree while this compact
    # representation exposes sh_degree only.
    trellis_gs.max_sh_degree = trellis_gs.sh_degree
    trellis_gs.load_ply(ply_path)
    # Normalize legacy PLYs written with infinite opacity logits as well as
    # newly decoded assets; this keeps the CUDA rasterizer deterministic.
    with torch.no_grad():
        opacity = torch.nan_to_num(trellis_gs.get_opacity, nan=1e-4, posinf=1.0 - 1e-4, neginf=1e-4)
        trellis_gs.from_opacity(opacity.clamp(1e-4, 1.0 - 1e-4))
    return trellis_gs

def _save_result(save_path, images, invdepths, absdepths, extrinsics, intrinsics, resolution, image_names=None):
    """ For InsertAny3D Pipeline
    save_path/
    ├── images/                      # 渲染图像
    │   ├── 0.png                   # 渲染图像序列
    │   ├── 1.png
    │   └── ...
    ├── sparse/                      # COLMAP格式相机参数
    │   └── 0/
    │       ├── cameras.txt         # 相机内参
    │       └── images.txt          # 相机外参
    └── depths/                      # 深度图
        ├── invdepth/               # 逆深度图(uint8)
        │   ├── 0.png
        │   ├── 1.png
        │   └── ...
        └── absdepth/               # 绝对深度图(float32)
            ├── 0.raw
            ├── 1.raw
            └── ...
    """
    if image_names is None:
        image_names = [f"{idx}.png" for idx in range(len(images))]
    else:
        image_names = [str(name) for name in image_names]
        if len(image_names) != len(images):
            raise ValueError(f"image_names 数量 ({len(image_names)}) 与图片数量 ({len(images)}) 不一致")
        image_names = [name if name.lower().endswith(".png") else f"{name}.png" for name in image_names]
        if any(os.path.basename(name) != name or name in {".", ".."} for name in image_names):
            raise ValueError("image_names 只能包含文件名")

    if images:
        os.makedirs(f"{save_path}/images", exist_ok=True)
        for idx, i in enumerate(images):
            cv2.imwrite(f"{save_path}/images/{image_names[idx]}", cv2.cvtColor(i, cv2.COLOR_RGB2BGR))
        save_colmap_image_camera(image_names, extrinsics, intrinsics, resolution, f"{save_path}/sparse/0")

    if invdepths:
        os.makedirs(f"{save_path}/depths/invdepth", exist_ok=True)
        for idx, d in enumerate(invdepths):
            invdepth_img = (d.squeeze(0) * 255).astype(np.uint8)
            stem = os.path.splitext(image_names[idx])[0]
            cv2.imwrite(f"{save_path}/depths/invdepth/{stem}.png", invdepth_img)
    
    if absdepths:
        os.makedirs(f"{save_path}/depths/absdepth", exist_ok=True)
        for idx, d in enumerate(absdepths): # 保存为raw格式(float32)
            absdepth_raw = d.squeeze(0).astype(np.float32)
            stem = os.path.splitext(image_names[idx])[0]
            with open(f"{save_path}/depths/absdepth/{stem}.raw", 'wb') as f:
                f.write(absdepth_raw.tobytes())
                
def _extract_3dgs_result(scene_path, result_path, gaussians_path, use_colmap=False):
    """ Match 3DGS Original output folder as closely as possible
    result_path/
    ├── point_cloud/                    # 高斯点云文件夹
    │   └── iteration_30000/           
    │       └── point_cloud.ply         # 转换为z-down坐标系的高斯点云
    ├── cameras.json                    # 相机参数(3DGS格式)
    └── cfg_args                        # 配置参数文件
    
    scene_path/
    └── sparse/                         # COLMAP格式相机参数
        └── 0/
            └─ + points3D.txt/bin       # 三维点云(COLMAP/Fake)
    """

    # 结果文件夹中的gaussian ply文件
    ply_save_path = f"{result_path}/point_cloud/iteration_30000"
    os.makedirs(ply_save_path, exist_ok=True)
    # Trellis的模型是y-up的，但是相机位姿是z-down的，需要转成z-down的
    transform_gaussian(gaussians_path, out_path=ply_save_path+"/point_cloud.ply",
        transform      = [[1,0,0],[0,0,-1],[0,1,0]],
        user_transform = [[1,0,0],[0,0,-1],[0,1,0]])
    # print(f"[INFO] Saved: - [Gaussian.ply] to '{ply_save_path}/point_cloud.ply'")

    # 结果文件夹中的cameras.json
    json_cams = []
    camlist = []
    cam_extrinsics = read_extrinsics_text(f"{scene_path}/sparse/0/images.txt")
    cam_intrinsics = read_intrinsics_text(f"{scene_path}/sparse/0/cameras.txt")
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder     = f"{scene_path}/images",
        )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: int(x.image_name) if x.image_name.isdigit() else x.image_name)
    camlist.extend(cam_infos)
    for id, cam in enumerate(camlist):  json_cams.append(camera_to_JSON(id, cam))
    with open(f"{result_path}/cameras.json", 'w') as file: 
        file.write('[\n')
        for i, cam in enumerate(json_cams):
            json.dump(cam, file, indent=None, separators=(',', ':'), ensure_ascii=False)
            if i < len(json_cams) - 1: file.write(',')
            file.write('\n')
        file.write(']\n')
    # print(f"[INFO] Saved: - [cameras.json] to '{result_path}/cameras.json'")


    # 结果文件夹中的cfg_args
    args = {
        "allow_principle_point_shift" : False,
        "data_device"                 : 'cuda',
        "eval"                        : False,
        "feature_dim"                 : 32,
        "feature_model_path"          : '',
        "images"                      : 'images',
        "init_from_3dgs_pcd"          : False,
        "model_path"                  : os.path.abspath(result_path),
        "need_features"               : False,
        "need_masks"                  : False,
        "resolution"                  : -1,
        "sh_degree"                   : 0,
        "source_path"                 : os.path.abspath(scene_path),
        "white_background"            : False
    }
    with open(f"{result_path}/cfg_args", "w") as f:
        f.write(str(Namespace(**args)))
    # print(f"[INFO] Saved: - [cfg_args]     to '{result_path}/cfg_args'")


    # 结果文件夹中的points3D.bin/points3D.txt
    if use_colmap is True:
        commands = [
            f"rm -rf {scene_path}/sparse/triangulated",
            f"mkdir {scene_path}/sparse/triangulated",
            f"colmap feature_extractor \
                --database_path {scene_path}/database.db \
                --image_path    {scene_path}/images \
                --ImageReader.single_camera 1",
            f"colmap exhaustive_matcher \
                --database_path {scene_path}/database.db",
            f"colmap model_converter \
                --input_path    {scene_path}/sparse/0 \
                --output_path   {scene_path}/sparse/0 \
                --output_type BIN",
            f"colmap point_triangulator \
                --database_path {scene_path}/database.db \
                --image_path    {scene_path}/images \
                --input_path    {scene_path}/sparse/0 \
                --output_path   {scene_path}/sparse/triangulated",
            f"cp {scene_path}/sparse/triangulated/points3D.bin \
                {scene_path}/sparse/0/points3D.bin",
            f"rm -rf {scene_path}/sparse/triangulated"
        ]
        for cmd in commands:
            result = subprocess.run(cmd, shell=True)
            if result.returncode != 0:
                print(f"[Warning] COLMAP raised error-({result.returncode}) while running {cmd}")

    else:
        with open(f'{scene_path}/sparse/0/points3D.txt', "w") as f: f.writelines(["# None"])
        os.system(f"rm -f {scene_path}/sparse/0/points3D.bin > /dev/null 2>&1")
        os.system(f"rm -f {scene_path}/sparse/0/points3D.ply > /dev/null 2>&1")
    # print(f"[INFO] Saved: - [points3D]     to '{scene_path}/sparse/0/'")

def _apply(gaussian_path, scene_path = None, resolution = 512, render_mode: Literal["sphere"]="sphere", fov=40, extract_3dgs_result_path = None):
    # Loading Gaussian
    trellis_gs = load_gaussian(gaussian_path)
    # Rendering
    if render_mode == "sphere": 
        render_pkg = render_sphere(
            sample=trellis_gs,
            r=1.5,
            latitudes_deg=[10,20,30],
            fov=fov,
            resolution=resolution,
            nviews_one_lat=30,
        )
        # TODO: key error 'absdepth'
        images, invdepths, absdepths, extrinsics, intrinsics = \
            render_pkg["color"], render_pkg["depth"], render_pkg["absdepth"], render_pkg["extr"], render_pkg["intr"]
    elif render_mode == "hammersley":
        raise NotImplementedError("Render by Hammersley is not implemented.")
        images, invdepths, absdepths, extrinsics, intrinsics = render_hammersley_with_depth(
            sample=trellis_gs,
            distances=2,
            fov=fov,
            resolution=resolution,
            nviews_one_lat=20,
        )
    # Saving
    if scene_path:
        _save_result(scene_path, images, invdepths, absdepths, extrinsics, intrinsics, resolution)
        print(f"[INFO] Saved: [Dataset File]   to '{scene_path}'")
        if extract_3dgs_result_path is not None:
            _extract_3dgs_result(scene_path, extract_3dgs_result_path, gaussian_path, use_colmap=False)
            print(f"[INFO] Saved: [Pretrain File]  to '{extract_3dgs_result_path}'")
    return images, invdepths, absdepths, extrinsics, intrinsics

def save_gaussian_result(gaussian:Gaussian, save_base, save_name, processed_image=None):
    save_path = f"{save_base}/{save_name+'/' if save_name is not None else ''}{current_time()}"
    print(f"[PIPE] Save to: {save_path}")
    
    os.makedirs(save_path, exist_ok=True)
    gaussian_path = f"{save_path}/sample.ply"
    gaussian.save_ply(gaussian_path)
    fov = 53.1301023542
    resolution=1024
    _apply(gaussian_path, os.path.join(save_path,"source"),\
        resolution, render_mode="sphere", fov=fov, \
        extract_3dgs_result_path=os.path.join(save_path,"model"))
    if processed_image is not None:
        p_image_path = os.path.join(save_path,"processed_image.png")
        processed_image.save(p_image_path)
    return ""
