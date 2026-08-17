import os
import torch
import utils3d
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from ..renderers import OctreeRenderer, GaussianRenderer, MeshRenderer
from ..renderers import GaussianRenderer_Absdepth
from ..representations import Octree, Gaussian, MeshExtractResult
from ..modules import sparse as sp
from .random_utils import sphere_hammersley_sequence


def yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, rs, fovs):
    is_list = isinstance(yaws, list)
    if not is_list:
        yaws = [yaws]
        pitchs = [pitchs]
    if not isinstance(rs, list):
        rs = [rs] * len(yaws)
    if not isinstance(fovs, list):
        fovs = [fovs] * len(yaws)
    extrinsics = []
    intrinsics = []
    for yaw, pitch, r, fov in zip(yaws, pitchs, rs, fovs):
        fov = torch.deg2rad(torch.tensor(float(fov))).cuda()
        yaw = torch.tensor(float(yaw)).cuda()
        pitch = torch.tensor(float(pitch)).cuda()
        orig = torch.tensor([
            torch.sin(yaw) * torch.cos(pitch),
            torch.cos(yaw) * torch.cos(pitch),
            torch.sin(pitch),
        ]).cuda() * r
        extr = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
        intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        extrinsics.append(extr)
        intrinsics.append(intr)
    if not is_list:
        extrinsics = extrinsics[0]
        intrinsics = intrinsics[0]
    return extrinsics, intrinsics


def get_renderer(sample, **kwargs):
    if isinstance(sample, Octree):
        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = kwargs.get('resolution', 512)
        renderer.rendering_options.near = kwargs.get('near', 0.8)
        renderer.rendering_options.far = kwargs.get('far', 1.6)
        renderer.rendering_options.bg_color = kwargs.get('bg_color', (0, 0, 0))
        renderer.rendering_options.ssaa = kwargs.get('ssaa', 4)
        renderer.pipe.primitive = sample.primitive
    elif isinstance(sample, Gaussian):
        renderer = GaussianRenderer_Absdepth()
        renderer.rendering_options.resolution = kwargs.get('resolution', 512)
        renderer.rendering_options.near = kwargs.get('near', 0.8)
        renderer.rendering_options.far = kwargs.get('far', 1.6)
        renderer.rendering_options.bg_color = kwargs.get('bg_color', (0, 0, 0))
        renderer.rendering_options.ssaa = kwargs.get('ssaa', 1)
        renderer.pipe.kernel_size = kwargs.get('kernel_size', 0.1)
        renderer.pipe.use_mip_gaussian = True
        # The absdepth CUDA extension used by this project does not reliably
        # evaluate SH colors.  Convert degree-0 SH to RGB in Python so the
        # RGB channel is populated while preserving the depth outputs.
        renderer.pipe.convert_SHs_python = True
    elif isinstance(sample, MeshExtractResult):
        renderer = MeshRenderer()
        renderer.rendering_options.resolution = kwargs.get('resolution', 512)
        renderer.rendering_options.near = kwargs.get('near', 1)
        renderer.rendering_options.far = kwargs.get('far', 100)
        renderer.rendering_options.ssaa = kwargs.get('ssaa', 4)
    else:
        raise ValueError(f'Unsupported sample type: {type(sample)}')
    return renderer


def render_frames(sample, extrinsics, intrinsics, options={}, colors_overwrite=None, verbose=True, **kwargs):
    renderer = get_renderer(sample, **options)
    returns = {}
    for j, (extr, intr) in tqdm(enumerate(zip(extrinsics, intrinsics)), desc='Rendering', disable=not verbose):

        if 'extr' not in returns: returns['extr'] = []
        if 'intr' not in returns: returns['intr'] = []
        returns['extr'].append(extr)
        returns['intr'].append(intr)
        
        if isinstance(sample, MeshExtractResult):
            render_result = renderer.render(sample, extr, intr)
            if 'normal' not in returns: returns['normal'] = []
            returns['normal'].append(np.clip(render_result['normal'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
        else:
            render_result = renderer.render(sample, extr, intr, colors_overwrite=colors_overwrite)
            if 'color'    not in returns: returns['color']    = [] # 创建初始列表
            if 'depth'    not in returns: returns['depth']    = [] # 创建初始列表
            if 'absdepth' not in returns: returns['absdepth'] = [] # 创建初始列表
            
            returns['color'].append(np.clip(render_result['color'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
            if 'percent_depth' in render_result: # 优先用percent_depth
                returns['depth'].append(render_result['percent_depth'].detach().cpu().numpy())
            elif 'depth' in render_result:
                returns['depth'].append(render_result['depth'].detach().cpu().numpy())
            elif 'invdepth' in render_result:
                returns['depth'].append(render_result['invdepth'].detach().cpu().numpy())
            else:
                returns['depth'].append(None)

            if 'absdepth' in render_result:
                returns['absdepth'].append(render_result['absdepth'].detach().cpu().numpy())
            else:
                returns['absdepth'].append(None)
    return returns


def render_video(sample, resolution=512, bg_color=(0, 0, 0), num_frames=300, r=2, fov=40, **kwargs):
    yaws = torch.linspace(0, 2 * 3.1415, num_frames)
    pitch = 0.25 + 0.5 * torch.sin(torch.linspace(0, 2 * 3.1415, num_frames))
    yaws = yaws.tolist()
    pitch = pitch.tolist()
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, r, fov)
    print(f"Render Function: render_video")
    return render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color}, **kwargs)

def render_multiview(sample, resolution=512, nviews=30):
    r = 2
    fov = 40
    cams = [sphere_hammersley_sequence(i, nviews) for i in range(nviews)]
    yaws = [cam[0] for cam in cams]
    pitchs = [cam[1] for cam in cams]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, r, fov)
    print(f"Render Function: render_multiview")
    res = render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': (0, 0, 0)})
    return res['color'], extrinsics, intrinsics


def render_snapshot(samples, resolution=512, bg_color=(0, 0, 0), offset=(-16 / 180 * np.pi, 20 / 180 * np.pi), r=10, fov=8, **kwargs):
    yaw = [0, np.pi/2, np.pi, 3*np.pi/2]
    yaw_offset = offset[0]
    yaw = [y + yaw_offset for y in yaw]
    pitch = [offset[1] for _ in range(4)]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, r, fov)
    print(f"Render Function: render_snapshot")
    return render_frames(samples, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color}, **kwargs)

# @InsertAny3D
def render_sphere(sample: Gaussian, r = 2, latitudes_deg = [10], fov = 40, resolution=512, nviews_one_lat=30):
    extrs, intrs = [],[]
    for la_deg in latitudes_deg:
        la_rad = la_deg*torch.pi/180
        yaws = torch.linspace(0, 2*torch.pi, nviews_one_lat+1)[:-1].tolist() # 去重，30个
        pitchs = [la_rad] * nviews_one_lat
        extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, r, fov)
        extrs += extrinsics
        intrs += intrinsics
    print(f"Render Function: render_sphere")
    return render_frames(sample, extrs, intrs, {'resolution': resolution, 'bg_color': (0, 0, 0)})

# @InsertAny3D
def save_colmap_image_camera(image_names, extrinsics, intrinsics, resolution, file_folder):
    if image_names == None: image_names = ["_"] * len(extrinsics)
    assert len(image_names) == len(extrinsics)
    assert len(extrinsics) == len(intrinsics)

    unique_intrinsics = []
    camera_id_map = []
    for i in intrinsics:
        if isinstance(i, torch.Tensor):i_np = i.cpu().numpy()
        else: i_np = i
        i_tuple = tuple(np.round(i_np.flatten(), 6))
        if i_tuple not in unique_intrinsics:
            unique_intrinsics.append(i_tuple)
        camera_id_map.append(unique_intrinsics.index(i_tuple) + 1)

    os.makedirs(file_folder, exist_ok=True)
    with open(file_folder + "/cameras.txt", "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: {}\n".format(len(unique_intrinsics)))
        for idx, i_tuple in enumerate(unique_intrinsics):
            i_np = np.array(i_tuple).reshape(3, 3)
            fx, fy, cx, cy = i_np[0, 0] * resolution, i_np[1, 1] * resolution, i_np[0, 2] * resolution, i_np[1, 2] * resolution
            camera_id = idx + 1
            f.write(f"{camera_id} PINHOLE {resolution} {resolution} {fx} {fy} {cx} {cy}\n")
        # print(f"[INFO] cameras saved to {file_folder + '/cameras.txt'}")

    with open(file_folder + "/images.txt", "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for idx, (name, e, cam_id) in enumerate(zip(image_names, extrinsics, camera_id_map)):
            if isinstance(e, torch.Tensor): e = e.cpu().numpy()
            r, t = e[:3, :3], e[:3, 3]
            qx, qy, qz, qw = R.from_matrix(r).as_quat()
            image_id = idx + 1
            f.write(f"{image_id} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} {cam_id} {name}\n")
            f.write("\n")
        # print(f"[INFO] images saved to {file_folder + '/images.txt'}")
