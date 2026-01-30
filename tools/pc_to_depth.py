import os
import numpy as np
from tqdm import tqdm

# ================= 配置路径 =================
PC_ROOT = "D:/1Ahaha/AA3d/ShapeNet_PointClouds"
CAM_ROOT = "D:/1Ahaha/AA3d/output_224"
DEPTH_ROOT = "D:/1Ahaha/AA3d/depth_maps"

IMG_SIZE = 224
Z_NEAR = 0.01
Z_FAR = 10.0


# ================= 你的原函数（加少量容错） =================
def pointcloud_to_depth(points, cam2world, K, img_size=224):
    # 新增：矩阵不可逆容错
    try:
        world2cam = np.linalg.inv(cam2world)
    except np.linalg.LinAlgError:
        return np.zeros((img_size, img_size), dtype=np.float32)
    
    world2cam[2, :] *= -1  # 修复 z 轴方向

    N = points.shape[0]
    points_h = np.concatenate([points, np.ones((N, 1))], axis=1)
    points_cam = (world2cam @ points_h.T).T[:, :3]

    z = points_cam[:, 2]
    valid = (z > Z_NEAR) & (z < Z_FAR)
    points_cam = points_cam[valid]

    if len(points_cam) == 0:
        return np.zeros((img_size, img_size), dtype=np.float32)

    proj = (K @ points_cam.T).T
    # 新增：避免除零错误
    if np.all(proj[:, 2] == 0):
        return np.zeros((img_size, img_size), dtype=np.float32)
    
    proj[:, 0] /= proj[:, 2]
    proj[:, 1] /= proj[:, 2]

    x = proj[:, 0].astype(np.int32)
    y = proj[:, 1].astype(np.int32)
    z = points_cam[:, 2]

    depth = np.zeros((img_size, img_size), dtype=np.float32)

    for xi, yi, zi in zip(x, y, z):
        if 0 <= xi < img_size and 0 <= yi < img_size:
            if depth[yi, xi] == 0 or zi < depth[yi, xi]:
                depth[yi, xi] = zi

    return depth


def normalize_depth(depth):
    mask = depth > 0
    if not np.any(mask):
        return depth
    d_min = depth[mask].min()
    d_max = depth[mask].max()
    # 新增：避免除以零（深度值全相同的极端情况）
    if d_max - d_min < 1e-6:
        depth[mask] = 0.0
    else:
        depth[mask] = (depth[mask] - d_min) / (d_max - d_min + 1e-6)
    return depth


def generate_depth_maps(pc_path, cam_param_path, output_dir):
    # 新增：前置文件存在性检查
    if not os.path.exists(pc_path):
        raise FileNotFoundError(f"Point cloud file not found: {pc_path}")
    if not os.path.exists(cam_param_path):
        raise FileNotFoundError(f"Camera params file not found: {cam_param_path}")
    
    os.makedirs(output_dir, exist_ok=True)

    points = np.load(pc_path)
    # 新增：点云维度校验
    if points.shape != (2048, 3):
        raise ValueError(f"Invalid point cloud shape: {points.shape}, expected (2048,3)")
    
    cam_data = np.load(cam_param_path)
    # 新增：相机参数键值校验
    if "cam2world" not in cam_data or "K" not in cam_data:
        raise KeyError("Camera params missing 'cam2world' or 'K'")
    
    cam2world = cam_data["cam2world"]
    K = cam_data["K"]

    depth_maps = []

    for i in range(len(cam2world)):
        depth = pointcloud_to_depth(points, cam2world[i], K)
        depth = normalize_depth(depth)
        depth_maps.append(depth)

        np.save(
            os.path.join(output_dir, f"depth_{i:02d}.npy"),
            depth.astype(np.float32)
        )

    depth_maps = np.stack(depth_maps, axis=0)
    np.save(os.path.join(output_dir, "depth_maps.npy"), depth_maps)


# ================= ShapeNet55 批处理主逻辑 =================
def batch_process_shapenet55():
    # 新增：根目录存在性检查
    for root in [PC_ROOT, CAM_ROOT]:
        if not os.path.exists(root):
            print(f"❌ Root directory not found: {root}")
            return
    
    categories = sorted(os.listdir(PC_ROOT))
    # 过滤非目录的文件（避免杂项文件干扰）
    categories = [c for c in categories if os.path.isdir(os.path.join(PC_ROOT, c))]

    # 新增：统计变量
    total_success = 0
    total_skip = 0
    total_failed = 0

    for category in categories:
        pc_cat_dir = os.path.join(PC_ROOT, category)
        cam_cat_dir = os.path.join(CAM_ROOT, f"{category}_multi_view")
        depth_cat_dir = os.path.join(DEPTH_ROOT, category)

        print(f"\n🚀 Processing category: {category}")

        # 获取该类别下所有npy点云文件
        pc_files = [f for f in os.listdir(pc_cat_dir) if f.endswith(".npy")]
        for pc_file in tqdm(pc_files, desc=f"Processing {category} objects"):
            obj_id = pc_file.replace(".npy", "")

            pc_path = os.path.join(pc_cat_dir, pc_file)
            cam_path = os.path.join(cam_cat_dir, obj_id, "camera_params.npz")
            out_dir = os.path.join(depth_cat_dir, obj_id)

            # 跳过缺失相机参数的
            if not os.path.exists(cam_path):
                total_skip += 1
                continue

            # 已生成则跳过（支持断点续跑）
            if os.path.exists(os.path.join(out_dir, "depth_maps.npy")):
                total_skip += 1
                continue

            try:
                generate_depth_maps(pc_path, cam_path, out_dir)
                total_success += 1
            except Exception as e:
                print(f"\n❌ Failed: {category}/{obj_id} → {e}")
                total_failed += 1

    # 新增：批量处理完成统计
    print("\n==================== 批量处理完成 ====================")
    print(f"✅ 成功生成：{total_success} 个物体")
    print(f"⏭️  跳过（已生成/缺参数）：{total_skip} 个物体")
    print(f"❌ 处理失败：{total_failed} 个物体")
    print(f"📁 深度图根目录：{DEPTH_ROOT}")


if __name__ == "__main__":
    batch_process_shapenet55()