"""
功能：从 OS-MN40-core 点云批量生成深度图。

说明：
    这个脚本会根据已有 RGB 视图文件名里的角度信息，
    旋转点云后进行光栅化，生成对应的深度图序列。
"""

import argparse
import math
import os
import re
import sys
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.exp_config import BASE_DIR

DATA_ROOT = os.path.join(BASE_DIR, "OS_MN40_core")
OUTPUT_ROOT = os.path.join(BASE_DIR, "OS_MN40_core_depth_maps")

ANGLE_PATTERN = re.compile(r"h_(\d+)\.jpg$", re.IGNORECASE)


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Generate depth maps for OS-MN40-core from point clouds."
    )
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--output_root", type=str, default=OUTPUT_ROOT)
    parser.add_argument(
        "--splits",
        type=str,
        default="train,query,target",
        help="Comma-separated subsets to process.",
    )
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--dilation_kernel", type=int, default=5)
    parser.add_argument("--blur_kernel", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_split_list(raw_value: str) -> List[str]:
    """把逗号分隔的 split 参数解析成列表。"""
    values = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one split name.")
    return values


def normalize_points(points: np.ndarray) -> np.ndarray:
    """
    功能：对点云做中心化和单位球归一化。
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected a point cloud with shape [N,3+], got {points.shape}")

    xyz = points[:, :3].copy()
    centroid = xyz.mean(axis=0, keepdims=True)
    xyz -= centroid
    radius = np.linalg.norm(xyz, axis=1).max()
    if radius > 0:
        xyz /= radius
    return xyz


def load_pts(path: str) -> np.ndarray:
    """
    功能：读取 `.pts` 点云文件，并做标准化。
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError(f"Empty point cloud file: {path}")

    try:
        declared_count = int(lines[0])
        point_lines = lines[1:]
    except ValueError:
        declared_count = None
        point_lines = lines

    points = np.loadtxt(point_lines, dtype=np.float32)
    if points.ndim == 1:
        points = points[None, :]

    if declared_count is not None and declared_count != points.shape[0]:
        raise ValueError(
            f"Point count mismatch in {path}: header says {declared_count}, "
            f"but parsed {points.shape[0]}"
        )

    return normalize_points(points)


def parse_view_angles(image_dir: str) -> List[Tuple[str, int]]:
    """
    功能：从 RGB 视图文件名中解析所有拍摄角度。
    """
    views = []
    for filename in os.listdir(image_dir):
        match = ANGLE_PATTERN.match(filename)
        if not match:
            continue
        angle = int(match.group(1))
        views.append((filename, angle))

    views.sort(key=lambda item: item[1])
    if not views:
        raise ValueError(f"No h_*.jpg views found in {image_dir}")
    return views


def rotate_points_z(points: np.ndarray, angle_deg: float) -> np.ndarray:
    """绕 z 轴旋转点云到指定角度。"""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    rot = np.array(
        [
            [cos_a, -sin_a, 0.0],
            [sin_a, cos_a, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return points @ rot.T


def rasterize_depth(points: np.ndarray, image_size: int) -> np.ndarray:
    """
    功能：把旋转后的点云光栅化成单张深度图。
    """
    if points.shape[0] == 0:
        return np.zeros((image_size, image_size), dtype=np.float32)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    extent = np.max(np.abs(np.stack([x, z], axis=1)))
    if extent <= 1e-8:
        return np.zeros((image_size, image_size), dtype=np.float32)

    px = ((x / extent) * 0.5 + 0.5) * (image_size - 1)
    py = ((-z / extent) * 0.5 + 0.5) * (image_size - 1)
    depth = -y

    depth_map = np.full((image_size, image_size), np.inf, dtype=np.float32)
    xi = np.clip(np.round(px).astype(np.int32), 0, image_size - 1)
    yi = np.clip(np.round(py).astype(np.int32), 0, image_size - 1)

    for cur_x, cur_y, cur_depth in zip(xi, yi, depth):
        if cur_depth < depth_map[cur_y, cur_x]:
            depth_map[cur_y, cur_x] = cur_depth

    valid = np.isfinite(depth_map)
    if not np.any(valid):
        return np.zeros((image_size, image_size), dtype=np.float32)

    valid_values = depth_map[valid]
    depth_min = valid_values.min()
    depth_max = np.percentile(valid_values, 95)
    if depth_max - depth_min < 1e-6:
        out = np.zeros_like(depth_map)
        out[valid] = 1.0
        return out

    out = np.zeros_like(depth_map)
    clipped = np.clip(depth_map[valid], depth_min, depth_max)
    out[valid] = 1.0 - (clipped - depth_min) / (depth_max - depth_min + 1e-6)
    return out


def postprocess_depth(
    depth: np.ndarray,
    dilation_kernel: int,
    blur_kernel: int,
) -> np.ndarray:
    """
    功能：对深度图做膨胀和模糊后处理。
    """
    depth = np.asarray(depth, dtype=np.float32)
    depth_uint8 = (np.clip(depth, 0.0, 1.0) * 255).astype(np.uint8)
    image = Image.fromarray(depth_uint8, mode="L")

    if dilation_kernel > 1:
        if dilation_kernel % 2 == 0:
            dilation_kernel += 1
        image = image.filter(ImageFilter.MaxFilter(size=dilation_kernel))

    if blur_kernel > 1:
        radius = max(0.5, (blur_kernel - 1) / 2.0)
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))

    return np.asarray(image, dtype=np.float32) / 255.0


def save_depth_stack(
    pointcloud_path: str,
    image_dir: str,
    output_dir: str,
    image_size: int,
    dilation_kernel: int,
    blur_kernel: int,
) -> int:
    """
    功能：为单个物体生成整套深度图序列。

    返回：
        实际保存的视图数量
    """
    os.makedirs(output_dir, exist_ok=True)
    points = load_pts(pointcloud_path)
    views = parse_view_angles(image_dir)

    for index, (_, angle_deg) in enumerate(views):
        rotated = rotate_points_z(points, angle_deg)
        depth = rasterize_depth(rotated, image_size=image_size)
        depth = postprocess_depth(
            depth,
            dilation_kernel=dilation_kernel,
            blur_kernel=blur_kernel,
        )
        depth_uint8 = (depth * 255).astype(np.uint8)
        Image.fromarray(depth_uint8, mode="L").save(
            os.path.join(output_dir, f"depth_{index:02d}.png")
        )

    return len(views)


def iter_train_objects(train_root: str) -> Iterable[Tuple[str, str, str]]:
    """遍历 train split 下的 `(class_name, object_id, object_dir)`。"""
    for class_name in sorted(os.listdir(train_root)):
        class_dir = os.path.join(train_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        for object_id in sorted(os.listdir(class_dir)):
            object_dir = os.path.join(class_dir, object_id)
            if not os.path.isdir(object_dir):
                continue
            yield class_name, object_id, object_dir


def iter_flat_objects(root: str) -> Iterable[Tuple[str, str]]:
    """遍历 query / target 这类扁平目录下的物体。"""
    for object_id in sorted(os.listdir(root)):
        object_dir = os.path.join(root, object_id)
        if not os.path.isdir(object_dir):
            continue
            yield object_id, object_dir


def main():
    """
    功能：按 split 批量生成 OS-MN40-core 深度图。
    """
    args = parse_args()
    splits = parse_split_list(args.splits)
    total_saved = 0
    total_skipped = 0

    print(f"Data root: {args.data_root}")
    print(f"Output root: {args.output_root}")
    print(f"Splits: {splits}")
    print(f"Image size: {args.image_size}")

    for split_name in splits:
        split_root = os.path.join(args.data_root, split_name)
        if not os.path.isdir(split_root):
            raise FileNotFoundError(f"Missing split root: {split_root}")

        if split_name == "train":
            iterable = list(iter_train_objects(split_root))
            progress = tqdm(iterable, desc=f"Depth {split_name}")
            for class_name, object_id, object_dir in progress:
                output_dir = os.path.join(args.output_root, split_name, class_name, object_id)
                if (
                    os.path.exists(os.path.join(output_dir, "depth_23.png"))
                    and not args.overwrite
                ):
                    total_skipped += 1
                    continue

                image_dir = os.path.join(object_dir, "image")
                pointcloud_path = os.path.join(object_dir, "pointcloud", "pt_1024.pts")
                # 根据 RGB 视图角度生成同顺序的深度图
                save_depth_stack(
                    pointcloud_path=pointcloud_path,
                    image_dir=image_dir,
                    output_dir=output_dir,
                    image_size=args.image_size,
                    dilation_kernel=args.dilation_kernel,
                    blur_kernel=args.blur_kernel,
                )
                total_saved += 1
        else:
            iterable = list(iter_flat_objects(split_root))
            progress = tqdm(iterable, desc=f"Depth {split_name}")
            for object_id, object_dir in progress:
                output_dir = os.path.join(args.output_root, split_name, object_id)
                if (
                    os.path.exists(os.path.join(output_dir, "depth_23.png"))
                    and not args.overwrite
                ):
                    total_skipped += 1
                    continue

                image_dir = os.path.join(object_dir, "image")
                pointcloud_path = os.path.join(object_dir, "pointcloud", "pt_1024.pts")
                # query / target 与 train 的目录结构不同，但生成流程一致
                save_depth_stack(
                    pointcloud_path=pointcloud_path,
                    image_dir=image_dir,
                    output_dir=output_dir,
                    image_size=args.image_size,
                    dilation_kernel=args.dilation_kernel,
                    blur_kernel=args.blur_kernel,
                )
                total_saved += 1

    print(f"Saved depth stacks: {total_saved}")
    print(f"Skipped existing objects: {total_skipped}")


if __name__ == "__main__":
    main()
