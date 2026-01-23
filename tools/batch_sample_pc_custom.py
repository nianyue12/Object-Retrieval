import os
import numpy
import trimesh
from misc_utils import trimesh_to_pc  # 导入调试好的采样函数


# ====================== 核心配置（按需修改） ======================
SHAPENET_ROOT = r"D:\1Ahaha\AA3d\ShapeNet"  # ShapeNet根目录（含所有类别子文件夹）
PC_OUTPUT_ROOT = r"D:\1Ahaha\AA3d\ShapeNet_PointClouds"  # 点云保存目录
N_SAMPLE_POINTS = 2048  # 采样点数量
# ==================================================================

def batch_sample_category(category_name, category_input_dir):
    """处理单个指定类别的所有GLB文件，返回统计结果"""
    # 1. 构造输出路径
    category_output_dir = os.path.join(PC_OUTPUT_ROOT, category_name)
    os.makedirs(category_output_dir, exist_ok=True)

    # 2. 遍历该类别下的所有GLB文件
    glb_files = [f for f in os.listdir(category_input_dir) if f.endswith(".glb")]
    total_count = len(glb_files)
    
    if total_count == 0:
        print(f"类别{category_name}：无GLB文件，跳过")
        return {
            "category": category_name,
            "total": 0,
            "success": 0,
            "failed": 0,
            "existing": 0
        }

    # 3. 过滤已采样的文件（断点续采）
    existing_npy = [f.replace(".npy", ".glb") for f in os.listdir(category_output_dir) if f.endswith(".npy")]
    existing_count = len(existing_npy)
    to_process = [f for f in glb_files if f not in existing_npy]
    to_process_count = len(to_process)
    
    if to_process_count == 0:
        print(f"类别{category_name}：所有文件已采样完成，无需处理")
        return {
            "category": category_name,
            "total": total_count,
            "success": existing_count,
            "failed": 0,
            "existing": existing_count
        }

    # 4. 逐个采样并保存
    print(f"\n开始处理类别：{category_name}（待处理{to_process_count}/{total_count}个模型）")
    success_count = 0
    fail_count = 0
    for idx, glb_file in enumerate(to_process, 1):
        try:
            # 加载GLB模型
            glb_path = os.path.join(category_input_dir, glb_file)
            mesh = trimesh.load(glb_path, force="scene")

            # 采样点云
            pc = trimesh_to_pc(mesh)

            # 裁剪/补全到2048个点
            if pc.shape[0] >= N_SAMPLE_POINTS:
                pc = pc[numpy.random.permutation(len(pc))[:N_SAMPLE_POINTS]]
            elif pc.shape[0] < N_SAMPLE_POINTS:
                pc = numpy.concatenate([pc, pc[numpy.random.randint(len(pc), size=[N_SAMPLE_POINTS - len(pc)])]])

            # 仅保留 XYZ
            xyz = pc[:, :3]

            # ======== 单位球归一化（HGM2R 必须） ========
            centroid = numpy.mean(xyz, axis=0)
            xyz = xyz - centroid
            max_dist = numpy.max(numpy.sqrt(numpy.sum(xyz ** 2, axis=1)))
            if max_dist > 0:
                xyz = xyz / max_dist
            # ==========================================

            # 保存
            output_npy_path = os.path.join(category_output_dir, glb_file.replace(".glb", ".npy"))
            numpy.save(output_npy_path, xyz)


            success_count += 1
            # 每处理10个文件打印进度
            if idx % 10 == 0:
                print(f"  进度：{idx}/{to_process_count} | 成功：{success_count} | 失败：{fail_count}")
        except Exception as e:
            fail_count += 1
            print(f" 处理{glb_file}失败：{str(e)[:50]}...")
            continue
    
    # 该类别最终统计（已存在的+本次成功的）
    final_success = existing_count + success_count
    print(f"类别{category_name}处理完成 | 总计：{total_count} | 成功：{final_success} | 失败：{fail_count}")
    
    return {
        "category": category_name,
        "total": total_count,
        "success": final_success,
        "failed": fail_count,
        "existing": existing_count
    }


if __name__ == "__main__":
    # 1. 自动遍历ShapeNet根目录下的所有子文件夹（即所有类别）
    all_categories = []
    for item in os.listdir(SHAPENET_ROOT):
        item_path = os.path.join(SHAPENET_ROOT, item)
        if os.path.isdir(item_path):  # 只处理文件夹（类别）
            all_categories.append((item, item_path))
    
    if not all_categories:
        print("ShapeNet根目录下无有效类别文件夹，程序退出")
        exit()
    
    print(f"\n检测到ShapeNet共{len(all_categories)}个类别，开始批量采样")
    print(f"采样结果将保存到：{PC_OUTPUT_ROOT}")
    
    # 2. 逐个处理所有类别，收集统计结果
    summary_stats = []
    for category_name, category_path in all_categories:
        stats = batch_sample_category(category_name, category_path)
        summary_stats.append(stats)
    
    # 3. 输出最终汇总表（清晰展示每个类别的数据）
    print("\n" + "="*80)
    print("所有类别采样完成！最终汇总统计：")
    print("="*80)
    # 打印表头
    print(f"{'类别名称':<15} {'文件总数':<8} {'成功采样':<8} {'采样失败':<8} {'已存在文件':<8}")
    print("-"*80)
    # 打印每个类别数据
    total_all = 0
    total_success = 0
    total_failed = 0
    for stats in summary_stats:
        print(f"{stats['category']:<15} {stats['total']:<8} {stats['success']:<8} {stats['failed']:<8} {stats['existing']:<8}")
        total_all += stats['total']
        total_success += stats['success']
        total_failed += stats['failed']
    # 打印总计
    print("-"*80)
    print(f"{'总计':<15} {total_all:<8} {total_success:<8} {total_failed:<8} -")
    print("="*80)
    print(f"\n所有点云文件保存路径：{PC_OUTPUT_ROOT}")