import os
import subprocess
import sys

# ========== 核心配置：强制所有操作在D盘 + 配置要渲染的类别列表 ==========
# 根路径
ROOT_DIR = r"D:\1Ahaha\AA3d"
# Blender可执行文件路径（手动安装后的路径）
BLENDER_EXE = r"D:\1Ahaha\AA3d\Blender\blender.exe"
# BlenderProc临时目录
BLENDER_PROC_TEMP = os.path.join(ROOT_DIR, "blenderproc_temp")
# 渲染相关路径
TOOLKIT_SCRIPT = os.path.join(ROOT_DIR, "3D-Data-Processing-Toolkit-main", "render_single_glb.py")

# 【关键配置】在这里添加/删除要渲染的类别（支持任意多个）
CATEGORIES_TO_RENDER = [ "bench","birdhouse","bookshelf","bottle" ]  # 可扩展：比如加 "chair", "car" 等

# ========== 强制环境变量（无C盘写入） ==========
os.environ["BLENDER_PROC_TEMP_DIR"] = BLENDER_PROC_TEMP
os.environ["TMP"] = BLENDER_PROC_TEMP
os.environ["TEMP"] = BLENDER_PROC_TEMP
os.environ["TMPDIR"] = BLENDER_PROC_TEMP
os.environ["BLENDER_EXECUTABLE"] = BLENDER_EXE  # 强制指定Blender路径

# ========== 验证Blender是否安装成功 ==========
def check_blender():
    if not os.path.exists(BLENDER_EXE):
        print(f" 未找到Blender：{BLENDER_EXE}")
        print(" 请先手动安装Blender 3.6.5到上述路径，下载地址：")
        print("   https://download.blender.org/release/Blender3.6/blender-3.6.5-windows-x64.msi")
        sys.exit(1)
    # 验证Blender能否运行
    try:
        result = subprocess.run(
            [BLENDER_EXE, "--version"],
            capture_output=True,
            encoding="utf-8",
            timeout=10
        )
        if "3.6.5" in result.stdout:
            print(f" Blender验证成功：{BLENDER_EXE}")
            return True
        else:
            print(f" Blender版本错误，需要3.6.5，当前版本：{result.stdout[:50]}")
            sys.exit(1)
    except Exception as e:
        print(f" Blender运行失败：{str(e)}")
        sys.exit(1)

# ========== 单个类别渲染函数（核心逻辑） ==========
def render_category(category):
    """
    渲染单个类别的所有GLB模型
    :param category: 类别名称（如bag、ashcan、airplane）
    """
    # 构建当前类别的路径
    category_dir = os.path.join(ROOT_DIR, "ShapeNet", category)
    output_root = os.path.join(ROOT_DIR, "output_224", f"{category}_multi_view")
    fail_log = os.path.join(output_root, f"{category}_render_fail.txt")
    
    # 检查类别目录是否存在
    if not os.path.exists(category_dir):
        print(f"\n 类别 {category} 的目录不存在：{category_dir}")
        return False
    
    # 创建必要目录
    for dir_path in [BLENDER_PROC_TEMP, output_root]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 清空旧的失败日志
    with open(fail_log, "w", encoding="utf-8") as f:
        f.write(f"{category.capitalize()}类别渲染失败模型列表：\n")
    
    # 获取所有GLB模型
    try:
        glb_files = [f for f in os.listdir(category_dir) if f.endswith(".glb")]
    except Exception as e:
        print(f"\n 读取类别 {category} 模型失败：{str(e)}")
        return False
    
    total_models = len(glb_files)
    if total_models == 0:
        print(f"\n 类别 {category} 下无GLB模型，跳过渲染")
        return True
    
    success_count = 0
    fail_count = 0
    
    print(f"\n{'='*60}")
    print(f"开始渲染 {category} 类别，共 {total_models} 个模型")
    print(f" 模型目录：{category_dir}")
    print(f" 输出目录：{output_root}")
    print(f"{'='*60}")
    
    # 批量渲染当前类别
    for idx, glb_file in enumerate(glb_files):
        model_path = os.path.join(category_dir, glb_file)
        model_name = os.path.splitext(glb_file)[0]
        out_dir = os.path.join(output_root, model_name)
        
        # 跳过已渲染的模型（检查是否有12张rgb图片）
        if os.path.exists(out_dir):
            rgb_files = [f for f in os.listdir(out_dir)
                         if f.startswith("rgb_") and f.endswith(".png")]
            cam_param_file = os.path.join(out_dir, "camera_params.npz")

            if len(rgb_files) >= 12 and os.path.exists(cam_param_file):
                print(f"[{idx+1}/{total_models}] 已渲染（含相机参数），跳过：{model_name}")
                success_count += 1
                continue

        
        # 渲染命令
        render_cmd = [
            "blenderproc", "run",
            TOOLKIT_SCRIPT,
            "--object_path", model_path,
            "--output_dir", out_dir,
            "--engine", "BLENDER_EEVEE_NEXT",
            "--num_images", "12",
            "--camera_dist", "1.2",
            "--resolution_x", "224",
            "--resolution_y", "224"
        ]
        
        try:
            print(f"[{idx+1}/{total_models}] 正在渲染：{model_name}")
            # 执行渲染
            result = subprocess.run(
                render_cmd,
                # shell=True,
                encoding="utf-8",
                timeout=300,  # 单模型渲染超时5分钟
                env=os.environ.copy()  # 传递所有D盘环境变量
            )
            
            # 检查渲染结果
            if result.returncode == 0:
                # 验证是否生成12张图片（防止空渲染）
                rgb_files = [f for f in os.listdir(out_dir) if f.startswith("rgb_") and f.endswith(".png")] if os.path.exists(out_dir) else []
                if len(rgb_files) >= 12:
                    print(f"[{idx+1}/{total_models}]  渲染成功：{model_name}")
                    success_count += 1
                else:
                    fail_info = f"{model_path} | 渲染完成但图片数量不足（仅{len(rgb_files)}张）\n"
                    with open(fail_log, "a", encoding="utf-8") as f:
                        f.write(fail_info)
                    print(f"[{idx+1}/{total_models}]  渲染失败：{model_name}（图片数量不足）")
                    fail_count += 1
            else:
                fail_info = f"{model_path} | 返回码：{result.returncode}\n"
                with open(fail_log, "a", encoding="utf-8") as f:
                    f.write(fail_info)
                print(f"[{idx+1}/{total_models}]  渲染失败：{model_name}（返回码非0）")
                fail_count += 1
        
        except subprocess.TimeoutExpired:
            fail_info = f"{model_path} | 错误：渲染超时（5分钟）\n"
            with open(fail_log, "a", encoding="utf-8") as f:
                f.write(fail_info)
            print(f"[{idx+1}/{total_models}]  渲染失败：{model_name}（超时）")
            fail_count += 1
        except Exception as e:
            fail_info = f"{model_path} | 未知错误：{str(e)}\n"
            with open(fail_log, "a", encoding="utf-8") as f:
                f.write(fail_info)
            print(f"[{idx+1}/{total_models}]  渲染失败：{model_name}（{str(e)}）")
            fail_count += 1
    
    # 输出当前类别渲染统计
    print(f"\n{'='*60}")
    print(f"{category} 类别渲染完成统计")
    print(f"总模型数：{total_models}")
    print(f" 成功：{success_count}（{success_count/total_models*100:.2f}%）")
    print(f" 失败：{fail_count}（{fail_count/total_models*100:.2f}%）")
    print(f" 输出目录：{output_root}")
    print(f" 失败日志：{fail_log if fail_count > 0 else '无'}")
    print(f"{'='*60}")
    
    return True

# ========== 主程序：依次渲染所有指定类别 ==========
if __name__ == "__main__":
    # 先验证Blender
    check_blender()
    
    # 统计全局渲染结果
    global_success = 0
    global_fail = 0
    total_categories = len(CATEGORIES_TO_RENDER)
    completed_categories = 0
    
    print(f"\n 开始批量渲染 {total_categories} 个类别：{', '.join(CATEGORIES_TO_RENDER)}")
    
    # 依次渲染每个类别
    for category in CATEGORIES_TO_RENDER:
        print(f"\n{'='*80}")
        print(f"开始处理第 {completed_categories+1}/{total_categories} 个类别：{category}")
        print(f"{'='*80}")
        
        # 渲染当前类别
        render_result = render_category(category)
        if render_result:
            completed_categories += 1
    
    # 输出全局统计
    print(f"\n{'='*80}")
    print(f" 所有类别批量渲染完成！")
    print(f"总类别数：{total_categories}")
    print(f"已完成类别数：{completed_categories}")
    print(f"未完成类别数：{total_categories - completed_categories}")
    print(f" 每个类别的详细统计已输出到对应输出目录的失败日志中")
    print(f"{'='*80}")