import os
import subprocess
import sys

def run_script(script_path):
    print("\n" + "#"*70)
    print(f"  执行脚本: {script_path}")
    print("#"*70)
    result = subprocess.run([sys.executable, script_path], capture_output=False)
    if result.returncode != 0:
        print(f"[错误] 脚本 {script_path} 执行失败!")
    else:
        print(f"[成功] 脚本 {script_path} 执行成功!")

def main():
    print("="*70)
    print("        OpenCV 3 从基础到进阶实战演示代码一键执行程序")
    print("="*70)

    # 1. 首先确保测试图片准备完毕
    run_script("demos/generate_test_images.py")

    # 2. 依次运行各个章节演示
    demo_scripts = [
        "demos/01_basic_operations.py",
        "demos/02_image_processing.py",
        "demos/03_feature_edge_contour.py",
        "demos/04_advanced_applications.py"
    ]

    for script in demo_scripts:
        run_script(script)

    print("\n" + "="*70)
    print("  所有演示脚本运行完毕！生成的演示图片均已保存在 'output_images/' 目录。")
    print("="*70)

if __name__ == "__main__":
    main()
