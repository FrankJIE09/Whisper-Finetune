import os
import math

def convert_size(size_bytes):
    """将字节大小转换为可读格式"""
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

def check_large_files(directory, size_limit_mb=100):
    """遍历目录，打印超过指定大小的文件路径"""
    size_limit_bytes = size_limit_mb * 1024 * 1024
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                file_size = os.path.getsize(file_path)
                if file_size > size_limit_bytes:
                    print(f"{file_path} - {convert_size(file_size)}")
            except Exception as e:
                print(f"Could not access {file_path}: {e}")

if __name__ == "__main__":
    directory = '.'  # 指定要遍历的目录
    check_large_files(directory, size_limit_mb=100)
