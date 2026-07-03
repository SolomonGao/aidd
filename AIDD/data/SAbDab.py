import pandas as pd
import requests
import zipfile
from pathlib import Path
import re

# ==================== 1. SAbDab 数据下载 ====================

SABDAB_SUMMARY_URL = "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/summary/all/"
SABDAB_ARCHIVE_URL = "https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/all/"

def download_sabdab_data(output_dir="sabdab_data"):
    """下载 SAbDab 摘要和结构文件"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # 下载摘要
    print("Downloading summary...")
    r = requests.get(SABDAB_SUMMARY_URL)
    summary_path = f"{output_dir}/sabdab_summary.tsv"
    with open(summary_path, "wb") as f:
        f.write(r.content)
    
    # 下载结构压缩包 (约几百MB，可选)
    print("Downloading structures archive...")
    r = requests.get(SABDAB_ARCHIVE_URL)
    archive_path = f"{output_dir}/all_structures.zip"
    with open(archive_path, "wb") as f:
        f.write(r.content)
    
    # 解压
    with zipfile.ZipFile(archive_path, 'r') as z:
        z.extractall(output_dir)
    
    print(f"Data saved to {output_dir}/")
    print(f"  - Summary: {summary_path}")
    print(f"  - Structures: {output_dir}/all_structures/imgt/")
    return summary_path

if __name__ == "__main__":
    download_sabdab_data()