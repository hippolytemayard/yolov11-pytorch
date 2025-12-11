#!/usr/bin/env python3
"""
Download YOLO v11 weights from Google Drive.

⚠️ LICENSE NOTICE:
These weights are transferred from Ultralytics and are licensed under AGPL-3.0.
Commercial use requires an Ultralytics Enterprise License.
See: https://ultralytics.com/license

Usage:
    python scripts/download_weights.py                    # Download all weights
    python scripts/download_weights.py --model n s       # Download specific sizes
    python scripts/download_weights.py --task detect     # Download specific task
"""

import argparse
import os
import sys
from pathlib import Path


# =============================================================================
# GOOGLE DRIVE CONFIGURATION
# =============================================================================

# Folder containing all weights
GDRIVE_FOLDER_ID = "1GB-Cy7iEms2_0_xKwsoETSuovcmbX6bG"
GDRIVE_FOLDER_URL = f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}"

# Individual file IDs for each weight
WEIGHT_FILES = {
    # Detection
    "yolo11n.pt": "175qOxekoJwXiNhE16Bo2iQr7IJPk4H-E",
    "yolo11s.pt": "1SA5N8BJulqPkMKaKnxP6fqJpENqvioc4",
    "yolo11m.pt": "1NpDLZjFCqMY8iM5812kSmHB04iiOYFkG",
    "yolo11l.pt": "1xWeN__ST4zio6O3efMHO6qBz-oIyyifo",
    "yolo11x.pt": "1NBHmAiMypj-KGHoN5GiIJtPF9h5XDQzq",
    # Segmentation
    "yolo11n-seg.pt": "1mUCOkkuGwS5ZVmCTx04OhVvQY9f31A1v",
    "yolo11s-seg.pt": "1xsJdQV2-IwwEeBFcgfSFmizxQGgZSF1k",
    "yolo11m-seg.pt": "1_U7fgSdK8ydWiWB2MJbN4hxcAc1x78b7",
    "yolo11l-seg.pt": "14uOrXi_1kmnTn-rWUXaVI1w20UdvjDID",
    "yolo11x-seg.pt": "1USVdODYo7_5bgik74-20tAhFWCdWM9Xm",
    # Pose
    "yolo11n-pose.pt": "1pJDTaHdbiJO_uHLeIDpwJt9WFfASsai9",
    "yolo11s-pose.pt": "1q4_hWMm0wMfQX4UsfDku80_tUyVjS9Gg",
    "yolo11m-pose.pt": "1LkUAZsyN8H1UsB51AWMJwUrEqjVTxS_8",
    "yolo11l-pose.pt": "1rWar9FvvI4KtYnFdGZT8FWu2hoF_Yzc9",
    "yolo11x-pose.pt": "1kgcjABj-QUl9eJbtYnNDwcDGoq5kGTLO",
}

# Model categories
MODELS_BY_TASK = {
    "detect": ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"],
    "segment": ["yolo11n-seg", "yolo11s-seg", "yolo11m-seg", "yolo11l-seg", "yolo11x-seg"],
    "pose": ["yolo11n-pose", "yolo11s-pose", "yolo11m-pose", "yolo11l-pose", "yolo11x-pose"],
}

MODELS_BY_SIZE = {
    "n": ["yolo11n", "yolo11n-seg", "yolo11n-pose"],
    "s": ["yolo11s", "yolo11s-seg", "yolo11s-pose"],
    "m": ["yolo11m", "yolo11m-seg", "yolo11m-pose"],
    "l": ["yolo11l", "yolo11l-seg", "yolo11l-pose"],
    "x": ["yolo11x", "yolo11x-seg", "yolo11x-pose"],
}


def install_gdown():
    """Install gdown if not available."""
    try:
        import gdown
        return gdown
    except ImportError:
        print("📦 Installing gdown...")
        os.system(f"{sys.executable} -m pip install gdown -q")
        import gdown
        return gdown


def get_models_to_download(sizes: list = None, tasks: list = None) -> list:
    """Get list of models to download based on filters."""
    models = set()
    
    # If no filters, download all
    if not sizes and not tasks:
        return [m.replace(".pt", "") for m in WEIGHT_FILES.keys()]
    
    # Filter by size
    if sizes:
        for size in sizes:
            models.update(MODELS_BY_SIZE.get(size, []))
    
    # Filter by task
    if tasks:
        task_models = set()
        for task in tasks:
            task_models.update(MODELS_BY_TASK.get(task, []))
        
        if sizes:
            models = models.intersection(task_models)
        else:
            models = task_models
    
    return sorted(list(models))


def download_file(file_id: str, output_path: Path, filename: str) -> bool:
    """Download a single file from Google Drive."""
    gdown = install_gdown()
    
    url = f"https://drive.google.com/uc?id={file_id}"
    dest = output_path / filename
    
    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  ⏭️  {filename} already exists ({size_mb:.1f} MB)")
        return True
    
    try:
        gdown.download(url, str(dest), quiet=False)
        
        if dest.exists():
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"  ✅ {filename} ({size_mb:.1f} MB)")
            return True
        return False
        
    except Exception as e:
        print(f"  ❌ {filename}: {e}")
        return False


def download_models(output_dir: Path, models_filter: list = None):
    """Download weights from Google Drive."""
    
    print(f"\n📥 Downloading from Google Drive...")
    print(f"   Output: {output_dir}\n")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = []
    failed = []
    skipped = []
    
    for filename, file_id in WEIGHT_FILES.items():
        model_name = filename.replace(".pt", "")
        
        # Check if this model should be downloaded
        if models_filter and model_name not in models_filter:
            continue
        
        if download_file(file_id, output_dir, filename):
            downloaded.append(model_name)
        else:
            failed.append(model_name)
    
    return downloaded, failed


def list_models():
    """List available models."""
    print("\n📋 Available YOLO v11 Models")
    print("=" * 50)
    
    for task, models in MODELS_BY_TASK.items():
        print(f"\n{task.upper()}:")
        for model in models:
            print(f"  • {model}")
    
    print(f"\n📁 Google Drive: {GDRIVE_FOLDER_URL}")
    print("\n⚠️  License: AGPL-3.0 (Ultralytics)")
    print("   Commercial use requires Ultralytics Enterprise License")
    print("   https://ultralytics.com/license")


def main():
    parser = argparse.ArgumentParser(
        description="Download YOLO v11 Open Source weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_weights.py                    # Download all (15 models)
  python download_weights.py --model n s        # Download nano & small
  python download_weights.py --task detect      # Download detection only
  python download_weights.py --list             # List available models
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        nargs="+",
        choices=["n", "s", "m", "l", "x"],
        help="Model sizes to download (default: all)"
    )
    parser.add_argument(
        "--task", "-t",
        nargs="+",
        choices=["detect", "segment", "pose"],
        help="Tasks to download (default: all)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path(__file__).parent.parent / "weights",
        help="Output directory (default: weights/)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available models"
    )
    
    args = parser.parse_args()
    
    # List mode
    if args.list:
        list_models()
        return
    
    # Get models to download
    models = get_models_to_download(args.model, args.task)
    
    # Header
    print("\n" + "=" * 60)
    print("  🚀 YOLO v11 - Weight Downloader")
    print("=" * 60)
    
    # License notice
    print("\n⚠️  LICENSE NOTICE")
    print("-" * 60)
    print("These weights are transferred from Ultralytics.")
    print("They are licensed under AGPL-3.0.")
    print("")
    print("• ✅ Free for research & personal use")
    print("• ⚠️  Commercial use requires Ultralytics Enterprise License")
    print("     https://ultralytics.com/license")
    print("-" * 60)
    
    response = input("\nDo you accept these terms? [y/N]: ")
    if response.lower() != 'y':
        print("\n❌ Download cancelled.")
        return
    
    print(f"\n📦 Models to download: {len(models)}")
    for m in models:
        print(f"   • {m}")
    
    # Download
    downloaded, failed = download_models(args.output, models)
    
    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  ✅ Downloaded: {len(downloaded)}")
    if failed:
        print(f"  ❌ Failed: {len(failed)}")
    
    if downloaded:
        print(f"\n💡 Quick start:")
        print(f"   import torch")
        print(f"   from yolo import yolo_v11_n")
        print(f"   ")
        print(f"   model = yolo_v11_n(num_classes=80)")
        print(f"   weights = torch.load('weights/yolo11n.pt')")
        print(f"   model.load_state_dict(weights['state_dict'])")


if __name__ == "__main__":
    main()
