import os
import platform

def install_torch():
    system = platform.system()
    if system == "Darwin":  # macOS
        print("Installing PyTorch for macOS...")
        os.system("pip install torch torchvision torchaudio")
    elif system in ["Linux", "Windows"]:
        print("Installing PyTorch with CUDA for Linux/Windows...")
        os.system(
            "pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118 "
            "--index-url https://download.pytorch.org/whl/cu118"
        )
    else:
        raise Exception(f"Unsupported platform: {system}")

def install_requirements():
    print("Installing other dependencies from requirements.txt...")
    os.system("pip install -r requirements.txt")

if __name__ == "__main__":
    install_torch()
    install_requirements()
