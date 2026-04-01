"""Setup Google Drive for Colab notebooks."""

from google.colab import drive
from pathlib import Path
import os


def setup_google_drive(mount_point="/content/drive"):
    """Mount Google Drive for Colab."""
    
    print("Mounting Google Drive...")
    drive.mount(mount_point)
    
    # Create project directories
    project_dir = Path(mount_point) / "MyDrive" / "STT-Indic-Project"
    project_dir.mkdir(parents=True, exist_ok=True)
    
    (project_dir / "datasets" / "raw").mkdir(parents=True, exist_ok=True)
    (project_dir / "datasets" / "processed").mkdir(parents=True, exist_ok=True)
    (project_dir / "models" / "checkpoints").mkdir(parents=True, exist_ok=True)
    (project_dir / "models" / "final").mkdir(parents=True, exist_ok=True)
    (project_dir / "logs").mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Google Drive mounted at {mount_point}")
    print(f"✅ Project directories created at {project_dir}")
    
    return str(project_dir)


if __name__ == "__main__":
    setup_google_drive()
