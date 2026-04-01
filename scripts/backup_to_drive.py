"""Backup checkpoints to Google Drive (for Colab)."""

import shutil
from pathlib import Path
from datetime import datetime
from loguru import logger


def backup_to_drive(checkpoint_dir: str, drive_backup_dir: str):
    """Backup training checkpoints to Google Drive."""
    
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint directory {checkpoint_dir} not found")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = Path(drive_backup_dir) / f"checkpoint_backup_{timestamp}"
    
    logger.info(f"Backing up {checkpoint_dir} to {backup_path}")
    
    shutil.copytree(checkpoint_path, backup_path, dirs_exist_ok=True)
    
    logger.info(f"✅ Backup complete!")
    return str(backup_path)


if __name__ == "__main__":
    # Example for Colab
    backup_to_drive(
        "/content/artifacts/marathi_lora",
        "/content/drive/MyDrive/STT-Indic-Project/models/checkpoints"
    )
