from setuptools import setup, find_packages

setup(
    name="speech-to-text-finetune-indic",
    version="0.1.0",
    description="Efficient multilingual Indic ASR with Whisper + LoRA",
    author="Ramnarayan Choudhary",
    author_email="choudharyramnarayan123@gmail.com",
    url="https://github.com/Ramnarayan-Choudhary/Speech_agent",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "transformers>=4.30.0",
        "peft>=0.4.0",
        "datasets[audio]>=2.13.0",
        "evaluate>=0.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
        "demo": [
            "gradio>=4.0.0",
            "streamlit>=1.24.0",
        ],
    },
)
