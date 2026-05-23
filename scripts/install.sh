# install cuda-toolkit to conda environment
conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit -y

# installing pytorch
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# installing transformers + accelerate + peft
pip install transformers accelerate peft deepspeed
pip install flash-attn==2.7.2.post1 --no-build-isolation

# install other packages
pip install rich numpy pandas nest-asyncio scikit-learn wandb