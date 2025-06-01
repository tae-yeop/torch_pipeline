conda create -n backend python=3.10 -y
conda activate backend
pip3 install torch torchvision torchaudio
pip install diffusers["torch"] transformers
pip install -r requirements.txt
