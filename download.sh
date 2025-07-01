# #!/bin/bash

# set -e  # Exit on error

# # 1. Check if Python tarball is present
# PYTHON_TARBALL="python-3.11.tar.xz"
# if [ ! -f "$PYTHON_TARBALL" ]; then
#     echo "[1] Python tarball not found. Downloading..."
#     wget https://www.python.org/ftp/python/3.11.0/$PYTHON_TARBALL
# else
#     echo "[1] Python tarball found."
# fi

# # 2. Install Python if not installed
# if ! command -v python3.11 &> /dev/null; then
#     echo "[2] Installing Python 3.11 from tarball..."
#     tar -xf $PYTHON_TARBALL
#     cd python-3.11.0
#     ./configure --enable-optimizations
#     make -j"$(nproc)"
#     sudo make altinstall
#     cd ..
# else
#     echo "[2] Python 3.11 is already installed."
# fi

# # 3. Create virtual environment if not exists
# if [ ! -d "venv" ]; then
#     echo "[3] Creating virtual environment..."
#     python3.11 -m venv venv
#     source venv/bin/activate
#     echo "[4] Installing requirements..."
#     pip install --upgrade pip
#     if [ -f "requirements.txt" ]; then
#         pip install -r requirements.txt
#     else
#         echo "[4] requirements.txt not found!"
#     fi
# else
#     echo "[3] Virtual environment already exists."
#     source venv/bin/activate
# fi

# # 4. Check if model file exists
# MODEL_PATH="model.pt"
# if [ ! -f "$MODEL_PATH" ]; then
#     echo "[5] Model not found. Downloading..."
#     # Replace with actual URL
#     wget -O $MODEL_PATH https://example.com/path/to/model.pt
# else
#     echo "[5] Model found."
# fi

# # 5. Run test check
# echo "[6] Running test check..."
# python -c "import torch; print('Torch version:', torch.__version__)" || echo "Torch not installed"

# # 6. Run main.py
# if [ -f "main.py" ]; then
#     echo "[7] Running main.py..."
#     python main.py
# else
#     echo "[7] main.py not found!"
# fi

