@echo off
echo ============================================
echo  MXDiscovery - Installation Script
echo ============================================
echo.

REM Step 1: Activate venv
echo [Step 1/4] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: venv activation failed. Run: python -m venv venv
    exit /b 1
)
echo   OK - venv activated
echo.

REM Step 2: Upgrade pip
echo [Step 2/4] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Step 3: PyTorch with CUDA
echo [Step 3/4] Installing PyTorch with CUDA 12.1...
echo   (This downloads ~2.5 GB, may take 10-20 minutes)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo WARNING: CUDA install failed. Trying CPU-only PyTorch...
    pip install torch torchvision
)
echo.

REM Step 4: All other dependencies
echo [Step 4/4] Installing project dependencies...
pip install numpy scipy pandas matplotlib scikit-learn
pip install ase pymatgen mp-api
pip install chgnet alignn
pip install ollama chromadb sentence-transformers
pip install requests beautifulsoup4 pdfplumber
pip install sqlalchemy
pip install py3Dmol plotly streamlit
pip install pyyaml tqdm loguru pydantic
echo.

echo ============================================
echo  Verifying installation...
echo ============================================
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import ase; print(f'ASE: {ase.__version__}')"
python -c "import chgnet; print(f'CHGNet: OK')"
python -c "import ollama; print(f'Ollama client: OK')"
python -c "import pymatgen; print(f'Pymatgen: OK')"
echo.

echo ============================================
echo  Installation complete!
echo ============================================
echo.
echo NEXT STEPS:
echo   1. Install Ollama app from https://ollama.com
echo   2. Run: ollama pull qwen2.5:14b
echo   3. Run: ollama pull nomic-embed-text
echo   4. Test: python -m src.pipeline fetch
echo.
echo NOTE: BoltzTraP2 must be installed in WSL2 (needs Fortran).
echo   This is only needed for Stage 6 (DFT). All other stages work now.
echo ============================================
