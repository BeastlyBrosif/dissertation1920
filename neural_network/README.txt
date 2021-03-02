Technologies used:
- Tensorflow2.0
- Python3.6.0 (64 bit version)
- Cuda 10.0
- cuDNN 10.0
- GTX 960 2GB
- virtualenv @ C:\Users\Nimrod\venv

To activate venv: 
"C:\Users\Nimrod\venv\Scripts\activate.ps1"
To deactivate venv:
"deactivate"

on linux:
source ~/venv/bin/activate

When installing tensorflow, we need to downgrade protobuf to 3.6.0 (for some reason)
pip install protobuf==3.6.0

scp psynw2@sproj16.nott.ac.uk:~/dissertation/dissertation/ ~/Documents/Year_4/Diss
