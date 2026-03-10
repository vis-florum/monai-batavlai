# ENVIRONMENT
# CUDA
for our machine with nvidia rtx a6000 - cuda 11.7 should be highest
obsolete? : make sure cuda 11.8 is installed
conclusion: we used cuda 12.2

## Create and activate the environment:
```bash
cd monai
python3.8 -m venv monaivenv
source monaivenv/bin/activate
```

## Install monai
```bash
python -m pip install --upgrade pip setuptools wheel
# Install latest stable version for pytorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu122

# Check if cuda enabled
python -c "import torch; print(torch.cuda.is_available())"

# Install latest milestone
pip install monailabel
```


# Download e.g. radiology app:
monailabel apps --download --name radiology --output apps

# Edit the models in:
apps/radiology/lib/configs/

# Check the readme in  apps/radiology

# start Segmentation server:   <--------------------
monailabel start_server --app apps/radiology --studies /media/Store-SSD/Stembank/pine/server --conf models segmentation --conf preload true

# start Deepedit module:
monailabel start_server --app apps/radiology --studies train-images/ --conf models deepedit

# Create SSL keys for https
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout uvicorn-selfsigned.key -out uvicorn-selfsigned.crt

# Serve via https:
monailabel start_server --app apps/radiology --studies /media/Store-SSD/Stembank/pine/server --conf models segmentation --ssl_keyfile uvicorn-selfsigned.key --ssl_certfile uvicorn-selfsigned.crt
# may cause problems fetching samples

# Samples can be in any folder, the uvicorn serving works inside LTU net at least via:
http://servername:8000



# However for basic production deployment, you might need to run Uvicorn independently. In such cases, you can following these simple steps.

## dryrun the MONAI Label CLI for pre-init and dump the env variables to .env or env.bat
monailabel start_server --app apps/radiology --studies datasets/Task09_Spleen/imagesTr --host 0.0.0.0 --port 8000 --dryrun

# Linux/Ubuntu
source .env
uvicorn monailabel.app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --log-config apps/radiology/logs/logging.json \
  --no-access-log
