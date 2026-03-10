# Start Server in training directory for starting training from Slicer
#SERVER_DIR="/media/Store-SSD/Stembank/pine/LPDsample/train"
SERVER_DIR="/media/Store-SSD/Stembank/pine/LPDsample-nrrd/train"

variant="LPDsample-CT"

monailabel start_server --app . --studies $SERVER_DIR --conf models segmentation --conf preload true --conf variant $variant
