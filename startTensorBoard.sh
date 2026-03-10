LOGDIR="/home/aime/monai/apps/bata-unet-4steps/model_3D_UNET_reconstructions_5_src_pos/segmentation"


echo "Starting TensorBoard on directory: $LOGDIR ..."
tensorboard --logdir=$LOGDIR --port=6006

echo "TensorBoard is running on directory: $LOGDIR"
echo "Now start a browser on the host machine and open the following URL:"
echo "http://localhost:6006/"
echo ""