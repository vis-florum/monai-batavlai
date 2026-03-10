# Activate the environment first!

# LPD variants
source ./lpdvariants.sh

chooser=$1  # first argument

if [ "$chooser" -eq 0 ]; then
    variant="LPDsample-CT"
    SERVER_DIR="/media/Store-SSD/Stembank/pine/LPDsample"
elif [ "$chooser" -gt 0 ] && [ "$chooser" -le ${#lpdvariants[@]} ]; then
    ((i = chooser - 1))
    variant=${lpdvariants[$i]}
    SERVER_DIR="$LPD_DIR/$variant"
else
    echo "Invalid argument"
    exit 1
fi

echo "Inferring for variant: $variant"


### Directories and files
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"   # this script's directory
MONAI_DIR=$SCRIPT_DIR
TRAIN_DIR="$SERVER_DIR/train"
VAL_DIR="$SERVER_DIR/val"
TEST_DIR="$SERVER_DIR/test"

infername="UNET_$variant"
TEST_LABEL_DIR="$TEST_DIR/labels/$infername"
VAL_LABEL_DIR="$VAL_DIR/labels/$infername"

APP_MAINFILE="$MONAI_DIR/main.py"
SCORE_FILE="$MONAI_DIR/scoring.py"


## Batch inference
echo "Running inference on $TEST_DIR and saving under labels/$infername ..."
python $APP_MAINFILE --studies $TEST_DIR --model segmentation --test batch_infer --variant $variant --inflabel $infername
echo -e "\n\n\n"

echo "Running inference on $VAL_DIR and saving under labels/$infername ..."
python $APP_MAINFILE --studies $VAL_DIR --model segmentation --test batch_infer --variant $variant --inflabel $infername
echo -e "\n\n\n"

## Scoring
# echo "Running scoring on $TEST_DIR ..."
# python $SCORE_FILE $TEST_LABEL_DIR
# echo -e "\n\n\n"

# echo "Running scoring on $VAL_DIR ..."
# python $SCORE_FILE $VAL_LABEL_DIR