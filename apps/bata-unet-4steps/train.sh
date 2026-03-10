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

echo "Training on variant: $variant"

TRAIN_DIR=$SERVER_DIR/train
VAL_DIR=$SERVER_DIR/val

python main.py --studies $TRAIN_DIR --model segmentation --test train --variant $variant --validation $VAL_DIR

