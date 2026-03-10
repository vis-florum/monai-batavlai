
# LPD variants
source ./lpdvariants.sh

for i in {0..3}
do
    variant=${lpdvariants[$i]}
    echo "Cross-inferring with CT-trained UNET on LPD variant: $variant"

    ### Directories and files
    SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"   # this script's directory
    MONAI_DIR=$SCRIPT_DIR

    SERVER_DIR="$LPD_DIR/$variant"
    TRAIN_DIR="$SERVER_DIR/train"
    VAL_DIR="$SERVER_DIR/val"
    TEST_DIR="$SERVER_DIR/test"

    unetCT="LPDsample-CT"   # always use the CT-trained UNET
    infername="UNET_$unetCT"
    TEST_LABEL_DIR="$TEST_DIR/labels/$infername"
    VAL_LABEL_DIR="$VAL_DIR/labels/$infername"

    APP_MAINFILE="$MONAI_DIR/main.py"
    SCORE_FILE="$MONAI_DIR/scoring.py"


    ## Batch inference
    echo "Running inference on $TEST_DIR and saving under labels/$infername ..."
    #python $APP_MAINFILE --studies $TEST_DIR --model segmentation --test batch_infer --variant $unetCT --inflabel $infername
    echo -e "\n\n\n"

    echo "Running inference on $VAL_DIR and saving under labels/$infername ..."
    #python $APP_MAINFILE --studies $VAL_DIR --model segmentation --test batch_infer --variant $unetCT --inflabel $infername
    echo -e "\n\n\n"

    ## Scoring
    echo "Running scoring on $TEST_DIR ..."
    python $SCORE_FILE $TEST_LABEL_DIR
    echo -e "\n\n\n"

    echo "Running scoring on $VAL_DIR ..."
    python $SCORE_FILE $VAL_LABEL_DIR
    echo -e "\n\n\n"
done