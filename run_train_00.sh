#!/usr/bin/env bash
DATA=cub
DATA_ROOT=data
Gallery_eq_Query=True
LOSS=margin_loss
DRO=DRO_TOPK
CHECKPOINTS=ckps
R=.pth.tar

if_exist_mkdir ()
{
    dirname=$1
    if [ ! -d "$dirname" ]; then
    mkdir $dirname
    fi
}

if_exist_mkdir ${CHECKPOINTS}
if_exist_mkdir ${CHECKPOINTS}/${LOSS}
if_exist_mkdir ${CHECKPOINTS}/${LOSS}/${DRO}
if_exist_mkdir ${CHECKPOINTS}/${LOSS}/${DRO}/${DATA}

if_exist_mkdir result
if_exist_mkdir result/${LOSS}
if_exist_mkdir result/${LOSS}/${DRO}
if_exist_mkdir result/${LOSS}/${DRO}/${DATA}

NET=BN-Inception
DIM=512
ALPHA=0.2
MARGIN=0.5
LR=1e-5
BatchSize=80
RATIO=0.16
K_LIST=(80 100 120 140)
SELECT_TOPK_ALL=1


for((k=0; k<4; k++));do
{
# if [ ! -n "$1" ] ;then
echo "Begin Training!"
SAVE_DIR=${CHECKPOINTS}/${LOSS}/${DRO}/${DATA}/${NET}-DRO-${DRO}-K-${K_LIST[$k]}-DIM-${DIM}-lr-${LR}-ratio-${RATIO}-BatchSize-${BatchSize}-batch-${SELECT_TOPK_ALL}
if_exist_mkdir ${SAVE_DIR}
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py --net ${NET} \
--data $DATA \
--data_root ${DATA_ROOT} \
--init random \
--lr ${LR} \
--dim $DIM \
--alpha $ALPHA \
--num_instances   5 \
--batch_size ${BatchSize} \
--epoch 1200 \
--loss $LOSS \
--DRO $DRO \
--width 227 \
--margin $MARGIN \
--save_dir ${SAVE_DIR} \
--save_step 50 \
--ratio ${RATIO} \
--K ${K_LIST[$k]} \
--select_TOPK_all ${SELECT_TOPK_ALL}

echo "Begin Testing!"
Model_LIST=`seq  1 50 1200`
for i in $Model_LIST; do
    CUDA_VISIBLE_DEVICES=0 python3 test.py --net ${NET} \
    --data $DATA \
    --data_root ${DATA_ROOT} \
    --batch_size 100 \
    -g_eq_q ${Gallery_eq_Query} \
    --width 227 \
    -r ${SAVE_DIR}/ckp_ep$i$R \
    --pool_feature ${POOL_FEATURE:-'False'} \
    | tee -a result/${LOSS}/${DRO}/${DATA}/${NET}-DIM-$DIM-K-${K_LIST[$k]}-Batchsize-${BatchSize}-ratio-${RATIO}-lr-$LR${POOL_FEATURE:+'-pool_feature'}.txt
done
}&
done

