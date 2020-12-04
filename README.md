# Conditional Variational Transformation for Diverse Machine Translation

```bash
DATA_PATH=
CHECKPOINT_DIR=
CUDA_VISIBLE_DEVICES=4,5,6,7 fairseq-train \
    $DATA_PATH \
    --source-lang en --target-lang de \
    --arch transformer_cvae_wmt_en_de --share-all-embeddings \
    --dropout 0.3 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy_with_kl_div --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.001 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --max-tokens 3584 --update-freq 16 \
    --max-update 30000 --chol-factor-cls DiagonalFactor \
    --alpha 1.0 --KL-lambda 1.0 \
    --save-dir $CHECKPOINT_DIR
```
