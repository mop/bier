TF_LOG_PATH=/media/nax/tensor-bier/products_new_release

mkdir -p $TF_LOG_PATH/fvecs

CUDA_VISIBLE_DEVICES="0" python -u eval_bier.py \
    --test-images /data/metric/bier/datasets/test_images_products.npy \
    --test-labels /data/metric/bier/datasets/test_labels_products.npy \
    --eval-every 5000 \
    --embedding-sizes '96,160,256' \
    --dump-only \
    --dump-prefix ${TF_LOG_PATH}/fvecs/stfd-adversarial-bier \
    --model ${TF_LOG_PATH}/stfd-adversarial-bier
