script="/本地路径/SD3-FLUX-Train"
train_data_dir="/本地路径/FLUX.1-宝可梦数据集"
model_dir="/本地路径"
repo_id="wd-v1-4-moat-tagger-v2"

# batch-size = 8
python $script/finetune/tag_images_by_wd14_tagger.py $train_data_dir --batch_size 8 --model_dir $model_dir --repo_id $repo_id --general_threshold 0.5 --character_threshold 0.5 --caption_extension ".txt"