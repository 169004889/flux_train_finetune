script="/本地路径/SD3-FLUX-Train"
train_data_dir="/本地路径/FLUX.1-宝可梦数据集"
BLIP="/本地路径/BLIP/model_large_caption.pth"

python $script/finetune/make_captions.py $train_data_dir --caption_weights $BLIP --batch_size=1 --beam_search --min_length=5 --max_length=100 --caption_extension=".caption" --max_data_loader_n_workers=2 --recursive