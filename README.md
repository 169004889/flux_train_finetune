# flux_train_finetune
Automatic label generation and title generation of image data with clean background, training of fine-tuning model.
Original project link, cited by https://zhuanlan.zhihu.com/p/684068402.
Only for learning and communication, not for business.

# 1.Create environment
there must be an environment, which can be the default, but it is best to create a new virtual environment. If it is a conda environment, you can use "conda create-name flux python = 3.10". Or use "python-m venv flux" under Python version 3.10.

# 2.Download dependence
you can use the mirror source, you can also download directly from the official.Make sure it is a GPU version, and the versions of PyTroch, CUDA and cuDNN are compatible.
"pip install torch==2.4.0 torchvision==0.19.0 xformers==0.0.27.post2 -i https://pypi.tuna.tsinghua.edu.cn/simple some-package"
"pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple some-package"

# 3.Set up the training environment
You also need to set the training environment parameters of FLUX.1 model, mainly using the ability of accelerate library, which can make the training and reasoning of PyTorch more efficient and concise. Enter the following commands at the command line, and fill in each setting one by one to complete the optimal configuration. The environment configuration process of single-machine single-card training and single-machine multi-card training is as follows.

(1)Configuration of single machine and single card training;
accelerate config

In which compute environment are you running? 
Please select a choice using the arrow or number keys, and selecting with enter
 ➔  This machine
    AWS (Amazon SageMaker)
                                                                                                       
Which type of machine are you using? Please select a choice using the arrow or number keys, and selecting with enter
 ➔  No distributed training
    multi-CPU
    multi-XPU
    multi-GPU
    multi-NPU
    multi-MLU
    multi-MUSA
    TPU

Do you want to run your training on CPU only (even if a GPU / Apple Silicon / Ascend NPU device is available)? [yes/NO]:NO

Do you wish to optimize your script with torch dynamo?[yes/NO]: # Press Enter.                                                                                                                                                         
Do you want to use DeepSpeed? [yes/NO]: # Press Enter.

What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]:all  # Choose which GPU to use for training. If you only have one GPU, simply enter "all". If you have multiple GPUs, say 8 GPUs, you can enter a number from 0 to 7 to specify the particular GPU to use for training.  
               
Would you like to enable numa efficiency? (Currently only supported on NVIDIA hardware). [yes/NO]:  # Press Enter.

Do you wish to use FP16 or BF16 (mixed precision)?                                                                                                        
Please select a choice using the arrow or number keys, and selecting with enter                                           
    no                                                                                                                                                    
➔  fp16
    bf16
    fp8                    

accelerate configuration saved at /root/.cache/huggingface/accelerate/default_config.yaml

(2)Configuration of single machine multi-card training;
accelerate config

In which compute environment are you running? 
Please select a choice using the arrow or number keys, and selecting with enter
 ➔  This machine
    AWS (Amazon SageMaker)
                                                                                                       
Which type of machine are you using?                                                                                                                      
Please select a choice using the arrow or number keys, and selecting with enter
    No distributed training                                                                                                                               
    multi-CPU                        
    multi-XPU                         
 ➔  multi-GPU
    multi-NPU
    multi-MLU
    multi-MUSA
    TPU

How many different machines will you use (use more than 1 for multi-node training)? [1]: 1

Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]: # Press Enter.                                                                                               
Do you wish to optimize your script with torch dynamo?[yes/NO]: # Press Enter.                                                                                                                                                              
Do you want to use DeepSpeed? [yes/NO]: # Press Enter.                                                                                                                                                                                 
Do you want to use FullyShardedDataParallel? [yes/NO]: # Press Enter.                                                                                                                                                                     
Do you want to use Megatron-LM ? [yes/NO]: # Press Enter.
 

How many GPU(s) should be used for distributed training? [1]:8   # How many GPUs to allocate for training? Assuming we have 8 GPUs, you can enter a number between 1 and 8.

Set the ID of the GPU card(s) used for training. If all GPUs are to be used for training, enter "all".

What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]:all # If only some GPUs are to be used for training, you need to enter the specific GPU card IDs. For example, if we have 8 cards and want to use 2 of them for training, you can enter 0,1 or 3,7, etc.
                     
Would you like to enable numa efficiency? (Currently only supported on NVIDIA hardware). [yes/NO]:  # Press Enter.

Do you wish to use FP16 or BF16 (mixed precision)?                                                                                                        
Please select a choice using the arrow or number keys, and selecting with enter                                                                           
    no                        
 ➔  fp16
    bf16
    fp8                    
                                                                             
accelerate configuration saved at /root/.cache/huggingface/accelerate/default_config.yaml

# 4.Set the text encoder
When training the FLUX.1 model, the Text Encoder model of FLUX.1 will call two configuration files, namely clip-vit-large-patch14 and t5-v1_1-xxl.
Under normal circumstances, FLUX.1 model will download configuration files from huggingface to ~/.cache/huggingface/ directory, but the download will probably fail due to network reasons, which will lead to the failure of training.
Therefore, three configuration files, namely clip-vit-large-patch14, Clip-Vit-Bigg-14-Laion2B-39b-B160K and t5-v1_1-xxl, have been put into the utils_json folder of the flux_finetune project, and the dependency paths have been configured for everyone. Just use Flux _ Fine. If you want to modify the call paths of the dependent folders clip-vit-large-patch14, Clip-Vit-Bigg-14-Laion2B-39b-B160K and t5-v1_1-xxl, the corresponding part in the library/strategy_flux.py script will be modified into its own local customized path, such as "/local path/utils _".

CLIP_L_TOKENIZER_ID = "./utils_json/clip-vit-large-patch14" # Lines 20-21 of the strategy_flux.py script
T5_XXL_TOKENIZER_ID = "./utils_json/t5-v1_1-xxl"

# 5.Making FLUX.1 Model Training Data Set
Data annotation can be divided into automatic annotation and manual annotation. Automatic labeling mainly depends on models such as BLIP and WaifuDiffusion Tagger, while manual labeling depends on labeling personnel.
Judging from the annotation content, the annotation content in AI painting field can be mainly divided into two tasks: Img2Caption and Img2Tag.
(1) using BLIP to automatically generate caption tags (natural language tags) of data.
Automatically label data sets with BLIP, and the output of BLIP is natural language label. Enter the path of flux_finetune/finetune/ and run the following code to get natural language label (caption label):
<img width="866" height="39" alt="image" src="https://github.com/user-attachments/assets/73e3920f-2132-4051-8c56-daf158c0f2f3" />
cd flux_finetune/finetune/
python make_captions.py "/Replace with dataset path" --caption_weights "../BLIP/model_large_caption.pth" --batch_size=1 --beam_search --min_length=5 --max_length=100 --debug --caption_extension=".caption" --max_data_loader_n_workers=2 --recursive
(2) Using WaifuDiffusion Tagger model to automatically generate data tag (permutation and combination of words)
Calling WaifuDiffusion Tagger model needs to install a specific version (2.10.0) of Tensorflow library, otherwise the runtime will report "DNN library is not found" error. Enter the following command at the command line to complete the version check and installation adaptation of Tensorflow library:

pip show tensorflow
Name: tensorflow
Version: 2.10.0
Summary: TensorFlow is an open source machine learning framework for everyone.

pip install tensorflow==2.10.0 -i https://pypi.tuna.tsinghua.edu.cn/simple some-package # If the Tenosrflow library is not installed or the version is wrong, you can re-install it by entering the following command.

Next, the training data can be automatically labeled using WaifuDiffusion Tagger model, and the output of WaifuDiffusion Tagger model is tag keyword tags, which are composed of keyword phrases:
<img width="1059" height="89" alt="image" src="https://github.com/user-attachments/assets/0e4bccfa-7d28-4353-b7d7-317afe8b1145" />
cd flux_finetune/finetune/
python tag_images_by_wd14_tagger.py "/Data path" --batch_size=8 --model_dir "../tag_models" --repo_id "wd-v1-4-moat-tagger-v2" --remove_underscore --general_threshold=0.35 --character_threshold=0.35 --caption_extension=".txt" --max_data_loader_n_workers=2 --debug --undesired_tags=""
(3) Supplementary custom special labels
After completing the automatic annotation of caption and tag, if we need to train some custom special annotations, we can also supplement the annotation of the data.
You can directly open the file flux_finetune/custom tag.ipynb, modify the parameters according to the comments provided in the code, and then run the code to supplement the annotation of the data set. If you think the code is too complicated, you just need to set the train_data_dir ="/ local data set path "and custom_tag ="WeThinkIn" as the local path of your own data set and the special tag you want to add, and then run the code, which is very simple and practical.
Generally, the special tags added manually will be put in the first place, because the first tag in a sentence has the largest weight, and the later tags have the smaller weight.
(4) Training data preprocessing
Making meta_data.json of training data: integrating the annotation files with suffixes of. caption and. txt just generated and storing them into a json format file, so as to facilitate the subsequent training of FLUX.1 model to retrieve training data and annotations according to the information in json.
Make meta_data.json file of training data:
<img width="1734" height="180" alt="image" src="https://github.com/user-attachments/assets/708a2b56-1913-4c13-8c2a-fbf6b2c8856e" />
cd flux_finetune
python ./finetune/merge_all_to_metadata.py "/Local data path ""/local data path/meta_data.json"

# 6.Fine-tuning training of FLUX.1 model
Fine-tuning training has two main goals:
(1)Enhance the image generation ability of FLUX.1 model.
(2)Increase the ability of FLUX.1 model to trigger and respond to the new prompt.
The parameter configuration and training script of FLUX.1 fine-tuning training include FLUX.1-dev and FLUX.1-schnell versions.

Find the corresponding training data parameter configuration file (data_config.toml) in the data _ config folder of the flux_finetune project, and find the flux_finetune.sh script in the flux _ finetune project, which contains the core training parameters.

[general] # define common settings here
flip_aug = false
color_aug = false
random_crop = false
shuffle_caption = false
caption_tag_dropout_rate = 0
caption_extension = ".txt"
keep_tokens = 1
keep_tokens_separator= "|||"

[[datasets]] # define the first resolution here
batch_size = 1
enable_bucket = true
resolution = [1024, 1024]
min_bucket_reso = 256
max_bucket_reso = 1024 
bucket_reso_steps = 64   
bucket_no_upscale = false

  [[datasets.subsets]]
  image_dir = "/Local path"
  num_repeats = 10

Finally, use the sh FLUX_finetune.sh command to start the training process of FLUX.1 model, and you can train your own FLUX.1 model:
accelerate launch  
  --num_cpu_threads_per_process 8 flux_train.py \  
  --pretrained_model_name_or_path "/Local path/flux1-dev.safetensors" \
  --clip_l "/Local path/clip_l.safetensors" \
  --t5xxl "/Local path/t5xxl_fp16.safetensors" \
  --ae "/Local path/ae.safetensors" \
  --save_model_as safetensors \
  --sdpa \
  --persistent_data_loader_workers \
  --max_data_loader_n_workers 8 \
  --seed 1024 \
  --gradient_checkpointing \
  --mixed_precision bf16 \
  --save_precision bf16 \
  --dataset_config "/Local path/data_config.toml" \
  --output_dir "/Local path/Model save address" \
  --output_name "FLUX_model" \
  --learning_rate 5e-5 \
  --max_train_epochs 10  \
  --sdpa \
  --highvram \
  --cache_text_encoder_outputs_to_disk \
  --cache_latents_to_disk \
  --save_every_n_epochs 1 \
  --optimizer_type adafactor \
  --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" \
  --lr_scheduler constant_with_warmup \
  --max_grad_norm 0.0 \
  --timestep_sampling shift \
  --discrete_flow_shift 3.1582 \
  --model_prediction_type raw \
  --guidance_scale 1.0 \
  --loss_type l2 \
  --fused_backward_pass \
  --blocks_to_swap 35 \
  --fp8_base 

# 7.Loading self-training FLUX.1 model for AI painting.
After the FLUX.1 model fine-tuning training is completed, the model weights will be saved in the output_dir path we set before. Next, you can use ComfyUI or webui as a framework to load FLUX.1 fine-tuning model for AI painting.
This part needs to use ComfyUI or webui framework. If I have time, I can supplement it later here, mainly to see if I can remember it.
In addition, there is a LoRA model based on FLUX.1 training, and I can supplement it later.


