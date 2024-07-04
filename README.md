# RVC + Audio separator project
---
## Features
- All you need is a model checkpoint and a song file to use as input, and the VOCAL and MR separations and even the MIX are automated.
## python version 3.8 <= 3.11
---
## Installation Guide
```bash
git clone https://github.com/ksl103177/RVC_mixed_audio-separator.git
```
```bash
cd RVC_mixed_audio-separator
```
```bash
conda create -n rvc python==3.10
```
```bash
activate rvc
```
```bash
pip install -r requirements.txt
```
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
---
## How to use
### Training
#### Enter the values from the rvc_train_config.yaml file in the load_yaml folder.

| parameter name | Value | options |
|:---------------|:------|:--------|
| **exp_dir1** | Name of the model to train | None |
| **sr2** | Adjust the sampling rate value | 32k, 40k, 48k |
| **if_f0_3** | Adjust whether to extract F0 | True, False |
| **spk_id5** | specify speaker ID | 0, 1, 2 (integer value) |
| **save_epoch10** | Specify the epoch to save the model checkpoint | 10, 20, 30 (integer value) |
| **total_epoch11** | Specify the total number of training epochs | 100, 200, 300 (integer value) |
| **batch_size12** | Specify the training batch size | options = 16, 32, 48 (integer value) |
| **in_save_latest13** | Specify whether to save the latest model checkpoint | True, False |
| **pretrained_G14** | Specifies the path to the pre-trained Generato model | None |
| **pretrained_D14** | Specifies the path to the pre-trained Discriminator model | None |
| **gpus16** | Specify the GPU number used for training | 0, 0,1 |
| **if_cache_gpu17** | Specify whether to cache data to GPU | True, False |
| **if_save_every_weights18** | Whether to save all checkpoints | True, False |
| **version19** | Specify the model version | v1, v2 |
| **n_p** | Specify the number of processes to use for model training | 4, 6, 8 (integer values) |
| **f0method** | Specify F0 extraction method | pm, harvest, dio, rmvpe, rmvpe_gpu |
| **gpus_rmvpe** | Specify GPU number to use for RMVPE model | 0, 0,1 (integer value) |
| **trainset_dir4** | Specify the path to the training data | None |
#### After specifying all the values needed for training, type python train_cli.py in the command.
### Inference
#### Enter the values from the rvc_main_config.yaml file in the load_yaml folder.

| parameter name | Value | options |
|:---------------|:------|:--------|
| trained_model_name | specifies the name of the trained model in the assets/weights path | None |
| file_index | Specifies the index file in the folder with the name of the trained model in the logs path | None |
| input_audio | Specifies the path to the song file for voice conversion | None |
| first_mr_output_dir | Specify the path where the MR of the song file to be voice converted is saved | None |
| output_info | Specify the path to the txt file where the log of the inference is recorded | None |
| output_audio | Specify the path where output files are saved | None |
| spk_id | specify the speaker ID specified in training | None |
| transform | Pitch transformation | -12 ~ 12 (integer value) |
| f0_file | Fixed to null value | None |
| f0_method | F0 extraction method | pm, harvest, crepe, rmvpe |
| index_rate | Index feature ratio | 0.0 ~ 1.0 (float value) |
| filter_radius | Center filtering radius | 1 ~ 5 (integer value) |
| resample_sr | Resampling sample rate | 16000, 22050, 44100, 48000 |
| rms_mix_rate | RMS mix rate | 0.0 ~ 1.0 (float value) |
| protect | Consonant and breath protection | 0.0 ~ 1.0 (float value) |
#### After specifying all the values needed for the inference, type python main.py in the command.
---
## Installing a pre-trained model
- (Path -> assets/pretrained_v2)
[Install pre-trained models(default)](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/pretrained_v2)

- (Path -> assets/pretrained_v2)
[Install Pre-trained models(korean)](https://huggingface.co/SeoulStreamingStation/KLM4/tree/main)
---
## Installing hubert
- (Path -> assets/hubert)
[Install hubert_base.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/hubert_base.pt)
---
## Installing rmvpe
- (Path -> assets/rmvpe)
[Install rmvpe.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt)