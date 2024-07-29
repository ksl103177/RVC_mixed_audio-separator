import logging
import os
from dotenv import load_dotenv
import yaml
from function.functions import extract_f0_feature, click_train, train_index, preprocess_dataset

# .env 파일 로드
load_dotenv('your_.env_path')

# YAML 설정 파일 로드
with open('your_train_yaml_path', 'r') as file:
    config_data = yaml.safe_load(file)

# 설정 값을 YAML 파일에서 가져오기
exp_dir1 = config_data['training']['exp_dir1']
sr2 = config_data['training']['sr2']
if_f0_3 = config_data['training']['if_f0_3']
spk_id5 = config_data['training']['spk_id5']
save_epoch10 = config_data['training']['save_epoch10']
total_epoch11 = config_data['training']['total_epoch11']
batch_size12 = config_data['training']['batch_size12']
if_save_latest13 = config_data['training']['if_save_latest13']
pretrained_G14 = config_data['training']['pretrained_G14']
pretrained_D15 = config_data['training']['pretrained_D15']
gpus16 = config_data['training']['gpus16']
if_cache_gpu17 = config_data['training']['if_cache_gpu17']
if_save_every_weights18 = config_data['training']['if_save_every_weights18']
version19 = config_data['training']['version19']
n_p = config_data['training']['n_p']
f0method = config_data['training']['f0method']
gpus_rmvpe = config_data['training']['gpus_rmvpe']
trainset_dir4 = config_data['training']['trainset_dir4']

# 로깅 설정
log_dir = os.path.join(os.getcwd(), "logs", exp_dir1)
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "train.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Step 1: 데이터 전처리
logger.info("훈련 데이터셋 전처리 진행 중...")
for output in preprocess_dataset(trainset_dir4, exp_dir1, sr2, n_p):
    logger.info(output)
logger.info("데이터 전처리 완료!")
logger.info("---------------------------------")

# Step 2: F0 및 특성 추출
logger.info("F0 및 특징 추출 진행 중...")
for output in extract_f0_feature(gpus16, n_p, f0method, if_f0_3, exp_dir1, version19, gpus_rmvpe):
    logger.info(output)
logger.info("F0 및 특징 추출 완료!")
logger.info("---------------------------------")

# Step 3: 모델 훈련
logger.info("모델 훈련 진행 중...")
training_result = click_train(
    exp_dir1,
    sr2,
    if_f0_3,
    spk_id5,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
)
logger.info(training_result)
logger.info("모델 훈련 완료!")
logger.info("---------------------------------")

# Step 4: 인덱스 훈련
logger.info("인덱스 훈련 진행 중...")
for output in train_index(exp_dir1, version19):
    logger.info(output)
logger.info("인덱스 훈련 완료!")
