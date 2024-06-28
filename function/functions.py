import os
import sys
import logging
import threading
import pathlib
import json
import numpy as np
import traceback
import faiss
import torch, platform

from dotenv import load_dotenv
from configs.config import Config
from infer.modules.vc.modules import VC
from subprocess import Popen, PIPE
from time import sleep
from random import shuffle
from i18n.i18n import I18nAuto
from sklearn.cluster import MiniBatchKMeans

load_dotenv()

now_dir = os.getcwd()
sys.path.append(now_dir)

os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "assets/weights"), exist_ok=True)

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

outside_index_root = os.getenv("outside_index_root")

if outside_index_root is None:
    outside_index_root = os.path.join(now_dir, "logs", "outside_index")

index_root = os.getenv("index_root")
outside_index_root = os.getenv("outside_index_root")

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

logger = logging.getLogger(__name__)

config = Config()
vc = VC(config)
i18n = I18nAuto()

sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}

def change_info_(ckpt_path):
    if not os.path.exists(ckpt_path.replace(os.path.basename(ckpt_path), "train.log")):
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}
    try:
        with open(ckpt_path.replace(os.path.basename(ckpt_path), "train.log"), "r") as f:
            lines = f.readlines()
            for line in lines:
                if "sample_rate" in line:
                    sr = line.split(":")[-1].strip()
                if "if_f0" in line:
                    f0 = line.split(":")[-1].strip()
                if "version" in line:
                    version = line.split(":")[-1].strip()
            return sr, f0, version
    except:
        traceback.print_exc()
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}


def if_done(done, p):
    while p.poll() is None:
        sleep(1)  # 1초 주기로 확인
    done[0] = True

def if_done_multi(done, ps):
    while not all(p.poll() is not None for p in ps):
        sleep(1)  # 1초 주기로 확인
    done[0] = True

def preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
    sr = sr_dict[sr]
    os.makedirs(f"{now_dir}/logs/{exp_dir}", exist_ok=True)
    log_path = f"{now_dir}/logs/{exp_dir}/preprocess.log"
    open(log_path, "w").close()

    cmd = f'"{config.python_cmd}" infer/modules/train/preprocess.py "{trainset_dir}" {sr} {n_p} "{now_dir}/logs/{exp_dir}" {config.noparallel} {config.preprocess_per:.1f}'
    logger.info("Execute: " + cmd)
    p = Popen(cmd, shell=True)
    
    done = [False]
    threading.Thread(target=if_done, args=(done, p)).start()

    # 프로세스가 완료될 때까지 기다림
    while not done[0]:
        sleep(1)
    
    # 프로세스 완료 후 한 번만 로그 파일 읽기
    with open(log_path, "r") as f:
        log_content = f.read()
    logger.info(log_content)
    yield log_content

def extract_f0_feature(gpus, n_p, f0method, if_f0, exp_dir, version19, gpus_rmvpe):
    gpus = gpus.split("-")
    os.makedirs(f"{now_dir}/logs/{exp_dir}", exist_ok=True)
    log_path = f"{now_dir}/logs/{exp_dir}/extract_f0_feature.log"
    open(log_path, "w").close()

    if if_f0:
        if f0method != "rmvpe_gpu":
            cmd = f'"{config.python_cmd}" infer/modules/train/extract/extract_f0_print.py "{now_dir}/logs/{exp_dir}" {n_p} {f0method}'
            logger.info("Execute: " + cmd)
            p = Popen(cmd, shell=True, cwd=now_dir)
            done = [False]
            threading.Thread(target=if_done, args=(done, p)).start()

            while not done[0]:
                sleep(1)
            
            with open(log_path, "r") as f:
                log_content = f.read()
            logger.info(log_content)
            yield log_content
        else:
            if gpus_rmvpe != "-":
                gpus_rmvpe = gpus_rmvpe.split("-")
                leng = len(gpus_rmvpe)
                ps = []
                for idx, n_g in enumerate(gpus_rmvpe):
                    cmd = (
                        f'"{config.python_cmd}" infer/modules/train/extract/extract_f0_rmvpe.py {leng} {idx} {n_g} "{now_dir}/logs/{exp_dir}" {config.is_half}'
                    )
                    logger.info("Execute: " + cmd)
                    p = Popen(cmd, shell=True, cwd=now_dir)
                    ps.append(p)
                
                done = [False]
                threading.Thread(target=if_done_multi, args=(done, ps)).start()

                while not done[0]:
                    sleep(1)
                
                with open(log_path, "r") as f:
                    log_content = f.read()
                logger.info(log_content)
                yield log_content
            else:
                cmd = (
                    f'{config.python_cmd} infer/modules/train/extract/extract_f0_rmvpe_dml.py "{now_dir}/logs/{exp_dir}"'
                )
                logger.info("Execute: " + cmd)
                p = Popen(cmd, shell=True, cwd=now_dir)
                p.wait()
                
                with open(log_path, "r") as f:
                    log_content = f.read()
                logger.info(log_content)
                yield log_content

    leng = len(gpus)
    ps = []
    for idx, n_g in enumerate(gpus):
        cmd = (
            f'"{config.python_cmd}" infer/modules/train/extract_feature_print.py {config.device} {leng} {idx} {n_g} "{now_dir}/logs/{exp_dir}" {version19} {config.is_half}'
        )
        logger.info("Execute: " + cmd)
        p = Popen(cmd, shell=True, cwd=now_dir)
        ps.append(p)

    done = [False]
    threading.Thread(target=if_done_multi, args=(done, ps)).start()

    while not done[0]:
        sleep(1)
    
    with open(log_path, "r") as f:
        log_content = f.read()
    logger.info(log_content)
    yield log_content

def click_train(
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
):
    exp_dir = os.path.join(now_dir, "logs", exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = os.path.join(exp_dir, "0_gt_wavs")
    feature_dir = os.path.join(exp_dir, "3_feature256" if version19 == "v1" else "3_feature768")
    train_log_path = os.path.join(exp_dir, "train.log")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(train_log_path)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)

    if if_f0_3:
        f0_dir = os.path.join(exp_dir, "2a_f0")
        f0nsf_dir = os.path.join(exp_dir, "2b-f0nsf")
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "{}|{}|{}|{}|{}".format(
                    os.path.join(gt_wavs_dir, f"{name}.wav").replace("\\", "\\\\"),
                    os.path.join(feature_dir, f"{name}.npy").replace("\\", "\\\\"),
                    os.path.join(f0_dir, f"{name}.wav.npy").replace("\\", "\\\\"),
                    os.path.join(f0nsf_dir, f"{name}.wav.npy").replace("\\", "\\\\"),
                    spk_id5,
                )
            )
        else:
            opt.append(
                "{}|{}|{}".format(
                    os.path.join(gt_wavs_dir, f"{name}.wav").replace("\\", "\\\\"),
                    os.path.join(feature_dir, f"{name}.npy").replace("\\", "\\\\"),
                    spk_id5,
                )
            )
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "{}|{}|{}|{}|{}".format(
                    os.path.join(now_dir, f"logs/mute/0_gt_wavs/mute{sr2}.wav"),
                    os.path.join(now_dir, f"logs/mute/3_feature{fea_dim}/mute.npy"),
                    os.path.join(now_dir, "logs/mute/2a_f0/mute.wav.npy"),
                    os.path.join(now_dir, "logs/mute/2b-f0nsf/mute.wav.npy"),
                    spk_id5,
                )
            )
    else:
        for _ in range(2):
            opt.append(
                "{}|{}|{}".format(
                    os.path.join(now_dir, f"logs/mute/0_gt_wavs/mute{sr2}.wav"),
                    os.path.join(now_dir, f"logs/mute/3_feature{fea_dim}/mute.npy"),
                    spk_id5,
                )
            )
    shuffle(opt)
    with open(os.path.join(exp_dir, "filelist.txt"), "w") as f:
        f.write("\n".join(opt))
    logger.debug("Write filelist done")
    logger.info("Use gpus: %s", str(gpus16))
    if pretrained_G14 == "":
        logger.info("No pretrained Generator")
    if pretrained_D15 == "":
        logger.info("No pretrained Discriminator")
    if version19 == "v1" or sr2 == "40k":
        config_path = f"v1/{sr2}.json"
    else:
        config_path = f"v2/{sr2}.json"
    config_save_path = os.path.join(exp_dir, "config.json")
    if not pathlib.Path(config_save_path).exists():
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(
                config.json_config[config_path],
                f,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            )
            f.write("\n")
    cmd = (
        f'"{config.python_cmd}" infer/modules/train/train.py -e "{exp_dir1}" -sr {sr2} -f0 {1 if if_f0_3 else 0} -bs {batch_size12} -te {total_epoch11} -se {save_epoch10} '
        f'{"-pg " + pretrained_G14 if pretrained_G14 else ""} {"-pd " + pretrained_D15 if pretrained_D15 else ""} -l {1 if if_save_latest13 else 0} -c {1 if if_cache_gpu17 else 0} '
        f'-sw {1 if if_save_every_weights18 else 0} -v {version19}'
    )
    if gpus16:
        cmd += f' -g "{gpus16}"'

    logger.info("Execute: " + cmd)

    with Popen(cmd, shell=True, cwd=now_dir, stdout=PIPE, stderr=PIPE, bufsize=1, universal_newlines=True) as p:
        with open(train_log_path, "a") as log_file:
            for line in p.stdout:
                if "Epoch" in line or "Saving model and optimizer state" in line:
                    log_file.write(line)
                    logger.info(line.strip())
            for line in p.stderr:
                if not ("RequestsDependencyWarning" in line or "UserWarning" in line or "resource_tracker" in line):
                    log_file.write(line)
                    logger.error(line.strip())

    p.wait()
    
    sr, f0, version = change_info_(train_log_path)
    logger.info(f"Training completed with sample rate: {sr}, f0: {f0}, version: {version}")
    
    return "훈련이 끝나면 콘솔 훈련 로그 또는 아래의 실험 폴더를 볼 수 있습니다. -> train.log"

def train_index(exp_dir1, version19):
    exp_dir = os.path.join("logs", exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = os.path.join(exp_dir, "3_feature256" if version19 == "v1" else "3_feature768")
    if not os.path.exists(feature_dir):
        return "먼저 특징 추출을 하세요!"
    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        return "먼저 특징 추출을 하세요!"
    infos = []
    npys = []
    for name in sorted(listdir_res):
        phone = np.load(os.path.join(feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    if big_npy.shape[0] > 2e5:
        infos.append(f"Trying doing kmeans {big_npy.shape[0]} shape to 10k centers.")
        yield "\n".join(infos)
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * config.n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except:
            info = traceback.format_exc()
            logger.info(info)
            infos.append(info)
            yield "\n".join(infos)

    np.save(os.path.join(exp_dir, "total_fea.npy"), big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos.append(f"{big_npy.shape},{n_ivf}")
    yield "\n".join(infos)
    index = faiss.index_factory(256 if version19 == "v1" else 768, f"IVF{n_ivf},Flat")
    infos.append("training")
    yield "\n".join(infos)
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        os.path.join(exp_dir, f"trained_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir1}_{version19}.index"),
    )
    infos.append("adding")
    yield "\n".join(infos)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    index_save_path = os.path.join(exp_dir, f"added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir1}_{version19}.index")
    faiss.write_index(index, index_save_path)
    infos.append(i18n("인덱스를 성공적으로 빌드했습니다.") + " " + index_save_path)
    link_target = os.path.join(outside_index_root, f"{exp_dir1}_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir1}_{version19}.index")
    try:
        link = os.link if platform.system() == "Windows" else os.symlink
        link(index_save_path, link_target)
        infos.append(i18n("외부 링크 인덱스") + " " + link_target)
    except:
        infos.append(i18n("외부 링크 인덱스") + " " + link_target + " " + i18n("실패(예: 실험)"))

    yield "\n".join(infos)
