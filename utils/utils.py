from pathlib import Path
from datetime import datetime

ckpt_root = "/opt/data/private/linkdom/model/minasr/checkpoints"

def save_ckpt(model_name, ckpt_path = Path(ckpt_root)):
    current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    model_ckpts = ckpt_path / model_name / current_time 
    model_ckpts.mkdir(parents=True, exist_ok=True)
    return model_ckpts

def get_previous_ckpt(model_name, ckpt_path = Path(ckpt_root)):
    model_ckpts = list(ckpt_path.glob(f"{model_name}/*"))
    if len(model_ckpts) == 0:
        return None
    model_ckpts.sort()
    return model_ckpts[-1]

# model_name = f"pretrain_minihubert_kmeans_512"
# print(f"model_name: {model_name}")
# prev_ckpt_path = get_previous_ckpt(model_name)
# print(f"prev_ckpt_path: {prev_ckpt_path}")