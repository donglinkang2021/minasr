from pathlib import Path
from datetime import datetime

def save_ckpt(model_name, ckpt_path = Path('checkpoints')):
    current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    model_ckpts = ckpt_path / model_name / current_time 
    model_ckpts.mkdir(parents=True, exist_ok=True)
    return model_ckpts