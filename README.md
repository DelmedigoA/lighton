# lighton

Simple starter repo to fine-tune **LightOnOCR-2-1B** with **LoRA** using a YAML config file.

## 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Configure

Edit `config.yaml` to change:
- model ID
- dataset/split
- prompt
- LoRA settings
- training hyperparameters

## 3) Train

```bash
python train.py --config config.yaml
```

The script will:
- load `lightonai/LightOnOCR-2-1B`
- preprocess `naver-clova-ix/cord-v2`
- apply LoRA (`q_proj`, `v_proj`)
- train with Hugging Face `Trainer`
- save adapters to `outputs/my-lightonocr-lora`

## Notes

- Device auto-detection prefers: `mps` -> `cuda` -> `cpu`
- Dtype defaults to `float32` on MPS, `bfloat16` elsewhere (when `dtype: auto`)
