# Free GPU compute

Checked 2026-10-27 ([GPUPerHour, Sep 2026](https://gpuperhour.com/blog/free-cloud-gpus-and-credits)).
Free tiers change without notice; check the provider before planning around them.

| Provider | GPU | Free amount | Session limit | Use for |
| --- | --- | --- | --- | --- |
| **Kaggle Notebooks** | T4 (16 GB, often 2x) or P100 | Weekly quota shown in settings, commonly about 30 h | 12 h | **Default for training** |
| Google Colab (free) | Usually T4 (16 GB) | Unpublished, varies | Up to 12 h, idle timeout | Backup, quick tests |
| Modal (Starter) | Various | US$30 / month credit | Serverless | SAM 3 auto labelling jobs (stretch) |
| Lightning AI (free) | Not stated | Unconfirmed | 1 Studio, 4 h restarts | Backup |
| Hugging Face ZeroGPU | RTX PRO 6000 slice | 5 min / day | 60 s per call | Demos only, not training |
| Local GTX 1650 | 4 GB | Unlimited | | Inference, debugging, tiny runs |

Closed: SageMaker Studio Lab (no new users), Colab Pro for Education (no new signups).

## Training on Kaggle

1. Create a Kaggle account, verify your phone number (needed for internet and GPUs in
   notebooks), and add `HF_TOKEN` (write access) under **Add ons > Secrets**.
2. New notebook: **Settings > Accelerator > GPU T4 x2**, **Internet on**.
3. In the first cell:
   ```bash
   !git clone https://github.com/AzeemChaudhry/pathsense.git
   %cd pathsense
   !pip install -q -e ".[data]"
   !python train/get_data.py openimages --max-train 20000 --max-val 2000
   ```
4. Train (T4 has 16 GB, so larger batches than locally):
   ```bash
   !yolo detect train model=yolo26n.pt data=data/openimages.yaml imgsz=640 epochs=50 \
        batch=32 device=0,1 project=/kaggle/working/runs name=yolo26n-oi
   ```
5. Push the weights to the Hub so nothing depends on the notebook surviving:
   ```python
   from kaggle_secrets import UserSecretsClient
   from huggingface_hub import HfApi
   token = UserSecretsClient().get_secret("HF_TOKEN")
   api = HfApi(token=token)
   api.create_repo("MuhammadAzeem/pathsense-detector", repo_type="model", exist_ok=True)
   api.upload_folder(
       folder_path="/kaggle/working/runs/yolo26n-oi/weights",
       repo_id="MuhammadAzeem/pathsense-detector", repo_type="model")
   ```
6. Record the run (data size, epochs, mAP, time) in NOTES.md.

Kaggle sessions stop after 12 hours: keep each run under that, and use `resume=True`
from the last checkpoint pushed to the Hub if a run is cut off.
