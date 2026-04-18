# dehazing-compression
Dehazing Model Compression, Quantization + Condition-Specific Distillation for Real-Time Inference

dehazing-compression/
│
├── configs/
│   ├── ptq_dehamer.yaml
│   ├── ptq_restormer.yaml
│   ├── distill_haze.yaml
│   └── distill_rain.yaml
│
├── data/
│   ├── reside.py          # RESIDE ITS/OTS/SOTS dataloader
│   └── rain13k.py         # Rain13K dataloader
│
├── models/
│   ├── teachers/
│   │   ├── dehamer.py     # thin wrapper: loads DeHamer checkpoint, exposes forward + hooks
│   │   └── restormer.py   # thin wrapper: loads Restormer checkpoint
│   ├── students/
│   │   └── nafnet_student.py  # NAFNet-32 with adapter layers for feature matching
│   └── quantized/
│       └── quant_utils.py     # PTQ helpers, sensitivity scan, mixed-precision config
│
├── phase1_quantize/
│   ├── run_ptq.py         # main script: quantize → evaluate → log
│   └── sensitivity.py     # per-layer sensitivity analysis
│
├── phase2_distill/
│   ├── train.py           # main training loop
│   └── losses.py          # L_pixel, L_feat, L_perceptual
│
├── evaluate/
│   ├── metrics.py         # PSNR, SSIM, FADE, FPS
│   └── benchmark.py       # latency table generator
│
├── scripts/
│   ├── download_reside.sh
│   ├── download_rain13k.sh
│   └── run_all_baselines.sh
│
├── experiments/           # gitignored outputs, checkpoints
├── results/               # tables, figures — committed
├── requirements.txt
└── README.md