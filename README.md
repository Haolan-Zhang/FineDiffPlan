# FineDiffPlan

This repository is partial implementation for a diffusion model for fp generation with fine-grain control. The code will be out after paper gets accepted. 

---

## Environment Setup

This project is built with Python and PyTorch. We recommend using `conda` to manage dependencies.

### 1. Clone the Repository

```bash
git clone  https://github.com/Haolan-Zhang/FineDiffPlan.git
cd floorplan-gen
```

### 2. Create the Conda Environment
```bash
conda env create -f environment.yml
conda activate finediffplan
```

### 3. Download Pretrained Model
Download the pretrained model from the following link:

```bash
[]
```
Once downloaded, place the model file in the ckpt/ directory. 

## Sampling

---

```bash
python test.py
```

## Post-Processing (Extract Input Conditions)

---

```bash
python utils/post_process.py
```