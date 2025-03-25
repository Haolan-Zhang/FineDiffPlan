# FineDiffPlan

This repository is partial implementation for a diffusion model for fp generation with fine-grain control. The code will be out after paper gets accepted. 

---

## Environment Setup

This project is built with Python and PyTorch. We recommend using `conda` to manage dependencies.

### 1. Clone the Repository

```bash
git clone  https://github.com/Haolan-Zhang/FineDiffPlan.git
cd FineDiffPlan
```

### 2. Create the Conda Environment
```bash
conda env create -f environment.yml
conda activate finediffplan
```

### 3. Download Pretrained Model
Download the pretrained model by filling the anonymous form (expires Apr.24th, with limited capability):

```bash
https://forms.office.com/r/tihwZQ0GLE
```
Once downloaded, place the model file in the ckpt/ directory. 

## Sampling

```bash
python test.py
```
There will be a few new directories appear and contains the result.

## Post-Processing (Extract Input Conditions )

```bash
python utils/post_process.py
```
Then the result image will appear under the directory.