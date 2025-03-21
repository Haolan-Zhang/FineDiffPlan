import os
import torch
import logging
from tqdm import tqdm
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from utils.download import find_model
from models import DiT_XL_4, DiT_L_4
from PIL import Image
from IPython.display import display
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils.floorplan_dataloader import ImageDataset
import matplotlib.pyplot as plt

from utils.compute_fid import compute_fid
from utils.fix_bgr import fix_images_in_directory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='output.log', filemode='w')
logger = logging.getLogger()

def setup_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("GPU not found. Using CPU instead.")
    return device

def load_model(device, image_size, latent_size, model_path):
    model = DiT_XL_4(input_size=latent_size, num_classes=1).to(device)
    state_dict = find_model(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"Model loaded from {model_path}")
    return model

def load_vae(device, vae_model):
    vae = AutoencoderKL.from_pretrained(vae_model).to(device)
    logger.info(f"VAE model loaded from {vae_model}")
    return vae

def create_save_dirs(save_dir, save_dir_batched):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir_batched):
        os.makedirs(save_dir_batched)
    logger.info(f"Directories created: {save_dir}, {save_dir_batched}")

def get_dataloader(data_path, transform, batch_size=32, img_size=256):
    dataset = ImageDataset(data_path, transform=transform, img_size=img_size)
    logger.info(f"Dataset loaded from {data_path}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def save_images(samples, cond, image, save_dir, save_dir_batched, img_path, seed, i, n, samples_per_row):
    os.makedirs(f'{save_dir}/samples_{seed}', exist_ok=True)
    os.makedirs(f'{save_dir}/cond', exist_ok=True)
    os.makedirs(f'{save_dir}/gt', exist_ok=True)
    os.makedirs(f'{save_dir_batched}/samples_{seed}', exist_ok=True)
    os.makedirs(f'{save_dir_batched}/cond', exist_ok=True)
    os.makedirs(f'{save_dir_batched}/gt', exist_ok=True)

    for j in range(n):
        save_image(samples[j], os.path.join(f'{save_dir}/samples_{seed}', f"{os.path.basename(img_path[j])}"), normalize=True, value_range=(-1, 1))
        save_image(cond[j], os.path.join(f'{save_dir}/cond', f"{os.path.basename(img_path[j])}"), normalize=True, value_range=(-1, 1))
        save_image(image[j], os.path.join(f'{save_dir}/gt', f"{os.path.basename(img_path[j])}"), normalize=True, value_range=(-1, 1))
    
    save_image(samples, f"{save_dir_batched}/samples_{seed}/sample_seed{seed}_{i}.png", nrow=int(samples_per_row), 
               normalize=True, value_range=(-1, 1))
    # display(Image.open(f"{save_dir_batched}/samples/sample_seed{seed}_{i}.png"))
    save_image(cond, f"{save_dir_batched}/cond/cond_seed{seed}_{i}.png", nrow=int(samples_per_row),
               normalize=True, value_range=(-1, 1))
    # display(Image.open(f"{save_dir_batched}/cond/cond_seed{seed}_{i}.png"))
    save_image(image, f"{save_dir_batched}/gt/gt_{i}.png", nrow=int(samples_per_row),
               normalize=True, value_range=(-1, 1))
    # display(Image.open(f"{save_dir_batched}/gt/gt_{i}.png"))
    logger.info(f"Images saved for iteration {i}")

def main(data_path, model_path, vae_model, save_dir, save_dir_batched, batch_size=32, img_size=256, seed=10):
    torch.set_grad_enabled(False)
    device = setup_device()
    image_size = img_size
    latent_size = int(image_size) // 8
    model = load_model(device, image_size, latent_size, model_path)
    vae = load_vae(device, vae_model)
    
    create_save_dirs(save_dir, save_dir_batched)
    
    seed = 10
    torch.manual_seed(seed)
    num_sampling_steps = 250
    samples_per_row = 4
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    dataloader = get_dataloader(data_path, transform, batch_size, img_size)
    all_img_paths = []
    # for i in tqdm(range(len(dataloader))):
    for i, (image, cond, y, img_path)in tqdm(enumerate(dataloader)):
        if i > 5:
            break
    # for i in tqdm(range(5)):
        # image, cond, y, img_path = next(dataloader)
        all_img_paths.extend(img_path)
        diffusion = create_diffusion(str(num_sampling_steps))
        n = batch_size
        z = torch.randn(n, 4, latent_size, latent_size, device=device)
        cond = cond.float().to(device)
        y = y.long().to(device)
        
        with torch.no_grad():
            cond_en = vae.encode(cond).latent_dist.sample().mul_(0.18215)
        
        model_kwargs = dict(y=y, cond=cond_en)
        samples = diffusion.p_sample_loop(
            model.forward, z.shape, z, clip_denoised=False, 
            model_kwargs=model_kwargs, progress=True, device=device
        )
        samples = vae.decode(samples / 0.18215).sample
        save_images(samples, cond, image, save_dir, save_dir_batched, img_path, seed, i, n, samples_per_row)
    # save the image paths
    with open(f'{save_dir}/image_paths.txt', 'w') as f:
        for item in all_img_paths:
            f.write("%s\n" % item)

if __name__ == "__main__":
    seed = 10

    data_path = '/home/donaldtrump/haolan/msd_dataset/DiT_opening/test'
    data_path = '/home/donaldtrump/haolan/msd_dataset/DiT_csv_1024/test'
    data_path = 'revision'
    # data_path = '/home/donaldtrump/haolan/msd_dataset/DIT_data_9/test'

    # model_path = 'results/opening/0066000.pt'
    model_path = '/home/donaldtrump/haolan/DiT/results/009-DiT-XL-4-all_cond-512/0192000.pt'
    # model_path = 'results/new_all_cond/0030000.pt'
    # model_path = 'results/009-DiT-XL-4-all_cond-512/0192000.pt'

    vae_model = "stabilityai/sd-vae-ft-ema"
    model_name = model_path.split('/')[1]
    model_step_name = model_path.split('/')[2].split('.')[0]
    save_dir = 'fid_images/' + model_name
    save_dir_batched = 'generated_images/' + model_name
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_batched, exist_ok=True)
    logger.info("Starting main process")
    main(data_path, model_path, vae_model, save_dir, save_dir_batched, batch_size=1, img_size=512, seed=seed)
    fid_score = compute_fid(f'{save_dir}/gt', f'{save_dir}/samples_{seed}')
    logger.info(f"FID score: {fid_score}")
    # move the log file to the save_dir
    os.rename('output.log', f'{save_dir}/output_{model_step_name}.log')
    # fix the BGR images (optional)
    # fix_images_in_directory(f'{save_dir}/gt', f'{save_dir}/gt')
    # fix_images_in_directory(f'{save_dir}/samples', f'{save_dir}/samples')
    # fix_images_in_directory(f'{save_dir}/cond', f'{save_dir}/cond')
