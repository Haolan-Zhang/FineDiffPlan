import os
import torch
from torchvision import transforms
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance


def load_images(image_dir, transform, max_images=None):
    """Load and preprocess images from a directory."""
    images = []
    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
    if max_images:
        image_paths = image_paths[:max_images]

    for img_path in image_paths:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img = transform(img).mul(255).byte()  # Convert to uint8 tensor
            images.append(img)

    return torch.stack(images)

def compute_fid(real_images_path, generated_images_path, image_size=299, batch_size=16):
    # Transform to preprocess the images
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    # Load images
    print("Loading real images...")
    real_images = load_images(real_images_path, transform)
    print("Loading generated images...")
    generated_images = load_images(generated_images_path, transform)

    # Initialize the FiD metric
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fid = FrechetInceptionDistance(feature=2048).to(device)
    real_images = real_images.to(device)
    generated_images = generated_images.to(device)

    # Add real images to FiD
    print("Adding real images to FiD...")
    for i in range(0, len(real_images), batch_size):
        batch = real_images[i:i + batch_size]
        fid.update(batch, real=True)

    # Add generated images to FiD
    print("Adding generated images to FiD...")
    for i in range(0, len(generated_images), batch_size):
        batch = generated_images[i:i + batch_size]
        fid.update(batch, real=False)

    # Compute FiD score
    print("Computing FiD score...")
    fid_score = fid.compute().item()
    print(f"FiD score: {fid_score}")
    return fid_score

def IoU(pred, target):
    """Compute Intersection over Union (IoU) for binary segmentation."""
    intersection = (pred & target).float().sum((1, 2))
    union = (pred | target).float().sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()



if __name__ == "__main__":
    # Example usage
    # all_img_dir = 'fid_images'
    # model_name = '001-DiT-XL-4'
    # real_images_path = f'{all_img_dir}/{model_name}/gt'
    # generated_images_path = f'{all_img_dir}/{model_name}/samples'

    real_images_path = '/home/donaldtrump/haolan/pytorch-CycleGAN-and-pix2pix/pix2pix_results/gt'
    generated_images_path = '/home/donaldtrump/haolan/pytorch-CycleGAN-and-pix2pix/pix2pix_results/samples'
    compute_fid(real_images_path, generated_images_path)
