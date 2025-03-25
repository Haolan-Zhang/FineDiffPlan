import cv2
import os
from tqdm import tqdm

def fix_image_colors(input_path, output_path):
    """
    Convert an image from BGR to RGB to fix color issues.
    
    Args:
        input_path (str): Path to the input image with wrong colors (BGR).
        output_path (str): Path to save the corrected image (RGB).
    
    Returns:
        None
    """
    # Read the image
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {input_path}")

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Save the corrected image
    cv2.imwrite(output_path, image_rgb)
    # print(f"Fixed image saved to {output_path}")


def fix_images_in_directory(input_dir, output_dir):
    """
    Fix all images in a directory by converting from BGR to RGB.
    
    Args:
        input_dir (str): Path to the directory containing images with wrong colors.
        output_dir (str): Path to save the corrected images.
    
    Returns:
        None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in tqdm(os.listdir(input_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            fix_image_colors(input_path, output_path)

if __name__ == "__main__":

    # # Example usage:
    # fix_images_in_directory("fid_images/008-DiT-XL-4-data_aug-512/gt", "fid_images/008-DiT-XL-4-data_aug-512/gt")
    # fix_images_in_directory("fid_images/008-DiT-XL-4-data_aug-512/samples", "fid_images/008-DiT-XL-4-data_aug-512/samples")
    # fix_images_in_directory("fid_images/008-DiT-XL-4-data_aug-512/cond", "fid_images/008-DiT-XL-4-data_aug-512/cond")

    # fix_images_in_directory("generated_images/008-DiT-XL-4-data_aug-512/gt", "generated_images/008-DiT-XL-4-data_aug-512/gt")
    # fix_images_in_directory("generated_images/008-DiT-XL-4-data_aug-512/samples", "generated_images/008-DiT-XL-4-data_aug-512/samples")
    # fix_images_in_directory("generated_images/008-DiT-XL-4-data_aug-512/cond", "generated_images/008-DiT-XL-4-data_aug-512/cond")

    # fix_images_in_directory("fid_images/007-DiT-XL-4-structure-512/gt", "fid_images/007-DiT-XL-4-structure-512/gt")
    # fix_images_in_directory("fid_images/007-DiT-XL-4-structure-512/samples", "fid_images/007-DiT-XL-4-structure-512/samples")
    # fix_images_in_directory("fid_images/007-DiT-XL-4-structure-512/cond", "fid_images/007-DiT-XL-4-structure-512/cond")

    # fix_images_in_directory("generated_images/007-DiT-XL-4-structure-512/gt", "generated_images/007-DiT-XL-4-structure-512/gt")
    # fix_images_in_directory("generated_images/007-DiT-XL-4-structure-512/samples", "generated_images/007-DiT-XL-4-structure-512/samples")
    # fix_images_in_directory("generated_images/007-DiT-XL-4-structure-512/cond", "generated_images/007-DiT-XL-4-structure-512/cond")
    # save_dir = 'fid_images/009-DiT-XL-4-all_cond-512'
    # fix_images_in_directory(f'{save_dir}/gt', f'{save_dir}/gt')
    # fix_images_in_directory(f'{save_dir}/samples', f'{save_dir}/samples')
    # fix_images_in_directory(f'{save_dir}/cond', f'{save_dir}/cond')
    # fix_images_in_directory(f'/home/donaldtrump/haolan/FlexiFloor/generated_images/less_cond/cond', f'/home/donaldtrump/haolan/FlexiFloor/generated_images/less_cond/cond')
    # fix_images_in_directory(f'/home/donaldtrump/haolan/FlexiFloor/generated_images/less_cond_50/cond', f'/home/donaldtrump/haolan/FlexiFloor/generated_images/less_cond_50/cond')
    # fix_images_in_directory(f'/home/donaldtrump/haolan/FlexiFloor/generated_images/less_cond_20/cond', f'/home/donaldtrump/haolan/FlexiFloor/generated_images/less_cond_20/cond')
    # fix_images_in_directory(f'/home/donaldtrump/haolan/FlexiFloor/generated_images/less_cond_20/samples', f'/home/donaldtrump/haolan/FlexiFloor/generated_images/less_cond_20/samples')
    fix_images_in_directory(f'examples', f'examples')


    

