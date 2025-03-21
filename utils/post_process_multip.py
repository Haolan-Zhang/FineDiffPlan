import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pandas as pd

warnings.filterwarnings("ignore")

def cluster_with_position(image_path, num_clusters=14, use_position=True, show=False):
    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, c = image_rgb.shape

    # Prepare pixel data: [R, G, B] + optional [X, Y]
    X, Y = np.meshgrid(np.arange(w), np.arange(h))  # Coordinate grid
    pixels = image_rgb.reshape(-1, 3)

    if use_position:
        # Append normalized spatial positions to the pixel colors
        positions = np.column_stack((X.ravel(), Y.ravel()))
        positions = positions / np.max(positions)  # Normalize to [0, 1]
        pixels = np.hstack((pixels, positions))

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(pixels)
    labels = kmeans.labels_
    
    # Repaint the image with cluster centers (color part only)
    cluster_centers = kmeans.cluster_centers_[:, :3].astype(int)  # Ignore positions
    clustered_image = cluster_centers[labels].reshape(h, w, c)
    if show:
        # Display the results
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(image_rgb)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(clustered_image)
        axes[1].set_title(f"Image Clustered with Position = {use_position}")
        axes[1].axis("off")

        plt.show()

    return clustered_image, cluster_centers

def find_closest_color(gt_color, predicted_colors):
    min_dist = float('inf')
    closest_color = None
    for pred_color in predicted_colors:
        dist = np.linalg.norm(np.array(gt_color) - np.array(pred_color))
        if dist < min_dist:
            min_dist = dist
            closest_color = pred_color
    return closest_color

def compute_iou(gt_path, predicted_path, cluster_num_df=None):
    if cluster_num_df is None:
        # get the number of clusters for gt image
        for i in range(1, 15):
            clustered_image, dominant_colors = cluster_with_position(gt_path, i, use_position=False)
            redundant_colors = [i for i in dominant_colors if np.array_equal(i, np.array([255, 255, 255]))]
            if len(redundant_colors) == 2:
                num_clusters = i - 1
                # print(f"Number of clusters for gt image: {num_clusters}")
                break
    else:
        if '_s' in os.path.basename(gt_path):
            new_name = os.path.basename(gt_path).replace('_s', '')
            num_clusters = cluster_num_df.loc[cluster_num_df['image_name'] == new_name]['num_clusters'].values[0]
        else:
            num_clusters = cluster_num_df.loc[cluster_num_df['image_name'] == os.path.basename(gt_path)]['num_clusters'].values[0]
    clustered_image, dominant_colors = cluster_with_position(gt_path, num_clusters, use_position=False, show=False)
    pred_clustered_image, pred_dominant_colors = cluster_with_position(predicted_path, num_clusters, use_position=False, show=False) 

    # find the closest color in the predicted colors to the gt color
    pairs = []
    for gt_color in dominant_colors:
        closest_color = find_closest_color(gt_color, pred_dominant_colors)
        pairs.append((gt_color, closest_color))

    # compute iou
    iou = []
    for gt_color, pred_color in pairs:
        if np.array_equal(gt_color, np.array([1, 1, 1])):
            continue
        gt_mask = (clustered_image == gt_color).all(axis=-1)
        pred_mask = (pred_clustered_image == pred_color).all(axis=-1)
        intersection = np.logical_and(gt_mask, pred_mask)
        union = np.logical_or(gt_mask, pred_mask)
        iou.append(np.sum(intersection) / np.sum(union))
    iou = np.mean(iou)
    return os.path.basename(predicted_path), iou

def process_pair(pair):
    gt, pred = pair
    num_cluster_file = 'fid_images/009-DiT-XL-4-all_cond-512_final/num_clusters.csv'
    return compute_iou(gt, pred, pd.read_csv(num_cluster_file))

if __name__ == "__main__":
    # base_dir = 'fid_images/009-DiT-XL-4-all_cond-512_final'
    base_dir = '/home/donaldtrump/haolan/pytorch-CycleGAN-and-pix2pix/pix2pix_results'
    # base_dir = 'fid_images/009-DiT-XL-4-all_cond-512_structure_final'
    # base_dir = 'fid_images/007-DiT-XL-4-structure-512_'

    # seed = 10
    seed = None
    predicted_dir = base_dir + f'/samples_{seed}' if seed else base_dir + '/samples'
    predicted_path = [os.path.join(predicted_dir, i) for i in os.listdir(predicted_dir)]
    # predicted_dir_2 = 'fid_images/007-DiT-XL-4-structure-512_/samples'
    # predicted_path_2 = [os.path.join(predicted_dir_2, i) for i in os.listdir(predicted_dir_2)]
    # predicted_path.extend(predicted_path_2)
    gt_path = [i.replace(f'samples_{seed}' if seed else 'samples', 'gt') for i in predicted_path]

    pairs = list(zip(gt_path, predicted_path))
    # write the pairs to a file
    # with open(f'{base_dir}/pairs.txt', 'w') as f:
    #     for item in pairs:
    #         f.write(f"{os.path.basename(item[0])} {os.path.basename(item[1])}\n")

    with Pool(cpu_count()) as p:
        ious = list(tqdm(p.imap(process_pair, pairs), total=len(pairs)))

    print("mean iou: ", np.mean([iou[1] for iou in ious]))
    # write the iou to a file under the predicted_dir
    with open(f'{base_dir}/iou.csv', 'w') as f:
        for img_name, iou in ious:
            f.write(f"{img_name},{iou}\n")
    # # Load the files
    # iou_file = f'{base_dir}/iou.txt'
    # pairs_file = f'{base_dir}/pairs.txt'

    # # Read data
    # with open(iou_file, 'r') as f:
    #     iou_values = [float(line.strip()) for line in f.readlines()]

    # with open(pairs_file, 'r') as f:
    #     pairs = [line.strip().split() for line in f.readlines()]

    # # Create DataFrame
    # df = pd.DataFrame(pairs, columns=['gt_file_names', 'sample_file_names'])
    # df['IoU'] = iou_values

    # # Save to CSV
    # output_csv = f'{base_dir}/iou.csv'
    # df.to_csv(output_csv, index=False)