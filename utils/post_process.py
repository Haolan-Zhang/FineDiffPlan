import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def cluster_with_position(image_path, num_clusters=14, use_position=True):
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

# Parameters

image_path = 'examples/post_processimg.png'


num_clusters = 7

# Run clustering with spatial positions
clustered_image, dominant_colors = cluster_with_position(image_path, num_clusters, use_position=False)

# Output dominant colors
print("Dominant Colors (RGB):")
plt.figure(figsize=(12, 15))
for idx, color in enumerate(dominant_colors):
    print(f"Cluster {idx+1}: {tuple(color)}")
    mask = (clustered_image == color).all(axis=-1)
    if np.array_equal(color, np.array([255, 255, 255])):
        continue
    plt.subplot(4, 3, idx+1)
    # use the color that is painted with the dominant color to represent the mask in matplotlib, the background is white
    cmap = plt.cm.colors.ListedColormap([[1, 1, 1], color/255])
    plt.imshow(mask, cmap=cmap)
    plt.title(f"Cluster {idx+1}")
    # remove the numbers on the x, y axis
    plt.xticks([])
    plt.yticks([])
plt.show()
plt.savefig('31_s_clusters.png')
