import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

from shapely.geometry import Polygon
from trimesh.creation import extrude_polygon
import trimesh
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------
# Global Definitions and Helper Functions
# ------------------------------------------------

COLORS_ROOMTYPE_HEX = [
    '#1f77b4', '#e6550d', '#fd8d3c', '#fdae6b', '#fdd0a2',
    '#72246c', '#5254a3', '#6b6ecf', '#2ca02c', '#000000',
    '#1f77b4', '#98df8a', '#d62728'
]

def hex2rgb(hex_color: str) -> tuple:
    """
    Convert a hex color (e.g. '#1f77b4') to an RGB tuple (e.g. (31, 119, 180)).
    """
    hex_color = hex_color.lstrip('#')
    hlen = len(hex_color)
    return tuple(int(hex_color[i:i+hlen//3], 16) for i in range(0, hlen, hlen//3))


def find_closest_color(gt_color: tuple, predicted_colors: list) -> tuple:
    """
    Find the closest color in `predicted_colors` to `gt_color`
    using Euclidean distance in RGB space.
    """
    min_dist = float('inf')
    closest_color = None
    for pred_color in predicted_colors:
        dist = np.linalg.norm(np.array(gt_color) - np.array(pred_color))
        if dist < min_dist:
            min_dist = dist
            closest_color = pred_color
    return closest_color


def cluster_with_position(
    image_path: str,
    num_clusters: int = 14,
    use_position: bool = True
) -> tuple:
    """
    Perform KMeans clustering on the given image. If `use_position` is True,
    it includes normalized (x, y) pixel positions in the feature set.
    
    Returns:
        clustered_image (ndarray): The image, repainted with cluster center colors.
        cluster_centers (ndarray): The (R, G, B) cluster centers from KMeans.
    """
    # Load and convert to RGB
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, c = image_rgb.shape

    # Prepare features
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    pixels = image_rgb.reshape(-1, 3)

    if use_position:
        positions = np.column_stack((X.ravel(), Y.ravel()))
        positions = positions / np.max(positions)  # normalize
        pixels = np.hstack((pixels, positions))

    # KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(pixels)
    labels = kmeans.labels_
    
    # Assign cluster centers to each pixel
    cluster_centers = kmeans.cluster_centers_[:, :3].astype(int)
    clustered_image = cluster_centers[labels].reshape(h, w, c)

    return clustered_image, cluster_centers


def clean_and_extract_polygon(
    mask: np.ndarray,
    open_kernel_size: tuple = (3, 3),
    approx_eps_factor: float = 0.008
) -> list:
    """
    Given a mask (0/1 or 0/255), perform morphological opening
    to remove noise, then find and approximate contours.
    Return a list of shapely Polygons.
    """
    # Morphological opening to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, open_kernel_size)
    clean_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    # Find and approximate contours
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        epsilon = approx_eps_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        polygons.append(Polygon(approx.squeeze()))
    return polygons


def clean_structure_polygon(
    mask: np.ndarray,
    open_kernel_size: tuple = (1, 1),
    close_kernel_size: tuple = (5, 5)
) -> Polygon:
    """
    For the 'structure' (black) region: remove noise with morphological
    opening, fill holes with closing, then create a shapely Polygon.
    """
    # Opening
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, open_kernel_size)
    noise_removed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    # Slight Gaussian smoothing
    noise_removed = cv2.GaussianBlur(noise_removed, (1, 1), 1)

    # Closing
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, close_kernel_size)
    filled = cv2.morphologyEx(noise_removed, cv2.MORPH_CLOSE, kernel_close)


    components = cv2.connectedComponents(filled.astype(np.uint8))
    max_area = 0
    max_idx = 0
    for i in range(1, components[0]):
        area = np.sum(components[1] == i)
        if area > max_area:
            max_area = area
            max_idx = i
    filled = (components[1] == max_idx).astype(np.uint8)
    # Contours
    contours, _ = cv2.findContours(filled, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Convert the outermost contour to polygon (index 0 is typically the outer contour)
    polygons = []
    for i, contour in enumerate(contours):
        # Slightly different approximation factor for the first contour
        if i == 0:
            epsilon = 0.002 * cv2.arcLength(contour, True)
        else:
            epsilon = 0.008 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        polygons.append(Polygon(approx.squeeze()))
    
    # Return the largest polygon or the first one
    # (Assumes polygons[0] is the main structure)
    if len(polygons) > 0:
        return polygons[0]
    else:
        return Polygon()


# ------------------------------------------------
# Main Processing Function
# ------------------------------------------------

def process_image(image_path: str) -> None:
    """
    Main function to:
      1. Determine number of clusters from CSV (depending on filename).
      2. Perform KMeans clustering (without position).
      3. Map cluster colors to closest known room colors.
      4. Extract polygons from each cluster.
      5. Build a 3D mesh for structure (black cluster) and rooms.
      6. Export as OBJ and visualize.
    """
    # Convert hex palette to RGB
    COLORS_ROOMTYPE = [hex2rgb(c) for c in COLORS_ROOMTYPE_HEX]
    # We add (255, 255, 255) at the end for 'ignored' color
    COLORS_ROOMTYPE.append((255, 255, 255))

    # 1) Read CSV to get num_clusters
    cluster_num_df = pd.read_csv('fid_images/009-DiT-XL-4-all_cond-512_final/num_clusters.csv')
    base_name = os.path.basename(image_path)
    if '_s' in base_name:
        new_name = base_name.replace('_s', '')
        num_clusters = cluster_num_df.loc[
            cluster_num_df['image_name'] == new_name
        ]['num_clusters'].values[0]
    else:
        num_clusters = cluster_num_df.loc[
            cluster_num_df['image_name'] == base_name
        ]['num_clusters'].values[0]

    # 2) Perform clustering
    clustered_image, dominant_colors = cluster_with_position(
        image_path, 
        num_clusters=num_clusters, 
        use_position=False
    )

    # 3) Map cluster centers to closest known colors
    new_colors = []
    for color in dominant_colors:
        closest_color = find_closest_color(color, COLORS_ROOMTYPE)
        new_colors.append(closest_color)

    # 4) Extract polygons for each cluster
    h, w, _ = clustered_image.shape
    room_polys = []
    structure_poly = None

    for idx, color in enumerate(dominant_colors):
        current_color = tuple(color)
        mapped_color = find_closest_color(current_color, COLORS_ROOMTYPE)

        # Skip if mapped color is white (ignored)
        if mapped_color == (255, 255, 255):
            continue
        
        # Create mask for the cluster
        mask = (clustered_image == current_color).all(axis=-1).astype(np.uint8)

        # If color is black -> structure
        if np.array_equal(mapped_color, np.array([0, 0, 0])):
            # Clean up structure and create one polygon
            structure_poly = clean_structure_polygon(mask)
        else:
            # Normal room polygon
            polys = clean_and_extract_polygon(mask)
            for p in polys:
                room_polys.append([p, mapped_color])

    # 5) Subtract room polygons from the structure polygon to avoid overlap
    #    (if structure_poly was found at all)
    if structure_poly is None:
        structure_poly = Polygon()
        print("No structure found in the image.")
    for poly, _ in room_polys:
        structure_poly = structure_poly.difference(poly)
    
    # 6) Create 3D meshes
    #    Extrude structure by a larger height, extrude rooms by a smaller height
    structure_mesh = extrude_polygon(structure_poly.buffer(1), 50, engine='triangle', color=[255, 0, 0])

    meshes = [structure_mesh]
    for poly, color in room_polys:
        room_mesh = extrude_polygon(poly, 1, engine='triangle')
        room_mesh.visual.face_colors = color
        meshes.append(room_mesh)

    # Combine all meshes
    final_mesh = trimesh.util.concatenate(meshes)

    # 7) Export and visualize
    os.makedirs('3d_models', exist_ok=True)
    export_name = os.path.basename(image_path).replace(".png", ".obj")
    output_path = os.path.join('3d_models', export_name)
    final_mesh.export(output_path)

    # Show the mesh (optional in headless environments)
    final_mesh.show()
    print(f"3D model exported to: {output_path}")
    return final_mesh


# Usage:

base_dir = 'fid_images/009-DiT-XL-4-all_cond-512_final/samples_10'
for image_name in os.listdir(base_dir)[:50]:
    if image_name.endswith('.png'):
        image_path = os.path.join(base_dir, image_name)
        print(f"Processing: {image_path}")
        try:
            final_mesh = process_image(image_path)
            final_mesh.show()
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
# final_mesh = process_image('fid_images/009-DiT-XL-4-all_cond-512_final/samples_10/85.png')
final_mesh.show()
