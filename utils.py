import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np

from PIL import Image
import cv2
import fiona
from shapely.geometry import shape, box
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import geopandas as gpd
import rasterio
from rasterio.errors import RasterioIOError


import torch
import torchvision
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import argparse

# Required folder structure
ROOT_DIR = os.path.abspath("")
RAWDATA_DIR = os.path.join(ROOT_DIR, "raw_data")
DATASET_DIR = os.path.join(ROOT_DIR, "train_data")
MODEL_DIR = os.path.join(ROOT_DIR,"model_logs")
TESTDATA_DIR = os.path.join(ROOT_DIR, "test_data")
os.makedirs(RAWDATA_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TESTDATA_DIR, exist_ok=True)

def get_tile_name_path(dst_dir:str, index:int, fileName:str):
    '''
    generating index specific tile name
    '''
    dst_tile_name = "{}_{}.tif".format(fileName, str(index).zfill(5))
    dst_tile_path = os.path.join(dst_dir, dst_tile_name)
    return dst_tile_name, dst_tile_path

def get_tile_transform(parent_transform, pixel_x:int,pixel_y:int):
    '''
    creating tile transform matrix from parent tif image
    '''
    crs_x = parent_transform.c + pixel_x * parent_transform.a
    crs_y = parent_transform.f + pixel_y * parent_transform.e
    tile_transform = rasterio.Affine(parent_transform.a, parent_transform.b, crs_x,
                                     parent_transform.d, parent_transform.e, crs_y)
    return tile_transform

def get_tile_profile(parent_tif:rasterio.io.DatasetReader, pixel_x:int, pixel_y:int):
    '''
    preparing tile profile
    '''
    tile_crs = parent_tif.crs
    tile_nodata = parent_tif.nodata if parent_tif.nodata is not None else 0
    tile_transform = get_tile_transform(parent_tif.transform, pixel_x, pixel_y)
    profile = dict(
                driver="GTiff",
                crs=tile_crs,
                nodata=tile_nodata,
                transform=tile_transform
            )
    return profile

def generate_tiles(tif:rasterio.io.DatasetReader, size:int, dst_dir:str, fileName:str):
    i = 0
    for x in range(0, tif.width, size):
        for y in range(0, tif.height, size):
            # creating the tile specific profile
            profile = get_tile_profile(tif, x, y)
            # extracting the pixel data (couldnt understand as i dont think thats the correct way to pass the argument)
            tile_data = tif.read(window=((y, y + size), (x, x + size)),
                                 boundless=True, fill_value=profile['nodata'])[:3]
            i+=1
            dst_name, dst_tile_path = get_tile_name_path(dst_dir, i, fileName)
            c, h, w = tile_data.shape
            profile.update(
                height=h,
                width=w,
                count=c,
                dtype=tile_data.dtype,
            )
            with rasterio.open(dst_tile_path, "w", **profile) as dst:
                dst.write(tile_data)


###############################################
# Code for creating smaller shapefiles
###############################################

def get_polygons(shp_path, target_crs):
    # Get all the polygons in the shapefile

    gdf = gpd.read_file(shp_path)
    gdf = gdf.to_crs(target_crs)
    polygons = []

    for index, row in gdf.iterrows():
        if row['geometry'].geom_type == 'Polygon':
            polygons.append((row['geometry'], row.drop('geometry').to_dict()))
        elif row['geometry'].geom_type == 'MultiPolygon':
            for geom in row['geometry'].geoms:
                polygons.append((geom, row.drop('geometry').to_dict()))
    return polygons

def get_tif_bounding_box(tif_dataset):
    bounds = tif_dataset.bounds
    min_x, min_y, max_x, max_y = bounds.left, bounds.bottom, bounds.right, bounds.top
    return (min_x, min_y, max_x, max_y)

def get_filtered_polygons(polygons, tif_dataset):
    # Get the bounding box of the TIF file
    tif_bbox = get_tif_bounding_box(tif_dataset)

    # Filter the polygons based on the intersection with the TIF file's bounding box:
    filtered_polygons = []
    for polygon, properties in polygons:
        polygon_bbox = polygon.bounds
        if box(*tif_bbox).intersects(box(*polygon_bbox)):
            filtered_polygons.append(polygon)
    return filtered_polygons

def fill_between(polygon, height, width):
    img = Image.new('1', (width, height), False)
    ImageDraw.Draw(img).polygon(polygon, outline=True, fill=True)
    mask = np.array(img)
    return mask


###############################################
# Code for Inference
###############################################

def is_mask_on_edge(mask, edge_percentage, mask_percentage):
    """
    Check if a mask is present on the edges of an image.

    Args:
        mask (numpy.ndarray): Binary mask.
        edge_percentage (float): Percentage of edges to consider (0 to 1).
        mask_percentage (float): Minimum percentage of mask pixels on an edge (0 to 1).

    Returns:
        bool: True if mask is on any edge, False otherwise.
    """
    height, width = mask.shape
    total_mask_area = np.sum(mask)
    edge_pixels = int(min(height, width) * edge_percentage)

    # Check top edge
    if np.sum(mask[:edge_pixels, :]) >= mask_percentage * total_mask_area:
        return True

    # Check bottom edge
    if np.sum(mask[-edge_pixels:, :]) >= mask_percentage * total_mask_area:
        return True

    # Check left edge
    if np.sum(mask[:, :edge_pixels]) >= mask_percentage * total_mask_area:
        return True

    # Check right edge
    if np.sum(mask[:, -edge_pixels:]) >= mask_percentage * total_mask_area:
        return True

    return False

def calculate_iou(poly1, poly2):
    if poly2.area > 0 and poly1.area > 0:
        intersection = poly1.intersection(poly2)
        if intersection.is_empty:
            return 0
        overlap = intersection.area / poly2.area
        return overlap
    else:
        return 0

def calculate_overlap(poly1, poly2):
    if poly2.area > 0 and poly1.area > 0:
        intersection = poly1.intersection(poly2)
        if intersection.is_empty:
            return 0
        overlap = max(intersection.area / poly2.area, intersection.area / poly1.area)
        return overlap
    else:
        return 0

def calculate_area(polygon):
    return polygon.area



def find_polygons(r, x, y, eps = 0.02):
    """
    Find polygons from instance segmentation results.

    Args:
        r (dict): Result dictionary containing masks and bounding boxes.
        x (int): X-coordinate shift.
        y (int): Y-coordinate shift.
        eps (float, optional): Epsilon value for polygon approximation.
            Controls the precision of the approximation. Default is 0.02.

    Returns:
        polygons (list): List of Shapely Polygon objects representing objects.
    """
    masks = r['masks']
    bboxes = r['boxes']
    n = bboxes.shape[0]
    polygons = []

    for i in range(n):
        mask = masks[i].squeeze().cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8) * 255

        # Ignore incomplete building masks on sides of images
        if is_mask_on_edge(mask, 0.2, 0.5):
            continue

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # epsilon = eps * cv2.arcLength(contour, True)
            # approx = cv2.approxPolyDP(contour, epsilon, closed=False)
            # shifted_polygon = np.array(approx) + np.array([x, y])

            # Convert the shifted_polygon to a GeoDataFrame Polygon
            shifted_polygon = np.array(contour) + np.array([x, y])

            poly_coords = [(point[0][0], point[0][1]) for point in shifted_polygon]
            if len(poly_coords) >= 4:
                polygon = Polygon(poly_coords)
                polygons.append(polygon)
    
    # combine intersecting polygons with iou > 0.5 
    # substract intersecting polygon with iou < 0.5
    for idx, polygon in enumerate(polygons):
        polygons[idx] = polygon.buffer(0)
    
    polygons = sorted(polygons, key=calculate_area, reverse=True)

    modified_tree_data = []
    for idx1, tree1 in enumerate(polygons):
        if not tree1.is_empty:
            for idx2, tree2 in enumerate(polygons):
                if idx1 < idx2 and not tree2.is_empty:
                    iou = calculate_overlap(tree1, tree2)
                    if iou > 0.5:
                        combined_polygon = tree1.union(tree2)
                        tree1 = combined_polygon
                        polygons[idx2] = Polygon()
                    elif iou > 0:
                        diff_polygon = tree1.difference(tree2)
                        tree1 = diff_polygon
            if not tree1.is_empty:
                if isinstance(tree1, Polygon):
                    modified_tree_data.append(tree1)
                elif isinstance(tree1, MultiPolygon):
                    combined_polygon = unary_union(tree1)
                    if isinstance(combined_polygon, Polygon):
                        modified_tree_data.append(combined_polygon)
                    elif isinstance(combined_polygon, MultiPolygon):
                        tree1 = list(combined_polygon.geoms)
                        modified_tree_data.extend(tree1)
                    else:
                        print("Something else")
                    # print("multi polygon",idx)
    return modified_tree_data

def apply_patch_overlap_adjustment(polygons, prev_polygons, overlap_threshold):
    adjusted_polygons = []
    for polygon in polygons:
        if polygon.is_valid:
            overlap = False  # Flag to track if there's any overlap with previous polygons
            for prev_poly in prev_polygons:
                if prev_poly.is_valid and polygon.intersection(prev_poly).area >= overlap_threshold * polygon.area:
                    overlap = True
                    break  # No need to check further, there's an overlap
            if not overlap:
                adjusted_polygons.append(polygon)
    return adjusted_polygons


def pad_patch(patch, target_size=(1024, 1024, 3)):
    """Pads the patch to the target size with zeros."""
    padded_patch = np.zeros(target_size, dtype=patch.dtype)
    original_shape = patch.shape
    
    # Copy the original patch into the padded array
    padded_patch[:original_shape[0], :original_shape[1], :original_shape[2]] = patch
    return padded_patch

def process_patches_with_overlap_adjustment(image_path, patch_size, step, overlap_threshold, output_shp_path, model, transform, device):
    # image_width, image_height = image.size
    # draw = ImageDraw.Draw(image)
    # with rasterio.open(image_path) as src:
    src = rasterio.open(image_path)
    transformPoly = src.transform
    crs = src.crs
    
    # img = Image.open(image_path)
    # width, height = img.size
    height = src.height
    width = src.width
    num_patches_y = (height + patch_size[0] - 1) // patch_size[0]
    num_patches_x = (width + patch_size[1] - 1) // patch_size[1]
    s_patch_y = 2*num_patches_y - 1
    s_patch_x = 2*num_patches_x - 1

    patch_polygons = [[[] for _ in range(s_patch_x)] for _ in range(s_patch_y)]
    all_polygons = []
    print("step", step)
    print("total patch",s_patch_x, s_patch_y)
    for i in range(s_patch_y):
        for j in range(s_patch_x):
            if j == 0:
                left = 0
            upper = i * step
            right = left + patch_size[0]
            lower = upper + patch_size[1]
            # print("L:", left, "U:", upper)
            try:
                window = rasterio.windows.Window(left, upper, 1024, 1024)
                left += step
                patch = src.read(window=window)
                patch = patch.transpose(1, 2, 0)[:,:,:3]
                if patch.shape[0] != 1024 or patch.shape[1] != 1024:
                    # print("size not 1024")
                    patch = pad_patch(patch)
                    # break
                conf_threshold = 0.1
                ig = transform(patch)

                with torch.no_grad():
                    r = model([ig.to(device)])[0]
                
                polygons = find_polygons(r, j * step, i * step)
                
                # Apply overlap adjustment with the previous row
                if i > 0:
                    prev_row_polygons = patch_polygons[i - 1][j]
                    polygons = apply_patch_overlap_adjustment(polygons, prev_row_polygons, overlap_threshold)

                # Apply overlap adjustment with the previous column
                if j > 0:
                    prev_col_polygons = patch_polygons[i][j - 1]
                    polygons = apply_patch_overlap_adjustment(polygons, prev_col_polygons, overlap_threshold)

                # Apply overlap adjustment with the diagonal patch (top-left)
                if i > 0 and j > 0:
                    prev_diag_polygons = patch_polygons[i - 1][j - 1]
                    polygons = apply_patch_overlap_adjustment(polygons, prev_diag_polygons, overlap_threshold)

                # Apply overlap adjustment with the diagonal patch (top-right)
                if i > 0 and j < s_patch_x-1:
                    prev_diag_polygons = patch_polygons[i - 1][j + 1]
                    polygons = apply_patch_overlap_adjustment(polygons, prev_diag_polygons, overlap_threshold)
                
                # Store the adjusted polygons in the 2D list
                patch_polygons[i][j] = polygons

                # Append the adjusted polygons to the list of all polygons
                # all_polygons.extend(polygons)

                # Print a message to indicate patch processing completion
                print(f"Patch {i},{j} completed")
            
            except RasterioIOError as e:
                print("Error:", e)
                print("Ignoring the error and continuing...")
                break
                
    # Transform polygon coordinates from pixel to the TIFF file's coordinate system
    all_polygons = []
    for i in range(s_patch_y):
        for j in range(s_patch_x):
            polygon_list = patch_polygons[i][j]
            all_polygons.extend(polygon_list)
    
    patch_polygons = None

    transformed_polygons = []
    for polygon in all_polygons:
        points = []
        for point in polygon.exterior.coords:
            x, y = point
            lon, lat = transformPoly * (x + 0.5, y + 0.5)
            points.append((lon, lat))
        transformed_polygons.append(Polygon(points))

    # Create a GeoDataFrame from the list of transformed polygons
    geometry = gpd.GeoSeries(transformed_polygons)
    print("All polygons", len(all_polygons))
    gdf = gpd.GeoDataFrame(geometry=geometry, crs=crs)  # You might need to adjust the CRS

    # Save the GeoDataFrame as a shapefile
    gdf.to_file(output_shp_path)
