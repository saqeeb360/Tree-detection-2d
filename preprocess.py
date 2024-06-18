from utils import *


'''
raw_data_list should contains the folder names of the training files.
you can either pass a variable list or all folders in the raw data folder.
'''
# raw_data_list = ["train cropped", "extra 1", "extra 2"]
raw_data_list = os.listdir(RAWDATA_DIR)





#################################################################
# 
# DONOT CHANGE ANYTHING BELOW HERE
# 
################################################################# 
print("Creating smaller patches...")
for folder in raw_data_list:
    print(folder)
    rawdata_path = os.path.join(RAWDATA_DIR, folder)
    src_path = os.path.join(rawdata_path, folder+".tif")
    data = rasterio.open(src_path)
    dst_image_dir = os.path.join(DATASET_DIR, folder, "images")
    os.makedirs(dst_image_dir, exist_ok=True)
    generate_tiles(data, 1024, dst_image_dir, folder)


print("Deleting tif files with no trees...")
# Deleting files which dont have any trees
for folder in raw_data_list:
    print(folder)
    dataset_dir = os.path.join(DATASET_DIR, folder, "images")

    for file in os.listdir(dataset_dir):
        if file.endswith(".tif"):
            tif_path = os.path.join(dataset_dir, file)
            with rasterio.open(tif_path) as dataset:
                bounds = dataset.bounds
                min_x, min_y, max_x, max_y = bounds.left, bounds.bottom, bounds.right, bounds.top
            
            filestrsplit = '_'.join(file.split("_")[:-1])
            rawdata_path = os.path.join(RAWDATA_DIR, filestrsplit)
            shp_path = os.path.join(rawdata_path, filestrsplit+".shp")
            # shp_path = "/content/drive/MyDrive/ThesisData/01/data/P1 FA/mask/P1 FA.shp"
            polygons = []
            with fiona.open(shp_path, "r") as shapefile:
                for feature in shapefile:
                    geometry = shape(feature["geometry"])
                    polygons.append((geometry, feature["properties"]))
            tif_bbox = (min_x, min_y, max_x, max_y)
            filtered_polygons = []
            for polygon, properties in polygons:
                polygon_bbox = polygon.bounds
                if box(*tif_bbox).intersects(box(*polygon_bbox)):
                    filtered_polygons.append(polygon)
            if(len(filtered_polygons) == 0):
                # print(file, len(filtered_polygons))
                try:
                    os.remove(tif_path)
                    # print(f"Removed: {tif_path} {len(filtered_polygons)}")
                except OSError:
                    print(f"Error while deleting file: {tif_path}")


#####################################################
# Code for creating smaller shapefiles
#####################################################

print("Creating smaller shapefiles...")
for place in raw_data_list:
# for place in os.listdir(DATASET_DIR):
    print(place)
    images = os.path.join(DATASET_DIR, place, "images")
    masks_path = os.path.join(RAWDATA_DIR, place)
    mask_file_name = place+".shp"
    dst_mask_path = os.path.join(DATASET_DIR, place, "mask")
    os.makedirs(dst_mask_path, exist_ok=True)

    for filename in os.listdir(images):
        tif_file_path = os.path.join(images, filename)
        tif_dataset = rasterio.open(tif_file_path)

        height = tif_dataset.height
        width = tif_dataset.width
        transform = tif_dataset.transform

        shp_file_path = os.path.join(masks_path, mask_file_name)
        polygons = get_polygons(shp_file_path, tif_dataset.crs)
        filtered_polygons = get_filtered_polygons(polygons, tif_dataset)

        # save to shape file filtered polygons
        gdf = gpd.GeoDataFrame(geometry=filtered_polygons)
        shp_filename = filename.split(".")[0] + ".shp"
        shp_path = os.path.join(dst_mask_path, shp_filename)
        gdf.to_file(shp_path, driver='ESRI Shapefile', crs=tif_dataset.crs)

#####################################################
# Code for removing corrupt shapefiles
#####################################################


print("Removing corrupt shapefiles...")
bad_imagesDATA = dict()
for place in raw_data_list:
# for place in os.listdir(DATASET_DIR):
    print(place)
    images = os.path.join(DATASET_DIR, place, "images")
    masks_path = os.path.join(DATASET_DIR, place, "mask")

    mask_file_name = place + ".shp"
    bad_images = set()
    for filename in os.listdir(images):
        tif_file_path = os.path.join(images, filename)
        # img = Image.open(tif_file_path).convert("RGB")

        tif_dataset = rasterio.open(tif_file_path)
        height = tif_dataset.height
        width = tif_dataset.width
        transform = tif_dataset.transform

        # open filtered polygon from shapefile
        shp_filename = filename.split(".")[0] + ".shp"

        filtered_polygons = gpd.read_file(os.path.join(masks_path, shp_filename))
        masks = np.zeros((len(filtered_polygons), height, width), dtype=np.uint8)
        for idx, polygon in enumerate(filtered_polygons["geometry"]):
            coordinates = list()
            for point in polygon.exterior.coords:
                x, y = point
                pixel_x, pixel_y = ~transform * (x, y)
                pixel_x = width - 1 if pixel_x > width else pixel_x
                pixel_y = height - 1 if pixel_y > height else pixel_y
                coordinates.append((pixel_x, pixel_y))
                # print(pixel_x, pixel_y)

            mask = fill_between(coordinates, height, width)
            masks[idx, :, :] = mask
            pos = np.where(masks[idx])
            if len(pos[0]) == 0 or len(pos[1]) == 0:
                print("Full blank mask",filename)
                bad_images.add(filename)
                break
            else:
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                if xmin == xmax:
                    # os.remove(tif_file_path)
                    # print(filename, "X")
                    bad_images.add(filename)
                    break
                elif ymin == ymax:
                    # print(filename, "Y")
                    bad_images.add(filename)
                    break
    
    # print(bad_images)
    bad_imagesDATA[place] = bad_images

# print(bad_imagesDATA)
for key, item in bad_imagesDATA.items():
    bad_images = bad_imagesDATA[key]
    # print(bad_images)
    for filename in bad_images:
        finalpth = os.path.join(DATASET_DIR, key, "images",filename)
        if os.path.exists(finalpth):
            os.remove(finalpth)
        else:
            print("Path not exist", finalpth)

print("Completed data preprocessing.")