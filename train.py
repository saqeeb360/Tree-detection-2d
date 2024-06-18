from utils import *

TOTAL_EPOCH = 100
Model_Filename = "model checkpoint"


#################################################################
# 
# DONOT CHANGE ANYTHING BELOW HERE
# 
################################################################# 

os.makedirs(MODEL_DIR, exist_ok=True)
places = os.listdir(DATASET_DIR)
images_data = []
masks_data = []

for place in places:
    images = os.path.join(DATASET_DIR, place, "images")
    masks_path = os.path.join(DATASET_DIR, place, "mask")
    for filename in os.listdir(images):
        shp_filename = filename.split(".")[0] + ".shp"
        images_data.append(os.path.join(images, filename))
        masks_data.append(os.path.join(masks_path, shp_filename))

class CustDat(torch.utils.data.Dataset):
    def __init__(self , images , masks):
        self.imgs = images
        self.masks = masks

    def __getitem__(self , idx):
        tif_file_path = self.imgs[idx]
        # print("the idx:",filename, " " , idx)
        # tif_file_path = os.path.join(images, filename)

        tif_dataset = rasterio.open(tif_file_path)
        height = tif_dataset.height
        width = tif_dataset.width
        transform = tif_dataset.transform
        img = Image.open(tif_file_path).convert("RGB")

        filtered_polygons = gpd.read_file(self.masks[idx])
        eachImg_mask = np.zeros((len(filtered_polygons), height, width), dtype=np.uint8)
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
            eachImg_mask[idx, :, :] = mask

        # eachImg_mask = masks_data[filename]
        num_objs = eachImg_mask.shape[0]
        boxes = []
        for i in range(num_objs):
            pos = np.where(eachImg_mask[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin , ymin , xmax , ymax])

        boxes = torch.as_tensor(boxes , dtype = torch.float32)
        labels = torch.ones((num_objs,) , dtype = torch.int64)
        masks = torch.as_tensor(eachImg_mask , dtype = torch.uint8)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = eachImg_mask

        return T.ToTensor()(img) , target

    def __len__(self):
        return len(self.imgs)
    
transform = T.ToTensor()
def custom_collate(data):
  return data

totalsize = 18 * (len(images_data) // 18)
images_data = images_data[:totalsize]
masks_data = masks_data[:totalsize]

num = int(0.9 * len(images_data))
num = num if num % 2 == 0 else num + 1
train_imgs_inds = np.random.choice(range(len(images_data)) , num , replace = False)
val_imgs_inds = np.setdiff1d(range(len(images_data)) , train_imgs_inds)
train_imgs = np.array(images_data)[train_imgs_inds]
train_masks = np.array(masks_data)[train_imgs_inds]

val_imgs = np.array(images_data)[val_imgs_inds]
val_masks = np.array(masks_data)[val_imgs_inds]


train_dl = torch.utils.data.DataLoader(CustDat(train_imgs , train_masks) ,
                                 batch_size = 2 ,
                                 shuffle = True ,
                                 collate_fn = custom_collate ,
                                 num_workers = 1 ,
                                 pin_memory = True if torch.cuda.is_available() else False)
val_dl = torch.utils.data.DataLoader(CustDat(val_imgs , val_masks) ,
                                 batch_size = 2 ,
                                 shuffle = True ,
                                 collate_fn = custom_collate ,
                                 num_workers = 1 ,
                                 pin_memory = True if torch.cuda.is_available() else False)


model = torchvision.models.detection.maskrcnn_resnet50_fpn()
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features , 2)
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask , hidden_layer , 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

def main():
    best_val_loss = float('inf')
    all_train_losses = []
    all_val_losses = []
    flag = False
    for epoch in range(TOTAL_EPOCH+1):
        train_epoch_loss = 0
        val_epoch_loss = 0
        model.train()

        for i , dt in enumerate(train_dl):
            imgs = [dt[0][0].to(device) , dt[1][0].to(device)]
            targ = [dt[0][1] , dt[1][1]]

            # targets = [{k: v.to(device) for k, v in t.items()} for t in targ]
            targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targ]

            loss = model(imgs , targets)
            if not flag:
                print(loss)
                flag = True

            losses = sum([l for l in loss.values()])
            train_epoch_loss += losses.cpu().detach().numpy()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        all_train_losses.append(train_epoch_loss)

        with torch.no_grad():
            for j , dt in enumerate(val_dl):
                imgs = [dt[0][0].to(device) , dt[1][0].to(device)]
                targ = [dt[0][1] , dt[1][1]]

                targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targ]

                loss = model(imgs , targets)
                losses = sum([l for l in loss.values()])
                val_epoch_loss += losses.cpu().detach().numpy()

            all_val_losses.append(val_epoch_loss)
        print(epoch , "  " , train_epoch_loss , "  " , val_epoch_loss)
        
        if val_epoch_loss < best_val_loss or epoch % 5 == 0:
            best_val_loss = min(val_epoch_loss, best_val_loss)
            checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            torch.save(checkpoint, os.path.join(MODEL_DIR,f"{Model_Filename} {epoch}.pth.tar"))
            # print(f"Model saved at epoch {epoch} with validation loss {val_epoch_loss}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train for')
    args = parser.parse_args()
    TOTAL_EPOCH = args.epochs
    print(f"The total number of epochs is {TOTAL_EPOCH}")
    main()