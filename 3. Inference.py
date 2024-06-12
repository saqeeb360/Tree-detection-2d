from utils import *

CHECKPOINT_FILENAME = "model checkpoint 20.pth.tar"



overlap_threshold = 0.5  # Adjust this threshold as needed
patch_size = (1024, 1024, 3)  # Adjust this to the desired patch size and channel count
step = 512

input_folder = os.path.join(ROOT_DIR, "test data")
output_folder = os.path.join(ROOT_DIR, "test result")
os.makedirs(output_folder, exist_ok=True)

places = [file for file in os.listdir(input_folder) if file.endswith(".tif")]




#################################################################
# 
# DONOT CHANGE ANYTHING BELOW HERE
# 
################################################################# 

if __name__ == "__main__":
    checkpoint_path = os.path.join(MODEL_DIR, CHECKPOINT_FILENAME)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    model = torchvision.models.detection.maskrcnn_resnet50_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features , 2)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask , hidden_layer , 2)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Assuming model and optimizer were saved in the checkpoint
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    transform = T.ToTensor()

    for filename in places:
        input_path = os.path.join(input_folder, filename)
        print("Working on", input_path)
        inside_folder = os.path.join(output_folder, f"{filename[:-4]}")
        if not os.path.exists(inside_folder):
            os.makedirs(inside_folder)
        output_path = os.path.join(inside_folder, f"output_final_{filename[:-4]}"+".shp")
        print("Output path:", output_path)
        
        process_patches_with_overlap_adjustment(input_path, patch_size, step, overlap_threshold, output_path, model, transform, device)
        