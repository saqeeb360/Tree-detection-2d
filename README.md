
# Tree Detection in 2d images
Link to the project: https://iitk-my.sharepoint.com/:f:/g/personal/saqeeb22_iitk_ac_in/EkUZLMnjW5lJl_ANKw0IBvsBKAcliczOJr_aZyBNlGUEKw?e=C8aQhe

### Steps:
#### Creating Dataset and Training the model
- Add folder in the raw data folder. Each folder should contain tif file and it's corresponding shapefile.
- Run the "preprocess.py" file.
- It will create train folder which will have all the training images and masks. 
- Run the "train.py" file.
- It will create multiple models in the "model_logs" folder.


##### Dataset used for training 
| File Name        | link                                                                                                                     |
|------------------|--------------------------------------------------------------------------------------------------------------------------|
| extra 1.tif   | https://ausindia-my.sharepoint.com/:i:/g/personal/jvora_aereo_io/ES7fY5Az-e5KtnQ-WosudVUB7dI5B2ADRkmLDlqPo0UX5Q?e=ecb4f9 |
| gevra.tif     | https://ausindia-my.sharepoint.com/:i:/g/personal/data_aus_co_in/EREaPDMdrbRFgX9zhLTpjCIBnKh16jrlnt8c13ToMa_Miw?e=mWz9SP |
| jsw 1.tif     | https://ausindia-my.sharepoint.com/:i:/g/personal/jvora_aereo_io/EQaRAnN35PlKq8TJRUqVZpUBDr5wRgOYmwikQAqQ0GplkA?e=7PHM71 |
| jsw 2.tif     | https://ausindia-my.sharepoint.com/:i:/g/personal/jvora_aereo_io/EQaRAnN35PlKq8TJRUqVZpUBDr5wRgOYmwikQAqQ0GplkA?e=7PHM71 |
| JUSCO_KML.tif | https://ausindia-my.sharepoint.com/:i:/g/personal/data_aus_co_in/ERyUaxZp9ExJjmcBfu2AxnMBis-M0tM9_su1NA820OuJuw?e=cIUQ4a |


#### Inference on the images
- Add test files in the "test data" folder.
- Run the "inference.py" file.
- It will generate the shapefile of all the test data.
- The link to the model is: https://iitk-my.sharepoint.com/:f:/g/personal/saqeeb22_iitk_ac_in/Ep48rXobNppLgnXLG8Y8shgBat7K9dLApOtbYiZQSQkTFg?e=FwKmfG
- Also the code downloads the base model automatically(Internet is required). And the link to the base model is https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth 


The folder structure is below:
```bash
tree detection 2d
│
├── model_logs
│   └── ... (trained models)
├── raw_data
│   ├───extra 1
│   │   ├── extra 1.tif
│   │   └── extra 1.shp and other files
│   └───gevra
│       ├── gevra.tif
│       └── gevra.shp and other files
├── train_data
│   └── extra 1
│   |       └── images
│   |       |      └── ... (tif files) 
│   |       └── mask
|   |
│   └── gevra
│           └── images
│           |      └── ... (tif files) 
│           └── mask
├── test_data
│   ├── extra 2.tif
│   └── jsw.tif
├── test_results
│   └── ... (test result files)
├── preprocess.py
├── train.py
├── inference.py
├── utils.py
├── Dockerfile
├── README.md
└── requirements.txt

```




## Steps to run this code

1. Open cmd and use the below commands to create directory, build a image, preprocess, train and infer.

2. Create 5 folders with the names
"raw_data", "train_data", "test_data", "test_results", "model_logs". Run the below command and ignore error(folder already exist).
```cmd
mkdir raw_data train_data test_data test_results model_logs
```

3. Install docker desktop and then run the below command on cmd to build a image.
```cmd
docker build -t treedetection:1 .
```

4. For training add folder with images in the raw_data folder where each folder will have large tiff file and it's corresponding shapefile. \
Naming convention: keep the folder name, tif file , shape file name same.

5. Run the below command to generate small files which will be used for training.
```cmd
 docker run -it --rm --gpus all -v "%cd%/:/myapp/" treedetection:1 python3 preprocess.py
```

6. To train a model run the below command, change 100 to number of epochs you want the model to train.
```cmd
docker run -it --rm --gpus all -v "%cd%/:/myapp/" treedetection:1 python3 train.py --epochs 100
```

7. For getting results, default test folder is "test_data". Keep all the tiif files in this folder to generate trees.
Change the model filename in the inference.py as per your need.\
Run the command to get the results. 

```cmd
docker run -it --rm --gpus all -v "%cd%/:/myapp/" treedetection:1 python3 inference.py
```

### Steps to Retrain the Model
1. Modify the retrain.py File:
    - Open the retrain.py file in your preferred text editor.
    - Update the following variables with the file name of your base model and the name you want to give to the new trained model:

```python
BASE_MODEL = "model checkpoint 80.pth.tar"
TRAIN_MODEL = "model checkpoint retrain"
```
2. Run the Docker Command:
    - Open your terminal or command prompt.
    - Navigate to the directory where your retrain.py file is located.
    - Execute the following command to start the retraining process:

```cmd
docker run -it --rm --gpus all -v "%cd%/:/myapp/" treedetection:1 python3 retrain.py --epochs 100
```

> Note: Make sure the `base model file (model checkpoint 80.pth.tar)` and all `necessary data` are present in the directory where you run the Docker command.
> Note: Adjust the `--epochs` parameter in the Docker command if you want to train for a different number of epochs.
