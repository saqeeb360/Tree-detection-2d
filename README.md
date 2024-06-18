
# Tree Detection in 2d images
### Steps:
#### Creating Dataset and Training the model
- Add folder in the raw data folder. Each folder should contain tif file and it's corresponding shapefile.
- Run the "preprocess.py" file.
- It will create train folder which will have all the training images and masks. 
- Run the "train.py" file.
- It will create multiple models in the "model_logs" folder.

#### Inference on the images
- Add test files in the "test data" folder.
- Run the "inference.py" file.
- It will generate the shapefile of all the test data.


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