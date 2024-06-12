
# Tree Detection in 2d images
### Steps:
#### Creating Dataset and Training the model
- Add folder in the raw data folder. Each folder should contain tif file and it's corresponding shapefile.
- Run the "1. Creating dataset.py" file.
- It will create train folder which will have all the training images and masks. 
- Run the "2. Train.py" file.
- It will create multiple models in the "model/logs" folder.

#### Inference on the images
- Add test files in the "test data" folder.
- Run the "3. Inference.py" file.
- It will generate the shapefile of all the test data.



The folder structure is below:
```bash
tree detection 2d
│
├── model/logs
│           └── ... (trained models)
├── raw data/
│   └── folder 1
│   |       └── ... (tif and shp file)
│   └── folder 2
│           └── ... (tif and shp file)
├── train data/
│   └── folder 1
│   |       └── images
│   |       |      └── ... (tif files) 
│   |       └── mask
|   |
│   └── folder 2
│           └── images
│           |      └── ... (tif files) 
│           └── mask
├── test_data
│   └── ... (test data tif files)
├── test_results
│   └── ... (test result files)
├── 1. Creating dataset.py
├── 2. Train.py
├── 3. Inference.py
├── utils.py
└── requirements.txt
```
