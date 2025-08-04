# YOLO: You Only Look Once - Fine-Tuned for a custom dataset

## 1. What is YOLO?

**YOLO (You Only Look Once)** is a real-time object detection system that predicts bounding boxes and class probabilities directly from full images simultaneously. 
  
  
## 2. Model Choice: YOLOv8 Nano

To deployment on a Raspberry Pi (low memory), we selected **YOLOv8n (Nano)**:

> *"YOLOv8 Nano is optimized for speed and resource efficiency at the cost of some accuracy, making it ideal for lightweight applications."*  
> — [Medium article on YOLOv8 Nano vs Large](https://medium.com/@elvenkim1/yolov8-nano-vs-yolov8-large-4f21324baa38)

The size of YOLOv8n is 3.2M parameters (vs 68.2M parameters for the largest version YOLOv8X).
  
  
## 3. Fine-tuning with a Custom Dataset 

As we supposed that we will have specific classes during our following internship, we decided to fine-tune YOLOv8n with a custom dataset.
We chose an [open-source dataset of plant leaves](https://huggingface.co/datasets/agyaatcoder/PlantDoc) that includes classes not originally present in YOLO’s pretrained models.

### 3.1 Dataset Structure

Before training, the dataset was reshaped to meet YOLOv8’s format:
- Upload dataset from Hugging Face hub  

`from datasets import load_dataset`  
`dataset = load_dataset("agyaatcoder/PlantDoc")`  

- Identify labels already annoted
- Create folders expected by the YOLO model:

```
yolo_plantdoc/   
├── dataset.yaml  
├── images/  
│ ├── train/  
│ └── val/  
└── labels/  
│ ├── train/  
│ └── val/  
```

- Convert every image and its annotations
- Save these images .jpg format
- Convert bounding boxes annotations to YOLO format : class_id x_center y_center width height (valeurs normalisées entre 0 et 1)
- Generate `dataset.yaml` expected by the YOLO model

`dataset.yaml` configuration  

```
path: yolo_plantdoc       # path to dataset folder
train: images/train       # path to images folder for training
val: images/val           # path to images folder for validation 
names:                    # mapping: numbers to classes
  0: Apple Scab Leaf
  1: Apple leaf
  2: Apple rust leaf
  3: Bell_pepper leaf
  4: Bell_pepper leaf spot
  5: Blueberry leaf
  6: Cherry leaf
   ...
```

### 3.2 Fine-Tuning the YOLOv8n Model

We fine-tuned the YOLOv8n model using 60 epochs (default):  
`!yolo detect train data=/content/yolo_plantdoc/dataset.yaml model=yolov8n.pt epochs=60`  

We uploaded the [fine tuned YOLOv8n model](https://huggingface.co/Louloubib/yolov8n-finetuned) on our Hugging Face hub, so we can re-use it later (see the Next steps section). 

We can see here the **confusion matrix**, who evaluate the model's **classification** performance, and the **F1 Box Confidence Curve** for the model's **object detection** performance:  
<img width="600" height="450" alt="Untitled 38 Colab" src="https://github.com/user-attachments/assets/4d20db7a-213b-4f10-809a-d145cb50fa28" />
<img width="600" height="450" alt="Untitled 38 Colab (1)" src="https://github.com/user-attachments/assets/0aa1246c-c69e-4144-b9c6-1a484dc2a324" />  

### 3.3 Inference on a New Image

We ran an example of inference on an unseen plant leaf image (displaying the image and then running inference):  

 ```
from PIL import Image
import requests

from io import BytesIO
from ultralytics import YOLO

url = "https://extension.umd.edu/sites/extension.umd.edu/files/styles/optimized/public/2021-05/hgic_veg_bacterial%20leaf%20spot_pepper_800.jpg?itok=0snxoX9B"
response = requests.get(url)
img = Image.open(BytesIO(response.content))
display(img)

model = YOLO("runs/detect/train/weights/best.pt")
results = model(img)
results[0].show()
```
<img width="800" height="773" alt="image" src="https://github.com/user-attachments/assets/d1ba0e15-4b7d-4326-a1e2-a5555c8a73d7" />


## 4. Pros and Cons of YOLOv8n

### 4.1 Pros
- Small
- Fast
- Optimized for low-memory devices

### 4.2 Cons
- YOLO is NOT actually open-source and you can't use it commercially: you have to pay an enterprise licence
- Re shape a custom dataset in the expected format by YOLOv8n

### 4.3 Alternatives
- RT-DETR: licence apache-2.0, open source, already used similar models within previous projects, simple structure of dataset expected. Size of R50 RT-DETR: 42M parameters (vs 3.2 parameters for YOLOv8n and 68M parameters for YOLOv8X). Frame per second: 108 FPS (vs 50 FPS for YOLOV8X).

<img width="926" height="435" alt="Screenshot 2025-08-04 at 10 49 45 AM" src="https://github.com/user-attachments/assets/fa1d5d16-604a-40a4-96e8-2fb784206eff" />

- YOLOs: licence apache-2.0, open source. Size of YOLO tiny: 6.5M parameters (vs 3.2 parameters for YOLOv8n). Frames per second: 84 FPS (vs 50 FPS for YOLOV8X).


## 5. Next steps
- Comparison between YOLOv8n and RT-DETRrunning inference. Metrics: latency, evaluation metrics on specific datasets, memory footprint
- Structure a dataset for a model with classes needed for the internship project
- Running inference with **YOLOv8n** on a Raspberry Pi (or the device wich will be used during the internship)
- Running inference with **RT-DETR** on a Raspberry Pi (or the device wich will be used during the internship)
- Running inference with **YOLOS tiny** on a Raspberry Pi (or the device wich will be used during the internship)
  - (Comparison of FPS on the wanted device)
 

## Sources
https://arxiv.org/pdf/2106.00666  
https://arxiv.org/pdf/2304.08069  
https://huggingface.co/PekingU/rtdetr_r50vd  
https://huggingface.co/hustvl/yolos-tiny  
https://docs.ultralytics.com/models/rtdetr/#pre-trained-models   






