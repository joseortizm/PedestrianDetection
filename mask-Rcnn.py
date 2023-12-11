#conda environment: myclone
import os
import torch

from torchvision.io import read_image

from torchvision.ops.boxes import masks_to_boxes

from torchvision import tv_tensors

from torchvision.transforms.v2 import functional as F

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from torchvision.transforms import v2 as T

import utils

from engine import train_one_epoch, evaluate

from tqdm import tqdm

import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

root = '../datasets/PedestrianDetection/PennFudanPed'
#imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
#print(len(imgs)) #170
#masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
#print(len(masks)) #170


#personalizar y preprar de dataset

class PedestrianDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        #return self.imgs[idx], self.masks[idx]
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        #return img_path, mask_path
        img = read_image(img_path)
        mask = read_image(mask_path)
        obj_ids = torch.unique(mask)
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        boxes = masks_to_boxes(masks)

        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


    def __len__(self):
        return len(self.imgs)




#temp = ''
#dataset = PedestrianDataset(root, temp)
#img, mask = dataset[0]
#print(img) #FudanPed00001.png
#print(mask) #FudanPed00001_mask.png
#print(len(dataset)) #170

#temp = ''
#idx = 0
#dataset = PedestrianDataset(root, temp)
#path_img, path_mask = dataset[0]
##print(path_img)
##../datasets/PedestrianDetection/PennFudanPed/PNGImages/FudanPed00001.png
##print(path_mask)
##../datasets/PedestrianDetection/PennFudanPed/PedMasks/FudanPed00001_mask.png
#img = read_image(path_img)
#print(img) #tensor of file png with 559 × 536
#print(len(img)) #3: output (Tensor[image_channels, image_height, image_width])
#print(img[2])
#print(len(img[2])) #536 la altura o ancho de la imagen está representada por la dimensión H o W en la forma del tensor.
#print(img[2][0])
#print(len(img[2][0])) #559
#mask = read_image(path_mask)
#print(img[0])
##print(mask) #tensor
#
#obj_ids = torch.unique(mask) #valores unicos en mask (del elemento dataset[0])
#print(obj_ids) #tensor([0, 1, 2], dtype=torch.uint8)
#obj_ids = obj_ids[1:] #eliminamos el primer elemento (usualmente es 0) que suele ser el fondo o regiones no etiquetadas en la mascara
#print(obj_ids) #tensor([1, 2], dtype=torch.uint8)
#num_objs = len(obj_ids)
#print(num_objs) #2
#
##compara la mascara original con los valores unicos y los vuelve matriz booleana
##luego las vuelve el TRUE en 1 y False en 0
#mask_ = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8) 
#print(mask.shape) #torch.Size([1, 536, 559])
#print(len(mask)) #1
#print(mask_.shape) #torch.Size([2, 536, 559])
#print(len(mask_)) #2
#
#boxes = masks_to_boxes(mask_) #get bounding box coordinates for each mask
#print(boxes) #tensor([[159., 181., 301., 430.], [419., 170., 534., 485.]])
#
##se establece en 1 para indicar que todos los objetos pertenecen a la misma clase
#label = torch.ones((num_objs,), dtype=torch.int64)
#print(label) #tensor([1, 1])
##nota: El modelo considera la clase 0 como fondo. 
##Si su conjunto de datos no contiene la clase de fondo, no debería tener 0 en sus etiquetas.
#
#image_id = idx
##el área de cada caja delimitadora (bounding box) en la imagen: area del rectangulo
#area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
#print(area) #tensor([35358., 36225.])
#
##Crea un tensor de ceros llamado iscrowd, que se utiliza para indicar si las instancias 
##(objetos) están amontonadas o no.
## suppose all instances are not crowd (0s)
#iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
#print(iscrowd) #tensor([0, 0])
## Wrap sample and targets into torchvision tv_tensors:
#img = tv_tensors.Image(img) #se utiliza para preparar imágenes en el formato esperado por modelos de PyTorch
#print(img) #Image([[[....]]], dtype=torch.uint8, )
#
##creamos diccionario target
#target = {}
##para representar las cajas delimitadoras en el formato especificado.
##"XYXY" (coordenadas x e y de las esquinas superior izquierda e inferior derecha de la caja)
##La transformación también toma en cuenta el tamaño del lienzo de la imagen con canvas_size
#target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
##para representar las máscaras (segmentaciones) de los objetos en la imagen
#target["masks"] = tv_tensors.Mask(masks)
#target["labels"] = labels
#target["image_id"] = image_id
#target["area"] = area
#target["iscrowd"] = iscrowd

#modelo

def get_model_instance_segmentation(num_classes):
    #cargar un modelo de segmentación de instancias previamente entrenado en COCO
    #maskrcnn_resnet50_fpn con los pesos preentrenados por defector
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    #obtener el número de características de entrada para el clasificador
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    #reemplazar el pre-trained head por uno nuevo
    #la parte final del modelo que se encarga de realizar la tarea específica de clasificación
    #(modelos  como Mask R-CNN puede incluir componentes para la predicción de máscaras, ya que estos modelos abordan simultáneamente 
    #la tarea de detección, clasificación y segmentación de instancias.)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    #obtener la cantidad de características de entrada para el clasificador de máscara
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    #reemplaza el predictor de máscara por uno nuevo
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes,
    )

    return model


#helper functions to simplify training and evaluating detection models (just one time)
#os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py")
#os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py")
#os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py")
#os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py")
#os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py")


#helper functions for data augmentation/transformation
def get_transform(train):
    transforms = []
    if train:
         #de forma aleatoria hay 0.5 probabilidad de que a la imagen se le aplique voltearlo horizontal
         #para aumentar la diversidad de los datos de entrenamiento sin cambiar la etiqueta
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True)) #Convierte la imagen a tipo de datos de punto flotante (float)
    #y, si se desea, escala(scale) los valores de píxeles en el rango [0, 1]
    transforms.append(T.ToPureTensor()) # Convierte la imagen a un tensor de PyTorch
    return T.Compose(transforms)

#entrenamiento y validacion

#mps
device = torch.device('mps') if torch.cuda.is_available() else torch.device('cpu')
# nuestro conjunto de datos tiene solo dos clases: antecedentes y persona
num_classes = 2
#usamos el conjunto de datos y transformaciones definidas
dataset = PedestrianDataset(root, get_transform(train=True))
dataset_test = PedestrianDataset(root, get_transform(train=False))
#dividir dataset en train y test 
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# definir training and validation data loaders
#num_workers = 4 causo erro, con 0 funciono.
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    collate_fn=utils.collate_fn
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=utils.collate_fn
)

#obtener el modelo usando nuestra helper function
model = get_model_instance_segmentation(num_classes)

model.to(device)

#construir el optimizador
#itera sobre todos los parámetros del modelo y selecciona aquellos que tienen 
#requires_grad establecido en True. La condición requires_grad indica si 
#los gradientes de esos parámetros se deben calcular durante el entrenamiento para 
#permitir la retropropagación y la actualización de los pesos.
params = [p for p in model.parameters() if p.requires_grad]
#usamos los parametros en el optimizador
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

#Learning rate
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

num_epochs = 5

for epoch in tqdm(range(num_epochs)):
    #se entrena durante una época y se muestra cada 10 iteraciones
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    #actualizar el learning rate
    lr_scheduler.step()
    #evaluatar con el test dataset
    evaluate(model, data_loader_test, device=device)


#verificar prediccion:
image = read_image("../datasets/PedestrianDetection/tv_image05.png")
eval_transform = get_transform(train=False)

model.eval()
with torch.no_grad():
    x = eval_transform(image)
    # convert RGBA -> RGB and move to device
    x = x[:3, ...].to(device)
    predictions = model([x, ])
    pred = predictions[0]

image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
image = image[:3, ...]
pred_labels = [f"pedestrian: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
pred_boxes = pred["boxes"].long()
output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

masks = (pred["masks"] > 0.7).squeeze(1)
output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")

plt.figure(figsize=(12, 12))
plt.imshow(output_image.permute(1, 2, 0))








