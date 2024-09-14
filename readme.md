
## План проекта

### 1. Подготовка данных
- **Сбор данных**: Использовать предоставленные 53 изображения и соответствующие JSON-файлы разметки.
- **Обработка аннотаций**: Преобразовать разметку из JSON-файлов в маски изображений, соответствующие областям для инъекций.
- **Аугментация данных**: Применить методы аугментации для увеличения объема данных и улучшения обобщающей способности модели.

### 2. Разработка модели
- **Выбор архитектуры**: Использовать проверенные модели для семантической сегментации, такие как U-Net или DeepLabV3.
- **Настройка гиперпараметров**: Определить оптимальные значения для скорости обучения, функции потерь, оптимизатора и т.д.

### 3. Обучение модели
- **Настройка даталоадеров**: Создать пользовательские `Dataset` и `DataLoader` для загрузки изображений и масок.
- **Обучение с использованием MLflow**: Интегрировать MLflow для отслеживания экспериментов, параметров и метрик.
- **Валидация модели**: Использовать часть данных для оценки производительности модели во время обучения.

### 4. Оценка и тестирование
- **Метрики оценки**: Выбрать метрики, такие как IoU (Intersection over Union) и Dice Coefficient.
- **Тестирование на новых данных**: Проверить модель на дополнительных изображениях для оценки ее обобщающей способности.

### 5. Развертывание и интеграция
- **Сохранение модели**: Экспортировать обученную модель для последующего использования.
- **Интеграция в приложение**: Разработать интерфейс для использования модели в реальном времени (при необходимости).

## Код для обработки областей изображения из JSON

Предположим, что JSON-файл имеет следующую структуру (пример):

```json
{
  "shapes": [
    {
      "label": "vein",
      "points": [[x1, y1], [x2, y2], ..., [xn, yn]],
      "shape_type": "polygon"
    }
  ],
  "imagePath": "image1.jpg",
  "imageHeight": 512,
  "imageWidth": 512
}
```

Код для преобразования разметки из JSON в маски:

```python
import json
import numpy as np
from PIL import Image, ImageDraw
import os

def json_to_mask(json_path, mask_save_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    image_height = data['imageHeight']
    image_width = data['imageWidth']
    
    mask = Image.new('L', (image_width, image_height), 0)
    draw = ImageDraw.Draw(mask)
    
    for shape in data['shapes']:
        if shape['shape_type'] == 'polygon':
            points = [tuple(point) for point in shape['points']]
            draw.polygon(points, outline=1, fill=1)
    
    mask = np.array(mask, dtype=np.uint8)
    mask = mask * 255  # Преобразование в диапазон [0, 255]
    
    mask_image = Image.fromarray(mask)
    mask_filename = os.path.splitext(os.path.basename(json_path))[0] + '_mask.png'
    mask_image.save(os.path.join(mask_save_path, mask_filename))

# Пример использования
json_to_mask('annotations/image1.json', 'masks/')
```

## Рекомендуемый пайплайн для обучения сети с использованием MLflow и PyTorch

### Шаг 1: Импорт необходимых библиотек

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import mlflow
import mlflow.pytorch
```

### Шаг 2: Определение кастомного Dataset

```python
class VeinDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_list = sorted(os.listdir(images_dir))
        self.masks_list = sorted(os.listdir(masks_dir))
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.images_list[idx])
        mask_path = os.path.join(self.masks_dir, self.masks_list[idx])
        
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask
```

### Шаг 3: Определение модели (например, U-Net)

```python
class UNet(nn.Module):
    # Определение слоев U-Net
    def __init__(self):
        super(UNet, self).__init__()
        # ...

    def forward(self, x):
        # ...
        return x
```

### Шаг 4: Настройка обучения с MLflow

```python
def train_model():
    mlflow.set_experiment('Vein Segmentation')

    with mlflow.start_run():
        # Инициализация модели, функции потерь и оптимизатора
        model = UNet()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Логирование параметров
        mlflow.log_param('learning_rate', 0.001)
        mlflow.log_param('optimizer', 'Adam')

        num_epochs = 25
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for images, masks in dataloader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(dataloader)
            mlflow.log_metric('loss', epoch_loss, step=epoch)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        
        # Сохранение модели
        mlflow.pytorch.log_model(model, 'model')

# Запуск обучения
train_model()
```

### Шаг 5: Оценка модели

```python
def evaluate_model(model, dataloader):
    model.eval()
    iou_scores = []
    with torch.no_grad():
        for images, masks in dataloader:
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5
            iou = compute_iou(preds, masks)
            iou_scores.append(iou)
    mean_iou = sum(iou_scores) / len(iou_scores)
    print(f'Mean IoU: {mean_iou:.4f}')
    mlflow.log_metric('mean_iou', mean_iou)
```

### Шаг 6: Определение функции вычисления метрики IoU

```python
def compute_iou(preds, masks):
    intersection = (preds & masks).float().sum((1, 2))
    union = (preds | masks).float().sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()
```

### Шаг 7: Основной блок запуска

```python
if __name__ == '__main__':
    # Настройка даталоадеров
    train_dataset = VeinDataset('images/train/', 'masks/train/', transform=...)
    val_dataset = VeinDataset('images/val/', 'masks/val/', transform=...)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Обучение модели
    train_model()
    
    # Загрузка лучшей модели и оценка
    model = mlflow.pytorch.load_model('runs:/{run_id}/model')
    evaluate_model(model, val_loader)
```

## Заключение

Предложенный план и код предоставляют основу для разработки системы семантической сегментации вен на изображениях руки. Использование PyTorch для модели и MLflow для отслеживания экспериментов позволит эффективно обучить и оценить нейронную сеть, а также упростит процесс воспроизводимости и последующего развертывания модели.
