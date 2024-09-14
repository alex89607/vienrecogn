## Код для преобразования разметки из JSON в маски

необходимо адаптировать код для обработки предоставленной структуры данных. 
В  JSON-файле разметка хранится в виде сжатого RLE (Run-Length Encoding), закодированного в массиве чисел.

Ниже приведен код, который:

- **Читает JSON-файл с разметкой**.
- **Извлекает RLE-данные** и распаковывает их.
- **Преобразует RLE в маску изображения**.
- **Сохраняет маску в виде PNG-файла**.

### Установка необходимых библиотек

Убедитесь, что у вас установлены следующие библиотеки:

```bash
pip install numpy pillow pycocotools
```

### Код

```python
import json
import numpy as np
from PIL import Image
import os
import zlib
from pycocotools import mask as maskutils

def decode_rle(rle_array, image_shape):
    """
    Декодирует RLE-массив в бинарную маску.

    Args:
        rle_array (list of int): RLE-данные в виде списка чисел.
        image_shape (tuple): Размер изображения в формате (height, width).

    Returns:
        np.ndarray: Бинарная маска.
    """
    try:
        # Преобразуем список чисел в байты
        rle_bytes = bytes(rle_array)
        # Распаковываем данные с помощью zlib
        decompressed_bytes = zlib.decompress(rle_bytes)
        # Декодируем байты в строку
        rle_string = decompressed_bytes.decode('utf-8')
        # Преобразуем строку в список целых чисел
        counts = [int(s) for s in rle_string.split()]
        # Создаем RLE-объект
        rle = {'counts': counts, 'size': list(image_shape)}
        # Декодируем маску с помощью pycocotools
        mask = maskutils.decode(rle)
        return mask
    except Exception as e:
        print(f"Ошибка при декодировании RLE: {e}")
        return None

def json_to_masks(json_file, mask_save_path):
    """
    Преобразует разметку из JSON-файла в маски и сохраняет их.

    Args:
        json_file (str): Путь к JSON-файлу с разметкой.
        mask_save_path (str): Директория для сохранения масок.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if not os.path.exists(mask_save_path):
        os.makedirs(mask_save_path)

    # Обходим каждый элемент в JSON
    for task in data:
        task_id = task['id']
        annotations = task.get('annotations', [])
        for annotation in annotations:
            annotation_id = annotation['id']
            results = annotation.get('result', [])
            for res in results:
                if res['type'] == 'brushlabels':
                    value = res['value']
                    rle_array = value.get('rle')
                    if not rle_array:
                        continue
                    width = res.get('original_width')
                    height = res.get('original_height')
                    if not width or not height:
                        continue
                    image_shape = (height, width)
                    # Декодируем RLE в маску
                    mask = decode_rle(rle_array, image_shape)
                    if mask is None:
                        continue
                    # Преобразуем маску в изображение
                    mask_image = Image.fromarray(mask * 255).convert('L')
                    # Формируем имя файла
                    mask_filename = f"task_{task_id}_annotation_{annotation_id}_result_{res['id']}_mask.png"
                    mask_filepath = os.path.join(mask_save_path, mask_filename)
                    # Сохраняем маску
                    mask_image.save(mask_filepath)
                    print(f"Маска сохранена: {mask_filepath}")

# Пример использования
json_file = 'path/to/your/annotations.json'  # Укажите путь к вашему JSON-файлу
mask_save_path = 'path/to/save/masks'       # Укажите директорию для сохранения масок

json_to_masks(json_file, mask_save_path)
```

### Объяснение кода

1. **Функция `decode_rle`**: 
    - Преобразует список чисел `rle_array` в байтовый формат.
    - Распаковывает байты с помощью `zlib.decompress`.
    - Декодирует байты в строку и разбивает ее на отдельные числа, получая список `counts`.
    - Создает объект `rle` с ключами `counts` и `size`.
    - Декодирует маску с помощью `maskutils.decode` из библиотеки `pycocotools`.

2. **Функция `json_to_masks`**:
    - Читает JSON-файл и обходит каждый элемент разметки.
    - Извлекает необходимые данные: `rle_array`, `width`, `height`.
    - Вызывает функцию `decode_rle` для получения маски.
    - Сохраняет маску как PNG-файл в указанной директории.

### Пример вывода

```
Маска сохранена: path/to/save/masks/task_128242458_annotation_43382384_result_K_hHZ8u7FH_mask.png
Маска сохранена: path/to/save/masks/task_128242458_annotation_43382384_result_Tx-atyfX0A_mask.png
Маска сохранена: path/to/save/masks/task_128242458_annotation_43382384_result_Ji-p_yoZOp_mask.png
Маска сохранена: path/to/save/masks/task_128242459_annotation_43382452_result_X78zH6e3tQ_mask.png
```

### Зависимости

- **Python 3.x**
- **numpy**
- **Pillow (PIL)**
- **pycocotools**

Установите зависимости, если они еще не установлены:

```bash
pip install numpy pillow pycocotools
```

### Дополнительные замечания

- **Обработка ошибок**: Код содержит обработку исключений при декодировании RLE. Если возникает ошибка, она будет выведена в консоль, и программа продолжит обработку следующих элементов.
- **Пути файлов**: Убедитесь, что пути к JSON-файлу и директории для сохранения масок указаны корректно.
- **Формат изображения**: Маски сохраняются в формате PNG с 8-битной глубиной цвета (градации серого).

## Заключение

Предоставленный код позволяет преобразовать разметку из вашего специфического JSON-файла в маски изображений. Это позволит вам подготовить данные для обучения модели семантической сегментации, используя предоставленные разметки.
