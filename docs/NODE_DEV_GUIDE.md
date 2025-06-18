# 🧠 Гайд по добавлению нод в NEUDC

Этот гайд объясняет, как добавлять новые ноды в пайплайн NEUDC, в том числе и потоковые (`BaseThreadedNode`), и процессные (`BaseProcessNode`). Взаимодействие между нодами осуществляется через единый интерфейс `mailbox`.

---

## 🧱 Архитектура Ноды

Все ноды должны наследоваться от одного из базовых классов:

- `BaseThreadedNode` — если нода работает в потоке (внутри одного процесса)
- `BaseProcessNode` — если нода должна запускаться как отдельный процесс (например, для инференса)

Нода также наследует **LogicMixin**, где реализуется логика метода `process()` (например, `ResizeLogicMixin`).

---

## 🧩 Как добавить новую ноду

### 1. Создай mixin с логикой

Пример: `core/node/processors/mixins/normalize_mixin.py`

```python
class NormalizeLogicMixin:
    def process(self, frame: Frame) -> Frame:
        frame.image = frame.image / 255.0
        return frame
```

### 2. Определи класс ноды

Выбери базовый класс в зависимости от типа:

```python
# Для потоковой ноды:
class NormalizeNode(NormalizeLogicMixin, BaseThreadedNode):
    ...

# Для процессной ноды:
class NormalizeProcessNode(NormalizeLogicMixin, BaseProcessNode):
    ...
```

Реализуй `__init__` и `from_config()`:

```python
@staticmethod
def from_config(config: dict[str, Any]) -> NormalizeNode:
    return NormalizeNode(
        mailbox=config["mailbox"],
        logger=config["logger"]
    )
```

📌 Можно передавать любые параметры в конфиге, включая пути, числа, структуры.

---

## 🏭 Зарегистрируй в фабрике

Добавь класс в `NODE_CLASS_MAP` внутри `NodeFactory`:

```python
NODE_CLASS_MAP = {
    "NormalizeNode": NormalizeNode,
    "NormalizeProcessNode": NormalizeProcessNode,
    # другие ноды...
}
```

---

## 🛠️ Опиши конфигурацию в YAML

Каждая нода описывается в `pipeline.yaml`:

```yaml
- id: normalize
  type: NormalizeNode
  outputs: [next_node_id]
```

Можно передавать любые дополнительные поля в конфиг — они попадут в `from_config()`.

---

## 🧠 Инференс-ноды

Для нод инференса:

- Наследуй от `BaseProcessInference`
- Реализуй `postprocess_result(result, frame)`

Пример:

```python
class MyModelNode(BaseProcessInference):
    def postprocess_result(self, result, item: Frame) -> Frame:
        # добавь боксы или метаданные в item
        return item
```

В фабрике нужно зарегистрировать через NODE\_CLASS\_MAP.

📌 Конфигурация модели передаётся в параметре `model_config` в YAML-файле:

```yaml
model_config:
  name: yolov8
  weights_path: ./models/yolov8.pt
```


---

## 🧪 Пример pipeline.yaml

```yaml
nodes:
  - id: reader
    type: FolderImageNode
    folder_path: "./test_images"
    mode: "loop"
    frame_delay: 0.5
    outputs: [model]

  - id: model
    type: ProcessDetInference
    model_config:
      type: YOLOv8
      path: "./models/yolov8n.torchscript"
      backend: TorchBackend
      device_id: -1
    outputs: [visualizer]

  - id: visualizer
    type: DrawNode
    outputs: [saver]

  - id: saver
    type: SaveImageNode
    save_dir: "./output_images"
    outputs: []
```

---
