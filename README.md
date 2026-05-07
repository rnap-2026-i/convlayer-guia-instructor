# Capa Convolucional
## ¿Qué es una capa convolucional?

Una **capa convolucional** (Conv Layer) es el bloque fundamental de las redes neuronales convolucionales (CNN). En lugar de conectar cada neurona a toda la entrada (como en una capa densa), aplica un pequeño filtro —llamado **kernel**— que se desliza sobre la imagen y detecta patrones locales: bordes, texturas, esquinas.

La idea clave es que un borde en la esquina superior izquierda y un borde en el centro de la imagen son *el mismo tipo de patrón*. El kernel lo detecta en ambas posiciones usando los **mismos pesos** — esto se conoce como *compartición de parámetros* y hace que las CNN sean mucho más eficientes que las capas densas para datos con estructura espacial.

---

## La operación de convolución

Dado un mapa de entrada `I` y un kernel `K` de tamaño `m × n`, la salida en la posición `(i, j)` es:

```
(I * K)[i, j] = Σ_p Σ_q  I[i + p, j + q] · K[p, q]
```

Para una imagen en color (3 canales RGB), la convolución suma la contribución de cada canal:

```
O[i, j] = Σ_c Σ_p Σ_q  I[c, i + p, j + q] · K[c, p, q]  +  b
```

donde `b` es el sesgo (bias) del filtro. El resultado `O` es un **mapa de características** (feature map).

---

## Hiperparámetros clave

### Tamaño del kernel (`kernel_size`)
Define el área de la imagen que el filtro "ve" a la vez. Un kernel 3×3 es el más común.

### Número de filtros (`out_channels`)
Cada filtro aprende a detectar un patrón distinto. Si usamos 32 filtros, la salida tendrá 32 mapas de características.

### Stride
Cuántos píxeles se desplaza el kernel en cada paso. `stride=1` produce una salida casi del mismo tamaño que la entrada; `stride=2` reduce las dimensiones espaciales a la mitad.

### Padding
Añade píxeles (generalmente ceros) alrededor de la imagen para controlar el tamaño de la salida. Con `padding=1` y `kernel_size=3`, la salida tiene el mismo tamaño espacial que la entrada.

**Fórmula del tamaño de salida:**

```
H_out = floor((H_in + 2·padding - kernel_size) / stride) + 1
```

**Ejemplo:** imagen 32×32, kernel 3×3, padding=1, stride=1
```
H_out = floor((32 + 2·1 - 3) / 1) + 1 = 32
```

---

## `torch.nn.Conv2d` en PyTorch

```python
import torch.nn as nn

# Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
capa = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
```

- `in_channels`: canales de la entrada (3 para RGB, 1 para escala de grises).
- `out_channels`: número de filtros = número de feature maps producidos.
- El tensor de entrada tiene forma `(batch, channels, height, width)`.

---

## Dataset: CIFAR-10

CIFAR-10 contiene 60 000 imágenes de 32×32 píxeles en 3 canales RGB, distribuidas en 10 clases (avión, automóvil, pájaro, gato, ciervo, perro, rana, caballo, barco, camión). Es ideal para experimentar con CNNs porque es pequeño pero no trivial.

```python
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2023, 0.1994, 0.2010)),
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
test_set  = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=64, shuffle=False)
```

---

## Explorar el DataLoader: visualizar un batch

El DataLoader es iterable: cada vez que se le pide un elemento entrega un **batch** de imágenes y sus etiquetas ya apiladas en tensores. La forma más directa de obtener el primer batch es:

```python
images, labels = next(iter(train_loader))

print(images.shape)   # torch.Size([64, 3, 32, 32])
print(labels.shape)   # torch.Size([64])
print(labels[:8])     # tensor([6, 9, 9, 4, 1, 1, 2, 7]) — índices de clase
```

Los ejes del tensor de imágenes son `(batch, canales, alto, ancho)`. Eso significa que `images[0]` es una sola imagen de forma `(3, 32, 32)`, con los píxeles ya normalizados, por lo que hay que revertir la normalización antes de mostrarlos:

```python
import numpy as np
import matplotlib.pyplot as plt

CLASES = ['avión', 'auto', 'pájaro', 'gato', 'ciervo',
          'perro', 'rana', 'caballo', 'barco', 'camión']

mean = np.array([0.4914, 0.4822, 0.4465])
std  = np.array([0.2023, 0.1994, 0.2010])

def desnormalizar(tensor):
    """Convierte un tensor (3, H, W) normalizado a imagen numpy (H, W, 3) en [0,1]."""
    img = tensor.permute(1, 2, 0).numpy()   # (H, W, 3)
    img = img * std + mean
    return np.clip(img, 0, 1)
```

### Cuadrícula de imágenes del batch

```python
images, labels = next(iter(train_loader))

fig, axes = plt.subplots(4, 8, figsize=(14, 7))
for i, ax in enumerate(axes.flat):
    ax.imshow(desnormalizar(images[i]))
    ax.set_title(CLASES[labels[i]], fontsize=8)
    ax.axis('off')

plt.suptitle(f'Primeras 32 imágenes del batch (batch_size={len(images)})')
plt.tight_layout()
plt.show()
```

Cada llamada a `next(iter(train_loader))` devuelve imágenes distintas porque el DataLoader fue creado con `shuffle=True`. Ejecutar la celda varias veces muestra combinaciones diferentes, lo que refleja cómo la red ve datos nuevos en cada época.

### Distribución de clases en el batch

Una verificación útil es comprobar que el batch no esté sesgado hacia pocas clases:

```python
import torch

conteo = torch.bincount(labels, minlength=10)

plt.figure(figsize=(8, 3))
plt.bar(CLASES, conteo.numpy())
plt.title('Número de imágenes por clase en el batch')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()
```

Con `batch_size=64` y `shuffle=True` se espera aproximadamente 6–7 imágenes por clase (distribución uniforme con algo de varianza aleatoria).

---

## Arquitectura de la red

```
Entrada: (batch, 3, 32, 32)

Conv2d(3  → 32, 3×3, padding=1) → ReLU → MaxPool2d(2×2)   →  (batch, 32, 16, 16)
Conv2d(32 → 64, 3×3, padding=1) → ReLU → MaxPool2d(2×2)   →  (batch, 64,  8,  8)
Conv2d(64 → 128, 3×3, padding=1) → ReLU → MaxPool2d(2×2)  →  (batch, 128,  4,  4)

Flatten                                                       →  (batch, 2048)
Linear(2048 → 256) → ReLU
Linear(256  → 10)                                            →  logits por clase
```

Cada bloque `Conv → ReLU → MaxPool` extrae características cada vez más abstractas:
- **Bloque 1:** bordes y gradientes de color.
- **Bloque 2:** texturas y formas simples.
- **Bloque 3:** partes de objetos.

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

---

## Entrenamiento

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model     = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()

    n = len(loader.dataset)
    return total_loss / n, correct / n

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * images.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n

EPOCHS = 10
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    test_loss,  test_acc  = evaluate(model, test_loader, criterion)
    print(f"Epoch {epoch:2d} | "
          f"Train loss: {train_loss:.4f}  acc: {train_acc:.3f} | "
          f"Test  loss: {test_loss:.4f}  acc: {test_acc:.3f}")
```

Con 10 épocas y sin data augmentation se espera ~70–72 % de accuracy en el conjunto de prueba.

---

## Visualizar los feature maps

Una forma de entender qué aprende la red es visualizar la salida de la primera capa convolucional ante una imagen real.

```python
import matplotlib.pyplot as plt

model.eval()
images, _ = next(iter(test_loader))
img = images[0:1].to(device)           # una sola imagen: (1, 3, 32, 32)

with torch.no_grad():
    feature_maps = model.features[0](img)   # salida de Conv2d(3→32)
    feature_maps = torch.relu(feature_maps) # aplicar ReLU

feature_maps = feature_maps.squeeze(0).cpu()  # (32, 16, 16) después del pool

fig, axes = plt.subplots(4, 8, figsize=(14, 7))
for i, ax in enumerate(axes.flat):
    ax.imshow(feature_maps[i], cmap='viridis')
    ax.axis('off')
plt.suptitle('Feature maps — primera capa convolucional')
plt.tight_layout()
plt.show()
```

Cada panel muestra qué "activa" uno de los 32 filtros: algunos responderán a bordes horizontales, otros a bordes diagonales, otros a zonas de color específico.

---

## Requisitos

```
torch>=2.0
torchvision>=0.15
matplotlib
```

Instalar con:

```bash
pip install torch torchvision matplotlib
```
