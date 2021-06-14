# White box attacks

Para los ataques de *caja blanca* se generaron ejemplos a la medida usando los modelos pre-entrenados de [Torchvision models](https://pytorch.org/vision/stable/models.html). Posteriormente se calculó el accuracy@1 y accuracy@5 de cada modelo usando el dataset limpio y con los ejemplos adversarios generados con cada algoritmo. En la tabla de abajo se observan el accuracy@1/accuracy@5 de cada modelo con el respectivo ataque de caja blanca.

|  **Modelo**  |  **Limpio** |   **FGSM**  |   **PGD**  | **MIFGSM** | **OnePixel** |
|:------------:|:-----------:|:-----------:|:----------:|:----------:|:------------:|
|    AlexNet   | 60.9 / 84.6 |  6.0 / 28.6 | 2.9 / 19.3 | 3.5 / 21.0 |  58.3 / 83.4 |
|   ResNet-18  | 82.5 / 95.4 |  3.5 / 24.3 | 0.8 / 14.6 | 1.0 / 13.5 |  78.7 / 94.5 |
| Inception v3 | 76.5 / 93.1 | 10.8 / 39.3 | 3.9 / 28.1 | 5.1 / 27.6 |  70.1 / 91.8 |
| MobileNet v2 | 85.0 / 97.3 |  3.5 / 24.4 | 0.5 / 11.3 |  0.6 / 8.8 |  81.4 / 96.6 |

<!-- Todo gordito y bonito el colibrí -->
<img src='../../Presentación/Images/hummingbird_alexnet_FGSM.png'>

# Black box attack to MobileNet v2

Para el ataque de *caja negra* generé los ejemplos adversarios con Inception v3 y los probé en MobileNet v2 (el cual funcionaba como mi modelo de caja negra). En este caso fui más agresivo con los ataques al cambiar algunos de los parámetros de cada algoritmo, lo que se nota en las imágenes. En la tabla de abajo se observan el accuracy@1/accuracy@5 con el respectivo ataque de caja negra a MobileNet v2.

|  **Modelo**  |  **Limpio** |   **FGSM**  |   **PGD**   |  **MIFGSM** | **OnePixel** |
|:------------:|:-----------:|:-----------:|:-----------:|:-----------:|:------------:|
| MobileNet v2 | 85.0 / 97.3 | 51.7 / 78.3 | 59.4 / 82.9 | 56.3 / 81.5 |  83.5 / 97.0 |

<!-- Todo gordito y bonito el colibrí -->
<img src='../../Presentación/Images/hummingbird_blackbox_FGSM.png'>git 
