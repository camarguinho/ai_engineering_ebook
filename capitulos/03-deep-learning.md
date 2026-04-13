# Capítulo 3: Deep Learning

### Introdução ao Deep Learning

O Deep Learning é uma subárea do Machine Learning que utiliza redes neurais artificiais com múltiplas camadas (deep neural networks) para modelar e resolver problemas complexos. Inspirado no funcionamento do cérebro humano, o Deep Learning permite que os modelos aprendam representações hierárquicas dos dados, extraindo características de baixo nível para alto nível automaticamente.

Ao contrário do Machine Learning tradicional, onde as características são engenheiradas manualmente, o Deep Learning descobre essas características diretamente dos dados. Isso é possível graças a arquiteturas com muitas camadas, onde cada camada transforma a entrada em uma representação mais abstrata.

### Conceitos Fundamentais

#### Redes Neurais Artificiais (ANNs)

Uma rede neural artificial é composta por:
- **Neurônios**: Unidades básicas que recebem entradas, aplicam uma função de ativação e produzem uma saída.
- **Camadas**: Grupos de neurônios organizados em camadas de entrada, ocultas e saída.
- **Pesos e Bias**: Parâmetros ajustáveis durante o treinamento que determinam a força das conexões.

#### Funções de Ativação

Funções que introduzem não-linearidade no modelo:
- **ReLU (Rectified Linear Unit)**: f(x) = max(0, x) – popular por evitar o problema do gradiente vanishing.
- **Sigmoid**: f(x) = 1 / (1 + e^-x) – usada para saídas binárias.
- **Tanh**: f(x) = (e^x - e^-x) / (e^x + e^-x) – similar ao sigmoid, mas centrada em zero.

#### Backpropagation e Otimização

O treinamento de redes profundas usa o algoritmo de backpropagation para calcular gradientes e atualizar pesos. Otimizadores como Adam, SGD (Stochastic Gradient Descent) e RMSProp ajustam os pesos para minimizar a função de perda.

#### Arquiteturas Comuns

- **Redes Neurais Convolucionais (CNNs)**: Ideais para dados de imagem, usam convoluções para detectar padrões locais.
- **Redes Neurais Recorrentes (RNNs)**: Adequadas para sequências, como texto ou séries temporais.
- **Transformers**: Baseados em atenção, revolucionaram o processamento de linguagem natural.

### Aplicações do Deep Learning

- **Visão Computacional**: Reconhecimento de imagens, detecção de objetos.
- **Processamento de Linguagem Natural (NLP)**: Tradução automática, geração de texto.
- **Jogos e Simulação**: Agentes inteligentes, como AlphaGo.
- **Medicina**: Análise de imagens médicas, descoberta de fármacos.

### Desafios

- **Overfitting**: Modelos muito complexos se ajustam demais aos dados de treinamento.
- **Dados e Computação**: Requer grandes volumes de dados e poder computacional (GPUs).
- **Interpretabilidade**: "Caixa preta" – difícil entender decisões do modelo.

### Exemplo Prático: Rede Neural Simples com TensorFlow

Aqui está um exemplo de uma rede neural convolucional simples para classificação de dígitos MNIST usando TensorFlow. Este código demonstra como construir e treinar um modelo de Deep Learning básico.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Carregar o dataset MNIST
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalizar as imagens
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Adicionar dimensão de canal
train_images = train_images[..., tf.newaxis]
test_images = test_images[..., tf.newaxis]

# Definir o modelo CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treinar o modelo
model.fit(train_images, train_labels, epochs=5, batch_size=64)

# Avaliar o modelo
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Acurácia no teste: {test_acc}')
```

Este exemplo ilustra como o Deep Learning pode ser aplicado para tarefas de classificação de imagens, alcançando alta precisão com relativamente pouco código.

### Referências

- LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.
- Coursera: Neural Networks and Deep Learning (Andrew Ng).

### Fechamento do Capítulo

Deep Learning amplia drasticamente a capacidade dos sistemas, mas também aumenta custo, opacidade e dependência de infraestrutura. Por isso, o engenheiro forte nessa área precisa pensar simultaneamente em arquitetura de modelo e arquitetura de sistema.

