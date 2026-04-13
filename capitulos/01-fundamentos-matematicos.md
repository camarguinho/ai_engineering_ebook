# Capítulo 1: Fundamentos Matemáticos

A Inteligência Artificial (IA) é construída sobre bases matemáticas sólidas. Este capítulo explora os conceitos matemáticos essenciais que fundamentam algoritmos de IA, incluindo otimização, cálculo, álgebra linear e probabilidade. Esses fundamentos permitem entender como os modelos aprendem e fazem previsões.

### 1.1 Otimização Matemática

A otimização matemática é o coração da IA, pois muitos algoritmos de aprendizado de máquina envolvem encontrar o melhor conjunto de parâmetros para minimizar ou maximizar uma função objetivo.

#### Conceitos Principais
- **Problema de Otimização**: Encontrar o valor ótimo de uma função f(x) em um conjunto A.
  - Minimização: f(x0) ≤ f(x) para todo x ∈ A
  - Maximização: f(x0) ≥ f(x) para todo x ∈ A
- **Pontos Críticos**: Pontos onde o gradiente é zero ou indefinido.
- **Convexidade**: Funções convexas têm um mínimo global único, facilitando a otimização.

#### Exemplos Práticos
Considere a regressão linear: minimizamos o erro quadrático médio (MSE) entre previsões e valores reais. Isso é um problema de otimização convexa resolvido por gradiente descendente.

```python
import numpy as np

# Exemplo simples de gradiente descendente para y = mx + b
def gradient_descent(X, y, m, b, learning_rate, iterations):
    for _ in range(iterations):
        y_pred = m * X + b
        dm = -2 * np.mean(X * (y - y_pred))
        db = -2 * np.mean(y - y_pred)
        m -= learning_rate * dm
        b -= learning_rate * db
    return m, b
```

#### Considerações Avançadas
Em problemas não convexos, como redes neurais profundas, algoritmos como Adam ou RMSProp são usados para escapar de mínimos locais.

### 1.2 Cálculo Diferencial

O cálculo é usado para entender taxas de mudança e derivadas, essenciais para algoritmos de aprendizado.

#### Conceitos Principais
- **Derivada**: Taxa de variação instantânea.
- **Regra da Cadeia**: Para funções compostas, d/dx f(g(x)) = f'(g(x)) * g'(x).
- **Gradiente**: Vetor de derivadas parciais.

#### Exemplos Práticos
No backpropagation de redes neurais, usamos a regra da cadeia para calcular gradientes e atualizar pesos.

### 1.3 Álgebra Linear

Vetores, matrizes e operações lineares são fundamentais para representar dados e transformações.

#### Conceitos Principais
- **Vetores e Matrizes**: Representam dados multidimensionais.
- **Produto Escalar e Vetorial**.
- **Autovalores e Autovetores**: Usados em PCA para redução de dimensionalidade.

#### Exemplos Práticos
Em visão computacional, imagens são matrizes, e convoluções são operações matriciais.

### 1.4 Probabilidade e Estatística

A IA lida com incerteza, então probabilidade é crucial.

#### Conceitos Principais
- **Distribuições**: Normal, Binomial, etc.
- **Teorema de Bayes**: P(A|B) = P(B|A) P(A) / P(B)
- **Esperança e Variância**.

#### Exemplos Práticos
Em classificação, usamos probabilidade para prever classes.

Este capítulo estabelece a base matemática. Os próximos capítulos constroem sobre esses conceitos, aplicando-os em Machine Learning e além.

### Fechamento do Capítulo

Sem fundamentos matemáticos, a Engenharia de IA vira apenas operação de ferramentas. Com eles, você ganha capacidade de diagnosticar comportamento de modelos, escolher abordagens com mais critério e discutir trade-offs de forma tecnicamente defensável.

