# Capítulo 2: Machine Learning

### Introdução ao Machine Learning

O Machine Learning (ML), ou Aprendizado de Máquina, é um subcampo da Inteligência Artificial que se concentra no desenvolvimento de algoritmos que podem aprender com dados e fazer previsões ou decisões sem serem explicitamente programados para cada tarefa específica. Em vez de seguir regras rígidas, os modelos de ML identificam padrões nos dados e generalizam para novos exemplos.

O termo foi cunhado em 1959 por Arthur Samuel, um pioneiro em jogos de computador e IA. Desde então, o ML evoluiu de abordagens simbólicas para métodos baseados em estatística e probabilidade, impulsionado pelo aumento de dados digitais e poder computacional.

### Tipos de Machine Learning

Existem três categorias principais de ML, baseadas na natureza do sinal ou feedback disponível para o sistema de aprendizado:

1. **Aprendizado Supervisionado**: O algoritmo recebe exemplos de entradas e saídas desejadas (rótulos). O objetivo é aprender uma regra geral que mapeie entradas para saídas. Exemplos incluem classificação (e.g., detectar spam em emails) e regressão (e.g., prever preços de imóveis).

2. **Aprendizado Não Supervisionado**: Não há rótulos fornecidos. O algoritmo identifica padrões ou estruturas nos dados, como agrupamentos (clustering) ou redução de dimensionalidade.

3. **Aprendizado por Reforço**: O agente interage com um ambiente dinâmico, recebendo recompensas ou punições para aprender a maximizar um objetivo, como em jogos ou robôs autônomos.

### Conceitos Fundamentais

- **Generalização**: A capacidade de um modelo de performar bem em dados não vistos, após treinado em um conjunto de dados.
- **Overfitting e Underfitting**: Overfitting ocorre quando o modelo aprende detalhes específicos dos dados de treinamento, incluindo ruído, resultando em baixa performance em dados novos. Underfitting acontece quando o modelo é muito simples para capturar os padrões.
- **Viés-Variância**: Trade-off entre viés (erro sistemático) e variância (sensibilidade a variações nos dados).

### Exemplos Práticos

Um exemplo clássico de ML supervisionado é a regressão linear, usada para prever valores contínuos. Vamos ver um exemplo simples em Python usando a biblioteca scikit-learn, inspirado em repositórios como o scikit-learn (um projeto open-source amplamente usado).

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Gerar dados sintéticos: y = 2*x + 1 + ruído
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2 * X.ravel() + 1 + np.random.randn(100) * 2

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar modelo de regressão linear
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Fazer previsões
y_pred = modelo.predict(X_test)

# Avaliar erro
mse = mean_squared_error(y_test, y_pred)
print(f"Erro Quadrático Médio: {mse:.2f}")
print(f"Coeficiente: {modelo.coef_[0]:.2f}, Intercepto: {modelo.intercept_:.2f}")
```

Este código gera dados lineares com ruído, treina um modelo de regressão linear e avalia seu desempenho. O coeficiente aproximado deve ser próximo de 2, e o intercepto próximo de 1.

### Aplicações do Machine Learning

O ML é usado em diversas áreas:
- **Visão Computacional**: Reconhecimento de imagens e objetos.
- **Processamento de Linguagem Natural**: Tradução automática e chatbots.
- **Medicina**: Diagnóstico assistido e descoberta de medicamentos.
- **Finanças**: Detecção de fraudes e previsões de mercado.
- **Transportes**: Veículos autônomos.

### Limitações e Considerações Éticas

Embora poderoso, o ML enfrenta desafios como viés nos dados (levando a decisões discriminatórias), falta de interpretabilidade ("caixa preta") e necessidade de grandes volumes de dados de qualidade. Questões éticas incluem privacidade, transparência e impacto social.

### Referências

- Mitchell, T. M. (1997). Machine Learning. McGraw-Hill.
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
- Repositório scikit-learn: https://github.com/scikit-learn/scikit-learn (exemplos de regressão linear).

### Fechamento do Capítulo

Machine Learning introduz a lógica de aprender a partir de dados, mas o ponto central para engenharia está em entender limites de generalização, critérios de avaliação e implicações de negócio. Esse repertório será reutilizado em todos os capítulos seguintes.

