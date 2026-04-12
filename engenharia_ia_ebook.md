# Engenharia de IA: Uma Base Sólida

## Prefácio

Engenharia de IA é, cada vez menos, um campo exclusivamente acadêmico e, cada vez mais, uma disciplina de construção de sistemas. Saber treinar modelos, conhecer artigos ou dominar bibliotecas isoladas já não basta. O diferencial prático está em conectar fundamentos matemáticos, engenharia de software, avaliação, infraestrutura, segurança e operação contínua em ambientes reais.

Este eBook foi escrito para profissionais que querem construir essa visão integrada. O objetivo não é apenas explicar o que são modelos, embeddings, agentes ou RAG, mas mostrar como essas peças se encaixam em uma arquitetura de produto e em uma rotina de engenharia.

Ao longo dos capítulos, a leitura progride do núcleo conceitual para a camada operacional. A recomendação é estudar cada capítulo com duas perguntas em mente: "que problema esta técnica resolve?" e "como eu a colocaria em produção com segurança, observabilidade e custo aceitável?". Essa mudança de perspectiva é o que separa consumo de tecnologia de engenharia de tecnologia.

## Introdução

Este eBook oferece uma base sólida em Engenharia de IA, cobrindo tópicos essenciais de forma evolutiva e natural. Começamos com os fundamentos matemáticos que sustentam a IA, progredindo para conceitos de Machine Learning, Deep Learning e avanços em engenharia como MLOps e segurança. Cada capítulo é escrito em nível intermediário, com explicações conceituais, exemplos práticos e referências para aprofundamento.

O conteúdo é organizado para aprendizado progressivo: dos princípios matemáticos básicos até aplicações práticas em engenharia, permitindo uma compreensão holística da IA como ferramenta de engenharia.

### Estrutura do eBook

1. [Fundamentos Matemáticos](#capitulo-1)
2. [Machine Learning](#capitulo-2)
3. [Deep Learning](#capitulo-3)
4. [Transformers e NLP Moderno](#capitulo-4)
5. [Prompt Engineering](#capitulo-5)
6. [Fine-Tuning](#capitulo-6)
7. [RAG (Retrieval-Augmented Generation)](#capitulo-7)
8. [Agentes e Uso de Ferramentas](#capitulo-8)
9. [Infraestrutura para IA](#capitulo-9)
10. [MLOps, Avaliação e Observabilidade](#capitulo-10)
11. [Segurança, Governança e IA Responsável](#capitulo-11)
12. [Repositórios GitHub Essenciais](#capitulo-12)

<a id="capitulo-1"></a>
## Capítulo 1: Fundamentos Matemáticos

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

<a id="capitulo-2"></a>
## Capítulo 2: Machine Learning

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

<a id="capitulo-3"></a>
## Capítulo 3: Deep Learning

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

<a id="capitulo-4"></a>
# Capítulo 4: Transformers e Processamento de Linguagem Natural (NLP)

## Introdução aos Transformers e ao Processamento de Linguagem Natural

O Processamento de Linguagem Natural (NLP) é uma subárea da Inteligência Artificial que se concentra em permitir que computadores entendam, interpretem e gerem linguagem humana. Desde os primeiros experimentos na década de 1950, como o Experimento de Georgetown em 1954, que traduzia frases russas para inglês, o NLP evoluiu de abordagens simbólicas baseadas em regras para métodos estatísticos e neurais.

O salto mais importante da última década veio com a arquitetura Transformer, apresentada em 2017 no artigo "Attention Is All You Need". Em vez de processar tokens de forma estritamente sequencial, os Transformers usam mecanismos de atenção para modelar dependências de curto e longo alcance com alta paralelização. Essa arquitetura tornou-se a base de modelos como BERT, GPT, T5, Gemini, Claude e muitos sistemas multimodais modernos.

### História e Evolução

- **Década de 1950-1960**: Abordagens simbólicas com sistemas como SHRDLU (1970), que trabalhava em mundos de blocos restritos, e ELIZA (1966), um simulador de psicoterapia.
- **Década de 1970-1980**: Desenvolvimento de ontologias conceituais e sistemas baseados em regras, como MARGIE e MYCIN.
- **Década de 1990**: Revolução estatística com modelos de Markov ocultos para marcação de partes da fala e tradução automática estatística.
- **Década de 2000-presente**: Aprendizado de máquina e redes neurais profundas, com modelos como Word2vec (2013), BERT (2018) e GPT (2018+), levando a avanços em tarefas como tradução, geração de texto e compreensão.

### Abordagens Principais

1. **Simbólica**: Baseada em regras manuais e dicionários.
2. **Estatística**: Usa modelos probabilísticos e corpora de texto.
3. **Neural**: Emprega redes neurais profundas para aprender representações de linguagem.

## Fundamentos de Transformers

### Tokenização

Modelos de linguagem não operam diretamente sobre palavras ou frases, mas sobre tokens. A tokenização divide o texto em unidades menores, que podem ser palavras, subpalavras ou caracteres, dependendo do algoritmo adotado. Em LLMs, técnicas como Byte Pair Encoding (BPE) e SentencePiece são comuns porque equilibram vocabulário, cobertura e eficiência.

### Embeddings

Cada token é convertido em um vetor numérico de alta dimensionalidade. Esses vetores capturam similaridades semânticas e sintáticas e servem como ponto de entrada para o modelo. Além do embedding do token, o modelo incorpora informação de posição, já que a atenção, por si só, não conhece a ordem da sequência.

### Self-Attention

O mecanismo de self-attention permite que cada token "olhe" para outros tokens da sequência para decidir quais são mais relevantes na construção de sua representação contextual. Esse processo usa três projeções principais:

- **Query (Q)**: O que o token atual procura.
- **Key (K)**: O que cada token oferece como informação.
- **Value (V)**: O conteúdo efetivamente agregado.

Em termos intuitivos, a atenção calcula quanta influência cada token deve exercer sobre os demais. Em engenharia, isso importa porque explica comportamentos como dependência de contexto, sensibilidade a instruções e limites de janela de contexto.

### Multi-Head Attention

Em vez de usar uma única atenção, os Transformers utilizam múltiplas "cabeças" em paralelo. Cada cabeça pode capturar padrões diferentes, como relações sintáticas, co-referência, dependências de longo prazo ou alinhamentos semânticos mais abstratos.

### Camadas Feed-Forward e Normalização

Após a etapa de atenção, as representações passam por camadas feed-forward, conexões residuais e normalizações. Esse conjunto melhora estabilidade de treinamento, profundidade efetiva e capacidade expressiva do modelo.

### Treinamento Prévio e Ajuste Posterior

O paradigma dominante em NLP moderno é:

1. **Pré-treinamento**: O modelo aprende padrões gerais em grandes volumes de texto.
2. **Adaptação**: O modelo é refinado por prompting, fine-tuning, alignment, RAG ou tool use para tarefas específicas.

Essa separação é central para Engenharia de IA porque desloca o esforço do treino do zero para o desenho de sistemas sobre modelos fundacionais.

## Limitações Práticas dos Transformers

- **Custo quadrático de atenção**: A atenção tradicional cresce aproximadamente com o quadrado do comprimento da sequência.
- **Janela de contexto finita**: Mesmo modelos com contexto amplo ainda exigem estratégias de chunking, retrieval ou memória externa.
- **Alucinação**: O modelo pode gerar texto plausível, porém incorreto.
- **Sensibilidade a prompt**: Pequenas mudanças na formulação podem alterar o comportamento.
- **Atualização de conhecimento**: Conhecimento paramétrico pode envelhecer, exigindo RAG, ferramentas ou retreinamento.

## Tarefas Comuns em NLP

O NLP abrange uma variedade de tarefas, desde processamento básico de texto até aplicações avançadas:

### Processamento de Texto e Fala

- **Reconhecimento Óptico de Caracteres (OCR)**: Converte imagens de texto impresso em texto digital.
- **Reconhecimento de Fala**: Transcreve áudio em texto.
- **Síntese de Fala**: Gera áudio a partir de texto.
- **Segmentação de Palavras**: Divide texto contínuo em palavras (importante em línguas como chinês).

### Análise Morfológica

- **Lematização**: Reduz palavras à sua forma base (ex.: "correu" → "correr").
- **Stemização**: Remove sufixos para obter raízes (ex.: "correu" → "corr").
- **Marcação de Partes da Fala (POS)**: Identifica substantivos, verbos, adjetivos, etc.
- **Segmentação Morfológica**: Divide palavras em morfemas.

### Análise Sintática

- **Análise Sintática (Parsing)**: Determina a estrutura gramatical de frases.
- **Quebra de Frases**: Identifica limites de sentenças.

### Semântica Lexical

- **Reconhecimento de Entidades Nomeadas (NER)**: Identifica pessoas, locais, organizações.
- **Análise de Sentimento**: Classifica texto como positivo, negativo ou neutro.
- **Desambiguação de Sentido de Palavras (WSD)**: Escolhe o significado correto de palavras ambíguas.

### Semântica Relacional

- **Extração de Relacionamentos**: Identifica relações entre entidades (ex.: "João trabalha na Google").
- **Resolução de Correferência**: Vincula pronomes a seus referentes (ex.: "João disse que ele...").

### Aplicações de Alto Nível

- **Tradução Automática**: Traduz texto entre idiomas.
- **Geração de Linguagem Natural**: Cria texto coerente (ex.: resumos, histórias).
- **Resposta a Perguntas**: Responde perguntas baseadas em contexto.
- **Resumo Automático**: Gera resumos concisos de textos longos.

## Exemplo Prático: Classificação de Texto com Hugging Face Transformers

Vamos implementar um exemplo de classificação de sentimento usando a biblioteca Transformers do Hugging Face, que é uma das principais ferramentas para NLP moderno.

```python
from transformers import pipeline

# Carregar pipeline de análise de sentimento
classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Texto de exemplo
text = "Este produto é incrível! Funciona perfeitamente e superou minhas expectativas."

# Classificar sentimento
result = classifier(text)
print(result)
# Saída: [{'label': '5 stars', 'score': 0.9}]

# Outro exemplo
text2 = "O atendimento foi terrível e o produto chegou com defeito."
result2 = classifier(text2)
print(result2)
# Saída: [{'label': '1 star', 'score': 0.8}]
```

Este exemplo usa um modelo BERT multilíngue pré-treinado para classificar o sentimento de textos em português. O modelo atribui uma pontuação de 1 a 5 estrelas baseada no sentimento expresso.

## Desafios e Tendências Atuais

- **Ambiguidade**: Linguagem humana é ambígua; uma palavra pode ter múltiplos significados.
- **Contexto**: Significado depende de contexto cultural e situacional.
- **Multilinguismo**: Muitos sistemas focam no inglês; há necessidade de modelos multilíngues.
- **Ética**: Preocupações com viés, privacidade e geração de conteúdo falso.

Tendências recentes incluem modelos de linguagem grandes (LLMs) como GPT, Gemini e Claude, aprendizado multimodal (combinação de texto com imagens/áudio) e aplicações em saúde, direito e educação.

## Referências

- Eisenstein, Jacob. *Introduction to Natural Language Processing*. MIT Press, 2019.
- Jurafsky, Daniel; Martin, James H. *Speech and Language Processing*. Pearson, 2023.
- Vaswani, Ashish et al. *Attention Is All You Need*. NeurIPS, 2017.
- Hugging Face Documentation: https://huggingface.co/docs/transformers/index
- Wikipedia: Processamento de Linguagem Natural (https://pt.wikipedia.org/wiki/Processamento_de_linguagem_natural)
- Coursera: Natural Language Processing Specialization (Andrew Ng)
- TensorFlow Documentation: https://www.tensorflow.org/

### Fechamento do Capítulo

Transformers mudaram o eixo da IA moderna porque fizeram da linguagem uma interface universal para software. Entender seus mecanismos e limites é indispensável para projetar sistemas robustos sobre modelos fundacionais.

<a id="capitulo-5"></a>
## Capítulo 5: Prompt Engineering

Prompt Engineering é a disciplina de desenhar instruções, contexto e exemplos de forma a obter respostas mais úteis, consistentes e seguras de modelos generativos. Em sistemas modernos, prompt engineering não é um truque periférico: é parte da especificação do comportamento do sistema.

### Princípios Fundamentais

- **Clareza de objetivo**: O modelo precisa saber exatamente o que deve produzir.
- **Contexto suficiente**: Incluir definições, restrições, exemplos e formato esperado reduz ambiguidades.
- **Delimitação de entradas**: Separar instruções, contexto e dados do usuário evita confusão.
- **Critérios de qualidade**: Especificar precisão, concisão, tom, idioma, formato e critérios de aceitação.
- **Iteração baseada em avaliação**: Bons prompts são refinados com evidência, não apenas por intuição.

### Técnicas Relevantes

- **Zero-shot**: Apenas instrução, sem exemplos.
- **Few-shot**: Inclui exemplos representativos.
- **Chain-of-thought guiado**: Útil em raciocínio, mas em produção convém privilegiar saídas estruturadas e verificáveis.
- **Structured output**: Restringe a saída a JSON, schemas ou formatos parseáveis.
- **Role prompting**: Define papel, responsabilidade e fronteiras do sistema.

### Erros Comuns

- Prompt excessivamente genérico.
- Misturar múltiplos objetivos conflitantes.
- Não definir formato de saída.
- Tentar compensar ausência de contexto com instruções vagas.
- Não testar contra casos adversos e entradas ambíguas.

### Boas Práticas de Engenharia

Em produção, prompts devem ser versionados, testados e observáveis. O ideal é tratá-los como artefatos de software: com revisão, histórico, métricas e rollout controlado.

### Fechamento do Capítulo

Prompt engineering, quando bem feito, deixa de ser improvisação e passa a ser design de interface com o modelo. O ganho real vem quando prompts entram no mesmo ciclo disciplinado de versionamento, teste e observabilidade que o restante do software.

<a id="capitulo-6"></a>
## Capítulo 6: Fine-Tuning

Fine-tuning é o processo de adaptar um modelo pré-treinado para uma tarefa, domínio ou estilo específico. Em vez de ensinar linguagem do zero, o objetivo é especializar capacidades já existentes.

### Quando Fine-Tuning Faz Sentido

- Quando há um padrão estável de saída desejada.
- Quando prompting sozinho não entrega consistência suficiente.
- Quando o domínio possui terminologia, estilo ou decisões próprias.
- Quando o custo de prompts extensos é alto demais.

### Quando Não É a Melhor Opção

- Quando o problema é falta de informação atualizada, caso em que RAG costuma ser melhor.
- Quando os requisitos mudam com frequência.
- Quando há pouco dado de qualidade.
- Quando um ajuste simples de prompt ou ferramentas resolve o caso.

### Abordagens Comuns

- **Supervised Fine-Tuning (SFT)**: Ajuste supervisionado com pares entrada-saída.
- **Instruction Tuning**: Especialização em seguir instruções.
- **Preference Optimization / RLHF / DPO**: Alinhamento a preferências humanas ou sintéticas.
- **PEFT / LoRA**: Ajustes eficientes em parâmetros, reduzindo custo computacional.

### Pipeline Recomendado

1. Definir objetivo de negócio e métricas.
2. Coletar e limpar dados.
3. Dividir datasets de treino, validação e teste.
4. Treinar e acompanhar métricas técnicas e métricas de tarefa.
5. Avaliar regressões, vieses e robustez.
6. Versionar modelo, dataset e configuração.

### Fechamento do Capítulo

Fine-tuning é poderoso, mas deve ser escolhido por critério econômico e arquitetural, não por moda. Em muitos casos, prompting, RAG ou tool use resolvem melhor o problema com menos custo operacional.

<a id="capitulo-7"></a>
## Capítulo 7: RAG (Retrieval-Augmented Generation)

RAG combina recuperação de informação com geração de linguagem. Em vez de depender apenas do conhecimento paramétrico do modelo, o sistema busca documentos relevantes e os injeta no contexto antes da resposta.

### Componentes de um Sistema RAG

- **Ingestão**: Coleta e normalização de documentos.
- **Chunking**: Divisão em partes semanticamente úteis.
- **Embeddings**: Representação vetorial dos trechos.
- **Indexação vetorial**: Armazenamento em banco vetorial ou mecanismo híbrido.
- **Retrieval**: Busca por similaridade, BM25 ou híbrida.
- **Re-ranking**: Reordenação dos resultados mais promissores.
- **Geração**: Resposta condicionada pelos documentos recuperados.

### Desafios Reais

- Chunking mal feito reduz recall.
- Embeddings inadequados prejudicam a recuperação.
- Contexto demais piora custo e foco.
- Dados desatualizados comprometem confiança.
- Sem citações ou evidências, a auditabilidade fica fraca.

### Boas Práticas

- Priorizar avaliação de retrieval separadamente da geração.
- Usar metadados fortes: fonte, data, domínio, permissão.
- Implementar filtros de autorização no momento da busca.
- Adotar respostas com referência explícita às fontes.
- Medir precisão factual, cobertura e taxa de resposta "não sei".

### Exemplo Prático: Pipeline RAG Mínimo em Python

O exemplo abaixo mostra um fluxo mínimo de RAG com embeddings locais, busca por similaridade e montagem de contexto para uma chamada posterior ao modelo.

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

documentos = [
    "O re-ranking melhora a qualidade final do contexto enviado ao modelo.",
    "Chunking em pedaços muito grandes reduz precisão de recuperação.",
    "RAG combina recuperação de informação com geração de linguagem.",
]

consulta = "Como o RAG melhora respostas factuais?"

modelo_embeddings = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings_docs = modelo_embeddings.encode(documentos)
embedding_consulta = modelo_embeddings.encode([consulta])[0]

similaridades = cosine_similarity([embedding_consulta], embeddings_docs)[0]
indices_ordenados = similaridades.argsort()[::-1][:2]

contexto = "\n".join(documentos[i] for i in indices_ordenados)

prompt = f"""
Responda usando apenas o contexto abaixo.

Contexto:
{contexto}

Pergunta: {consulta}
"""

print(prompt)
```

Esse padrão é propositalmente simples: em produção, você substituiria a lista em memória por um pipeline de ingestão, um banco vetorial, filtros por permissão, re-ranking e observabilidade de retrieval.

### Fechamento do Capítulo

RAG é uma das técnicas mais importantes para reduzir alucinação e atualizar conhecimento sem retreinar o modelo. O valor prático, porém, só aparece quando retrieval, autorização e avaliação são tratados como partes independentes do sistema.

<a id="capitulo-8"></a>
## Capítulo 8: Agentes e Uso de Ferramentas

Agentes são sistemas que combinam um modelo com memória operacional, regras, ferramentas externas e laços de decisão para executar tarefas compostas. O termo é amplo, mas, em engenharia, deve ser tratado com rigor: um agente é útil quando precisa decidir, observar, agir e se adaptar ao estado do ambiente.

### Elementos de um Sistema Agêntico

- **Planejamento**: Decompor tarefa em subtarefas.
- **Tool use**: Chamar APIs, bancos de dados, navegadores, interpretadores ou sistemas internos.
- **Memória**: Manter estado de execução e histórico relevante.
- **Critérios de parada**: Saber quando encerrar, pedir ajuda ou falhar com segurança.
- **Observabilidade**: Registrar decisões, chamadas e erros.

### Quando Usar Agentes

- Fluxos multi-etapa.
- Integração com múltiplas ferramentas.
- Necessidade de autonomia limitada e auditável.
- Tarefas que exigem busca, cálculo, execução de código ou navegação.

### Riscos

- Loops improdutivos.
- Chamada excessiva de ferramentas.
- Vazamento de dados para ferramentas não confiáveis.
- Execução de ações irreversíveis sem confirmação.

Por isso, agentes em produção exigem limites de orçamento, permissões explícitas, logs e mecanismos de aprovação humana para ações sensíveis.

### Exemplo Prático: Loop de Agente com Ferramentas e Limite de Iteração

Um agente em produção não deve ser modelado como autonomia irrestrita, mas como um laço controlado de decisão e execução. O exemplo abaixo mostra um esqueleto simples dessa ideia.

```python
def buscar_preco(produto: str) -> str:
    tabela = {"notebook": "R$ 4.500", "monitor": "R$ 1.200"}
    return tabela.get(produto.lower(), "Produto não encontrado")


TOOLS = {
    "buscar_preco": buscar_preco,
}


def planner(user_input: str, historico: list[dict]) -> dict:
    if "preço" in user_input.lower() or "preco" in user_input.lower():
        return {"type": "tool", "tool": "buscar_preco", "args": {"produto": "notebook"}}
    return {"type": "final", "content": "Posso responder diretamente sem usar ferramentas."}


def run_agent(user_input: str, max_steps: int = 3) -> str:
    historico = []

    for _ in range(max_steps):
        acao = planner(user_input, historico)

        if acao["type"] == "final":
            return acao["content"]

        if acao["type"] == "tool":
            ferramenta = TOOLS[acao["tool"]]
            resultado = ferramenta(**acao["args"])
            historico.append({"tool": acao["tool"], "result": resultado})
            return f"Usei a ferramenta {acao['tool']} e obtive: {resultado}"

    return "Limite de iterações atingido; encaminhar para revisão humana."


print(run_agent("Qual é o preço do notebook?"))
```

Embora simplificado, esse padrão reforça quatro elementos fundamentais: limite de passos, catálogo explícito de ferramentas, registro de estado e fallback seguro.

### Fechamento do Capítulo

Agentes são úteis quando há necessidade real de coordenação entre raciocínio, memória e ação. Fora disso, eles adicionam custo e superfície de risco sem retorno proporcional.

<a id="capitulo-9"></a>
## Capítulo 9: Infraestrutura para IA

Infraestrutura para IA envolve as camadas necessárias para treinar, servir, monitorar e evoluir sistemas inteligentes com confiabilidade operacional.

### Blocos Principais

- **Compute**: CPU, GPU, TPU e aceleradores especializados.
- **Armazenamento**: Data lake, feature store, object storage, bancos relacionais e vetoriais.
- **Serving**: APIs síncronas, assíncronas, batch, streaming e realtime.
- **Filas e eventos**: Para desacoplamento e escalabilidade.
- **Observabilidade**: Logs, traces, métricas e alertas.
- **Segurança**: IAM, gestão de segredos, isolamento e criptografia.

### Trade-offs Relevantes

- **Latency vs. quality**: Modelos maiores tendem a custar mais e responder mais lentamente.
- **Hosted vs. self-hosted**: Serviços gerenciados reduzem operação, mas limitam controle.
- **GPU centralizada vs. distribuída**: Impacta custo, utilização e complexidade.
- **Batch vs. online inference**: Depende de requisitos de tempo de resposta.

### Exemplo Prático: Serving Local com vLLM e API Compatível com OpenAI

Em equipes de plataforma, um padrão comum é servir modelos localmente ou em infraestrutura própria com uma API compatível com clientes já existentes.

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000
```

Depois de subir o servidor, um cliente Python pode consumi-lo como se estivesse falando com uma API compatível com OpenAI:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[
                {"role": "system", "content": "Você é um assistente técnico conciso."},
                {"role": "user", "content": "Explique o que é batching contínuo."},
        ],
)

print(response.choices[0].message.content)
```

Esse arranjo é importante porque desacopla cliente e provedor de inferência, permitindo testes locais, redução de custo e maior controle operacional.

### Fechamento do Capítulo

Infraestrutura para IA deixa de ser detalhe quando latência, custo e confiabilidade passam a ser parte do produto. Saber servir modelos de forma eficiente é tão estratégico quanto escolher o modelo certo.

<a id="capitulo-10"></a>
## Capítulo 10: MLOps, Avaliação e Observabilidade

MLOps amplia práticas de DevOps para o ciclo de vida de modelos e sistemas de IA. Em LLMOps, a disciplina inclui não apenas versionamento de modelos, mas também prompts, datasets, políticas, traces e avaliações automáticas.

### Capacidades Essenciais

- **Versionamento**: Código, dataset, modelo, prompt e configuração.
- **CI/CD**: Testes automatizados e implantação progressiva.
- **Avaliação contínua**: Benchmarks offline e testes online.
- **Observabilidade**: Monitorar custo, latência, falhas, drift e qualidade percebida.
- **Rollback**: Reverter versões com rapidez e segurança.

### Métricas Importantes

- Latência p50, p95 e p99.
- Custo por requisição e por sessão.
- Taxa de erro e timeout.
- Precisão factual.
- Taxa de uso correto de ferramenta.
- Satisfação do usuário e resolução da tarefa.

Em sistemas generativos, uma boa prática é separar métricas em três camadas: infraestrutura, qualidade de resposta e impacto de produto.

### Exemplo Prático: Harness Simples de Avaliação Offline

Nem toda avaliação precisa começar com uma plataforma complexa. Um harness simples já ajuda a transformar percepções subjetivas em critérios explícitos.

```python
casos_de_teste = [
    {
        "pergunta": "O que é RAG?",
        "resposta_esperada": ["recuperação", "geração"],
    },
    {
        "pergunta": "O que é fine-tuning?",
        "resposta_esperada": ["adaptação", "modelo pré-treinado"],
    },
]


def avaliar_resposta(resposta_modelo: str, termos_esperados: list[str]) -> float:
    resposta_normalizada = resposta_modelo.lower()
    acertos = sum(1 for termo in termos_esperados if termo in resposta_normalizada)
    return acertos / len(termos_esperados)


def modelo(pergunta: str) -> str:
    respostas = {
        "O que é RAG?": "RAG é uma abordagem que combina recuperação de informação com geração de texto.",
        "O que é fine-tuning?": "Fine-tuning é a adaptação de um modelo pré-treinado para uma tarefa específica.",
    }
    return respostas[pergunta]


scores = []
for caso in casos_de_teste:
    resposta = modelo(caso["pergunta"])
    score = avaliar_resposta(resposta, caso["resposta_esperada"])
    scores.append(score)
    print(caso["pergunta"], score)

print("Score médio:", sum(scores) / len(scores))
```

Esse tipo de avaliação não substitui frameworks como OpenAI Evals, mas já cria uma disciplina importante: definir casos, comparar versões e medir regressões antes de publicar mudanças.

### Fechamento do Capítulo

Sem avaliação, não existe melhoria contínua confiável em IA. MLOps e LLMOps só se tornam maduros quando qualidade deixa de ser opinião e passa a ser medida sistematicamente.

<a id="capitulo-11"></a>
## Capítulo 11: Segurança, Governança e IA Responsável

Segurança em IA não se limita a proteger infraestrutura. É necessário proteger o comportamento do sistema, os dados usados como contexto, as integrações externas e a forma como respostas são consumidas.

### Riscos Importantes

- **Prompt injection**: Instruções maliciosas vindas de documentos, web ou entradas de usuário.
- **Data leakage**: Exposição de segredos, PII ou conteúdo restrito.
- **Tool abuse**: Uso indevido de conectores ou ferramentas privilegiadas.
- **Model misuse**: Geração de conteúdo enganoso, discriminatório ou inseguro.
- **Supply chain risk**: Dependências, modelos e conectores inseguros.

### Práticas de Mitigação

- Isolar dados confiáveis e não confiáveis.
- Aplicar políticas de menor privilégio em ferramentas.
- Redigir logs com cuidado para não vazar segredos.
- Adotar avaliações de segurança e red teaming.
- Definir políticas de retenção, auditoria e revisão humana.

Governança madura em IA também exige responsabilização: saber qual versão do modelo respondeu, com qual prompt, com quais documentos e em qual contexto operacional.

### Fechamento do Capítulo

Segurança e governança não são camadas opcionais adicionadas no fim do projeto. Em IA, elas precisam nascer junto com a arquitetura, porque os riscos se manifestam no comportamento do sistema, não apenas na infraestrutura.

<a id="capitulo-12"></a>
## Capítulo 12: Repositórios GitHub Essenciais para Engenharia de IA

Uma base sólida em Engenharia de IA não vem apenas de teoria. Ela depende de contato com SDKs, frameworks de orquestração, bibliotecas de serving, exemplos de avaliação e projetos de referência usados pelo mercado. A curadoria abaixo privilegia repositórios públicos consolidados, mantidos por grandes players ou por projetos de infraestrutura que influenciam diretamente a prática profissional.

### 12.1 OpenAI e ecossistema GPT

- **[openai/openai-cookbook](https://github.com/openai/openai-cookbook)**: Excelente ponto de entrada para padrões práticos de uso da API, avaliações, RAG, function calling, multimodalidade e integrações. Útil para aprender como transformar capacidades de modelo em soluções de produto.
- **[openai/openai-python](https://github.com/openai/openai-python)**: SDK oficial em Python. Ensina padrões reais de cliente, streaming, retries, paginação, upload de arquivos, webhooks, autenticação e integração segura em aplicações.
- **[openai/evals](https://github.com/openai/evals)**: Repositório fundamental para aprender avaliação sistemática de LLMs e sistemas baseados em LLMs. Se você quer sair do empirismo e adotar critérios objetivos, este é um dos repositórios mais importantes.

### 12.2 Google, Gemini e ecossistema multimodal

- **[google-gemini/cookbook](https://github.com/google-gemini/cookbook)**: Cookbook oficial do Gemini, com quickstarts e exemplos práticos de grounding, file search, Live API, multimodalidade e aplicações end-to-end.
- **[googleapis/python-genai](https://github.com/googleapis/python-genai)**: SDK oficial Python para modelos generativos do Google. Relevante para aprender integração de API em aplicações reais e fluxos de produção.
- **[google-gemini/gemini-cli](https://github.com/google-gemini/gemini-cli)**: Agente open-source para terminal baseado em Gemini. Útil para estudar ergonomia de agentes, interação via CLI e patterns de uso assistido por ferramentas.

### 12.3 GitHub Copilot e assistentes de desenvolvimento

- **[github/copilot.vim](https://github.com/github/copilot.vim)**: Embora o produto GitHub Copilot não seja totalmente open-source, este cliente oficial para Vim/Neovim é uma referência pública útil para entender integração de assistente de código no editor, autenticação e experiência de uso em fluxo de desenvolvimento.

### 12.4 Microsoft e sistemas agênticos

- **[microsoft/semantic-kernel](https://github.com/microsoft/semantic-kernel)**: Framework de orquestração agêntica com foco corporativo, plugins, memória, conectores e multi-agent systems. Muito útil para aprender arquitetura de aplicações LLM enterprise.
- **[microsoft/autogen](https://github.com/microsoft/autogen)**: Projeto historicamente importante para multiagentes. Continua valendo como referência conceitual e de padrões, embora esteja em maintenance mode.
- **[microsoft/agent-framework](https://github.com/microsoft/agent-framework)**: Sucessor enterprise-ready do AutoGen para construir, orquestrar e implantar agentes e fluxos multiagente em Python e .NET.

### 12.5 Meta, open models e uso prático de Llama

- **[meta-llama/llama-cookbook](https://github.com/meta-llama/llama-cookbook)**: Um dos melhores materiais para aprender inferência, fine-tuning, RAG e casos end-to-end com a família Llama. Muito útil para quem quer compreender o lado open-weight da engenharia de IA.

### 12.6 Anthropic e SDKs de produção

- **[anthropics/anthropic-sdk-python](https://github.com/anthropics/anthropic-sdk-python)**: SDK oficial do Claude em Python. Bom para entender padrões de integração, versionamento de cliente e uso programático de modelos comerciais de ponta.

### 12.7 Hugging Face e base do ecossistema open-source

- **[huggingface/transformers](https://github.com/huggingface/transformers)**: Biblioteca central do ecossistema open-source de modelos. Essencial para entender inferência, treino, fine-tuning e interoperabilidade com centenas de arquiteturas e frameworks.

### 12.8 Serving e performance

- **[vllm-project/vllm](https://github.com/vllm-project/vllm)**: Referência obrigatória para serving de LLMs em alta escala. Ensina temas críticos como throughput, uso eficiente de memória, batching contínuo, prefix caching, API compatível com OpenAI e serving distribuído.

### Fechamento do Capítulo

Estudar bons repositórios encurta o caminho entre teoria e prática. O ganho maior não está em decorar APIs, mas em observar como projetos maduros organizam exemplos, testes, versionamento, integrações e decisões de arquitetura.

## Como Estudar Esses Repositórios de Forma Eficiente

Uma abordagem pragmática é não tentar "ler tudo". O ideal é explorar os repositórios por problema de engenharia:

1. Se você quer aprender integração de API e padrões de aplicação, comece por SDKs e cookbooks.
2. Se quer aprender RAG e avaliação, estude cookbooks, evals e exemplos de retrieval.
3. Se quer aprender agentes, compare Semantic Kernel, Agent Framework, AutoGen e Gemini CLI.
4. Se quer aprender serving e infraestrutura, aprofunde em vLLM e no ecossistema Hugging Face.
5. Se quer aprender open models e adaptação, combine Transformers com Llama Cookbook.

## Conclusão

Engenharia de IA exige uma combinação rara de fundamentos matemáticos, entendimento de modelos, disciplina de software, infraestrutura, segurança e avaliação contínua. O profissional forte nessa área não é apenas alguém que sabe chamar uma API, nem apenas alguém que conhece teoria de redes neurais. É alguém capaz de construir sistemas confiáveis, mensuráveis, evolutivos e alinhados a restrições reais de negócio.

Os repositórios públicos listados neste eBook oferecem um caminho concreto para sair da abstração e observar como o mercado realmente estrutura SDKs, frameworks, pipelines de avaliação, agentes e serving. Estudar esses projetos com intencionalidade acelera muito a formação de uma base sólida em Engenharia de IA.

## Base Sólida em Engenharia de IA: Checklist de Competências

Use este checklist como referência para validar se sua formação está equilibrada entre teoria, construção de sistemas e operação.

1. **Fundamentos quantitativos**: você entende otimização, probabilidade e álgebra linear o suficiente para interpretar falhas e ajustar abordagens.
2. **Modelagem**: você distingue quando usar ML clássico, Deep Learning, prompting, fine-tuning, RAG ou agentes.
3. **Arquitetura de sistemas**: você consegue desenhar um fluxo fim a fim com ingestão de dados, inferência, armazenamento e observabilidade.
4. **Avaliação**: você mede qualidade com critérios explícitos e compara versões de forma reprodutível.
5. **Operação e custo**: você acompanha latência, throughput e custo por requisição para guiar decisões técnicas.
6. **Segurança e governança**: você trata prompt injection, controle de acesso, proteção de dados e rastreabilidade desde o desenho da solução.
7. **Capacidade de entrega**: você transforma protótipos em aplicações documentadas, testáveis e mantidas por equipe.

Se algum item estiver fraco, use a trilha de 30, 60 e 90 dias como plano de recuperação direcionado.

## Trilha de Estudos: 30, 60 e 90 Dias

### Primeiros 30 Dias: Fundamentos e Fluência Conceitual

Objetivo: construir vocabulário técnico e visão sistêmica.

1. Estude os capítulos 1 a 4 com foco em entendimento conceitual.
2. Reproduza os exemplos simples de ML, Deep Learning e Transformers.
3. Leia o README e os quickstarts de [openai/openai-cookbook](https://github.com/openai/openai-cookbook), [google-gemini/cookbook](https://github.com/google-gemini/cookbook) e [huggingface/transformers](https://github.com/huggingface/transformers).
4. Monte um glossário pessoal com termos como embeddings, tokenização, fine-tuning, retrieval, tool calling e observabilidade.

Entregável sugerido: um documento curto explicando, com suas palavras, como um sistema baseado em LLM funciona de ponta a ponta.

### Dias 31 a 60: Construção de Sistemas Pequenos

Objetivo: sair do estudo passivo e começar a compor sistemas.

1. Implemente um mini projeto de RAG com um pequeno conjunto de documentos.
2. Crie um agente simples com uma ou duas ferramentas controladas.
3. Suba um modelo local ou servidor compatível com OpenAI usando [vllm-project/vllm](https://github.com/vllm-project/vllm) ou estude sua arquitetura de serving.
4. Experimente uma rotina mínima de avaliação offline para comparar prompts, versões ou respostas.

Entregável sugerido: um protótipo funcional com README, métricas básicas e logs simples.

### Dias 61 a 90: Maturidade de Engenharia

Objetivo: transformar protótipos em sistemas avaliáveis e operáveis.

1. Adicione versionamento de prompts, dados e configurações.
2. Implante observabilidade mínima: latência, custo, erros e qualidade percebida.
3. Estude [openai/evals](https://github.com/openai/evals), [microsoft/semantic-kernel](https://github.com/microsoft/semantic-kernel) e [microsoft/agent-framework](https://github.com/microsoft/agent-framework) comparando filosofia e arquitetura.
4. Faça uma revisão de segurança do seu sistema: prompt injection, dados sensíveis, escopo de ferramentas e logs.
5. Reescreva partes do projeto pensando em rollback, testes e manutenção por equipe.

Entregável sugerido: uma aplicação pequena, mas com padrão profissional de documentação, avaliação e operação.

### Resultado Esperado ao Final de 90 Dias

Se a trilha for seguida com disciplina, você termina este ciclo não apenas sabendo usar modelos, mas sabendo tomar decisões de engenharia sobre quando usar prompting, fine-tuning, RAG, agentes, serving próprio e avaliação contínua. Esse é o ponto em que a formação deixa de ser apenas técnica e passa a ser profissional.