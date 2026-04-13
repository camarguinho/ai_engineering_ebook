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

