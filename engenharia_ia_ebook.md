# Engenharia de IA: Uma Base Sólida

## Prefácio

Engenharia de IA é, cada vez menos, um campo exclusivamente acadêmico e, cada vez mais, uma disciplina de construção de sistemas. Saber treinar modelos, conhecer artigos ou dominar bibliotecas isoladas já não basta. O diferencial prático está em conectar fundamentos matemáticos, engenharia de software, avaliação, infraestrutura, segurança e operação contínua em ambientes reais.

Este eBook foi escrito para profissionais que querem construir essa visão integrada. O objetivo não é apenas explicar o que são modelos, embeddings, agentes ou RAG, mas mostrar como essas peças se encaixam em uma arquitetura de produto e em uma rotina de engenharia.

Ao longo dos capítulos, a leitura progride do núcleo conceitual para a camada operacional. A recomendação é estudar cada capítulo com duas perguntas em mente: "que problema esta técnica resolve?" e "como eu a colocaria em produção com segurança, observabilidade e custo aceitável?". Essa mudança de perspectiva é o que separa consumo de tecnologia de engenharia de tecnologia.

## Introdução

Este eBook oferece uma base sólida em Engenharia de IA, cobrindo tópicos essenciais de forma evolutiva e natural. Começamos com os fundamentos matemáticos que sustentam a IA, progredindo para conceitos de Machine Learning, Deep Learning e avanços em engenharia como MLOps e segurança. Cada capítulo é escrito em nível intermediário, com explicações conceituais, exemplos práticos e referências para aprofundamento.

O conteúdo é organizado para aprendizado progressivo: dos princípios matemáticos básicos até aplicações práticas em engenharia, permitindo uma compreensão holística da IA como ferramenta de engenharia.

### Estrutura do eBook

1. [Capítulo 1: Fundamentos Matemáticos](./capitulos/01-fundamentos-matematicos.md)
2. [Capítulo 2: Machine Learning](./capitulos/02-machine-learning.md)
3. [Capítulo 3: Deep Learning](./capitulos/03-deep-learning.md)
4. [Capítulo 4: Transformers e Processamento de Linguagem Natural (NLP)](./capitulos/04-transformers-nlp.md)
5. [Capítulo 5: Prompt Engineering](./capitulos/05-prompt-engineering.md)
6. [Capítulo 6: Fine-Tuning](./capitulos/06-fine-tuning.md)
7. [Capítulo 7: RAG (Retrieval-Augmented Generation)](./capitulos/07-rag.md)
8. [Capítulo 8: Agentes e Uso de Ferramentas](./capitulos/08-agentes-e-uso-de-ferramentas.md)
9. [Capítulo 9: Infraestrutura para IA](./capitulos/09-infraestrutura-para-ia.md)
10. [Capítulo 10: MLOps, Avaliação e Observabilidade](./capitulos/10-mlops-avaliacao.md)
11. [Capítulo 11: Segurança, Governança e IA Responsável](./capitulos/11-seguranca-governanca.md)
12. [Capítulo 12: Repositórios GitHub Essenciais para Engenharia de IA](./capitulos/12-repositorios-github.md)

Os capítulos numerados foram extraídos para arquivos independentes na pasta capitulos. Este arquivo passa a ser o índice principal do ebook.

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