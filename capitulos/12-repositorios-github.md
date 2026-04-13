# Capítulo 12: Repositórios GitHub Essenciais para Engenharia de IA

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

