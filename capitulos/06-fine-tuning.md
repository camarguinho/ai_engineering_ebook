# Capítulo 6: Fine-Tuning

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

