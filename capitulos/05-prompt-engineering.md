# Capítulo 5: Prompt Engineering

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

