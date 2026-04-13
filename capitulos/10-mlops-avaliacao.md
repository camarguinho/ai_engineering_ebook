# Capítulo 10: MLOps, Avaliação e Observabilidade

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

