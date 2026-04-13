# Capítulo 8: Agentes e Uso de Ferramentas

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

