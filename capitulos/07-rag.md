# Capítulo 7: RAG (Retrieval-Augmented Generation)

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

