# Capítulo 9: Infraestrutura para IA

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

