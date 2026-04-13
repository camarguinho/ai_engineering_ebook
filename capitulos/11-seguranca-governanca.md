# Capítulo 11: Segurança, Governança e IA Responsável

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

