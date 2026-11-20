# Keen IA — núcleo experimental

Backend sem interface visual, sem chamadas a provedores de IA: tokenizador reversível UTF-8 por byte, Transformer decoder autoregressivo Keen-1, treinamento local, checkpoints em safetensors, inferência CPU, SQLite e API HTTP/SSE. O código e os pesos incluídos são uma **prova de conceito**, não um LLM de produção.

## Limitações essenciais

- **Qualidade:** o Keen-1 foi treinado do zero em apenas 20 exemplos sintéticos de estilo; não foi pré-treinado em corpus amplo. Ele pode produzir fragmentos incoerentes, repetição, erros e UTF-8 substituído. Não é adequado para raciocínio, matemática, código, aconselhamento ou decisões reais. Não há demonstração de personalidade consistente, segurança ou capacidade de seguir instruções. Perplexidade não comprova essas capacidades.
- **Contexto:** 128 tokens que são *bytes*, não palavras. A inferência mantém apenas os últimos 128 bytes do prompt; mensagens longas, histórico e memórias podem ser parcial ou totalmente descartados. Não há resumo de conversas nem recuperação semântica. Memórias explícitas são recuperadas por sobreposição lexical; isso não é raciocínio nem garante que influenciem a resposta.
- **Treinamento:** exemplos comportamentais realmente atualizam os pesos, mas são insuficientes para ensinar personalidade geral. `--resume` retoma os pesos, porém **reinicia Adam** (não restaura momentos nem estado aleatório); não é retomada exata. Não há pipeline automático de melhoria, pré-treinamento massivo, RLHF ou avaliação confiável de raciocínio. Pesos de arquiteturas externas não são automaticamente compatíveis.
- **Embeddings:** média dos embeddings de bytes treinados para predição, não embeddings semânticos validados.
- **Infraestrutura:** executar localmente é gratuito além dos custos de computador/energia. Gratuidade, disponibilidade, CPU, armazenamento persistente e política de cartão em um provedor são condições externas que podem mudar; não há garantia de hospedagem gratuita ou 24h. Com limites gratuitos pequenos, inferência em CPU pode ser lenta. O treino inicial foi realizado em CPU, sem API paga.
- **Segurança:** não há moderação, prevenção confiável de prompt injection, revisão jurídica de dados, nem isolamento por usuário final quando múltiplas pessoas compartilham a mesma API key. Não use em produção nem com dados sensíveis sem controles adicionais. Texto salvo pode influenciar prompts; trate memória como entrada não confiável.

## Iniciar localmente

Requer Python 3.11, compilador `clang` disponível no PATH (tinygrad compila kernels em tempo de execução) e cerca de 1 GB de RAM ou mais. Linux x86-64 é o alvo testado. No Debian/Ubuntu: `apt-get install clang`.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
export KEEN_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
echo "$KEEN_API_KEY" # guarde fora do repositório
uvicorn keen.api:app --host 0.0.0.0 --port 8080
```

A API não inicia sem uma chave de ao menos 16 caracteres. Documentação OpenAPI: `http://localhost:8080/docs`. Saúde sem autenticação: `GET /v1/health`. Chamadas restantes exigem `Authorization: Bearer <chave>`. Nunca exponha a chave no frontend ou no Git.

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $KEEN_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"keen-1-experimental","messages":[{"role":"user","content":"O que é coragem?"}],"max_tokens":32,"conversation_id":"sessao1"}'
```

Para streaming, acrescente `"stream":true` e use `curl -N`; recebe eventos SSE, terminando em `data: [DONE]`. Também há `POST /v1/completions`, `GET /v1/models`, `POST /v1/embeddings`. Os parâmetros suportados estão no `/docs`; a estrutura é parcialmente compatível com APIs LLM comuns, não uma implementação completa do padrão OpenAI. `max_tokens` é número máximo de bytes gerados; pode terminar no meio de um caractere UTF-8. O primeiro pedido de cada tamanho pode levar tempo extra por compilação.

## Memória e isolamento

Por padrão `KEEN_API_KEY` identifica um único tenant chamado `default`. Para vários usuários use `KEEN_API_KEYS`, um JSON de **chaves secretas para IDs de tenant**, por exemplo `{"<chave-longa-alice>":"alice","<chave-longa-bob>":"bob"}`; cada usuário precisa de chave própria. A identidade é derivada da chave, não de um ID informado na requisição. Chaves associadas ao mesmo tenant compartilham memória. Rotação exige manter o mesmo ID. Configure segredos no servidor, nunca em código cliente público.

Com `conversation_id`, pares usuário/resposta concluídos são guardados automaticamente em SQLite; até 12 turnos anteriores são recuperados. Requisições simultâneas para a mesma conversa não têm ordem garantida; serialize-as no cliente. Stream interrompido não salva o par. Não envie ao mesmo tempo o histórico anterior completo e `conversation_id`, pois ele será duplicado.

`PUT /v1/memories/{key}` com `{"content":"..."}` adiciona/atualiza uma memória explícita. `DELETE /v1/memories/{key}` e `DELETE /v1/conversations/{id}` apagam dados daquele tenant. Não existe listagem/exportação nem extração ou resumo automáticos. `KEEN_DB_PATH` controla a localização (padrão `data/keen.sqlite3`). Para sobreviver a reinícios/redeploy, use volume persistente e faça backups seguros; criptografia em repouso e retenção são responsabilidade da infraestrutura. SQLite não é apropriado para réplicas independentes com arquivos locais diferentes.

Limite padrão: 30 requisições/minuto por tenant e por processo (`KEEN_RATE_LIMIT`), incluindo memória e modelos. Não é distribuído entre réplicas; use gateway externo para proteção efetiva. Processamento da geração é serializado por processo; limitação de concorrência/filas e proteção contra abuso devem ser adicionadas antes de exposição pública.

## Northflank / implantação

O `Dockerfile` instala dependências e contém os pesos; expõe `PORT` (padrão 8080). Conectar a integração GitHub **não cria automaticamente um serviço nem configura segredos ou armazenamento**. Crie um projeto e um serviço de build/deploy a partir deste repositório e branch, escolha Dockerfile, habilite deploy automático, configure `KEEN_API_KEY` como secret, exponha a porta HTTP e configure health check em `/v1/health`. Monte volume persistente em `/app/data`, gravável pelo usuário não-root da imagem (UID gerado pela imagem), ou a memória será perdida no redeploy. O provisionamento de volume e disponibilidade de plano gratuito dependem do serviço. Execute uma única réplica enquanto usar SQLite. A API não precisa de cartão nem de API externa, mas a plataforma pode impor requisitos próprios.

Exemplo Docker:

```bash
docker build -t keen-ia .
docker run --rm -p 8080:8080 -e KEEN_API_KEY="sua-chave-secreta-longa" keen-ia
```

Sem volume esse comando é apenas demonstração e perde memórias ao encerrar. Use TLS ao expor a API e preserve credenciais fora do histórico do shell. A conexão do repositório com o Arena também não executa a API nem configura uma implantação automaticamente; a interface existente deve apontar para o backend implantado por meio de um servidor que proteja a chave.

## Treinamento, fine-tuning e avaliação

```bash
# Treina do zero e salva config + model.safetensors + métricas
python -m keen.training --data datasets/personality.jsonl --steps 100 --output models/keen-1-new
# Continua os pesos com mais exemplos (Adam é reiniciado)
python -m keen.training --data datasets/personality.jsonl --resume models/keen-1-new --steps 100 --output models/keen-1-v2
# Perplexidade em exemplos de validação e amostras para revisão humana
python -m keen.evaluation --model models/keen-1
pip install -r requirements-dev.txt
pytest -q
```

Cada linha JSONL precisa conter `prompt` e `response`. O split determinístico por exemplo separa 10% para validação; os exemplos são curtos e o conjunto é minúsculo. Apenas compartilhe dados com licença e autorização apropriadas. Ver `datasets/README.md`. Versione pesos, configuração, origem/licença e métricas juntos. `models/keen-1/training.json` registra esta execução. Os arquivos de pesos são carregados exclusivamente como safetensors locais; não carregue pesos/configuração de fonte não confiável. O modelo é extensível via `keen/model.py`, mas um modelo útil requer corpus licenciado, avaliações independentes, recursos de computação e provavelmente uma arquitetura/pretreinamento compatível mais robusto.

## Módulos

`keen/tokenizer.py`: bytes; `model.py`: embeddings, posições, self-attention causal, MLP, decoder e pesos; `training.py`: JSONL, treino, split e validação; `inference.py`: amostragem; `memory.py`: persistência; `api.py`: HTTP e SSE; `evaluation.py`: avaliação inicial. Nenhuma interface visual foi criada.
