# Keen-1 experimental: cartão de modelo

- Arquitetura: decoder Transformer de 2 camadas, largura 64, 4 cabeças, 141.440 parâmetros; contexto 128 bytes. Tokenizador UTF-8 de 259 IDs.
- Origem: pesos treinados neste projeto a partir de inicialização aleatória, sem modelos ou APIs externos, nos 20 exemplos sintéticos originais de `datasets/personality.jsonl`; licença MIT do repositório. Sem pré-treinamento de larga escala.
- Execução: CPU, 100 passos, batch 2, Adam, taxa 0,001, semente 42. Separação determinística por exemplos: 18 treino e 2 validação. `training.json` acompanha os pesos; o ponto de partida teve loss 5,819 e o final 2,526 na mesma amostragem de validação. A perplexidade final por byte foi 12,50. Os dados e a amostra são pequenos demais para sustentar conclusões gerais.
- Revisão exploratória: com `O que é coragem?`, decodificação greedy e 24 bytes, a saída observada foi `e on: conco enconco a en`. **Isso não é resposta útil.** A API e o treinamento funcionam tecnicamente, mas os pesos não entregam compreensão, raciocínio, personalidade estável ou qualidade conversacional. Não utilize para decisões ou conteúdos de risco.
- A melhoria depende de dados apropriados e licenciados, recursos de treinamento, arquitetura compatível, avaliações fora da amostra e revisão humana; não está assegurada pela redução da loss.
