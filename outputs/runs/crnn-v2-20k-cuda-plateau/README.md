# CRNN D — checkpoint original publicado

Arquivo: [best.pt](best.pt), 14.321.743 bytes. É o checkpoint original do
treinamento CUDA na NVIDIA RTX A2000; não foi convertido, extraído, retreinado
ou substituído por `last.pt`. O arquivo inclui os metadados e estados gravados
pelo treinamento original.

| Propriedade | Valor |
| --- | --- |
| Experimento | D — crnn-v2-20k-cuda-plateau |
| Época interna de best.pt | 46 |
| Encerramento do treinamento | Época 65, por early stopping |
| Imagens sintéticas de treino | 20.000 |
| Validação sintética histórica | 300 imagens, 1.507 caracteres |
| Exact match sintético em CPU | 268/300 — 89,3333% |
| CER sintético em CPU | 40/1507 — 2,6543% |
| Parâmetros | 1.186.815 |
| Charset literal | 23456789abcdefghiklmnprstuvwyz |

SHA-256:

```text
da2c608b9776f7ea54e15996e8bcb1c8bcfa54f2c4fed0f8c536ff7f093bfddf
```

Na pasta deste arquivo, confira com:

```bash
sha256sum -c best.pt.sha256
```

A avaliação usa `ml.train.load_checkpoint`, com `map_location="cpu"` e
`weights_only=True`, e a CRNN existente. Não é necessário executar treinamento
para carregar o modelo ou reproduzir as métricas de inferência.

O [protocolo do experimento E](../../../experiments/real-transfer-expanded.json)
fixa este hash e os checkpoints A/B. Os resultados reais estão no
[relatório comparativo](../../analysis/real-transfer-expanded/REPORT.md).
A/B e o dataset de treino de 20.000 imagens não integram esta publicação.
O conjunto `real/test` permaneceu reservado durante a avaliação E.
