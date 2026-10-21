# Experimento E — avaliação real ampliada, exclusivamente offline

Compara os checkpoints `best.pt` de A/B/D em CPU usando a arquitetura,
pré-processamento, greedy CTC e alinhamento já existentes. Não treina, não
gera CAPTCHAs e não faz requisições HTTP. Não enumera nem abre `real/test`.

O protocolo [experiments/real-transfer-expanded.json](../experiments/real-transfer-expanded.json)
fixa os checkpoints por SHA-256, épocas e tamanho de treino, a validação
sintética histórica e a decisão humana anterior às previsões para `6giL`.
Os caminhos do protocolo são relativos à raiz do repositório.

```bash
.venv-ml/bin/python -m ml.compare_expanded \
  --protocol experiments/real-transfer-expanded.json \
  --output outputs/analysis/real-transfer-expanded-rerun \
  --batch-size 16 --threads 6
```

Use uma saída nova para cada reprodução. A CLI recusa resultados existentes
e saídas dentro dos datasets/modelos. As únicas subpastas preexistentes
aceitas são `preflight/` e `incoming/`, para preservar a auditoria inicial
e os arquivos recebidos por SCP sem sobrescrevê-los.

O manifesto inclui todas as imagens de train/validation, labels originais,
status, exclusão, pendências, charset e hashes. `l` minúsculo exclui a imagem
antes de qualquer previsão real, igualmente nos três modelos. `L`/`I`
maiúsculos, labels ausentes/incertos ou incompatíveis geram uma pendência
separada; a comparação não prossegue sem uma decisão humana explícita.
Não existe normalização, substituição ou equivalência entre caracteres.
O usuário autorizou excluir `6giL`, separadamente da regra de `l`.

Antes da inferência real, A/B/D precisam reproduzir as contagens sintéticas
fixadas no protocolo. Cada imagem/label sintético também é conferido contra
o manifesto histórico. A inferência reutiliza `ml.evaluate.frozen_inference`:
CPU, eval(), inference_mode(), pesos congelados e buffers conferidos.
Hashes de todas as imagens permitidas, labels e checkpoints são reconferidos
após a galeria. Symlinks, junctions e imagens com hardlinks são recusados
sem precisar inspecionar o conjunto reservado.

Artefatos: `REPORT.md`, `summary.json`, `eligibility.csv`, `eligibility.json`,
`predictions.csv`, `paired-comparison.csv`, `comparison-gallery.png`, páginas
numeradas da galeria, regressões sintéticas e `test-results.txt`.

CER é ponderado pelo total de caracteres. Os três pares são avaliados em
cada split e no agregado, com listas completas das mudanças de acerto e
distância. Wilson 95% e McNemar exato são exploratórios; não há correção de
múltiplas comparações e ausência de significância não comprova equivalência.
`train` é diagnóstico, `validation` é desenvolvimento já consultado e a união
não é um teste independente. D modificou amostragem, orçamento e scheduler,
além da quantidade de dados; os efeitos não são isoláveis nesta comparação.

Os testes novos estão em `tests/test_ml_compare_expanded.py`, com fixtures
pequenas, sem treinamento ou gerador. **Não foram executados**, assim como
a suíte completa, por decisão posterior explícita do usuário. A avaliação
conserva as verificações de integridade e as regressões sintéticas necessárias.
