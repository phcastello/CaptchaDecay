# Expansão sintética e learning rate

Este protocolo prepara dois treinamentos manuais para separar o efeito do
tamanho do dataset do efeito do learning rate adaptativo. A arquitetura CRNN,
o pré-processamento, o decoder CTC e o gerador visual permanecem iguais.
Os argumentos verificáveis estão em [dataset-expansion.json](dataset-expansion.json);
esse arquivo descreve os experimentos e não inicia processos.

| Experimento | Treino | Validação | Épocas | LR | Updates com batch 16 |
| --- | ---: | ---: | ---: | --- | ---: |
| A, concluído | 2.700 | 300 | 100 | 0,001 fixo | 16.900 |
| B, preparado | 9.000 | As mesmas 300 | 30 | 0,001 fixo | 16.890 |
| C, preparado | Os mesmos 9.000 | As mesmas 300 | 30 | Plateau, inicial 0,001 | 16.890 |

A melhor época de A é 95: exact match 185/300 = 61,6667%, CER
160/1.507 = 0,106171201062 e train loss 0,004282. A não é treinado novamente;
`outputs/runs/crnn-v2-baseline` permanece preservado.

## Preparação e preservação

Há uma divergência entre a descrição histórica e o estado local encontrado:
`dataset/synthetic/v2_main` já tinha 9.000 imagens de treino e 300 de validação
antes desta preparação. O script não modifica essa fonte. Ele identifica o
prefixo histórico de 3.000 amostras e exige que seus fingerprints coincidam
com o dataset registrado no `run_config.json` do baseline:

- SHA-256 do CSV histórico: `6f553b7fe3acf17d2a603381133b20d8e3da139808219e8fa20816efad156bf1`.
- Fingerprint agregado dos PNGs históricos: `bdaeab77d32f9c6ae98b9ba7b1d6cf349865481bb36aa41455d766f905013b97`.

Essa referência comprovada é copiada para
`outputs/experiments/dataset-expansion/baseline-dataset` (2.700/300), permitindo
reproduzir a identidade do dataset A sem reescrever `v2_main`. O manifesto
`outputs/experiments/dataset-expansion/baseline-manifest.json` registra fonte,
contagens por split, hashes do CSV e checkpoints, cada PNG/label de validação,
metadata, configuração visual, versão Securimage e esquema RNG. Preserva também
a identidade da fonte já ampliada para detectar alterações posteriores.

Execute na raiz do repositório, com PHP CLI e GD/FreeType disponíveis:

```powershell
.\.venv-ml\Scripts\python.exe scripts/prepare_expanded_dataset.py --php php
```

Se PHP não estiver no PATH, substitua `php` pelo caminho do executável entre
aspas. A preparação usa cópias convencionais independentes dos 3.000 PNGs
históricos, CSV/metadata e labels; não usa links. Só então chama o gerador
existente para gerar 6.300 novas imagens no **novo destino**:

```text
php generator/generate.php --count 6300 --seed 42 --train-ratio 1 --validation-ratio 0 --output dataset/synthetic/v2_9k_train
```

O comando acima ilustra a etapa interna; use o script para realizar as
verificações antes/depois. Os índices novos continuam de 3.001 a 9.300.
Nenhuma amostra histórica é regenerada. As configurações de charset, resolução,
comprimentos, domain randomization, `xoshiro256starstar-sha256-v2` e Securimage
4.0.2 precisam coincidir com a referência. Os parâmetros visuais não mudam.

O resultado obrigatório é `dataset/synthetic/v2_9k_train/{train,validation}`
e seu `labels.csv`, com exatamente 9.000/300 PNGs, labels completos, todas as
referências existentes e nenhuma duplicata exata SHA-256. As 300 imagens de
validação precisam ter os mesmos nomes, labels, metadata e bytes de A; a
verificação falha para qualquer diferença. Os 2.700 PNGs de treino históricos
também são comparados por hash. A fonte, seu CSV e os checkpoints são conferidos
novamente após a preparação, sem alterações.

Uma reexecução confere o destino e informa conclusão se estiver completo e
válido. Um destino incompatível ou uma preparação interrompida recebe
diagnóstico; os dados são preservados e nenhuma exclusão automática ocorre.
O script não completa automaticamente uma expansão interrompida após a geração
ter começado. Preserve os arquivos e revise o diagnóstico antes de uma
recuperação manual. O esquema RNG é derivado do índice de cada amostra e
permite reproduzir seus parâmetros independentemente da invocação; isso não
dispensa verificar CSV, arquivos e metadata depois de uma interrupção.
Para conferir todos os artefatos completos sem criar ou escrever arquivos:

```powershell
.\.venv-ml\Scripts\python.exe scripts/prepare_expanded_dataset.py --verify-only
```

Se o PHP não estiver no PATH, informe também `--php "caminho\php.exe"` na
verificação; ela consulta as receitas do gerador sem produzir imagens.

## Scheduler de C

`--scheduler none` mantém o comportamento fixo e é o padrão.
`--scheduler plateau` usa `ReduceLROnPlateau(mode="min")` sobre **CER da validação
sintética**, nunca sobre train loss. Defaults:

| Argumento | Valor |
| --- | ---: |
| `--plateau-start-epoch` | 10 |
| `--plateau-patience` | 8 |
| `--plateau-factor` | 0.5 |
| `--plateau-min-delta` | 0.002 |
| `--plateau-cooldown` | 2 |
| `--plateau-min-lr` | 0.00001 |

As épocas 1–10 registram métricas, mas não chamam o scheduler nem contam
estagnação. A época 11 estabelece a primeira referência. `threshold_mode=abs`
significa uma melhoria absoluta **estritamente maior** que 0,002 no CER;
não é um percentual relativo. Patience 8 tolera oito observações ruins e reduz
na nona. Sem qualquer melhoria, a primeira redução possível é após a época 20.
Cooldown 2 pausa a contagem após a redução; o LR nunca fica abaixo de 0,00001.
NaN/Inf no CER interrompe com diagnóstico.

Cada época faz treino, validação, métricas, `step(CER)` quando habilitado,
registro de LR e checkpoint. `history.csv` preserva as colunas anteriores e
acrescenta `lr_used`, `lr_next` e `lr_reduced`. O LR usado é o aplicado naquela
época; o seguinte incorpora a decisão para a próxima. O console destaca as
reduções. O scheduler recebe uma observação por época e nenhuma por batch.

`best.pt` continua selecionado pelo menor CER; no empate, pelo maior exact
match; o empate em ambos preserva o best anterior. Essa seleção é independente
do threshold e da contagem de estagnação do scheduler.

## Comandos manuais

Os dois comandos começam do zero, sem `--resume`. Não carregue pesos de A em B
ou C, nem pesos de B em C. Mesmo dataset, seed 42, arquitetura, batch size,
threads e protocolo de embaralhamento garantem a comparação inicial controlada.
Mantenha o dataset inalterado entre as execuções e use as mesmas versões.

Experimento B — dataset ampliado, LR fixo:

```powershell
.\.venv-ml\Scripts\python.exe -m ml.train --data dataset/synthetic/v2_9k_train --device cpu --epochs 30 --batch-size 16 --learning-rate 0.001 --seed 42 --scheduler none --run-dir outputs/runs/crnn-v2-9k-fixed
```

Experimento C — dataset ampliado, LR adaptativo:

```powershell
.\.venv-ml\Scripts\python.exe -m ml.train --data dataset/synthetic/v2_9k_train --device cpu --epochs 30 --batch-size 16 --learning-rate 0.001 --seed 42 --scheduler plateau --plateau-start-epoch 10 --plateau-patience 8 --plateau-factor 0.5 --plateau-min-delta 0.002 --plateau-cooldown 2 --plateau-min-lr 0.00001 --run-dir outputs/runs/crnn-v2-9k-plateau
```

Diretórios de execução novos precisam estar vazios. Os treinamentos B/C não
foram iniciados pela preparação. Nenhum comando usa GPU, DirectML ou servidor.

## Verificações curtas e estimativa CPU

```powershell
.\.venv-ml\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-ml\Scripts\python.exe -m ml.train --data dataset/synthetic/v2_9k_train --device cpu --batch-size 16 --scheduler none --dry-run
.\.venv-ml\Scripts\python.exe -m ml.train --data dataset/synthetic/v2_9k_train --device cpu --batch-size 16 --scheduler plateau --dry-run
.\.venv-ml\Scripts\python.exe -m scripts.benchmark_cpu --data dataset/synthetic/v2_9k_train --output outputs/experiments/dataset-expansion/cpu-benchmark.json
git diff --check
```

Os testes usam fixtures pequenos, simulações de CER e treinamento curto em
temporários; não geram milhares de imagens nem iniciam B/C. Os dry-runs
verificam forward, CTC, backward, gradientes finitos e mudança dos pesos em
memória, além de um batch de validação, sem checkpoints. O dry-run plateau
exercita uma observação diagnóstica após o período inicial virtual.

O benchmark usa CPU, batch 16, seis threads, dois batches de warmup e seis
batches medidos de treino; validação tem um warmup e três batches medidos.
Não completa uma época. O JSON registra média por batch, 563 batches de treino
e 19 de validação por época, estimativa por época e por 30 épocas, e memória
do processo quando disponível. A estimativa inclui validação e pode variar
com carga e I/O. Não grava checkpoints.

Medição local de preparação (CPU, seis threads, batch 16): média de **0,242813 s
por batch de treino** e **0,084493 s por batch de validação**. Com 563 + 19
batches, a estimativa é **138,31 s por época** e **4.149 s (69,2 minutos) por
experimento de 30 épocas**. O processo usava 443,8 MiB ao final e atingiu pico
de 637,8 MiB. A inspeção inicial dos arquivos levou 9,08 s, medida separadamente.
São oito atualizações em memória (dois warmups e seis batches medidos); nenhuma
época completa ou checkpoint. A projeção exclui gravação de checkpoints e
variação entre épocas; B e C podem ter durações diferentes com carga diferente.
Os valores completos estão em `outputs/experiments/dataset-expansion/cpu-benchmark.json`.

Na verificação desta preparação, **234 testes passaram**, os dois dry-runs CPU
passaram, e a preparação repetida e `--verify-only` mantiveram o resultado sem
chamar a geração. A auditoria conferiu 9.314 arquivos protegidos sem alterações,
incluindo todos os PNGs da fonte, o CSV, os quatro artefatos do baseline e o
gerador/configuração/modelo. As 6.300 imagens novas geradas independentemente
também reproduziram os hashes da expansão já encontrada na fonte local.

## Checkpoint e retomada

Cada execução mantém `last.pt`, `best.pt`, `history.csv` e `run_config.json`
no seu próprio diretório. O checkpoint guarda pesos, AdamW com LR atual,
época concluída, métricas, best, charset, identidade do dataset e RNGs
Python/NumPy/PyTorch/DataLoader. Com plateau guarda também `scheduler_config`
e `scheduler_state_dict`, após a decisão da época.

Para retomar B, repita seu comando e acrescente
`--resume outputs/runs/crnn-v2-9k-fixed/last.pt`. Para C, repita **todos** os seus
argumentos e acrescente `--resume outputs/runs/crnn-v2-9k-plateau/last.pt`.
`--epochs 30` significa o total desejado, incluindo as épocas já concluídas.
A retomada começa na época seguinte, restaura otimizador, LR, scheduler e RNGs,
e não repete a decisão já salva. Uma interrupção no meio de uma época perde
as atualizações não salvas dessa época.

Checkpoints antigos sem scheduler carregam com `--scheduler none`.
Ativar plateau ao retomar um checkpoint legado, trocar a política ou alterar
parâmetros do scheduler resulta em erro claro. Dataset, arquitetura, batch,
seed, workers e threads também precisam ser compatíveis. Não retome nem
modifique A durante esta preparação.

## Comparação posterior e limites

Compare A/B para investigar volume de dados com LR fixo e B/C para investigar
a política de LR no mesmo dataset. Use CER e exact match nas mesmas 300 imagens,
época do melhor checkpoint, curvas de losses, LR usado/seguinte, eventos de
redução, tempo e número de updates. Preserve hashes e versões junto dos resultados.
Para avaliar A, use o snapshot `baseline-dataset`, cujo CSV coincide com o
fingerprint histórico; use `v2_9k_train` para B/C. Só validação sintética participa.

Os orçamentos são `ceil(2700/16) × 100 = 16900` para A e
`ceil(9000/16) × 30 = 16890` para B/C. Épocas e cardinalidade diferem: cada imagem
de A é apresentada 100 vezes e cada imagem de B/C, 30 vezes. O total de
apresentações é, entretanto, **270.000 nos três experimentos**. O último batch
tem 12 imagens em A e 8 em B/C. A comparação é preliminar: uma única seed
não oferece conclusões estatísticas fortes e a validação já orientou A.

Se C não reduzir o LR em 30 épocas, registre que a política não entrou em ação
nesse orçamento. B e C podem terminar equivalentes; isso não demonstra
superioridade ou inferioridade do scheduler. Não prolongue o treinamento
automaticamente para forçar um resultado. Não há fine-tuning, dados reais,
avaliação `real/test`, acesso remoto, A2000 ou alterações na Aegis nesta etapa.
