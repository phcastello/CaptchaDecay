# Treinamento CUDA no Ubuntu — experimento D

Ambiente preparado e verificado em 2026-10-20 para a NVIDIA RTX A2000 12GB.
Dataset, testes, dry-run e benchmark foram concluídos nessa preparação.
Posteriormente, o experimento D foi treinado até a época 65, encerrando por
early stopping; o melhor checkpoint é o da época 46, com 268/300 acertos
sintéticos e CER 40/1507, reproduzidos em CPU no experimento E.

O [checkpoint publicado](../outputs/runs/crnn-v2-20k-cuda-plateau/README.md)
documenta o arquivo original e seu SHA-256. A comparação real está no
[relatório do experimento E](../outputs/analysis/real-transfer-expanded/REPORT.md).
Os inventários em `docs/environment/` registram a preparação de 2026-10-20;
seus campos sobre treinamento ainda não iniciado descrevem aquele momento.
Os comandos de treinamento abaixo são mantidos para reprodução manual.

## Servidor e dependências

| Item | Versão/resultado |
| --- | --- |
| Sistema | Ubuntu 24.04.4 LTS (Noble), x86_64 |
| Kernel | Linux 6.8.0-139-generic |
| Python | 3.12.3 em .venv-ml, sem pacotes globais |
| GPU física | 0, NVIDIA RTX A2000 12GB; nvidia-smi indica 11514 MiB |
| UUID | GPU-bc252296-7ce0-02fb-acbd-4f371c943715 |
| Driver | 535.309.01, preservado |
| PyTorch | 2.14.1+cu126 |
| Runtime CUDA do PyTorch | 12.6 |
| NumPy / Pillow | 2.5.4 / 12.3.0, conforme requirements-ml.txt |
| PHP do sistema | 8.3.6 CLI, pacote Ubuntu 8.3.6-0ubuntu0.24.04.12 |
| GD | 2.3.3, suporte FreeType e PNG confirmado |
| FreeType do sistema | libfreetype6 2.13.2+dfsg-1ubuntu0.2 |
| Composer do sistema | 2.7.1 |
| Securimage | 4.0.2, instalação pelo composer.lock |
| Recursos iniciais | 23 GiB RAM, 21 GiB disponíveis; disco livre aproximadamente 17 GiB |

O repositório foi atualizado por fast-forward para
`13d7cad3023278600080a80d1c52bd9bbc86e4f2`, que contém a implementação
original dos experimentos A/B/C. As alterações locais foram preservadas.
A arquitetura original não foi modificada: 1.186.815 parâmetros, entrada
1 × 80 × 215, CNN/BatchNorm, BiLSTM e classificação CTC.

O `nvidia-smi` mostra CUDA 12.2 como versão associada ao driver, enquanto o
wheel usa runtime 12.6. Essa combinação usa a
[compatibilidade entre versões menores CUDA 12.x](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html),
com driver mínimo 525. A execução real na A2000 foi verificada; não depende
somente desse número de versão. A versão 2.14.1 está publicada nos
[canais oficiais do PyTorch](https://dev-discuss.pytorch.org/t/pytorch-2-14-1-general-availability/3455);
a [documentação de lançamento](https://pytorch.org/blog/pytorch-2-14-release-blog/)
identifica o canal CUDA 12.6.

## Instalação específica deste servidor

Na raiz `/home/phc/CaptchaDecay`:

```bash
python3 -m venv .venv-ml
./.venv-ml/bin/python -m pip install --no-cache-dir -r requirements-ml-cuda.txt
./.venv-ml/bin/python -m pip check
```

O venv usa `include-system-site-packages = false` e desabilita o user site.
Todas as instalações pip foram realizadas com seu executável, sem sudo.
O arquivo CUDA inclui os requisitos originais e fixa explicitamente
`torch==2.14.1+cu126`; a configuração Windows permanece intacta.
O inventário exato está em `docs/gpu-python-freeze.txt`.

Os wheels trazem as bibliotecas necessárias. O metapacote pip chamado
`cuda-toolkit` é uma dependência do próprio wheel com extras de runtime;
não foi instalado o Toolkit completo do sistema, nvcc ou drivers.

O usuário instalou no sistema:

```bash
sudo apt-get update
sudo apt-get install --no-install-recommends php-cli php-gd php-mbstring php-xml php-curl php-zip composer
composer install --no-interaction --prefer-dist
composer check-platform-reqs
php --version
php -m
php -r 'var_export(gd_info());'
composer --version
```

Não foi necessário editar `/etc/php/8.3/cli/php.ini`. GD/FreeType, PNG, fonte
AHGBold.ttf e requisitos Composer foram conferidos. O lockfile não foi alterado.
O Composer informa que Securimage está abandonado upstream; foi mantida a versão
histórica exigida pelo experimento.

Antes da instalação do sistema, foi preparado um runtime opcional em `.tools/`.
Ele continua disponível para reproduzir a geração já em andamento:
`scripts/setup-php-local.sh`, `scripts/php-local` (PHP 8.3.6) e
`scripts/composer-local` (Composer 2.10.3). O setup usa pacotes oficiais Ubuntu,
extraídos sem scripts de instalação; pode ser reexecutado. O instalador Composer
foi validado por SHA-384 conforme a
[documentação oficial](https://getcomposer.org/doc/faqs/how-to-install-composer-programmatically.md).

## Dataset e validação histórica

Fonte original: `dataset/synthetic/validation-300/`. Os 304 arquivos versionados
coincidiram byte a byte com Git antes da preparação. Há 300 PNGs de 215 × 80,
1.227.375 bytes, 1.507 caracteres; comprimentos 4/5/6: 84/125/91.

| Identificação histórica | SHA-256 |
| --- | --- |
| labels.csv | `2158f8746674f893770245220177634d69cd77a6298373e8b74d0f9a9b1d7b95` |
| manifest.json | `0bbdc3f7a7b74206466444f98855bab13d28d8016e1615bef3382fc65abad597` |
| Registros dos 300 PNGs | `393d05912b302a8363b8186c6a7074e23f5a4e5f62d1564d334d063c30387b63` |

Preparar e verificar sem sobrescrever datasets existentes:

```bash
./.venv-ml/bin/python generator/prepare_experiment_d.py --php "$PWD/scripts/php-local"
./.venv-ml/bin/python generator/prepare_experiment_d.py --verify-only
```

Destino: `dataset/synthetic/v2_20k_train/`, com `train/`, `validation/` e
`labels.csv`. Como as 9.000 imagens históricas de treino não estavam disponíveis,
as 20.000 imagens de treino são novas, geradas localmente com seed **20261020**.
A seed de treinamento é **42**, independente da geração. A validação foi
copiada, sem regeneração, preservando nomes, PNGs, labels e todos os 19 campos
de metadados. O CSV histórico permanece como prefixo byte a byte do CSV final.

O gerador oficial permanece intacto: Securimage 4.0.2, RNG
`xoshiro256starstar-sha256-v2`, charset
`abcdefghiklmnprstuvwyz23456789`, comprimentos 4/5/6 com pesos 27/46/27,
mesma configuração visual v2 e resolução. Novos índices: 3000 até 22999.
A seed histórica 42 não é reutilizada na geração.

O helper verifica fonte e resultado, PNGs decodificados/CRC/dimensões, CSV,
metadados, contagens, charset, hashes únicos e ausência de validação no treino.
A publicação final é atômica; destinos existentes são apenas verificados.
Uma falha preserva o diretório de staging indicado no erro, sem substituir dados.

Arquivos de proveniência no dataset:

- `historical-validation/{labels.csv,manifest.json}`: cópias históricas originais.
- `generation.json`: parâmetros, seeds, versões, fonte/biblioteca e hashes do código/configuração/lockfile.
- `integrity-manifest.json`: SHA-256 de todos os 20.300 PNGs, CSV e registros.

Imagens geradas, vendor, venv, ferramentas e checkpoints são ignorados no Git.
A comparação com B/C envolve nova amostragem do gerador além do aumento de
quantidade. A validação fixa não é um teste independente. PNGs regenerados
podem variar entre plataformas/versões PHP/GD/FreeType; o runtime local,
o hash da fonte e da biblioteca estão registrados.

## CUDA, smoke test e benchmark

Verificação básica do runtime:

```bash
CUDA_VISIBLE_DEVICES=GPU-bc252296-7ce0-02fb-acbd-4f371c943715 ./.venv-ml/bin/python scripts/check_cuda.py
```

Esse script testa primitivas CUDA; o dry-run abaixo usa a CRNN original.
Não há fallback silencioso para CPU. A loss move targets para o dispositivo
dos logits, mantém comprimentos CPU/int64 e executa CTC/backward na GPU.
As verificações incluem CNN, BatchNorm, BiLSTM, Linear, LogSoftmax, gradientes,
AdamW, clipping, atualização de pesos, validação, greedy e checkpoint/resume.

```bash
CUDA_VISIBLE_DEVICES=GPU-bc252296-7ce0-02fb-acbd-4f371c943715 ./.venv-ml/bin/python -m ml.train \
  --data dataset/synthetic/v2_20k_train --device cuda --batch-size 16 --seed 42 \
  --scheduler plateau --plateau-start-epoch 10 \
  --early-stopping-patience 25 --early-stopping-min-delta 0.002 \
  --early-stopping-start-epoch 11 --dry-run
```

Benchmark curto, 3 batches de warmup + 8 cronometrados de treino e
1 warmup + 3 cronometrados de validação:

```bash
CUDA_VISIBLE_DEVICES=GPU-bc252296-7ce0-02fb-acbd-4f371c943715 ./.venv-ml/bin/python -m scripts.benchmark_cuda \
  --data dataset/synthetic/v2_20k_train --device cuda:0 --batch-size 16 \
  --warmup 3 --batches 8 --validation-batches 3 \
  --output docs/environment/benchmark-cuda.json
```

O caminho de relatório deve estar livre. O benchmark sincroniza CUDA antes
e depois de cada batch, incluindo carregamento, transferência, treino e
validação; usa o train_step original. São 1250 batches de treino e 19 de
validação por época. As estimativas excluem inicialização, I/O de checkpoints
e variação entre épocas; a verificação do manifesto aquece o cache de disco.
Nenhuma época inteira é executada e os pesos temporários são descartados.
Treino e benchmark usam float32 estrito, sem AMP nem TF32.

## Early stopping, scheduler e checkpoints

Early stopping permanece desabilitado por padrão (`--early-stopping-patience 0`).
No D monitora CER a partir da época 11, paciência 25, delta absoluto 0.002.
A primeira observação estabelece a referência; somente
`CER < referência - 0.002` reinicia a paciência. Melhorias menores podem
atualizar best.pt sem reiniciar essa contagem.

O ReduceLROnPlateau existente continua após a validação, uma decisão por época,
modo min, factor 0.5, patience 8, threshold 0.002 absoluto, cooldown 2,
min_lr 0.00001. `--plateau-start-epoch 10` ignora épocas 1–10 e observa
a partir de 11. CER constante produz reduções nas épocas 20 e 31 antes de
o early stopping encerrar na 36. LR utilizado/próximo, reduções e contadores
de estagnação ficam no histórico.

Diretório: `outputs/runs/crnn-v2-20k-cuda-plateau/`:

- `best.pt`: menor CER, com desempate histórico por maior exact match.
- `last.pt`: última época completa, inclusive a decisão que encerra o treino.
- `history.csv` e `run_config.json`: métricas, LR, configuração e fingerprint.

Os checkpoints guardam modelo, otimizador, scheduler, early stopping (configuração,
referência, contadores, observações e stopped), histórico, versão e RNG Python,
NumPy, Torch CPU/CUDA e DataLoader. O carregamento usa map_location CPU; a
restauração leva os estados ao modelo/otimizador no dispositivo escolhido.

Resume recusa mudança de configuração, dados ou charset e não zera a paciência.
Um checkpoint antigo sem estado de early stopping é aceito com o recurso
desabilitado, mas não pode ativá-lo silenciosamente. Um checkpoint já encerrado
por early stopping não reinicia o treinamento: a CLI verifica e repara best.pt e
history.csv a partir de last.pt caso a interrupção tenha ocorrido entre as
gravações, informa que a execução já terminou e retorna sem novos passos.

## Treinamento manual e resume

O launcher seleciona explicitamente a A2000 por UUID, verifica a integridade do
dataset e passa todos os argumentos abaixo. O primeiro comando inicia do zero;
não usa checkpoints A/B/C. Ele não foi executado durante a preparação.

```bash
./scripts/train-experiment-d.sh
```

Após interrupção, restaurar last.pt com a mesma configuração:

```bash
./scripts/train-experiment-d.sh --resume
```

Comando expandido de início, equivalente ao launcher:

```bash
CUDA_VISIBLE_DEVICES=GPU-bc252296-7ce0-02fb-acbd-4f371c943715 ./.venv-ml/bin/python -m ml.train \
  --data dataset/synthetic/v2_20k_train --device cuda \
  --epochs 100 --batch-size 16 --learning-rate 0.001 --seed 42 \
  --scheduler plateau --plateau-start-epoch 10 --plateau-patience 8 \
  --plateau-factor 0.5 --plateau-min-delta 0.002 --plateau-cooldown 2 \
  --plateau-min-lr 0.00001 --early-stopping-patience 25 \
  --early-stopping-min-delta 0.002 --early-stopping-start-epoch 11 \
  --run-dir outputs/runs/crnn-v2-20k-cuda-plateau
```

Para retomar pelo comando expandido, acrescente
`--resume outputs/runs/crnn-v2-20k-cuda-plateau/last.pt` a todos os mesmos argumentos.
`--epochs 100` é o total pretendido, não 100 épocas adicionais.

Acompanhar a GPU:

```bash
watch -n 1 nvidia-smi
```

Avaliação sintética posterior, somente após existir best.pt:

```bash
CUDA_VISIBLE_DEVICES=GPU-bc252296-7ce0-02fb-acbd-4f371c943715 ./.venv-ml/bin/python -m ml.evaluate \
  --data dataset/synthetic/v2_20k_train --split validation \
  --checkpoint outputs/runs/crnn-v2-20k-cuda-plateau/best.pt --device cuda --batch-size 16
```

## Verificação automatizada

```bash
./.venv-ml/bin/python -m unittest discover -s tests -v
git diff --check
```

PHP já está no PATH do sistema; para reproduzir com o runtime opcional, prefixe
o comando de testes com `PHP_BINARY="$PWD/scripts/php-local"`.
Testes GPU são identificados e pulados explicitamente quando CUDA não está
disponível; as demais verificações funcionam em CPU. Testes do helper de
publicação Linux são pulados fora desse sistema.

A suíte usa fixtures sintéticas temporárias e mocks de rede. Nenhuma imagem
real foi coletada, avaliada ou utilizada para treinar; não houve acesso a
Correios nem uso de real/test. Não foram criados serviços, jobs, execução de
treino em background, commits ou pushes.

Suíte completa executada com o Python do venv: **300 testes passaram**, incluindo
os testes GPU, sem skips. O processo levou 29,182 s. Os testes CPU emitiram
avisos de NNPACK por instruções não disponíveis na CPU virtualizada, sem falhas;
os testes CUDA passaram normalmente. A ocultação explícita de CUDA também foi
verificada: os testes CPU continuam passando e os três testes da CRNN GPU são
pulados com motivo explícito.

Resultado final do dataset: **20000 treino + 300 validação**, 20300 SHA-256
únicos, nenhuma duplicata ou sobreposição entre splits. Todos os PNGs decodificados
têm 215 × 80; labels/CSV/metadados válidos. As 300 validações, nomes, labels e
metadados coincidiram com o histórico antes e depois da preparação. O CSV
histórico mantém SHA-256 `2158f8746674f893770245220177634d69cd77a6298373e8b74d0f9a9b1d7b95`.

Fingerprint usado pelo pipeline:
`b96386a19db32c50e837926b17937a1a7d3f30efc79dfcb28b402c7673abdb78`.
O manifesto completo fica no dataset; a identificação e receita para preservar
em Git estão em [experiment-d-dataset.json](environment/experiment-d-dataset.json).

O [dry-run CUDA](environment/cuda-smoke.json) passou com batch 16, entrada
16 × 1 × 80 × 215, saída 53 × 16 × 31 e 1.186.815 parâmetros. CTC e gradientes
finitos, pesos atualizados, validação/greedy e checkpoint temporário restaurado
em CUDA, incluindo otimizador/RNG/scheduler/early stopping. Tempo de modelo:
1,17 s; tempo total com verificação do dataset: 21,52 s. Não representa uma
medição de qualidade do modelo: os pesos estavam recém-inicializados.

[Benchmark completo](environment/benchmark-cuda.json), batch 16 e float32
estrito, sincronizado, incluindo os custos do pipeline:

| Medida | Resultado |
| --- | --- |
| Batch de treino, média de 8 após 3 de warmup | 49,63 ms |
| Desvio padrão por batch de treino | 0,22 ms |
| Batch de validação | 20,49 ms |
| Batches por época | 1250 treino + 19 validação |
| Estimativa por época, treino + validação | 62,42 s |
| Estimativa de 30 épocas | 31,21 min |
| Estimativa de 100 épocas | 104,04 min (1 h 44 min) |
| Pico de memória CUDA alocada | 289,77 MiB |
| Pico de memória reservada | 356 MiB |
| Memória CUDA livre antes do modelo | 10,91 GiB |

Foram apenas 11 atualizações em memória (3 de warmup + 8 medidas), sem uma época
completa e sem salvar pesos de benchmark. A carga inicial/verificação levou
20,06 s, fora das estimativas por época. As estimativas excluem I/O de checkpoints
e variação futura do modelo; não são uma promessa de duração. Batch 16 não causou
OOM. Após os diagnósticos não ficou processo de treinamento em execução.

## Arquivos desta preparação

Modificados: `.gitignore`, `ml/train.py` e `ml/README.md`.

Criados:

- `requirements-ml-cuda.txt`;
- `generator/prepare_experiment_d.py`;
- `scripts/train-experiment-d.sh`, `scripts/benchmark_cuda.py`, `scripts/check_cuda.py`;
- `scripts/setup-php-local.sh`, `scripts/php-local`, `scripts/composer-local`;
- `tests/test_ml_early_stopping.py`, `tests/test_ml_cuda.py`, `tests/test_cuda_runtime.py`,
  `tests/test_cuda_benchmark.py`, `tests/test_experiment_d_dataset.py`;
- `docs/gpu-training.md`, `docs/gpu-python-freeze.txt`;
- `docs/environment/cuda-smoke.json`, `docs/environment/benchmark-cuda.json`,
  `docs/environment/experiment-d-dataset.json`, `docs/environment/server.json`.

Arquivos locais ignorados: `.venv-ml/`, `.tools/`, `vendor/` e o dataset preparado
em `dataset/synthetic/v2_20k_train/`. Nenhuma imagem gerada foi adicionada ao Git.
O run de produção `outputs/runs/crnn-v2-20k-cuda-plateau/` ainda não foi criado.
