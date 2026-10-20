# CRNN sintética — experimentos controlados

## Avaliação de transferência em arquivos reais locais

`ml.compare_transfer` compara dois checkpoints congelados somente em
`dataset/real/validation`, exclusivamente em CPU. O treinamento e as CLIs
sintéticas continuam recusando dados reais. Não há integração web ou produção.

```powershell
.\.venv-ml\Scripts\python.exe -m ml.compare_transfer --checkpoint-a outputs/runs/crnn-v2-baseline/best.pt --checkpoint-b outputs/runs/crnn-v2-9k-fixed/best.pt --data dataset/real --output outputs/analysis/real-transfer
```

O diretório de saída deve ser novo ou vazio e separado dos dados/checkpoints.
Para auditar sem inferência, acrescente `--audit-only` e escolha outra saída.
A CLI aceita somente `--split validation`, batch 16 e seis threads por padrão;
`--batch-size` e `--threads` são configuráveis, sem opção de GPU.

O CSV compartilhado e `labeling_state.json`, se existir, são consultados, mas
nenhum PNG de train/test é aberto. Correspondência CSV/PNG, integridade, hashes,
nomes duplicados e charset são verificados. Duplicatas que envolvam outros
splits são verificadas somente pelos hashes do CSV, sem abrir suas imagens.
Todos os membros de uma duplicata são excluídos, assim como labels fora do
charset ou marcados para revisão. **Qualquer label vazio em validation bloqueia
toda a avaliação**, com exit code 2 e somente artefatos de auditoria. Nenhum
label é preenchido automaticamente. A revisão manual usa:

```powershell
.\.venv-ml\Scripts\python.exe -m labeling.labeler --dataset dataset/real --split validation
```

Neste protocolo, uma decisão humana anterior à inferência exclui também todas
as imagens cujo label contém `l` minúsculo: a ambiguidade visual com `L`/`I`
maiúsculos impede uma referência confiável em alguns casos. Não há conversão
de caracteres ou caixa nem equivalência entre eles. Os percentuais passam a
descrever apenas imagens elegíveis sem `l`. O rotulador registra labels salvos
e marcas de revisão, mas não mantém uma confirmação positiva de dupla revisão.

`eligibility.json`, `review-manifest.json`, `checkpoints.json` e `exclusions.csv`
registram a seleção antes das previsões. `predictions.csv` contém cada imagem
elegível com previsões, acertos, distâncias e comprimentos de A/B;
`summary.json` contém métricas globais, por comprimento, S/D/I, confusões,
concordância e listas de melhorias/pioras por distância. `comparison-gallery.png`
inclui todas as elegíveis e `REPORT.md` apresenta resultados e limitações.
As métricas reutilizam o alinhamento existente; CER é ponderado pelo total de
caracteres de referência. Os modelos usam exatamente o mesmo subconjunto;
nenhuma exclusão é contada como acerto ou erro. Não são realizados testes de
significância ou ajustes de hiperparâmetros. Labels, PNGs e checkpoints têm
seus hashes conferidos ao final, e os pesos em memória permanecem idênticos.

Os testes de transferência usam pesos aleatórios congelados em temporários,
sem treinamento. A suíte completa histórica também contém testes de treino
e geração com fixtures temporários; esses não são parte da avaliação A/B.

O pipeline usa exclusivamente arquivos sintéticos locais, PNG de 215 × 80 pixels
e labels com 4–6 caracteres. O experimento A já foi concluído: 2.700 imagens de
treino, 300 de validação, 100 épocas, batch 16 e LR fixo 0,001. Seu melhor
checkpoint, na época 95, obteve exact match 61,6667%, CER 0,106171 e train loss
0,004282. Seus artefatos em `outputs/runs/crnn-v2-baseline` são preservados.

Esta etapa prepara B (9.000 imagens, LR fixo) e C (os mesmos dados, LR adaptativo),
ambos com 30 épocas e inicialização nova. **Os treinamentos B e C não são
executados na preparação.** O dry-run faz uma atualização em memória; testes
de treinamento e retomada usam somente fixtures pequenos em temporários.
O protocolo e os comandos estão também em [experiments/README.md](../experiments/README.md).

## Instalação no PowerShell

Execute na raiz do repositório. Python 3.12 é recomendado. O ambiente `.venv-ml`
é independente de `.venv-directml` e não modifica o Python global.

```powershell
py -3.12 -m venv .venv-ml
.\.venv-ml\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv-ml\Scripts\python.exe -m pip install -r requirements-ml.txt
.\.venv-ml\Scripts\python.exe -m pip check
```

Instale o wheel CPU **antes** do requirements para selecionar explicitamente
a distribuição oficial CPU do PyTorch. `requirements-ml.txt` fixa as dependências
diretas e permite uma instalação CUDA explícita futura. O `requirements.txt`
original permanece independente. Não é necessário ativar o ambiente.

Ambiente verificado: Python 3.12.10, PyTorch 2.14.1+cpu, NumPy 2.5.4 e Pillow
12.3.0. NumPy faz a conversão dos pixels; não usamos torchvision, TensorBoard,
serviços de tracking ou bibliotecas externas para CER.

Referências oficiais: [instalação PyTorch](https://pytorch.org/get-started/locally/)
e [CTCLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html).

## Dataset e charset

```text
dataset/synthetic/v2_9k_train/
  train/*.png
  validation/*.png
  labels.csv
```

O CSV precisa conter `filename,label,split,width,height`. Metadata adicional do
gerador é preservada; `length`, quando presente, precisa coincidir com o label.
Labels vêm do CSV, nunca do nome do arquivo. O loader verifica todas as linhas,
splits, nomes duplicados inclusive entre splits, caminhos, existência dos
arquivos, formato/integridade PNG, dimensões e labels. Erros identificam a
entrada. A identificação do dataset inclui SHA-256 do CSV e de todos os PNGs;
a retomada recusa mudanças de conteúdo. O dataset deve permanecer inalterado
durante a execução. A validação inicial lê todos os PNGs uma vez e os splits
reutilizam o mesmo manifesto.

O charset é a união **ordenada** dos caracteres dos labels de treino completo,
inclusive no modo overfit. O dataset atual tem 30 caracteres:
`23456789abcdefghiklmnprstuvwyz`. O blank CTC ocupa 0; caracteres ocupam 1–30,
totalizando 31 classes. Charset e mapeamentos ficam no checkpoint. Não há
conversão de caixa, expansão na validação, substituição de desconhecidos nem
alteração dos labels. Caracteres desconhecidos geram erro com imagem e caractere.

Cada imagem inteira é convertida para grayscale, float32, intervalo `[-1,1]`,
com shape `[1,80,215]`. Não há recorte, threshold, remoção de linhas, resize
ou augmentation. O collate empilha imagens e concatena targets, mantendo seus
comprimentos verdadeiros. Os splits permanecem separados. O pipeline recusa
`dataset/real` e fontes reais identificadas por metadata; não oferece split test.
O dataset real, seus backups e `real/test` não participam desta etapa.

Há uma particularidade do estado local: `dataset/synthetic/v2_main` já continha
9.000 imagens de treino e 300 de validação antes desta tarefa. Ele é preservado
exatamente como encontrado. As primeiras 3.000 linhas do CSV e seus PNGs
correspondem ao dataset A de 2.700/300; os fingerprints do CSV e dos PNGs
coincidem com os registrados no `run_config.json` histórico. A preparação cria
uma cópia independente dessa referência comprovada em
`outputs/experiments/dataset-expansion/baseline-dataset`, sem reescrever a fonte.
Esse snapshot permite reproduzir a identidade do dataset associada ao checkpoint
antigo, cujo fingerprint global não coincide com o `v2_main` já ampliado.

O dataset de B/C, `dataset/synthetic/v2_9k_train`, contém cópias convencionais dos
2.700 PNGs de treino e 300 PNGs de validação de A, mais 6.300 novos PNGs somente
de treino. Nomes, labels, splits, metadata e bytes das amostras antigas são
preservados. Não há hardlinks nem symlinks entre as cópias. O gerador PHP
existente continua do índice 3.001, com seed 42, `--train-ratio 1` e
`--validation-ratio 0`, conservando charset, resolução, distribuição de
comprimentos, configuração visual, RNG e versão Securimage. O gerador e sua
configuração visual não são alterados nesta etapa.

O manifesto de referência, os hashes de cada PNG, os labels e os parâmetros
de geração ficam em `outputs/experiments/dataset-expansion/baseline-manifest.json`.
A verificação exige 9.000/300 imagens, referências existentes e labels completos,
metadata antiga preservada, ausência de duplicatas SHA-256, igualdade byte a byte
dos 3.000 PNGs antigos e dos 300 labels/PNGs de validação. Ela também verifica
que a fonte e os checkpoints históricos permanecem iguais aos hashes anteriores.
A execução repetida confere um destino completo; uma preparação incompleta é
diagnosticada e preservada para recuperação explícita, sem apagar arquivos.

Use o script de preparação descrito em
[experiments/README.md](../experiments/README.md), em vez de executar o gerador
diretamente sobre qualquer dataset histórico. A CLI de treinamento não gera
imagens automaticamente se o CSV estiver ausente.

## Arquitetura e CTC

Implementação própria com módulos padrão, inicialização aleatória e sem pesos
pré-treinados. Cada bloco CNN contém Conv2d 3×3 com padding 1, BatchNorm2d,
ReLU e MaxPool2d. Os canais são 1 → 32 → 64 → 128 → 256; os pools são
2×2, 2×2, 2×1, 2×1. A altura final é reduzida por média, preservando a largura
como eixo temporal. A BiLSTM tem duas camadas, hidden size 128 por direção.

| Etapa | Shape |
| --- | --- |
| Entrada | `[B,1,80,215]` |
| CNN bloco 1 | `[B,32,40,107]` |
| CNN bloco 2 | `[B,64,20,53]` |
| CNN bloco 3 | `[B,128,10,53]` |
| CNN bloco 4 | `[B,256,5,53]` |
| Média vertical + transposição | `[B,53,256]` |
| BiLSTM | `[B,53,256]` |
| Linear e transposição | `[53,B,31]` no dataset atual |

São **1.186.815 parâmetros treináveis** com 31 classes. A quantidade de classes
é calculada a partir do charset. O forward retorna logits; o helper CTC aplica
`log_softmax(dim=2)` antes de `nn.CTCLoss(blank=0, zero_infinity=True)`.
O comprimento de entrada é obtido de `logits.shape[0]`, nunca de uma constante.

A CTC soma alinhamentos possíveis entre posições temporais e caracteres sem
exigir localização individual dos caracteres. Um alvo precisa de pelo menos
`len(label) + número de repetições consecutivas` posições: `aa` exige 3,
`ab` exige 2. Alinhamentos impossíveis são recusados explicitamente antes da
loss, para que `zero_infinity=True` não oculte problemas.

O greedy decoder escolhe a maior classe por timestep, colapsa repetições
consecutivas e remove blank. Assim, `a,a,blank,b,b` vira `ab`, enquanto
`a,blank,a` preserva `aa`. Não há beam search ou modelo de linguagem.

## Dry-run

```powershell
.\.venv-ml\Scripts\python.exe -m ml.train --data dataset/synthetic/v2_9k_train --device cpu --batch-size 16 --scheduler none --dry-run
.\.venv-ml\Scripts\python.exe -m ml.train --data dataset/synthetic/v2_9k_train --device cpu --batch-size 16 --scheduler plateau --dry-run
```

Valida o dataset, constrói charset e modelo, executa um batch de treino,
forward/CTC/backward, checa gradientes finitos e não nulos na CNN, clipping,
uma atualização AdamW e mudança efetiva dos pesos. Em seguida avalia um batch
de validação sem gradientes, decodifica e calcula métricas. Exibe dispositivo,
parâmetros, shapes, losses, métricas, previsões, duração total (`seconds`, com
validação dos arquivos) e das operações do modelo (`model_seconds`), terminando
em `DRY RUN PASSED` somente se tudo passar. Não cria run-dir nem checkpoints.

## Treinamento manual: B e C

Experimento B, dataset ampliado e LR fixo:

```powershell
.\.venv-ml\Scripts\python.exe -m ml.train --data dataset/synthetic/v2_9k_train --device cpu --epochs 30 --batch-size 16 --learning-rate 0.001 --seed 42 --scheduler none --run-dir outputs/runs/crnn-v2-9k-fixed
```

Experimento C, exatamente o mesmo dataset e LR adaptativo:

```powershell
.\.venv-ml\Scripts\python.exe -m ml.train --data dataset/synthetic/v2_9k_train --device cpu --epochs 30 --batch-size 16 --learning-rate 0.001 --seed 42 --scheduler plateau --plateau-start-epoch 10 --plateau-patience 8 --plateau-factor 0.5 --plateau-min-delta 0.002 --plateau-cooldown 2 --plateau-min-lr 0.00001 --run-dir outputs/runs/crnn-v2-9k-plateau
```

Ambos começam do zero, sem `--resume` e sem carregar pesos de A ou B. Com a
mesma seed 42, arquitetura, workers, threads e dataset, B/C têm a mesma
inicialização e sequência de embaralhamento. Seus diretórios são independentes.
Não use o diretório histórico do baseline como destino de novas execuções.

Defaults: CPU, 30 épocas, batch 16, learning rate 0.001, seed 42,
`--num-workers 0`, `--threads 6`. AdamW usa weight decay 0.0001;
clipping usa max norm 5.0. NaN/Inf na entrada, logits, loss, gradientes ou
parâmetros interrompem a execução com diagnóstico. A validação usa `eval()` e
`no_grad()`, sem atualizar pesos ou estatísticas BatchNorm.

Seeds controlam Python, NumPy, PyTorch e gerador de embaralhamento do DataLoader;
workers recebem seeds derivadas do PyTorch. São registradas versões, threads e
configurações. Isso não garante resultados bit a bit entre versões, sistemas
ou hardwares diferentes. No Windows, mantenha workers 0 inicialmente; o entry
point protegido permite aumentar workers explicitamente no futuro.

CPU sempre é o padrão. CUDA só é usada com `--device cuda`; se indisponível,
a CLI informa erro. Uma instalação PyTorch compatível com CUDA e uma GPU
NVIDIA serão necessárias nesse ambiente futuro. **DirectML não é suportado**:
o benchmark anterior encontrou incompatibilidade em BiLSTM, CTCLoss e Adam.
Nenhuma VM, A2000, configuração SSH ou serviço Aegis é acessado nesta etapa.

## ReduceLROnPlateau e CER

`--scheduler none` é o padrão e mantém o LR fixo. `--scheduler plateau` habilita
`torch.optim.lr_scheduler.ReduceLROnPlateau` com `mode=min`, monitorando somente
o CER da validação sintética. Train loss e validation loss não controlam o LR.

| Opção | Default | Significado |
| --- | ---: | --- |
| `--plateau-start-epoch` | 10 | Épocas iniciais sem observações do scheduler |
| `--plateau-patience` | 8 | Observações ruins toleradas antes de reduzir |
| `--plateau-factor` | 0.5 | Multiplicador do LR em cada redução |
| `--plateau-min-delta` | 0.002 | Melhoria absoluta mínima do CER |
| `--plateau-cooldown` | 2 | Observações de pausa na contagem após redução |
| `--plateau-min-lr` | 0.00001 | Limite inferior do LR |

Nas épocas 1–10 as métricas são registradas normalmente, sem chamar `step()`
e sem acumular estagnação. A época 11 fornece a primeira observação e estabelece
a referência. Isso protege a fase inicial de alinhamento CTC, em que o baseline
ficou sem exact matches durante as primeiras 30 épocas. `threshold_mode=abs`
exige `CER atual < melhor CER do scheduler - 0.002`; uma queda de exatamente
0,002 não é suficiente pela comparação estrita do PyTorch. O valor é uma
diferença absoluta de CER, sem interpretação como porcentagem relativa.

Uma melhoria suficiente zera a contagem de estagnação. Com patience 8, a redução
ocorre na nona observação ruim após a referência/melhoria, respeitando cooldown
e min LR. A primeira redução possível, com os defaults e CER sempre igual,
é após a validação da época 20 e afeta a época 21. CER NaN/Inf interrompe a
execução com diagnóstico antes de atualizar o scheduler.

O fluxo por época é treino, validação, cálculo das métricas, uma única chamada
`scheduler.step(CER)` após o período inicial, registro do LR seguinte e
checkpoint. `lr_used` é o LR efetivamente aplicado nos batches da época;
`lr_next` é o LR após a decisão, usado na próxima época; `lr_reduced` indica
uma redução. Esses campos são acrescentados a `history.csv`, preservando as
colunas anteriores. O console mostra os dois LRs e identifica cada redução.
Nenhuma decisão ocorre por batch.

A seleção de `best.pt` continua usando menor CER, com maior exact match no
desempate. Ela é independente da referência interna e do threshold do scheduler:
uma melhoria pequena pode selecionar um novo best sem reiniciar a paciência.

## Checkpoints, histórico e retomada

Uma execução nova exige run-dir vazio para proteger resultados existentes.
Artefatos locais em `outputs/` e pesos `.pt/.pth` são ignorados pelo Git.

- `last.pt`: estado ao final da época mais recente concluída.
- `best.pt`: menor CER de validação; em empate, maior exact match. Empate nas
  duas métricas preserva o checkpoint anterior.
- `history.csv`: losses médias ponderadas pelo número de imagens de cada batch,
  exact match, CER, imagens, exact match para comprimentos 4/5/6, duração,
  `lr_used`, `lr_next` e `lr_reduced`.
- `run_config.json`: argumentos, arquitetura, charset, identificação dos dados
  e versões.

Checkpoints contêm modelo, otimizador, época, configuração, seed, charset e
mapeamentos, métricas, melhor resultado anterior, histórico e RNGs de
Python/NumPy/PyTorch/DataLoader; CUDA RNG é guardado quando usado. Quando plateau
está habilitado, incluem também `scheduler_state_dict` e sua configuração;
o estado AdamW já contém o LR atual. O checkpoint corresponde à decisão de LR
já concluída naquela época. A escrita usa temporário no mesmo diretório,
flush/fsync e `os.replace`.

```powershell
.\.venv-ml\Scripts\python.exe -m ml.train --data dataset/synthetic/v2_9k_train --device cpu --epochs 30 --batch-size 16 --learning-rate 0.001 --seed 42 --scheduler none --run-dir outputs/runs/crnn-v2-9k-fixed --resume outputs/runs/crnn-v2-9k-fixed/last.pt
```

`--epochs` é o **total desejado**, não a quantidade adicional. A retomada começa
na época seguinte à salva e recupera o AdamW e o melhor checkpoint. Use o
`last.pt` e run-dir da execução original. Mantenha batch size, learning rate,
seed, workers, threads e opção de overfit originais; incompatibilidades de
dataset, charset, modelo ou configuração são recusadas. Para C, reutilize seu
comando completo com todos os argumentos plateau e acrescente
`--resume outputs/runs/crnn-v2-9k-plateau/last.pt`. Scheduler, configuração,
otimizador, LR e RNGs são restaurados antes da época seguinte, sem repetir a
decisão da época salva. Alterar o caminho do dataset sem mudar seus bytes é
permitido. Avaliação usa o charset salvo.

Checkpoints antigos sem scheduler continuam compatíveis com `--scheduler none`.
Solicitar plateau para um checkpoint legado, trocar o scheduler ou mudar seus
parâmetros durante a retomada produz erro explícito. Nenhum scheduler é criado
silenciosamente para um checkpoint incompatível. A preparação não retoma nem
sobrescreve os checkpoints históricos de A.

Uma interrupção no meio de uma época perde as atualizações daquela época;
retoma-se a partir da última concluída. `last.pt` é a fonte do histórico na
retomada. Se houver interrupção entre a gravação de `last.pt` e um novo
`best.pt`, a retomada reconstrói esse best a partir de last. Use um único
processo por run-dir. Não edite ou substitua arquivos de uma execução ativa.

## Overfit deliberado, separado do baseline

Disponível para execução posterior pelo usuário; não foi realizado um teste
prolongado de memorização:

```powershell
.\.venv-ml\Scripts\python.exe -m ml.train --data outputs/experiments/dataset-expansion/baseline-dataset --device cpu --epochs 100 --batch-size 16 --overfit-samples 16 --seed 42 --run-dir outputs/runs/crnn-overfit16
```

Essa opção seleciona 16 imagens de treino deterministicamente pela seed.
O charset continua vindo do treino completo e a validação permanece separada.
O teste ajuda a investigar memorização; não mede generalização. Use um run-dir
exclusivo e preserve a opção ao retomar.

## Avaliação offline

Após produzir um checkpoint manualmente:

```powershell
.\.venv-ml\Scripts\python.exe -m ml.evaluate --data outputs/experiments/dataset-expansion/baseline-dataset --split validation --checkpoint outputs/runs/crnn-v2-baseline/best.pt --device cpu --output-csv outputs/experiments/dataset-expansion/A-validation-predictions.csv
```

Aceita apenas os splits sintéticos `train` e `validation`. Restaura arquitetura,
pesos e charset; exibe loss média, exact match, CER, número de imagens e
acurácia por comprimento. `--output-csv` é opcional e grava
`filename,expected,predicted,correct`, recusando arquivo existente.
Nenhuma inferência real, página web, requisição HTTP ou envio de resposta
é implementado.

Exact match exige a string inteira idêntica. CER é a soma das distâncias de
Levenshtein dividida pela soma de caracteres reais em **todo** o split; não
é média de CERs individuais. Pode ultrapassar 1 por inserções. Acurácia por
comprimento usa o comprimento do ground truth, com quantidade de exemplos.
Comprimentos sem exemplos não recebem uma acurácia inventada.

## Testes e limites da verificação

```powershell
.\.venv-ml\Scripts\python.exe -m unittest discover -s tests -v
git diff --check
```

Os testes ML usam pequenos fixtures locais: charset, PNG/CSV, shapes reais,
CTC com repetições/alinhamentos inválidos, gradientes, decoder, métricas,
validação sem mutação, atualização de pesos, falhas numéricas, checkpoint
atômico, RNGs, otimizador e retomada equivalente à continuação. A integração
CLI usa dois exemplos artificiais de treino, uma única atualização por época,
e uma retomada curta em diretório temporário; verifica também a avaliação CSV.
Os testes existentes usam rede mockada e não coletam imagens reais. Para os
testes PHP existentes, PHP CLI com GD/FreeType precisa estar no PATH ou em
`$env:PHP_BINARY`; sem ele, essa parte da suíte informa skips.

Os testes adicionais verificam cópia independente, hashes e metadata, contagens,
duplicatas, reexecução segura e interrupção usando fixtures pequenos; testes de
scheduler simulam CER, paciência, threshold absoluto, cooldown, min LR e métricas
inválidas. Uma execução curta contínua é comparada com interrupção/retomada,
incluindo pesos, AdamW, RNGs, estado do scheduler e sequência de learning rates.
Os testes do protocolo conferem os argumentos de B/C sem iniciar treinamentos.

O dry-run no dataset ampliado testa a integração dos componentes, sem medir
qualidade do reconhecimento. Um benchmark CPU com warmup e poucos batches
estima custo, sem executar uma época completa; o relatório local fica em
`outputs/experiments/dataset-expansion`. O tempo real varia com carga, I/O e
temperatura. O reconhecimento real e fine-tuning ficam para outra etapa.

```powershell
.\.venv-ml\Scripts\python.exe scripts/prepare_expanded_dataset.py --verify-only
.\.venv-ml\Scripts\python.exe -m scripts.benchmark_cpu --data dataset/synthetic/v2_9k_train --output outputs/experiments/dataset-expansion/cpu-benchmark.json
```

O benchmark usa batch 16, seis threads, dois batches de warmup e seis medidos
de treino; validação usa um warmup e três medidos. Estima uma época com 563
batches de treino e 19 de validação e projeta 30 épocas. Registra memória do
processo quando disponível, sem checkpoints nem época completa.

## Análise do checkpoint congelado

Para diagnosticar o modelo já treinado, a CLI de análise reutiliza a mesma
inferência de `ml.evaluate` e o greedy decoder original. A arquitetura, os pesos,
os hiperparâmetros e a decodificação não são alterados.

```powershell
.\.venv-ml\Scripts\python.exe -m ml.analyze --data outputs/experiments/dataset-expansion/baseline-dataset --split validation --checkpoint outputs/runs/crnn-v2-baseline/best.pt --device cpu --output outputs/analysis/crnn-v2-baseline-check
```

A saída precisa ser nova ou vazia. Para repetir a execução, escolha outro
`--output`, por exemplo `outputs/analysis/crnn-v2-baseline-rerun`. A análise
aceita apenas validação sintética, usa `eval()` + `torch.inference_mode()` e
congela os parâmetros. Somente os PNGs do split solicitado são abertos; o CSV
compartilhado é lido para validar metadata. CPU continua sendo o padrão.

São exportados `predictions.csv` (todas as imagens com distância, CER individual
e S/D/I), `summary.json`, `REPORT.md`, tabelas por comprimento e repetições,
substituições, matriz de confusão e uma galeria de até 20 acertos e 20 erros.
Os gráficos usam Pillow quando Matplotlib não está instalado, sem dependências
novas. A galeria usa seed 42, estratificação por comprimento/tipos de erro e
até quatro extremos explicitamente identificados.

O alinhamento Levenshtein usa custo unitário e desempate no backtrace:
match exato, substituição, deleção e inserção. A decomposição não é uma
comparação ingênua de posições; S+D+I coincide com a distância mínima.
Confusões incluem somente substituições reais do alinhamento. Deleções e
inserções são contadas separadamente. Métricas globais e por grupo usam CER
ponderado pelos caracteres de referência.

O comando confere exact match, CER, quantidade e validation loss contra os
registros da época efetiva do checkpoint, sem ajustar resultados. Divergências
geram relatório explícito e exit code 1. A identidade do CSV precisa coincidir
com a salva. Hashes do checkpoint, labels e PNGs de validação são conferidos
antes/depois, assim como o estado do modelo em memória. O checkpoint antigo
possui fingerprint global dos PNGs, sem hashes individuais por split; essa
limitação histórica é documentada no relatório.

Na análise inicial do checkpoint da época 95, as 300 previsões reproduziram
185 acertos, exact match 61,6667%, CER 160/1.507 = 0,106171201062 e validation
loss 0,548292247852. O relatório inclui a preparação metodológica para comparar
futuramente 2.700 versus aproximadamente 9.000 imagens de treino, preservando
estas 300 imagens de validação. Nenhuma nova época ou imagem foi produzida
pela análise.

## Comparação posterior de A, B e C

| Experimento | Treino/validação | Épocas | LR | Updates com batch 16 |
| --- | --- | ---: | --- | ---: |
| A, concluído | 2.700/300 | 100 | 0,001 fixo | 16.900 |
| B, preparado | 9.000/300 | 30 | 0,001 fixo | 16.890 |
| C, preparado | 9.000/300 | 30 | Plateau, inicial 0,001 | 16.890 |

Compare os melhores CERs e seus exact matches nas mesmas 300 imagens,
época do best, curvas de treino/validação, `lr_used`/`lr_next`, número e momento
das reduções, duração e atualizações. Registre também os fingerprints e versões.
A/B investiga o aumento dos dados com LR fixo; B/C investiga a política de LR
com dados, inicialização e orçamento iguais.

O orçamento de updates é quase igual, mas cardinalidade, épocas e repetições
por imagem diferem: 100 apresentações por exemplo de A versus 30 em B/C.
O total de apresentações de treino é, por coincidência, **270.000 nos três**,
sem contar validação. O último batch tem 12 imagens em A e 8 em B/C. Uma única
seed não sustenta conclusões estatísticas fortes; a mesma validação já usada
em A também limita afirmações independentes de generalização.

Se C não reduzir o LR nas 30 épocas, B/C podem ter comportamento equivalente.
Registre que o scheduler não entrou em ação nesse orçamento; não atribua
superioridade ou inferioridade à política e não prolongue automaticamente os
treinamentos para provocar uma redução. Não há avaliação em `real/test`,
fine-tuning, acesso remoto ou uso de GPU nesta preparação.
