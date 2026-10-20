# CaptchaDecay — viabilidade de PyTorch + DirectML

Data: 18/10/2026. **Classificação C — DirectML incompatível com o treinamento nativo da CRNN representativa neste ambiente.**

**A RX 6700 XT é adequada para treinar nossa futura CRNN com PyTorch + DirectML? Não, com os pacotes e o driver testados.** A GPU executa operações convolucionais e seus gradientes, mas a BiLSTM não completa o forward nativo, a CTC Loss exige fallback para CPU e o Adam também encontra uma operação sem suporte. Não foi possível aprovar nem medir treinamento integrado DirectML. Não se trata de uma conclusão sobre capacidade computacional ou VRAM da placa, nem sobre outros backends AMD.

Recomendação: **CPU para desenvolvimento local e NVIDIA A2000 com CUDA para o treinamento completo**, após validar o mesmo caminho nessa alternativa. A A2000 não foi acessada nem medida nesta tarefa. A arquitetura definitiva continua pendente.

## Ambiente observado

| Item | Valor |
|---|---|
| Windows | Windows 11 Pro, 10.0.26300, build 26300 |
| CPU | AMD Ryzen 5 5600X, 6 núcleos / 12 threads lógicas |
| RAM informada pelo Windows | 34.255.376.384 bytes, aproximadamente 31,90 GiB |
| GPU física | AMD Radeon RX 6700 XT; 12 GB conforme inventário fornecido pelo usuário |
| Driver AMD | 32.0.21045.5002; data CIM 2026-08-17 UTC |
| Outros adaptadores Windows | Parsec Virtual Display Adapter; Microsoft Remote Display Adapter |
| Adaptadores enumerados pelo DirectML | Somente AMD Radeon RX 6700 XT, índice 0 |
| Dispositivo selecionado | `privateuseone:0` (GPU) e `cpu` |
| Python do benchmark | CPython 3.12.10, Windows AMD64 |
| Ambiente | `.venv-directml`, `include-system-site-packages = false` |
| PyTorch do benchmark | distribuição 2.4.1; runtime `2.4.1+cpu` |
| torch-directml | 0.2.5.dev240914 |
| torchvision | 0.19.1, dependência obrigatória declarada pelo torch-directml |
| Outras dependências observadas | NumPy 2.5.4; Pillow 13.0.0; lista completa em `requirements-directml-lock.txt` |
| Threads do PyTorch | 6 intra-op / 6 inter-op |
| Precisão | float32 para imagens, parâmetros, logits, loss e gradientes; índices CTC int64 |
| CUDA/ROCm | Nenhum instalado por esta tarefa; `torch.version.cuda = None` no venv |

O sufixo `+cpu` descreve a distribuição PyTorch base; o plugin DirectML acrescenta o backend `PrivateUse1`. Ele não significa que todos os tensores ficaram na CPU. As saídas e os gradientes dos componentes aprovados foram conferidos em `privateuseone:0`, com cópias explícitas para observação.

O README e a estrutura foram inspecionados antes das edições: o projeto contém geração, organização, rotulagem e inspeção de dados; ainda não contém um reconhecedor. O Python global já tinha `torch 2.14.1` e não tinha torch-directml. Esse ambiente não foi alterado. O venv `.venv-directml` **já existia com os pacotes acima**: foi inspecionado e reutilizado, sem reinstalação ou substituição de dependências. `pip check` retornou `No broken requirements found`.

### Disponibilidade dos pacotes e compatibilidade da instalação

A consulta ao índice nesta sessão encontrou PyTorch 2.14.1 como versão mais recente para o Python instalado. Para torch-directml, `pip index versions --pre` encontrou 0.2.5.dev240914, 0.2.4.dev240913, 0.2.4.dev240815, 0.2.3.dev240715, 0.2.2.dev240614 e 0.2.1.dev240521. Isso descreve as distribuições visíveis ao Python/plataforma desta máquina.

O [PyPI publica wheel CPython 3.12 para Windows x86-64](https://pypi.org/project/torch-directml/0.2.5.dev240914/). Seus [metadados](https://pypi.org/pypi/torch-directml/0.2.5.dev240914/json) exigem `torch==2.4.1` e `torchvision==0.19.1`. A [documentação Microsoft](https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows) ainda cita suporte até PyTorch 2.3.1; a escolha desta sessão segue o wheel e as dependências efetivamente publicados, além da execução local. Portanto Python 3.12 foi validado para instalação/importação, sem presumir compatibilidade das operações.

A soma de tensores `[1, 2]` em DirectML retornou `3.0` por leitura CPU bloqueante. Essa verificação inicial passou, mas não foi usada como critério de aprovação do treinamento.

## Protocolo

Script independente: `backend_probe.py`. Nenhum módulo de aplicação, imagem, label, charset real ou dataset do CaptchaDecay é carregado. Não há HTTP no script.

- Batches 4, 8 e 16; seed 42; mesmos dados artificiais, pesos iniciais, shapes e precisão nos dois backends.
- Cada diagnóstico aprovado completou 5 iterações; o script exige pelo menos 3 mesmo com `--iterations` menor.
- Conv2d, BatchNorm2d, ReLU, MaxPool2d e Linear têm testes separados; também há um bloco CNN combinado.
- BiLSTM isolada: `input_size=128`, `hidden_size=128`, 2 camadas, bidirecional, entrada `24 × B × 128`.
- CTC isolada: logits `24 × B × 17`, blank 0, `zero_infinity=True`, targets de comprimento 5. Cada target começa com um símbolo repetido e exige 6 passos; tanto T=24 quanto T=53 permitem alinhamento válido. Classes dos targets estão em 1..16. Não há perdas infinitas deliberadamente mascaradas.
- Gradientes são copiados explicitamente para CPU para conferir NaN/Inf, presença e magnitude. O gradiente da entrada deve ser não nulo; os gradientes treináveis devem ser não nulos em conjunto. Um bias individual pode ter gradiente zero por simetria. Todos os gradientes existentes devem ser finitos e permanecer no dispositivo correto.
- Adam com `foreach=False`, lr 0,001. O diagnóstico só aprova atualização se algum parâmetro mudar após cada step. ReLU, MaxPool e CTC isoladas não têm parâmetros treináveis.
- Sem mixed precision, `torch.compile`, ajuste de arquitetura para contornar incompatibilidades ou execução híbrida.

Modelo temporário integrado, definido exclusivamente no benchmark:

```text
B×1×80×215
→ Conv(1,8,3,pad=1), BN, ReLU, MaxPool(2)
→ Conv(8,16,3,pad=1), BN, ReLU, MaxPool(2)
→ B×16×20×53
→ permute/flatten → 53×B×320
→ Linear(320,128)
→ BiLSTM(128,128,2,bidirectional=True)
→ Linear(256,17)
→ log_softmax → CTCLoss → backward → Adam.step
```

Esse modelo serve apenas para diagnóstico sintético e não define a futura arquitetura do projeto.

## Compatibilidade medida

Resultados iguais nos três batches. “Bloqueado” significa que a operação exigia fallback ou não tinha implementação nativa; “não executado” não é aprovação.

| Operação / etapa | CPU | DirectML | Observações |
|---|---|---|---|
| Conv2d forward | Passou | Passou | Saída `B×8×80×215` |
| Conv2d backward | Passou | Passou | Entrada/pesos/bias com gradientes finitos |
| BatchNorm2d forward/backward | Passou | Passou | Modo treino; gradientes finitos |
| ReLU forward/backward | Passou | Passou | 5 iterações em ambos |
| MaxPool2d forward/backward | Passou | Passou | 5 iterações em ambos |
| CNN combinada forward/backward | Passou | Passou | Saída `B×16×20×53`; step DML falha em seguida |
| BiLSTM forward | Passou | Bloqueado / falhou | `aten::_thnn_fused_lstm_cell` |
| BiLSTM backward | Passou | Não executado | Forward nativo indisponível |
| Linear forward/backward | Passou | Passou | Step DML falha em seguida |
| CTC Loss forward | Passou | Bloqueado | `aten::_ctc_loss.Tensor` sem kernel DML/composite |
| CTC Loss backward | Passou | Não executado | Loss nativa indisponível |
| Adam / atualização dos pesos | Passou | Bloqueado | `aten::lerp.Scalar_out`; atingido em Conv, BN, CNN e Linear |
| Treinamento integrado | Passou | Pulado por incompatibilidade | CNN/Adam, BiLSTM e CTC reprovados como componentes nativos |

Na CPU, a rede integrada apresentou loss finita, backward válido e parâmetros alterados em cada uma das 5 iterações. No batch 8, as losses foram aproximadamente 25,1685; 22,6222; 18,3777; 12,6583; 7,1953. Isso demonstra atualização sintética; não é treinamento de um reconhecedor. A CTC isolada, sem otimizador, repetiu loss 11,1820 e backward válido nas 5 iterações.

No DirectML, Conv, BN, CNN e Linear chegaram à verificação de gradientes na primeira iteração, mas não completaram o step. Não foi demonstrada estabilidade de treinamento com atualização nesses componentes. ReLU e MaxPool completaram as 5 iterações. A BiLSTM e a CTC não têm resultados de backward DML.

### Erros e fallbacks

Todos os erros abaixo ocorreram com `torch 2.4.1+cpu`, `torch-directml 0.2.5.dev240914`, dispositivo `privateuseone:0`, RX 6700 XT, em batches 4/8/16.

**BiLSTM — etapa forward.** No diagnóstico inicial, o backend avisou que `aten::_thnn_fused_lstm_cell` usaria CPU. A tentativa de fallback terminou em `NotImplementedError`, com a mensagem:

```text
Could not run 'aten::_thnn_fused_lstm_cell' with arguments from the 'CPU' backend.
```

A mensagem integral, lista de registros do dispatcher, stack trace, versões e dispositivo foram preservados em `directml-warning-diagnostics.json`. Nessa chamada interna C++, promover warnings Python a erros não bastou para impedir a tentativa. O script final usa adicionalmente `TorchDispatchMode` e consulta os registros do dispatcher **antes da execução**. A chamada final foi interrompida com:

```text
aten::_thnn_fused_lstm_cell: no PrivateUse1 or composite kernel; CPU fallback blocked before execution
```

**CTC Loss — etapa loss, correspondente ao forward da CTCLoss.** O aviso original identificou `aten::_ctc_loss` sem suporte e fallback para CPU. O diagnóstico final bloqueou antes de executar:

```text
aten::_ctc_loss.Tensor: no PrivateUse1 or composite kernel; CPU fallback blocked before execution
```

No JSON, `ctc.stages.forward` é a preparação dos logits pela Identity; o forward da CTCLoss é `ctc.stages.loss`, indicado também por `operation_forward_stage`. Logits válidos não significam que a CTCLoss executou.

**Adam — etapa optimizer.** O backend emitiu o aviso abaixo, promovido a exceção `UserWarning`, impedindo aprovação do step:

```text
The operator 'aten::lerp.Scalar_out' is not currently supported on the DML backend and will fall back to run on the CPU. This may have performance implications.
```

O aviso aponta para `torch_directml/csrc/dml/dml_cpu_fallback.cpp:17`. Os JSONs contêm mensagem e traceback completos. O teste escolheu Adam; não conclui que todo otimizador possível é incompatível. Trocar o otimizador, porém, não resolve a ausência de BiLSTM/CTC nativas demonstrada aqui.

**Não há treinamento DirectML aprovado com fallback escondido.** A tentativa interna de fallback da BiLSTM no primeiro diagnóstico foi exposta e falhou. Nas execuções finais, o script combina auditoria dos operadores DML com erros em avisos de fallback. Kernels compostos são permitidos, com auditoria das suboperações despachadas; cópias explícitas de observação/sincronização são identificadas. Essa auditoria depende de APIs internas do PyTorch e deve ser revalidada ao trocar de versão. Não substitui um profiler de kernels, mas os bloqueios observados já são suficientes para reprovar o caminho nativo. Nenhum experimento híbrido foi realizado; custo CPU↔GPU e backward híbrido não foram medidos.

## Comparação numérica

Mesma seed, ordem de inicialização, pesos e entradas; primeira iteração antes de atualização; `rtol=1e-3`, `atol=1e-5`. Todas as saídas, losses auxiliares e gradientes que puderam ser comparados passaram nessas tolerâncias, nos três batches. A tabela mostra batch 8; o máximo de gradientes abrange entrada e parâmetros disponíveis.

| Componente | Maior diferença absoluta na saída | Maior diferença absoluta nos gradientes | Limite da comparação |
|---|---:|---:|---|
| Conv2d | 5,96e-7 | 1,49e-7 | Step DML indisponível |
| BatchNorm2d | 4,77e-7 | 4,47e-8 | Step DML indisponível |
| ReLU | 0 | 3,64e-12 | Forward/backward comparáveis |
| MaxPool2d | 0 | 1,46e-11 | Forward/backward comparáveis |
| CNN combinada | 2,86e-6 | 1,58e-7 | Step DML indisponível |
| Linear | 5,96e-7 | 9,31e-9 | Step DML indisponível |
| BiLSTM | Não comparável | Não comparável | Forward DML falhou |
| CTC Loss | Não comparável | Não comparável | Apenas os logits de entrada são iguais |
| Rede integrada | Não comparável | Não comparável | DML não executado |

Não foi demonstrada equivalência numérica da CRNN, da BiLSTM, da CTCLoss ou das atualizações Adam. Valores relativos altos perto de zero, presentes no JSON, são avaliados juntamente com o termo absoluto da tolerância; nenhuma diferença significativa foi encontrada nas partes comparáveis.

## Desempenho

Para cada batch e modelo aprovado: 3 medições com modelos/otimizadores reinicializados no mesmo processo, 2 iterações de warmup por medição e 5 medidas por medição, totalizando 15 amostras. Os batches foram executados sequencialmente. O script usa os mesmos dados artificiais pré-carregados em todas as iterações. As medições não incluem criação de modelo, transferência inicial ou geração dos dados.

Sincronização: leitura bloqueante da loss em CPU ao fim do forward; após `optimizer.step()`, cópias bloqueantes de **todos** os parâmetros para CPU, garantindo conclusão dos ramos de atualização. Cópias e clones entram nos tempos; a checagem de finitude posterior fica fora deles. O tempo de forward inclui zero_grad e loss; backward inclui Adam e sincronização. A auditoria Python do dispatcher fica no diagnóstico prévio, fora do trecho cronometrado; avisos de fallback continuam invalidando medições. Nenhuma chamada CUDA é usada. Se um backend acelerado passasse, os números seriam conservadores por incluir essas transferências; não representariam tempo exclusivo de kernels.

### Treinamento integrado

| Backend | Batch | Média ms/iteração | Mediana ms/iteração | Imagens/s | Speedup |
|---|---:|---:|---:|---:|---:|
| CPU | 4 | 16,610 | 16,698 | 240,82 | 1,00× |
| CPU | 8 | 25,304 | 25,220 | 316,15 | 1,00× |
| CPU | 16 | 41,464 | 41,714 | 385,88 | 1,00× |
| DirectML | 4/8/16 | Não medido | Não medido | Não medido | Indisponível |

| Batch CPU | Forward + loss ms | Backward + Adam ms | Desvio padrão entre iterações ms | Médias das 3 medições ms | Desvio padrão entre médias ms |
|---|---:|---:|---:|---|---:|
| 4 | 6,069 | 10,541 | 0,676 | 16,901 / 16,875 / 16,053 | 0,482 |
| 8 | 9,796 | 15,508 | 0,549 | 25,708 / 24,938 / 25,267 | 0,386 |
| 16 | 16,799 | 24,664 | 1,052 | 41,300 / 42,401 / 40,690 | 0,867 |

### CNN com objetivo auxiliar e Adam

| Backend | Batch | Forward + loss ms | Backward + Adam ms | Média total ms | Mediana total ms | Imagens/s |
|---|---:|---:|---:|---:|---:|---:|
| CPU | 4 | 2,508 | 3,158 | 5,667 | 5,596 | 705,89 |
| CPU | 8 | 4,509 | 4,805 | 9,315 | 9,309 | 858,86 |
| CPU | 16 | 9,430 | 10,115 | 19,545 | 19,495 | 818,63 |
| DirectML | 4/8/16 | Não medido | Não medido | Não medido | Não medido | Não medido |

Médias das três medições CNN: batch 4 = 5,726 / 5,525 / 5,749 ms; batch 8 = 9,230 / 9,247 / 9,468 ms; batch 16 = 19,561 / 19,519 / 19,555 ms. Desvios entre médias: 0,123 / 0,133 / 0,023 ms, respectivamente. Tempos brutos e demais estatísticas estão nos JSONs.

Não foi medido desempenho DirectML de treinamento CNN porque o Adam falhou. Também não foi medido treinamento integrado DirectML: os componentes já haviam reprovado. Não existe speedup GPU válido, e as linhas CPU são somente uma referência preliminar para esta arquitetura pequena, sem carregamento de dados. Não extrapolar para a arquitetura futura ou para a A2000.

## Reprodução

Use o venv existente. Para recriar em **uma nova pasta de benchmark vazia**, com Python 3.12 Windows x64:

```powershell
py -3.12 -m venv .venv-directml
.\.venv-directml\Scripts\python.exe -m pip install -r benchmarks/requirements-directml.txt
.\.venv-directml\Scripts\python.exe -m pip check
```

`requirements-directml.txt` fixa os pacotes principais; `requirements-directml-lock.txt` registra todas as versões de dependências efetivamente encontradas no venv desta medição e pode ser usado em seu lugar para reproduzi-las. Não aplicar esses arquivos ao ambiente principal do projeto.

```powershell
.\.venv-directml\Scripts\python.exe benchmarks/backend_probe.py --device cpu
.\.venv-directml\Scripts\python.exe benchmarks/backend_probe.py --device directml
.\.venv-directml\Scripts\python.exe benchmarks/backend_probe.py --device both --batch-size 8 --iterations 5 --warmup 2 --repeats 3 --seed 42 --output benchmarks/directml-results.json
.\.venv-directml\Scripts\python.exe benchmarks/backend_probe.py --device both --batch-size 4 --output benchmarks/directml-results-b4.json
.\.venv-directml\Scripts\python.exe benchmarks/backend_probe.py --device both --batch-size 16 --output benchmarks/directml-results-b16.json
```

Defaults: batch 4, 5 iterações medidas, 2 warmups, 3 medições, seed 42, 6 threads. `--device-index` permite selecionar outro adaptador explicitamente. Cada comando sobrescreve seu arquivo `--output`; use outro destino para preservar resultados anteriores. Exit code 1 significa que algum backend solicitado não passou no integrado; o JSON ainda é gravado. CPU aprovada retorna 0. Falhas de instalação também ficam registradas como backend indisponível quando a importação principal de torch funciona.

## Testes e escopo das alterações

- Nove testes novos sem necessidade de GPU: alinhamento CTC com repetições, CLI, unidades/throughput, bloqueio de warnings de fallback, comparação parcial, NaN/gradiente zero, dados/pesos reproduzíveis, treinamento CPU completo e rejeição de otimizador que não atualiza pesos. Testes dependentes de torch são pulados se ele não estiver instalado.
- No venv do benchmark: `python -m unittest discover -s tests -p test_backend_probe.py -v` — 9 testes passaram.
- No Python principal: `python -m unittest discover -s tests -v` — 96 testes descobertos, 74 passaram, 22 pulados, sem falhas. Os 22 são do gerador, pois PHP CLI não está no PATH; GD/FreeType não puderam ser verificados. Nenhuma dependência global foi instalada para isso.
- `git diff --check` executado sem erros de whitespace.
- `.gitignore` passou a ignorar `.venv-directml/`. Nenhum ambiente, peso ou checkpoint foi adicionado ao Git. Sem commit/push.
- A alteração prévia em `dataset/real/labels.csv` foi preservada: SHA-256 antes/depois `0B09CF061DD897D788E8E0583AE39482C95E4AE638443D92AD23C1C3B6BB28C4`.
- Nenhuma alteração em generator/, labeling/, inspection/, captcha_dataset.py, dataset/real ou dataset/synthetic. Sem requisições aos Correios, OCR, novos datasets, treinamento do CaptchaDecay ou conexão com VM NVIDIA.

Arquivos entregues: este relatório; `backend_probe.py`; `requirements-directml.txt`; `requirements-directml-lock.txt`; `directml-results.json` (batch 8); `directml-results-b4.json`; `directml-results-b16.json`; `directml-warning-diagnostics.json`; `../tests/test_backend_probe.py`; alteração em `../.gitignore`.

## Decisão de backend

**C — DirectML incompatível neste ambiente.** Há evidência funcional positiva para CNN e Linear, porém faltam operações críticas da CRNN e uma atualização Adam nativa aprovada. Não é B: treinamento completo não funcionou. Não é D: a instalação e a GPU foram verificadas e os bloqueios foram reproduzidos. Não há evidência que permita escolher A ou anunciar aceleração.

Adotar CPU no desenvolvimento local e planejar a validação/treinamento completo na NVIDIA A2000 com CUDA. Reavaliar DirectML somente se uma nova combinação de pacotes oferecer os operadores necessários e passar novamente por forward, backward, atualização, comparação numérica e medições completas.
