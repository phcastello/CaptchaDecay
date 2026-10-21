# PoC de consulta end-to-end

O executor recebe **um objeto**, abre um contexto Chromium novo, captura o
CAPTCHA daquela sessão, usa a CRNN D congelada e clica em **Consultar**.
O resultado da consulta, e não o reconhecimento isolado, determina o sucesso.
Nenhum teste automatizado consulta o site dos Correios.

## Ambiente e instalação

Na raiz `/home/phc/CaptchaDecay`, Ubuntu 24.04, Python 3.12.3:

```bash
./.venv-ml/bin/python -m pip install -r requirements-poc.txt
./.venv-ml/bin/python -m playwright install chromium
./.venv-ml/bin/python -m pip check
```

Dependências acrescentadas nesta entrega: Playwright 1.64.0, pyee 14.0.0 e
greenlet 3.5.6. Chromium/Headless Shell 156.0.8078.4, revisão Playwright 1248,
e FFmpeg 1013 foram instalados pelo instalador do Playwright no cache do
usuário. Não foi necessário instalar bibliotecas do sistema. Não houve pip
global, alteração de drivers ou reinstalação do PyTorch.

Antes e depois: PyTorch **2.14.1+cu126**, runtime CUDA **12.6**, CUDA disponível;
NumPy 2.5.4 e Pillow 12.3.0. `pip check` passou. O padrão do executor é CPU;
`--device cuda` é uma opção explícita e reutiliza `resolve_device()`.
O ambiente ML está documentado em [gpu-training.md](gpu-training.md).

Referências usadas: [browsers](https://playwright.dev/python/docs/browsers),
[network](https://playwright.dev/python/docs/network),
[Response.body](https://playwright.dev/python/docs/api/class-response#response-body),
[locators](https://playwright.dev/python/docs/locators) e
[actionability](https://playwright.dev/python/docs/actionability).

## Checkpoint e reconhecimento sem ground truth

Caminho padrão: `outputs/runs/crnn-v2-20k-cuda-plateau/best.pt`.
Época interna obrigatória: **46**. SHA-256 obrigatório:

```text
da2c608b9776f7ea54e15996e8bcb1c8bcfa54f2c4fed0f8c536ff7f093bfddf
```

Outro caminho só é aceito se contiver exatamente o mesmo checkpoint.
Arquivo ausente, hash, época, classes ou pesos incompatíveis interrompem
antes da navegação. O arquivo não é convertido, atualizado ou substituído.

`poc.recognizer.Recognizer(checkpoint, device="cpu", threads=6)` carrega uma
vez. `recognize(png_bytes)` retorna `text`, `image_sha256`, identificação/hash/
época do checkpoint, dispositivo, `model_load_seconds` e `inference_seconds`.
Reutiliza `load_checkpoint(path)` de `ml.train`, `CRNN(**model_config)`,
`Charset.from_dict`, `read_image_tensor(io.BytesIO(png_bytes))` e
`greedy_decode(logits, charset)`. O arquivo é carregado com
`map_location="cpu", weights_only=True` pelo loader existente.

Entrada: PNG original estático de **215×80**, grayscale, float32,
`pixel / 127.5 - 1`, batch `[1,1,80,215]`. Não há resize. `eval()`,
`requires_grad_(False)` e `torch.inference_mode()` congelam a execução.
Pesos, buffers de BatchNorm e arquivo são comparados ao final, inclusive
em falhas. CUDA usa sincronização antes/depois das medições.

Não há dataset/target/label no reconhecedor, perda CTC, treinamento, solver
externo, confiança calibrada ou correção do texto. Previsão vazia ou logits
não finitos impedem a submissão. O `l` armazenado no charset é usado sem
exclusão ou heurística. Nenhum conteúdo de `dataset/real/test` é acessado.
A equivalência com `frozen_inference()` anterior é testada em uma fixture
rotulada exclusiva do teste; o novo caminho recebe somente bytes PNG.
Nenhuma função ML existente precisou ser alterada.

## CLI e início explícito

Sem `--live`, a CLI mostra ajuda e não abre o site. A exceção local explícita
é `--smoke-image`, que também não acessa a rede.

Uma consulta, CPU, uma tentativa, com solicitação interativa oculta do código:

```bash
cd /home/phc/CaptchaDecay
./.venv-ml/bin/python -m poc.run \
  --live \
  --checkpoint outputs/runs/crnn-v2-20k-cuda-plateau/best.pt \
  --device cpu \
  --max-attempts 1
```

Também aceita `--tracking-code` ou a variável `CAPTCHADECAY_TRACKING_CODE`.
Não salve valores reais em configurações versionadas. A entrada exige duas
letras maiúsculas, nove dígitos e duas letras, sem listas, CPF/CNPJ,
intervalos ou geração de códigos. O texto previsto do CAPTCHA é submetido
exatamente como retornado pelo decoder, sem transformação.

Headless é o padrão e foi validado neste servidor. Para exibir a janela no
Windows, execute na raiz do projeto, pelo PowerShell:

```powershell
.\.venv-ml\Scripts\python.exe -m poc.run --live --headed --device cpu --max-attempts 1
```

No Linux, `--headed` exige `DISPLAY` ou `WAYLAND_DISPLAY`; na ausência deles,
falha antes da navegação. Windows e macOS usam o display nativo, sem exigir
essas variáveis. Não havia display no servidor Linux original: nenhuma janela
foi validada visualmente naquele ambiente.
`--timeout-ms` vale 30000 por padrão, limitado a 1–120000 ms por operação.
Não existe opção de URL remota arbitrária, proxy, stealth ou troca de sessão.

Inspeção pública explícita, sem preencher/submeter e sem exigir modelo:

```bash
./.venv-ml/bin/python -m poc.run --live --inspect --device cpu
```

A inspeção confirma apenas presença de controles e captura da imagem.
Seu resultado tem `success=false`; ela não comprova consulta nem aceitação.

## Interface pública observada

Uma navegação pública sem código/formulário foi realizada em 2026-10-21,
usando TLS normal e os recursos carregados pela própria página:
`https://rastreamento.correios.com.br/app/index.php`.
Não foram sondados endpoints alternativos nem feitas consultas nessa inspeção.

| Parte | Evidência observada |
| --- | --- |
| Campo do objeto | `input#objeto`, nome `objeto` |
| Campo do CAPTCHA | `input#captcha`, nome `captcha` |
| Imagem exibida | `img#captcha_image` |
| Resposta da imagem | `/core/securimage/securimage_show.php` |
| Botão normal | `button#b-pesquisar`, texto Consultar |
| Refresh normal | `#captcha_refresh_btn` |
| Submissão normal do frontend | fetch de `/app/resultado.php`, parâmetros `objeto`, `captcha`, `mqs=S` |
| Rejeição explícita no JS | `r.erro` e `r.mensagem === "Captcha inválido"` |
| Renderização | `codObjeto`, lista `eventos`, `descricao`, `dtHrCriado.date`; `#tabs-rastreamento` |

O adaptador **observa** o fetch disparado pelo clique; não constrói uma
consulta direta. O JS faz refresh da imagem depois de receber a resposta.
Essa renovação posterior não transforma o desafio submetido em inconsistente.
Os contratos estão centralizados em `poc/correios.py` e `poc/results.py`.

Hashes dos recursos recebidos na inspeção, sem novas requisições:

- `/app/index.js`: `705c9e92e379618e006ae3be567b78bfde5f9bfce11ce5d8aec98c425ca8aefc`.
- `/core/js/rastroUnico.js`: `61d36a932e1f72a27ef4a000fc4847cb466ef3a110b79cb002382cf8099293e3`.

A estrutura de sucesso foi inicialmente inferida do renderer público e testada
localmente; a consulta real ao final desta entrega confirmou esse contrato
para o objeto autorizado. Erro de sessão, expiração e mensagem de objeto
inexistente continuam sem observação no serviço real. Não há literal presumido de
"objeto não encontrado" na configuração de produção. O classificador aceita
uma afirmação explícita de que o objeto não foi encontrado/localizado;
essa regra semântica foi testada com mensagens **fabricadas**, sem atribuí-las
ao serviço. Mensagens ambíguas permanecem desconhecidas.

## Desafio e sessão

Os observadores de requisições/respostas e de alterações da imagem são
registrados antes da navegação. A captura associa `currentSrc` da imagem
carregada à resposta de imagem recebida na mesma página/contexto. Os bytes
vêm de `Response.body()`: não há GET adicional para baixar ou confirmar o
CAPTCHA, nem screenshot usada como entrada do modelo.

Formato, dimensões naturais, integridade PNG, versão DOM, quantidade de
requisições e SHA-256 identificam o desafio. Antes do envio, o adaptador
confere novamente a identidade em memória. Há uma guarda no clique e uma
barreira para requisições duplicadas/inconsistentes. Elas apenas cancelam
um envio indevido; não alteram a validação JS, o resultado ou os dados do
serviço. Um desafio alterado encerra a tentativa sem enviar a previsão antiga.
Cookies ficam exclusivamente no contexto, até seu encerramento.

## Resultado e tentativas

`tracking_success` exige uma resposta nova da requisição normal da tentativa
atual, objeto exato correspondente em memória e eventos não vazios com
descrição e data estruturadas. HTTP 200, eco do input, conteúdo antigo,
lista vazia ou ausência de alerta nunca bastam. O corpo pode comprovar o
retorno do serviço mesmo que o renderer tenha um problema visual.

Estados: `tracking_success`, `captcha_rejected`, `object_not_found`,
`challenge_expired_or_changed`, `challenge_not_present`, `rate_limited`,
`access_blocked`, `network_error`, `site_changed_or_unrecognized`,
`inference_error`. Erros de serviço não reconhecidos recebem estado
desconhecido e `success=false`. Uma falha de sessão não prova erro de OCR.
Objeto não encontrado também não comprova aceitação do CAPTCHA.

Padrão: **uma tentativa, uma submissão**. `--max-attempts 2` ou `3` é opt-in
anterior ao início. Somente rejeição explícita permite continuar. Há espera
de cinco segundos entre tentativas e uso do refresh normal da interface
(ou da renovação automática já feita pelo frontend), na mesma sessão.
Todas as tentativas, inclusive rejeitadas, permanecem nos resultados.

HTTP 429 do domínio de rastreamento registra `Retry-After` convertido em
segundos e encerra. HTTP 401/403 ou documento explícito de bloqueio encerram.
Nenhum bloqueio, timeout ou estado desconhecido permite nova tentativa.
Não há troca de IP/identidade, sondagem de liberação ou reinício automático.

## Evidências e privacidade

Cada execução cria `outputs/poc/<timestamp>-<id>/`, sem sobrescrever:

- `result.json`: ambiente, versão Git/hash do código de trabalho, checkpoint,
  tempos, orçamento, todas as tentativas, evidências e classificação final.
- `attempts.csv`: status/tempos por tentativa e previsão efetivamente submetida.
- `run.log`: eventos estruturados com campos selecionados.
- `captcha-01.png`, etc.: bytes originais dos desafios capturados.

A pasta tem permissão 0700 e está coberta pelo `/outputs/` já ignorado pelo
Git. O código completo só existe em memória e no envio normal à página;
relatórios guardam identificação mascarada. Não são persistidos cookies,
cabeçalhos de autorização, tokens, URLs/query strings, corpo JSON do serviço,
nomes, endereços, HAR, tracing ou vídeo. A evidência da resposta contém hash,
status HTTP, correlação, contagem de eventos e regra de classificação.

Screenshots brutas são omitidas por padrão: não há uma máscara segura para
todos os dados pessoais que o serviço possa renderizar. `result.json`
registra explicitamente essa limitação; não há `result-redacted.png` quando
não se pode assegurar a redação. O PNG do CAPTCHA não é screenshot da página.

Carregamento do modelo, navegação, captura/extração da resposta já recebida,
inferência e submissão são medidos separadamente. O total inclui o
carregamento. `queries_completed / queries_started` conta consultas
concluídas no protocolo; um resultado individual não estima uma taxa
populacional. Não há exact match do CAPTCHA ao vivo nem taxa de aceitação
inferida da ausência de erros. A inspeção e o smoke test iniciam zero consultas.

## Validação local e reprodução

Unidades e integração com Chromium local (reconhecedor **simulado**):

**27 testes novos aprovados**: 12 de unidade e 15 de navegador local.
Somados aos 79 existentes abaixo, são **106 testes distintos aprovados**.
As rodadas intermediárias repetiram parte desses testes, sem consultas reais.

```bash
./.venv-ml/bin/python -m unittest \
  tests.test_poc_recognizer tests.test_poc_results tests.test_poc_browser -v
```

`tests/poc_support.py` cria um servidor de loopback, sessões e desafios
independentes. A resposta esperada da fixture é fixa, contém `l` e não é
escolhida a partir da previsão do modelo. A infraestrutura bloqueia todo
acesso externo, inclusive um recurso externo introduzido deliberadamente.
Só a infraestrutura de testes pode injetar outro destino no executor.

Cobertura: bytes originais sem GET extra, previsão exata, objeto correto,
erro HTTP 200, rejeição, objeto inexistente, resultado antigo/outro objeto,
mudança no preenchimento/no clique, troca de sessão, 429, bloqueio, timeout
sem reenvio, controles ausentes, ausência de desafio, dimensões inválidas,
previsão vazia, logits inválidos, pesos/BatchNorm, limite e histórico de
tentativas, isolamento de rede, privacidade, ausência de `--live`, inspeção
sem submissão e fechamento/persistência após falhas.

Regressões existentes executadas, incluindo o experimento E pendente:

```bash
./.venv-ml/bin/python -m unittest \
  tests.test_ml_model tests.test_ml_dataset tests.test_ml_charset \
  tests.test_ml_ctc tests.test_ml_analyze \
  tests.test_ml_compare_transfer tests.test_ml_compare_expanded -v
```

Foram 79 testes existentes aprovados, sem treinamento ou acesso ao conjunto
real reservado. A suíte completa de treinamento/geração/CUDA não foi
executada: essas funções não foram alteradas nem são necessárias ao fluxo.

Inferência com o checkpoint **D real**, separada do navegador e do serviço:

```bash
./.venv-ml/bin/python -m poc.run \
  --smoke-image dataset/synthetic/validation-300/validation/00000986.png \
  --checkpoint outputs/runs/crnn-v2-20k-cuda-plateau/best.pt --device cpu
```

Esse smoke test produziu `u2hw`, em aproximadamente 0,045 s de inferência na
primeira medição. Checkpoint e buffers preservados; não houve leitura de
label ou cálculo de acerto. É evidência de integração de inferência local,
sem demonstrar que um CAPTCHA real foi aceito.

## Consulta real desta entrega

A atualização do usuário autoriza uma sessão, um objeto fornecido
privadamente e no máximo uma submissão. A execução ocorre somente após os
testes necessários, sem refresh deliberado para seleção de desafio,
correção humana, segunda tentativa ou repetição em busca de sucesso.

Execução concluída em **2026-10-21, 02:46:41–02:46:44 UTC**:

| Medida | Resultado individual |
| --- | --- |
| Objeto | `LZ*********CN` (correspondência completa somente em memória) |
| Estado | `tracking_success` |
| Consultas iniciadas / concluídas | 1 / 1 |
| Tentativas / submissões | 1 / 1 |
| Previsão de D efetivamente enviada | `y5hv`, sem alterações |
| Evidência | Resposta nova da requisição da tentativa, objeto exato e 3 eventos com descrição/data válidas |
| Checkpoint / dispositivo | D, época 46, SHA-256 acima; CPU |
| Carregamento | 0,191 s |
| Navegação | 1,506 s |
| Captura / extração | 0,082 s |
| Inferência | 0,043 s |
| Submissão / resposta | 0,573 s |
| Total, incluindo carregamento e fechamento | 3,334 s |
| Navegador | Headless; encerrado |
| Preservação | Arquivo, pesos e BatchNorm conferidos |

Artefatos locais: `outputs/poc/20261021T024641-8cbb3cda34/`.
Hash do PNG capturado:
`a7dce3c1389df03a7d706a182f53d231146d1c7d5be89608701abbefe1251f94`.
Hash do corpo da resposta, que não foi persistido:
`80157a8cc7c1e0378b3d26f1ef99900f4ddc8565c136f0097a87aec6352f06e0`.

Houve uma requisição de imagem antes do envio, sem GET adicional de captura.
O frontend solicitou outra imagem **após receber a resposta**, como faz seu
refresh normal. A contagem final de imagens é 2; não houve mudança antes
da submissão, seleção de desafio ou nova tentativa. Não se calculou exact
match do CAPTCHA. O retorno individual do rastreamento não estima eficácia
populacional. Fixtures, inferência sintética e consulta real são evidências
distintas. Screenshots foram omitidas pela política de privacidade descrita.

Não há consulta ao vivo pendente nesta entrega. Novas execuções dependem de
nova iniciativa do usuário; não existe rotina automática de repetição.
