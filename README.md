# CaptchaDecay

PoC acadêmica para estudar a resistência de CAPTCHAs visuais legados diante
de visão computacional moderna. Esta etapa prepara os dados; não implementa
reconhecimento, treinamento, inferência ou OCR.

## Metodologia

O dataset é híbrido. O conjunto **sintético** é gerado localmente com a biblioteca
oficial [Securimage](https://github.com/dapphp/securimage), usada somente como
gerador de imagens. Isso permite produzir grandes quantidades de dados sem
requisições adicionais ao serviço analisado. O conjunto **real** será usado
para adaptação e avaliação após rotulagem manual.

O reconhecedor futuro será desenvolvido independentemente. Nenhuma arquitetura,
código, peso, dataset, preprocessing ou pipeline de projetos existentes de
CAPTCHA solving, incluindo `securimage_solver`, é utilizado.

| Conjunto | Split | Proporção | Finalidade futura |
| --- | --- | --- | --- |
| synthetic | train | 90% | Treinamento inicial |
| synthetic | validation | 10% | Validação sintética |
| real | train | 60% | Fine-tuning |
| real | validation | 20% | Validação / seleção de checkpoint |
| real | test | 20% | Avaliação final intocável |

Não existe `synthetic/test`. As proporções são arredondadas por lote; lotes
pequenos podem ter splits vazios. Arquivos antigos nunca são redistribuídos.

```text
generator/
  generate.php         # geração local via CLI
  config.php           # defaults e faixas visuais
  organize_real.py     # organização segura e índice para rotulagem manual
  README.md
dataset/
  synthetic/{train,validation}/
  synthetic/labels.csv # criado pelo gerador; ignorado pelo Git
  real/{train,validation,test}/
  real/labels.csv      # label vazio significa ground truth ainda desconhecido
tests/
captcha_dataset.py    # aquisição auxiliar limitada de amostras reais
composer.json
composer.lock
```

## Gerar dados sintéticos

Requisitos: PHP CLI 8.2–8.5, GD com FreeType e Composer. Não é necessário
servidor web. A instalação requer internet; a geração e os testes são locais.

```powershell
composer install
php generator/generate.php --count 1000 --seed 42
```

Cada execução adiciona a quantidade solicitada em `dataset/synthetic`, sem
sobrescrever PNGs ou linhas anteriores do CSV. Nomes como `00000001.png` são
independentes do label. O CSV contém label, split, dimensões, parâmetros visuais,
seed, índice, seed de renderização, hash da configuração, versão do Securimage
e `rng_scheme=xoshiro256starstar-sha256-v2`. O esquema novo usa SHA-256 binário
completo para inicializar Xoshiro256StarStar; a mesma seed produz uma sequência
diferente da implementação anterior. CSVs sintéticos antigos são validados e
migrados com backup integral e substituição atômica, preservando os valores
anteriores com `rng_scheme=legacy-unknown`, sem regenerar imagens.

O gerador verifica SHA-256 dos bytes dos PNGs em train e validation conjuntamente,
tanto ao abrir o dataset quanto antes de salvar cada imagem nova. Uma duplicata
interrompe a execução com diagnóstico e preserva o progresso anterior. Labels
repetidos são permitidos. Essa proteção trata de arquivos exatamente iguais;
não detecta imagens quase iguais nem pixels iguais com codificações diferentes.

Edite [generator/config.php](generator/config.php) para mudar charset,
comprimentos, resolução (215×80), faixas de distorção, linhas, ruído, fonte e
paletas. A calibração inicial usa somente os 63 labels de `real/train` e inspeção
manual: charset `abcdefghiklmnprstuvwyz23456789` e pesos de comprimento
4/5/6 = 27/46/27. O único `L` observado foi tratado como ambíguo, sem editar o
label real; charset e faixas são revisáveis com novas evidências de treino.
As novas faixas aumentam perturbação, ocupação e interferência, mantendo a
randomização. **Não são parâmetros medidos dos Correios**; `real/validation` e
`real/test` não participam da calibração. Consulte [generator/README.md](generator/README.md)
para pesos configuráveis, reprodução e o piloto separado `dataset/synthetic/pilot_v2`.

## Organizar e rotular dados reais

As 105 imagens encontradas foram organizadas com seed 42: 63 em train, 21 em
validation e 21 em test. A migração copia os bytes e conserva as aquisições
originais em `dataset/train`, `dataset/validation` e `dataset/test` como backup.
Essas pastas antigas não fazem parte do dataset canônico em `dataset/real`.

```powershell
python generator/organize_real.py --source dataset --output dataset/real --seed 42
```

O script verifica SHA-256, evita duplicatas entre os splits e cria
`real/labels.csv` com `filename,label,split,sha256,source,split_seed`. Preencha
somente a coluna `label` manualmente; label vazio é explícito e nenhum texto
é inferido pelo nome ou por OCR. Não use imagens de `real/test` para calibrar
configurações ou selecionar checkpoints.

Novas execuções preservam labels preenchidos e os splits existentes. A seed
controla apenas a divisão das imagens ainda não organizadas; alterar a seed
não move imagens já organizadas. O CSV é substituído atomicamente após escrita
completa, mantendo os rótulos existentes.

Arquivos já presentes em `real/<split>` e ainda sem registro recebem
`split_seed` vazio: a seed original é desconhecida. A seed atual só é registrada
quando o organizador escolhe o split agora. Registros existentes são preservados;
seeds históricas potencialmente incorretas não são reescritas sem evidência.

## Preparação e rotulagem do dataset

Requer Python 3.10+, Tkinter e Pillow. Tkinter costuma acompanhar o Python para
Windows; no Linux, instale também o pacote `python3-tk` da distribuição.

```powershell
python -m pip install -r requirements.txt
python labeling/labeler.py
```

O rotulador abre `real/train` e `real/validation`, com original e ampliação até 3×,
foco automático e interpolação suave/nearest-neighbor. Não altera PNGs nem sugere
labels. Aceita comprimentos/caracteres livres, preserva maiúsculas e espaços
internos e remove apenas espaços nas extremidades ao confirmar.

| Ação | Atalho |
| --- | --- |
| Salvar e avançar | Enter |
| Salvar sem avançar | Ctrl+S |
| Anterior / próxima | Ctrl+← / Ctrl+→ |
| Pular sem salvar o texto digitado | Esc |

Cada confirmação atualiza **somente o label selecionado** em
`dataset/real/labels.csv`, usando temporário validado no mesmo diretório e
substituição atômica, preservando a metadata. Um lock coordena gravações e as
alterações externas são conferidas antes da substituição. Em conflito, use
**Recarregar CSV** e reaplique sua edição. Não edite externamente durante uma
gravação. Um `.labeling.lock` deixado por interrupção só deve ser removido depois
de confirmar que nenhuma instância está gravando.

Ao reabrir, o filtro **Pendentes** começa na primeira imagem ainda não analisada.
Use **Todas** para editar labels antigos e **Revisão** para imagens duvidosas.
**Marcar revisão** persiste split/nome/SHA-256 em `dataset/real/labeling_state.json`,
sem salvar texto digitado nem inventar label. Um label anterior permanece até ser
editado ou salvo vazio. Salvar confirma a revisão e remove a marca; label vazio
significa não rotulada. Navegar, pular e recarregar descartam texto não confirmado.
Ctrl+S mantém a seleção e, se necessário, muda para **Todas**.

O teste fica isolado da inspeção/calibração e só pode ser rotulado explicitamente,
independentemente de previsões de qualquer modelo:

```powershell
python labeling/labeler.py --split test --allow-test-labeling
```

Para escolher outros dados locais, use `--dataset caminho/real`; `--split train`
restringe a sessão ao treino. A comparação aceita **somente reais de train**:

```powershell
php generator/generate.php --count 100 --seed 42 --output dataset/synthetic/pilot
python inspection/compare.py --count 20 --seed 42 --synthetic-dir dataset/synthetic/pilot/train --output inspection/output/pilot-comparison.png
```

O piloto fica separado do dataset definitivo. O comparador aceita `--real-dir`,
`--synthetic-dir`, `--count`, `--seed` e `--output`; defaults: `dataset/real/train`,
`dataset/synthetic/train` e `inspection/output/comparison.png`. A seleção é
reproduzível para entradas inalteradas, com escala comum e proporções preservadas.
O número de pares é limitado pelo menor dataset. Saídas existentes são recusadas.
O JSON ao lado do PNG registra resoluções e estatísticas de **todas as imagens
das entradas**. Nos reais, `labeled` conta apenas labels confirmados; os preenchidos
em revisão aparecem em `labeled_in_review` e não entram em comprimento/charset.
`unlabeled_or_unknown` conta imagens sem label, incluindo a parcela marcada em
`unlabeled_in_review`. Assim, os três primeiros grupos somam o total disponível;
`unlabeled_in_review` é um subconjunto, não um grupo adicional. Sem arquivo de
estado, nenhuma imagem está em revisão; estado inválido interrompe a comparação.
As estatísticas sintéticas permanecem baseadas nos labels preenchidos. Sem
label/CSV, comprimento e charset ficam desconhecidos. Contraste, ocupação e
distorção exigem inspeção manual; não há OCR ou calibração automática.

A CLI aceita somente `dataset/real/train` como entrada real e diretórios dentro
de `dataset/synthetic` como entrada sintética, incluindo `train` e `pilot/train`.
Caminhos canônicos são validados antes da leitura: reais e backups legados
`dataset/{train,validation,test}` são recusados como sintéticos, inclusive por
`..`, symlinks ou aliases que apontem para fora da raiz permitida.

PNG e JSON são preparados e validados em temporários no diretório de saída,
depois publicados com criação exclusiva. Falhas tratadas removem temporários e
saídas ainda identificadas como pertencentes à execução, preservando arquivos
anteriores ou de outra execução. O par não é uma transação indivisível: uma
interrupção abrupta pode exigir recuperação manual antes de reutilizar a saída.
Piloto, relatórios e estado auxiliar são locais e ignorados pelo Git.

## Aquisição real auxiliar

`captcha_dataset.py` permanece disponível para aquisição limitada de amostras
reais complementares. O destino padrão agora é `dataset/real`, dividido em
60/20/20 por lote. Cada comando adiciona imagens novas; SHA-256 evita duplicatas.
Após a coleta, execute `organize_real.py` para indexar os novos arquivos no CSV.

```powershell
python captcha_dataset.py --count 20 --delay 5
python generator/organize_real.py
```

O coletor mantém uma sessão com cookies, carrega a página uma vez em execução
normal e usa o refresh do endpoint de imagem. Identifica-se como
`CaptchaDecay-Academic-PoC/1.0`. HTTP 429 com `Retry-After` válido (segundos ou
data HTTP) permite uma nova tentativa após o prazo + 2 segundos. Novo 429 ou
ausência de prazo válido interrompe imediatamente, preservando o progresso.
Erros temporários 408/500/502/503/504 e de conexão mantêm backoff configurável
(`--retries`, `--backoff`); o delay normal continua configurável (`--delay`,
padrão 5 s). A ferramenta não tenta contornar rate limiting ou mecanismos anti-bot.
Uma coleta interrompida pode ser retomada posteriormente com o mesmo comando.

## Testes e atribuição

Após instalar as dependências, execute:

```powershell
python -m unittest discover -s tests -v
```

Os testes cobrem nosso código e usam poucas imagens locais; respostas do coletor
e sleeps são mockados. Não acessam os Correios. PHP deve estar no PATH ou ser
informado em `PHP_BINARY`; sem PHP, os testes do gerador são explicitamente pulados.

Synthetic CAPTCHA generation uses the open-source Securimage library.
A versão oficial 4.0.2 é instalada via Composer e fixada no lockfile. O projeto
upstream está sem manutenção; aqui é utilizado apenas localmente como gerador.
Copyright e licenças upstream, incluindo os da fonte, permanecem em
`vendor/dapphp/securimage`; consulte seu `LICENSE.txt` e `README.FONT.txt`.
Nenhum arquivo upstream é copiado manualmente ou alterado neste projeto.
