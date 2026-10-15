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
seed, índice, seed de renderização, hash da configuração e versão do Securimage.

Edite [generator/config.php](generator/config.php) para mudar charset,
comprimento (inicialmente 4–6), resolução (215×80), faixas de distorção, linhas,
ruído, tamanho relativo da fonte e paletas. Esses valores iniciais **não são
estimativas dos parâmetros dos Correios**; devem ser calibrados após rotulagem
das amostras reais. Consulte [generator/README.md](generator/README.md) para
opções, reprodução e tratamento de interrupções.

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
