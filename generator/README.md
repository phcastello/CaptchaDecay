# Gerador local Securimage

Requer PHP CLI 8.2–8.5, extensão GD com FreeType, Composer e, para organizar
amostras reais/testar nosso código, Python 3.10+. Execute na raiz do repositório:

```powershell
composer install
php generator/generate.php --count 1000 --seed 42
php generator/generate.php --count 20 --output dataset/synthetic --seed 7 --train-ratio 0.9 --validation-ratio 0.1
```

Defaults: count=1000, seed=42, saída=`dataset/synthetic`, train=0.9 e
validation=0.1. Ratios devem estar em [0,1] e somar 1; são frações, não
percentuais inteiros. Count deve estar entre 1 e 1.000.000 e a seed entre 0 e
2.147.483.647. Use `--help` para a CLI. A geração não realiza chamadas de rede.

## Configuração e API

Edite `config.php`, ou passe `--config arquivo.php`. O arquivo opcional deve
retornar um array PHP com overrides das chaves existentes; ele é código local
confiável, não um arquivo de dados remoto. Opções da CLI prevalecem sobre ele.

| Configuração | Valor/faixa inicial | API Securimage |
| --- | --- | --- |
| width / height | 215 / 80 | image_width / image_height |
| min_length / max_length | 4 / 6, uniforme | texto via display_value |
| charset | charset inicial do upstream, centralizado | texto via display_value |
| perturbation | 0.65–1.0 | perturbation |
| num_lines | 3–7 | num_lines |
| noise_level | 1–4 | noise_level |
| font_ratio | 0.35–0.45 | font_ratio |
| image_bg_color | branco / cinza muito claro | image_bg_color |
| text_color / line_color | paletas de cinza | text_color / line_color |

As faixas são infraestrutura inicial de domain randomization, próximas aos
defaults do gerador, não medições dos Correios. TODO: calibrar charset,
distribuição de comprimento e faixas visuais após rotulagem de dados reais de
adaptação/validação. Não é feita inferência automática desses parâmetros.
O charset atual admite caracteres ASCII alfanuméricos únicos; alfabetos Unicode
exigirão suporte explícito e fonte compatível em uma alteração futura.

A API [oficial 4.0.2](https://github.com/dapphp/securimage/blob/4.0.2/securimage.php)
oferece `display_value` para desenhar o label escolhido por nosso código. A
saída de `show()` é capturada em buffer com `no_session=true`, `no_exit=true`
e `send_headers=false`; nenhuma implementação interna é copiada ou sobrescrita.
Usa-se a fonte padrão fornecida pela dependência. Não é necessário servidor,
sessão PHP, banco de dados ou sistema de validação de CAPTCHAs.

## Saída, metadata e geração incremental

Cada PNG fica em `train/` ou `validation/`. Não existe split sintético de teste.
O arquivo `labels.csv` contém:

```csv
filename,label,split,length,width,height,perturbation,num_lines,noise_level,font_ratio,image_bg_color,text_color,line_color,seed,index,render_seed,config_sha256,securimage_version,rng_scheme
```

Filename é um contador de oito dígitos, independente do label. A sequência
continua a partir do maior índice registrado. Imagem e dimensões são validadas
antes da escrita exclusiva; o CSV só recebe a linha após a escrita da imagem.
Linhas antigas são preservadas e cada linha nova é imediatamente descarregada
para o arquivo. Um lock no diretório impede gerações simultâneas na mesma saída.

O formato anterior é reconhecido exatamente pelo mesmo cabeçalho sem a coluna
`rng_scheme`. Somente após validar o CSV e os arquivos, o gerador salva um backup
integral `labels.csv.pre-rng-v2-<id>.bak` e substitui o CSV atomicamente por uma
versão com a coluna adicional. Todos os campos/valores antigos são preservados
e recebem `legacy-unknown`: o formato sozinho não comprova qual versão gerou
essas amostras. As novas linhas recebem `xoshiro256starstar-sha256-v2`. Imagens
existentes não são regeneradas. Cabeçalhos desconhecidos, linhas de formatos
misturados ou datasets inconsistentes interrompem a operação para revisão.

Um índice SHA-256 em memória reúne os bytes dos PNGs dos dois splits. Arquivos
diferentes com conteúdo idêntico no dataset existente impedem novas adições.
Após renderizar e validar cada PNG novo, seu hash é conferido antes de qualquer
escrita da imagem/linha. Duplicatas interrompem com erro, identificando arquivos,
splits e eventuais labels conflitantes, sem tentativas automáticas nem remoções;
o progresso anterior permanece. O lote interrompido pode ter proporções
incompletas. Labels repetidos com imagens diferentes são válidos. Não se usam
label, filename ou `render_seed` como identidade do conteúdo. O hash não detecta
semelhança visual nem necessariamente pixels iguais codificados diferentemente;
não constitui garantia matemática contra qualquer colisão possível.

Em erro, o processo sai com código 1 e preserva o progresso. Uma interrupção
abrupta entre imagem e CSV pode deixar um PNG órfão, ou uma última linha CSV
incompleta. A próxima execução detecta esses casos e para sem sobrescrever
dados. Revise e recupere essa imagem/linha manualmente antes de continuar;
o gerador não apaga arquivos nem inventa metadata de recuperação.

`vendor/`, PNGs sintéticos, seu CSV e lock são ignorados pelo Git. `composer.lock`
é mantido para fixar a versão oficial e preservar a resolução da dependência.

## Seed e limites de reprodução

O esquema `xoshiro256starstar-sha256-v2` usa `Randomizer`/`Xoshiro256StarStar`,
inicializado com os **32 bytes binários completos** de
`SHA-256("captchadecay:rng-v2:<seed>:<contexto>")`, conforme a
[documentação oficial do PHP](https://www.php.net/manual/en/random-engine-xoshiro256starstar.construct.php).
O contexto `sample:<índice>` controla label e parâmetros independentemente do
split e do diretório de saída; `splits:<início do lote>` controla a divisão.
Esse esquema substitui a derivação antiga de 28 bits e **muda a sequência
produzida pela mesma seed**, inclusive o embaralhamento dos splits.

Cada amostra usa seu próprio estado, sem compartilhar a aleatoriedade global
`mt_rand()` consumida pelo Securimage. Renderizar outras imagens não muda sua
metadata. Splits são embaralhados com outro engine e arredondados por lote.
Para o esquema novo, com a mesma configuração,
seed, índice inicial e quantidade do lote, labels, parâmetros, metadata e splits
são reproduzíveis. Lotes pequenos podem arredondar validation para zero.

Dividir um lote em várias execuções mantém labels e parâmetros por índice,
mas pode mudar os splits, pois as proporções/embaralhamento são calculados por
lote. Mudar faixas ou charset também muda a sequência de metadata; o hash da
configuração identifica as configurações visuais utilizadas.

A `render_seed` também inicializa `mt_rand` antes de `show()`. Ainda assim, não
se promete igualdade byte a byte entre versões de PHP, GD/FreeType, plataformas
ou fontes. Para reprodução visual, conserve config, lockfile e versões dessas
ferramentas. A dependência legada pode emitir depreciações em PHP recente;
apenas `E_DEPRECATED` é suprimido durante a renderização, sem editar `vendor/`.
Os demais erros continuam sendo reportados em stderr.

## Organização real e testes

```powershell
python generator/organize_real.py --source dataset --output dataset/real --seed 42
python -m unittest discover -s tests -v
```

A organização real conserva os originais, cria cópias 60/20/20 sem duplicatas
por SHA-256 e escreve labels vazios. Reexecutar adiciona imagens ao índice sem
mover splits existentes ou perder labels manuais. `source` e `split_seed` no
CSV permitem auditar a migração. Os diretórios antigos são somente backup;
use apenas `dataset/real` para adaptação e avaliação.

`split_seed` registra a seed apenas quando o organizador decide o split nessa
execução. Para arquivos que já estavam em `real/<split>` sem linha anterior no
CSV (por exemplo, vindos do coletor), o split é conservado e `split_seed` fica
vazio, indicando procedência desconhecida. Linhas já indexadas, incluindo labels,
hashes, origens e seeds, são preservadas mesmo ao mudar `--seed`. Seeds históricas
registradas pela versão anterior podem ser imprecisas nesse caso; não há
reescrita em massa nem inferência da seed original sem evidência específica.

Os testes incluem receitas sem renderização para os índices 3057 e 21576 com
seed 42, isolamento de RNG, poucos PNGs com renderização injetada para deduplicação,
migração de CSV e coleta simulada. A integração local gera 20 imagens, adiciona
5 e compara metadata com outra saída limpa. Requer as dependências acima
instaladas via `composer install`; não usa OCR nem acessa os Correios. Informe
`PHP_BINARY` se o executável PHP não estiver no PATH.
