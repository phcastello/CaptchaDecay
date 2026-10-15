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
filename,label,split,length,width,height,perturbation,num_lines,noise_level,font_ratio,image_bg_color,text_color,line_color,seed,index,render_seed,config_sha256,securimage_version
```

Filename é um contador de oito dígitos, independente do label. A sequência
continua a partir do maior índice registrado. Imagem e dimensões são validadas
antes da escrita exclusiva; o CSV só recebe a linha após a escrita da imagem.
Linhas antigas são preservadas e cada linha nova é imediatamente descarregada
para o arquivo. Um lock no diretório impede gerações simultâneas na mesma saída.

Em erro, o processo sai com código 1 e preserva o progresso. Uma interrupção
abrupta entre imagem e CSV pode deixar um PNG órfão, ou uma última linha CSV
incompleta. A próxima execução detecta esses casos e para sem sobrescrever
dados. Revise e recupere essa imagem/linha manualmente antes de continuar;
o gerador não apaga arquivos nem inventa metadata de recuperação.

`vendor/`, PNGs sintéticos, seu CSV e lock são ignorados pelo Git. `composer.lock`
é mantido para fixar a versão oficial e preservar a resolução da dependência.

## Seed e limites de reprodução

Labels e parâmetros usam um `Randomizer`/`Mt19937` separado por seed e índice,
sem compartilhar o estado aleatório consumido pelo Securimage. Splits são
embaralhados com outro engine e arredondados por lote. Com a mesma configuração,
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
