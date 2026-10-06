# VulnCaptcha

Módulo Python para coletar o `src` de `img#captcha_image` na página de
[rastreamento dos Correios](https://rastreamento.correios.com.br/app/index.php#).
Usa somente a biblioteca padrão do Python 3.10 ou superior.

## Terminal

```powershell
python -m captcha_dataset --count 100 --output dataset --delay 5
```

Esse comando adiciona 100 imagens novas: 85 em `dataset/train`, 10 em
`dataset/validation` e 5 em `dataset/test`. A quantidade padrão é 100.
As pastas são criadas automaticamente e as imagens são salvas como PNG.
O caminho de saída é relativo ao diretório de execução.

## Uso na PoC

```python
from captcha_dataset import collect_captchas

resultado = collect_captchas(count=200, output_dir="dataset", delay=5.0)
print(resultado)  # {'train': 170, 'validation': 20, 'test': 10}
```

Importar o módulo não inicia downloads. A sessão preserva cookies e o endereço
da imagem é extraído do elemento da página, com URLs relativas resolvidas.
Os downloads são sequenciais, com um intervalo configurável entre tentativas.
O intervalo padrão é de cinco segundos; o servidor ainda pode limitar a coleta.
Falhas temporárias de conexão e HTTP 408, 429, 500, 502, 503 e 504 são repetidas
até três vezes por requisição (`--retries 3`), inclusive ao abrir a página inicial.
As esperas padrão são de 60, 120 e 240 segundos (`--backoff 60`). Em HTTP 429,
a espera mínima é sempre de 60 segundos. Se o servidor enviar `Retry-After`,
em segundos ou data HTTP, o módulo aguarda pelo menos esse prazo.
Outros erros HTTP interrompem a coleta imediatamente.

### Limites e timeout

`HTTP Error 429: Too Many Requests` indica um limite imposto pelo servidor.
O módulo aguarda e tenta novamente, mas não há garantia de que o acesso será
liberado nessas tentativas. Se o erro persistir, encerra a coleta preservando os
arquivos. Aguarde mais tempo antes de reiniciar. Aumentar `--timeout` não
remove esse limite.

`timed out` indica que uma operação de rede excedeu o tempo de espera. Nesse
caso, é possível aumentar `--timeout`, por exemplo para 60 segundos.

```powershell
python -m captcha_dataset --count 20 --delay 10 --retries 3 --backoff 60
```

`--delay` controla a pausa normal entre downloads; `--backoff` controla a espera
progressiva após uma falha temporária. `--retries 0` desativa novas tentativas.

A divisão 85/10/5 se aplica a cada lote concluído. Para quantidades que não são
múltiplos de 20, as contagens são arredondadas pelos maiores restos, mantendo o
total solicitado. A divisão é embaralhada com a semente configurável `--seed`
(padrão: 42). Lotes anteriores são preservados, sem redistribuição.

Os nomes usam SHA-256 do conteúdo. Imagens idênticas, inclusive de execuções
anteriores, são ignoradas para evitar duplicatas entre os subconjuntos. Após
três vezes a quantidade solicitada em tentativas, a coleta incompleta termina
com erro. Em falhas ou interrupções, os arquivos já salvos permanecem e o lote
parcial pode não manter a proporção. As imagens não recebem rótulos com o texto
do CAPTCHA.
