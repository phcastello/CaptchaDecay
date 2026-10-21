# Experimento E — transferência ampliada dos modelos A, B e D

## 12.1. Objetivo

Avaliar a transferência de CRNNs treinadas com dados sintéticos para imagens reais, com ênfase nas mudanças pareadas B→D.

## 12.2. Metodologia

Arquitetura existente: CNN 1→32→64→128→256, média vertical, BiLSTM de duas camadas com hidden size 128 por direção, projeção para 31 classes CTC; 1.186.815 parâmetros em cada modelo.

CPU para A/B/D; model.eval(), requires_grad_(False), torch.inference_mode(). O loader usa map_location="cpu" e weights_only=True, inclusive para D treinado em CUDA. Pesos e buffers de BatchNorm são comparados em memória antes/depois; nenhum otimizador é instanciado.

Batch=16; threads=6; workers=0. Sistema: Linux-6.8.0-139-generic-x86_64-with-glibc2.39; Python 3.12.3, PyTorch 2.14.1+cu126, NumPy 2.5.4, Pillow 12.3.0. Ambiente completo em summary.json.

Pré-processamento e inferência reutilizados de ml.dataset, ml.evaluate e ml.model: PNG original 215×80, grayscale float32, normalização [-1,1], sem resize, recorte, threshold ou augmentation. Decoder greedy CTC de ml.decode; nenhum ajuste com base em labels.

Exact match exige igualdade literal. CER global = soma das distâncias de Levenshtein / soma dos comprimentos humanos, sem média simples dos CERs individuais. O alinhamento determinístico de ml.metrics/ml.analysis_metrics desempata por match, substituição, deleção e inserção; S+D+I é conferido contra a distância mínima.

**Separação metodológica:** real/train é diagnóstico; suas características calibraram o gerador sintético v2, embora seus PNGs não tenham atualizado os pesos. real/validation é desenvolvimento já consultado. A união é exploratória, não um teste independente. real/test permaneceu completamente reservado, sem listagem de conteúdo, leitura de imagens ou inferência.

Wilson 95% é apresentado para exact match (inclusive subgrupos pequenos, sujeitos a grande incerteza). McNemar exato bicaudal usa somente os pares discordantes, com probabilidade binomial 0,5. Estatísticas exploratórias, sem correção de múltiplas comparações; ausência de significância não demonstra equivalência.

## 12.3. Integridade

| Modelo | Treino sintético | Época interna | Checkpoint | SHA-256 |
|---|---:|---:|---|---|
| A | 2700 | 95 | outputs/runs/crnn-v2-baseline/best.pt | fe315effbcf0dc6dafaaf5856a0b175f7bde62fef1ba7aabb2eeae506c656e1b |
| B | 9000 | 28 | outputs/runs/crnn-v2-9k-fixed/best.pt | 90f7dd05955e38a6b2f0a54cca0883db8a70b38cad2a2aac512b22234fd2e8c9 |
| D | 20000 | 46 | outputs/runs/crnn-v2-20k-cuda-plateau/best.pt | da2c608b9776f7ea54e15996e8bcb1c8bcfa54f2c4fed0f8c536ff7f093bfddf |

Charset idêntico e literal: `23456789abcdefghiklmnprstuvwyz`; blank=0.

CSV real original SHA-256: `322a305cd3d6238a7a42afc3f6284a1a719fbb01a66af2b71f0517568fb1a43c`. Manifesto canônico: `fbeae3902255fed8c55f4025da5a825596be3f404aab52f2672552a3698aff81`. CSV de elegibilidade congelado: `c29ed87ef6ccaf7795cff79e9188d340d7cafc4bca571e428e0606e9504ac72e`.

| Split | Imagens | Labels preenchidos | Elegíveis | Excluídas | Pendentes |
|---|---:|---:|---:|---:|---:|
| train | 63 | 63 | 46 | 17 | 0 |
| validation | 21 | 21 | 17 | 4 | 0 |

As imagens permitidas foram conferidas contra SHA-256 do CSV, formato PNG, dimensões e decodificação completa. Correspondência CSV/PNG e nomes duplicados foram conferidos. Duplicatas de bytes entre splits permitidos: {}.

O CSV compartilhado foi filtrado para train/validation antes da construção de caminhos. Nenhum loader percorreu automaticamente todos os splits. Não se inspecionaram PNGs de test para duplicatas.

Estado de revisão presente: False. Label salvo sem marca é o critério operacional; o rotulador não fornece confirmação positiva de dupla revisão.

A elegibilidade foi gravada e congelada antes das previsões reais. A regra de l é literal e baseada somente no label original. Não houve conversão de caixa, equivalências, alteração de labels ou seleção por previsão. Exclusões individuais:

| Split | Arquivo | Label original | Motivo |
|---|---|---|---|
| train | 038b97956e78ec4343e0e1f033f6e4a09e65d0f0177907db59db6f27751b225c.png | `z3lzv5` | human_rule_contains_lowercase_l |
| train | 0c9cb566a8a6cf6085ed7507235182c00aa3ca3048a66e3ff65ef8b6f1a58afe.png | `65ldg` | human_rule_contains_lowercase_l |
| train | 1fc24ba4d51c0944cf1dc458b8949f56cfc0d671c4082aeb2b7bc1e6f2c28979.png | `ull52` | human_rule_contains_lowercase_l |
| train | 2d1b5c0aa60916826ff47cbb818fbba56154cd492d799d40e724b322453fe7b0.png | `3kwl` | human_rule_contains_lowercase_l |
| train | 3fb223adb7dd5314d2f1565b73ca7e232266d31c6fab6e583492db5b47cccc34.png | `8czgl6` | human_rule_contains_lowercase_l |
| train | 5c15329e54a612fc368620d61f7ee68c2c2d298007a65298ffa6d4e62c3abf12.png | `slav2` | human_rule_contains_lowercase_l |
| train | 656820ed8ee578335eba17324baff00776a4ea1d39cf2ea3feaeb6f0c33c99fd.png | `83sl3` | human_rule_contains_lowercase_l |
| train | 6baf7d4c43d6864a5a4c3309d6080977d05e6c53f82681fbde9514b3b8148dd3.png | `l7abke` | human_rule_contains_lowercase_l |
| train | 703f1908f2559fe337be21f2ae3b1803108359164adacc2b772c8972f836b598.png | `ddl3c` | human_rule_contains_lowercase_l |
| train | 824b495f5d0805a4b35e0b229ad26c833c1aa4a9b09c22b38aa8349515719696.png | `rt9l3` | human_rule_contains_lowercase_l |
| train | a1e370f46df44c9212b414f0bff6169ad0a4b4d8720d32e22d12d2a4fd19b088.png | `gcatl` | human_rule_contains_lowercase_l |
| train | ab8de8944013a300f78f6dd3e35716910677dc450f256c636c606446745e8b9b.png | `rgl9r` | human_rule_contains_lowercase_l |
| train | c1ca18e528d2d6891332bdbafb05f03f4f376c681e558fdc003f06ab1991fdfb.png | `p87ml` | human_rule_contains_lowercase_l |
| train | d70c4ac40e6afbf3e8a183c387d814d0b0a4bddc3718a06f483254ea56d20161.png | `lls7l` | human_rule_contains_lowercase_l |
| train | dab14e783ff8903078d65d2de825e043e5cebb1a6a4e173aef85a2092c37cc76.png | `g8n6l` | human_rule_contains_lowercase_l |
| train | eecc594b8381ea96c8c8f73939bfd6cbd7df6b5d41a38d0a98694a154ecbb678.png | `lmbc` | human_rule_contains_lowercase_l |
| train | f25eba4f1ecc9f210a7652ce51bdb5e82f847c11e2b85ddc8ecf3ea7d502f874.png | `6giL` | human_excluded_uppercase_L_outside_charset |
| validation | 028a7568d63f4e030bb5f69951c62d9e1068256e72b8861e4e5faf995347902a.png | `9yluu7` | human_rule_contains_lowercase_l |
| validation | ca7c819782e44088dc10cbf071ce62bcc7e0224c1abdc893ef2baa6da40b9073.png | `tuel` | human_rule_contains_lowercase_l |
| validation | cb58a438df8dc3f7638b674850484b453c0e4352629b016cbbb81a4aa23997bb.png | `l345` | human_rule_contains_lowercase_l |
| validation | dda52e15a46c4f3dc6b1d1edbb612f650b350503f20c238938be1d85f637f04e.png | `ltnn75` | human_rule_contains_lowercase_l |

**Exclusão adicional explicitamente autorizada pelo usuário:** a imagem train/f25eba4f1ecc9f210a7652ce51bdb5e82f847c11e2b85ddc8ecf3ea7d502f874.png, label original `6giL`, foi retirada por conter L maiúsculo fora do charset. Essa decisão foi tomada antes da inferência real e é separada das 20 exclusões por l minúsculo. O label foi preservado.

### Regressão sintética em CPU

| Modelo | Corretas | CER | Histórico reproduzido |
|---|---:|---|---|
| A | 185/300 | 160/1507 | True |
| B | 253/300 | 60/1507 | True |
| D | 268/300 | 40/1507 | True |

Os 300 PNGs/labels atuais coincidem com o manifesto histórico preservado no dataset D. A/B usam essa mesma validação, embora os CSVs completos de seus datasets de treino tenham identidades distintas. O fingerprint da avaliação cobre o CSV completo e somente PNGs de validation, diferindo por escopo do fingerprint de treino. Nenhum dado foi regenerado.

### Reprodução real anterior A/B

Reproduziu: **True**. Medido: `{"num_images": 17, "reference_characters": 83, "a_correct": 11, "a_edits": 9, "b_correct": 14, "b_edits": 3, "both_correct": 11, "both_incorrect": 3, "first_only_correct": 0, "second_only_correct": 3}`. Divergências: `{}`.

## 12.4. Resultados por split

### real/train — diagnóstico

| Modelo | n | Corretas | Incorretas | Exact match | Wilson 95% | CER | Edições/caracteres | S/D/I |
|---|---:|---:|---:|---:|---|---:|---|---|
| A | 46 | 28 | 18 | 60.8696% | 46.4568%–73.6068% | 9.1304% | 21/230 | 18/2/1 |
| B | 46 | 41 | 5 | 89.1304% | 76.9622%–95.2669% | 2.1739% | 5/230 | 3/2/0 |
| D | 46 | 46 | 0 | 100.0000% | 92.2926%–100.0000% | 0.0000% | 0/230 | 0/0/0 |

### real/validation — desenvolvimento

| Modelo | n | Corretas | Incorretas | Exact match | Wilson 95% | CER | Edições/caracteres | S/D/I |
|---|---:|---:|---:|---:|---|---:|---|---|
| A | 17 | 11 | 6 | 64.7059% | 41.3004%–82.6903% | 10.8434% | 9/83 | 9/0/0 |
| B | 17 | 14 | 3 | 82.3529% | 58.9705%–93.8089% | 3.6145% | 3/83 | 2/1/0 |
| D | 17 | 16 | 1 | 94.1176% | 73.0180%–98.9540% | 1.2048% | 1/83 | 1/0/0 |

## 12.5. Resultado agregado

**União train + validation: análise exploratória de desenvolvimento; não é avaliação final independente.**

| Modelo | n | Corretas | Incorretas | Exact match | Wilson 95% | CER | Edições/caracteres | S/D/I |
|---|---:|---:|---:|---:|---|---:|---|---|
| A | 63 | 39 | 24 | 61.9048% | 49.5586%–72.8825% | 9.5847% | 30/313 | 27/2/1 |
| B | 63 | 55 | 8 | 87.3016% | 76.8930%–93.4227% | 2.5559% | 8/313 | 5/3/0 |
| D | 63 | 62 | 1 | 98.4127% | 91.5415%–99.7193% | 0.3195% | 1/313 | 1/0/0 |

## 12.6. Comparações pareadas

| Escopo | Par | Ambos acertam | Ambos erram | Só primeiro | Só segundo | Menor distância primeiro | Menor distância segundo | Empate distância | Discordantes | McNemar p |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | A_vs_B | 27 | 4 | 1 | 14 | 1 | 15 | 30 | 15 | 0.000976562 |
| train | A_vs_D | 28 | 0 | 0 | 18 | 0 | 18 | 28 | 18 | 7.62939e-06 |
| train | B_vs_D | 41 | 0 | 0 | 5 | 0 | 5 | 41 | 5 | 0.0625 |
| validation | A_vs_B | 11 | 3 | 0 | 3 | 0 | 4 | 13 | 3 | 0.25 |
| validation | A_vs_D | 11 | 1 | 0 | 5 | 0 | 5 | 12 | 5 | 0.0625 |
| validation | B_vs_D | 14 | 1 | 0 | 2 | 0 | 2 | 15 | 2 | 0.5 |
| aggregate | A_vs_B | 38 | 7 | 1 | 17 | 1 | 19 | 43 | 18 | 0.000144958 |
| aggregate | A_vs_D | 39 | 1 | 0 | 23 | 0 | 23 | 40 | 23 | 2.38419e-07 |
| aggregate | B_vs_D | 55 | 1 | 0 | 7 | 0 | 7 | 56 | 7 | 0.015625 |

### Mudanças individuais B→D

| Split | Arquivo | Referência | B | D | Distância B→D | Mudança de acerto |
|---|---|---|---|---|---|---|
| train | 46f6c13ecd658265370bea77bf24885ab6d3b3e1859cf773ce665e9077351051.png | `crvc` | `ervc` | `crvc` | 1→0 | D corrigiu B |
| train | 5730a5127c3570701d7aa91ec30d69a9e5cebfcc4ce810f80d758e270844f4b7.png | `rzhva` | `rzhvu` | `rzhva` | 1→0 | D corrigiu B |
| train | 5821982ab965e6d0e41a54f19c48f4f8dbf010fcfa236807b35f64902e28e8e4.png | `e265gk` | `e285gk` | `e265gk` | 1→0 | D corrigiu B |
| train | 638badd0a4a53fe908554965e8021f0e939a04de8abdd0a3085da4f7c68ea12b.png | `zc4t4` | `zc44` | `zc4t4` | 1→0 | D corrigiu B |
| train | f603886d9d16a4b2d3bff2ea9ea3c4c12865b8dfc6a65a94a613cf51fda26641.png | `ubzuy6` | `uzuy6` | `ubzuy6` | 1→0 | D corrigiu B |
| validation | 2e60a26c303edca7dc3f434d023353265bd99d32cb1390675dec811332d2b2f9.png | `y3pyu` | `y3pyd` | `y3pyu` | 1→0 | D corrigiu B |
| validation | 9f3ecf17279431223e10a67ef61e5a4b2bb116c6392bae1555b5f65fb0fe1c58.png | `89y8` | `89y3` | `89y8` | 1→0 | D corrigiu B |
| validation | fed397be919b3d9d0b9b24def049650b879a57018b0adecc8c2f507e3aa6e3e1.png | `a8tt` | `a8t` | `a8ut` | 1→1 | ambos erram; ver distância |

Todos os pares A/B, B/D e A/D, inclusive empates, constam em paired-comparison.csv e summary.json.

## 12.7. Análise de erros

| Escopo | Modelo | Comprimento real | n | Corretas | Exact match | CER | Edições/caracteres |
|---|---|---:|---:|---:|---:|---:|---|
| train | A | 4 | 14 | 7 | 50.0000% | 14.2857% | 8/56 |
| train | A | 5 | 18 | 13 | 72.2222% | 7.7778% | 7/90 |
| train | A | 6 | 14 | 8 | 57.1429% | 7.1429% | 6/84 |
| train | B | 4 | 14 | 13 | 92.8571% | 1.7857% | 1/56 |
| train | B | 5 | 18 | 16 | 88.8889% | 2.2222% | 2/90 |
| train | B | 6 | 14 | 12 | 85.7143% | 2.3810% | 2/84 |
| train | D | 4 | 14 | 14 | 100.0000% | 0.0000% | 0/56 |
| train | D | 5 | 18 | 18 | 100.0000% | 0.0000% | 0/90 |
| train | D | 6 | 14 | 14 | 100.0000% | 0.0000% | 0/84 |
| validation | A | 4 | 7 | 5 | 71.4286% | 10.7143% | 3/28 |
| validation | A | 5 | 5 | 3 | 60.0000% | 8.0000% | 2/25 |
| validation | A | 6 | 5 | 3 | 60.0000% | 13.3333% | 4/30 |
| validation | B | 4 | 7 | 5 | 71.4286% | 7.1429% | 2/28 |
| validation | B | 5 | 5 | 4 | 80.0000% | 4.0000% | 1/25 |
| validation | B | 6 | 5 | 5 | 100.0000% | 0.0000% | 0/30 |
| validation | D | 4 | 7 | 6 | 85.7143% | 3.5714% | 1/28 |
| validation | D | 5 | 5 | 5 | 100.0000% | 0.0000% | 0/25 |
| validation | D | 6 | 5 | 5 | 100.0000% | 0.0000% | 0/30 |
| aggregate | A | 4 | 21 | 12 | 57.1429% | 13.0952% | 11/84 |
| aggregate | A | 5 | 23 | 16 | 69.5652% | 7.8261% | 9/115 |
| aggregate | A | 6 | 19 | 11 | 57.8947% | 8.7719% | 10/114 |
| aggregate | B | 4 | 21 | 18 | 85.7143% | 3.5714% | 3/84 |
| aggregate | B | 5 | 23 | 20 | 86.9565% | 2.6087% | 3/115 |
| aggregate | B | 6 | 19 | 17 | 89.4737% | 1.7544% | 2/114 |
| aggregate | D | 4 | 21 | 20 | 95.2381% | 1.1905% | 1/84 |
| aggregate | D | 5 | 23 | 23 | 100.0000% | 0.0000% | 0/115 |
| aggregate | D | 6 | 19 | 19 | 100.0000% | 0.0000% | 0/114 |

### Frequência dos erros de comprimento

| Escopo | Modelo | Mais curtas | Mais longas | Total com comprimento incorreto | Frequência | Mesmo comprimento, texto incorreto |
|---|---|---:|---:|---:|---:|---:|
| train | A | 2 | 1 | 3 | 6.5217% | 15 |
| train | B | 2 | 0 | 2 | 4.3478% | 3 |
| train | D | 0 | 0 | 0 | 0.0000% | 0 |
| validation | A | 0 | 0 | 0 | 0.0000% | 6 |
| validation | B | 1 | 0 | 1 | 5.8824% | 2 |
| validation | D | 0 | 0 | 0 | 0.0000% | 1 |
| aggregate | A | 2 | 1 | 3 | 4.7619% | 21 |
| aggregate | B | 3 | 0 | 3 | 4.7619% | 5 |
| aggregate | D | 0 | 0 | 0 | 0.0000% | 1 |

A distribuição completa comprimento real→previsto está em summary.json. Mesmo comprimento não implica somente substituições: inserções/deleções podem se compensar.

### Confusões de caracteres

- train, A: comprimentos `{"shorter": 2, "longer": 1, "same_correct": 28, "same_incorrect": 15}`; substituições: c→e (3), u→v (2), 4→g (1), 9→5 (1), c→r (1), f→r (1), f→t (1), k→r (1), m→n (1), p→n (1), r→t (1), r→y (1), t→r (1), v→w (1), z→r (1); caracteres deletados `{"4": 1, "6": 1}`; inseridos `{"p": 1}`.
- train, B: comprimentos `{"shorter": 2, "longer": 0, "same_correct": 41, "same_incorrect": 3}`; substituições: 6→8 (1), a→u (1), c→e (1); caracteres deletados `{"b": 1, "t": 1}`; inseridos `{}`.
- train, D: comprimentos `{"shorter": 0, "longer": 0, "same_correct": 46, "same_incorrect": 0}`; substituições: nenhuma; caracteres deletados `{}`; inseridos `{}`.
- validation, A: comprimentos `{"shorter": 0, "longer": 0, "same_correct": 11, "same_incorrect": 6}`; substituições: 3→9 (1), 4→a (1), 8→3 (1), 8→6 (1), 8→9 (1), 9→5 (1), t→r (1), u→l (1), u→y (1); caracteres deletados `{}`; inseridos `{}`.
- validation, B: comprimentos `{"shorter": 1, "longer": 0, "same_correct": 14, "same_incorrect": 2}`; substituições: 8→3 (1), u→d (1); caracteres deletados `{"t": 1}`; inseridos `{}`.
- validation, D: comprimentos `{"shorter": 0, "longer": 0, "same_correct": 16, "same_incorrect": 1}`; substituições: t→u (1); caracteres deletados `{}`; inseridos `{}`.
- aggregate, A: comprimentos `{"shorter": 2, "longer": 1, "same_correct": 39, "same_incorrect": 21}`; substituições: c→e (3), 9→5 (2), t→r (2), u→v (2), 3→9 (1), 4→a (1), 4→g (1), 8→3 (1), 8→6 (1), 8→9 (1), c→r (1), f→r (1), f→t (1), k→r (1), m→n (1), p→n (1), r→t (1), r→y (1), u→l (1), u→y (1), v→w (1), z→r (1); caracteres deletados `{"4": 1, "6": 1}`; inseridos `{"p": 1}`.
- aggregate, B: comprimentos `{"shorter": 3, "longer": 0, "same_correct": 55, "same_incorrect": 5}`; substituições: 6→8 (1), 8→3 (1), a→u (1), c→e (1), u→d (1); caracteres deletados `{"b": 1, "t": 2}`; inseridos `{}`.
- aggregate, D: comprimentos `{"shorter": 0, "longer": 0, "same_correct": 62, "same_incorrect": 1}`; substituições: t→u (1); caracteres deletados `{}`; inseridos `{}`.

### Imagens persistentemente difíceis

| Split | Arquivo | Referência | A | B | D | Distâncias A/B/D |
|---|---|---|---|---|---|---|
| validation | fed397be919b3d9d0b9b24def049650b879a57018b0adecc8c2f507e3aa6e3e1.png | `a8tt` | `a8rt` | `a8t` | `a8ut` | 1/1/1 |

Frequências de confusões são contagens absolutas do alinhamento, sem controle por exposição ou condições visuais; não demonstram dificuldade intrínseca ou causas. Inserções/deleções não são convertidas em substituições por posição.

A galeria inclui todas as elegíveis em comparison-gallery.png e páginas numeradas de até 12 exemplos, com originais em resolução nativa, ground truth e previsões A/B/D. Etiquetas identificam acertos/erros compartilhados, correções A→B, correções B→D e regressões B→D. Nenhum label foi revisto a partir dessas previsões.

## 12.8. Interpretação

1. **D supera B nas imagens reais?** Sim, nesta amostra, em exact match e CER. Exact match B=55/63, D=62/63, diferença de +11.1111 pontos percentuais; CER B=8/313, D=1/313.

2. **Consistência/concentração:** D corrige 7 dos 8 erros de B e regride em 0 acertos; possui menor distância em 7 imagens, maior em 0 e empata em 56. As correções se concentram em 5 imagens train e 2 validation. A direção de exact match é favorável em ambos os splits: **True**. Os números absolutos são pequenos e não constituem replicações independentes.

3. **Mais dados sintéticos continuam trazendo benefícios?** Sim, nesta amostra, em exact match e CER. Isso descreve o experimento completo: D também modificou amostragem, orçamento de treinamento e scheduler. Não se pode atribuir o ganho exclusivamente às 20.000 imagens nem extrapolar uma curva de escala a partir de três execuções.

4. **Erros compartilhados:** total de imagens erradas pelos três modelos: **1**; as previsões estão na tabela de casos persistentemente difíceis.

5. **Domain gap:** a tabela abaixo compara os domínios descritivamente. A avaliação real exclui l e tem distribuição diferente; diferenças não isolam efeito de domínio, dificuldade ou comprimento. Não há evidência causal suficiente para explicar erros pela aparência das imagens.

| Modelo | Exact match sintético | CER sintético | Exact match real agregado | CER real agregado |
|---|---:|---:|---:|---:|
| A | 61.6667% | 10.6171% | 61.9048% | 9.5847% |
| B | 84.3333% | 3.9814% | 87.3016% | 2.5559% |
| D | 89.3333% | 2.6543% | 98.4127% | 0.3195% |

Não se observou piora real em exact match ou CER em nenhum dos três modelos; portanto, estes números não sustentam afirmar um domain gap adverso aqui. Isso não demonstra ausência de domain gap na população: as referências reais excluem l, são poucas e participaram do desenvolvimento, enquanto o sintético inclui l. Os domínios não foram comparados sob distribuição controlada.

6. **Limitações:** amostra pequena e de desenvolvimento, exclusões que restringem a população, labels sem comprovação de dupla revisão, subgrupos muito pequenos, alinhamentos mínimos eventualmente não únicos, múltiplas comparações exploratórias e alterações simultâneas em D. Wilson e McNemar tratam as imagens como observações independentes; possível correlação da coleta não foi estimada. Nenhum resultado estatístico foi usado para escolher checkpoints ou ajustar hiperparâmetros. O conjunto real/test continua reservado.

McNemar B/D agregado: 7 pares discordantes, p=0.015625; train p=0.0625, validation p=0.5. O agregado cruza o limiar nominal de 5% nesta execução, sem ajuste para múltiplas comparações; os splits separados não o cruzam. Os três escopos se sobrepõem e não são três testes independentes. Ausência de significância estatística não demonstra equivalência.

## Reprodutibilidade e testes

Protocolo versionável: experiments/real-transfer-expanded.json, incluindo hashes dos originais, épocas, regressão e decisão humana sobre 6giL. Hashes do protocolo, fontes e ambiente estão em summary.json. Repetir usando uma saída nova:

```bash
.venv-ml/bin/python -m ml.compare_expanded --protocol experiments/real-transfer-expanded.json --output outputs/analysis/real-transfer-expanded-rerun --batch-size 16 --threads 6
```

**Testes automatizados não executados nesta etapa, por solicitação explícita do usuário.** A suíte completa e os testes novos de tests/test_ml_compare_expanded.py foram deixados de lado. A pré-auditoria anterior executou 32 testes existentes com sucesso; esses não equivalem à suíte completa. O log anterior está preservado em preflight/. Verificações de integridade/regressão/metrificação foram executadas como parte da avaliação. Resultado de git diff --check consta em test-results.txt.

Nenhum treinamento, fine-tuning, geração de CAPTCHA, download, requisição HTTP, alteração de modelos/dados originais, commit ou push. As únicas alterações de código são avaliação, estatísticas, protocolo, testes e documentação.

Arquivos criados no projeto: `ml/compare_expanded.py`, `ml/expanded_metrics.py`, `experiments/real-transfer-expanded.json`, `tests/test_ml_compare_expanded.py` e `docs/experiment-e.md`. Alterações preexistentes em outros arquivos foram preservadas.

Artefatos em `outputs/analysis/real-transfer-expanded/`: REPORT.md, summary.json, eligibility.csv, eligibility.json, predictions.csv, paired-comparison.csv, comparison-gallery.png, comparison-gallery-01.png a comparison-gallery-06.png, synthetic-regression.json, synthetic-regression-a.csv, synthetic-regression-b.csv, synthetic-regression-d.csv e test-results.txt. A pasta preflight preserva o estágio preparatório; incoming preserva os checkpoints recebidos.

# RESULTADO DO EXPERIMENTO E

**Sim, nesta amostra, em exact match e CER.** B reconhece 55/63; D reconhece 62/63. D corrige 7 erros de B e perde 0 acertos. CER passa de 8/313 para 1/313. Resultado exploratório, limitado à amostra e às exclusões declaradas.
