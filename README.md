# Classificação de Condições Maculares em OCT

Repositório do projeto final da disciplina **Inteligência Computacional em Saúde - 2026/1** da Universidade Federal do Espírito Santo (UFES).

O projeto investiga a classificação automática de imagens de tomografia de coerência óptica (OCT) de retina em quatro classes: `CNV`, `DME`, `DRUSEN` e `NORMAL`. O estudo compara uma abordagem clássica, baseada em descritores HOG com regressão logística, com uma rede convolucional compacta treinada do zero.

## Grupo

- Jheam Storch Ross
- Felipe Mattos Vanetti de Albuquerque
- Gabriel Zuany Duarte Vargas

## Entregas

- Artigo: [article/conference_101719.pdf](article/conference_101719.pdf)
- Vídeo de apresentação: `https://youtu.be/a0HlH45dhNw`

## Dataset

Foi usada a versão 3 do subconjunto OCT do dataset de Kermany et al., publicado no Mendeley Data:

<https://data.mendeley.com/datasets/rscbjbr9sj/3>

A porção usada contém B-scans maculares em tons de cinza, organizados nas classes `CNV`, `DME`, `DRUSEN` e `NORMAL`. A porção de raio-X do pacote original não faz parte do escopo deste projeto.

O protocolo preserva o conjunto de teste oficial do dataset, com 1.000 imagens balanceadas, 250 por classe. A validação é criada apenas a partir do conjunto de treino, com agrupamento por paciente inferido do nome do arquivo, evitando vazamento entre treino, validação e teste.

## Métodos

O artigo reporta duas abordagens principais:

- `configs/hog_logreg.yaml`: descritores HOG com regressão logística.
- `configs/simple_cnn.yaml`: rede convolucional compacta treinada do zero.

O repositório também inclui `configs/mobilenetv3.yaml`, usado como experimento exploratório com transfer learning. A comparação principal do artigo é entre HOG + regressão logística e a CNN compacta.

O desbalanceamento é tratado no caminho principal. O baseline usa `class_weight="balanced"` na regressão logística. Os modelos neurais usam `WeightedRandomSampler` por padrão, configurado em `imbalance.strategy`.

## Resultados Principais

Resultados no conjunto de teste oficial:

| Modelo | Acurácia | F1 macro | AUC macro (OvR) |
| --- | ---: | ---: | ---: |
| HOG + regressão logística | 0,636 | 0,63 | 0,864 |
| CNN compacta | 0,938 | 0,94 | 0,994 |

A CNN compacta superou o baseline clássico nas quatro classes. A diferença foi estatisticamente significativa pelo teste de McNemar (`p < 0,001`). O modelo convolucional tem cerca de 94 mil parâmetros e foi treinado sem transferência de aprendizado.

O principal modo de erro observado foi a confusão entre `DRUSEN` e `CNV`, classes relacionadas ao espectro da degeneração macular relacionada à idade.

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dados

Baixe o dataset manualmente e aponte `--archive` para o arquivo compactado. O conteúdo será extraído em `--raw-data-dir`.

```bash
python scripts/download_data.py --archive /caminho/para/OCT2017.zip --raw-data-dir data/raw
```

Também é possível extrair manualmente o arquivo em `data/raw` e seguir para o preparo.

## Preparo

```bash
python scripts/prepare_data.py --raw-data-dir data/raw --manifest data/processed/manifest.csv
```

O manifesto final fica em `data/processed/manifest.csv` com as colunas `image_path`, `label`, `patient_id` e `split`. O resumo de distribuição por classe e split fica em `data/processed/summary.json`.

## Treinamento

Para reproduzir os modelos principais do artigo:

```bash
python scripts/train.py --config configs/hog_logreg.yaml
python scripts/train.py --config configs/simple_cnn.yaml
```

Cada execução salva os artefatos em `outputs/<run_id>/`: `config.yaml`, `metrics.json`, `classification_report.txt`, `confusion_matrix.png`, `roc_curve.png`, e também `history.csv`/`best_model.pt` para modelos neurais ou `model.joblib` para o baseline HOG.

## Avaliação

Para reavaliar uma execução salva:

```bash
python scripts/evaluate.py --run outputs/<run_id> --split test
```

As métricas principais são `accuracy`, `balanced_accuracy`, `macro_precision`, `macro_recall`, `macro_f1` e `roc_auc_macro_ovr`, além de matriz de confusão, curvas ROC por classe e relatório de classificação.

## Comparação Estatística

Para comparar dois modelos treinados no mesmo conjunto de teste:

```bash
python scripts/compare_models.py --run-a outputs/<run_hog> --run-b outputs/<run_cnn> --split test
```

O script aplica o teste de McNemar exato sobre predições pareadas por amostra e calcula intervalos de confiança de 95% por bootstrap para acurácia e F1 macro de cada modelo, além do intervalo de confiança da diferença entre eles. Os dois runs precisam usar o mesmo manifesto e split para que o pareamento seja válido.

## Estrutura do Repositório

```text
article/      Artigo em LaTeX e PDF final
configs/      Configurações dos experimentos
data/         Dados brutos e processados, não versionados integralmente
outputs/      Artefatos gerados por treino e avaliação
scripts/      Preparação, treino, avaliação e comparação estatística
src/          Código-fonte reutilizável do projeto
```

## Limitações

Os resultados não têm finalidade diagnóstica e não substituem avaliação clínica. A interpretação depende da qualidade do dataset, do protocolo de separação por paciente e da distribuição da base de Kermany et al. O projeto não modela estágios de doença, não usa dados em vídeo e não aborda a porção de raio-X do pacote original.
