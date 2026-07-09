# OCT Macular Classification

Projeto de classificação multiclasse de imagens OCT de retina nas classes `CNV`, `DME`, `DRUSEN` e `NORMAL`, usando o dataset de Kermany et al. publicado no Mendeley Data.

## Dataset

Dataset base: Kermany et al., *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*.

Fonte: <https://data.mendeley.com/datasets/rscbjbr9sj/3>

Esta base foi escolhida porque usa imagem médica real, possui rótulos categóricos de diagnóstico e evita a ambiguidade maior de rótulos por estágio de evolução. A porção usada é a de OCT, com `CNV`, `DME`, `DRUSEN` e `NORMAL`; a porção de raio-X não entra no escopo.

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dados

Faça o download do dataset e aponte `--archive` para ele. O conteúdo será extraído em `--raw-data-dir`.

```bash
python scripts/download_data.py --archive /caminho/para/OCT2017.zip --raw-data-dir data/raw
```

Também é possível extrair manualmente o arquivo em `data/raw` e seguir para o preparo.

## Preparo

```bash
python scripts/prepare_data.py --raw-data-dir data/raw --manifest data/processed/manifest.csv
```

O preparo preserva o teste oficial quando as pastas `train` e `test` existem no dataset. A validação é criada apenas a partir do conjunto de treino, separando pacientes pelo `patient_id` extraído do nome do arquivo. Se houver uma pasta `val` no pacote bruto, ela é incorporada ao pool de treino antes de criar a validação do projeto. O teste final não é usado para decisões de treino.

O manifesto final fica em `data/processed/manifest.csv` com as colunas `image_path`, `label`, `patient_id` e `split`. O resumo de distribuição por classe e split fica em `data/processed/summary.json`.

## Modelos

Três abordagens são configuradas:

- `configs/hog_logreg.yaml`: HOG + Logistic Regression.
- `configs/simple_cnn.yaml`: CNN pequena treinada do zero.
- `configs/mobilenetv3.yaml`: MobileNetV3-Small com transfer learning.

O desbalanceamento é tratado no caminho principal. O baseline usa `class_weight="balanced"` na regressão logística. Os modelos neurais usam `WeightedRandomSampler` por padrão, configurado em `imbalance.strategy`.

## Execução

```bash
python scripts/train.py --config configs/hog_logreg.yaml
python scripts/train.py --config configs/simple_cnn.yaml
python scripts/train.py --config configs/mobilenetv3.yaml
```

Ou execute tudo em sequência:

```bash
python scripts/run_all.py --raw-data-dir data/raw --manifest data/processed/manifest.csv
```

Para reavaliar uma execução salva:

```bash
python scripts/evaluate.py --run outputs/<run_id> --split test
```

Cada execução salva em `outputs/<run_id>/`: `config.yaml`, `metrics.json`, `classification_report.txt`, `confusion_matrix.png`, `roc_curve.png`, e também `history.csv`/`best_model.pt` para modelos neurais ou `model.joblib` para o baseline HOG.

## Métricas

As métricas principais são `accuracy`, `balanced_accuracy`, `macro_precision`, `macro_recall`, `macro_f1` e `roc_auc_macro_ovr`, além de matriz de confusão, curvas ROC por classe e relatório de classificação.

## Limitações

Os resultados não têm finalidade diagnóstica. A avaliação depende da qualidade do split por paciente inferido dos nomes dos arquivos e deve ser interpretada no contexto da distribuição do dataset. O projeto não usa estágio de doença, não usa vídeo e não aborda a porção de raio-X.

