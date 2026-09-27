# Detecção de Fraude em Cartão de Crédito

Modelo de classificação para identificar transações fraudulentas em um dataset extremamente desbalanceado (0,17% de fraudes).

## Dataset
Carregado diretamente via link (não incluído no repositório): `storage.googleapis.com/download.tensorflow.org/data/creditcard.csv`.

## Preparação dos Dados
- `Amount_log`: log1p do valor da transação, para reduzir assimetria.
- `Time_scaled`: Time padronizado com `StandardScaler`.
- Colunas originais `Amount` e `Time` removidas do X.
- Split treino/teste (70/30) com `stratify=y`.

## Modelos Comparados
- Regressão Logística (baseline)
- Regressão Logística + SMOTE (aplicado só no treino)
- Random Forest (`class_weight="balanced"`)
- Pipeline (Scaler + Regressão Logística)
- XGBoost (`scale_pos_weight` calculado pela proporção real das classes)
- XGBoost otimizado via `GridSearchCV` (scoring="recall")

| Modelo | Recall (fraude) | Precisão (fraude) | F1 (fraude) |
|---|---|---|---|
| Regressão Logística (baseline) | 0.62 | 0.86 | 0.72 |
| Regressão Logística + SMOTE | 0.86 | 0.06 | 0.12 |
| Random Forest | 0.78 | 0.82 | 0.80 |
| Pipeline (Scaler + Regressão Logística) | 0.62 | 0.86 | 0.72 |
| XGBoost | 0.65 | 0.79 | 0.83 |
| XGBoost + GridSearchCV | 0.72 | 0.84 | 0.78 |

## Avaliação
- Curvas ROC e Precisão-Recall (baseline). AUC: `~0.96`.
- Limiar de decisão testado: `0.3` (padrão é `0.5`), para aumentar o recall de fraude.
- Importância das variáveis via `feature_importances_` do XGBoost e explicação individual via SHAP.

## Como Rodar
(caso não use o google colab)

```bash
pip install -r requirements.txt
```
Depois, abra `Deteccao_de_Fraudes.ipynb` no Google Colab, Jupyter ou VSCode e execute as células em ordem.
