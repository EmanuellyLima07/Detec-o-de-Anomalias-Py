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
| Regressão Logística (baseline) | _preencher_ | _preencher_ | _preencher_ |
| Regressão Logística + SMOTE | _preencher_ | _preencher_ | _preencher_ |
| Random Forest | _preencher_ | _preencher_ | _preencher_ |
| XGBoost | _preencher_ | _preencher_ | _preencher_ |
| XGBoost + GridSearchCV | _preencher_ | _preencher_ | _preencher_ |

## Avaliação
- Curvas ROC e Precisão-Recall (baseline). AUC: `_preencher_`.
- Limiar de decisão testado: `0.3` (padrão é `0.5`), para aumentar o recall de fraude.
- Importância das variáveis via `feature_importances_` do XGBoost e explicação individual via SHAP.

## Como Rodar
```bash
pip install -r requirements.txt
```
Depois, abra `Deteccao_de_Fraudes.ipynb` no Google Colab, Jupyter ou VSCode e execute as células em ordem.
