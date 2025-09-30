# 📊 Pipeline de Modelagem e Avaliação de Predições de Umidade do Solo

Este repositório contém o pipeline completo para **treinamento, predição e avaliação de modelos de aprendizado de máquina** aplicados à estimativa de **umidade do solo** em diferentes profundidades, épocas e locais experimentais.

O fluxo de trabalho está dividido em duas etapas principais:

1. **Modelagem e geração de predições (`modelagem.py`)**  
2. **Avaliação de desempenho dos modelos (`avaliacao_metricas.R`)**

---

## 🧠 1. Modelagem (`modelagem.py`)

Este script realiza todo o processo de modelagem:

- Pré-processamento dos dados de entrada  
- Treinamento de múltiplos modelos (e.g., CatBoost, SVR, Lasso, etc.)  
- Geração de predições para os conjuntos de **treino** e **teste**  
- Exportação das predições em formato `.csv` para uso posterior na avaliação


## 📈 2. Avaliação de Métricas (avaliacao_metricas.R)

O script em R realiza a avaliação detalhada do desempenho dos modelos utilizando as predições geradas na etapa anterior.
Ele calcula métricas clássicas como:

- RMSE – Raiz do erro quadrático médio
- MAE – Erro Absoluto Médio
- R² – Coeficiente de Determinação
- CCC – Coeficiente de Correlação de Concordância de Lin

As métricas são calculadas de forma global e também por diferentes níveis de agrupamento:

- Por Época
- Por Local 
- Por Profundidade
- Por Local × Profundidade

📂 Estrutura do objeto retornado

- resultados$dados_end → Dados completos mesclados (observados + previstos)
- resultados$metricas_geral → Métricas globais por modelo
- resultados$list_epoca → Métricas por época
- resultados$list_site → Métricas por época
- resultados$list_prof → Métricas por profundidade
- resultados$list_site_prof → Métricas por época × profundidade

## 📦 Dependências

Python:
- pandas
- numpy
- scikit-learn
- catboost
- joblib

R:
- tidyverse
- readxl
- lubridate
- caret
- DescTools
- purrr
