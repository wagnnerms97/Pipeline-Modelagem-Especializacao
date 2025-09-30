"""
Este script realiza a etapa de **treinamento e geração de predições** de modelos de aprendizado de máquina
utilizando a biblioteca PyCaret. Ele carrega os dados, prepara o conjunto de treino e teste,
compara e seleciona modelos, gera predições e exporta os resultados em arquivos.csv
para posterior avaliação no R.

Fluxo geral:
1. Leitura e preparação dos dados
2. Configuração do ambiente de modelagem (PyCaret)
3. Comparação e seleção de modelos
4. Geração de predições no treino e teste
5. Exportação dos resultados

Autor: Thieres George Freire da Silva
"""

import pandas as pd
from pycaret.regression import setup, compare_models, predict_model, pull
from sklearn.model_selection import train_test_split

# ==========================================================
# 1. LEITURA E PRÉ-PROCESSAMENTO DOS DADOS
# ==========================================================
# Carrega o dataset original contendo variáveis observadas e preditoras.
# O arquivo Excel deve estar no diretório especificado.
dados = pd.read_excel("./resultados/dados/dados_semanal_prof.xlsx")

# Converte a coluna "Profundidade" para string para tratá-la como categórica
dados['Profundidade'] = dados['Profundidade'].astype(str)

# Define a variável resposta (target) e colunas categóricas
vary = "Umidade"
categorical_features = ['Site', "Classificação", 'Profundidade']

# Define as colunas que devem ser ignoradas no processo de modelagem
ignore_features = ['Time', 'ID']

# Cria um subconjunto de dados apenas com as colunas relevantes
dados_select = dados[ignore_features + [vary, "Rain_sum", "TM_mean", "URM_mean", "DPVM_mean", "Rg_mean"] + categorical_features]

# Define parâmetros de salvamento
folder_save = "u10_u60"
type_save = "_week_" + vary

# ==========================================================
# 2. DIVISÃO DOS DADOS EM TREINO E TESTE
# ==========================================================
# O conjunto de dados é dividido em treino (70%) e teste (30%) para avaliar a performance dos modelos.
train_data, test_data = train_test_split(
    dados_select,
    test_size=0.3,
    random_state=42
)

# ==========================================================
# 3. CONFIGURAÇÃO DO AMBIENTE DE MODELAGEM (PYCARET)
# ==========================================================
# A função setup() inicializa o ambiente de modelagem:
# - Define a variável alvo
# - Realiza normalização
# - Configura colunas categóricas e ignora colunas não utilizadas
clf = setup(
    data=train_data,
    target=vary,
    use_gpu=True,
    normalize=True,
    normalize_method='zscore',
    remove_outliers=False,
    session_id=42,
    fold=10,
    log_experiment=False,
    ignore_features=ignore_features,
    n_jobs=-1,
    categorical_features=categorical_features
)

# ==========================================================
# 4. COMPARAÇÃO E SELEÇÃO DE MODELOS
# ==========================================================
# Compara automaticamente diversos modelos de regressão e seleciona os 26 melhores.
models_train = clf.compare_models(
    verbose=True,
    n_select=26,
    turbo=False
)

# Coleta a tabela de resultados de comparação de modelos
resultados = pull().reset_index()
names_models = resultados["index"]
print(names_models)

# ==========================================================
# 5. FUNÇÕES PARA GERAÇÃO E ORGANIZAÇÃO DAS PREDIÇÕES
# ==========================================================

def extract_pred_classe(model, data, name_model):
    """
    Gera as predições de um modelo específico sobre um conjunto de dados.
    Renomeia as colunas de predição para incluir o nome do modelo.
    """
    predictions = predict_model(model, data)
    predictions = predictions.filter(like='predic')  # mantém apenas as colunas de predição

    old_name_columns = predictions.columns.tolist()
    new_name_columns = [name_model + "_" + item for item in old_name_columns]

    # Renomeia colunas para identificar qual modelo produziu cada predição
    predictions = predictions.rename(columns=dict(zip(old_name_columns, new_name_columns)))
    return predictions


def unite_predicts_models(models_predict, data, col_name_target, col_name_id, names_models):
    """
    Combina as predições de todos os modelos treinados em um único DataFrame.
    Inclui a variável observada e identificadores originais.
    """
    if isinstance(col_name_id, str):
        col_name_id = [col_name_id]

    result_predict = []
    # Adiciona a variável observada
    result_predict.append(data[col_name_target])

    # Gera e adiciona predições para cada modelo treinado
    for i, model in enumerate(models_predict):
        predictions = extract_pred_classe(model, data, names_models[i])
        result_predict.append(predictions)

    # Concatena identificadores, observações e predições
    result_predict = pd.concat([data[col_name_id]] + result_predict, axis=1)
    return result_predict

# ==========================================================
# 6. GERAÇÃO DE PREDIÇÕES E SALVAMENTO
# ==========================================================
# Gera predições para o conjunto de teste e treino utilizando todos os modelos
results_test = unite_predicts_models(models_train, test_data, vary, ["Time", "ID", "Profundidade"], names_models)
results_train = unite_predicts_models(models_train, train_data, vary, ["Time", "ID", "Profundidade"], names_models)

# Exporta os resultados em formato CSV para posterior análise no R
results_test.to_csv("./resultados/" + folder_save + "/test_predict" + type_save + ".csv", index=False, sep=";")
results_train.to_csv("./resultados/" + folder_save + "/train_predict" + type_save + ".csv", index=False, sep=";")

print("✅ Predições geradas e salvas com sucesso.")


