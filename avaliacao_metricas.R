# avaliacao_metricas.R
# --------------------
# Script para avaliação de métricas de modelos a partir dos arquivos de predição gerados em Python.
#
# Dependências sugeridas: tidyverse, readxl, lubridate, caret, DescTools, purrr
#
# ------------------------------------------------------------------------------

# Carregamento de bibliotecas necessárias
library(tidyverse)
library(readxl)
library(lubridate)
library(caret)
library(DescTools)
library(purrr)

# ---------------------------------------------------------------------
# Função principal: calcular_metricas_completas
# Objetivo: carregar predições (treino e teste), mesclar com dados observacionais
# e calcular métricas agregadas (global) e por grupos (época/classificação, site, profundidade).
# Retorna uma lista com os conjuntos finais e diferentes agregações de métricas.
# ---------------------------------------------------------------------
calcular_metricas_completas <-
  function(
    val_obs,
    val_id,
    file_treino,
    file_teste,
    file_base
  ){
    
    # Carrega os resultados de predição do conjunto de treino.
    dados_treino <- 
      read.table(
        file_treino,h=T,sep = ";"
      ) |> 
      as_tibble() |> 
      # Renomeia todas as colunas que contêm "prediction_label" para "pred" para padronização.
      rename_all(
        ~{gsub("prediction_label","pred",.x)}
      )
    
    # Carrega os resultados de predição do conjunto de teste.
    dados_teste <- 
      read.table(file_teste,h=T,sep = ";") |> 
      as_tibble() |> 
      rename_all(
        ~{gsub("prediction_label","pred",.x)}
      )
    
    
    # Junta os dados de teste e treino em um único dataframe.
    # - Adiciona coluna 'particao' para distinguir treino/teste
    # - Normaliza nomes (Data -> Time) e garante que Time seja datetime
    # - Separa a coluna ID em Site e Classificação (assume formato "Site_Classificação")
    dados_geral <-
      dados_teste |> 
      mutate(particao = "Teste") |> 
      bind_rows(
        dados_treino |> 
          mutate(particao = "Treino")
      ) |> 
      rename_all(function(x) gsub("Data","Time",x)) |> 
      relocate(Time) |> 
      mutate(Time = lubridate::as_datetime(Time)) |> 
      arrange(Time) |> 
      mutate(Data = as.Date(Time)) |> 
      separate(ID,c("Site","Classificação"),sep = "_")
    
    
    # Carrega os dados observacionais de base de um arquivo Excel.
    # Seleciona apenas as colunas de interesse (IDs e variável observada).
    dados_base <- 
      readxl::read_excel(file_base) |> 
      select(all_of(val_id),all_of(val_obs)) 
    
    # Une os dados de predição com os dados observacionais pela chave definida em val_id + val_obs.
    dados_end <-
      dados_geral |>
      left_join(dados_base,by = set_names(c(val_id, val_obs))) |> 
      relocate(Site,Classificação) |> 
      arrange(all_of(val_id[1]));dados_end
    
    # Cria diferentes partições/agrupamentos para cálculo de métricas:
    # - por época/classificação
    # - por site
    # - por profundidade (prefixando com "u")
    # - por site x profundidade (interação)
    dados_epoca <- split(dados_end,dados_end$Classificação)
    dados_site <- split(dados_end,dados_end$Site)
    dados_prof <- split(dados_end,paste0("u",dados_end$Profundidade))
    dados_site_prof <- split(dados_end,interaction(dados_end$Site,dados_end$Profundidade))
    
    # Identifica os nomes das colunas que terminam com "_pred", que são as predições dos modelos.
    colunas_pred <- names(dados_teste)[grepl("_pred$", names(dados_teste))]
    
    
    # -----------------------------------------------------------------
    # 3. Análise de Métricas de Performance
    # - Calcula métricas globais (treino e teste)
    # - Calcula métricas por agrupamentos usando funções auxiliares
    # -----------------------------------------------------------------
    
    # Calcula as métricas globais para o conjunto de teste e treino, organiza e marca partição.
    metricas_geral <-
      fn_metricas_globais(
        dados = dados_teste,
        col_obs = val_obs,
        vec_cols_pred = colunas_pred
      ) |>
      arrange(desc(CCC)) |>
      mutate(particao = "teste") |> 
      bind_rows(
        fn_metricas_globais(
          dados = dados_treino,
          col_obs = val_obs,
          vec_cols_pred = colunas_pred
        ) |>
          arrange(desc(CCC)) |>
          mutate(particao = "treino")
      ) |> 
      mutate(vary = val_obs)
    
    
    # Calcula as métricas para cada 'época', retornando uma lista de resultados.
    list_epoca <-
      fn_list_metricas_id(
        dados = dados_epoca,
        col_obs = val_obs,
        vec_cols_pred = colunas_pred,
        vec_ids = c(val_id,"particao")
      )
    
    # Calcula as métricas para cada 'local de estudo', retornando uma lista de resultados.
    list_site <-
      fn_list_metricas_id(
        dados = dados_site,
        col_obs = val_obs,
        vec_cols_pred = colunas_pred,
        vec_ids = c(val_id,"particao")
      )
    
    # Calcula as métricas para cada 'profundidade', retornando uma lista de resultados.
    list_prof <-
      fn_list_metricas_id(
        dados = dados_prof,
        col_obs = val_obs,
        vec_cols_pred = colunas_pred,
        vec_ids = c(val_id,"particao")
      )
    
    # Calcula as métricas para cada 'local e profundidade', retornando uma lista de resultados.
    list_site_prof <-
      fn_list_metricas_id(
        dados = dados_site_prof,
        col_obs = val_obs,
        vec_cols_pred = colunas_pred,
        vec_ids = c(val_id,"particao")
      )
    
    
    
    
    #retornar lista com resultados estruturados
    return(
      list(
        dados_end = dados_end,
        metricas_geral = metricas_geral,
        list_epoca = list_epoca,
        list_site = list_site,
        list_prof = list_prof,
        list_site_prof = list_site_prof
      )
    )
  }

# ---------------------------------------------------------------------
# Função auxiliar: fn_metricas_globais
# Objetivo: dado um data.frame/tibble com observações e colunas de predição,
# calcular métricas agregadas (RMSE, MAE, R2, CCC) para cada coluna de predição.
# Retorna um tibble com uma linha por modelo/predição.
# ---------------------------------------------------------------------
fn_metricas_globais <-
  function(
    dados,
    col_obs,
    vec_cols_pred
  ){
    metricas_globais <-
      map_dfr(
        vec_cols_pred,
        ~{
          pred <- dados[[.x]]
          obs <- dados[[col_obs]]
          
          # Métricas globais
          rmse_val <- caret::RMSE(pred = pred, obs = obs)
          mae_val  <- caret::MAE(pred = pred, obs = obs)
          r2_val   <- caret::R2(pred = pred, obs = obs)
          ccc_val  <- DescTools::CCC(pred, obs)$rho.c[1]
          
          # Tabela linha a linha com colunas fixas para as métricas globais
          tibble(
            RMSE = rmse_val,
            MAE = mae_val,
            R2 = r2_val,
            CCC = as.numeric(ccc_val),
            modelo = .x,
          )
        }
      )
    return(metricas_globais)
  }

# ---------------------------------------------------------------------
# Função auxiliar: fn_metricas_pontuais
# Objetivo: calcular métricas pontuais (por observação) para cada modelo/predição.
# Atualmente retorna MAE por observação; estrutura para estender é simples.
# ---------------------------------------------------------------------
fn_metricas_pontuais <-
  function(
    dados,
    col_obs,
    vec_cols_pred,
    col_id 
  ){
    metricas_pontuais <-
      map_dfr(
        vec_cols_pred,
        ~{
          dados |> 
            select(all_of(col_id)) |> 
            mutate(
              pred = dados[[.x]],
              obs = dados[[col_obs]]
            ) |> 
            as_tibble() |> 
            rowwise() |> 
            mutate(
              # Calcula MAE ponto a ponto; pode-se expandir para RMSE local etc.
              MAE = caret::MAE(pred = pred,obs = obs)
            ) |> 
            mutate(modelo =.x) |> 
            ungroup()
        }
      )
    
    return(metricas_pontuais)
  }

# ---------------------------------------------------------------------
# Função auxiliar: fn_list_metricas_id
# Objetivo: para uma lista de data.frames (por agrupamento), calcular:
# - métricas globais separadas por partição (treino/teste)
# - métricas pontuais (por observação)
# Retorna uma lista com elementos contendo 'global' e 'pontual'.
# ---------------------------------------------------------------------
fn_list_metricas_id <- 
  function(
    dados,
    col_obs,
    vec_cols_pred,
    vec_ids
  ){
    list_result <-
      map(
        dados,
        ~{
          
          map_treino <-
            .x |> 
            filter(particao == "Treino")
          map_teste <-
            .x |> 
            filter(particao == "Teste")
          
          treino <-
            fn_metricas_globais(
              dados = map_treino,
              col_obs = col_obs,
              vec_cols_pred = vec_cols_pred
            ) |> 
            mutate(particao = "treino")
          
          teste <-
            fn_metricas_globais(
              dados = map_teste,
              col_obs = col_obs,
              vec_cols_pred = vec_cols_pred
            ) |> 
            mutate(particao = "teste")
          
          pontual <- 
            fn_metricas_pontuais(
              dados = .x,
              col_obs = col_obs,
              vec_cols_pred = vec_cols_pred,
              col_id = vec_ids
            ) 
          
          list(
            global = bind_rows(treino,teste),
            pontual = pontual
          )
        }
      )
    return(list_result)
  }

# ---------------------------
# Exemplo de uso (comentado)
# ---------------------------
# source("avaliacao_metricas.R")
# resultados <- calcular_metricas_completas(
#   val_obs = "Umidade",
#   val_id = c("Time","ID","Profundidade"),
#   file_treino = "./resultados/u10_u60/train_predict_week_Umidade.csv",
#   file_teste = "./resultados/u10_u60/test_predict_week_Umidade.csv",
#   file_base = "./resultados/dados/dados_semanal_prof.xlsx"
# )
#
# # Acessar métricas globais:
# resultados$metricas_geral
#
# # Acessar métricas por época:
# resultados$list_epoca[[1]]$global   # exemplo para o primeiro elemento da lista de épocas
#
# # Observação: as listas retornadas (list_epoca, list_site, ...) contêm elementos
# # com duas partes: $global (tibble) e $pontual (tibble com MAE por observação).
