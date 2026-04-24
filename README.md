# 🎵 Prevendo Músicas Populares do Spotify com Machine Learning

> Projeto desenvolvido como parte do desafio **[7 Days of Code - Machine Learning](https://7daysofcode.io/)** da **[Alura](https://www.alura.com.br/)**.

---

## 📌 Sobre o Projeto

Este projeto tem como objetivo **prever se uma música do Spotify será popular** utilizando técnicas de Machine Learning. A partir de características sonoras de cada faixa — como dançabilidade, energia, volume e positividade — foi construído um classificador binário capaz de identificar potenciais *hits*.

O pipeline completo cobre desde a ingestão dos dados até a exportação do modelo treinado, passando por análise exploratória, pré-processamento, tratamento de desbalanceamento de classes, seleção e otimização de modelos.

---

## 📂 Dataset

- **Fonte:** [Spotify Tracks Dataset — Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
- **Acesso:** Download automatizado via `kagglehub`
- **Conteúdo:** Faixas musicais com atributos como popularidade, dançabilidade, energia, volume, modo, tempo, entre outros

---

## 🔄 Pipeline do Projeto

### 1. 📥 Importação e Tradução dos Dados
- Carregamento do CSV com `pandas`
- Remoção de colunas desnecessárias (`Unnamed: 0`)
- Tradução dos nomes das colunas para português (ex: `danceability` → `dancabilidade`)

---

### 2. 🔍 Análise Exploratória (EDA)
Geração de rankings para entender o dataset:

| Ranking | Critério |
|---|---|
| Artistas mais populares | Média de popularidade por artista |
| Músicas mais longas | Duração média em milissegundos |
| Músicas mais dançáveis | Índice de danceability |

Verificação de valores nulos, tipos de dados e distribuições estatísticas com `.describe()` e `.info()`.

---

### 3. 🛠️ Pré-processamento
- Remoção de **duplicatas** e **valores nulos**
- Criação da **variável-alvo binária:**
  ```
  popular_binario = 1  →  popularidade ≥ 60
  popular_binario = 0  →  popularidade < 60
  ```
- Remoção de **colunas qualitativas** (nome da música, artista, gênero)
- **Normalização Min-Max** de todas as features numéricas para o intervalo [0, 1]

---

### 4. ✂️ Divisão dos Dados

A estratégia adotada separa os dados em três conjuntos distintos:

```
Dataset Completo
│
├── 80% → Dados de Treino
│         ├── Treino     (StratifiedKFold — 5 folds)
│         └── Validação  (comparação de modelos e hiperparâmetros)
│
└── 20% → Dados de Teste (reservado para avaliação final)
```

O **StratifiedKFold** garante a proporção das classes em cada fold, essencial dado o desbalanceamento do dataset.

---

### 5. 📊 Modelo Baseline — Regressão Logística

A Regressão Logística sem ajustes foi usada como ponto de partida:

| Métrica | Resultado |
|---|:---:|
| Acurácia geral | ~87% |
| Precisão (músicas populares) | 0.00 |
| Recall (músicas populares) | 0.00 |
| F1-Score (músicas populares) | 0.00 |

> ⚠️ O modelo ignorava completamente a classe positiva — um sinal claro de desbalanceamento de classes.

---

### 6. ⚖️ Tratamento de Desbalanceamento

Três técnicas de reamostragem foram comparadas, cada uma aplicada sobre a Regressão Logística:

| Técnica | Descrição |
|---|---|
| **SMOTE** | Oversampling sintético — cria novos exemplos da classe minoritária |
| **RUS** *(Random Under Sampling)* | Reduz a classe majoritária aleatoriamente |
| **ROS** *(Random Over Sampling)* | Duplica exemplos da classe minoritária |

---

### 7. 🌲 Random Forest

Motivado pelas limitações lineares da Regressão Logística, o **Random Forest** (100 estimadores) foi testado com as mesmas três técnicas de reamostragem, apresentando desempenho significativamente superior.

---

### 8. 🔧 Ajuste de Hiperparâmetros — RandomizedSearchCV

Otimização do Random Forest com busca aleatória:

```python
param_dist_rf = {
    'n_estimators':      [100, 200, 300],
    'max_depth':         [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4],
    'bootstrap':         [True, False]
}
```

- **Iterações:** 10 combinações avaliadas
- **Validação cruzada:** 3 folds
- **Métrica de otimização:** F1-Score
- **Combinação vencedora:** Random Forest Otimizado + ROS

---

### 9. 📈 Avaliação Final

#### Curva ROC
O modelo final foi avaliado com a **Curva ROC**, obtendo um **AUC de 0.88** — o que significa que há 88% de chance de o modelo rankear corretamente uma música popular acima de uma não popular.

#### Importância das Features
A importância de cada variável foi extraída do Random Forest, revelando quais características musicais mais influenciam na previsão de popularidade.

---

## 🏆 Resultados Comparativos

| Métrica | Baseline (Reg. Logística) | Random Forest Final |
|---|:---:|:---:|
| **Acurácia Geral** | 86.95% | **92.11%** |
| **Precisão (hits)** | 0.00 | **0.92** |
| **Recall (hits)** | 0.00 | **0.43** |
| **F1-Score (hits)** | 0.00 | **0.59** |
| **AUC-ROC** | — | **0.88** |

> O Random Forest aprendeu padrões reais que distinguem *hits* das demais faixas, enquanto o baseline simplesmente ignorava a classe minoritária.

---

## 💾 Exportação do Modelo

O modelo final foi serializado com `joblib` para uso futuro:

```python
import joblib
joblib.dump(best_rf_ros, 'modelo_spotify_random_forest.pkl')
```

---

## 🧰 Tecnologias e Bibliotecas

| Categoria | Ferramentas |
|---|---|
| Linguagem | Python 3 |
| Manipulação de dados | `pandas`, `numpy` |
| Visualização | `matplotlib`, `seaborn` |
| Machine Learning | `scikit-learn` |
| Reamostragem | `imbalanced-learn` |
| Serialização | `joblib` |
| Ambiente | Google Colab |

---

## 🗂️ Estrutura do Projeto

```
📁 Machine-Learning-com-dados-do-Spotify/
│
├── 📓 Analise_dos_dados_com_ML.ipynb   # Notebook principal
├── 🤖 modelo_spotify_random_forest.pkl # Modelo exportado
└── 📄 README.md                        # Este arquivo
```

---

## 💡 Inspiração

Este projeto foi inspirado e guiado pela atividade **[7 Days of Code](https://7daysofcode.io/)** da **[Alura](https://www.alura.com.br/)**, um desafio prático onde desenvolvedores aplicam conceitos de Data Science e Machine Learning em problemas reais ao longo de 7 dias.

---

## 👤 Autor

Desenvolvido como projeto de aprendizado prático em Machine Learning.

---

*⭐ Se este projeto foi útil para você, considere deixar uma estrela no repositório!*