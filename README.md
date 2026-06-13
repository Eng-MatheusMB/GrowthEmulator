# 🧫 GrowthEmulator v1.2 — Simulador de Crescimento Celular Microbiano

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io/)

**GrowthEmulator** é uma plataforma interativa de modelagem cinética microbiana desenvolvida para
pesquisadores, engenheiros de bioprocessos e estudantes.

---

## ✨ Recursos

| Recurso | Detalhe |
|---------|---------|
| 🔬 **Modelos cinéticos** | 30+ modelos em 11 categorias (Malthus, Monod, Gompertz, Baranyi-Roberts…) |
| 📁 **Importação inteligente** | `.csv`, `.xlsx`, `.txt` com detecção automática de delimitadores BR/EN |
| 📈 **Visualização** | Plotly interativo com banda ±1σ, resíduos e distribuição normal overlay |
| 📐 **Métricas** | RMSE, R² Ajustado, AIC, BIC + teste de Shapiro-Wilk/D'Agostino |
| 🌱 **Fases de crescimento** | Detecção automática de lag, exponencial, estacionária, declínio |
| 📄 **Exportação** | PDF (relatório) + Excel 5 abas (Dados, Parâmetros, Métricas, Ajuste, Fases) |
| ♿ **Acessibilidade** | Modo escuro/claro, alto contraste, deuteranopia, protanopia |
| 🌐 **Idiomas** | Português · English · Español · 中文 |

---

## 🚀 Instalação rápida

### Pré-requisitos
- Python 3.9 ou superior
- pip

### 1. Clonar o repositório
```bash
git clone https://github.com/matheusmonteirobatista/GrowthEmulator.git
cd GrowthEmulator
```

### 2. Instalar dependências
```bash
pip install -r requirements.txt
```

### 3. Executar
```bash
streamlit run app.py
```

O simulador abrirá automaticamente em `http://localhost:8501`.

---

## 📦 Dependências

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
scipy>=1.11.0
openpyxl>=3.1.0
fpdf2>=2.7.0
```

---

## 🗂️ Estrutura do projeto

```
GrowthEmulator/
├── app.py                   ← Aplicação Streamlit principal
├── requirements.txt         ← Dependências Python
├── index.html               ← Landing page (HTML5 + Tailwind CSS)
├── README.md                ← Este arquivo
├── assets/
│   └── logo.png             ← Logo do simulador
└── .streamlit/
    └── config.toml          ← Tema padrão (dark)
```

---

## 🔬 Famílias de modelos disponíveis

| Box | Categoria | Modelos |
|-----|-----------|---------|
| 1 | Exponencial & Linear | Malthus, Linear |
| 2 | Logísticos | Verhulst, Gompertz Mod., Logístico Mod. |
| 3 | Mecanísticos | Baranyi-Roberts, Contois, Herbert-Pirt |
| 4 | Sigmoides | Richards, Von Bertalanffy |
| 5 | Cinética Homogênea | Michaelis-Menten, Inibição Competitiva |
| 6 | Cinética Heterogênea | Pirt – Rendimento Variável |
| 7 | Turn-over Celular | Luedeking-Piret, Chick |
| 8 | Clássicos (S-dep.) | Monod, Tessier, Moser, Haldane/Andrews, Aiba |
| 9 | Empíricos | Buchanan Trifásico |
| 10 | Ajuste Linear | Polinomial, Log-linear |
| 11 | Ajuste Não Linear | Gaussiana, Lei de Potência |

---

## 📖 Fluxo de trabalho

```
📊 Dados → 🔬 Modelos Cinéticos → 📈 Resultados → 📤 Exportar
```

1. **Dados**: Faça upload do arquivo ou insira dados manualmente. Mapeie as colunas para as variáveis do simulador.
2. **Modelos**: Visualize os 11 boxes. Somente os compatíveis com seus dados são ativados.
3. **Resultados**: Configure a otimização, execute, analise parâmetros e métricas.
4. **Exportar**: Baixe PDF ou Excel com resultados completos.

---

## 📚 Principais referências

- Monod, J. (1949). *The growth of bacterial cultures.* Annu. Rev. Microbiol., 3, 371–394.
- Zwietering, M.H. et al. (1990). *Modeling of the Bacterial Growth Curve.* Appl. Environ. Microbiol., 56(6), 1875–1881.
- Baranyi, J. & Roberts, T.A. (1994). *A dynamic approach to predicting bacterial growth in food.* Int. J. Food Microbiol., 23(3–4), 277–294.
- Bailey, J.E. & Ollis, D.F. (1986). *Biochemical Engineering Fundamentals* (2nd ed.). McGraw-Hill.
- Shuler, M.L. & Kargi, F. (2002). *Bioprocess Engineering* (2nd ed.). Prentice Hall.

Lista completa de referências disponível na aba **📖 Guia & Referências** dentro do simulador.

---

## ⚖️ Licença

**GrowthEmulator v1.2** © 2026 by **Matheus Monteiro Batista** is licensed under
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).

Uso acadêmico e pessoal permitido com atribuição.
Para uso comercial, entre em contato: matheus@example.com

---

## 📧 Contato

- **E-mail**: matheus@example.com
- **LinkedIn**: [linkedin.com/in/matheusmonteirobatista](https://www.linkedin.com/in/matheusmonteirobatista/)
- **GitHub**: [github.com/matheusmonteirobatista/GrowthEmulator](https://github.com/matheusmonteirobatista/GrowthEmulator)
