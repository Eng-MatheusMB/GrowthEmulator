#!/usr/bin/env python3
"""
GrowthEmulator v1.2 — Simulador de Crescimento Celular Microbiano
© 2026 Matheus Monteiro Batista | CC BY-NC-ND 4.0
"""
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. IMPORTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import streamlit as st
# components.html kept for AdSense (st.html strips scripts)
from streamlit.components.v1 import html as _components_html
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit, minimize, differential_evolution
from scipy.integrate import solve_ivp
from io import BytesIO
from pathlib import Path
import base64, os, warnings, re, copy
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="GrowthEmulator v1.2",
    page_icon="🧫",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "GrowthEmulator v1.2 © 2026 Matheus Monteiro Batista"},
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. TRANSLATIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_T = {
    "pt": {
        "app_name": "GrowthEmulator v1.2",
        "nav_data": "📊 Dados",
        "nav_models": "🔬 Modelos Cinéticos",
        "nav_tools": "🔧 Ferramentas",
        "nav_results": "📈 Resultados",
        "nav_about": "ℹ️ Sobre",
        "nav_guide": "📖 Guia & Referências",
        "nav_contact": "📧 Entre em contato",
        "dark_mode": "🌙 Escuro",
        "light_mode": "☀️ Claro",
        "contrast": "♿ Contraste",
        "lang_label": "Idioma",
        # --- Data Tab ---
        "dt_title": "Entrada de Dados",
        "dt_upload_label": "Enviar arquivo de dados",
        "dt_upload_types": "Suporte para .csv, .xlsx e .txt (delimitadores BR e EN detectados automaticamente)",
        "dt_header_row": "Linha do cabeçalho",
        "dt_header_note": "💡 Informe a linha onde estão os nomes das colunas (1 = primeira linha)",
        "dt_mapping_title": "Mapeamento de variáveis",
        "dt_mapping_note": "Associe cada coluna do seu arquivo às variáveis do simulador",
        "dt_none": "— Não mapeado —",
        "dt_time_lbl": "Tempo (h)",
        "dt_biomass_lbl": "Concentração (g/L, OD)",
        "dt_substrate_lbl": "Substrato (g/L)",
        "dt_ph_lbl": "pH",
        "dt_product_lbl": "Produto (g/L)",
        "dt_drymass_lbl": "Massa seca (g)",
        "dt_tool": "Mapeie **Tempo**, **Biomassa** e **Produto** na aba 📊 Dados para habilitar.",
        "dt_info": "Mapeie **Tempo** e **Biomassa** para ajustar o modelo de Chick",
        "dt_manual_title": "Entrada manual de dados",
        "dt_manual_note": "Use Ctrl+V para colar dados de uma planilha",
        "dt_manual_add_col": "＋ Coluna",
        "dt_manual_rem_col": "－ Coluna",
        "dt_manual_add_row": "＋ Linha",
        "dt_manual_rem_row": "－ Linha",
        "dt_manual_clear": "🗑 Limpar tabela",
        "dt_files_merge": " arquivos mesclados (outer join no eixo do tempo)",
        "dt_manual_add_data": " Aplicar dados manuais",
        "dt_summary_title": "Resumo dos dados",
        "dt_nan_warn": "⚠️ Linhas com valores ausentes serão desconsideradas nas análises.",
        "dt_neg_time": "🚫 Valores negativos de tempo detectados!",
        "dt_dup_time": "⚠️ Valores de tempo duplicados detectados!",
        "dt_preview_title": "Prévia — Dispersão",
        "dt_preview_log": "Eixo Y em escala logarítmica",
        "dt_avail_title": "Modelos disponíveis",
        "dt_avail_note": "com os headers atuais",
        "dt_avail_complement0": " Modelos cinéticos",
        "dt_avail_complement1":"Ferramentas",
        "dt_avail_prep":"de",
        "dt_select_btn": "🔬 Selecionar modelo para análise →",
        "dt_select_disabled": "Mapeie pelo menos **Tempo** e **Biomassa** para habilitar",
        "dt_excl_cols": "Colunas excluídas da análise",
        "dt_excl_note": "Selecione as colunas que deseja remover dos dados ativos",
        "dt_multi_file": "Múltiplos arquivos carregados — combinados por posição de linhas",
        # --- Tools Tab ---
        "tl_title": "Ferramentas",
        "tl_enzyme_title": "Cinética Enzimática (Michaelis-Menten)",
        "tl_enzyme_note": "Analise a relação velocidade-substrato para reações enzimáticas",
        "tl_turnover_title": "Turn-over Celular e Formação de Produto",
        "tl_turnover_note": "Modelos de inativação e acoplamento de produção",
        "tl_yield_title": "Rendimento Variável (Pirt)",
        "tl_yield_note": "Calcule o coeficiente de rendimento real com manutenção",
        "tl_params_title": "Calculadora de Parâmetros Cinéticos",
        "tl_params_note": "Estimativas rápidas sem ajuste de curva",
        "tl_run": "▶ Calcular",
        "tl_vmax": "Velocidade máxima (Vmax)",
        "tl_km": "Constante de Michaelis (Km)",
        "tl_kcat": "Constante catalítica (kcat)",
        "tl_efficiency": "Eficiência catalítica (kcat/Km)",
        "tl_alpha": "Parâmetro associado ao crescimento (α)",
        "tl_beta": "Parâmetro não associado ao crescimento (β)",
        "tl_td": "Tempo de duplicação",
        "tl_gen_per_h": "Gerações por hora",
        "tl_yield_obs": "Rendimento observado (Y_obs)",
        "tl_yield_true": "Rendimento verdadeiro (Y_max)",
        # --- Models Tab ---
        "md_title": "Modelos Cinéticos",
        "md_selected": "✅ Selecionado",
        "md_select_this": "Selecionar",
        "md_deselect": "Remover seleção",
        "md_compare": "🔁 Comparar modelos selecionados",
        "md_selected_count": "modelos selecionados",
        "md_go_results": "📈 Ver Resultados →",
        "md_disabled_tip": "Requer: ",
        "md_back": "← Voltar para Dados",
        "md_points_detection": " Pontos detectados — Tempo, Biomassa e Produto mapeados",
        "rs_compare_title": "Comparação de Modelos",
        "rs_compare_note": "Análise comparativa dos modelos selecionados",
        # --- Results Tab ---
        "rs_title": "Resultados da Análise",
        "rs_opt_title": "Configurações de Otimização",
        "rs_est_mode": "Estimativa de parâmetros",
        "rs_est_auto": "Automático",
        "rs_est_manual": "Manual",
        "rs_tolerance": "Tolerância",
        "rs_maxiter": "Iterações máximas",
        "rs_restarts": "Reinicializações",
        "rs_restart_std": "Padrão (1×)",
        "rs_restart_rob": "Robusto (3×)",
        "rs_restart_glb": "Global (DE)",
        "rs_speed_fast": "Rápido",
        "rs_speed_std": "Padrão",
        "rs_speed_prec": "Preciso",
        "rs_run_btn": "▶ Executar análise",
        "rs_running": "Ajustando modelo…",
        "rs_params_title": "Parâmetros calculados",
        "rs_mu_max": "Velocidade máxima (μmax)",
        "rs_mu_avg": "Velocidade média (μ̄)",
        "rs_td": "Tempo de duplicação (td)",
        "rs_t_total": "Tempo total de análise",
        "rs_x0": "Concentração inicial (X₀)",
        "rs_xm": "Concentração máxima (Xm)",
        "rs_yield_obs": "Rend. observado (Yobs)",
        "rs_yield_teo": "Rend. teórico (Yteo)",
        "rs_metrics_title": "Métricas estatísticas",
        "rs_rmse": "RMSE",
        "rs_r2adj": "R² Ajustado",
        "rs_aic": "AIC",
        "rs_bic": "BIC",
        "rs_fit_chart": "Curva ajustada",
        "rs_resid_chart": "Resíduos",
        "rs_phases_title": "Fases de crescimento",
        "rs_phase_lag": "Fase Lag",
        "rs_phase_acc": "Fase de Aceleração",
        "rs_phase_exp": "Fase Exponencial",
        "rs_phase_dec": "Fase de Desaceleração",
        "rs_phase_stat": "Fase Estacionária",
        "rs_phase_decl": "Fase de Declínio",
        "rs_interval": "Intervalo (h)",
        "rs_points": "Pontos",
        "rs_duration": "Duração (h)",
        "rs_export_pdf": "📄 Exportar PDF",
        "rs_export_xlsx": "📊 Exportar Excel",
        "rs_export_results": " Exportar Resultados",
        "rs_download_excel": " Baixar Excel",
        "rs_error_lib": " Instale fpdf2 para exportar PDF",
        "rs_back": "← Voltar para Modelos",
        "rs_no_model": "Nenhum modelo selecionado. Acesse 🔬 Modelos Cinéticos e escolha um modelo.",
        "rs_no_data": "Nenhum dado disponível. Vá para 📊 Dados primeiro.",
        "rs_sub_plot1":" Resíduos x Tempo",
        "rs_sub_plot2":" Distribuição",
        "rs_error_adjust": " Erro de Ajuste",
        "rs_named_col1": " Parâmetro",
        "rs_named_col2": " Valor",
        # --- About Tab ---
        "ab_title": "Sobre o GrowthEmulator",
        "ab_version": "Versão",
        "ab_desc": (
            "GrowthEmulator é um simulador de crescimento celular microbiano desenvolvido para "
            "análise, ajuste e modelagem de dados cinéticos de culturas microbianas. "
            "Oferece suporte a mais de 30 modelos cinéticos clássicos e contemporâneos, "
            "múltiplos métodos de otimização e ferramentas de visualização interativa."
        ),
        "ab_author": "Desenvolvedor",
        "ab_lic": "Licença",
        "ab_contact_btn": "📧 Entrar em contato",
        # --- Guide Tab ---
        "gd_title": "Guia & Referências",
        "gd_intro_title": "O que é o GrowthEmulator?",
        "gd_intro_body": (
            "GrowthEmulator v1.2 é uma plataforma interativa de modelagem cinética microbiana "
            "construída para pesquisadores, engenheiros de bioprocessos e estudantes. "
            "Permite carregar dados experimentais de crescimento celular, selecionar e ajustar "
            "modelos matemáticos, analisar métricas estatísticas e exportar resultados prontos "
            "para publicação — tudo em uma interface intuitiva e multilíngue."
        ),
        "gd_intro_background1":" Modelos Cinéticos",
        "gd_intro_background2":" 4 Idiomas",
        "gd_intro_background3":" Exportação PDF & Excel",
        "gd_intro_background4":" Acessibilidade",
        "gd_decision_title": "Qual modelo utilizar?",
        "gd_decision_note": "Guia rápido para seleção de modelo baseado no comportamento dos seus dados",
        "gd_refs_title": "Referências bibliográficas",
        "gd_param_title": "Interpretação dos parâmetros",
        "gd_param_note": "Significado e faixas típicas dos principais parâmetros cinéticos",
        "gd_param_descript1":" Intervalo de Análises",
        "gd_param_descript2":" Interpretação",
        "gd_label":"Modelo Recomendado",
        "gd_data":"### 🚀 Fluxo de trabalho recomendado",
        "gd_flux_recommended": """
        1. **📊 Dados** → Faça upload do arquivo ou insira manualmente → Mapeie as variáveis → Visualize a curva  
        2. **🔬 Modelos Cinéticos** → Explore os modelos disponíveis → Selecione o mais adequado para seus dados  
        3. **📈 Resultados** → Configure a otimização → Execute a análise → Interprete os parâmetros e métricas → Exporte  

        > 💡 **Dica:** Use escala logarítmica na prévia dos dados para identificar visualmente a fase exponencial e estimar o λ.  
        > Se R² < 0,95 ou RMSE > 10% do Xm, tente outro modelo ou revise os dados.""", 
        # --- Contact Dialog ---
        "ct_title": "Entre em contato",
        "ct_email": "E-mail",
        "ct_linkedin": "LinkedIn",
        "ct_close": "Fechar",
        "ab_integrators":       "Integradores",
        "dt_rename_category_hint": "Clique para renomear esta categoria",
        # ── Fitness labels ─────────────────────────────────────
        "fit_excellent":       "Excelente",
        "fit_good":            "Bom",
        "fit_acceptable":      "Aceitável",
        "fit_insufficient":    "Insuficiente",
        "fit_quality": "Qualidade do ajuste",
        # ── Comparison chart ───────────────────────────────────
        "rs_individual_results": "Resultados individuais",
        "rs_top_n_label":      "Curvas no gráfico overlay",
        "rs_top_n_omitted":    "modelo(s) omitido(s). Ajuste o slider para ver mais.",
        "rs_params_adjusted":  "Parâmetros ajustados",
        "rs_residuals_normal": "Resíduos normais",
        "rs_residuals_abnormal": "Resíduos não-normais",
        "rs_run_hint":         "Clique em **▶ Executar análise** para ajustar o(s) modelo(s).",
        "rs_tol_per_model": "🔧 Tolerância por modelo (avançado)",
        "rs_tol_hint":         "Defina tolerâncias individuais. Modelos ODE precisam de valores menores (1e-8).",
        "rs_model_series_col": "Modelo / Série",
        # ── Data tab ───────────────────────────────────────────
        "dt_files_identical":  "arquivos com colunas idênticas → empilhados",
        "dt_files_merged":     "arquivos mesclados (outer join no tempo)",
        "dt_col_rename_hint": "✏️ Renomear cabeçalhos — clique no campo e edite:",
        "dt_data_applied": "✅ Dados aplicados! Role para baixo e mapeie os cabeçalhos.",
        "dt_chart_error": "Erro no gráfico:",
        "dt_estimated_mu":  "μ estimado (h⁻¹)",
        "dt_estimated_td":  "td estimado (h)",
        "dt_estimated_x0": "**X₀** inativação (g/L)",
        "dt_estimated_yield": "**S₀** deve ser maior que **Sf**",
        # ── Tools tab (static UI) ──────────────────────────────
        "tl_enzyme_config": "Configurar análise enzimática",
        "tl_data_source_lbl": "Fonte de dados",
        "tl_use_loaded_data": "Usar dados carregados (aba Dados)",
        "tl_enter_manually": "Inserir S e v manualmente",
        "tl_enzyme_conc_lbl": "[E] — Concentração de enzima (µM, opcional)",
        "tl_enzyme_conc_help": "Se fornecido, calcula kcat = Vmax / [E]",
        "tl_enzyme_results":   "Resultados — Cinética Enzimática",
        "tl_ci_result":        "Inibição Competitiva ajustada",
        "tl_pts_extracted":    "pontos de [S] vs v extraídos dos dados",
        "tl_map_sub_time":     "Mapeie Substrato e Tempo na aba Dados para usar esta opção",
        "tl_enter_sv_pairs":   "Insira pares [S] (µM ou mM) e v (µM/s ou mM/h) separados por ';'",
        "tl_format_invalid": "Formato inválido — use números separados por ';'",
        "tl_mm_run_btn":       "▶ Calcular — Michaelis-Menten",
        "tl_lp_config":        "Configurar análise de formação de produto",
        "tl_chick_section": "Chick — Inativação Celular de 1ª Ordem",
        "tl_min_pts":          "pontos — mínimo 4 necessário",
        "tl_lp_run_btn": "▶ Ajustar Luedeking-Piret",
        "tl_chick_run_btn": "▶ Ajustar Chick (kd)",
        "tl_prod_primary": "Produto primário (crescimento-associado)",
        "tl_prod_secondary":   "Produto secundário (não-associado ao crescimento)",
        "tl_prod_mixed":       "Produto misto (α e β ambos significativos)",
        "tl_classification":   "Classificação:",
        "tl_lp_results":       "Resultados — Luedeking-Piret",
        "tl_ck_results":       "Resultados — Chick (Inativação)",
        "tl_halflife": "t½ (tempo de meia-vida)",
        "tl_pirt_config":      "Configurar análise de rendimento",
        "tl_pirt_run_btn": "▶ Ajustar Pirt — Rendimento Variável",
        "tl_pirt_results":     "Resultados — Pirt (Rendimento Variável)",
        "t1_pirt_mark": "Insira pares **μ (h⁻¹)** e **Y_obs (g/g)** para ajustar o modelo de Pirt:",
        "tl_maintenance_coef": "ms — Manutenção",
        "tl_calc_quick":       "Calculadora rápida",
        "tl_td_gen_section":   "⏱ Tempo de duplicação e taxa de geração",
        "tl_mu_from_pts":      "📈 Estimativa de μ a partir de dois pontos",
        "tl_yield_coef_sect":  "⚗️ Coeficiente de rendimento observado",
        "tl_chick_const_sect": "📐 Constante de inativação (Chick)",
        "tl_s_values_lbl": "Valores de [S]:",
        "tl_v_values_lbl": "Valores de v:",
        # ── About tab ─────────────────────────────────────────
        "ab_stack":            "Stack tecnológico",
        "ab_algorithms":       "Algoritmos",
        "tl_subtitle": "Modelos especializados de cinética enzimática, rendimento e turn-over celular",
        # ── Table (metric) ─────────────────────────────────────────
        "tm_metric_col": " Colunas",
        "tm_metric_line": " Linhas",
        "tm_metric_model": " Modelos disponíveis",
    },
    "en": {
        "app_name": "GrowthEmulator v1.2",
        "nav_data": "📊 Data",
        "nav_models": "🔬 Kinetic Models",
        "nav_tools": "🔧 Tools",
        "nav_results": "📈 Results",
        "nav_about": "ℹ️ About",
        "nav_guide": "📖 Guide & References",
        "nav_contact": "📧 Contact",
        "dark_mode": "🌙 Dark",
        "light_mode": "☀️ Light",
        "contrast": "♿ Contrast",
        "lang_label": "Language",
        "dt_title": "Data Input",
        "dt_upload_label": "Upload data file",
        "dt_upload_types": "Supports .csv, .xlsx and .txt (BR and EN delimiters auto-detected)",
        "dt_header_row": "Header row",
        "dt_header_note": "💡 Enter the row number containing column names (1 = first row)",
        "dt_mapping_title": "Variable mapping",
        "dt_mapping_note": "Associate each column with a simulator variable",
        "dt_none": "— Not mapped —",
        "dt_time_lbl": "Time (h)",
        "dt_biomass_lbl": "Concentration (g/L, OD)",
        "dt_substrate_lbl": "Substrate (g/L)",
        "dt_ph_lbl": "pH",
        "dt_product_lbl": "Product (g/L)",
        "dt_drymass_lbl": "Dry mass (g)",
        "dt_tool": "Map **Time**, **Biomass**, and **Product** in the 📊 Data tab to enable.",
        "dt_info": "Map **Time and **Biomass** to fit the Chick model.",
        "dt_manual_title": "Manual data entry",
        "dt_manual_note": "Use Ctrl+V to paste data from a spreadsheet",
        "dt_manual_add_col": "＋ Column",
        "dt_manual_rem_col": "－ Column",
        "dt_manual_add_row": "＋ Row",
        "dt_manual_rem_row": "－ Row",
        "dt_manual_clear": "🗑 Clear table",
        "dt_files_merge": " merged files (outer join on time axis)",
        "dt_manual_add_data": " Apply manual data",
        "dt_summary_title": "Data summary",
        "dt_nan_warn": "⚠️ Rows with missing values will be excluded from analyses.",
        "dt_neg_time": "🚫 Negative time values detected!",
        "dt_dup_time": "⚠️ Duplicate time values detected!",
        "dt_preview_title": "Preview — Scatter Plot",
        "dt_preview_log": "Y-axis in logarithmic scale",
        "dt_avail_title": "Available models",
        "dt_avail_note": "with current headers",
        "dt_avail_complement0": " Kinetic models",
        "dt_avail_complement1":"Tools",
        "dt_avail_prep":"of",
        "dt_select_btn": "🔬 Select model for analysis →",
        "dt_select_disabled": "Map at least **Time** and **Biomass** to enable",
        "dt_excl_cols": "Excluded columns",
        "dt_excl_note": "Select columns to remove from active data",
        "dt_multi_file": "Multiple files loaded — combined by row position",
        "tl_title": "Tools",
        "tl_enzyme_title": "Enzyme Kinetics (Michaelis-Menten)",
        "tl_enzyme_note": "Analyse velocity-substrate relationship for enzymatic reactions",
        "tl_turnover_title": "Cell Turnover & Product Formation",
        "tl_turnover_note": "Inactivation and production coupling models",
        "tl_yield_title": "Variable Yield (Pirt)",
        "tl_yield_note": "Calculate true yield coefficient with maintenance",
        "tl_params_title": "Kinetic Parameter Calculator",
        "tl_params_note": "Quick estimates without curve fitting",
        "tl_run": "▶ Calculate",
        "tl_vmax": "Maximum velocity (Vmax)",
        "tl_km": "Michaelis constant (Km)",
        "tl_kcat": "Catalytic constant (kcat)",
        "tl_efficiency": "Catalytic efficiency (kcat/Km)",
        "tl_alpha": "Growth-associated parameter (α)",
        "tl_beta": "Non-growth-associated parameter (β)",
        "tl_td": "Doubling time",
        "tl_gen_per_h": "Generations per hour",
        "tl_yield_obs": "Observed yield (Y_obs)",
        "tl_yield_true": "True yield (Y_max)",
        "md_title": "Kinetic Models",
        "md_selected": "✅ Selected",
        "md_select_this": "Select",
        "md_deselect": "Deselect",
        "md_compare": "🔁 Compare selected models",
        "md_selected_count": "models selected",
        "md_go_results": "📈 View Results →",
        "md_disabled_tip": "Requires: ",
        "md_back": "← Back to Data",
        "md_points_detection": " Points detected — Time, Biomass and Product mapped",
        "rs_compare_title": "Model Comparison",
        "rs_compare_note": "Comparative analysis of selected models",
        "rs_title": "Analysis Results",
        "rs_opt_title": "Optimization Settings",
        "rs_est_mode": "Parameter estimation",
        "rs_est_auto": "Automatic",
        "rs_est_manual": "Manual",
        "rs_tolerance": "Tolerance",
        "rs_maxiter": "Max iterations",
        "rs_restarts": "Restarts",
        "rs_restart_std": "Standard (1×)",
        "rs_restart_rob": "Robust (3×)",
        "rs_restart_glb": "Global (DE)",
        "rs_speed_fast": "Fast",
        "rs_speed_std": "Standard",
        "rs_speed_prec": "Precise",
        "rs_run_btn": "▶ Run analysis",
        "rs_running": "Fitting model…",
        "rs_params_title": "Computed parameters",
        "rs_mu_max": "Maximum rate (μmax)",
        "rs_mu_avg": "Average rate (μ̄)",
        "rs_td": "Doubling time (td)",
        "rs_t_total": "Total analysis time",
        "rs_x0": "Initial concentration (X₀)",
        "rs_xm": "Maximum concentration (Xm)",
        "rs_yield_obs": "Observed yield (Yobs)",
        "rs_yield_teo": "Theoretical yield (Yteo)",
        "rs_metrics_title": "Statistical metrics",
        "rs_rmse": "RMSE",
        "rs_r2adj": "Adjusted R²",
        "rs_aic": "AIC",
        "rs_bic": "BIC",
        "rs_fit_chart": "Fitted curve",
        "rs_resid_chart": "Residuals",
        "rs_phases_title": "Growth phases",
        "rs_phase_lag": "Lag phase",
        "rs_phase_acc": "Acceleration phase",
        "rs_phase_exp": "Exponential phase",
        "rs_phase_dec": "Deceleration phase",
        "rs_phase_stat": "Stationary phase",
        "rs_phase_decl": "Decline phase",
        "rs_interval": "Interval (h)",
        "rs_points": "Points",
        "rs_duration": "Duration (h)",
        "rs_export_pdf": "📄 Export PDF",
        "rs_export_xlsx": "📊 Export Excel",
        "rs_export_results": " Export Results",
        "rs_download_excel": " Download Excel",
        "rs_error_lib": " Install fpdf2 to export PDF",
        "rs_back": " ← Back to Models",
        "rs_no_model": "No model selected. Go to 🔬 Kinetic Models and choose one.",
        "rs_no_data": "No data available. Go to 📊 Data first.",
        "rs_sub_plot1":" Waste × Time",
        "rs_sub_plot2":" Distribution",
        "rs_error_adjust": " Adjustment error",
        "rs_named_col1": " Parameter",
        "rs_named_col2": " Value",
        "ab_title": "About GrowthEmulator",
        "ab_version": "Version",
        "ab_desc": (
            "GrowthEmulator is a microbial cell growth simulator developed for the analysis, "
            "fitting and modeling of microbial culture kinetic data. It supports 30+ classic "
            "and contemporary kinetic models, multiple optimization methods and interactive "
            "visualization tools."
        ),
        "ab_author": "Developer",
        "ab_lic": "License",
        "ab_contact_btn": "📧 Get in touch",
        "gd_title": "Guide & References",
        "gd_intro_title": "What is GrowthEmulator?",
        "gd_intro_body": (
            "GrowthEmulator v1.2 is an interactive microbial kinetic modeling platform built for "
            "researchers, bioprocess engineers and students. Load experimental growth data, "
            "select and fit mathematical models, analyze statistical metrics and export "
            "publication-ready results — all in an intuitive multilingual interface."
        ),
        "gd_intro_background1":" Kinetic Models",
        "gd_intro_background2":" 4 Languages",
        "gd_intro_background3":" PDF & Excel Export",
        "gd_intro_background4":" Accessibility",
        "gd_decision_title": "Which model should I use?",
        "gd_decision_note": "Quick selection guide based on your data's behavior",
        "gd_refs_title": "Bibliography",
        "gd_param_title": "Parameter interpretation",
        "gd_param_note": "Meaning and typical ranges of the main kinetic parameters",
        "gd_param_descript1":" Analysis interval",
        "gd_param_descript2":" Interpretation",
        "gd_data":"### 🚀 Recommended Workflow",
        "gd_flux_recommended": """
        1. **📊 Data** → Upload file or enter manually → Map variables → Visualize curve  
        2. **🔬 Kinetic Models** → Explore available models → Select the most suitable for your data  
        3. **📈 Results** → Configure optimization → Run analysis → Interpret parameters and metrics → Export  

        > 💡 **Tip:** Use a logarithmic scale in the data preview to visually identify the exponential phase and estimate λ.  
        > If R² < 0.95 or RMSE > 10% of Xm, try another model or review the data.""",
        "gd_label":"Recommended model",
        "ct_title": "Get in touch",
        "ct_email": "E-mail",
        "ct_linkedin": "LinkedIn",
        "ct_close": "Close",
        "ab_integrators":       "Integrators",
        "dt_rename_category_hint": "Click to rename this category",
        "fit_excellent":       "Excellent",
        "fit_good":            "Good",
        "fit_acceptable":      "Acceptable",
        "fit_insufficient":    "Insufficient",
        "fit_quality":         "Fit quality",
        "rs_individual_results": "Individual results",
        "rs_top_n_label":      "Curves in overlay chart",
        "rs_top_n_omitted":    "model(s) omitted. Adjust slider to see more.",
        "rs_params_adjusted":  "Adjusted parameters",
        "rs_residuals_normal": "Normal residuals",
        "rs_residuals_abnormal": "Non-normal residuals",
        "rs_run_hint":         "Click **▶ Run analysis** to fit the model(s).",
        "rs_tol_per_model":    "🔧 Tolerance per model (advanced)",
        "rs_tol_hint":         "Set individual tolerances. ODE models usually need lower values (1e-8).",
        "rs_model_series_col": "Model / Series",
        "dt_files_identical":  "files with identical columns → stacked",
        "dt_files_merged":     "files merged (outer join on time)",
        "dt_col_rename_hint":  "✏️ Rename headers — click a field to edit:",
        "dt_data_applied":     "✅ Data applied! Scroll down and map the headers.",
        "dt_chart_error":      "Chart error:",
        "dt_estimated_mu":     "Estimated μ (h⁻¹)",
        "dt_estimated_td":     "Estimated td (h)",
        "dt_estimated_x0": "**X₀** inactivation (g/L)",
        "dt_estimated_yield": "**S₀** must be greater than **Sf**",
        "tl_enzyme_config":    "Configure enzyme kinetics analysis",
        "tl_data_source_lbl":  "Data source",
        "tl_use_loaded_data":  "Use loaded data (Data tab)",
        "tl_enter_manually":   "Enter S and v manually",
        "tl_enzyme_conc_lbl":  "[E] — Enzyme concentration (µM, optional)",
        "tl_enzyme_conc_help": "If provided, calculates kcat = Vmax / [E]",
        "tl_enzyme_results":   "Results — Enzyme Kinetics",
        "tl_ci_result":        "Competitive Inhibition fitted",
        "tl_pts_extracted":    "[S] vs v datapoints extracted from data",
        "tl_map_sub_time":     "Map Substrate and Time in the Data tab to use this option",
        "tl_enter_sv_pairs":   "Enter [S] (µM or mM) and v (µM/s or mM/h) pairs separated by ';'",
        "tl_format_invalid":   "Invalid format — use numbers separated by ';'",
        "tl_mm_run_btn":       "▶ Calculate — Michaelis-Menten",
        "tl_lp_config":        "Configure product formation analysis",
        "tl_chick_section":    "Chick — First-Order Cell Inactivation",
        "tl_min_pts":          "datapoints — minimum 4 required",
        "tl_lp_run_btn":       "▶ Fit Luedeking-Piret",
        "tl_chick_run_btn":    "▶ Fit Chick (kd)",
        "tl_prod_primary":     "Primary product (growth-associated)",
        "tl_prod_secondary":   "Secondary product (non-growth-associated)",
        "tl_prod_mixed":       "Mixed product (α and β both significant)",
        "tl_classification":   "Classification:",
        "tl_lp_results":       "Results — Luedeking-Piret",
        "tl_ck_results":       "Results — Chick (Inactivation)",
        "tl_halflife":         "t½ (half-life)",
        "tl_pirt_config":      "Configure yield analysis",
        "tl_pirt_run_btn":     "▶ Fit Pirt — Variable Yield",
        "tl_pirt_results":     "Results — Pirt (Variable Yield)",
        "t1_pirt_mark": "Enter **μ (h⁻¹)** and **Y_obs (g/g)** pairs to fit the Pirt model:",
        "tl_maintenance_coef": "ms — Maintenance",
        "tl_calc_quick":       "Quick calculator",
        "tl_td_gen_section":   "⏱ Doubling time and generation rate",
        "tl_mu_from_pts":      "📈 Estimate μ from two data points",
        "tl_yield_coef_sect":  "⚗️ Observed yield coefficient",
        "tl_chick_const_sect": "📐 Inactivation constant (Chick)",
        "tl_s_values_lbl":     "[S] values:",
        "tl_v_values_lbl":     "v values:",
        "ab_stack":            "Tech stack",
        "ab_algorithms":       "Algorithms",
        "tl_subtitle": "Specialized models for enzyme kinetics, yield and cell turnover",
        "tm_metric_col": " Columns",
        "tm_metric_line": " Lines",
        "tm_metric_model": " Available Models",

    },
    "es": {
        "app_name": "GrowthEmulator v1.2",
        "nav_data": "📊 Datos",
        "nav_models": "🔬 Modelos Cinéticos",
        "nav_tools": "🔧 Herramientas",
        "nav_results": "📈 Resultados",
        "nav_about": "ℹ️ Acerca de",
        "nav_guide": "📖 Guía & Referencias",
        "nav_contact": "📧 Contacto",
        "dark_mode": "🌙 Oscuro",
        "light_mode": "☀️ Claro",
        "contrast": "♿ Contraste",
        "lang_label": "Idioma",
        "dt_title": "Entrada de Datos",
        "dt_upload_label": "Subir archivo de datos",
        "dt_upload_types": "Compatible con .csv, .xlsx y .txt (delimitadores BR y EN detectados automáticamente)",
        "dt_header_row": "Fila de encabezado",
        "dt_header_note": "💡 Indique la fila donde están los nombres de columnas (1 = primera fila)",
        "dt_mapping_title": "Mapeo de variables",
        "dt_mapping_note": "Asocie cada columna del archivo con las variables del simulador",
        "dt_none": "— No mapeado —",
        "dt_time_lbl": "Tiempo (h)",
        "dt_biomass_lbl": "Concentración (g/L, OD)",
        "dt_substrate_lbl": "Sustrato (g/L)",
        "dt_ph_lbl": "pH",
        "dt_product_lbl": "Producto (g/L)",
        "dt_drymass_lbl": "Masa seca (g)",
        "dt_tool": "Mapee **Tiempo**, **Biomasa** y **Producto** en la pestaña 📊 Datos para habilitar.",
        "dt_info": "Mapee **Tiempo** y Biomasa** para ajustar el modelo de Chick.",
        "dt_manual_title": "Entrada manual de datos",
        "dt_manual_note": "Use Ctrl+V para pegar datos desde una hoja de cálculo",
        "dt_manual_add_col": "＋ Columna",
        "dt_manual_rem_col": "－ Columna",
        "dt_manual_add_row": "＋ Fila",
        "dt_manual_rem_row": "－ Fila",
        "dt_manual_clear": "🗑 Limpiar tabla",
        "dt_files_merge": " archivos fusionados (unión externa en el eje de tiempo)",
        "dt_manual_add_data": " Aplicar datos manuales",
        "dt_summary_title": "Resumen de datos",
        "dt_nan_warn": "⚠️ Las filas con valores faltantes serán excluidas del análisis.",
        "dt_neg_time": "🚫 ¡Valores negativos de tiempo detectados!",
        "dt_dup_time": "⚠️ ¡Valores de tiempo duplicados detectados!",
        "dt_preview_title": "Vista previa — Diagrama de dispersión",
        "dt_preview_log": "Eje Y en escala logarítmica",
        "dt_avail_title": "Modelos disponibles",
        "dt_avail_note": "con los encabezados actuales",
        "dt_avail_complement0": " Modelos Cinéticos",
        "dt_avail_complement1":"Herramientas",
        "dt_avail_prep":"de",
        "dt_select_btn": "🔬 Seleccionar modelo para análisis →",
        "dt_select_disabled": "Mapee al menos **Tiempo** y **Biomasa** para habilitar",
        "dt_excl_cols": "Columnas excluidas",
        "dt_excl_note": "Seleccione las columnas a eliminar de los datos activos",
        "dt_multi_file": "Varios archivos cargados — combinados por posición de filas",
        "tl_title": "Herramientas",
        "tl_enzyme_title": "Cinética Enzimática (Michaelis-Menten)",
        "tl_enzyme_note": "Analice la relación velocidad-sustrato para reacciones enzimáticas",
        "tl_turnover_title": "Recambio Celular y Formación de Producto",
        "tl_turnover_note": "Modelos de inactivación y acoplamiento de producción",
        "tl_yield_title": "Rendimiento Variable (Pirt)",
        "tl_yield_note": "Calcule el coeficiente de rendimiento real con mantenimiento",
        "tl_params_title": "Calculadora de Parámetros Cinéticos",
        "tl_params_note": "Estimaciones rápidas sin ajuste de curva",
        "tl_run": "▶ Calcular",
        "tl_vmax": "Velocidad máxima (Vmax)",
        "tl_km": "Constante de Michaelis (Km)",
        "tl_kcat": "Constante catalítica (kcat)",
        "tl_efficiency": "Eficiencia catalítica (kcat/Km)",
        "tl_alpha": "Parámetro asociado al crecimiento (α)",
        "tl_beta": "Parámetro no asociado al crecimiento (β)",
        "tl_td": "Tiempo de duplicación",
        "tl_gen_per_h": "Generaciones por hora",
        "tl_yield_obs": "Rendimiento observado (Y_obs)",
        "tl_yield_true": "Rendimiento verdadero (Y_max)",
        "md_title": "Modelos Cinéticos",
        "md_selected": "✅ Seleccionado",
        "md_select_this": "Seleccionar",
        "md_deselect": "Deseleccionar",
        "md_compare": "🔁 Comparar modelos seleccionados",
        "md_selected_count": "modelos seleccionados",
        "md_go_results": "📈 Ver Resultados →",
        "md_disabled_tip": "Requiere: ",
        "md_back": "← Volver a Datos",
        "md_points_detection": " Puntos detectados: tiempo, biomasa y producto mapeados",
        "rs_compare_title": "Comparación de Modelos",
        "rs_compare_note": "Análisis comparativo de los modelos seleccionados",
        "rs_title": "Resultados del Análisis",
        "rs_opt_title": "Configuración de Optimización",
        "rs_est_mode": "Estimación de parámetros",
        "rs_est_auto": "Automático",
        "rs_est_manual": "Manual",
        "rs_tolerance": "Tolerancia",
        "rs_maxiter": "Iteraciones máximas",
        "rs_restarts": "Reinicios",
        "rs_restart_std": "Estándar (1×)",
        "rs_restart_rob": "Robusto (3×)",
        "rs_restart_glb": "Global (DE)",
        "rs_speed_fast": "Rápido",
        "rs_speed_std": "Estándar",
        "rs_speed_prec": "Preciso",
        "rs_run_btn": "▶ Ejecutar análisis",
        "rs_running": "Ajustando modelo…",
        "rs_params_title": "Parámetros calculados",
        "rs_mu_max": "Velocidad máxima (μmax)",
        "rs_mu_avg": "Velocidad media (μ̄)",
        "rs_td": "Tiempo de duplicación (td)",
        "rs_t_total": "Tiempo total de análisis",
        "rs_x0": "Concentración inicial (X₀)",
        "rs_xm": "Concentración máxima (Xm)",
        "rs_yield_obs": "Rendimiento observado (Yobs)",
        "rs_yield_teo": "Rendimiento teórico (Yteo)",
        "rs_metrics_title": "Métricas estadísticas",
        "rs_rmse": "RMSE",
        "rs_r2adj": "R² Ajustado",
        "rs_aic": "AIC",
        "rs_bic": "BIC",
        "rs_fit_chart": "Curva ajustada",
        "rs_resid_chart": "Residuales",
        "rs_phases_title": "Fases de crecimiento",
        "rs_phase_lag": "Fase Lag",
        "rs_phase_acc": "Fase de Aceleración",
        "rs_phase_exp": "Fase Exponencial",
        "rs_phase_dec": "Fase de Desaceleración",
        "rs_phase_stat": "Fase Estacionaria",
        "rs_phase_decl": "Fase de Declive",
        "rs_interval": "Intervalo (h)",
        "rs_points": "Puntos",
        "rs_duration": "Duración (h)",
        "rs_export_pdf": "📄 Exportar PDF",
        "rs_export_xlsx": "📊 Exportar Excel",
        "rs_export_results": " Exportar resultados",
        "rs_download_excel": " Descargar Excel",
        "rs_error_lib": " Instale fpdf2 para exportar PDF",
        "rs_back": "← Volver a Modelos",
        "rs_no_model": "Ningún modelo seleccionado. Ve a 🔬 Modelos Cinéticos y elige uno.",
        "rs_no_data": "Sin datos disponibles. Ve a 📊 Datos primero.",
        "rs_sub_plot1":" Desperdicio × Tiempo",
        "rs_sub_plot2":" Distribución",
        "rs_error_adjust": " Error de ajuste",
        "rs_named_col1": " Parámetro",
        "rs_named_col2": " Valor",
        "ab_title": "Acerca de GrowthEmulator",
        "ab_version": "Versión",
        "ab_desc": (
            "GrowthEmulator es un simulador de crecimiento celular microbiano para análisis, "
            "ajuste y modelado de datos cinéticos. Soporta más de 30 modelos clásicos y "
            "contemporáneos, múltiples métodos de optimización y visualización interactiva."
        ),
        "ab_author": "Desarrollador",
        "ab_lic": "Licencia",
        "ab_contact_btn": "📧 Ponerse en contacto",
        "gd_title": "Guía & Referencias",
        "gd_intro_title": "¿Qué es GrowthEmulator?",
        "gd_intro_body": (
            "GrowthEmulator v1.2 es una plataforma interactiva de modelado cinético microbiano "
            "para investigadores, ingenieros de bioprocesos y estudiantes. Cargue datos "
            "experimentales, seleccione y ajuste modelos matemáticos, analice métricas "
            "estadísticas y exporte resultados listos para publicación."
        ),
        "gd_intro_background1":" Modelos Cinéticos",
        "gd_intro_background2":" 4 Idiomas",
        "gd_intro_background3":" Exportación PDF & Excel",
        "gd_intro_background4":" Accesibilidad",
        "gd_decision_title": "¿Qué modelo debo usar?",
        "gd_decision_note": "Guía rápida de selección basada en el comportamiento de sus datos",
        "gd_refs_title": "Bibliografía",
        "gd_param_title": "Interpretación de parámetros",
        "gd_param_note": "Significado y rangos típicos de los principales parámetros cinéticos",
        "gd_param_descript1":" Intervalo de análisis",
        "gd_param_descript2":" Interpretación",
        "gd_data":"### 🚀 Flujo de trabajo recomendado",
        "gd_flux_recommended": """
        1. **📊 Datos** → Suba el archivo o introdúzcalo manualmente → Mapée las variables → Visualice la curva  
        2. **🔬 Modelos Cinéticos** → Explore los modelos disponibles → Seleccione el más adecuado para sus datos  
        3. **📈 Resultados** → Configure la optimización → Ejecute el análisis → Interprete los parámetros y métricas → Exporte  

        > 💡 **Consejo:** Utilice la escala logarítmica en la vista previa de los datos para identificar visualmente la fase exponencial y estimar el λ.  
        > Si R² < 0,95 o RMSE > 10% de Xm, intente con otro modelo o revise los datos.""",
        "gd_label":"Modelo recomendado",
        "ct_title": "Ponerse en contacto",
        "ct_email": "Correo electrónico",
        "ct_linkedin": "LinkedIn",
        "ct_close": "Cerrar",
        "ab_integrators":       "Integradores",
        "dt_rename_category_hint": "Haga clic para renombrar esta categoría",
        "fit_excellent":       "Excelente",
        "fit_good":            "Bueno",
        "fit_acceptable":      "Aceptable",
        "fit_insufficient":    "Insuficiente",
        "fit_quality":         "Calidad del ajuste",
        "rs_individual_results": "Resultados individuales",
        "rs_top_n_label":      "Curvas en gráfico superpuesto",
        "rs_top_n_omitted":    "modelo(s) omitido(s). Ajuste el control para ver más.",
        "rs_params_adjusted":  "Parámetros ajustados",
        "rs_residuals_normal": "Residuales normales",
        "rs_residuals_abnormal": "Residuales no normales",
        "rs_run_hint":         "Haga clic en **▶ Ejecutar análisis** para ajustar el/los modelo(s).",
        "rs_tol_per_model":    "🔧 Tolerancia por modelo (avanzado)",
        "rs_tol_hint":         "Tolerancias individuales. Los modelos ODE suelen necesitar valores menores (1e-8).",
        "rs_model_series_col": "Modelo / Serie",
        "dt_files_identical":  "archivos con columnas idénticas → apilados",
        "dt_files_merged":     "archivos fusionados (outer join en el tiempo)",
        "dt_col_rename_hint":  "✏️ Renombrar encabezados — haga clic en el campo para editar:",
        "dt_data_applied":     "✅ ¡Datos aplicados! Desplácese hacia abajo y mapee los encabezados.",
        "dt_chart_error":      "Error en el gráfico:",
        "dt_estimated_mu":     "μ estimado (h⁻¹)",
        "dt_estimated_td":     "td estimado (h)",
        "dt_estimated_x0": "**X₀** inactivación (g/L)",
        "dt_estimated_yield": "**S₀** debe ser mayor que **Sf**",
        "tl_enzyme_config":    "Configurar análisis enzimático",
        "tl_data_source_lbl":  "Fuente de datos",
        "tl_use_loaded_data":  "Usar datos cargados (pestaña Datos)",
        "tl_enter_manually":   "Ingresar S y v manualmente",
        "tl_enzyme_conc_lbl":  "[E] — Concentración de enzima (µM, opcional)",
        "tl_enzyme_conc_help": "Si se proporciona, calcula kcat = Vmax / [E]",
        "tl_enzyme_results":   "Resultados — Cinética Enzimática",
        "tl_ci_result":        "Inhibición Competitiva ajustada",
        "tl_pts_extracted":    "puntos [S] vs v extraídos de los datos",
        "tl_map_sub_time":     "Mapee Sustrato y Tiempo en la pestaña Datos para usar esta opción",
        "tl_enter_sv_pairs":   "Ingrese pares [S] (µM o mM) y v (µM/s o mM/h) separados por ';'",
        "tl_format_invalid":   "Formato inválido — use números separados por ';'",
        "tl_mm_run_btn":       "▶ Calcular — Michaelis-Menten",
        "tl_lp_config":        "Configurar análisis de formación de producto",
        "tl_chick_section":    "Chick — Inactivación Celular de 1er Orden",
        "tl_min_pts":          "puntos — mínimo 4 requerido",
        "tl_lp_run_btn":       "▶ Ajustar Luedeking-Piret",
        "tl_chick_run_btn":    "▶ Ajustar Chick (kd)",
        "tl_prod_primary":     "Producto primario (asociado al crecimiento)",
        "tl_prod_secondary":   "Producto secundario (no asociado al crecimiento)",
        "tl_prod_mixed":       "Producto mixto (α y β ambos significativos)",
        "tl_classification":   "Clasificación:",
        "tl_lp_results":       "Resultados — Luedeking-Piret",
        "tl_ck_results":       "Resultados — Chick (Inactivación)",
        "tl_halflife":         "t½ (vida media)",
        "tl_pirt_config":      "Configurar análisis de rendimiento",
        "tl_pirt_run_btn":     "▶ Ajustar Pirt — Rendimiento Variable",
        "tl_pirt_results":     "Resultados — Pirt (Rendimiento Variable)",
        "t1_pirt_mark": "Ingrese pares de **μ (h⁻¹)** y **Y_obs (g/g)** para ajustar el modelo de Pirt:",
        "tl_maintenance_coef": "ms — Mantenimiento",
        "tl_calc_quick":       "Calculadora rápida",
        "tl_td_gen_section":   "⏱ Tiempo de duplicación y tasa de generación",
        "tl_mu_from_pts":      "📈 Estimación de μ a partir de dos puntos",
        "tl_yield_coef_sect":  "⚗️ Coeficiente de rendimiento observado",
        "tl_chick_const_sect": "📐 Constante de inactivación (Chick)",
        "tl_s_values_lbl":     "Valores de [S]:",
        "tl_v_values_lbl":     "Valores de v:",
        "ab_stack":            "Stack tecnológico",
        "ab_algorithms":       "Algoritmos",
        "tl_subtitle": "Modelos especializados de cinética enzimática, rendimiento y recambio celular",
        "tm_metric_col": " Columnas",
        "tm_metric_line": " Pauta",
        "tm_metric_model": " Modelos disponibles",

    },
    "zh": {
        "app_name": "GrowthEmulator v1.2",
        "nav_data": "📊 数据",
        "nav_models": "🔬 动力学模型",
        "nav_tools": "🔧 工具",
        "nav_results": "📈 结果",
        "nav_about": "ℹ️ 关于",
        "nav_guide": "📖 指南与参考文献",
        "nav_contact": "📧 联系我们",
        "dark_mode": "🌙 深色",
        "light_mode": "☀️ 浅色",
        "contrast": "♿ 对比度",
        "lang_label": "语言",
        "dt_title": "数据输入",
        "dt_upload_label": "上传数据文件",
        "dt_upload_types": "支持 .csv, .xlsx 和 .txt（自动识别巴西和国际分隔符）",
        "dt_header_row": "标题行",
        "dt_header_note": "💡 输入列名所在行号（1 = 第一行）",
        "dt_mapping_title": "变量映射",
        "dt_mapping_note": "将文件中的每列与模拟器变量对应",
        "dt_none": "— 未映射 —",
        "dt_time_lbl": "时间（h）",
        "dt_biomass_lbl": "生物量 / 浓度（g/L, OD）",
        "dt_substrate_lbl": "底物（g/L）",
        "dt_ph_lbl": "pH",
        "dt_product_lbl": "产物（g/L）",
        "dt_drymass_lbl": "干重（g）",
        "dt_tool": "请在 📊 数据 选项卡中映射时间、生物质和产物以启用",
        "dt_info": "映射时间和生物质以拟合 Chick 模型",
        "dt_manual_title": "手动输入数据",
        "dt_manual_note": "使用 Ctrl+V 从电子表格粘贴数据",
        "dt_manual_add_col": "＋ 列",
        "dt_manual_rem_col": "－ 列",
        "dt_manual_add_row": "＋ 行",
        "dt_manual_rem_row": "－ 行",
        "dt_manual_clear": "🗑 清除表格",
        "dt_files_merge": " 合并文件（时间轴上的外连接)",
        "dt_manual_add_data": " 应用手动数据",
        "dt_summary_title": "数据摘要",
        "dt_nan_warn": "⚠️ 含缺失值的行将在分析中被排除。",
        "dt_neg_time": "🚫 检测到负时间值！",
        "dt_dup_time": "⚠️ 检测到重复时间值！",
        "dt_preview_title": "预览 — 散点图",
        "dt_preview_log": "Y 轴使用对数刻度",
        "dt_avail_title": "可用模型",
        "dt_avail_note": "基于当前标题",
        "dt_avail_complement0": " 动力学模型",
        "dt_avail_complement1":"工具",
        "dt_avail_prep":"的",
        "dt_select_btn": "🔬 选择分析模型 →",
        "dt_select_disabled": "请至少映射**时间**和**生物量**以启用",
        "dt_excl_cols": "已排除的列",
        "dt_excl_note": "选择要从活动数据中删除的列",
        "dt_multi_file": "已加载多个文件 — 按行位置合并",
        "tl_title": "工具",
        "tl_enzyme_title": "酶动力学 (Michaelis-Menten)",
        "tl_enzyme_note": "分析酶促反应的速率-底物关系",
        "tl_turnover_title": "细胞更新与产物生成",
        "tl_turnover_note": "失活和产物偶联模型",
        "tl_yield_title": "可变产率 (Pirt)",
        "tl_yield_note": "计算含维持能的真实产率系数",
        "tl_params_title": "动力学参数计算器",
        "tl_params_note": "无需曲线拟合的快速估算",
        "tl_run": "▶ 计算",
        "tl_vmax": "最大速率 (Vmax)",
        "tl_km": "Michaelis 常数 (Km)",
        "tl_kcat": "催化常数 (kcat)",
        "tl_efficiency": "催化效率 (kcat/Km)",
        "tl_alpha": "生长相关参数 (α)",
        "tl_beta": "非生长相关参数 (β)",
        "tl_td": "倍增时间",
        "tl_gen_per_h": "每小时代数",
        "tl_yield_obs": "观测产率 (Y_obs)",
        "tl_yield_true": "真实产率 (Y_max)",
        "md_title": "动力学模型",
        "md_selected": "✅ 已选择",
        "md_select_this": "选择",
        "md_deselect": "取消选择",
        "md_compare": "🔁 比较所选模型",
        "md_selected_count": "个模型已选择",
        "md_go_results": "📈 查看结果 →",
        "md_disabled_tip": "需要：",
        "md_back": "← 返回数据",
        "md_points_detection": " 检测到的点 - 时间、生物量和产品映射",
        "rs_compare_title": "模型比较",
        "rs_compare_note": "所选模型的比较分析",
        "rs_title": "分析结果",
        "rs_opt_title": "优化设置",
        "rs_est_mode": "参数估算",
        "rs_est_auto": "自动",
        "rs_est_manual": "手动",
        "rs_tolerance": "容差",
        "rs_maxiter": "最大迭代次数",
        "rs_restarts": "重启次数",
        "rs_restart_std": "标准 (1×)",
        "rs_restart_rob": "稳健 (3×)",
        "rs_restart_glb": "全局 (DE)",
        "rs_speed_fast": "快速",
        "rs_speed_std": "标准",
        "rs_speed_prec": "精确",
        "rs_run_btn": "▶ 运行分析",
        "rs_running": "拟合模型中…",
        "rs_params_title": "计算参数",
        "rs_mu_max": "最大速率 (μmax)",
        "rs_mu_avg": "平均速率 (μ̄)",
        "rs_td": "倍增时间 (td)",
        "rs_t_total": "总分析时间",
        "rs_x0": "初始浓度 (X₀)",
        "rs_xm": "最大浓度 (Xm)",
        "rs_yield_obs": "观测产量系数 (Yobs)",
        "rs_yield_teo": "理论产量系数 (Yteo)",
        "rs_metrics_title": "统计指标",
        "rs_rmse": "RMSE",
        "rs_r2adj": "调整 R²",
        "rs_aic": "AIC",
        "rs_bic": "BIC",
        "rs_fit_chart": "拟合曲线",
        "rs_resid_chart": "残差",
        "rs_phases_title": "生长阶段",
        "rs_phase_lag": "延滞期",
        "rs_phase_acc": "加速期",
        "rs_phase_exp": "指数期",
        "rs_phase_dec": "减速期",
        "rs_phase_stat": "稳定期",
        "rs_phase_decl": "衰退期",
        "rs_interval": "区间 (h)",
        "rs_points": "点数",
        "rs_duration": "持续 (h)",
        "rs_export_pdf": "📄 导出 PDF",
        "rs_export_xlsx": "📊 导出 Excel",
        "rs_export_results": " 导出结果",
        "rs_download_excel": " 下载Excel",
        "rs_error_lib": " 安装fpdf2导出PDF",
        "rs_back": "← 返回模型",
        "rs_no_model": "未选择模型。请前往 🔬 动力学模型 选择一个。",
        "rs_no_data": "无数据。请先前往 📊 数据。",
        "rs_sub_plot1":" 浪费 × 时间",
        "rs_sub_plot2":" 分配",
        "rs_error_adjust": " 调整误差",
        "rs_named_col1": " 范围",
        "rs_named_col2": " 价值",
        "ab_title": "关于 GrowthEmulator",
        "ab_version": "版本",
        "ab_desc": (
            "GrowthEmulator 是一款微生物细胞生长模拟器，用于分析、拟合和建模微生物培养动力学数据。"
            "支持 30 余个经典和现代动力学模型、多种优化方法和交互式可视化工具。"
        ),
        "ab_author": "开发者",
        "ab_lic": "许可证",
        "ab_contact_btn": "📧 联系我们",
        "gd_title": "指南与参考文献",
        "gd_intro_title": "GrowthEmulator 是什么？",
        "gd_intro_body": (
            "GrowthEmulator v1.2 是面向研究人员、生物过程工程师和学生的交互式微生物动力学建模平台。"
            "可加载实验生长数据、选择并拟合数学模型、分析统计指标并导出可发表的结果。"
        ),
        "gd_intro_background1":" 动力学模型",
        "gd_intro_background2":" 4 种语言",
        "gd_intro_background3":" 导出 PDF 和 Excel",
        "gd_intro_background4":" 无障碍辅助",
        "gd_decision_title": "我应该使用哪个模型？",
        "gd_decision_note": "基于数据行为的快速模型选择指南",
        "gd_refs_title": "参考文献",
        "gd_param_title": "参数解读",
        "gd_param_note": "主要动力学参数的含义及典型范围",
        "gd_param_descript1":" 分析区间",
        "gd_param_descript2":" 解释",
        "gd_data":"### 🚀 推荐工作流",
        "gd_flux_recommended": """
        1. **📊 数据** → 上传文件或手动输入 → 映射变量 → 可视化曲线  
        2. **🔬 动力学模型** → 探索可用模型 → 选择最适合您数据的模型  
        3. **📈 配置优化结果** → 配置优化 → 运行分析 → 解释参数与指标 → 导出  

        > 💡 **提示:** 在数据预览中使用对数坐标轴，以直观识别指数生长期并估算 λ。  
        > 如果 R² < 0.95 或 RMSE > Xm 的 10%，请尝试其他模型或检查数据。""",
        "gd_label":"推荐型号",
        "ct_title": "联系我们",
        "ct_email": "电子邮件",
        "ct_linkedin": "LinkedIn",
        "ct_close": "关闭",
        "ab_integrators":       "积分器",
        "dt_rename_category_hint": "点击重命名此类别",
        "fit_excellent":       "优秀",
        "fit_good":            "良好",
        "fit_acceptable":      "可接受",
        "fit_insufficient":    "不足",
        "fit_quality":         "拟合质量",
        "rs_individual_results": "单独结果",
        "rs_top_n_label":      "叠加图中的曲线数",
        "rs_top_n_omitted":    "个模型已省略。调整滑块以查看更多。",
        "rs_params_adjusted":  "拟合参数",
        "rs_residuals_normal": "残差正态",
        "rs_residuals_abnormal": "残差非正态",
        "rs_run_hint":         "点击 **▶ 运行分析** 拟合模型。",
        "rs_tol_per_model":    "🔧 每模型容差（高级）",
        "rs_tol_hint":         "设置各模型容差。ODE 模型通常需要更小的值 (1e-8)。",
        "rs_model_series_col": "模型 / 系列",
        "dt_files_identical":  "个列名相同的文件 → 已堆叠",
        "dt_files_merged":     "个文件已合并（时间轴外连接）",
        "dt_col_rename_hint":  "✏️ 重命名标题 — 点击字段编辑：",
        "dt_data_applied":     "✅ 数据已应用！向下滚动并映射标题。",
        "dt_chart_error":      "图表错误：",
        "dt_estimated_mu":     "估算 μ (h⁻¹)",
        "dt_estimated_td":     "估算 td (h)",
        "dt_estimated_x0": "**X₀** 灭活 (g/L)",
        "dt_estimated_yield": "**S₀** 必须大于 **Sf**",
        "tl_enzyme_config":    "配置酶动力学分析",
        "tl_data_source_lbl":  "数据来源",
        "tl_use_loaded_data":  "使用已加载数据（数据标签页）",
        "tl_enter_manually":   "手动输入 S 和 v",
        "tl_enzyme_conc_lbl":  "[E] — 酶浓度（µM，可选）",
        "tl_enzyme_conc_help": "若提供，则计算 kcat = Vmax / [E]",
        "tl_enzyme_results":   "结果 — 酶动力学",
        "tl_ci_result":        "竞争性抑制拟合结果",
        "tl_pts_extracted":    "个 [S] vs v 数据点已从数据中提取",
        "tl_map_sub_time":     "在数据标签页映射底物和时间以使用此选项",
        "tl_enter_sv_pairs":   "输入 [S]（µM或mM）和 v（µM/s或mM/h）对，用';'分隔",
        "tl_format_invalid":   "格式无效 — 请使用';'分隔的数字",
        "tl_mm_run_btn":       "▶ 计算 — Michaelis-Menten",
        "tl_lp_config":        "配置产物生成分析",
        "tl_chick_section":    "Chick — 一阶细胞失活",
        "tl_min_pts":          "个数据点 — 至少需要 4 个",
        "tl_lp_run_btn":       "▶ 拟合 Luedeking-Piret",
        "tl_chick_run_btn":    "▶ 拟合 Chick (kd)",
        "tl_prod_primary":     "初级产物（与生长相关）",
        "tl_prod_secondary":   "次级产物（与生长无关）",
        "tl_prod_mixed":       "混合型产物（α 和 β 均显著）",
        "tl_classification":   "分类：",
        "tl_lp_results":       "结果 — Luedeking-Piret",
        "tl_ck_results":       "结果 — Chick（失活）",
        "tl_halflife":         "t½（半衰期）",
        "tl_pirt_config":      "配置产率分析",
        "tl_pirt_run_btn":     "▶ 拟合 Pirt — 可变产率",
        "tl_pirt_results":     "结果 — Pirt（可变产率）",
        "t1_pirt_mark": "输入 **μ (h⁻¹)** 和 **Y_obs (g/g)** 数据对以拟合 Pirt 模型：",
        "tl_maintenance_coef": "ms — 维持系数",
        "tl_calc_quick":       "快速计算器",
        "tl_td_gen_section":   "⏱ 倍增时间与代时",
        "tl_mu_from_pts":      "📈 从两点估算 μ",
        "tl_yield_coef_sect":  "⚗️ 观测产率系数",
        "tl_chick_const_sect": "📐 失活常数（Chick）",
        "tl_s_values_lbl":     "[S] 值：",
        "tl_v_values_lbl":     "v 值：",
        "ab_stack":            "技术栈",
        "ab_algorithms":       "算法",
        "tl_subtitle": "酶动力学、产率和细胞更新的专业模型",
        "tm_metric_col": " 专栏",
        "tm_metric_line": " 线路",
        "tm_metric_model": " 可用型号",

    },
}


def t(key):
    lang = st.session_state.get("lang", "pt")
    return _T.get(lang, _T["pt"]).get(key, _T["pt"].get(key, key))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. MODEL REGISTRY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BOXES = [
    {
        "id": "box1",
        "title": {
            "pt": "Modelos Exponenciais e Lineares Simples", 
            "en": "Simple Exponential & Linear Models",
            "es": "Modelos Exponenciales y Lineales Simples", 
            "zh": "简单指数与线性模型"
            },
        "models": [
            {"key": "malthus", 
             "name": {
                 "pt": "Malthus — Crescimento Exponencial", 
                 "en": "Malthus — Exponential Growth", 
                 "es": "Malthus — Crecimiento Exponencial", 
                 "zh": "Malthus — 指数增长"
                 },
             "author":{
                 "pt": " Thomas R. Malthus (1798)",
                 "en": " Thomas R. Malthus (1798)",
                 "es": " Thomas R. Malthus (1798)",
                 "zh": " Thomas R. Malthus (1798)",
                },
             "category": {
                 "pt": "Modelo exponencial simples", 
                 "en": "Simple exponential model",
                 "es": "Modelo exponencial simple", 
                 "zh": "简单指数模型"
                 },
             "latex": r"X(t) = X_0 \cdot e^{\mu \cdot t}",
             "params": ["X₀ (g/L)", "μ (h⁻¹)"],
             "requires": ["time", "biomass"], "ode": False},
            {"key": "linear", 
             "name": {
                 "pt": "Crescimento Linear", 
                 "en": "Linear Growth", 
                 "es": "Crecimiento Lineal", 
                 "zh": "线性增长"
                 },
             "author":{
                 "pt": " Modelo genérico",
                 "en": " Generic model",
                 "es": " Modelo genérico",
                 "zh": " 通用模型",
                },
             "category": {
                 "pt": "Modelo linear", 
                 "en": "Linear model", 
                 "es": "Modelo lineal", 
                 "zh": "线性模型"
                 },
             "latex": r"X(t) = X_0 + \mu \cdot t",
             "params": ["X₀ (g/L)", "μ (g/L·h)"],
             "requires": ["time", "biomass"], "ode": False},
        ],
    },
    {
        "id": "box2",
        "title": {"pt": "Modelos Logísticos", 
                  "en": "Logistic Models",
                  "es": "Modelos Logísticos", 
                  "zh": "Logistic 模型"
                  },
        "models": [
            {"key": "verhulst", 
             "name": {
                 "pt": "Verhulst — Logístico Clássico", 
                 "en": "Verhulst — Classic Logistic", 
                 "es": "Verhulst — Logístico Clásico", 
                 "zh": "Verhulst — 经典 Logistic"
                 },
             "author":{
                 "pt": " Pierre François Verhulst (1845)",
                 "en": " Pierre François Verhulst (1845)",
                 "es": " Pierre François Verhulst (1845)",
                 "zh": " Pierre François Verhulst (1845)",
                },
             "category": {
                 "pt": "Logístico clássico", 
                 "en": "Classic logistic", 
                 "es": "Logístico clásico", 
                 "zh": "经典 Logistic"
                 
                 },
             "latex": r"X(t)=\frac{X_m}{1+\left(\frac{X_m}{X_0}-1\right)e^{-\mu_{max}t}}",
             "params": ["X₀", "Xm", "μmax"], "requires": ["time", "biomass"], "ode": False},
            {"key": "gompertz_mod", 
             "name": {
                 "pt": "Gibson — Gompertz Modificado", 
                 "en": "Gibson — Modified Gompertz", 
                 "es": "Gibson — Gompertz Modificado", 
                 "zh": "Gibson — 改进 Gompertz"
                 },
             "author":{
                 "pt": " Gibson et al. (1987) · Zwietering et al. (1990)",
                 "en": " Gibson et al. (1987) · Zwietering et al. (1990)",
                 "es": " Gibson et al. (1987) · Zwietering et al. (1990)",
                 "zh": " Gibson et al. (1987) · Zwietering et al. (1990)",
                },
             "category": {
                 "pt": "Gompertz modificado", 
                 "en": "Modified Gompertz", 
                 "es": "Gompertz modificado", 
                 "zh": "改进 Gompertz"
                 },
             "latex": r"\ln\!\frac{X}{X_0}=A\exp\!\left(-\exp\!\left(\frac{\mu_{max}\,e}{A}(\lambda-t)+1\right)\right)",
             "params": ["A=ln(Xm/X0)", "μmax", "λ (lag)"], "requires": ["time", "biomass"], "ode": False},
            {"key": "logistic_mod", 
             "name": {
                 "pt": "Zwietering — Logístico Modificado", 
                 "en": "Zwietering — Modified Logistic", 
                 "es": "Zwietering — Logístico Modificado", 
                 "zh": "Zwietering — 改进 Logistic"
                 },
             "author":{
                 "pt": " Zwietering et al. (1990)",
                 "en": " Zwietering et al. (1990)",
                 "es": " Zwietering et al. (1990)",
                 "zh": " Zwietering et al. (1990)",
                },
             "category": {
                 "pt": "Logístico modificado", 
                 "en": "Modified logistic", 
                 "es": "Logístico modificado", 
                 "zh": "改进 Logistic"
                 },
             "latex": r"\ln\!\frac{X}{X_0}=\frac{A}{1+\exp\!\left(\frac{4\mu_{max}}{A}(\lambda-t)+2\right)}",
             "params": ["A", "μmax", "λ (lag)"], "requires": ["time", "biomass"], "ode": False},
        ],
    },
    {
        "id": "box3",
        "title": {
            "pt": "Modelos Mecanísticos", 
            "en": "Mechanistic Models",
            "es": "Modelos Mecanísticos", 
            "zh": "机理模型"
            },
        "models": [
            {"key": "baranyi", 
             "name": {
                 "pt": "Baranyi & Roberts — Modelo Mecanístico", 
                 "en": "Baranyi & Roberts — Mechanistic Model", 
                 "es": "Baranyi & Roberts — Modelo Mecanístico", 
                 "zh": "Baranyi & Roberts — 机理模型"
                 },
             "author":{
                 "pt": " József Baranyi & Terry A. Roberts (1994)",
                 "en": " József Baranyi & Terry A. Roberts (1994)",
                 "es": " József Baranyi & Terry A. Roberts (1994)",
                 "zh": " József Baranyi & Terry A. Roberts (1994)",
                },
             "category": {
                 "pt": "Mecanístico com fase lag", 
                 "en": "Mechanistic with lag", 
                 "es": "Mecanístico con fase lag", 
                 "zh": "含滞后期机理模型"
                 },
             "latex": r"\frac{dX}{dt}=\mu_{max}\frac{q}{1+q}\!\left(1-\frac{X}{X_m}\right)\!X;\quad\frac{dq}{dt}=\nu q",
             "params": ["X₀", "Xm", "μmax", "q₀", "ν"], "requires": ["time", "biomass"], "ode": True},
            {"key": "contois", 
             "name": {
                 "pt": "Contois — Cinética de Contois", 
                 "en": "Contois — Contois Kinetics", 
                 "es": "Contois — Cinética de Contois", 
                 "zh": "Contois — Contois 动力学"
                 },
             "author":{
                 "pt": " D.E. Contois (1959)",
                 "en": " D.E. Contois (1959)",
                 "es": " D.E. Contois (1959)",
                 "zh": " D.E. Contois (1959)",
                },
             "category": {
                 "pt": "Inibição por biomassa", 
                 "en": "Biomass inhibition", 
                 "es": "Inhibición por biomasa", 
                 "zh": "生物量抑制"
                 },
             "latex": r"\mu=\frac{\mu_{max} S}{K_s X + S}",
             "params": ["μmax", "Ks", "X₀", "Y"], "requires": ["time", "biomass", "substrate"], "ode": True},
            {"key": "herbert", 
             "name": {
                 "pt": "Herbert-Pirt — Com Manutenção", 
                 "en": "Herbert-Pirt — With Maintenance", 
                 "es": "Herbert-Pirt — Con Mantenimiento", 
                 "zh": "Herbert-Pirt — 含维持能"
                 },
             "author":{
                 "pt": " Dennis Herbert (1958) · Sidney Pirt (1965)",
                 "en": " Dennis Herbert (1958) · Sidney Pirt (1965)",
                 "es": " Dennis Herbert (1958) · Sidney Pirt (1965)",
                 "zh": " Dennis Herbert (1958) · Sidney Pirt (1965)",
                },
             "category": {
                 "pt": "Mecanístico com manutenção", 
                 "en": "Mechanistic with maintenance", 
                 "es": "Mecanístico con mantenimiento", 
                 "zh": "含维持能机理模型"
                 },
             "latex": r"\frac{dX}{dt}=(\mu-k_d)X;\quad\frac{dS}{dt}=-\!\left(\frac{\mu}{Y}+m_s\right)\!X",
             "params": ["μmax", "Ks", "Y", "kd", "ms"], "requires": ["time", "biomass", "substrate"], "ode": True},
        ],
    },
    {
        "id": "box4",
        "title": {
            "pt": "Modelos Sigmoides — Curva Completa", 
            "en": "Sigmoid Models — Full Growth Curve",
            "es": "Modelos Sigmoideos — Curva Completa", 
            "zh": "S 型模型 — 完整生长曲线"
            },
        "models": [
            {"key": "richards", 
             "name": {
                 "pt": "Richards — Curva Generalizada", 
                 "en": "Richards — Generalised Curve", 
                 "es": "Richards — Curva Generalizada", 
                 "zh": "Richards — 广义增长曲线"
                 },
             "author":{
                 "pt": " F.J. Richards (1959)",
                 "en": " F.J. Richards (1959)",
                 "es": " F.J. Richards (1959)",
                 "zh": " F.J. Richards (1959)",
                },
             "category": {
                 "pt": "Sigmoide generalizado", 
                 "en": "Generalized sigmoid", 
                 "es": "Sigmoide generalizado", 
                 "zh": "广义 S 型"
                 },
             "latex": r"X(t)=\frac{X_m}{\left(1+\nu\,e^{-k(t-\lambda)}\right)^{1/\nu}}",
             "params": ["Xm", "ν", "k", "λ"], "requires": ["time", "biomass"], "ode": False},
            {"key": "von_bertalanffy", 
             "name": {
                 "pt": "Von Bertalanffy — Crescimento com Saturação", 
                 "en": "Von Bertalanffy — Saturation Growth", 
                 "es": "Von Bertalanffy — Crecimiento con Saturación", 
                 "zh": "Von Bertalanffy — 饱和增长"
                 },
             "author":{
                 "pt": " Ludwig von Bertalanffy (1957)",
                 "en": " Ludwig von Bertalanffy (1957)",
                 "es": " Ludwig von Bertalanffy (1957)",
                 "zh": " Ludwig von Bertalanffy (1957)",
                },
             "category": {
                 "pt": "Crescimento allométrico", 
                 "en": "Allometric growth", 
                 "es": "Crecimiento alométrico", 
                 "zh": "异速生长"
                 },
             "latex": r"X(t)=\!\left(X_\infty^{1/3}-\!\left(X_\infty^{1/3}-X_0^{1/3}\right)e^{-\beta t/3}\right)^{\!3}",
             "params": ["X₀", "X∞", "β"], "requires": ["time", "biomass"], "ode": False},
        ],
    },
    {
        "id": "box8",
        "title": {"pt": "Modelos Cinéticos Clássicos (Dependentes de Substrato)",
                  "en": "Classic Substrate-Dependent Kinetic Models",
                  "es": "Modelos Cinéticos Clásicos (Dependientes de Sustrato)", 
                  "zh": "经典底物依赖动力学模型"
                  },
        "models": [
            {"key": "monod", 
             "name": {
                 "pt": "Monod — Cinética de Monod", 
                 "en": "Monod — Monod Kinetics", 
                 "es": "Monod — Cinética de Monod", 
                 "zh": "Monod — Monod 动力学"
                 },
             "author":{
                 "pt": " Jacques Monod (1949)",
                 "en": " Jacques Monod (1949)",
                 "es": " Jacques Monod (1949)",
                 "zh": " Jacques Monod (1949)",
                },
             "category": {
                 "pt": "Limitação por substrato", 
                 "en": "Substrate limitation",
                 "es": "Limitación por sustrato", 
                 "zh": "底物限制"
                 },
             "latex": r"\mu=\frac{\mu_{max}\,S}{K_s+S}",
             "params": ["μmax", "Ks", "X₀", "Y"], "requires": ["time", "biomass", "substrate"], "ode": True},
            {"key": "tessier", 
             "name": {
                 "pt": "Tessier — Saturação Exponencial", 
                 "en": "Tessier — Exponential Saturation", 
                 "es": "Tessier — Saturación Exponencial", 
                 "zh": "Tessier — 指数饱和"
                 },
             "author":{
                 "pt": " Georges Tessier (1942)",
                 "en": " Georges Tessier (1942)",
                 "es": " Georges Tessier (1942)",
                 "zh": " Georges Tessier (1942)",
                },
             "category": {
                 "pt": "Saturação exponencial", 
                 "en": "Exponential saturation",
                 "es": "Saturación exponencial", 
                 "zh": "指数饱和"
                 },
             "latex": r"\mu=\mu_{max}\!\left(1-e^{-S/K_s}\right)",
             "params": ["μmax", "Ks"], "requires": ["time", "biomass", "substrate"], "ode": True},
            {"key": "moser", 
             "name": {
                 "pt": "Moser — Modelo de Potência", 
                 "en": "Moser — Power Model", 
                 "es": "Moser — Modelo de Potencia", 
                 "zh": "Moser — 幂次模型"
                 },
             "author":{
                 "pt": " H. Moser (1958)",
                 "en": " H. Moser (1958)",
                 "es": " H. Moser (1958)",
                 "zh": " H. Moser (1958)",
                },
             "category": {
                 "pt": "Modelo de potência modificado", 
                 "en": "Modified power model",
                 "es": "Modelo de potencia modificado", 
                 "zh": "幂次模型"
                 },
             "latex": r"\mu=\frac{\mu_{max}}{1+\left(K_s/S\right)^n}",
             "params": ["μmax", "Ks", "n"], "requires": ["time", "biomass", "substrate"], "ode": True},
            {"key": "haldane", 
             "name": {
                 "pt": "Haldane/Andrews — Inibição por Substrato", 
                 "en": "Haldane/Andrews — Substrate Inhibition", 
                 "es": "Haldane/Andrews — Inhibición por Sustrato", 
                 "zh": "Haldane/Andrews — 底物抑制"
                 },
             "author":{
                 "pt": " J.B.S. Haldane (1930) · J.F. Andrews (1968)",
                 "en": " J.B.S. Haldane (1930) · J.F. Andrews (1968)",
                 "es": " J.B.S. Haldane (1930) · J.F. Andrews (1968)",
                 "zh": " J.B.S. Haldane (1930) · J.F. Andrews (1968)",
                },
             "category": {
                 "pt": "Inibição por substrato", 
                 "en": "Substrate inhibition",
                 "es": "Inhibición por sustrato",
                 "zh": "底物抑制"
                 },
             "latex": r"\mu=\frac{\mu_{max}\,S}{K_s+S+S^2/K_i}",
             "params": ["μmax", "Ks", "Ki"], "requires": ["time", "biomass", "substrate"], "ode": True},
            {"key": "aiba", 
             "name": {
                 "pt": "Aiba — Inibição Mista (Exponencial)", 
                 "en": "Aiba — Mixed Inhibition (Exponential)", 
                 "es": "Aiba — Inhibición Mixta (Exponencial)", 
                 "zh": "Aiba — 混合抑制（指数型）"
                 },
             "author":{
                 "pt": " Shuichi Aiba et al. (1965)",
                 "en": " Shuichi Aiba et al. (1965)",
                 "es": " Shuichi Aiba et al. (1965)",
                 "zh": " Shuichi Aiba et al. (1965)",
                },
             "category": {
                 "pt": "Inibição exponencial mista", 
                 "en": "Mixed exponential inhibition",
                 "es": "Inhibición exponencial mixta", 
                 "zh": "混合指数抑制"
                 },
             "latex": r"\mu=\frac{\mu_{max}\,S\,e^{-S/K_i}}{K_s+S}",
             "params": ["μmax", "Ks", "Ki"], "requires": ["time", "biomass", "substrate"], "ode": True},
        ],
    },
    {
        "id": "box9",
        "title": {
            "pt": "Modelos Empíricos", 
            "en": "Empirical Models",
            "es": "Modelos Empíricos", 
            "zh": "经验模型"
            },
        "models": [
            {"key": "buchanan", 
             "name": {
                 "pt": "Buchanan — Modelo Trifásico Linear", 
                 "en": "Buchanan — Three-Phase Linear", 
                 "es": "Buchanan — Modelo Trifásico Lineal", 
                 "zh": "Buchanan — 三阶段线性模型"
                 },
             "author":{
                 "pt": " Robert L. Buchanan (1993)",
                 "en": " Robert L. Buchanan (1993)",
                 "es": " Robert L. Buchanan (1993)",
                 "zh": " Robert L. Buchanan (1993)",
                },
             "category": {
                 "pt": "Trifásico empírico", 
                 "en": "Empirical three-phase",
                 "es": "Trifásico empírico", 
                 "zh": "经验三阶段"
                 },
             "latex": r"\ln X=\begin{cases}\ln X_0 & t\le\lambda\\\ln X_0+\mu_{max}(t-\lambda) & \lambda<t\le t_{max}\\\ln X_m & t>t_{max}\end{cases}",
             "params": ["X₀", "Xm", "μmax", "λ", "tmax"], "requires": ["time", "biomass"], "ode": False},
        ],
    },
    {
        "id": "box10",
        "title": {
            "pt": "Modelos Lineares de Ajuste de Curva", 
            "en": "Linear Curve Fitting",
            "es": "Modelos Lineales de Ajuste de Curva", 
            "zh": "线性曲线拟合"
            },
        "models": [
            {"key": "polynomial", 
             "name": {
                 "pt": "Regressão Polinomial", 
                 "en": "Polynomial Regression", 
                 "es": "Regresión Polinomial", 
                 "zh": "多项式回归"
                 },
             "author":{
                 "pt": " Ajuste numérico clássico",
                 "en": " Classic numeric adjustment",
                 "es": " Ajuste numérico clásico",
                 "zh": " 经典数值调整",
                },
             "category": {
                 "pt": "Regressão polinomial", 
                 "en": "Polynomial regression",
                 "es": "Regresión polinomial", 
                 "zh": "多项式回归"
                 },
             "latex": r"X=\sum_{i=0}^{n}a_i\,t^i",
             "params": ["a₀", "a₁", "a₂", "…"], "requires": ["time", "biomass"], "ode": False},
            {"key": "lin_logspace", 
             "name": {
                 "pt": "Regressão Linear em Espaço Log", 
                 "en": "Linear Regression in Log Space", 
                 "es": "Regresión Lineal en Espacio Log", 
                 "zh": "对数空间线性回归"
                 },
             "author":{
                 "pt": " Método clássico",
                 "en": " Classic method",
                 "es": " Método clásico",
                 "zh": " 经典方法",
                },
             "category": {
                 "pt": "Regressão log-linear", 
                 "en": "Log-linear regression",
                 "es": "Regresión log-lineal", 
                 "zh": "对数线性回归"
                 },
             "latex": r"\ln X=\ln X_0+\mu\,t",
             "params": ["X₀", "μ"], "requires": ["time", "biomass"], "ode": False},
        ],
    },
    {
        "id": "box11",
        "title": {
            "pt": "Modelos Não Lineares de Ajuste de Curva", 
            "en": "Non-linear Curve Fitting",
            "es": "Modelos No Lineales de Ajuste de Curva", 
            "zh": "非线性曲线拟合"
            },
        "models": [
            {"key": "gaussian", 
             "name": {
                 "pt": "Gaussiana — Ajuste por Curva Normal", 
                 "en": "Gaussian — Normal Curve Fit", 
                 "es": "Gaussiana — Ajuste por Curva Normal", 
                 "zh": "高斯 — 正态曲线拟合"
                 },
             "author":{
                 "pt": " Modelo estatístico clássico",
                 "en": " Classic statistical model",
                 "es": " Modelo estadístico clásico",
                 "zh": " 经典统计模型",
                },
             "category": {
                 "pt": "Ajuste gaussiano", 
                 "en": "Gaussian fit",
                 "es": "Ajuste gaussiano", 
                 "zh": "高斯拟合"
                 },
             "latex": r"X=A\,e^{-\frac{(t-\mu_t)^2}{2\sigma^2}}+C",
             "params": ["A", "μt", "σ", "C"], "requires": ["time", "biomass"], "ode": False},
            {"key": "power_law", 
             "name": {
                 "pt": "Lei de Potência", 
                 "en": "Power Law", "es": 
                 "Ley de Potencia", "zh": 
                 "幂律"
                 },
             "author": {
                 "pt": " Modelo Empírico Geral",
                 "en": " General Empirical Model",
                 "es": " Modelo Empírico General",
                 "zh": " 一般经验模型",
                },
             "category": {
                 "pt": "Lei de potência", 
                 "en": "Power law",
                 "es": "Ley de potencia", 
                 "zh": "幂律"
                 },
             "latex": r"X=X_0\,t^n",
             "params": ["X₀", "n"], "requires": ["time", "biomass"], "ode": False},
        ],
    },
]

ALL_MODELS = {m["key"]: m for b in BOXES for m in b["models"]}

# ── Tools models (moved from main boxes, now in Ferramentas tab) ──────────
TOOLS_BOXES = [
    {
        "id": "tbox1",
        "title": {"pt": "Cinética Enzimática Homogênea", 
                  "en": "Homogeneous Enzyme Kinetics",
                  "es": "Cinética Enzimática Homogénea", 
                  "zh": "均相酶动力学"
                  },
        "icon": "🧫",
        "models": [
            {"key": "michaelis", 
             "name": {
                "pt": "Michaelis-Menten — Cinética Enzimática", 
                "en": "Michaelis-Menten — Enzyme Kinetics", 
                "es": "Michaelis-Menten — Cinética Enzimática", 
                "zh": "Michaelis-Menten — 酶动力学"
                },
             "author": "Leonor Michaelis & Maud Menten (1913)",
             "category": {
                 "pt": "Enzimática clássica", 
                 "en": "Classic enzyme kinetics", 
                 "es": "Cinética enzimática clásica", 
                 "zh": "经典酶动力学"
                 },
             "latex": r"v=\frac{V_{max}\,S}{K_m+S}",
             "params": ["Vmax (mmol/L·h)", "Km (mmol/L)"],
             "requires": ["time", "substrate"], "ode": False},
            {"key": "inhib_comp", 
             "name": {
                "pt": "Inibição Competitiva", 
                "en": "Competitive Inhibition", 
                "es": "Inhibición Competitiva", 
                "zh": "竞争性抑制"
                },
             "author": "Briggs & Haldane (1925)",
             "category": {
                 "pt": "Inibição competitiva", 
                 "en": "Competitive inhibition", 
                 "es": "Inhibición competitiva", 
                 "zh": "竞争性抑制"
                 },
             "latex": r"v=\frac{V_{max}\,S}{K_m\!\left(1+I/K_i\right)+S}",
             "params": ["Vmax", "Km", "Ki"],
             "requires": ["time", "substrate"], "ode": False},
        ],
    },
    {
        "id": "tbox2",
        "title": {
            "pt": "Rendimento Variável e Cinética Heterogênea", 
            "en": "Variable Yield & Heterogeneous Kinetics",
            "es": "Rendimiento Variable y Cinética Heterogénea", 
            "zh": "可变产率与非均相动力学"
            },
        "icon": "⚗️",
        "models": [
            {"key": "pirt", 
             "name": {
                 "pt": "Pirt — Rendimento Variável", 
                 "en": "Pirt — Variable Yield", "es": 
                 "Pirt — Rendimiento Variable", "zh": 
                 "Pirt — 可变产率"
                 },
             "author": "Sidney John Pirt (1965)",
             "category": {
                 "pt": "Rendimento com manutenção", 
                 "en": "Yield with maintenance", 
                 "es": "Rendimiento con mantenimiento", 
                 "zh": "含维持的产率"
                 },
             "latex": r"Y_{obs}=\frac{Y_{max}}{1+m_s\,Y_{max}/\mu}",
             "params": ["Ymax (g/g)", "ms (g/g·h)"],
             "requires": ["time", "biomass", "substrate"], "ode": False},
        ],
    },
    {
        "id": "tbox3",
        "title": {
            "pt": "Turn-over Celular e Formação de Produto", 
            "en": "Cell Turnover & Product Formation",
            "es": "Recambio Celular y Formación de Producto", 
            "zh": "细胞更新与产物生成"
            },
        "icon": "🔄",
        "models": [
            {"key": "luedeking_piret", 
             "name": {
                 "pt": "Luedeking-Piret — Formação de Produto", 
                 "en": "Luedeking-Piret — Product Formation", 
                 "es": "Luedeking-Piret — Formación de Producto", 
                 "zh": "Luedeking-Piret — 产物生成"
                 },
             "author": "Robert Luedeking & E.L. Piret (1959)",
             "category": {
                 "pt": "Produto acoplado ao crescimento", 
                 "en": "Growth-coupled product", 
                 "es": "Producto acoplado al crecimiento", 
                 "zh": "生长偶联产物"
                 },
             "latex": r"\frac{dp}{dt}=\alpha\frac{dX}{dt}+\beta X",
             "params": ["α (g/g)", "β (g/g·h)"],
             "requires": ["time", "biomass"], "ode": False},
            {"key": "chick", 
             "name": {
                 "pt": "Chick — Inativação/Morte Celular", 
                 "en": "Chick — Cell Inactivation/Death", 
                 "es": "Chick — Inactivación/Muerte Celular", 
                 "zh": "Chick — 细胞失活/死亡"
                 },
             "author": "Harriette Chick (1908)",
             "category": {
                 "pt": "Morte/inativação de 1ª ordem", 
                 "en": "First-order inactivation", 
                 "es": "Inactivación de 1er orden", 
                 "zh": "一阶失活"
                 },
             "latex": r"X(t)=X_0\,e^{-k_d t}",
             "params": ["X₀ (g/L)", "kd (h⁻¹)"],
             "requires": ["time", "biomass"], "ode": False},
        ],
    }
]
ALL_TOOLS = {m["key"]: m for b in TOOLS_BOXES for m in b["models"]}


def _box_title(box):
    lang = st.session_state.get("lang", "pt")
    return box["title"].get(lang, box["title"]["en"])


def _model_category(m):
    lang = st.session_state.get("lang", "pt")
    return m["category"].get(lang, m["category"]["en"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. KINETIC MODEL FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def m_malthus(t, X0, mu):
    return np.maximum(X0 * np.exp(mu * t), 1e-12)

def m_linear(t, X0, mu):
    return X0 + mu * t

def m_verhulst(t, X0, Xm, mu_max):
    ratio = np.where(X0 > 0, Xm / X0 - 1, 1e6)
    return Xm / (1 + ratio * np.exp(-mu_max * t))

def m_gompertz_mod(t, A, mu_max, lam):
    return A * np.exp(-np.exp(mu_max * np.e / A * (lam - t) + 1))

def m_logistic_mod(t, A, mu_max, lam):
    return A / (1 + np.exp(4 * mu_max / A * (lam - t) + 2))

def m_baranyi(t_eval, X0, Xm, mu_max, q0, v=None):
    if v is None: v = mu_max
    # solve_ivp requires t_eval sorted ascending, unique, within integration span
    t_eval = np.asarray(t_eval, dtype=float)
    t_eval = np.unique(t_eval)          # sort + remove duplicates in one step
    if len(t_eval) < 2:
        return np.full(max(len(t_eval), 1), X0, dtype=float)
    def odes(t, y):
        X, q = max(y[0], 0.0), max(y[1], 0.0)
        alpha = q / (1 + q)
        dX = mu_max * alpha * (1 - X / max(Xm, 1e-9)) * X
        dq = v * q
        return [dX, dq]
    sol = solve_ivp(odes, [t_eval[0], t_eval[-1]], [max(X0, 1e-9), max(q0, 1e-9)],
                    t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-9, dense_output=False)
    if sol.success: return sol.y[0]
    return np.full_like(t_eval, X0, dtype=float)

def m_richards(t, Xm, nu, k, lam):
    nu = max(nu, 1e-6)
    return Xm / (1 + nu * np.exp(-k * (t - lam))) ** (1 / nu)

def m_von_bertalanffy(t, X0, X_inf, beta):
    X0p, Xip = max(X0, 1e-9) ** (1/3), max(X_inf, 1e-6) ** (1/3)
    return (Xip - (Xip - X0p) * np.exp(-beta / 3 * t)) ** 3

def m_michaelis(S, Vmax, Km):
    return Vmax * S / (Km + S + 1e-12)

def m_chick(t, X0, kd):
    return X0 * np.exp(-kd * t)

def m_buchanan(t, X0, Xm, mu_max, lam, tmax):
    lX0, lXm = np.log(max(X0, 1e-9)), np.log(max(Xm, 1e-9))
    r = np.where(t <= lam, lX0,
        np.where(t <= tmax, lX0 + mu_max * (t - lam), lXm))
    return np.exp(r)

def m_polynomial(t, *coeffs):
    return sum(c * t**i for i, c in enumerate(coeffs))

def m_lin_logspace(t, X0, mu):
    return X0 * np.exp(mu * t)

def m_gaussian(t, A, mu_t, sigma, C):
    return A * np.exp(-(t - mu_t)**2 / (2 * sigma**2)) + C

def m_power_law(t, X0, n):
    t_s = np.where(t > 0, t, 1e-10)
    return X0 * t_s ** n

def _monod_mu(S, mu_max, Ks): return mu_max * S / (Ks + S + 1e-12)
def _tessier_mu(S, mu_max, Ks): return mu_max * (1 - np.exp(-S / (Ks + 1e-12)))
def _moser_mu(S, mu_max, Ks, n): return mu_max / (1 + (Ks / (S + 1e-12))**n)
def _haldane_mu(S, mu_max, Ks, Ki): return mu_max * S / (Ks + S + S**2 / Ki + 1e-12)
def _aiba_mu(S, mu_max, Ks, Ki): return mu_max * S * np.exp(-S / Ki) / (Ks + S + 1e-12)

def _ode_substrate_model(t_eval, X0, S0, Y, mu_func):
    # Guarantee t_eval is sorted and has no duplicates (solve_ivp requirement)
    t_eval = np.unique(np.asarray(t_eval, dtype=float))
    if len(t_eval) < 2:
        return (np.full(max(len(t_eval), 1), X0, dtype=float),
                np.full(max(len(t_eval), 1), S0, dtype=float))
    def odes(t, y):
        X, S = max(y[0], 0.0), max(y[1], 0.0)
        mu = mu_func(S)
        return [mu * X, -mu * X / max(Y, 1e-12)]
    sol = solve_ivp(odes, [t_eval[0], t_eval[-1]], [max(X0, 1e-9), max(S0, 0.0)],
                    t_eval=t_eval, method="LSODA", rtol=1e-6, atol=1e-9)
    if sol.success: return sol.y[0], sol.y[1]
    return np.full_like(t_eval, X0, dtype=float), np.full_like(t_eval, S0, dtype=float)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. FITTING ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _auto_p0(key, t, X, S=None):
    """Heuristic initial parameter estimates."""
    X0 = float(X[0]) if X[0] > 0 else 0.01
    Xm = float(np.max(X)) * 1.1
    A  = float(np.log(Xm / max(X0, 1e-9)))
    # Estimate mu_max from steepest log-slope
    if len(t) > 2:
        lX = np.log(np.maximum(X, 1e-9))
        slopes = np.gradient(lX, t)
        mu_max = float(np.max(slopes))
    else:
        mu_max = 0.5
    mu_max = max(mu_max, 0.01)
    # Estimate lag as inflection point
    inflx = t[np.argmax(np.gradient(np.gradient(np.log(np.maximum(X, 1e-9)), t), t) > 0.01)
               ] if len(t) > 3 else t[0]
    lam = float(inflx)
    td = float(t[-1])
    S0 = float(S[0]) if S is not None and len(S) > 0 else 1.0
    Ks = S0 * 0.1
    p0_map = {
        "malthus":       [X0, mu_max],
        "linear":        [X0, (float(X[-1]) - X0) / max(t[-1], 1)],
        "verhulst":      [X0, Xm, mu_max],
        "gompertz_mod":  [max(A, 0.01), mu_max, lam],
        "logistic_mod":  [max(A, 0.01), mu_max, lam],
        "baranyi":       [X0, Xm, mu_max, 0.1],
        "richards":      [Xm, 1.0, mu_max, lam],
        "von_bertalanffy": [X0, Xm, mu_max],
        "michaelis":     [1.0, S0 * 0.5],
        "inhib_comp":    [1.0, S0 * 0.5, S0],
        "pirt":          [0.5, 0.05],
        "luedeking_piret": [0.5, 0.05],
        "chick":         [X0, 0.1],
        "monod":         [mu_max, Ks, X0, 0.5, S0],
        "tessier":       [mu_max, Ks, X0, 0.5, S0],
        "moser":         [mu_max, Ks, 1.0, X0, 0.5, S0],
        "haldane":       [mu_max, Ks, S0 * 2, X0, 0.5, S0],
        "aiba":          [mu_max, Ks, S0 * 2, X0, 0.5, S0],
        "contois":       [mu_max, 0.1, X0, 0.5, S0],
        "herbert":       [mu_max, Ks, 0.5, 0.01, 0.01, X0, S0],
        "buchanan":      [X0, Xm, mu_max, lam, td * 0.8],
        "polynomial":    [X0, mu_max, 0.0],
        "lin_logspace":  [X0, mu_max],
        "gaussian":      [Xm, td * 0.5, td * 0.2, X0 * 0.1],
        "power_law":     [X0, 0.5],
    }
    return p0_map.get(key, [0.1] * 3)


def _bounds_for(key):
    """(lower, upper) bounds for each model."""
    big = np.inf
    b_map = {
        "malthus":       ([1e-6, -big],        [big, big]),
        "linear":        ([0, -big],            [big, big]),
        "verhulst":      ([1e-6, 1e-5, 1e-6], [big, big, big]),
        "gompertz_mod":  ([1e-4, 1e-4, -big],  [big, big, big]),
        "logistic_mod":  ([1e-4, 1e-4, -big],  [big, big, big]),
        "baranyi":       ([1e-6, 1e-4, 1e-4, 0], [big]*4),
        "richards":      ([1e-4, 1e-4, 1e-4, -big], [big, big, big, big]),
        "von_bertalanffy": ([1e-6, 1e-4, 1e-6], [big, big, big]),
        "michaelis":     ([1e-6, 1e-6], [big, big]),
        "chick":         ([1e-6, 1e-4], [big, big]),
        "buchanan":      ([1e-6, 1e-4, 1e-4, 0, 0], [big]*5),
        "polynomial":    ([-big]*3, [big]*3),
        "lin_logspace":  ([1e-6, -big], [big, big]),
        "gaussian":      ([0, -big, 1e-4, -big], [big, big, big, big]),
        "power_law":     ([1e-6, -big], [big, big]),
    }
    return b_map.get(key, ([-big]*10, [big]*10))


def predict_for_plot(key, params, t_fine, X0_data=None, S0=None):
    """
    Generate a smooth, fine-grid prediction for plotting.
    X0_data: first observed X value (needed for Gompertz/Logistic scaling).
    S0: initial substrate concentration (needed for ODE substrate models).
    """
    try:
        p = params
        if key == "malthus":         return m_malthus(t_fine, *p)
        if key == "linear":          return m_linear(t_fine, *p)
        if key == "verhulst":        return m_verhulst(t_fine, *p)
        if key == "gompertz_mod":
            X0 = X0_data if X0_data and X0_data > 0 else 1.0
            return np.maximum(X0 * np.exp(m_gompertz_mod(t_fine, *p)), 1e-9)
        if key == "logistic_mod":
            X0 = X0_data if X0_data and X0_data > 0 else 1.0
            return np.maximum(X0 * np.exp(m_logistic_mod(t_fine, *p)), 1e-9)
        if key == "baranyi":         return m_baranyi(t_fine, *p)
        if key == "richards":        return m_richards(t_fine, *p)
        if key == "von_bertalanffy": return m_von_bertalanffy(t_fine, *p)
        if key == "michaelis":       return m_michaelis(t_fine, *p)
        if key == "chick":           return m_chick(t_fine, *p)
        if key == "buchanan":        return m_buchanan(t_fine, *p)
        if key == "polynomial":      return m_polynomial(t_fine, *p)
        if key == "lin_logspace":    return m_lin_logspace(t_fine, *p)
        if key == "gaussian":        return m_gaussian(t_fine, *p)
        if key == "power_law":       return m_power_law(t_fine, *p)
        # ── ODE substrate-dependent models ──
        if key in ("monod", "tessier", "moser", "haldane", "aiba", "contois", "herbert") \
                and S0 is not None:
            # p layout: [mu_params...] + [X0, Y]
            X0_ode = abs(p[-2]) if len(p) >= 2 else (X0_data or 0.1)
            Y_ode  = max(abs(p[-1]), 1e-4) if len(p) >= 1 else 0.5
            if key == "monod":
                mu_f = lambda S_: _monod_mu(S_, abs(p[0]), abs(p[1]))
            elif key == "tessier":
                mu_f = lambda S_: _tessier_mu(S_, abs(p[0]), abs(p[1]))
            elif key == "moser":
                mu_f = lambda S_: _moser_mu(S_, abs(p[0]), abs(p[1]), abs(p[2]))
            elif key in ("haldane",):
                mu_f = lambda S_: _haldane_mu(S_, abs(p[0]), abs(p[1]), abs(p[2]))
            elif key == "aiba":
                mu_f = lambda S_: _aiba_mu(S_, abs(p[0]), abs(p[1]), abs(p[2]))
            elif key == "contois":
                mu_f = lambda S_: abs(p[0]) * S_ / (abs(p[1]) * X0_ode + S_ + 1e-12)
            else:  # herbert
                mu_f = lambda S_: _monod_mu(S_, abs(p[0]), abs(p[1]))
            Xp, _ = _ode_substrate_model(t_fine, X0_ode, S0, Y_ode, mu_f)
            return Xp
    except Exception:
        pass
    return np.zeros_like(t_fine)


def fitness_label(r2):
    """Return (label, color, emoji) based on R² quality — language-aware."""
    if r2 >= 0.99: return t("fit_excellent"),    "#56d364", "🟢"
    if r2 >= 0.95: return t("fit_good"),         "#a5d96a", "🟡"
    if r2 >= 0.85: return t("fit_acceptable"),   "#e3b341", "🟠"
    return          t("fit_insufficient"),        "#f85149", "🔴"


def fit_model(key, t, X, S=None, settings=None):
    """
    Fit the selected model to data.
    Returns: (params, y_pred, fit_info_dict)
    """
    if settings is None:
        settings = {"tolerance": 1e-6, "max_iter": 1000, "restarts": "standard"}

    t = np.asarray(t, dtype=float)
    X = np.asarray(X, dtype=float)
    p0 = _auto_p0(key, t, X, S)
    bounds = _bounds_for(key)
    method = "trf" if len(p0) > 1 else "lm"
    tol = settings.get("tolerance", 1e-6)
    maxfev = settings.get("max_iter", 1000) * 100

    def _try_curvefit(func, y_target, p0_, bounds_):
        try:
            popt, _ = curve_fit(func, t, y_target, p0=p0_, bounds=bounds_,
                                 method="trf", ftol=tol, xtol=tol, max_nfev=maxfev)
            return popt
        except Exception:
            return None

    # ── Direct algebraic models ──────────────────────────────
    if key == "malthus":
        popt = _try_curvefit(m_malthus, X, p0, bounds)
        if popt is None: popt = [X[0], 0.3]
        y_pred = m_malthus(t, *popt)

    elif key == "linear":
        popt = _try_curvefit(m_linear, X, p0, bounds)
        if popt is None: popt = [X[0], 0.1]
        y_pred = m_linear(t, *popt)

    elif key == "verhulst":
        popt = _try_curvefit(m_verhulst, X, p0, bounds)
        if popt is None: popt = [X[0], max(X)*1.1, 0.3]
        y_pred = m_verhulst(t, *popt)

    elif key in ("gompertz_mod", "logistic_mod"):
        X0 = max(float(X[0]), 1e-9)
        y_target = np.log(np.maximum(X / X0, 1e-9))
        func = m_gompertz_mod if key == "gompertz_mod" else m_logistic_mod
        popt = _try_curvefit(func, y_target, p0, bounds)
        if popt is None: popt = [max(np.log(max(X)/X0), 0.01), 0.3, t[len(t)//4]]
        y_pred = X0 * np.exp(func(t, *popt))

    elif key == "baranyi":
        def _bar_wrap(t_, X0, Xm, mu_max, q0):
            return m_baranyi(t_, X0, Xm, mu_max, q0)
        popt = _try_curvefit(_bar_wrap, X, [p0[0], p0[1], p0[2], p0[3]],
                              ([1e-6, 1e-4, 1e-4, 0], [np.inf]*4))
        if popt is None: popt = p0[:4]
        y_pred = m_baranyi(t, *popt)

    elif key == "richards":
        popt = _try_curvefit(m_richards, X, p0, bounds)
        if popt is None: popt = [max(X)*1.1, 1.0, 0.3, t[len(t)//4]]
        y_pred = m_richards(t, *popt)

    elif key == "von_bertalanffy":
        popt = _try_curvefit(m_von_bertalanffy, X, p0, bounds)
        if popt is None: popt = [X[0], max(X)*1.2, 0.2]
        y_pred = m_von_bertalanffy(t, *popt)

    elif key == "chick":
        popt = _try_curvefit(m_chick, X, p0, ([1e-6, 1e-4], [np.inf, np.inf]))
        if popt is None: popt = [X[0], 0.1]
        y_pred = m_chick(t, *popt)

    elif key == "buchanan":
        popt = _try_curvefit(m_buchanan, X, p0, bounds)
        if popt is None: popt = p0
        y_pred = m_buchanan(t, *popt)

    elif key == "polynomial":
        deg = 3
        coeffs = np.polyfit(t, X, deg)[::-1]
        popt = list(coeffs)
        y_pred = m_polynomial(t, *popt)

    elif key == "lin_logspace":
        lX = np.log(np.maximum(X, 1e-9))
        try:
            popt2, _ = curve_fit(lambda t_, X0, mu: np.log(X0) + mu * t_,
                                  t, lX, p0=[p0[0], p0[1]], bounds=([1e-9, -np.inf], [np.inf, np.inf]),
                                  max_nfev=maxfev)
            popt = [popt2[0], popt2[1]]
        except Exception:
            popt = [X[0], 0.3]
        y_pred = m_lin_logspace(t, *popt)

    elif key == "gaussian":
        popt = _try_curvefit(m_gaussian, X, p0, bounds)
        if popt is None: popt = [max(X), t[len(t)//2], t[-1]/5, min(X)]
        y_pred = m_gaussian(t, *popt)

    elif key == "power_law":
        popt = _try_curvefit(m_power_law, X, p0, bounds)
        if popt is None: popt = [X[0], 0.5]
        y_pred = m_power_law(t, *popt)

    elif key == "michaelis" and S is not None:
        S_arr = np.asarray(S, dtype=float)
        popt = _try_curvefit(lambda S_, Vm, Km: m_michaelis(S_, Vm, Km),
                              np.gradient(S_arr, t) * (-1), p0[:2],
                              ([1e-6, 1e-6], [np.inf, np.inf]))
        if popt is None: popt = [1.0, S_arr[0]*0.5]
        y_pred = X  # rate fit, show X as-is

    # ── ODE substrate-dependent ──────────────────────────────
    elif key in ("monod", "tessier", "moser", "haldane", "aiba", "contois", "herbert") \
            and S is not None:
        S_arr = np.asarray(S, dtype=float)
        S0_ = float(S_arr[0])

        def _ode_residuals(p, mu_f):
            X0_, Y_ = abs(p[0]), abs(p[1])
            Xp, _ = _ode_substrate_model(t, X0_, S0_, Y_, mu_f)
            return float(np.sum((X - Xp)**2))

        if key == "monod":
            def _mu(p_mu): return lambda S_: _monod_mu(S_, abs(p_mu[0]), abs(p_mu[1]))
            p_mu0, p_ode0 = [p0[0], p0[1]], [p0[2], p0[3]]
        elif key == "tessier":
            def _mu(p_mu): return lambda S_: _tessier_mu(S_, abs(p_mu[0]), abs(p_mu[1]))
            p_mu0, p_ode0 = [p0[0], p0[1]], [p0[2], p0[3]]
        elif key == "moser":
            def _mu(p_mu): return lambda S_: _moser_mu(S_, abs(p_mu[0]), abs(p_mu[1]), abs(p_mu[2]))
            p_mu0, p_ode0 = [p0[0], p0[1], p0[2]], [p0[3], p0[4]]
        elif key in ("haldane", "aiba"):
            _m = _haldane_mu if key == "haldane" else _aiba_mu
            def _mu(p_mu): return lambda S_: _m(S_, abs(p_mu[0]), abs(p_mu[1]), abs(p_mu[2]))
            p_mu0, p_ode0 = [p0[0], p0[1], p0[2]], [p0[3], p0[4]]
        elif key == "contois":
            def _mu(p_mu): return lambda S_: abs(p_mu[0]) * S_ / (abs(p_mu[1]) * X[0] + S_ + 1e-12)
            p_mu0, p_ode0 = [p0[0], p0[1]], [p0[2], p0[3]]
        elif key == "herbert":
            def _mu(p_mu): return lambda S_: _monod_mu(S_, abs(p_mu[0]), abs(p_mu[1]))
            p_mu0, p_ode0 = [p0[0], p0[1]], [p0[2], p0[3]]

        try:
            res = minimize(lambda p: _ode_residuals(p, _mu(p_mu0)), p_ode0,
                           method="Nelder-Mead",
                           options={"xatol": tol, "fatol": tol, "maxiter": settings.get("max_iter", 500)})
            p_ode_fit = res.x
        except Exception:
            p_ode_fit = p_ode0

        popt = list(p_mu0) + list(p_ode_fit)
        mu_f = _mu(p_mu0)
        Xp, _ = _ode_substrate_model(t, abs(p_ode_fit[0]), S0_, abs(p_ode_fit[1]), mu_f)
        y_pred = Xp

    else:
        popt = p0
        y_pred = X.copy()

    y_pred = np.asarray(y_pred, dtype=float)
    fit_info = {"params": popt, "model_key": key}
    return popt, y_pred, fit_info


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. STATISTICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def rmse(obs, pred):
    return float(np.sqrt(np.mean((obs - pred) ** 2)))

def r2_adj(obs, pred, k):
    n = len(obs)
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    return 1 - (1 - r2) * (n - 1) / max(n - k - 1, 1)

def aic(obs, pred, k):
    n = len(obs)
    mse = np.mean((obs - pred) ** 2)
    return n * np.log(mse + 1e-12) + 2 * k

def bic(obs, pred, k):
    n = len(obs)
    mse = np.mean((obs - pred) ** 2)
    return n * np.log(mse + 1e-12) + k * np.log(n)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. DATA UTILITIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def detect_delimiter(text_sample):
    """Detect BR (;,) vs EN (,,.) delimiter style."""
    sc = text_sample.count(";")
    cm = text_sample.count(",")
    if sc > cm:
        return ";", ","    # BR: ; = col sep, , = decimal
    return ",", "."        # EN/default


@st.cache_data(show_spinner=False)
def load_file(uf_bytes, uf_name, header_row=0):
    """Load uploaded file into DataFrame with smart delimiter detection."""
    name = uf_name.lower()
    try:
        if name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(BytesIO(uf_bytes), header=header_row)
        text = uf_bytes.decode("utf-8", errors="replace")
        sep, dec = detect_delimiter(text[:2000])
        return pd.read_csv(BytesIO(uf_bytes), sep=sep, decimal=dec, header=header_row)
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        return None


def clean_df(df, time_col, biomass_col):
    """Drop NaN, validate time."""
    cols = [c for c in [time_col, biomass_col] if c]
    sub = df[cols].dropna()
    return sub.reset_index(drop=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. PHASE DETECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def detect_phases(t, X):
    """Simple heuristic phase detection using smoothed growth rate."""
    t = np.asarray(t, dtype=float)
    X = np.asarray(X, dtype=float)
    n = len(t)
    if n < 5:
        return {}
    lX = np.log(np.maximum(X, 1e-9))
    mu = np.gradient(lX, t)
    mu_smooth = pd.Series(mu).rolling(max(2, n//6), center=True, min_periods=1).mean().values
    mu_max = float(np.max(mu_smooth))
    mu_thr  = mu_max * 0.05   # 5% of max = near zero

    phases = {}
    # Lag: initial region with near-zero growth
    lag_end = 0
    for i in range(n):
        if mu_smooth[i] > mu_thr:
            lag_end = i; break
    if lag_end > 0:
        phases["lag"] = (t[0], t[lag_end], list(range(0, lag_end + 1)))

    # Exponential: where mu > 70% of max
    exp_thr = mu_max * 0.7
    exp_idx = [i for i in range(lag_end, n) if mu_smooth[i] >= exp_thr]
    if exp_idx:
        phases["exp"] = (t[exp_idx[0]], t[exp_idx[-1]], exp_idx)
        # Acceleration: between lag and exp
        acc_idx = list(range(lag_end, exp_idx[0]))
        if acc_idx:
            phases["acc"] = (t[acc_idx[0]], t[acc_idx[-1]], acc_idx)
        # Deceleration: after exp, before stationary
        Xmax_i = int(np.argmax(X))
        dec_idx = list(range(exp_idx[-1], Xmax_i + 1))
        if len(dec_idx) > 1:
            phases["dec"] = (t[dec_idx[0]], t[dec_idx[-1]], dec_idx)
        # Stationary & decline
        stat_thr = mu_max * 0.05
        stat_idx = [i for i in range(Xmax_i, n) if abs(mu_smooth[i]) <= stat_thr]
        dec2_idx = [i for i in range(Xmax_i, n) if mu_smooth[i] < -stat_thr]
        if stat_idx:
            phases["stat"] = (t[stat_idx[0]], t[stat_idx[-1]], stat_idx)
        if dec2_idx:
            phases["decl"] = (t[dec2_idx[0]], t[dec2_idx[-1]], dec2_idx)
    return phases


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. CSS INJECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def inject_css():
    dark      = st.session_state.get("dark_mode", True)
    contrast  = st.session_state.get("contrast", None)

    if contrast == "high":
        bg, bg2, bgc = "#000000", "#111111", "#111111"
        fg, fg2      = "#ffffff", "#eeeeee"
        acc, acc2    = "#ffff00", "#00ffff"
        border       = "#ffffff"
    elif contrast == "deuter":       # Deuteranopia – blue/orange safe
        bg, bg2, bgc = "#0a0a1e", "#12122e", "#12122e"
        fg, fg2      = "#e8e8ff", "#b0b0dd"
        acc, acc2    = "#4fc3f7", "#f57c00"
        border       = "#4fc3f7"
    elif contrast == "protan":       # Protanopia – blue/yellow
        bg, bg2, bgc = "#050510", "#0d0d20", "#0d0d20"
        fg, fg2      = "#f0f0ff", "#aaaacc"
        acc, acc2    = "#1de9b6", "#fdd835"
        border       = "#1de9b6"
    elif dark:
        bg, bg2, bgc = "#0d1117", "#161b22", "#1c2128"
        fg, fg2      = "#e6edf3", "#8b949e"
        acc, acc2    = "#00c8b4", "#7c3aed"
        border       = "#30363d"
    else:
        bg, bg2, bgc = "#f0f4f8", "#ffffff", "#ffffff"
        fg, fg2      = "#1e293b", "#64748b"
        acc, acc2    = "#009688", "#6d28d9"
        border       = "#cbd5e1"

    st.markdown(f"""
<style>
/* ── Base ── */
:root {{
    --bg:   {bg};   --bg2:  {bg2};  --bgc: {bgc};
    --fg:   {fg};   --fg2:  {fg2};
    --acc:  {acc};  --acc2: {acc2}; --brd: {border};
    --radius: 10px; --shadow: 0 2px 12px rgba(0,0,0,.35);
}}
.stApp {{ background:{bg} !important; color:{fg} !important; }}
section[data-testid="stSidebar"] {{
    background:{bg2} !important;
    border-right:1px solid {border} !important;
    padding-top:0 !important;
}}
section[data-testid="stSidebar"] * {{ color:{fg} !important; }}
[data-testid="stHeader"]  {{ background:{bg2} !important; }}
[data-testid="stToolbar"] {{ background:{bg2} !important; }}
/* ── Cards ── */
.bio-card {{
    background:{bgc}; border:1px solid {border};
    border-radius:var(--radius); padding:16px 20px;
    box-shadow:var(--shadow); margin-bottom:14px;
}}
.bio-card.selected  {{ border-color:{acc}; box-shadow:0 0 0 2px {acc}; }}
.bio-card.disabled  {{ opacity:.45; pointer-events:none; filter:grayscale(60%); }}
.bio-card h4 {{ color:{acc};  margin:0 0 4px 0; font-size:0.92rem; font-weight:700; }}
.bio-card .author {{ color:{fg2}; font-size:0.78rem; margin-bottom:6px; }}
.bio-card .category {{ color:{acc2}; font-size:0.75rem; letter-spacing:.04em; text-transform:uppercase; }}
/* ── Nav buttons ── */
.stButton>button {{
    width:100%; text-align:left; background:transparent;
    border:1px solid transparent; border-radius:8px; color:{fg};
    padding:8px 12px; font-size:.87rem; transition:.15s;
}}
.stButton>button:hover {{ background:{bg}; border-color:{border}; }}
.stButton>button.active-nav {{ background:{bg}; border-color:{acc}; color:{acc}; font-weight:700; }}
/* ── Metric boxes ── */
.metric-box {{
    background:{bg2}; border:1px solid {border}; border-radius:8px;
    padding:12px 16px; text-align:center;
}}
.metric-box .val {{ font-size:1.35rem; font-weight:700; color:{acc}; }}
.metric-box .lbl {{ font-size:0.73rem; color:{fg2}; margin-top:2px; }}
/* ── Phase pill ── */
.phase-pill {{
    display:inline-block; padding:2px 10px; border-radius:20px;
    font-size:.75rem; font-weight:600; margin:2px;
}}
.phase-lag  {{ background:#1a365d; color:#90cdf4; }}
.phase-acc  {{ background:#1c4532; color:#9ae6b4; }}
.phase-exp  {{ background:#234e52; color:#81e6d9; }}
.phase-dec  {{ background:#322659; color:#e9d8fd; }}
.phase-stat {{ background:#2d3748; color:#e2e8f0; }}
.phase-decl {{ background:#63171b; color:#feb2b2; }}
/* ── Header ── */
#bio-header {{
    display:flex; align-items:center; justify-content:space-between;
    padding:10px 24px 10px 0; border-bottom:1px solid {border};
    background:{bg2}; margin-bottom:18px;
}}
#bio-header .logo-area {{ display:flex; align-items:center; gap:10px; }}
#bio-header .app-name  {{ font-size:1.1rem; font-weight:800; color:{acc}; letter-spacing:.02em; }}
#bio-header .app-sub   {{ font-size:.72rem; color:{fg2}; }}
/* ── Footer ── */
#bio-footer {{
    border-top:1px solid {border}; padding:12px 0 6px;
    text-align:center; font-size:.72rem; color:{fg2}; margin-top:40px;
}}
#bio-footer a {{ color:{acc}; text-decoration:none; }}
/* ── Scrollbar ── */
::-webkit-scrollbar {{ width:6px; height:6px; }}
::-webkit-scrollbar-track {{ background:{bg}; }}
::-webkit-scrollbar-thumb {{ background:{border}; border-radius:3px; }}
/* ── Misc ── */
.stDataFrame, .stTable {{ border-radius:8px; overflow:hidden; }}
hr {{ border-color:{border}; }}
.stAlert {{ border-radius:8px; }}

/* ── Mobile / Responsive ─────────────────────────── */
@media screen and (max-width: 768px) {{
    .main .block-container {{
        padding: 0.75rem 0.6rem !important;
        max-width: 100% !important;
    }}
    [data-testid="stHorizontalBlock"] {{
        flex-direction: column !important;
        gap: 6px !important;
    }}
    [data-testid="column"] {{
        width: 100% !important;
        flex: 0 0 100% !important;
        min-width: 100% !important;
    }}
    .bio-card {{ padding: 12px 14px; }}
    .bio-card h4 {{ font-size: .88rem; }}
    .stButton > button {{
        min-height: 44px !important;
        font-size: .86rem !important;
    }}
    .metric-box .val {{ font-size: 1.1rem; }}
    .metric-box .lbl {{ font-size: .68rem; }}
    .phase-pill {{ font-size: .66rem; padding: 2px 7px; }}
    .stDataFrame {{ font-size: .78rem; }}
    .tag-label {{ font-size: .72rem; padding: 2px 8px; }}
    #bio-footer {{ font-size: .65rem; }}
    section[data-testid="stSidebar"] {{ min-width: 0 !important; }}
}}
@media screen and (max-width: 480px) {{
    .main .block-container {{ padding: 0.5rem !important; }}
    .metric-box .val {{ font-size: .95rem; }}
    h1, h2 {{ font-size: 1.2rem !important; }}
    h3 {{ font-size: 1rem !important; }}
}}
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. SESSION STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _init():
    defs = {
        "tab":             "data",
        "lang":            "pt",
        "dark_mode":       True,
        "contrast":        None,
        "df":              None,          # active DataFrame (after exclusions)
        "df_raw":          None,          # original loaded DataFrame
        "df_clean":        None,          # cleaned (NaN removed) DataFrame
        "excluded_cols":   [],            # columns excluded by user
        "headers":         {},            # {"time":col, "biomass":col, ...}
        "selected_models": [],            # list of model keys for comparison
        "fit_results":     {},            # {model_key: result_dict}
        "use_manual":      False,
        "manual_rows":     [["", ""], ["", ""], ["", ""]],
        "manual_cols":     ["t (h)", "X (g/L)"],
        "tools_inputs":    {},            # inputs for tools tab
        "tools_results":   {},            # results from tools tab
        "opt": {
            "estimate":      "auto",
            "tolerance":     1e-6,
            "max_iter":      1000,
            "restarts":      "standard",
            "initial_guess": {},
            "per_model_tol": {},          # {model_key: tolerance}
        },
    }
    for k, v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 11. LOGO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_LOGO_PATH  = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
_FONTS_DIR  = os.path.join(os.path.dirname(__file__), "fonts")
_FONT_REG   = os.path.join(_FONTS_DIR, "DejaVuSans.ttf")
_FONT_BOLD  = os.path.join(_FONTS_DIR, "DejaVuSans-Bold.ttf")
_HAS_DEJAVU = os.path.exists(_FONT_REG) and os.path.exists(_FONT_BOLD)
_ADSENSE_PATH = os.path.join(os.path.dirname(__file__), "static", "adsense.html")


def _render_adsense_banner(height: int = 100):
    """
    Render the AdSense leaderboard from a standalone HTML file.
    Uses st.iframe (Streamlit >= 1.56) when available, which auto-detects
    HTML-string content and renders it in a proper iframe document — this
    gives the AdSense script its own <head>/<body> context, which is more
    reliable than injecting a bare <ins> fragment.
    Falls back to streamlit.components.v1.html on older Streamlit versions.
    """
    try:
        with open(_ADSENSE_PATH, "r", encoding="utf-8") as f:
            ad_html = f.read()
    except Exception:
        return  # static/adsense.html missing — skip silently

    if hasattr(st, "iframe"):
        st.iframe(ad_html, height=height)
    else:
        _components_html(ad_html, height=height, scrolling=False)
if os.path.exists(_LOGO_PATH):
    with open(_LOGO_PATH, "rb") as _lf:
        LOGO_B64 = base64.b64encode(_lf.read()).decode()
    LOGO_HTML = f'<img src="data:image/png;base64,{LOGO_B64}" style="height:42px;width:42px;object-fit:cover;border-radius:8px;">'
else:
    LOGO_HTML = '<div style="width:42px;height:42px;border-radius:8px;background:var(--acc);display:flex;align-items:center;justify-content:center;font-size:20px;">🧫</div>'


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 12. CONTACT DIALOG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.dialog(f'{t("nav_contact")}')
def contact_dialog():
    st.markdown(f"""
<div style="padding:8px 0">
  <p style="font-size:.85rem;color:var(--fg2)">
    {t("ct_email")}: <a href="mailto:eng.matheusmbatista@gmail.com" style="color:var(--acc)">eng.matheusmbatista@gmail.com</a>
  </p>
  <p style="font-size:.85rem;color:var(--fg2)">
    {t("ct_linkedin")}:
    <a href="https://www.linkedin.com/in/matheusmonteirobatista/" target="_blank"
       style="color:var(--acc)">linkedin.com/in/matheusmonteirobatista</a>
  </p>
</div>
""", unsafe_allow_html=True)
    if st.button(t("ct_close"), use_container_width=True): st.rerun()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 13. SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
<div style="padding:14px 0 8px 8px; border-bottom:1px solid var(--brd); margin-bottom:12px;">
  <div style="display:flex;align-items:center;gap:8px">
    {LOGO_HTML}
    <div><div style="font-weight:800;color:var(--acc);font-size:.95rem">{t("app_name")}</div>
    <div style="font-size:.7rem;color:var(--fg2)">Microbial Growth Simulator</div></div>
  </div>
</div>
""", unsafe_allow_html=True)
        nav_items = [
            ("data",    t("nav_data")),
            ("models",  t("nav_models")),
            ("tools",   t("nav_tools")),
            ("results", t("nav_results")),
            ("about",   t("nav_about")),
            ("guide",   t("nav_guide")),
        ]
        for tab_key, label in nav_items:
            active = st.session_state.tab == tab_key
            if st.button(label, key=f"nav_{tab_key}",
                         use_container_width=True,
                         type="primary" if active else "secondary"):
                st.session_state.tab = tab_key
                st.rerun()

        st.markdown("<div style='height:1px;background:var(--brd);margin:12px 0'></div>",
                    unsafe_allow_html=True)
        if st.button(t("nav_contact"), use_container_width=True):
            contact_dialog()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 14. HEADER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def render_header():
    c1, c2 = st.columns([5, 3])
    with c1:
        st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;padding:4px 0">
  {LOGO_HTML}
  <div>
    <span style="font-weight:800;font-size:1.2rem;color:var(--acc)">{t("app_name")}</span>
    <span style="font-size:.72rem;color:var(--fg2);margin-left:8px">Microbial Growth Simulator</span>
  </div>
</div>
""", unsafe_allow_html=True)
    with c2:
        hc1, hc2, hc3 = st.columns(3)
        with hc1:
            dm_label = t("light_mode") if st.session_state.dark_mode else t("dark_mode")
            if st.button(dm_label, use_container_width=True):
                st.session_state.dark_mode = not st.session_state.dark_mode
                st.rerun()
        with hc2:
            contrast_opts = {"— Normal —": None, "Alto Contraste": "high",
                              "Deuteranopia": "deuter", "Protanopia": "protan"}
            contrast_labels = list(contrast_opts.keys())
            # Resolve current selection by value
            curr_contrast_val = st.session_state.get("contrast", None)
            curr_contrast_idx = next(
                (i for i, k in enumerate(contrast_labels) if contrast_opts[k] == curr_contrast_val), 0)
            selected_c = st.selectbox(t("contrast"), contrast_labels,
                                       index=curr_contrast_idx,
                                       label_visibility="collapsed", key="contrast_sel")
            new_contrast = contrast_opts[selected_c]
            if new_contrast != st.session_state.contrast:
                st.session_state.contrast = new_contrast
                st.rerun()
        with hc3:
            lang_opts   = {"Português": "pt", "English": "en", "Español": "es", "中文": "zh"}
            lang_labels = list(lang_opts.keys())
            curr_lang   = st.session_state.get("lang", "pt")
            curr_lang_idx = next(
                (i for i, k in enumerate(lang_labels) if lang_opts[k] == curr_lang), 0)
            lang_sel = st.selectbox(t("lang_label"), lang_labels,
                                     index=curr_lang_idx,
                                     label_visibility="collapsed", key="lang_sel")
            new_lang = lang_opts[lang_sel]
            if new_lang != curr_lang:
                st.session_state.lang = new_lang
                st.rerun()
    st.divider()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 15. FOOTER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def render_footer():
    # ── AdSense leaderboard (728×90) — non-intrusive, centrado no rodapé ──
    _render_adsense_banner(height=100)
    st.markdown("""
<div id="bio-footer">
  <a href="https://github.com/matheusmonteirobatista/growthemulator">GrowthEmulator</a> © 2026 by
  <a href="https://www.linkedin.com/in/matheusmonteirobatista/">Matheus Monteiro Batista</a>
  is licensed under
  <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/">CC BY-NC-ND 4.0</a>
  <img src="https://mirrors.creativecommons.org/presskit/icons/cc.svg"
       style="max-width:1em;max-height:1em;margin-left:.3em;vertical-align:middle">
  <img src="https://mirrors.creativecommons.org/presskit/icons/by.svg"
       style="max-width:1em;max-height:1em;vertical-align:middle">
  <img src="https://mirrors.creativecommons.org/presskit/icons/nc.svg"
       style="max-width:1em;max-height:1em;vertical-align:middle">
  <img src="https://mirrors.creativecommons.org/presskit/icons/nd.svg"
       style="max-width:1em;max-height:1em;vertical-align:middle">
  &nbsp;|&nbsp;
  <a href="mailto:eng.matheusmbatista@gmail.com">eng.matheusmbatista@gmail.com</a>
</div>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 16. HELPERS — available model count
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ── Header access helpers ─────────────────────────────────────
def _hdr_list(var_key: str) -> list:
    """Return mapped column(s) for var_key as a clean list (handles str or list)."""
    val = st.session_state.headers.get(var_key)
    if val is None:
        return []
    if isinstance(val, list):
        return [v for v in val if v]
    return [val] if val else []


def _hdr_primary(var_key: str):
    """Return the first mapped column for var_key, or None."""
    lst = _hdr_list(var_key)
    return lst[0] if lst else None


def _mapped_vars() -> set:
    h = st.session_state.headers
    mapped = set()
    if h.get("time"):           mapped.add("time")
    if _hdr_list("biomass"):    mapped.add("biomass")
    if _hdr_list("substrate"):  mapped.add("substrate")
    if h.get("ph"):             mapped.add("ph")
    if _hdr_list("product"):    mapped.add("product")
    return mapped


def _count_available_models():
    mv = _mapped_vars()
    return sum(1 for m in ALL_MODELS.values() if set(m["requires"]).issubset(mv))


def _model_available(m):
    return set(m["requires"]).issubset(_mapped_vars())


def _model_name(model_or_meta: dict) -> str:
    """Return the model name in the currently selected language.
    Accepts either a BOXES model dict or an ALL_MODELS meta dict.
    Falls back to PT if the key is missing, then to the raw string if
    'name' is still a plain string (backward-compat).
    """
    lang = st.session_state.get("lang", "pt")
    name_val = model_or_meta.get("name", "")
    if isinstance(name_val, dict):
        return name_val.get(lang) or name_val.get("pt") or next(iter(name_val.values()), "")
    return str(name_val)  # legacy plain string


def _extract_sorted_series(df_src, t_col: str, x_col: str, s_col=None):
    """
    Extract time, biomass (and optional substrate) from df_src,
    drop NaNs, sort by time, remove duplicate time points.
    Returns (t_arr, x_arr, s_arr | None) as float64 numpy arrays.
    """
    t_raw = pd.to_numeric(df_src[t_col], errors="coerce")
    x_raw = pd.to_numeric(df_src[x_col], errors="coerce")
    mask  = t_raw.notna() & x_raw.notna()
    t_a   = t_raw[mask].values.astype(float)
    x_a   = x_raw[mask].values.astype(float)
    n     = min(len(t_a), len(x_a))
    t_a, x_a  = t_a[:n], x_a[:n]

    sort_idx          = np.argsort(t_a, kind="stable")
    t_a, x_a          = t_a[sort_idx], x_a[sort_idx]
    _, uni_idx         = np.unique(t_a, return_index=True)
    t_a, x_a           = t_a[uni_idx], x_a[uni_idx]

    s_a = None
    if s_col and s_col in df_src.columns:
        s_raw   = pd.to_numeric(df_src[s_col], errors="coerce")
        s_aligned = s_raw[mask].values.astype(float)[:n][sort_idx][uni_idx]
        s_a     = np.nan_to_num(s_aligned, nan=0.0)
    return t_a, x_a, s_a


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 17. TAB: DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def tab_data():
    st.markdown(f"## {t('dt_title')}")

    # ── Box 1: Upload (collapsed by default) ─────────────────
    with st.expander("📂 " + t("dt_upload_label"), expanded=False):
        st.caption(t("dt_upload_types"))
        col_up, col_hr = st.columns([3, 1])
        with col_up:
            ufs = st.file_uploader(
                t("dt_upload_label"),
                type=["csv", "xlsx", "xls", "txt"],
                accept_multiple_files=True,
                label_visibility="collapsed",
                key="uploader_multiplo",
            )
        with col_hr:
            hr = st.number_input(t("dt_header_row"), min_value=1, max_value=50, value=1,
                                  key="header_row_num")
            st.caption(t("dt_header_note"))

        if ufs:
            dfs, names = [], []
            for uf in ufs:
                df_i = load_file(uf.read(), uf.name, header_row=int(hr) - 1)
                if df_i is not None:
                    dfs.append(df_i)
                    names.append(uf.name)

            if dfs:
                # ── Bug 7: detect new uploads and reset analysis ──────
                current_sig = frozenset(names)
                prev_sig    = st.session_state.get("_uploaded_file_sig", frozenset())
                if current_sig != prev_sig:
                    st.session_state.selected_models  = []
                    st.session_state.fit_results       = {}
                    st.session_state.headers           = {}
                    st.session_state.df_clean          = None
                    st.session_state.excluded_cols     = []
                    st.session_state._uploaded_file_sig = current_sig

                if len(dfs) == 1:
                    # Single file — use directly
                    st.session_state.df_raw = dfs[0]
                    st.success(f"✅ {names[0]} — {dfs[0].shape[0]} × {dfs[0].shape[1]}")

                else:
                    # ── Multiple files: merge all ─────────────────────
                    col_sets     = [frozenset(df.columns.astype(str)) for df in dfs]
                    all_identical = len(set(col_sets)) == 1

                    if all_identical:
                        # Same structure → stack vertically
                        combined = pd.concat(dfs, ignore_index=True)
                        st.session_state.df_raw = combined
                        st.success(
                            f"✅ {len(dfs)} arquivos com colunas idênticas "
                            f"→ empilhados: {combined.shape[0]} × {combined.shape[1]} linhas")
                    else:
                        # Different columns → outer merge on first column (time)
                        # Use the file with the widest time range as the base
                        ranges = []
                        for df_i in dfs:
                            try:
                                t_vals = pd.to_numeric(
                                    df_i.iloc[:, 0], errors="coerce").dropna()
                                ranges.append(float(t_vals.max() - t_vals.min())
                                              if len(t_vals) > 1 else 0.0)
                            except Exception:
                                ranges.append(0.0)

                        order     = sorted(range(len(dfs)), key=lambda i: -ranges[i])
                        base_df   = dfs[order[0]].copy()
                        base_t    = str(base_df.columns[0])   # time column name
                        result    = base_df

                        for idx in order[1:]:
                            df_other = dfs[idx].copy()
                            short    = (names[idx]
                                        .replace(".csv","").replace(".xlsx","")
                                        .replace(".xls","").replace(".txt",""))[:10]
                            t_other  = str(df_other.columns[0])

                            # Rename non-time columns with filename prefix
                            rmap = {c: f"{c}_{short}"
                                    for c in df_other.columns if c != t_other}
                            df_other = df_other.rename(columns=rmap)
                            # Align time column name to base
                            if t_other != base_t:
                                df_other = df_other.rename(
                                    columns={t_other: base_t})

                            # Outer merge on the time column
                            result = pd.merge(result, df_other,
                                              on=base_t, how="outer")
                            result = (result.sort_values(base_t)
                                            .reset_index(drop=True))

                        st.session_state.df_raw = result
                        st.info(
                            f"📂 {len(dfs)} {t('dt_files_merge')} "
                            f"→ {result.shape[0]} × {result.shape[1]}")
                        # Show column mapping per file
                        for i, (nm, df_i) in enumerate(zip(names, dfs)):
                            icon  = "🏆" if i == order[0] else "📄"
                            cols_ = list(df_i.columns.astype(str))
                            st.caption(f"{icon} {nm}: {cols_}")

                # Apply any active column exclusions
                if st.session_state.df_raw is not None:
                    excl = st.session_state.excluded_cols
                    st.session_state.df = st.session_state.df_raw.drop(
                        columns=[c for c in excl
                                 if c in st.session_state.df_raw.columns],
                        errors="ignore")

        # ── Manual data toggle  (Original, retornar se a alteração não for bem sucedida)────────────────────────────────
        #st.markdown("---")
        #prev_manual = st.session_state.use_manual
        #st.session_state.use_manual = st.checkbox(t("dt_manual_title"),
                                                    #value=st.session_state.use_manual)

        # — Manual data toggle (nova implementação)——————————————————
        st.markdown("---")
        prev_manual = st.session_state.get("use_manual", False) # Pega o valor atual com segurança

        # Adicionamos a key="use_manual_multiplo" e lemos o valor atual
        st.session_state.use_manual = st.checkbox(
        t("dt_manual_title"), 
        value=st.session_state.get("use_manual", False),
        key="use_manual_multiplo"
)                                            
        # Clear file data when manual is activated
        if st.session_state.use_manual and not prev_manual:
            st.session_state.df = None
            st.session_state.df_raw = None
            st.session_state.df_clean = None
            st.session_state.excluded_cols = []
            st.session_state.headers = {}
            st.rerun()

        if st.session_state.use_manual:
            st.caption(t("dt_manual_note"))

            # ── Toolbar ──────────────────────────────────────────
            mc = st.columns(5)
            if mc[0].button(t("dt_manual_add_col"), key="mc_add_col"):
                new_name = f"Col{len(st.session_state.manual_cols)}"
                st.session_state.manual_cols.append(new_name)
                # CRITICAL: keep each row in sync with col count
                st.session_state.manual_rows = [
                    list(r) + [""] for r in st.session_state.manual_rows
                ]
                st.rerun()
            if mc[1].button(t("dt_manual_rem_col"), key="mc_rem_col") \
                    and len(st.session_state.manual_cols) > 2:
                st.session_state.manual_cols.pop()
                st.session_state.manual_rows = [
                    list(r)[:-1] for r in st.session_state.manual_rows
                ]
                st.rerun()
            if mc[2].button(t("dt_manual_add_row"), key="mc_add_row"):
                st.session_state.manual_rows.append(
                    [""] * len(st.session_state.manual_cols))
                st.rerun()
            if mc[3].button(t("dt_manual_rem_row"), key="mc_rem_row") \
                    and len(st.session_state.manual_rows) > 1:
                st.session_state.manual_rows.pop()
                st.rerun()
            if mc[4].button(t("dt_manual_clear"), key="mc_clear"):
                n = len(st.session_state.manual_cols)
                st.session_state.manual_rows = [[""] * n for _ in range(3)]
                st.rerun()

            # ── Inline header renaming ────────────────────────────
            st.caption(t("dt_col_rename_hint"))
            n_cols_now  = len(st.session_state.manual_cols)
            hdr_cols    = st.columns(max(n_cols_now, 1))
            new_names   = list(st.session_state.manual_cols)
            for ci, (hcol, old_name) in enumerate(
                    zip(hdr_cols, st.session_state.manual_cols)):
                renamed = hcol.text_input(
                    f"H{ci}", value=old_name,
                    key=f"hdr_rename_{ci}",
                    label_visibility="collapsed",
                    placeholder=f"Coluna {ci+1}",
                )
                new_names[ci] = renamed.strip() or old_name
            if new_names != list(st.session_state.manual_cols):
                st.session_state.manual_cols = new_names
                st.rerun()

            # ── Editable table ────────────────────────────────────
            # Guarantee rows are aligned with cols before building df
            n_c = len(st.session_state.manual_cols)
            aligned = []
            for r in st.session_state.manual_rows:
                row = list(r)
                if len(row) < n_c:
                    row += [""] * (n_c - len(row))
                elif len(row) > n_c:
                    row = row[:n_c]
                aligned.append(row)
            st.session_state.manual_rows = aligned

            man_df = pd.DataFrame(aligned, columns=st.session_state.manual_cols)
            edited = st.data_editor(
                man_df, use_container_width=True,
                num_rows="dynamic", key="manual_editor")

            # Sync editor changes back to session state
            st.session_state.manual_rows = edited.values.tolist()

            # Coerce numerics
            for c in edited.columns:
                try:
                    edited[c] = pd.to_numeric(
                        edited[c].astype(str).str.replace(",", "."),
                        errors="coerce")
                except Exception:
                    pass

            if st.button(f"✅ {t('dt_manual_add_data')}", key="apply_manual"):
                st.session_state.df_raw = edited.copy()
                st.session_state.df     = edited.copy()
                st.session_state.excluded_cols = []
                st.session_state.headers       = {}  # reset mapping for fresh column names
                st.success(t("dt_data_applied"))
                st.rerun()

    # ── Variable mapping — tag-input style ───────────────────
    if st.session_state.df is not None:
        df = st.session_state.df
        none_label = t("dt_none")
        cols = [none_label] + list(df.columns.astype(str))

        with st.expander(f"🏷️ {t('dt_mapping_title')}", expanded=True):
            st.caption(t("dt_mapping_note"))
            st.markdown("""
<style>
.cat-row{
  display:flex; align-items:center; gap:4px;
  margin:8px 0 2px 2px; font-size:.82rem; line-height:1;
}
.cat-icon{ font-size:.92rem; }
.cat-req { color:var(--acc); font-weight:700; }
/* Collapse extra padding around text inputs inside the mapping grid */
div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"]
  > div.stTextInput > div {
  padding-top: 0 !important;
  margin-top: 0 !important;
}
div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"]
  > div.stTextInput input {
  border:1px solid var(--brd) !important;
  background:var(--bg2) !important;
  border-radius:8px !important;
  padding:3px 10px !important;
  font-size:.8rem !important;
  font-weight:600 !important;
  height:28px !important;
  min-height:28px !important;
}
div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"]
  > div.stSelectbox, div.stMultiSelect {
  margin-top: 2px !important;
}
</style>
""", unsafe_allow_html=True)
            lang = st.session_state.get("lang", "pt")
            var_defs = [
                ("time",      "⏱",  f'{t("dt_time_lbl")}',      True,  False),
                ("biomass",   "🦠", f'{t("dt_biomass_lbl")}',   True,  True),
                ("substrate", "🍬",  f'{t("dt_substrate_lbl")}', False, True),
                ("ph",        "⚗️",  f'{t("dt_ph_lbl")}',        False, False),
                ("product",   "🧪",  f'{t("dt_product_lbl")}',   False, True),
                ("drymass",   "⚖️",  f'{t("dt_drymass_lbl")}',   False, False),
            ]

            col_opts  = list(df.columns.astype(str))
            cols_none = [none_label] + col_opts

            if "category_labels" not in st.session_state:
                st.session_state.category_labels = {}

            grid_cols = st.columns(3)
            for idx, (var_key, icon, label_key, required, multi) in enumerate(var_defs):
                gcol = grid_cols[idx % 3]
                default_label = t(label_key)

                if multi:
                    curr_multi = _hdr_list(var_key)
                    curr_multi = [c for c in curr_multi if c in col_opts]
                    mapped = len(curr_multi) > 0
                else:
                    current = st.session_state.headers.get(var_key, none_label)
                    if isinstance(current, list):
                        current = current[0] if current else none_label
                    if current not in cols_none:
                        current = none_label
                    mapped = current != none_label

                req_badge  = '<span class="cat-req">*</span>' if required else ""
                # Single-column layout — no sub-split, no whitespace gap
                gcol.markdown(
                    f'<div class="cat-row">'
                    f'<span class="cat-icon">{icon}</span>'
                    f'{req_badge}'
                    f'</div>',
                    unsafe_allow_html=True)

                stored_label = st.session_state.category_labels.get(var_key, default_label)
                edited_label = gcol.text_input(
                    f"catlabel_{var_key}",
                    value=stored_label,
                    key=f"catlabel_{var_key}",
                    label_visibility="collapsed",
                    placeholder=default_label,
                    help=t("dt_rename_category_hint"),
                )
                final_label = edited_label.strip() or default_label
                st.session_state.category_labels[var_key] = final_label

                if multi:
                    sel = gcol.multiselect(
                        final_label, col_opts,
                        default=curr_multi,
                        key=f"hdr_{var_key}",
                        label_visibility="collapsed",
                        placeholder=t("dt_none"),
                    )
                    st.session_state.headers[var_key] = sel if sel else None
                else:
                    sel = gcol.selectbox(
                        final_label, cols_none,
                        index=cols_none.index(current),
                        key=f"hdr_{var_key}",
                        label_visibility="collapsed",
                    )
                    st.session_state.headers[var_key] = sel if sel != none_label else None

    # ── Box 2: Data Summary ───────────────────────────────────
    if st.session_state.df is not None:
        df    = st.session_state.df
        t_col = st.session_state.headers.get("time")
        x_col = _hdr_primary("biomass")   # primary biomass for validation checks

        with st.expander(f"📋 {t('dt_summary_title')}", expanded=True):
            if df.isnull().any().any():
                st.warning(t("dt_nan_warn"))
            if t_col and t_col in df.columns:
                t_s = pd.to_numeric(df[t_col], errors="coerce")
                if (t_s < 0).any():
                    st.error(t("dt_neg_time"))
                if t_s.dropna().duplicated().any():
                    st.warning(t("dt_dup_time"))

            # Column exclusion — filter stale defaults first
            all_cols = list(df.columns.astype(str))
            if all_cols:
                # Only keep previously-excluded cols that still exist in this df
                valid_defaults = [c for c in st.session_state.excluded_cols
                                  if c in all_cols]
                # Sync session state if stale items were removed
                if set(valid_defaults) != set(st.session_state.excluded_cols):
                    st.session_state.excluded_cols = valid_defaults

                excl_sel = st.multiselect(
                    t("dt_excl_cols"),
                    all_cols,
                    default=valid_defaults,
                    help=t("dt_excl_note"),
                    key="excl_cols_sel",
                )
                if set(excl_sel) != set(st.session_state.excluded_cols):
                    st.session_state.excluded_cols = excl_sel
                    if st.session_state.df_raw is not None:
                        st.session_state.df = st.session_state.df_raw.drop(
                            columns=[c for c in excl_sel if c in st.session_state.df_raw.columns],
                            errors="ignore")
                    # Clear headers pointing to excluded cols
                    for hk, hv in list(st.session_state.headers.items()):
                        if hv in excl_sel:
                            st.session_state.headers[hk] = None
                    st.rerun()

            def _highlight_neg(val):
                try:
                    return "background-color:#6b1212;color:#ffcccc" if float(val) < 0 else ""
                except Exception:
                    return ""

            st.dataframe(df.head(50).style.map(_highlight_neg),
                         use_container_width=True, height=220)
            lang = st.session_state.get("lang", "pt")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f'{t("tm_metric_line")}', df.shape[0])
            m2.metric(f'{t("tm_metric_col")}', df.shape[1])
            m3.metric("NaN", f"{df.isnull().mean().mean()*100:.1f}%")
            m4.metric(f'{t("tm_metric_model")}', _count_available_models())

    # ── Box 3: Scatter preview ────────────────────────────────
    t_col_p  = st.session_state.headers.get("time")
    x_cols_p = _hdr_list("biomass")
    if st.session_state.df is not None and t_col_p and x_cols_p:
        df = st.session_state.df
        try:
            t_arr_full = pd.to_numeric(df[t_col_p], errors="coerce")

            with st.expander(f"📉 {t('dt_preview_title')}", expanded=True):
                use_log = st.checkbox(t("dt_preview_log"), value=True, key="prev_log")

                CHART_COLORS = [
                    "#00c8b4","#f97316","#a855f7","#22c55e","#f43f5e",
                    "#38bdf8","#fbbf24","#e879f9","#4ade80","#fb7185",
                    "#34d399","#c084fc","#fdba74","#67e8f9","#fde68a",
                ]
                fig = go.Figure()

                # Plot every mapped biomass column as a separate series
                for bi, b_col in enumerate(x_cols_p):
                    if b_col not in df.columns:
                        continue
                    b_vals = pd.to_numeric(df[b_col], errors="coerce")
                    mask_b = t_arr_full.notna() & b_vals.notna() & (b_vals > 0)
                    if not mask_b.any():
                        continue
                    color = CHART_COLORS[bi % len(CHART_COLORS)]
                    fig.add_trace(go.Scatter(
                        x=t_arr_full[mask_b].values,
                        y=b_vals[mask_b].values,
                        mode="markers+lines",
                        marker=dict(color=color, size=7),
                        line=dict(color=color, width=1.5, dash="dot"),
                        name=str(b_col)))

                # Substrate on secondary Y axis
                s_col_p = _hdr_primary("substrate")
                if s_col_p and s_col_p in df.columns:
                    s_vals  = pd.to_numeric(df[s_col_p], errors="coerce")
                    mask_s  = t_arr_full.notna() & s_vals.notna()
                    if mask_s.any():
                        fig.add_trace(go.Scatter(
                            x=t_arr_full[mask_s].values,
                            y=s_vals[mask_s].values,
                            mode="markers+lines", yaxis="y2",
                            marker=dict(color="#f97316", size=6, symbol="diamond"),
                            line=dict(color="#f97316", width=1, dash="dot"),
                            name=str(s_col_p)))
                        fig.update_layout(
                            yaxis2=dict(overlaying="y", side="right",
                                        showgrid=False, color="#f97316",
                                        tickformat=".4g", exponentformat="none"))

                # Build df_clean from the first valid biomass mask
                if x_cols_p:
                    x0_vals = pd.to_numeric(df[x_cols_p[0]], errors="coerce")
                    clean_mask = (t_arr_full.notna() & x0_vals.notna()
                                  & (t_arr_full >= 0) & (x0_vals > 0))
                    st.session_state.df_clean = df[clean_mask].reset_index(drop=True)

                fig.update_layout(
                    yaxis_type="log" if use_log else "linear",
                    yaxis=dict(
                        color="#8b949e", gridcolor="#30363d",
                        title=" / ".join(x_cols_p[:3]),
                        tickformat=".4g" if not use_log else None,
                        exponentformat="none",
                    ),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(color="#8b949e", gridcolor="#30363d",
                               title=str(t_col_p)),
                    legend=dict(bgcolor="rgba(0,0,0,0)"),
                    uirevision="scatter_preview",   # keeps zoom on rerun
                    margin=dict(l=40, r=40, t=20, b=40), height=340,
                )
                st.plotly_chart(fig, use_container_width=True,
                                config={"scrollZoom": True})
        except Exception as e:
            st.error(f"{t('dt_chart_error')} {e}")

    # ── Box 4+5: Available models & Select ────────────────────
    n_avail = _count_available_models()
    ca, cb = st.columns([2, 1])
    with ca:
        st.markdown(f"""
<div class="bio-card">
  <h4>🔢 {t("dt_avail_title")}</h4>
  <div class="author">{t("dt_avail_note")}</div>
  <div style="font-size:2rem;font-weight:900;color:var(--acc)">{n_avail}</div>
  <div style="font-size:.75rem;color:var(--fg2)">{t("dt_avail_prep")} {len(ALL_MODELS)}{t("dt_avail_complement0")} · + {len(ALL_TOOLS)} {t("dt_avail_complement1")}</div>
</div>
""", unsafe_allow_html=True)
    with cb:
        can_select = bool(st.session_state.headers.get("time") and
                          st.session_state.headers.get("biomass"))
        if can_select:
            if st.button(t("dt_select_btn"), use_container_width=True, type="primary"):
                st.session_state.tab = "models"
                st.rerun()
        else:
            st.info(t("dt_select_disabled"))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 18. TAB: MODELS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def tab_models():
    st.markdown(f"## {t('md_title')}")

    # Action bar
    ab1, ab2, ab3 = st.columns([1, 1, 2])
    if ab1.button(t("md_back")):
        st.session_state.tab = "data"; st.rerun()

    n_sel = len(st.session_state.selected_models)
    if n_sel > 0:
        ab2.markdown(
            f'<div style="padding:7px 0;font-size:.82rem;color:var(--acc)">'
            f'✅ {n_sel} {t("md_selected_count")}</div>',
            unsafe_allow_html=True)
        if ab3.button(t("md_go_results"), type="primary",
                      use_container_width=True, key="go_results_top"):
            st.session_state.tab = "results"; st.rerun()

    st.markdown("---")

    # Inline CSS for model cards with selection border
    st.markdown("""
<style>
.model-card{
  background:var(--bgc);border:2px solid var(--brd);border-radius:10px;
  padding:12px 14px;margin-bottom:6px;transition:.18s;cursor:default;
}
.model-card.sel{border-color:var(--acc);box-shadow:0 0 0 2px var(--acc)40;}
.model-card.dis{opacity:.4;filter:grayscale(60%);pointer-events:none;}
.model-card h5{margin:0 0 3px;font-size:.88rem;color:var(--acc);}
.model-card .auth{font-size:.75rem;color:var(--fg2);margin-bottom:4px;}
.model-card .catg{font-size:.72rem;color:var(--acc2);text-transform:uppercase;letter-spacing:.04em;}
</style>
""", unsafe_allow_html=True)

    for box in BOXES:
        box_title = _box_title(box)
        with st.expander(box_title, expanded=False):
            n_cols = min(len(box["models"]), 3)
            cols = st.columns(n_cols)
            for i, model in enumerate(box["models"]):
                avail  = _model_available(model)
                is_sel = model["key"] in st.session_state.selected_models
                card_cls = ("sel" if is_sel else "") + (" dis" if not avail else "")
                col = cols[i % n_cols]
                with col:
                    lang = st.session_state.get("lang", "pt")
                    col.markdown(f"""
<div class="model-card {card_cls}">
  <h5>{_model_name(model)}</h5>
  <div class="auth">{model['author'].get(lang,model['author']['en'])}</div>
  <div class="catg">{_model_category(model)}</div>
</div>
""", unsafe_allow_html=True)
                    col.latex(model["latex"])
                    col.caption("Params: " + " · ".join(model["params"]))
                    if avail:
                        btn_lbl = t("md_deselect") if is_sel else t("md_select_this")
                        btn_type = "primary" if is_sel else "secondary"
                        if col.button(btn_lbl, key=f"sel_{model['key']}",
                                      use_container_width=True, type=btn_type):
                            sms = st.session_state.selected_models
                            if model["key"] in sms:
                                sms.remove(model["key"])
                            else:
                                sms.append(model["key"])
                            # clear cached fits when selection changes
                            st.session_state.fit_results = {}
                            # ── Bug 3: no auto-redirect; user navigates manually ──
                            st.rerun()
                    else:
                        col.caption(f"⚠️ {t('md_disabled_tip')}" +
                                    ", ".join(model["requires"]))

    st.markdown("---")
    if st.session_state.selected_models:
        if st.button(t("md_go_results"), type="primary",
                     use_container_width=True, key="go_results_bottom"):
            st.session_state.tab = "results"; st.rerun()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 19. TAB: RESULTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _run_fit_for(model_key, t_arr, x_arr, s_arr, opt_settings):
    try:
        popt, y_pred, _ = fit_model(model_key, t_arr, x_arr, s_arr, opt_settings)
        n_k  = len(popt)
        _r2  = r2_adj(x_arr, y_pred, n_k)
        return {
            "popt":      popt, "y_pred":    y_pred,
            "t":         t_arr, "X":         x_arr, "S":         s_arr,
            "X0_fit":    float(x_arr[0]),
            "S0_fit":    float(s_arr[0]) if s_arr is not None else None,
            "model_key": model_key,
            "rmse":      rmse(x_arr, y_pred), "r2adj": _r2,
            "aic":       aic(x_arr, y_pred, n_k), "bic": bic(x_arr, y_pred, n_k),
            "phases":    detect_phases(t_arr, x_arr), "n_params": n_k,
        }
    except Exception as ex:
        return {"error": str(ex), "model_key": model_key}


def _render_single_result(fr, model_key, t_col, x_col):
    from scipy import stats as _stats
    m_meta = ALL_MODELS.get(model_key, {})
    t_d, X_d, y_p = fr["t"], fr["X"], fr["y_pred"]
    resid_std = float(np.std(X_d - y_p))

    # Metrics
    fit_lbl, fit_col, fit_ico = fitness_label(fr["r2adj"])
    mc = st.columns(5)
    mc[0].metric(t("rs_rmse"),  f"{fr['rmse']:.5f}")
    mc[1].metric(t("rs_r2adj"), f"{fr['r2adj']:.5f}")
    mc[2].metric(t("rs_aic"),   f"{fr['aic']:.3f}")
    mc[3].metric(t("rs_bic"),   f"{fr['bic']:.3f}")
    mc[4].markdown(
        f'<div class="metric-box" style="border-color:{fit_col};margin-top:4px">'
        f'<div class="val" style="color:{fit_col};font-size:1rem">{fit_ico} {fit_lbl}</div>'
        f'<div class="lbl">{t("fit_quality")}</div></div>',
        unsafe_allow_html=True)

    # Computed kinetic parameters
    mu_all   = np.gradient(np.log(np.maximum(y_p, 1e-9)), t_d)
    mu_max_v = float(np.max(mu_all))
    mu_avg_v = float(np.mean(mu_all))
    td_v     = np.log(2) / mu_max_v if mu_max_v > 1e-6 else float("nan")
    X0_v, Xm_v = float(X_d[0]), float(np.max(y_p))
    S_data   = fr.get("S")
    yield_obs = ((Xm_v - X0_v) / (-S_data[-1] + S_data[0] + 1e-9)
                 if S_data is not None else None)

    st.markdown(f"##### {t('rs_params_title')}")
    pb = st.columns(4)
    pb[0].markdown(f'<div class="metric-box"><div class="val">{mu_max_v:.4f}</div><div class="lbl">{t("rs_mu_max")} h⁻¹</div></div>', unsafe_allow_html=True)
    pb[1].markdown(f'<div class="metric-box"><div class="val">{mu_avg_v:.4f}</div><div class="lbl">{t("rs_mu_avg")} h⁻¹</div></div>', unsafe_allow_html=True)
    td_str = f"{td_v:.3f}" if not np.isnan(td_v) else "—"
    pb[2].markdown(f'<div class="metric-box"><div class="val">{td_str}</div><div class="lbl">{t("rs_td")} h</div></div>', unsafe_allow_html=True)
    yobs_str = f"{yield_obs:.3f}" if yield_obs is not None else "—"
    pb[3].markdown(f'<div class="metric-box"><div class="val">{yobs_str}</div><div class="lbl">{t("rs_yield_obs")} g·g⁻¹</div></div>', unsafe_allow_html=True)

    param_names = m_meta.get("params", [f"p{i}" for i in range(len(fr["popt"]))])
    with st.expander(t("rs_params_adjusted"), expanded=False):
        lang = st.session_state.get("lang", "pt")
        param_rows = [{t("rs_named_col1"): pn, t("rs_named_col2"): f"{pv:.6g}"}
                      for pn, pv in zip(param_names, fr["popt"])]
        if param_rows:
            st.dataframe(pd.DataFrame(param_rows), use_container_width=True, hide_index=True)

    # Fitted curve
    st.markdown(f"##### 📈 {t('rs_fit_chart')}")
    t_fine = np.linspace(t_d[0], t_d[-1], 400)
    y_fine = predict_for_plot(model_key, fr["popt"], t_fine,
                              X0_data=fr.get("X0_fit"), S0=fr.get("S0_fit"))
    fig_fit = go.Figure()
    fig_fit.add_trace(go.Scatter(
        x=np.concatenate([t_fine, t_fine[::-1]]),
        y=np.concatenate([y_fine + resid_std, (y_fine - resid_std)[::-1]]),
        fill="toself", fillcolor="rgba(0,200,180,0.08)",
        line=dict(color="rgba(0,0,0,0)"), name="±1σ", hoverinfo="skip"))
    fig_fit.add_trace(go.Scatter(x=t_fine, y=y_fine, mode="lines", name="Modelo",
                                  line=dict(color="#00c8b4", width=2.5)))
    fig_fit.add_trace(go.Scatter(x=t_d, y=X_d, mode="markers", name="Dados obs.",
                                  marker=dict(color="#f97316", size=8,
                                              line=dict(color="#fff", width=1))))
    fig_fit.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(color="#8b949e", gridcolor="#30363d", title=str(t_col)),
        yaxis=dict(color="#8b949e", gridcolor="#30363d", title=str(x_col),
                   tickformat=".4g", exponentformat="none"),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.08),
        uirevision="fit_chart",
        margin=dict(l=40, r=20, t=30, b=40), height=340,
    )
    st.plotly_chart(fig_fit, use_container_width=True, config={"scrollZoom": True})

    # Residuals
    resid = X_d - y_p
    norm_msg = ""
    try:
        if len(resid) >= 3:
            fn = _stats.shapiro if len(resid) <= 50 else _stats.normaltest
            nm = "Shapiro-Wilk" if len(resid) <= 50 else "D'Agostino"
            _, pval = fn(resid)
            norm_msg = (f"{t('rs_residuals_normal')} ({nm} p={pval:.4f})" if pval > 0.05
                        else f"{t('rs_residuals_abnormal')} ({nm} p={pval:.4f})")
    except Exception:
        pass
    st.markdown(f"##### 📉 {t('rs_resid_chart')}")
    fig_res = make_subplots(rows=1, cols=2,
                             subplot_titles=[f"{t('rs_sub_plot1')}", f"{t('rs_sub_plot2')}"])
    rcol = ["#f85149" if abs(r) > 2*resid_std else "#7c3aed" for r in resid]
    fig_res.add_trace(go.Scatter(x=t_d, y=resid, mode="markers",
                                  marker=dict(color=rcol, size=6, opacity=.85)), row=1, col=1)
    fig_res.add_hline(y=0, line_dash="dot", line_color="#8b949e", row=1, col=1)
    fig_res.add_hline(y= resid_std, line_dash="dash", line_color="#e3b341", line_width=.8, row=1, col=1)
    fig_res.add_hline(y=-resid_std, line_dash="dash", line_color="#e3b341", line_width=.8, row=1, col=1)
    fig_res.add_trace(go.Histogram(x=resid, nbinsx=min(15, max(5, len(resid)//3)),
                                    marker_color="#00c8b4", opacity=.7), row=1, col=2)
    try:
        xn = np.linspace(resid.min(), resid.max(), 80)
        yn = _stats.norm.pdf(xn, resid.mean(), resid.std())
        yn_s = yn * len(resid) * (resid.max()-resid.min()) / min(15, max(5, len(resid)//3))
        fig_res.add_trace(go.Scatter(x=xn, y=yn_s, mode="lines",
                                      line=dict(color="#f97316", width=2)), row=1, col=2)
    except Exception:
        pass
    fig_res.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           uirevision="residuals",
                           showlegend=False, margin=dict(l=40, r=20, t=40, b=40), height=260)
    st.plotly_chart(fig_res, use_container_width=True)
    if norm_msg:
        st.caption(norm_msg)

    # Growth phases
    phases = fr.get("phases", {})
    phase_rows = []
    if phases:
        st.markdown(f"##### 🌱 {t('rs_phases_title')}")
        phase_map = [("lag","rs_phase_lag","phase-lag"),("acc","rs_phase_acc","phase-acc"),
                     ("exp","rs_phase_exp","phase-exp"),("dec","rs_phase_dec","phase-dec"),
                     ("stat","rs_phase_stat","phase-stat"),("decl","rs_phase_decl","phase-decl")]
        pills = ""
        for pk, pk_key, pcls in phase_map:
            if pk in phases:
                t0, t1, pts = phases[pk]
                dur = t1 - t0
                pills += f'<span class="phase-pill {pcls}">{t(pk_key)}: {t0:.1f}–{t1:.1f} h</span>'
                phase_rows.append({
                    "Fase": t(pk_key),
                    t("rs_interval"): f"{t0:.2f} – {t1:.2f}",
                    t("rs_points"):   len(pts),
                    t("rs_duration"): f"{dur:.2f}",
                })
        st.markdown(f'<div style="margin-bottom:10px">{pills}</div>', unsafe_allow_html=True)
        if phase_rows:
            st.dataframe(pd.DataFrame(phase_rows), use_container_width=True, hide_index=True)

    return {
        "mu_max": mu_max_v, "mu_avg": mu_avg_v, "td": td_v,
        "X0": X0_v, "Xm": Xm_v, "yield_obs": yield_obs,
        "resid": resid, "t_d": t_d, "X_d": X_d, "y_p": y_p,
        "phase_rows": phase_rows,
    }


def tab_results():
    st.markdown(f"## {t('rs_title')}")
    ab1, _ = st.columns([1, 3])
    if ab1.button(t("rs_back")):
        st.session_state.tab = "models"; st.rerun()

    if not st.session_state.selected_models:
        st.info(t("rs_no_model")); return
    df_src = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df
    if df_src is None:
        st.info(t("rs_no_data")); return

    h      = st.session_state.headers
    t_col  = h.get("time")
    x_cols = _hdr_list("biomass")          # may be multiple after merge
    s_col  = _hdr_primary("substrate")     # use primary substrate for all fits

    if not t_col or not x_cols:
        st.info(t("rs_no_data")); return

    # Validate columns exist in the source df
    x_cols = [c for c in x_cols if c in df_src.columns]
    if not x_cols:
        st.info(t("rs_no_data")); return

    # Use the first biomass column for the optimization settings UI
    x_col_primary = x_cols[0]
    t_arr_ui, x_arr_ui, s_arr_ui = _extract_sorted_series(
        df_src, t_col, x_col_primary, s_col)

    # Optimization settings
    with st.expander(f"⚙️ {t('rs_opt_title')}", expanded=True):
        oc1, oc2, oc3 = st.columns(3)
        est_mode = oc1.radio(t("rs_est_mode"),
                              [t("rs_est_auto"), t("rs_est_manual")],
                              horizontal=True, key="est_mode_r")
        speed = oc2.select_slider("Velocidade",
                    [t("rs_speed_fast"), t("rs_speed_std"), t("rs_speed_prec")],
                    value=t("rs_speed_std"), key="speed_sl")
        restarts = oc3.selectbox(t("rs_restarts"),
                    [t("rs_restart_std"), t("rs_restart_rob"), t("rs_restart_glb")],
                    key="restarts_sel")
        speed_map = {t("rs_speed_fast"): (1e-4, 200),
                     t("rs_speed_std"):  (1e-6, 1000),
                     t("rs_speed_prec"): (1e-8, 5000)}
        tol, max_it = speed_map[speed]

        if len(st.session_state.selected_models) > 1:
            with st.expander(t("rs_tol_per_model"), expanded=False):
                st.caption(t("rs_tol_hint"))
                per_tol = {}
                # 2 colunas com nomes completos + quebra de linha no label
                n_models = len(st.session_state.selected_models)
                ptcols   = st.columns(2)
                for idx, mk in enumerate(st.session_state.selected_models):
                    m_info  = ALL_MODELS.get(mk, {})
                    # Full name: "Modelo (Autor)"
                    lang = st.session_state.get("lang", "pt")
                    full_name = _model_name(m_info) or mk
                    author    = m_info.get(m_info["author"].get(lang, m_info["author"]["en"]), "")
                    col = ptcols[idx % 2]
                    col.markdown(
                        f'<div style="font-size:.78rem;color:var(--acc);'
                        f'font-weight:600;margin-bottom:2px">{full_name}</div>'
                        f'<div style="font-size:.7rem;color:var(--fg2);'
                        f'margin-bottom:6px">{author}</div>',
                        unsafe_allow_html=True)
                    per_tol[mk] = col.number_input(
                        "Tolerância",
                        value=tol,
                        format="%.2e",
                        key=f"ptol_{mk}",
                        min_value=1e-12,
                        max_value=1e-1,
                        label_visibility="collapsed",
                    )
                    # Visual separator inside column
                    col.markdown(
                        '<div style="height:1px;background:var(--brd);'
                        'margin:8px 0 10px"></div>',
                        unsafe_allow_html=True)
                st.session_state.opt["per_model_tol"] = per_tol

        if est_mode == t("rs_est_manual"):
            for mk in st.session_state.selected_models:
                p0 = _auto_p0(mk, t_arr_ui, x_arr_ui, s_arr_ui)
                pn = ALL_MODELS.get(mk, {}).get("params", [])
                st.caption(f"**{mk}**")
                mg_cols = st.columns(min(len(p0), 4))
                for i, (pname, pval) in enumerate(zip(pn, p0)):
                    v = mg_cols[i % 4].number_input(pname, value=float(pval),
                                                     format="%.4g", key=f"ig_{mk}_{i}")
                    st.session_state.opt["initial_guess"][f"{mk}_{pname}"] = v

        st.session_state.opt.update({"tolerance": tol, "max_iter": max_it,
                                      "estimate": est_mode, "restarts": restarts})

    if st.button(t("rs_run_btn"), type="primary", use_container_width=True):
        results = {}
        with st.spinner(t("rs_running")):
            for x_col_i in x_cols:
                t_a, x_a, s_a = _extract_sorted_series(
                    df_src, t_col, x_col_i, s_col)
                for mk in st.session_state.selected_models:
                    mk_opt = dict(st.session_state.opt)
                    per_tol = st.session_state.opt.get("per_model_tol", {})
                    if mk in per_tol:
                        mk_opt["tolerance"] = per_tol[mk]
                    # Key encodes both model and biomass series
                    rkey = f"{mk}__{x_col_i}" if len(x_cols) > 1 else mk
                    res  = _run_fit_for(mk, t_a, x_a, s_a, mk_opt)
                    res["series"] = x_col_i
                    results[rkey] = res
        st.session_state.fit_results = results

    if not st.session_state.fit_results:
        st.info(t("rs_run_hint"))
        return

    fit_results = st.session_state.fit_results

    # Sort valid results by R²adj descending
    valid_keys        = [k for k, v in fit_results.items() if "error" not in v and v]
    valid_keys_sorted = sorted(valid_keys,
                               key=lambda k: fit_results[k].get("r2adj", -999),
                               reverse=True)
    error_keys        = [k for k in fit_results if k not in valid_keys]

    if len(valid_keys) > 1:
        st.markdown(f"### 📊 {t('rs_compare_title')}")
        st.caption(t("rs_compare_note"))

        # ── Ranked comparison table ───────────────────────────
        comp_rows = []
        for rank, rkey in enumerate(valid_keys_sorted + error_keys, 1):
            fr      = fit_results[rkey]
            mk_disp = rkey.split("__")[0] if "__" in rkey else rkey
            series  = fr.get("series", "")
            suffix  = f"  [{series}]" if len(x_cols) > 1 and series else ""
            label   = (_model_name(ALL_MODELS.get(mk_disp, {})) or mk_disp)[:38] + suffix
            if "error" in fr:
                comp_rows.append({
                    "#": "—", t("rs_model_series_col"): label,
                    "Status": "❌ " + fr["error"][:40],
                    "RMSE": "—", "R²adj": "—", "AIC": "—", "BIC": "—"})
            else:
                fl, _, fi = fitness_label(fr["r2adj"])
                comp_rows.append({
                    "#":                      f"{rank}°",
                    t("rs_model_series_col"): label,
                    "Status":                 f"{fi} {fl}",
                    "RMSE":                   f"{fr['rmse']:.5f}",
                    "R²adj":                  f"{fr['r2adj']:.5f}",
                    "AIC":                    f"{fr['aic']:.2f}",
                    "BIC":                    f"{fr['bic']:.2f}",
                })
        if comp_rows:
            st.dataframe(pd.DataFrame(comp_rows),
                         use_container_width=True, hide_index=True)

        # ── Top-N overlay chart ───────────────────────────────
        n_valid     = len(valid_keys_sorted)
        DEFAULT_TOP = min(5, n_valid)

        if n_valid > DEFAULT_TOP:
            sl_col, inf_col = st.columns([2, 1])
            top_n = sl_col.slider(
                t("rs_top_n_label"),
                min_value=2, max_value=min(n_valid, 20),
                value=DEFAULT_TOP, key="top_n_slider")
            inf_col.markdown(
                f'<div style="padding-top:28px;font-size:.78rem;'
                f'color:var(--fg2)">Top <b style="color:var(--acc)">'
                f'{top_n}</b> / {n_valid}</div>',
                unsafe_allow_html=True)
        else:
            top_n = n_valid

        chart_keys = valid_keys_sorted[:top_n]

        COLORS_CMP = [
            "#00c8b4","#f97316","#a855f7","#22c55e","#f43f5e",
            "#38bdf8","#fbbf24","#e879f9","#4ade80","#fb7185",
            "#34d399","#c084fc","#fdba74","#67e8f9","#86efac",
            "#fde68a","#f9a8d4","#6ee7b7","#a5b4fc","#fca5a5",
        ]
        fig_cmp         = go.Figure()
        plotted_series  = set()

        for ci, rkey in enumerate(chart_keys):
            fr      = fit_results[rkey]
            mk_disp = rkey.split("__")[0] if "__" in rkey else rkey
            series  = fr.get("series", "")
            rank    = valid_keys_sorted.index(rkey) + 1
            suffix  = f" [{series}]" if len(x_cols) > 1 and series else ""
            name    = f"{rank}° {_model_name(ALL_MODELS.get(mk_disp, {}))[:22] or mk_disp}{suffix}"
            color   = COLORS_CMP[ci % len(COLORS_CMP)]
            t_fine  = np.linspace(fr["t"][0], fr["t"][-1], 400)
            y_fine  = predict_for_plot(mk_disp, fr["popt"], t_fine,
                                       X0_data=fr.get("X0_fit"),
                                       S0=fr.get("S0_fit"))
            fig_cmp.add_trace(go.Scatter(
                x=t_fine, y=y_fine, mode="lines", name=name,
                line=dict(color=color, width=2.2),
                hovertemplate=(f"<b>{name}</b><br>"
                               "t=%{x:.2f}<br>X=%{y:.4g}<extra></extra>")))
            if series not in plotted_series and fr.get("X") is not None:
                plotted_series.add(series)
                d_name = f"Dados [{series}]" if series else "Dados"
                fig_cmp.add_trace(go.Scatter(
                    x=fr["t"], y=fr["X"], mode="markers", name=d_name,
                    marker=dict(color="#f97316", size=8, symbol="circle",
                                line=dict(color="#fff", width=1)),
                    hovertemplate="<b>Dados</b><br>t=%{x:.2f}<br>"
                                  "X=%{y:.4g}<extra></extra>"))

        legend_cfg = (
            dict(bgcolor="rgba(0,0,0,0)", orientation="v",
                 x=1.01, y=1, xanchor="left", font=dict(size=10))
            if top_n > 7
            else dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.08))

        fig_cmp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="#8b949e", gridcolor="#30363d", title=str(t_col)),
            yaxis=dict(color="#8b949e", gridcolor="#30363d",
                       title=" / ".join(x_cols[:3]),
                       tickformat=".4g", exponentformat="none"),
            legend=legend_cfg,
            uirevision="comparison_chart",
            margin=dict(l=40, r=10 if top_n > 7 else 20, t=50, b=40),
            height=430 if top_n <= 7 else 470)
        st.plotly_chart(fig_cmp, use_container_width=True,
                        config={"scrollZoom": True,
                                "displayModeBar": True})
        if n_valid > top_n:
            st.caption(
                f"💡 {n_valid - top_n} {t('rs_top_n_omitted')}")
        st.divider()
        st.markdown(f"### 🔍 {t('rs_individual_results')}")

    # ── Individual results ────────────────────────────────────
    all_export_data = {}
    iter_keys = valid_keys_sorted if valid_keys_sorted else list(fit_results.keys())
    for rkey in iter_keys:
        fr      = fit_results.get(rkey, {})
        if not fr or "error" in fr:
            st.error(f"{fr.get('error','?')}"); continue
        mk_disp = rkey.split("__")[0] if "__" in rkey else rkey
        series  = fr.get("series", "")
        m_name  = _model_name(ALL_MODELS.get(mk_disp, {})) or mk_disp
        suffix  = f"  [{series}]" if len(x_cols) > 1 and series else ""
        exp_label = (f"**{m_name}**{suffix}  ·  "
                     f"R²={fr['r2adj']:.4f}  ·  RMSE={fr['rmse']:.4f}")
        if len(valid_keys) > 1:
            with st.expander(exp_label, expanded=False):
                extras = _render_single_result(
                    fr, mk_disp, t_col, series or x_cols[0])
        else:
            st.markdown(
                f"**Modelo:** `{m_name}`  —  "
                f"*{ALL_MODELS.get(mk_disp,{}).get('author','')}*")
            st.divider()
            extras = _render_single_result(
                fr, mk_disp, t_col, series or x_cols[0])
        all_export_data[rkey] = {"fr": fr, **extras}

    # Export — bottom, full-width
    if all_export_data:
        st.divider()
        st.markdown(f"### 📤 {t('rs_export_results')}")
        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            if st.button(t("rs_export_xlsx"), use_container_width=True, type="secondary"):
                buf = BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    df_src.to_excel(writer, sheet_name="Dados", index=False)
                    for rkey, ed in all_export_data.items():
                        fr      = ed["fr"]
                        mk_disp = rkey.split("__")[0] if "__" in rkey else rkey
                        series  = fr.get("series", "")
                        # Compose a safe sheet-name prefix (≤31 chars)
                        pfx = (mk_disp[:8] + ("_" + series[:6] if series else ""))[:12]
                        pd.DataFrame({
                            "Parâmetro": [t("rs_mu_max"), t("rs_mu_avg"), t("rs_td"),
                                          t("rs_x0"), t("rs_xm")],
                            "Valor": [ed["mu_max"], ed["mu_avg"], ed["td"],
                                      ed["X0"], ed["Xm"]],
                            "Unidade": ["h⁻¹","h⁻¹","h","g/L","g/L"],
                        }).to_excel(writer, sheet_name=f"Params_{pfx}", index=False)
                        pd.DataFrame({
                            "Métrica": [t("rs_rmse"), t("rs_r2adj"),
                                        t("rs_aic"),  t("rs_bic")],
                            "Valor":   [fr["rmse"], fr["r2adj"],
                                        fr["aic"],  fr["bic"]],
                        }).to_excel(writer, sheet_name=f"Metrics_{pfx}", index=False)
                        pd.DataFrame({
                            "t": ed["t_d"], "X_obs": ed["X_d"],
                            "X_pred": ed["y_p"], "Residual": ed["resid"],
                        }).to_excel(writer, sheet_name=f"Fit_{pfx}", index=False)
                        if ed.get("phase_rows"):
                            pd.DataFrame(ed["phase_rows"]).to_excel(
                                writer, sheet_name=f"Fases_{pfx}", index=False)
                st.download_button(
                    f"⬇ {t('rs_download_excel')}", data=buf.getvalue(),
                    file_name="GrowthEmulator_results.xlsx",
                    mime=("application/vnd.openxmlformats-officedocument"
                          ".spreadsheetml.sheet"),
                    use_container_width=True)

        with exp_col2:
            try:
                from fpdf import FPDF
                if st.button(t("rs_export_pdf"), use_container_width=True, type="secondary"):
                    pdf = FPDF()
                    pdf.add_page()

                    # ── Register DejaVu (UTF-8 / accents / Greek letters) ──
                    if _HAS_DEJAVU:
                        pdf.add_font("DejaVu", "",  _FONT_REG)
                        pdf.add_font("DejaVu", "B", _FONT_BOLD)
                        FONT = "DejaVu"
                    else:
                        # Fallback: Helvetica only supports latin-1 → strip
                        # unsupported chars (μ, ², °, etc.) to avoid crashes.
                        FONT = "Helvetica"

                    def _safe(txt: str) -> str:
                        if FONT == "DejaVu":
                            return txt
                        repl = {"μ": "u", "²": "2", "°": "deg", "α": "alpha",
                                "β": "beta", "ν": "v", "λ": "lambda",
                                "₀": "0", "·": "*", "—": "-", "–": "-",
                                "→": "->", "✓": "OK", "≈": "~"}
                        for k, v in repl.items():
                            txt = txt.replace(k, v)
                        return txt.encode("latin-1", "replace").decode("latin-1")

                    pdf.set_font(FONT, "B", 16)
                    pdf.cell(0, 10, _safe("GrowthEmulator v1.2 - "
                            f"{t('rs_title')}"), ln=True)
                    pdf.set_font(FONT, "", 9)
                    pdf.cell(0, 5, _safe(
                        f"{t('rs_individual_results')}: "
                        f"{len(all_export_data)}"), ln=True)
                    pdf.ln(3)
                    for rkey, ed in all_export_data.items():
                        fr      = ed["fr"]
                        mk_disp = rkey.split("__")[0] if "__" in rkey else rkey
                        series  = fr.get("series", "")
                        m_name  = _model_name(ALL_MODELS.get(mk_disp, {})) or mk_disp
                        suffix  = f" [{series}]" if series else ""
                        pdf.set_font(FONT, "B", 12)
                        pdf.cell(0, 8, _safe(f"{m_name}{suffix}"), ln= True)
                        pdf.set_font(FONT, "", 10)
                        pdf.cell(0, 5, _safe(
                            f"  {t('ab_author')}: "
                            f"{ALL_MODELS.get(mk_disp,{}).get('author','')}"),
                            ln=True)
                        for label, val in [
                            (t("rs_rmse"),   fr["rmse"]),
                            (t("rs_r2adj"),  fr["r2adj"]),
                            (t("rs_aic"),    fr["aic"]),
                            (t("rs_bic"),    fr["bic"]),
                            (t("rs_mu_max"), ed["mu_max"]),
                            (t("rs_mu_avg"), ed["mu_avg"]),
                            (t("rs_td"),     ed["td"]),
                        ]:
                            v_str = (f"{val:.5f}"
                                     if isinstance(val, float) and not np.isnan(val)
                                     else "N/A")
                            pdf.cell(0, 5, _safe(f"  {label}: {v_str}"), ln=True)
                        pnames = ALL_MODELS.get(mk_disp, {}).get("params", [])
                        for pn, pv in zip(pnames, fr["popt"]):
                            pdf.cell(0, 5, _safe(f"  {pn}: {pv:.5g}"), ln=True)
                        pdf.ln(3)
                    buf_pdf = BytesIO(pdf.output())
                    st.download_button(
                        "⬇ Baixar PDF", data=buf_pdf.getvalue(),
                        file_name="GrowthEmulator_results.pdf",
                        mime="application/pdf",
                        use_container_width=True)
            except ImportError:
                st.caption(f"{t('rs_error_lib')}")

        # ── AdSense — after export, non-intrusive ─────────────
        st.markdown("<br>", unsafe_allow_html=True)
        _render_adsense_banner(height=100)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 20. TAB: FERRAMENTAS / TOOLS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def tab_tools():
    from scipy import stats as _stats

    st.markdown(f"## 🔧 {t('tl_title')}")
    st.caption(t("tl_subtitle"))
    st.divider()

    # ── Show TOOLS_BOXES models ───────────────────────────────
    for box in TOOLS_BOXES:
        lang = st.session_state.get("lang", "pt")
        box_title = box["title"].get(lang, box["title"]["en"])
        icon = box.get("icon", "🔬")
        with st.expander(f"{icon} {box_title}", expanded=False):
            cols = st.columns(min(len(box["models"]), 2))
            for i, model in enumerate(box["models"]):
                col = cols[i % len(cols)]
                mc = model["category"].get(lang, model["category"]["en"])
                col.markdown(f"""
<div class="bio-card">
  <h4>{_model_name(model)}</h4>
  <div class="author">{model['author']}</div>
  <div class="category">{mc}</div>
</div>
""", unsafe_allow_html=True)
                col.latex(model["latex"])
                col.caption("Params: " + " · ".join(model["params"]))

    st.divider()

    # ═══════════════════════════════════════════════════════════
    # SECTION 1 — ENZYME KINETICS (Michaelis-Menten)
    # ═══════════════════════════════════════════════════════════
    st.markdown(f"### 🧫 {t('tl_enzyme_title')}")
    st.caption(t("tl_enzyme_note"))

    with st.expander(f"⚙️ {t('tl_enzyme_config')}", expanded=True):
        ek1, ek2 = st.columns(2)
        ek_mode = ek1.radio(t("tl_data_source_lbl"),
                             [t("tl_use_loaded_data"),
                              t("tl_enter_manually")],
                             key="ek_mode")
        ek_enzyme_conc = ek2.number_input(
            t("tl_enzyme_conc_lbl"),
            value=0.0, min_value=0.0, format="%.4f", key="ek_enzyme_conc",
            help=t("tl_enzyme_conc_help"))

        S_enz = v_enz = None
        if ek_mode == t("tl_use_loaded_data"):
            # Use the correct session state keys — df_clean and df
            _dc  = st.session_state.get("df_clean")
            _df  = st.session_state.get("df")
            df_src = _dc if _dc is not None else _df
            s_col  = _hdr_primary("substrate")
            t_col2 = st.session_state.headers.get("time")
            if df_src is not None and s_col and t_col2 and s_col in df_src.columns:
                S_raw  = pd.to_numeric(df_src[s_col], errors="coerce").dropna().values
                t_raw  = pd.to_numeric(df_src[t_col2], errors="coerce").dropna().values
                n      = min(len(S_raw), len(t_raw))
                if n > 2:
                    # v = -dS/dt
                    v_raw  = -np.gradient(S_raw[:n], t_raw[:n])
                    # only positive v
                    mask   = (S_raw[:n] > 0) & (v_raw > 0)
                    S_enz  = S_raw[:n][mask]
                    v_enz  = v_raw[mask]
                    st.success(f"✅ {len(S_enz)} {t('tl_pts_extracted')}")
                else:
                    st.warning("Dados insuficientes — mapeie Tempo e Substrato na aba Dados")
            else:
                st.warning(t("tl_map_sub_time"))
        else:
            st.caption(t("tl_enter_sv_pairs"))
            raw_s = st.text_input(t("tl_s_values_lbl"), "0.5;1;2;4;8;12;20", key="ek_s_raw")
            raw_v = st.text_input(t("tl_v_values_lbl"), "0.8;1.3;2.0;2.8;3.5;3.8;4.0", key="ek_v_raw")
            try:
                sep = ";" if ";" in raw_s else ","
                S_enz = np.array([float(x.replace(",",".")) for x in raw_s.split(sep)])
                v_enz = np.array([float(x.replace(",",".")) for x in raw_v.split(sep)])
            except Exception:
                st.error(t("tl_format_invalid"))

    if S_enz is not None and v_enz is not None and len(S_enz) >= 3:
        if st.button(f"▶ {t('tl_run')} — Michaelis-Menten", key="run_mm"):
            try:
                # Fit MM
                popt_mm, pcov = curve_fit(m_michaelis, S_enz, v_enz,
                                           p0=[max(v_enz)*1.1, np.median(S_enz)],
                                           bounds=([0,0],[np.inf,np.inf]), maxfev=5000)
                Vmax_fit, Km_fit = popt_mm
                v_pred_mm = m_michaelis(S_enz, *popt_mm)
                r2_mm = r2_adj(v_enz, v_pred_mm, 2)

                # Try competitive inhibition if product column exists
                p_col   = _hdr_primary("product")
                _dc2    = st.session_state.get("df_clean")
                _df2    = st.session_state.get("df")
                df_src2 = _dc2 if _dc2 is not None else _df2
                popt_ci = None
                if p_col and df_src2 is not None and p_col in df_src2.columns:
                    I_arr = pd.to_numeric(df_src2[p_col], errors="coerce").dropna().values
                    I_arr = I_arr[:len(S_enz)]
                    if len(I_arr) >= len(S_enz):
                        try:
                            def _mm_ci(SI, Vm, Km_, Ki):
                                S_, I_ = SI
                                return Vm * S_ / (Km_ * (1 + I_ / Ki) + S_)
                            popt_ci, _ = curve_fit(
                                lambda SI, Vm, Km_, Ki: _mm_ci(SI, Vm, Km_, Ki),
                                (S_enz, I_arr[:len(S_enz)]), v_enz,
                                p0=[Vmax_fit, Km_fit, np.mean(I_arr)],
                                bounds=([0,0,0],[np.inf,np.inf,np.inf]), maxfev=5000)
                        except Exception:
                            pass

                kcat = Vmax_fit / ek_enzyme_conc if ek_enzyme_conc > 0 else None
                efficiency = kcat / Km_fit if kcat is not None and Km_fit > 0 else None

                # Store results
                st.session_state.tools_results["michaelis"] = {
                    "Vmax": Vmax_fit, "Km": Km_fit, "r2": r2_mm,
                    "kcat": kcat, "efficiency": efficiency,
                    "S": S_enz, "v_obs": v_enz, "v_pred": v_pred_mm,
                    "popt_ci": popt_ci,
                }
            except Exception as ex:
                st.error(f"{t('rs_error_adjust'): ex}")

    if "michaelis" in st.session_state.tools_results:
        res = st.session_state.tools_results["michaelis"]
        st.markdown(f"#### 📊 {t('tl_enzyme_results')}")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric(t("tl_vmax"),   f"{res['Vmax']:.4g}")
        mc2.metric(t("tl_km"),     f"{res['Km']:.4g}")
        mc3.metric("R² Ajustado",  f"{res['r2']:.5f}")
        if res["kcat"]:
            mc4.metric(t("tl_kcat"), f"{res['kcat']:.4g}")

        if res["efficiency"]:
            st.metric(t("tl_efficiency"), f"{res['efficiency']:.4g} µM⁻¹·h⁻¹")

        if res.get("popt_ci") is not None:
            ci = res["popt_ci"]
            st.info(f"🔬 {t('tl_ci_result')} — Vmax={ci[0]:.4g}  Km={ci[1]:.4g}  Ki={ci[2]:.4g}")

        S_enz, v_obs, v_pred_mm = res["S"], res["v_obs"], res["v_pred"]
        S_fine = np.linspace(S_enz.min()*0.1, S_enz.max()*1.1, 300)
        v_fine = m_michaelis(S_fine, res["Vmax"], res["Km"])

        tab_mm1, tab_mm2, tab_mm3 = st.tabs(["Curva MM", "Lineweaver-Burk", "Eadie-Hofstee"])
        with tab_mm1:
            fig_mm = go.Figure()
            fig_mm.add_trace(go.Scatter(x=S_fine, y=v_fine, mode="lines", name="MM fit",
                                         line=dict(color="#00c8b4", width=2.5)))
            fig_mm.add_trace(go.Scatter(x=S_enz, y=v_obs, mode="markers", name="Dados",
                                         marker=dict(color="#f97316", size=8)))
            fig_mm.add_hline(y=res["Vmax"], line_dash="dash", line_color="#8b949e",
                              annotation_text=f"Vmax={res['Vmax']:.4g}")
            fig_mm.add_vline(x=res["Km"],  line_dash="dash", line_color="#7c3aed",
                              annotation_text=f"Km={res['Km']:.4g}")
            fig_mm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  xaxis=dict(color="#8b949e", gridcolor="#30363d", title="[S]"),
                                  yaxis=dict(color="#8b949e", gridcolor="#30363d", title="v",
                                             tickformat=".4g", exponentformat="none"),
                                  legend=dict(bgcolor="rgba(0,0,0,0)"),
                                  uirevision="mm_chart",
                                  margin=dict(l=40, r=20, t=20, b=40), height=300)
            st.plotly_chart(fig_mm, use_container_width=True, config={"scrollZoom": True})

        with tab_mm2:
            # Lineweaver-Burk (double reciprocal)
            inv_S = 1.0 / (S_enz + 1e-12)
            inv_v = 1.0 / (v_obs  + 1e-12)
            try:
                lr = _stats.linregress(inv_S, inv_v)
                x_lb = np.linspace(inv_S.min()*0.5, inv_S.max()*1.1, 100)
                y_lb = lr.slope * x_lb + lr.intercept
            except Exception:
                lr, x_lb, y_lb = None, inv_S, inv_v
            fig_lb = go.Figure()
            if lr:
                fig_lb.add_trace(go.Scatter(x=x_lb, y=y_lb, mode="lines", name="Regressão",
                                             line=dict(color="#00c8b4", width=2)))
            fig_lb.add_trace(go.Scatter(x=inv_S, y=inv_v, mode="markers", name="1/S vs 1/v",
                                         marker=dict(color="#f97316", size=8)))
            fig_lb.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(color="#8b949e", gridcolor="#30363d", title="1/[S]"),
                yaxis=dict(color="#8b949e", gridcolor="#30363d", title="1/v",
                           tickformat=".4g", exponentformat="none"),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                uirevision="lb_chart",
                margin=dict(l=40, r=20, t=20, b=40), height=280)
            st.plotly_chart(fig_lb, use_container_width=True)
            if lr:
                st.caption(f"Slope = {lr.slope:.4g}  ·  Intercept = {lr.intercept:.4g}"
                           f"  ·  Vmax = {1/lr.intercept:.4g}  ·  Km = {lr.slope*lr.intercept:.4g}")

        with tab_mm3:
            # Eadie-Hofstee
            ef_x = v_obs / (S_enz + 1e-12)
            ef_y = v_obs
            try:
                lr2 = _stats.linregress(ef_x, ef_y)
                x_ef = np.linspace(ef_x.min()*0.8, ef_x.max()*1.1, 100)
                y_ef = lr2.slope * x_ef + lr2.intercept
            except Exception:
                lr2, x_ef, y_ef = None, ef_x, ef_y
            fig_ef = go.Figure()
            if lr2:
                fig_ef.add_trace(go.Scatter(x=x_ef, y=y_ef, mode="lines", name="Regressão",
                                             line=dict(color="#7c3aed", width=2)))
            fig_ef.add_trace(go.Scatter(x=ef_x, y=ef_y, mode="markers", name="v/S vs v",
                                         marker=dict(color="#f97316", size=8)))
            fig_ef.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(color="#8b949e", gridcolor="#30363d", title="v/[S]"),
                yaxis=dict(color="#8b949e", gridcolor="#30363d", title="v",
                           tickformat=".4g", exponentformat="none"),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                uirevision="ef_chart",
                margin=dict(l=40, r=20, t=20, b=40), height=280)
            st.plotly_chart(fig_ef, use_container_width=True)
            if lr2:
                st.caption(f"Vmax (intercept) = {lr2.intercept:.4g}  ·  Km (-slope) = {-lr2.slope:.4g}")

    st.divider()

    # ═══════════════════════════════════════════════════════════
    # SECTION 2 — LUEDEKING-PIRET (product formation)
    # ═══════════════════════════════════════════════════════════
    st.markdown(f"### 🔄 {t('tl_turnover_title')}")
    st.caption(t("tl_turnover_note"))

    # Correct: use df_clean if available, otherwise fall back to df
    _dc3   = st.session_state.get("df_clean")
    _df3   = st.session_state.get("df")
    df_src3 = _dc3 if _dc3 is not None else _df3
    t_col3   = st.session_state.headers.get("time")
    x_col3   = _hdr_primary("biomass")
    p_col3   = _hdr_primary("product")

    lp_ready = (df_src3 is not None and t_col3 and x_col3 and p_col3
                and x_col3 in df_src3.columns and p_col3 in df_src3.columns)

    with st.expander(f"⚙️ {t('tl_lp_config')}", expanded=bool(lp_ready)):
        if lp_ready:
            t_lp  = pd.to_numeric(df_src3[t_col3], errors="coerce").dropna().values
            X_lp  = pd.to_numeric(df_src3[x_col3], errors="coerce").dropna().values
            P_lp  = pd.to_numeric(df_src3[p_col3], errors="coerce").dropna().values
            n_lp  = min(len(t_lp), len(X_lp), len(P_lp))
            t_lp, X_lp, P_lp = t_lp[:n_lp], X_lp[:n_lp], P_lp[:n_lp]
            if n_lp >= 4:
                st.success(f"✅ {n_lp} {t('md_points_detection')}")
                if st.button(t("tl_lp_run_btn"), key="run_lp"):
                    try:
                        dX_dt = np.gradient(X_lp, t_lp)
                        dP_dt = np.gradient(P_lp, t_lp)
                        # Linear fit: dP/dt = α·dX/dt + β·X
                        A = np.column_stack([dX_dt, X_lp])
                        result = np.linalg.lstsq(A, dP_dt, rcond=None)
                        alpha_lp, beta_lp = result[0]
                        dP_pred = alpha_lp * dX_dt + beta_lp * X_lp
                        r2_lp = r2_adj(dP_dt, dP_pred, 2)
                        P_pred_int = np.cumsum(dP_pred) * np.mean(np.diff(t_lp))
                        P_pred_int = P_pred_int + P_lp[0] - P_pred_int[0]
                        st.session_state.tools_results["luedeking_piret"] = {
                            "alpha": alpha_lp, "beta": beta_lp, "r2": r2_lp,
                            "t": t_lp, "X": X_lp, "P_obs": P_lp,
                            "P_pred": P_pred_int, "dP_obs": dP_dt, "dP_pred": dP_pred,
                        }
                    except Exception as ex:
                        st.error(f"Erro: {ex}")
            else:
                st.warning(f"{n_lp} {t('tl_min_pts')}")
        else:
            st.info(t("dt_tool"))

        # Chick inactivation — always available if biomass data present
        st.markdown("---")
        st.markdown(f"**{t('tl_chick_section')}**")
        if df_src3 is not None and x_col3 and t_col3:
            t_ck = pd.to_numeric(df_src3[t_col3], errors="coerce").dropna().values
            X_ck = pd.to_numeric(df_src3[x_col3], errors="coerce").dropna().values
            n_ck = min(len(t_ck), len(X_ck))
            t_ck, X_ck = t_ck[:n_ck], X_ck[:n_ck]
            if st.button(t("tl_chick_run_btn"), key="run_chick"):
                try:
                    popt_ck, _ = curve_fit(m_chick, t_ck, X_ck,
                                            p0=[X_ck[0], 0.1],
                                            bounds=([0,0],[np.inf,np.inf]))
                    y_ck_pred = m_chick(t_ck, *popt_ck)
                    r2_ck = r2_adj(X_ck, y_ck_pred, 2)
                    st.session_state.tools_results["chick"] = {
                        "X0": popt_ck[0], "kd": popt_ck[1], "r2": r2_ck,
                        "t": t_ck, "X_obs": X_ck, "X_pred": y_ck_pred,
                    }
                except Exception as ex:
                    st.error(f"Erro: {ex}")
        else:
            st.info(t("dt_info"))

    if "luedeking_piret" in st.session_state.tools_results:
        res = st.session_state.tools_results["luedeking_piret"]
        st.markdown(f"#### 📊 {t('tl_lp_results')}")
        lc1, lc2, lc3 = st.columns(3)
        lc1.metric(t("tl_alpha"), f"{res['alpha']:.5g}")
        lc2.metric(t("tl_beta"),  f"{res['beta']:.5g}")
        lc3.metric("R² Ajustado", f"{res['r2']:.5f}")

        prod_type = (t("tl_prod_primary")
                     if abs(res["alpha"]) > abs(res["beta"]) * 5
                     else (t("tl_prod_secondary")
                           if abs(res["beta"]) > abs(res["alpha"]) * 5
                           else t("tl_prod_mixed")))
        st.info(f"🔬 {t('tl_classification')} **{prod_type}**")

        fig_lp = make_subplots(rows=1, cols=2,
                                subplot_titles=["P observado vs ajustado", "dP/dt: obs vs pred"])
        fig_lp.add_trace(go.Scatter(x=res["t"], y=res["P_obs"], mode="markers",
                                     name="P obs", marker=dict(color="#f97316", size=7)), row=1, col=1)
        fig_lp.add_trace(go.Scatter(x=res["t"], y=res["P_pred"], mode="lines",
                                     name="P pred", line=dict(color="#00c8b4", width=2)), row=1, col=1)
        fig_lp.add_trace(go.Scatter(x=res["t"], y=res["dP_obs"], mode="markers",
                                     name="dP/dt obs", marker=dict(color="#a78bfa", size=6)), row=1, col=2)
        fig_lp.add_trace(go.Scatter(x=res["t"], y=res["dP_pred"], mode="lines",
                                     name="dP/dt pred", line=dict(color="#7c3aed", width=2)), row=1, col=2)
        fig_lp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               showlegend=True, legend=dict(bgcolor="rgba(0,0,0,0)"),
                               uirevision="lp_chart",
                               margin=dict(l=40, r=20, t=40, b=40), height=300)
        st.plotly_chart(fig_lp, use_container_width=True)

    if "chick" in st.session_state.tools_results:
        res = st.session_state.tools_results["chick"]
        st.markdown(f"#### 📊 {t('tl_ck_results')}")
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("X₀", f"{res['X0']:.4g}")
        cc2.metric("kd (h⁻¹)", f"{res['kd']:.5g}")
        cc3.metric("R² Ajustado", f"{res['r2']:.5f}")
        t_half = np.log(2) / res["kd"] if res["kd"] > 0 else float("nan")
        st.caption(f"{t('tl_halflife')} = {t_half:.3f} h")
        fig_ck = go.Figure()
        t_ck_fine = np.linspace(res["t"][0], res["t"][-1], 300)
        fig_ck.add_trace(go.Scatter(x=t_ck_fine,
                                     y=m_chick(t_ck_fine, res["X0"], res["kd"]),
                                     mode="lines", name="Chick fit",
                                     line=dict(color="#00c8b4", width=2.5)))
        fig_ck.add_trace(go.Scatter(x=res["t"], y=res["X_obs"], mode="markers",
                                     name="Dados", marker=dict(color="#f97316", size=8)))
        fig_ck.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="#8b949e", gridcolor="#30363d", title="Tempo (h)"),
            yaxis=dict(color="#8b949e", gridcolor="#30363d", title="X (g/L)",
                       tickformat=".4g", exponentformat="none"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            uirevision="ck_chart",
            margin=dict(l=40, r=20, t=20, b=40), height=280)
        st.plotly_chart(fig_ck, use_container_width=True)

    st.divider()

    # ═══════════════════════════════════════════════════════════
    # SECTION 3 — PIRT (variable yield)
    # ═══════════════════════════════════════════════════════════
    st.markdown(f"### ⚗️ {t('tl_yield_title')}")
    st.caption(t("tl_yield_note"))

    with st.expander(f"⚙️ {t('tl_pirt_config')}", expanded=False):
        pt1, pt2 = st.columns(2)
        pt1.markdown(t("t1_pirt_mark"))
        raw_mu_p  = pt1.text_input("μ (h⁻¹):",    "0.05;0.1;0.2;0.3;0.4;0.5", key="pirt_mu")
        raw_yobs  = pt1.text_input("Y_obs (g/g):", "0.30;0.38;0.44;0.48;0.50;0.51", key="pirt_y")

        def _parse_pirt(s):
            sep = ";" if ";" in s else ","
            return np.array([float(x.replace(",", ".")) for x in s.split(sep)])

        if st.button(t("tl_pirt_run_btn"), key="run_pirt"):
            try:
                mu_arr  = _parse_pirt(raw_mu_p)
                yobs_arr = _parse_pirt(raw_yobs)
                n_min = min(len(mu_arr), len(yobs_arr))
                mu_arr, yobs_arr = mu_arr[:n_min], yobs_arr[:n_min]

                def pirt_func(mu, Ymax, ms):
                    return Ymax / (1 + ms * Ymax / (mu + 1e-12))

                popt_pt, _ = curve_fit(pirt_func, mu_arr, yobs_arr,
                                        p0=[max(yobs_arr)*1.1, 0.05],
                                        bounds=([0,0],[5,5]))
                Ymax_fit, ms_fit = popt_pt
                y_pred_pt = pirt_func(mu_arr, *popt_pt)
                r2_pt = r2_adj(yobs_arr, y_pred_pt, 2)
                st.session_state.tools_results["pirt"] = {
                    "Ymax": Ymax_fit, "ms": ms_fit, "r2": r2_pt,
                    "mu": mu_arr, "Yobs": yobs_arr, "Ypred": y_pred_pt,
                }
            except Exception as ex:
                st.error(f"Erro: {ex}")

    if "pirt" in st.session_state.tools_results:
        res = st.session_state.tools_results["pirt"]
        st.markdown(f"#### 📊 {t('tl_pirt_results')}")
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric(t("tl_yield_true"), f"{res['Ymax']:.5g} g/g")
        pc2.metric(t("tl_maintenance_coef"), f"{res['ms']:.5g} g/g·h")
        pc3.metric("R² Ajustado",      f"{res['r2']:.5f}")

        mu_fine = np.linspace(res["mu"].min()*0.5, res["mu"].max()*1.1, 200)
        y_fine_pt = res["Ymax"] / (1 + res["ms"] * res["Ymax"] / (mu_fine + 1e-12))
        fig_pt = go.Figure()
        fig_pt.add_trace(go.Scatter(x=mu_fine, y=y_fine_pt, mode="lines", name="Pirt",
                                     line=dict(color="#00c8b4", width=2.5)))
        fig_pt.add_trace(go.Scatter(x=res["mu"], y=res["Yobs"], mode="markers", name="Dados",
                                     marker=dict(color="#f97316", size=8)))
        fig_pt.add_hline(y=res["Ymax"], line_dash="dash", line_color="#8b949e",
                          annotation_text=f"Ymax={res['Ymax']:.4g}")
        fig_pt.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="#8b949e", gridcolor="#30363d", title="μ (h⁻¹)"),
            yaxis=dict(color="#8b949e", gridcolor="#30363d", title="Y (g/g)",
                       tickformat=".4g", exponentformat="none"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            uirevision="pt_chart",
            margin=dict(l=40, r=20, t=20, b=40), height=280)
        st.plotly_chart(fig_pt, use_container_width=True)

    st.divider()

    # ═══════════════════════════════════════════════════════════
    # SECTION 4 — KINETIC PARAMETER CALCULATOR
    # ═══════════════════════════════════════════════════════════
    st.markdown(f"### 🧮 {t('tl_params_title')}")
    st.caption(t("tl_params_note"))

    with st.expander(f"⚙️ {t('tl_calc_quick')}", expanded=True):
        calc_cols = st.columns(2)
        with calc_cols[0]:
            st.markdown(f"**{t('tl_td_gen_section')}**")
            mu_calc = st.number_input("μmax (h⁻¹):", min_value=1e-6, value=0.693,
                                       format="%.4f", key="calc_mu")
            td_calc  = np.log(2) / mu_calc
            gen_calc = 1.0 / td_calc
            st.metric(t("tl_td"),       f"{td_calc:.4f} h")
            st.metric(t("tl_gen_per_h"), f"{gen_calc:.4f} gen/h")

            st.markdown("---")
            st.markdown(f"**{t('tl_mu_from_pts')}**")
            xA = st.number_input("X₁ (g/L):", min_value=1e-6, value=0.1, format="%.4f", key="xA")
            xB = st.number_input("X₂ (g/L):", min_value=1e-6, value=1.0, format="%.4f", key="xB")
            tA = st.number_input("t₁ (h):", value=0.0, format="%.2f", key="tA")
            tB = st.number_input("t₂ (h):", min_value=1e-4, value=5.0, format="%.2f", key="tB")
            if tB > tA and xB > 0 and xA > 0:
                mu_est = np.log(xB / xA) / (tB - tA)
                td_est = np.log(2) / mu_est if mu_est > 0 else float("nan")
                st.metric(t("dt_estimated_mu"), f"{mu_est:.5f}")
                st.metric(t("dt_estimated_td"),   f"{td_est:.4f}" if not np.isnan(td_est) else "—")

        with calc_cols[1]:
            st.markdown(f"**{t('tl_yield_coef_sect')}**")
            X0_yc = st.number_input("X₀ (g/L):", min_value=0.0, value=0.05, format="%.4f", key="X0_yc")
            Xf_yc = st.number_input("Xf (g/L):", min_value=0.0, value=4.0,  format="%.4f", key="Xf_yc")
            S0_yc = st.number_input("S₀ (g/L):", min_value=0.0, value=20.0, format="%.4f", key="S0_yc")
            Sf_yc = st.number_input("Sf (g/L):", min_value=0.0, value=0.5,  format="%.4f", key="Sf_yc")
            dX = Xf_yc - X0_yc
            dS = S0_yc - Sf_yc
            if dS > 0:
                Y_obs_calc = dX / dS
                st.metric(t("tl_yield_obs"), f"{Y_obs_calc:.5f} g·g⁻¹")
                st.caption(f"ΔX = {dX:.4g} g/L  ·  ΔS = {dS:.4g} g/L")
            else:
                st.warning(t("dt_estimated_yield"))

            st.markdown("---")
            st.markdown(f"**{t('tl_chick_const_sect')}**")
            X0_ck = st.number_input(t("dt_estimated_x0"), min_value=1e-6,
                                     value=5.0, format="%.4f", key="X0_ck_calc")
            Xt_ck = st.number_input("X(t) (g/L):", min_value=1e-8,
                                     value=0.5, format="%.4f", key="Xt_ck_calc")
            t_ck_c = st.number_input("t (h):", min_value=1e-4,
                                      value=5.0, format="%.2f", key="t_ck_calc")
            if X0_ck > 0 and Xt_ck > 0 and t_ck_c > 0:
                kd_calc = -np.log(Xt_ck / X0_ck) / t_ck_c
                t_half_c = np.log(2) / kd_calc if kd_calc > 0 else float("nan")
                st.metric("kd (h⁻¹)", f"{kd_calc:.5f}")
                st.metric("t½ (h)", f"{t_half_c:.4f}" if not np.isnan(t_half_c) else "—")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 21. TAB: ABOUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def tab_about():
    st.markdown(f"## {t('ab_title')}")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(LOGO_HTML, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
<div class="bio-card">
  <h4>{t('app_name')}</h4>
  <div class="author">{t('ab_version')} 1.2  ·  2026</div>
  <p style="margin-top:8px;font-size:.87rem;color:var(--fg)">{t('ab_desc')}</p>
  <p style="font-size:.82rem;color:var(--fg2)">
    <strong>{t('ab_author')}:</strong>
    <a href="https://www.linkedin.com/in/matheusmonteirobatista/" style="color:var(--acc)">Matheus Monteiro Batista</a><br>
    <strong>E-mail:</strong>
    <a href="mailto:eng.matheusmbatista@gmail.com" style="color:var(--acc)">eng.matheusmbatista@gmail.com</a><br>
    <strong>{t('ab_lic')}:</strong>
    <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/" style="color:var(--acc)">CC BY-NC-ND 4.0</a>
  </p>
</div>
""", unsafe_allow_html=True)
    if st.button(t("ab_contact_btn")):
        contact_dialog()
    st.divider()
    c3, c4 = st.columns(2)
    with c3:
        st.markdown(f"""
**{t('ab_stack')}**
- Python 3.9+ · Streamlit ≥ 1.32
- Plotly · SciPy · Pandas · NumPy
- openpyxl · fpdf2
""")
    with c4:
        st.markdown(f"""
**{t('ab_algorithms')}**
- Levenberg-Marquardt · Trust-Region Reflective
- Nelder-Mead (Simplex) · Differential Evolution
- {t('ab_integrators')}: RK4 · RK45 · LSODA
""")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 21. TAB: GUIDE & REFERENCES  ← NEW
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_DECISION_GUIDE = [
    {
        "scenario": {
            "pt": "Crescimento exponencial puro sem limite",
            "en": "Pure exponential growth, no upper limit",
            "es": "Crecimiento exponencial puro sin límite",
            "zh": "纯指数增长，无上限",
        },
        "model": {
            "pt": "Malthus (Exponencial)",
            "en": "Malthus (Exponential)",
            "es": "Malthus (Exponencial)",
            "zh": "Malthus 指数型 (Exponential)"
        },

        "reason": {
            "pt": "Apenas 2 parâmetros. Ideal para fase log inicial ou poucos pontos experimentais.",
            "en": "Only 2 parameters. Ideal for the early log phase or few data points.",
            "es": "Solo 2 parámetros. Ideal para la fase log inicial o pocos datos.",
            "zh": "仅 2 个参数。适用于早期对数期或数据点较少的情况。",
        },
        "data_needed": "time, biomass",
        "color": "#1a365d",
        "text_color": "#90cdf4",
    },
    {
        "scenario": {
            "pt": "Curva S simples com platô (sem fase lag visível)",
            "en": "Simple S-shaped curve with plateau (no visible lag)",
            "es": "Curva S simple con meseta (sin fase lag visible)",
            "zh": "简单 S 形曲线带平台（无明显滞后期）",
        },
        "model": {
            "pt": "Verhulst — Logístico",
            "en": "Verhulst — Logistic",
            "es": "Verhulst — Logístico",
            "zh": "Verhulst — 逻辑斯蒂型 (Logistic)"
        },

        "reason": {
            "pt": "Modelo logístico clássico. 3 parâmetros: X₀, Xm, μmax. Robusto e amplamente validado.",
            "en": "Classic logistic model. 3 parameters: X₀, Xm, μmax. Robust and widely validated.",
            "es": "Modelo logístico clásico. 3 parámetros: X₀, Xm, μmax. Robusto y validado.",
            "zh": "经典 Logistic 模型，3 个参数：X₀, Xm, μmax，稳健且广泛验证。",
        },
        "data_needed": "time, biomass",
        "color": "#1c4532",
        "text_color": "#9ae6b4",
    },
    {
        "scenario": {
            "pt": "Curva sigmoide com fase lag e assíntota superior clara",
            "en": "Sigmoid curve with lag phase and clear upper asymptote",
            "es": "Curva sigmoide con fase lag y asíntota superior clara",
            "zh": "具有明显滞后期和上渐近线的 S 形曲线",
        },
        "model": {
            "pt": "Gompertz Modificado (Gibson/Zwietering)",
            "en": "Gompertz Modified (Gibson/Zwietering)",
            "es": "Gompertz Modificado (Gibson/Zwietering)",
            "zh": "Gompertz 修正型 (Gibson/Zwietering)"
        },

        "reason": {
            "pt": "Padrão ouro em microbiologia de alimentos. Parâmetros biológicos diretos: lag (λ), μmax e A. "
                  "Preferido ao logístico quando a curva é assimétrica.",
            "en": "Gold standard in food microbiology. Direct biological parameters: lag (λ), μmax and A. "
                  "Preferred over logistic for asymmetric curves.",
            "es": "Estándar de oro en microbiología de alimentos. Parámetros biológicos directos: lag (λ), μmax y A.",
            "zh": "食品微生物学金标准。直接生物学参数：滞后期 (λ)、μmax 和 A。",
        },
        "data_needed": "time, biomass",
        "color": "#234e52",
        "text_color": "#81e6d9",
    },
    {
        "scenario": {
            "pt": "Modelagem mecanística com previsão de comportamento futuro",
            "en": "Mechanistic modeling with future behavior prediction",
            "es": "Modelado mecanístico con predicción de comportamiento futuro",
            "zh": "具有未来行为预测的机理建模",
        },
        "model": {
            "pt": "Baranyi & Roberts",
            "en": "Baranyi & Roberts",
            "es": "Baranyi & Roberts",
            "zh": "Baranyi & Roberts"
        },

        "reason": {
            "pt": "Derivado de equações diferenciais com significado fisiológico. Melhor modelo global "
                  "para cultura em batelada. Recomendado quando a fase lag é longa ou irregular.",
            "en": "Derived from ODEs with physiological meaning. Best overall model for batch culture. "
                  "Recommended when lag is long or irregular.",
            "es": "Derivado de EDOs con significado fisiológico. Mejor modelo global para cultivo discontinuo.",
            "zh": "源自具有生理意义的常微分方程。批培养的最佳综合模型，适用于滞后期较长或不规则的情况。",
        },
        "data_needed": "time, biomass",
        "color": "#322659",
        "text_color": "#e9d8fd",
    },
    {
        "scenario": {
            "pt": "Crescimento limitado por substrato (biorreator, quimiostato)",
            "en": "Substrate-limited growth (bioreactor, chemostat)",
            "es": "Crecimiento limitado por sustrato (biorreactor, quimiostato)",
            "zh": "底物限制生长（生物反应器，恒化器）",
        },
        "model": {
            "pt": "Monod",
            "en": "Monod",
            "es": "Monod",
            "zh": "Monod"
        },
        "reason": {
            "pt": "Modelo fundamental em engenharia de bioprocessos. Requer medições de substrato (S). "
                  "μmax e Ks são parâmetros cinéticos do microrganismo.",
            "en": "Fundamental model in bioprocess engineering. Requires substrate measurements (S). "
                  "μmax and Ks are organism kinetic parameters.",
            "es": "Modelo fundamental en ingeniería de bioprocesos. Requiere mediciones de sustrato.",
            "zh": "生物过程工程中的基础模型。需要底物测量 (S)，μmax 和 Ks 为菌株动力学参数。",
        },
        "data_needed": "time, biomass, substrate",
        "color": "#2a1f45",
        "text_color": "#d6bcfa",
    },
    {
        "scenario": {
            "pt": "Substrato inibidor em altas concentrações (fenol, etanol, antibiótico)",
            "en": "Inhibitory substrate at high concentrations (phenol, ethanol, antibiotics)",
            "es": "Sustrato inhibidor a altas concentraciones (fenol, etanol, antibiótico)",
            "zh": "高浓度抑制性底物（苯酚、乙醇、抗生素）",
        },
        "model": {
            "pt": "Haldane & Andrews",
            "en": "Haldane & Andrews",
            "es": "Haldane & Andrews",
            "zh": "Haldane & Andrews"
        },
        "reason": {
            "pt": "Adiciona o termo Ki ao modelo de Monod. A curva de taxa tem formato de sino. "
                  "Identifica concentração ótima de substrato.",
            "en": "Adds Ki inhibition term to Monod. Rate curve has a bell shape. "
                  "Identifies optimal substrate concentration.",
            "es": "Añade el término Ki al modelo de Monod. La curva de tasa tiene forma de campana.",
            "zh": "在 Monod 模型中添加 Ki 抑制项，速率曲线呈钟形，可识别最佳底物浓度。",
        },
        "data_needed": "time, biomass, substrate",
        "color": "#63171b",
        "text_color": "#feb2b2",
    },
    {
        "scenario": {
            "pt": "Cultivo contínuo com energia de manutenção significativa",
            "en": "Continuous culture with significant maintenance energy",
            "es": "Cultivo continuo con energía de mantenimiento significativa",
            "zh": "具有显著维持能的连续培养",
        },
        "model": {
            "pt": "Herbert-Pirt",
            "en": "Herbert-Pirt",
            "es": "Herbert-Pirt",
            "zh": "Herbert-Pirt"
        },
        "reason": {
            "pt": "Inclui coeficiente de manutenção (ms) e coeficiente de decaimento (kd). "
                  "Mais realista que Monod para culturas limitadas por energia.",
            "en": "Includes maintenance coefficient (ms) and decay (kd). "
                  "More realistic than Monod for energy-limited cultures.",
            "es": "Incluye coeficiente de mantenimiento (ms) y decaimiento (kd).",
            "zh": "包含维持系数 (ms) 和衰减系数 (kd)，对能量限制培养比 Monod 更准确。",
        },
        "data_needed": "time, biomass, substrate",
        "color": "#1a3a2a",
        "text_color": "#68d391",
    },
    {
        "scenario": {
            "pt": "Cinética enzimática in vitro (S → P)",
            "en": "In vitro enzyme kinetics (S → P)",
            "es": "Cinética enzimática in vitro (S → P)",
            "zh": "体外酶动力学 (S → P)",
        },
        "model":  {
            "pt": "Michaelis-Menten",
            "en": "Michaelis-Menten",
            "es": "Michaelis-Menten",
            "zh": "Michaelis-Menten"
        },
        "reason": {
            "pt": "Modelo universal para reações enzimáticas. Vmax = velocidade máxima; Km = constante de afinidade.",
            "en": "Universal model for enzymatic reactions. Vmax = maximum velocity; Km = affinity constant.",
            "es": "Modelo universal para reacciones enzimáticas. Vmax = velocidad máxima; Km = constante de afinidad.",
            "zh": "酶催化反应的通用模型。Vmax = 最大速率；Km = 亲和力常数。",
        },
        "data_needed": "time, substrate",
        "color": "#1a2c4a",
        "text_color": "#63b3ed",
    },
    {
        "scenario": {
            "pt": "Produção de metabólito (ácido lático, antibiótico, etanol)",
            "en": "Metabolite production (lactic acid, antibiotic, ethanol)",
            "es": "Producción de metabolito (ácido láctico, antibiótico, etanol)",
            "zh": "代谢物生产（乳酸、抗生素、乙醇）",
        },
        "model": {
            "pt": "Luediking-Piret",
            "en": "Luediking-Piret",
            "es": "Luediking-Piret",
            "zh": "Luediking-Piret"
        },
        "reason": {
            "pt": "Distingue produção associada ao crescimento (α) e não associada (β). "
                  "Fundamental para otimização de bioprodução.",
            "en": "Separates growth-associated (α) and non-growth-associated (β) production. "
                  "Essential for bioproduction optimization.",
            "es": "Distingue producción asociada al crecimiento (α) y no asociada (β).",
            "zh": "区分生长相关产物 (α) 和非生长相关产物 (β)，对生物生产优化至关重要。",
        },
        "data_needed": "time, biomass",
        "color": "#2d2a1f",
        "text_color": "#fbd38d",
    },
    {
        "scenario": {
            "pt": "Ajuste empírico rápido sem interpretação mecanística",
            "en": "Quick empirical fit without mechanistic interpretation",
            "es": "Ajuste empírico rápido sin interpretación mecanística",
            "zh": "无需机理解释的快速经验拟合",
        },
        "model": {
            "pt": "Regressão Polinomial / Gaussiana",
            "en": "Polynomial / Gaussian Regression",
            "es": "Regresión Polinómica / Gaussiana",
            "zh": "多项式 / 高斯回归 (Polynomial / Gaussian Regression)"
        },
        "reason": {
            "pt": "Flexível e rápido. Use quando precisar de uma curva suave para interpolação "
                  "ou quando os dados não se adequam a nenhum modelo cinético.",
            "en": "Flexible and fast. Use when a smooth interpolation curve is needed "
                  "or when data doesn't fit any kinetic model.",
            "es": "Flexible y rápido. Úselo cuando necesite interpolación suave.",
            "zh": "灵活快速，适用于需要平滑插值或数据不符合任何动力学模型时。",
        },
        "data_needed": "time, biomass",
        "color": "#2d3748",
        "text_color": "#a0aec0",
    },
    {
        "scenario": {
            "pt": "Curva completa com todas as fases (forma de S generalizada)",
            "en": "Full growth curve with all phases (generalized S-shape)",
            "es": "Curva completa con todas las fases (forma S generalizada)",
            "zh": "含所有生长阶段的完整生长曲线（广义 S 形）",
        },
        "model": {
            "pt": "Richards",
            "en": "Richards",
            "es": "Richards",
            "zh": "Richards"
        },
        "reason": {
            "pt": "4 parâmetros: Xm, ν (assimetria), k, λ. Generaliza o Gompertz e o Logístico. "
                  "Máxima flexibilidade para curvas não simétricas.",
            "en": "4 parameters: Xm, ν (asymmetry), k, λ. Generalizes Gompertz and Logistic. "
                  "Maximum flexibility for asymmetric curves.",
            "es": "4 parámetros: Xm, ν (asimetría), k, λ. Generaliza Gompertz y Logístico.",
            "zh": "4 个参数：Xm、ν（不对称性）、k、λ，是 Gompertz 和 Logistic 的广义形式。",
        },
        "data_needed": "time, biomass",
        "color": "#1a2c3a",
        "text_color": "#76e4f7",
    },
]

_REFERENCES = [
    {"category": {
        "pt": "📂 Modelos Exponencial e Logístico",
        "en": "📂 Exponential and Logistic Models",
        "es": "📂 Modelos Sigmoides",
        "zh": "📂 指数与逻辑增长模型 (Exponential and Logistic Models)",
     },
     "refs": [
        "Malthus, T.R. (1798). An Essay on the Principle of Population. J. Johnson, London.",
        "Verhulst, P.F. (1845). Recherches mathématiques sur la loi d'accroissement de la population. Nouv. Mém. Acad. Roy. Sci. B.-Arts Belg., 18, 1–41.",
        "Pearl, R. & Reed, L.J. (1920). On the Rate of Growth of the Population of the United States. PNAS, 6(6), 275–288.",
        "Zwietering, M.H., Jongenburger, I., Rombouts, F.M., & van't Riet, K. (1990). Modeling of the Bacterial Growth Curve. Appl. Environ. Microbiol., 56(6), 1875–1881.",
        "Gibson, A.M., Bratchell, N., & Roberts, T.A. (1987). The effect of NaCl and temperature on the rate and extent of growth of Clostridium botulinum. J. Appl. Bacteriol., 62(6), 479–490.",
     ]},
    {"category":{
        "pt": "📂 Modelos Sigmoides",
        "en": "📂 Sigmoid Models",
        "es": "📂 Modelos Sigmoides",
        "zh": "📂 S型曲线模型 (Sigmoid Models)",
     },
     "refs": [
        "Richards, F.J. (1959). A flexible growth function for empirical use. J. Exp. Bot., 10(29), 290–301.",
        "Von Bertalanffy, L. (1957). Quantitative laws in metabolism and growth. Q. Rev. Biol., 32(3), 217–231.",
        "Baranyi, J. & Roberts, T.A. (1994). A dynamic approach to predicting bacterial growth in food. Int. J. Food Microbiol., 23(3–4), 277–294.",
        "Buchanan, R.L., Whiting, R.C., & Damert, W.C. (1997). When is simple good enough: a comparison of the Gompertz, Baranyi, and three-phase linear models for fitting bacterial growth curves. Food Microbiol., 14(4), 313–326.",
     ]},
    {"category":{
        "pt": "📂 Modelos Cinéticos Clássicos (Monod e variantes)",
        "en": "📂 Classical Kinetic Models (Monod and variants)",
        "es": "📂 Modelos Cinéticos Clásicos (Monod y variantes)",
        "zh": "📂 经典动力学模型（Monod及变体",
     },
     "refs": [
        "Monod, J. (1949). The growth of bacterial cultures. Annu. Rev. Microbiol., 3(1), 371–394.",
        "Tessier, G. (1942). Croissance des populations bactériennes et quantité d'aliment disponible. Rev. Sci., 80, 209–214.",
        "Moser, H. (1958). The dynamics of bacterial populations maintained in the chemostat. Carnegie Inst. Wash. Publ., 614.",
        "Contois, D.E. (1959). Kinetics of bacterial growth: relationship between population density and specific growth rate of continuous cultures. J. Gen. Microbiol., 21(1), 40–50.",
        "Andrews, J.F. (1968). A mathematical model for the continuous culture of microorganisms utilizing inhibitory substrates. Biotechnol. Bioeng., 10(6), 707–723.",
        "Haldane, J.B.S. (1930). Enzymes. Longmans, Green and Co., London.",
        "Aiba, S., Humphrey, A.E., & Millis, N.F. (1965). Biochemical Engineering. Academic Press, New York.",
     ]},
    {"category":{
        "pt": "📂 Modelos Mecanísticos e com Manutenção",
        "en": "📂 Mechanistic and Maintenance Models",
        "es": "📂 Modelos Mecanísticos y con Mantenimiento",
        "zh": "📂 机理与维持模型 (Mechanistic and Maintenance Models)",
     },
     "refs": [
        "Herbert, D., Elsworth, R., & Telling, R.C. (1956). The continuous culture of bacteria: a theoretical and experimental study. J. Gen. Microbiol., 14(3), 601–622.",
        "Pirt, S.J. (1965). The maintenance energy of bacteria in growing cultures. Proc. R. Soc. Lond. B, 163(991), 224–231.",
     ]},
    {"category":{
        "pt": "📂 Cinética Enzimática e Inibição",
        "en": "📂 Enzyme Kinetics and Inhibition",
        "es": "📂 Cinética Enzimática e Inhibición",
        "zh": "📂 酶动力学与 inhibiton (酶动力学与抑制)",
     },
     "refs": [
        "Michaelis, L. & Menten, M.L. (1913). Die Kinetik der Invertinwirkung. Biochem. Z., 49, 333–369.",
        "Briggs, G.E. & Haldane, J.B.S. (1925). A Note on the Kinetics of Enzyme Action. Biochem. J., 19(2), 338–339.",
     ]},
    {"category":{
        "pt": "📂 Formação de Produto",
        "en": "📂 Product Formation",
        "es": "📂 Formación de Producto",
        "zh": "📂 产物形成 (Product Formation)",
     },
     "refs": [
        "Luedeking, R. & Piret, E.L. (1959). A kinetic study of the lactic acid fermentation. Batch process at controlled pH. J. Biochem. Microbiol. Technol. Eng., 1(4), 393–412.",
        "Chick, H. (1908). An investigation of the laws of disinfection. J. Hyg. (Lond.), 8(1), 92–158.",
     ]},
    {"category":{
        "pt": "📂 Livros de Referência em Engenharia de Bioprocessos",
        "en": "📂 Reference Books in Bioprocess Engineering",
        "es": "📂 Libros de Referencia en Ingeniería de Bioprocesos",
        "zh": "📂 生物过程工程参考书目",
     },
     "refs": [
        "Bailey, J.E. & Ollis, D.F. (1986). Biochemical Engineering Fundamentals (2nd ed.). McGraw-Hill, New York.",
        "Shuler, M.L. & Kargi, F. (2002). Bioprocess Engineering: Basic Concepts (2nd ed.). Prentice Hall, Upper Saddle River.",
        "Nielsen, J., Villadsen, J., & Lidén, G. (2003). Bioreaction Engineering Principles (2nd ed.). Kluwer Academic, New York.",
        "Doran, P.M. (2012). Bioprocess Engineering Principles (2nd ed.). Academic Press, Waltham.",
        "Blanch, H.W. & Clark, D.S. (1997). Biochemical Engineering. Marcel Dekker, New York.",
     ]},
]

_PARAMS_INFO = [
    {
    "variable": {
      "pt": "μmax (h⁻¹)",
      "en": "μmax (h⁻¹)",
      "es": "μmax (h⁻¹)",
      "zh": "μmax (h⁻¹)"
        },
    "category": {
      "pt": "Velocidade específica máxima de crescimento",
      "en": "Maximum specific growth rate",
      "es": "Velocidad específica máxima de crecimiento",
      "zh": "最大比生长速率"
        },
    "value": {
      "pt": "Faixa típica: 0,05 - 2,5 h⁻¹ para bactérias; 0,01 - 0,5 para leveduras",
      "en": "Typical range: 0.05 - 2.5 h⁻¹ for bacteria; 0.01 - 0.5 for yeasts",
      "es": "Rango típico: 0,05 - 2,5 h⁻¹ para bacterias; 0,01 - 0,5 para levaduras",
      "zh": "典型范围：细菌为 0.05 - 2.5 h⁻¹；酵母为 0.01 - 0.5 h⁻¹"
        },
    "interpretation": {
      "pt": "Quanto maior, mais rápido o organismo cresce na fase exponencial.",
      "en": "The higher it is, the faster the organism grows in the exponential phase.",
      "es": "Cuanto mayor sea, más rápido crecerá el organismo en la fase exponencial.",
      "zh": "该值越大，生物体在对数生长期的生长速度就越快。"
        }
    },
    {
    "variable": {
      "pt": "Xm (g/L)",
      "en": "Xm (g/L)",
      "es": "Xm (g/L)",
      "zh": "Xm (g/L)"
        },
    "category": {
      "pt": "Concentração máxima de biomassa atingida na cultura",
      "en": "Maximum biomass concentration achieved in the culture",
      "es": "Concentración máxima de biomasa alcanzada en el cultivo",
      "zh": "培养物达到的最大生物量浓度"
        },
    "value": {
      "pt": "Depende do meio e da capacidade de suporte do sistema",
      "en": "Depends on the medium and the system's carrying capacity",
      "es": "Depende del medio y de la capacidad de soporte del sistema",
      "zh": "取决于培养基和系统的环境容纳量"
        },
    "interpretation": {
      "pt": "Representa o limite superior da curva S (capacidade de carga).",
      "en": "Represents the upper limit of the S-curve (carrying capacity).",
      "es": "Representa el límite superior de la curva S (capacidad de carga).",
      "zh": "代表S型曲线的上限（环境承载力）。"
        }
    },
    {
    "variable": {
      "pt": "λ (h)",
      "en": "λ (h)",
      "es": "λ (h)",
      "zh": "λ (h)"
        },
    "category": {
      "pt": "Fase lag — tempo de adaptação antes do crescimento ativo",
      "en": "Lag phase — adaptation time before active growth",
      "es": "Fase lag — tiempo de adaptación antes del crecimiento activo",
      "zh": "延滞期（Lag phase）— 活跃生长前的适应时间"
        },
    "value": {
      "pt": "0 (sem lag) a dezenas de horas em condições de estresse",
      "en": "0 (no lag) to dozens of hours under stress conditions",
      "es": "0 (sin lag) a decenas de horas en condiciones de estrés",
      "zh": "从 0（无延滞期）到应激条件下的数十小时不等"
        },
    "interpretation": {
      "pt": "Parâmetro crítico em microbiologia preditiva; afeta tempo de processo.",
      "en": "Critical parameter in predictive microbiology; affects process time.",
      "es": "Parámetro crítico en microbiología predictiva; afecta el tiempo de proceso.",
      "zh": "预测微生物学中的关键参数；直接影响工艺时间。"
        }
    },
    {
    "variable": {
      "pt": "Ks (g/L)",
      "en": "Ks (g/L)",
      "es": "Ks (g/L)",
      "zh": "Ks (g/L)"
        },
    "category": {
      "pt": "Constante de meia saturação (Monod e variantes)",
      "en": "Half-saturation constant (Monod and variants)",
      "es": "Constante de media saturación (Monod y variantes)",
      "zh": "半饱和常数（Monod及变体）"
        },
    "value": {
      "pt": "0,001 – 10 g/L dependendo do substrato e organismo",
      "en": "0.001 – 10 g/L depending on substrate and organism",
      "es": "0,001 – 10 g/L dependiendo del sustrato y organismo",
      "zh": "0.001 – 10 g/L，取决于底物和生物体"
        },
    "interpretation": {
      "pt": "Concentração de substrato na qual μ = μmax/2. Mede afinidade pelo substrato.",
      "en": "Substrate concentration at which μ = μmax/2. Measures substrate affinity.",
      "es": "Concentración de sustrato en la cual μ = μmax/2. Mide la afinidad por el sustrato.",
      "zh": "μ = μmax/2 时的底物浓度。用于衡量对底物的亲和力。"
        }
    },
    {
    "variable": {
      "pt": "Ki (g/L)",
      "en": "Ki (g/L)",
      "es": "Ki (g/L)",
      "zh": "Ki (g/L)"
        },
    "category": {
      "pt": "Constante de inibição por substrato (Haldane, Aiba)",
      "en": "Substrate inhibition constant (Haldane, Aiba)",
      "es": "Constante de inhibición por sustrato (Haldane, Aiba)",
      "zh": "底物抑制常数（Haldane, Aiba）"
        },
    "value": {
      "pt": "Geralmente > Ks; quanto menor, mais tóxico o substrato",
      "en": "Usually > Ks; the lower it is, the more toxic the substrate",
      "es": "Generalmente > Ks; cuanto menor sea, más tóxico es el sustrato",
      "zh": "通常 > Ks；该值越小，底物毒性越强"
        },
    "interpretation": {
      "pt": "Acima de Ki, o crescimento é inibido. Parâmetro chave para processos com fenol, etanol etc.",
      "en": "Above Ki, growth is inhibited. Key parameter for processes with phenol, ethanol, etc.",
      "es": "Por encima de Ki, el crecimiento se inhibe. Parámetro clave para procesos con fenol, etanol, etc.",
      "zh": "高于 Ki 时生长受阻。是处理苯酚、乙醇等工艺的关键参数。"
        }
    },
    {
    "variable": {
      "pt": "Y (g·g⁻¹)",
      "en": "Y (g·g⁻¹)",
      "es": "Y (g·g⁻¹)",
      "zh": "Y (g·g⁻¹)"
        },
    "category": {
      "pt": "Coeficiente de rendimento celular (biomassa por substrato consumido)",
      "en": "Cell yield coefficient (biomass produced per substrate consumed)",
      "es": "Coeficiente de rendimiento celular (biomasa por sustrato consumido)",
      "zh": "细胞得率系数（消耗每克底物生成的生物量）"
        },
    "value": {
      "pt": "0,01 – 0,8 g/g dependendo da fonte de carbono e condições",
      "en": "0.01 – 0.8 g/g depending on carbon source and conditions",
      "es": "0,01 – 0,8 g/g dependiendo de la fuente de carbono y condiciones",
      "zh": "0.01 – 0.8 g/g，取决于碳源和培养条件"
        },
    "interpretation": {
      "pt": "Y = ΔX / (-ΔS). Fundamental para balanço de massa e dimensionamento de biorreatores.",
      "en": "Y = ΔX / (-ΔS). Fundamental for mass balance and bioreactor sizing.",
      "es": "Y = ΔX / (-ΔS). Fundamental para el balance de masa y dimensionamiento de biorreactores.",
      "zh": "Y = ΔX / (-ΔS)。是物料衡算和生物反应器选型设计的基石。"
        }
    },
    {
    "variable": {
      "pt": "kd (h⁻¹)",
      "en": "kd (h⁻¹)",
      "es": "kd (h⁻¹)",
      "zh": "kd (h⁻¹)"
        },
    "category": {
      "pt": "Coeficiente de decaimento / morte celular endógena",
      "en": "Endogenous decay / cell death coefficient",
      "es": "Coeficiente de decaimiento / muerte celular endógena",
      "zh": "内源衰减 / 细胞死亡系数"
        },
    "value": {
      "pt": "0,001 – 0,1 h⁻¹ em condições normais",
      "en": "0.001 – 0.1 h⁻¹ under normal conditions",
      "es": "0,001 – 0,1 h⁻¹ en condiciones normales",
      "zh": "正常条件下为 0.001 – 0.1 h⁻¹"
        },
    "interpretation": {
      "pt": "Relevante em culturas prolongadas ou em condições limitantes.",
      "en": "Relevant in prolonged cultures or under limiting conditions.",
      "es": "Relevante en cultivos prolongados o en condiciones limitantes.",
      "zh": "在延长培养或限制性条件下尤为重要。"
        }
    },
    {
    "variable": {
      "pt": "α, β",
      "en": "α, β",
      "es": "α, β",
      "zh": "α, β"
        },
    "category": {
      "pt": "Parâmetros do modelo de Luedeking-Piret",
      "en": "Luedeking-Piret model parameters",
      "es": "Parámetros del modelo de Luedeking-Piret",
      "zh": "Luedeking-Piret 模型参数"
        },
    "value": {
      "pt": "α ≥ 0 (associado ao crescimento); β ≥ 0 (não associado)",
      "en": "α ≥ 0 (growth-associated); β ≥ 0 (non-growth-associated)",
      "es": "α ≥ 0 (asociado al crecimiento); β ≥ 0 (no asociado)",
      "zh": "α ≥ 0（与生长偶联）；β ≥ 0（与生长非偶联）"
        },
    "interpretation": {
      "pt": "α > 0, β ≈ 0: produto primário. α ≈ 0, β > 0: produto secundário.",
      "en": "α > 0, β ≈ 0: primary product. α ≈ 0, β > 0: secondary product.",
      "es": "α > 0, β ≈ 0: producto primario. α ≈ 0, β > 0: producto secundario.",
      "zh": "α > 0, β ≈ 0：初级代谢产物。α ≈ 0, β > 0：次级代谢产物。"
        }
    },
    {
    "variable": {
      "pt": "td (h)",
      "en": "td (h)",
      "es": "td (h)",
      "zh": "td (h)"
        },
    "category": {
      "pt": "Tempo de duplicação (doubling time)",
      "en": "Doubling time",
      "es": "Tiempo de duplicación (doubling time)",
      "zh": "加倍时间 (Doubling time)"
        },
    "value": {
      "pt": "td = ln(2) / μmax ≈ 0,693 / μmax",
      "en": "td = ln(2) / μmax ≈ 0.693 / μmax",
      "es": "td = ln(2) / μmax ≈ 0,693 / μmax",
      "zh": "td = ln(2) / μmax ≈ 0.693 / μmax"
        },
    "interpretation": {
      "pt": "Tempo necessário para a biomassa duplicar na fase exponencial.",
      "en": "Time required for biomass to double in the exponential phase.",
      "es": "Tiempo necesario para que la biomasa se duplique en la fase exponencial.",
      "zh": "在对数生长期内生物量翻倍所需的时间。"
        }
    }
]



def tab_guide():
    lang = st.session_state.get("lang", "pt")

    st.markdown(f"## {t('gd_title')}")
    st.divider()

    # ── Seção 1: O que é o GrowthEmulator ─────────────────────────
    with st.container():
        c1, c2 = st.columns([1, 3])
        with c1:
            st.markdown(LOGO_HTML + """
<div style="text-align:center;margin-top:8px">
  <span style="font-size:.75rem;color:var(--fg2)">v 1.2</span>
</div>
""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
<div class="bio-card">
  <h4 style="font-size:1rem">{t('gd_intro_title')}</h4>
  <p style="color:var(--fg);font-size:.88rem;line-height:1.6">{t('gd_intro_body')}</p>
  <div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:10px">
    <span style="background:var(--bg2);border:1px solid var(--brd);border-radius:20px;padding:3px 12px;font-size:.72rem;color:var(--acc)">🔬 {len(ALL_MODELS)} {t("gd_intro_background1")}</span>
    <span style="background:var(--bg2);border:1px solid var(--brd);border-radius:20px;padding:3px 12px;font-size:.72rem;color:var(--acc)">🌐 {t("gd_intro_background2")}</span>
    <span style="background:var(--bg2);border:1px solid var(--brd);border-radius:20px;padding:3px 12px;font-size:.72rem;color:var(--acc)">📊 {t("gd_intro_background3")}</span>
    <span style="background:var(--bg2);border:1px solid var(--brd);border-radius:20px;padding:3px 12px;font-size:.72rem;color:var(--acc)">♿ {t("gd_intro_background4")}</span>
  </div>
</div>
""", unsafe_allow_html=True)

    st.divider()

    # ── Seção 2: Qual modelo usar ─────────────────────────────
    st.markdown(f"### 🧭 {t('gd_decision_title')}")
    st.caption(t("gd_decision_note"))
    st.markdown("<br>", unsafe_allow_html=True)

    for entry in _DECISION_GUIDE:
        sc  = entry["scenario"].get(lang, entry["scenario"]["en"])
        rsn = entry["reason"].get(lang,  entry["reason"]["en"])
        c   = entry["color"]
        tc  = entry["text_color"]
        st.markdown(f"""
<div style="border-left:4px solid {tc};background:{c}22;border-radius:0 10px 10px 0;
     padding:12px 16px;margin-bottom:10px;cursor:default">
  <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:8px">
    <div style="flex:2;min-width:180px">
      <div style="font-weight:700;color:{tc};font-size:.88rem;margin-bottom:4px">📌 {sc}</div>
      <div style="font-size:.78rem;color:#c9d1d9;line-height:1.5">{rsn}</div>
    </div>
    <div style="flex:1;min-width:150px;text-align:right">
      <div style="background:{tc}22;border:1px solid {tc};border-radius:8px;
           padding:6px 12px;display:inline-block">
        <div style="font-size:.7rem;color:var(--fg2)">{t("gd_label")}</div>
        <div style="font-weight:800;color:{tc};font-size:.85rem">{entry['model'].get(lang, entry["model"]["en"])}</div>
        <div style="font-size:.65rem;color:var(--fg2);margin-top:3px">📋 {entry['data_needed']}</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.divider()

    # ── Seção 3: Interpretação de parâmetros ──────────────────
    st.markdown(f"### 📐 {t('gd_param_title')}")
    st.caption(t("gd_param_note"))
    st.markdown("<br>", unsafe_allow_html=True)

    for param in _PARAMS_INFO:
        lang = st.session_state.get("lang", "pt")
        with st.expander(f"**{param['variable'].get(lang, param['variable']['en'])}** — {param['category'].get(lang, param['category']['en'])}", expanded= False):
            pc1, pc2 = st.columns(2)
            pc1.markdown(f"""
<div style="background:var(--bg2);border:1px solid var(--brd);border-radius:8px;padding:10px 14px">
  <div style="font-size:.73rem;color:var(--fg2);margin-bottom:4px">📏 {t("gd_param_descript1")}</div>
  <div style="font-size:.85rem;color:var(--fg)">{param['value'].get(lang, param['value']['en'])}</div>
</div>
""", unsafe_allow_html=True)
            pc2.markdown(f"""
<div style="background:var(--bg2);border:1px solid var(--brd);border-radius:8px;padding:10px 14px">
  <div style="font-size:.73rem;color:var(--fg2);margin-bottom:4px">💡 {t("gd_param_descript2")}</div>
  <div style="font-size:.85rem;color:var(--fg)">{param['interpretation'].get(lang, param['interpretation']['en'])}</div>
</div>
""", unsafe_allow_html=True)

    st.divider()

    # ── Seção 4: Referências ──────────────────────────────────
    st.markdown(f"### 📚 {t('gd_refs_title')}")
    st.markdown("<br>", unsafe_allow_html=True)

    for cat in _REFERENCES:
        lang = st.session_state.get("lang", "pt")
        with st.expander(f" {cat['category'].get(lang, cat['category']['en'])}", expanded=False):
            for i, ref in enumerate(cat["refs"], 1):
                st.markdown(f"""
<div style="background:var(--bg2);border-left:3px solid var(--acc);
     border-radius:0 6px 6px 0;padding:8px 12px;margin-bottom:6px;font-size:.82rem;color:var(--fg)">
  <strong style="color:var(--acc2)">[{i}]</strong> {ref}
</div>
""", unsafe_allow_html=True)

    st.divider()
    # Quick usage tips
    st.markdown(t("gd_data"))
    st.markdown(t("gd_flux_recommended"))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 23. MAIN ROUTER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    inject_css()
    render_sidebar()
    with st.container():
        render_header()

    tab = st.session_state.get("tab", "data")
    if   tab == "data":    tab_data()
    elif tab == "models":  tab_models()
    elif tab == "tools":   tab_tools()
    elif tab == "results": tab_results()
    elif tab == "about":   tab_about()
    elif tab == "guide":   tab_guide()

    render_footer()


if __name__ == "__main__" or True:
    main()