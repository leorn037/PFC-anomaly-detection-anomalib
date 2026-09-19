import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carrega os dados do Grid Search
csv_path = 'analises\sweep_resultados.csv'
df = pd.read_csv(csv_path)

print("="*70)
print(" RELATÓRIO DE ANÁLISE DO GRID SEARCH — PATCHCORE")
print("="*70)
print(f"Total de combinações testadas: {len(df)}")

# 2. Top 5 Modelos com Maior AUC-ROC (Foco em Precisão Pura)
print("\n--- [1] TOP 5 MODELOS POR AUC-ROC (Mais precisos) ---")
top_auc = df.sort_values('auc_roc', ascending=False)[
    ['backbone', 'layers', 'coreset_sampling_ratio', 'num_neighbors', 'auc_roc', 'melhor_f1', 'tempo_inferencia_medio_s']
].head(5)
print(top_auc.to_string(index=False))

# 3. Melhor Custo-Benefício (Equilíbrio entre AUC alta e baixa latência)
# Criamos um índice simples: AUC dividida pelo tempo de inferência
df['indice_eficiencia'] = df['auc_roc'] / df['tempo_inferencia_medio_s']
print("\n--- [2] TOP 5 MELHOR CUSTO-BENEFÍCIO (Ideais para Raspberry Pi) ---")
top_eff = df.sort_values('indice_eficiencia', ascending=False)[
    ['backbone', 'layers', 'coreset_sampling_ratio', 'num_neighbors', 'auc_roc', 'tempo_inferencia_medio_s']
].head(5)
print(top_eff.to_string(index=False))

# 4. Resumo Médio por Tipo de Backbone
print("\n--- [3] DESEMPENHO MÉDIO POR BACKBONE ---")
resumo_backbone = df.groupby('backbone')[['auc_roc', 'melhor_f1', 'tempo_inferencia_medio_s']].mean().reset_index()
print(resumo_backbone.to_string(index=False))

# 5. Resumo Médio por Camadas
print("\n--- [4] DESEMPENHO MÉDIO POR CAMADAS ---")
resumo_layers = df.groupby('layers')[['auc_roc', 'melhor_f1', 'tempo_inferencia_medio_s']].mean().reset_index()
print(resumo_layers.to_string(index=False))

# 6. Geração de Gráfico Científico para o TCC (Trade-off: Acurácia vs Latência)
print("\n[Gráfico] Gerando gráfico de dispersão (Trade-off)...")
plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df, 
    x='tempo_inferencia_medio_s', 
    y='auc_roc', 
    hue='backbone', 
    style='layers', 
    s=120, 
    palette='Set1',
    alpha=0.8
)
plt.title('Trade-off: Acurácia (AUC-ROC) vs Latência (Tempo de Inferência)', fontsize=12, fontweight='bold')
plt.xlabel('Tempo Médio de Inferência por Frame (s)', fontsize=10)
plt.ylabel('AUC-ROC', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
plt.tight_layout()

grafico_path = 'analises/tradeoff_auc_latencia.png'
plt.savefig(grafico_path, dpi=300)
plt.close()
print(f"[Sucesso] Gráfico salvo com alta resolução em: '{grafico_path}'")
print("="*70)