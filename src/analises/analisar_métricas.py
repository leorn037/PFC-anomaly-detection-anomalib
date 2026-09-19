"""
Análise de métricas de detecção de anomalias — abordagem A (sessões curadas).

Carrega dois logs de inferência (uma sessão gravada só sobre trecho(s) normal(is),
outra só sobre trecho(s) com anomalia confirmada), calcula um conjunto amplo de
métricas de classificação binária, encontra thresholds candidatos (incluindo o
"ideal" quando há separação perfeita entre as classes) e gera os gráficos mais
usados nesse tipo de avaliação — prontos pra colar no TCC.

Uso:
    python analisar_metricas.py --normal caminho/metricas_normal.csv --anomalo caminho/metricas_anomalo.csv

Espera CSVs com pelo menos as colunas 'Score' (e opcionalmente 'Tempo_Inferencia_s'),
no mesmo formato gerado pelo seu log de inferência (delimitador ';').
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, matthews_corrcoef,
    balanced_accuracy_score, f1_score, ConfusionMatrixDisplay,
)
import sys
# Adiciona a pasta pai (src/) ao caminho de busca do Python
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models import MODEL_CONFIGS

def carregar_dados(caminho_normal, caminho_anomalo):
    df_normal = pd.read_csv(caminho_normal, delimiter=';')
    df_normal['Classe_Real'] = 0  # 0 = Normal

    df_anomalo = pd.read_csv(caminho_anomalo, delimiter=';')
    df_anomalo['Classe_Real'] = 1  # 1 = Anômalo

    df = pd.concat([df_normal, df_anomalo], ignore_index=True)
    return df, df_normal, df_anomalo


def estatisticas_descritivas(df_normal, df_anomalo):
    linhas = []
    for nome, df in [("Normal", df_normal), ("Anômalo", df_anomalo)]:
        s = df['Score']
        linhas.append({
            'Classe': nome, 'N': len(s), 'Média': s.mean(), 'Desvio Padrão': s.std(),
            'Mínimo': s.min(), 'Q1': s.quantile(0.25), 'Mediana': s.median(),
            'Q3': s.quantile(0.75), 'Máximo': s.max(),
        })
    return pd.DataFrame(linhas)


def separacao_perfeita(df_normal, df_anomalo):
    max_normal = df_normal['Score'].max()
    min_anomalo = df_anomalo['Score'].min()
    perfeita = max_normal < min_anomalo
    margem = min_anomalo - max_normal
    threshold_meio = (max_normal + min_anomalo) / 2 if perfeita else None
    return {
        'max_normal': max_normal, 'min_anomalo': min_anomalo,
        'perfeita': perfeita, 'margem': margem, 'threshold_meio_faixa': threshold_meio,
    }


def melhor_threshold_youden(y_true, scores):
    """Maximiza TPR - FPR (estatística de Youden) sobre a curva ROC."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thresholds[idx]), fpr, tpr, thresholds, float(j[idx])


def melhor_threshold_f1(y_true, scores):
    """Maximiza F1 sobre a curva Precisão-Recall."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    idx = int(np.argmax(f1s[:-1]))  # thresholds tem 1 elemento a menos que precisions/recalls
    return float(thresholds[idx]), float(f1s[idx])


def metricas_no_threshold(y_true, scores, threshold):
    y_pred = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precisao = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc_balanceada = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred) if len(set(y_pred)) > 1 else 0.0
    return {
        'threshold': threshold, 'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp),
        'Precisão': precisao, 'Recall (Sensibilidade)': recall,
        'Especificidade': especificidade, 'F1': f1,
        'Acurácia Balanceada': acc_balanceada, 'MCC': mcc, 'y_pred': y_pred,
    }


def plotar_distribuicao(df_normal, df_anomalo, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = 30
    ax.hist(df_normal['Score'], bins=bins, alpha=0.6, label='Normal', color='#2E8B57')
    ax.hist(df_anomalo['Score'], bins=bins, alpha=0.6, label='Anômalo', color='#B22222')
    ax.set_xlabel('Score de anomalia')
    ax.set_ylabel('Frequência (nº de frames)')
    ax.set_title('Distribuição do score de anomalia por classe')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'histograma_scores.png', dpi=150)
    plt.close(fig)


def plotar_boxplot(df, out_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    dados = [df[df['Classe_Real'] == 0]['Score'], df[df['Classe_Real'] == 1]['Score']]
    ax.boxplot(dados, tick_labels=['Normal', 'Anômalo'], showmeans=True)
    ax.set_ylabel('Score de anomalia')
    ax.set_title('Distribuição do score por classe (boxplot)')
    fig.tight_layout()
    fig.savefig(out_dir / 'boxplot_scores.png', dpi=150)
    plt.close(fig)


def plotar_roc(y_true, scores, auc, out_dir):
    fpr, tpr, _ = roc_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f'AUC = {auc:.4f}', color='#1f77b4', linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Classificador aleatório')
    ax.set_xlabel('Taxa de Falsos Positivos (1 - Especificidade)')
    ax.set_ylabel('Taxa de Verdadeiros Positivos (Recall)')
    ax.set_title('Curva ROC')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'curva_roc.png', dpi=150)
    plt.close(fig)


def plotar_precision_recall(y_true, scores, ap, out_dir):
    precisions, recalls, _ = precision_recall_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recalls, precisions, color='#d62728', linewidth=2, label=f'AP = {ap:.4f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precisão')
    ax.set_title('Curva Precisão-Recall')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'curva_precision_recall.png', dpi=150)
    plt.close(fig)


def plotar_threshold_sweep(y_true, scores, out_dir):
    thresholds = np.linspace(scores.min(), scores.max(), 200)
    precisoes, recalls, f1s = [], [], []
    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precisoes.append(p)
        recalls.append(r)
        f1s.append(f1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precisoes, label='Precisão')
    ax.plot(thresholds, recalls, label='Recall')
    ax.plot(thresholds, f1s, label='F1', linewidth=2)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Valor da métrica')
    ax.set_title('Precisão / Recall / F1 em função do threshold')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'threshold_sweep.png', dpi=150)
    plt.close(fig)


def plotar_matriz_confusao(y_true, y_pred, titulo, nome_arquivo, out_dir):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anômalo'])
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(titulo)
    fig.tight_layout()
    fig.savefig(out_dir / nome_arquivo, dpi=150)
    plt.close(fig)


def analisar(caminho_normal, caminho_anomalo, out_dir="analise_metricas", threshold_operacional=0.5, modelo = ""):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, df_normal, df_anomalo = carregar_dados(caminho_normal, caminho_anomalo)
    y_true = df['Classe_Real'].values
    scores = df['Score'].values

    linhas = []
    linhas.append("=" * 70)
    linhas.append(f"RELATÓRIO DE AVALIAÇÃO DO MODELO ({titulo})")
    linhas.append("=" * 70)
    linhas.append(f"Total de frames: {len(df)} ({len(df_normal)} normais, {len(df_anomalo)} anômalos)")
    linhas.append("")

    # 1. Estatísticas descritivas
    desc = estatisticas_descritivas(df_normal, df_anomalo)
    linhas.append("--- Estatísticas descritivas do score ---")
    linhas.append(desc.to_string(index=False))
    linhas.append("")

    # 2. Separação perfeita?
    sep = separacao_perfeita(df_normal, df_anomalo)
    linhas.append("--- Separabilidade das classes ---")
    linhas.append(f"Score máximo entre os normais:  {sep['max_normal']:.4f}")
    linhas.append(f"Score mínimo entre os anômalos: {sep['min_anomalo']:.4f}")
    if sep['perfeita']:
        linhas.append(f"SEPARAÇÃO PERFEITA — existe uma margem de {sep['margem']:.4f} sem nenhuma amostra "
                       f"de nenhuma classe.")
        linhas.append(f"   Threshold no meio da margem: {sep['threshold_meio_faixa']:.4f}")
    else:
        linhas.append("NÃO há separação perfeita — existe sobreposição entre as distribuições.")
    linhas.append("")

    # 3. AUC-ROC e Average Precision
    auc = roc_auc_score(y_true, scores)
    ap = average_precision_score(y_true, scores)
    linhas.append(f"AUC-ROC: {auc:.4f}  (1.0 = separação perfeita, 0.5 = aleatório)")
    linhas.append(f"Average Precision (área da curva PR): {ap:.4f}")
    linhas.append("")

    # 4. Thresholds candidatos
    thresh_youden, fpr, tpr, thresholds_roc, j_max = melhor_threshold_youden(y_true, scores)
    thresh_f1, f1_max = melhor_threshold_f1(y_true, scores)

    linhas.append("--- Thresholds candidatos ---")
    linhas.append(f"Operacional (usado ao vivo hoje): {threshold_operacional}")
    linhas.append(f"Ótimo por Youden's J (ROC):       {thresh_youden:.4f}  (J = {j_max:.4f})")
    linhas.append(f"Ótimo por F1 (Precisão-Recall):   {thresh_f1:.4f}  (F1 = {f1_max:.4f})")
    if sep['perfeita']:
        linhas.append(f"Meio da margem de separação:      {sep['threshold_meio_faixa']:.4f}  "
                       f"(mais robusto a ruído, se há separação perfeita)")
    linhas.append("")

    # 5. Métricas completas por threshold candidato
    candidatos = {
        'Operacional': threshold_operacional,
        'Youden (ROC)': thresh_youden,
        'F1 ótimo': thresh_f1,
    }
    if sep['perfeita']:
        candidatos['Meio da margem'] = sep['threshold_meio_faixa']

    linhas.append("--- Métricas completas por threshold ---")
    resultados = {}
    for nome, t in candidatos.items():
        m = metricas_no_threshold(y_true, scores, t)
        resultados[nome] = m
        linhas.append(f"\n[{nome}] threshold = {t:.4f}")
        linhas.append(f"  TN={m['TN']}  FP={m['FP']}  FN={m['FN']}  TP={m['TP']}")
        linhas.append(f"  Precisão: {m['Precisão']:.4f}  Recall: {m['Recall (Sensibilidade)']:.4f}  "
                       f"Especificidade: {m['Especificidade']:.4f}")
        linhas.append(f"  F1: {m['F1']:.4f}  Acurácia Balanceada: {m['Acurácia Balanceada']:.4f}  "
                       f"MCC: {m['MCC']:.4f}")

    linhas.append("")
    linhas.append("--- Classification report (threshold operacional) ---")
    y_pred_op = (scores >= threshold_operacional).astype(int)
    linhas.append(classification_report(y_true, y_pred_op, target_names=['Normal', 'Anômalo'], zero_division=0))

    # 6. Tempo de inferência, se existir a coluna
    if 'Tempo_Inferencia_s' in df.columns:
        t = df['Tempo_Inferencia_s']
        linhas.append("--- Tempo de inferência ---")
        linhas.append(f"Média: {t.mean():.4f}s | Mediana: {t.median():.4f}s | "
                       f"Mín: {t.min():.4f}s | Máx: {t.max():.4f}s | "
                       f"FPS médio equivalente: {1/t.mean():.2f}")
        linhas.append("")

    relatorio_txt = "\n".join(linhas)
    (out_dir / 'relatorio_metricas.txt').write_text(relatorio_txt, encoding='utf-8')
    print(relatorio_txt)

    # --- Gráficos ---
    plotar_distribuicao(df_normal, df_anomalo, out_dir)
    plotar_boxplot(df, out_dir)
    plotar_roc(y_true, scores, auc, out_dir)
    plotar_precision_recall(y_true, scores, ap, out_dir)
    plotar_threshold_sweep(y_true, scores, out_dir)

    for nome, m in resultados.items():
        nome_arquivo = ("matriz_confusao_" + nome.lower().replace(' ', '_')
                         .replace('(', '').replace(')', '') + ".png")
        plotar_matriz_confusao(y_true, m['y_pred'], f"Matriz de Confusão — {nome} (thr={m['threshold']:.4f})",
                                nome_arquivo, out_dir)

    print(f"\nRelatório e gráficos salvos em: {out_dir.resolve()}")


if __name__ == "__main__":

    anormal = r"C:\Users\Leonardo\Downloads\Programas\PFC\src\inference_results\inference_2026-09-17_15-52-22\metricas_deteccao.csv"
    normal = r"C:\Users\Leonardo\Downloads\Programas\PFC\src\inference_results\inference_2026-09-17_15-53-41\metricas_deteccao.csv"

    p = MODEL_CONFIGS["PatchCore"]["params"]
    layers_str = f"{p['layers'][0]}-{p['layers'][1]}"

    # Gerando a string do título
    titulo = f"PatchCore | {p['backbone']} | {layers_str} | Coreset: {p['coreset_sampling_ratio']} | k-NN: {p['num_neighbors']}"

    print(titulo)

    parser = argparse.ArgumentParser(description="Análise de métricas de detecção de anomalias (abordagem A)")
    parser.add_argument("--normal", type=str, default=normal, help="CSV da sessão só com trechos normais")
    parser.add_argument("--anomalo", type=str, default=anormal, help="CSV da sessão só com trechos com anomalia")
    parser.add_argument("--saida", type=str, default=r"analises\analise_metricas", help="Pasta de saída dos gráficos/relatório")
    parser.add_argument("--threshold_operacional", type=float, default=0.5,
                         help="Threshold usado ao vivo hoje, pra comparação")
    args = parser.parse_args()

    analisar(args.normal, args.anomalo, args.saida, args.threshold_operacional, titulo)