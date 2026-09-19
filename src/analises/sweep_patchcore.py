"""
Grid search de hiperparâmetros do PatchCore.

Retreina para cada combinação de (backbone, layers, coreset_sampling_ratio), e
para cada uma dessas, reavalia com vários valores de num_neighbors SEM retreinar
(só o parâmetro do k-NN muda na hora da inferência, não o banco de memória).

Salva os resultados incrementalmente em CSV (retomável — pula combinações já
feitas se você rodar de novo), e mantém em disco só os N melhores checkpoints
por AUC-ROC, apagando o resto pra não lotar o disco.

Uso:
    python sweep_patchcore.py
(edite as constantes na seção CONFIGURAÇÕES abaixo antes de rodar)
"""
import csv
import itertools
import shutil
import time
import warnings
from pathlib import Path

import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

from utils import Colors, CONFIG as config
from models import setup_datamodule, create_patchcore_custom
from inference import predict_image

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import cv2
from cabo_tracker import CaboTracker


# ============================== CONFIGURAÇÕES ===============================
BACKBONES = ["resnet18", "resnet50", "wide_resnet50_2"]
LAYERS_OPTS = [("layer2", "layer3"),("layer1", "layer2")]
CORESET_RATIOS = [0.025, 0.05, 0.1]
NUM_NEIGHBORS_OPTS = [1, 3, 9]


DATASET_ROOT = Path(r"C:\Users\Leonardo\Downloads\Programas\PFC\src\train")  #Path(config["dataset_root"])          # pasta de treino (normais)
PASTA_TESTE_NORMAL = Path(r"C:\Users\Leonardo\Downloads\Programas\PFC\src\simulacao\normal")     #!# AJUSTE pro caminho real
PASTA_TESTE_ANOMALO = Path(r"C:\Users\Leonardo\Downloads\Programas\PFC\src\simulacao\abnormal")   #!# AJUSTE pro caminho real
IMAGE_SIZE = config["image_size"]

RESULTADOS_CSV = Path("sweep_resultados.csv")
CHECKPOINTS_DIR = Path("sweep_checkpoints")
MANTER_TOP_N = 3   # quantos checkpoints (por AUC) ficam salvos em disco ao final
# =============================================================================

COLUNAS_CSV = [
    "backbone", "layers", "coreset_sampling_ratio", "num_neighbors",
    "auc_roc", "max_score_normal", "min_score_anomalo", "separacao_perfeita",
    "margem_separacao", "melhor_f1", "threshold_melhor_f1",
    "tempo_treino_s", "tempo_inferencia_medio_s", "checkpoint_path", "erro",
]

def preparar_dataset_tracked(pasta_origem, pasta_destino, image_size):
    """
    Lê as imagens da origem, aplica o CaboTracker e salva no destino.
    Se a pasta de destino já existir com arquivos, pula para economizar tempo.
    """
    pasta_destino = Path(pasta_destino)
    arquivos_origem = listar_imagens(pasta_origem)
    
    # Se já processou antes e tem a mesma quantidade de arquivos, pula (cache)
    if pasta_destino.exists() and len(listar_imagens(pasta_destino)) == len(arquivos_origem):
        print(f"[{Colors.GREEN}Cache{Colors.RESET}] Pasta {pasta_destino.name} já processada pelo CaboTracker.")
        return pasta_destino

    pasta_destino.mkdir(parents=True, exist_ok=True)
    tracker = CaboTracker(crop_output_size=image_size)
    
    print(f"[{Colors.CYAN}Tracker{Colors.RESET}] Aplicando CaboTracker em {pasta_origem.name}...")
    
    for img_path in arquivos_origem:
        # Lê a imagem em BGR (padrão OpenCV, que o tracker espera)
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
            
        # Aplica o mesmo pré-processamento do robô
        frame_tracked = tracker.track(frame)
        
        # Salva na nova pasta
        caminho_salvar = pasta_destino / Path(img_path).name
        cv2.imwrite(str(caminho_salvar), frame_tracked)
        
    return pasta_destino

def montar_pre_processor(image_size):
    import torchvision.transforms.v2 as v2
    from anomalib.pre_processing import PreProcessor
    transform = v2.Compose([
        v2.Resize((image_size, image_size)),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return PreProcessor(transform=transform)


def montar_transform_avaliacao(image_size):
    import torchvision.transforms.v2 as v2
    return v2.Compose([
        v2.Resize((image_size, image_size)),
        v2.ToTensor(),
    ])


def listar_imagens(pasta):
    return sorted(list(Path(pasta).glob("*.jpg")) + list(Path(pasta).glob("*.png")))


def avaliar_modelo(modelo, transform, pasta_normal, pasta_anomalo, image_size):
    """Roda o modelo já treinado sobre as duas pastas curadas, retorna scores + tempos."""
    modelo.eval()

    def rodar_pasta(pasta):
        scores, tempos = [], []
        for img_path in listar_imagens(pasta):
            image = Image.open(img_path).convert("RGB")
            t0 = time.time()
            with torch.no_grad():
                _, anomaly_map, score, _ = predict_image(modelo, image, transform, image_size)
            
            '''# --- DEBUG VISUAL IGUAL AO MAIN.PY ---
            import cv2
            import numpy as np
            
            # Converte a PIL Image para padrão OpenCV (BGR)
            img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Prepara o mapa de calor
            mapa_norm = (anomaly_map * 255).astype(np.uint8)
            mapa_color = cv2.applyColorMap(mapa_norm, cv2.COLORMAP_JET)
            mapa_color = cv2.resize(mapa_color, (img_bgr.shape[1], img_bgr.shape[0]))
            
            # Junta as duas e escreve o score
            tela = np.hstack((img_bgr, mapa_color))
            cv2.putText(tela, f"Score: {score:.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("Debug Sweep - Aperte qualquer tecla para a proxima", tela)
            cv2.waitKey(0) # Pausa o código até você apertar uma tecla'''
            # =====================================
            tempos.append(time.time() - t0)
            scores.append(score)
        return scores, tempos

    scores_normal, tempos_normal = rodar_pasta(pasta_normal)
    scores_anomalo, tempos_anomalo = rodar_pasta(pasta_anomalo)

    if not scores_normal or not scores_anomalo:
        raise ValueError(f"Pasta de teste vazia: normal={len(scores_normal)}, anomalo={len(scores_anomalo)}")

    y_true = [0] * len(scores_normal) + [1] * len(scores_anomalo)
    scores = scores_normal + scores_anomalo
    tempo_medio = sum(tempos_normal + tempos_anomalo) / len(tempos_normal + tempos_anomalo)

    return y_true, scores, tempo_medio


def resumo_metricas(y_true, scores):
    auc = roc_auc_score(y_true, scores)

    max_normal = max(s for s, y in zip(scores, y_true) if y == 0)
    min_anomalo = min(s for s, y in zip(scores, y_true) if y == 1)
    separacao_perfeita = max_normal < min_anomalo
    margem = min_anomalo - max_normal

    precisions, recalls, thresholds_pr = precision_recall_curve(y_true, scores)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    idx_melhor_f1 = int(f1s[:-1].argmax()) if len(thresholds_pr) > 0 else 0
    melhor_f1 = float(f1s[idx_melhor_f1]) if len(thresholds_pr) > 0 else 0.0
    threshold_melhor_f1 = float(thresholds_pr[idx_melhor_f1]) if len(thresholds_pr) > 0 else 0.0

    return {
        "auc_roc": auc,
        "max_score_normal": max_normal,
        "min_score_anomalo": min_anomalo,
        "separacao_perfeita": separacao_perfeita,
        "margem_separacao": margem,
        "melhor_f1": melhor_f1,
        "threshold_melhor_f1": threshold_melhor_f1,
    }


def carregar_combinacoes_ja_feitas(csv_path):
    """Lê o CSV existente (se houver) e devolve o conjunto de (backbone, layers, coreset, neighbors) já feitos."""
    feitas = set()
    if not csv_path.exists():
        return feitas
    with open(csv_path, newline='', encoding='utf-8') as f:
        for linha in csv.DictReader(f):
            if linha.get("erro"):
                continue  # combinações que falharam podem ser tentadas de novo
            chave = (linha["backbone"], linha["layers"], linha["coreset_sampling_ratio"], linha["num_neighbors"])
            feitas.add(chave)
    return feitas


def escrever_linha_csv(csv_path, linha, primeira_vez):
    modo = 'a' if csv_path.exists() and not primeira_vez else 'w'
    with open(csv_path, mode=modo, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=COLUNAS_CSV)
        if modo == 'w':
            writer.writeheader()
        writer.writerow(linha)


def gerenciar_checkpoints_finais(csv_path, checkpoints_dir, manter_top_n):
    """Lê o CSV final, mantém só os top-N checkpoints por AUC, apaga o resto."""
    with open(csv_path, newline='', encoding='utf-8') as f:
        linhas = [l for l in csv.DictReader(f) if not l.get("erro") and l.get("checkpoint_path")]

    linhas.sort(key=lambda l: float(l["auc_roc"]), reverse=True)
    manter = {l["checkpoint_path"] for l in linhas[:manter_top_n]}

    for ckpt_dir in checkpoints_dir.iterdir():
        if ckpt_dir.is_dir() and str(ckpt_dir) not in manter:
            shutil.rmtree(ckpt_dir, ignore_errors=True)

    print(f"\n{Colors.GREEN}Checkpoints mantidos (top {manter_top_n} por AUC):{Colors.RESET}")
    for l in linhas[:manter_top_n]:
        print(f"  AUC={float(l['auc_roc']):.4f} | {l['backbone']} | {l['layers']} | "
              f"coreset={l['coreset_sampling_ratio']} | n_neighbors={l['num_neighbors']} -> {l['checkpoint_path']}")


def rodar_sweep():
    from anomalib.engine import Engine
    from anomalib.data import Folder
    from anomalib.data import Folder
    from anomalib.data.utils import TestSplitMode, ValSplitMode

    combinacoes_treino = list(itertools.product(BACKBONES, LAYERS_OPTS, CORESET_RATIOS))
    print(f"{Colors.CYAN}{len(combinacoes_treino)} combinações de treino x {len(NUM_NEIGHBORS_OPTS)} "
          f"valores de num_neighbors = {len(combinacoes_treino) * len(NUM_NEIGHBORS_OPTS)} linhas de resultado.{Colors.RESET}")

    ja_feitas = carregar_combinacoes_ja_feitas(RESULTADOS_CSV)
    primeira_escrita = not RESULTADOS_CSV.exists()

    NORMAL_TRACKED = preparar_dataset_tracked(PASTA_TESTE_NORMAL, "dataset_sweep_tracked/test/normal", IMAGE_SIZE)
    ANOMALO_TRACKED = preparar_dataset_tracked(PASTA_TESTE_ANOMALO, "dataset_sweep_tracked/test/abnormal", IMAGE_SIZE)

    print(f"{Colors.GREEN}Datasets prontos. Iniciando Sweep.{Colors.RESET}")

    # 2. Configura o Datamodule usando a SUA função original
    # Assim o sweep vai usar exatamente a mesma quantidade de imagens,
    # os mesmos splits (SYNTHETIC) e os mesmos transforms ocultos do seu código principal.
    
    # Adicione a mesma configuração de diretórios que o código normal usa
    config["folder_name"] = "dataset_rasp_sweep"
    config["normal_dir"] = str(DATASET_ROOT)
    
    datamodule = setup_datamodule(config)

    transform_avaliacao = montar_transform_avaliacao(IMAGE_SIZE)

    tempo_total_estimado_impresso = False

    for i, (backbone, layers, coreset_ratio) in enumerate(combinacoes_treino):
        layers_str = "-".join(layers)
        chaves_dessa_combo = [(backbone, layers_str, str(coreset_ratio), str(n)) for n in NUM_NEIGHBORS_OPTS]

        if all(chave in ja_feitas for chave in chaves_dessa_combo):
            print(f"[{i+1}/{len(combinacoes_treino)}] JÁ FEITO — pulando: {backbone}, {layers_str}, coreset={coreset_ratio}")
            continue

        print(f"\n{Colors.BLUE}[{i+1}/{len(combinacoes_treino)}] Treinando: backbone={backbone}, "
              f"layers={layers_str}, coreset_sampling_ratio={coreset_ratio}{Colors.RESET}")

        combo_dir = CHECKPOINTS_DIR / f"{backbone}_{layers_str}_{coreset_ratio}"

        try:
            pre_processor = montar_pre_processor(IMAGE_SIZE)
            modelo = create_patchcore_custom(backbone, layers, coreset_ratio, NUM_NEIGHBORS_OPTS[0], pre_processor)

            inicio = time.time()
            engine = Engine(logger=False, accelerator="cpu", max_epochs=1, default_root_dir=str(combo_dir))
            engine.fit(model=modelo, datamodule=datamodule)
            tempo_treino = time.time() - inicio

            if not tempo_total_estimado_impresso:
                estimativa_min = (tempo_treino * len(combinacoes_treino)) / 60
                print(f"{Colors.YELLOW}Estimativa grosseira do tempo total do sweep: "
                      f"~{estimativa_min:.1f} min (baseado só nesse primeiro treino, "
                      f"backbones maiores tendem a ser mais lentos).{Colors.RESET}")
                tempo_total_estimado_impresso = True

        except Exception as e:
            print(f"{Colors.RED}FALHOU o treino de {backbone}/{layers_str}/{coreset_ratio}: {e}{Colors.RESET}")
            for n in NUM_NEIGHBORS_OPTS:
                escrever_linha_csv(RESULTADOS_CSV, {
                    "backbone": backbone, "layers": layers_str, "coreset_sampling_ratio": coreset_ratio,
                    "num_neighbors": n, "erro": str(e),
                }, primeira_escrita)
                primeira_escrita = False
            continue

        scores_por_neighbor = {}
        for j, num_neighbors in enumerate(NUM_NEIGHBORS_OPTS):
            chave = (backbone, layers_str, str(coreset_ratio), str(num_neighbors))
            if chave in ja_feitas:
                continue

            modelo.model.num_neighbors = num_neighbors  #!# reavaliação sem retreinar

            try:
                inicio_infer = time.time()
                y_true, scores, tempo_medio = avaliar_modelo(
                    modelo, transform_avaliacao, NORMAL_TRACKED, ANOMALO_TRACKED, IMAGE_SIZE)
                metricas = resumo_metricas(y_true, scores)
                scores_por_neighbor[num_neighbors] = scores

                linha = {
                    "backbone": backbone, "layers": layers_str, "coreset_sampling_ratio": coreset_ratio,
                    "num_neighbors": num_neighbors, "tempo_treino_s": round(tempo_treino, 2),
                    "tempo_inferencia_medio_s": round(tempo_medio, 4),
                    "checkpoint_path": str(combo_dir), "erro": "",
                    **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in metricas.items()},
                }
                escrever_linha_csv(RESULTADOS_CSV, linha, primeira_escrita)
                primeira_escrita = False

                print(f"  num_neighbors={num_neighbors}: AUC={metricas['auc_roc']:.4f}  "
                      f"separação_perfeita={metricas['separacao_perfeita']}  F1={metricas['melhor_f1']:.4f}")

            except Exception as e:
                print(f"{Colors.RED}  FALHOU avaliação num_neighbors={num_neighbors}: {e}{Colors.RESET}")
                escrever_linha_csv(RESULTADOS_CSV, {
                    "backbone": backbone, "layers": layers_str, "coreset_sampling_ratio": coreset_ratio,
                    "num_neighbors": num_neighbors, "erro": str(e),
                }, primeira_escrita)
                primeira_escrita = False

        #!# Checagem de sanidade: se variar num_neighbors não muda NADA nos scores,
        #!# a otimização "reavalia sem retreinar" pode não estar funcionando nessa
        #!# versão do Anomalib — avisa em vez de deixar passar batido.
        if len(scores_por_neighbor) >= 2:
            vals = list(scores_por_neighbor.values())
            if all(v == vals[0] for v in vals[1:]):
                print(f"{Colors.YELLOW}⚠️  ATENÇÃO: scores idênticos pra todos os num_neighbors testados nessa "
                      f"combinação. Mudar model.model.num_neighbors pode não estar tendo efeito na sua versão "
                      f"do Anomalib — resultados de num_neighbors podem não ser confiáveis.{Colors.RESET}")

    gerenciar_checkpoints_finais(RESULTADOS_CSV, CHECKPOINTS_DIR, MANTER_TOP_N)
    print(f"\n{Colors.GREEN}Sweep concluído. Resultados em: {RESULTADOS_CSV.resolve()}{Colors.RESET}")


if __name__ == "__main__":
    rodar_sweep()