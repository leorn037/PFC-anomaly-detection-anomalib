"""
Calibração interativa do CaboTracker.

Uso:
    python calibrar_cabo_tracker.py                       # câmera ao vivo
    python calibrar_cabo_tracker.py --pasta caminho/imgs   # revisa imagens já salvas

Controles (também aparecem na própria janela, no painel "Corte final"):
    [n] / [espaço] -> próxima imagem (aceita o resultado atual como "memória" de entrada
                      da próxima imagem, simulando frame seguinte de vídeo)
    [p]            -> imagem anterior (volta usando a memória com que ela foi processada
                      originalmente, não o que os sliders atuais produziriam)
    [r]            -> reset completo do tracker (e da memória da sessão de calibração)
    [s]            -> imprime no terminal os valores atuais dos sliders
    [q] / [ESC]    -> sair

Sliders numa janela separada ("Controles"). Só ROI/Tolerância/Fallback/HistN
ficam ajustáveis aqui — margin_percent e MIN_VOTOS_CONFIANCA saíram dos sliders
(ver comentário perto de criar_sliders() se quiser saber por quê / como reativar).

Qualquer erro durante o processamento (inclusive no 'r') é capturado e impresso
no terminal com traceback completo, em vez de derrubar o programa — assim dá
pra ver a causa real e me mandar o erro, se acontecer de novo.
"""
import argparse
import glob
import os
import time
import traceback
import cv2
import numpy as np
from cabo_tracker import CaboTracker

try:
    from picamera2 import Picamera2

    def setup_camera(image_size):
        picam2 = Picamera2()
        config = picam2.create_video_configuration(main={"size": (image_size, image_size)})
        picam2.configure(config)
        picam2.start()
        return picam2

    def get_frame(camera, image_size):
        frame = camera.capture_array()
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

except ImportError:
    def setup_camera(image_size):
        return cv2.VideoCapture(0)

    def get_frame(camera, image_size):
        ret, frame = camera.read()
        if not ret:
            return None
        return cv2.resize(frame, (image_size, image_size))


# --- Cores (BGR) ---
COR_ROI = (255, 255, 0)      # ciano
COR_CANDIDATO_ESQ = (255, 0, 255)  # magenta
COR_CANDIDATO_DIR = (0, 0, 255)    # vermelho
COR_CORTE = (0, 130, 0)      # verde escuro
COR_CENTRO = (0, 140, 255)   # laranja (linha fina)
COR_INFO = (0, 255, 255)     # amarelo — texto de info, com contorno preto
COR_LEGENDA_CTRL = (0, 0, 255)  # vermelho — legenda de controles

JANELA_IMG = "Calibracao CaboTracker"
JANELA_CTRL = "Controles"
LARGURA_MIN_PAINEL_CORTE = 280  # painel do corte final nunca fica mais estreito que isso (espaço pra legenda)

NOME_ROI = "ROI"
NOME_TOLER = "Toler%"
NOME_FALLBACK = "Fallback"
NOME_HIST_N = "HistN"
#!# Margin% e MinVotos saíram dos sliders (ver criar_sliders)


def nothing(_):
    pass


def criar_sliders(tracker):
    cv2.namedWindow(JANELA_CTRL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(JANELA_CTRL, 380, 200)

    cv2.createTrackbar(NOME_ROI, JANELA_CTRL, tracker.roi_window, 300, nothing)
    cv2.createTrackbar(NOME_TOLER, JANELA_CTRL, int(tracker.LARGURA_TOLERANCIA * 100), 100, nothing)
    cv2.createTrackbar(NOME_FALLBACK, JANELA_CTRL, tracker.LARGURA_FALLBACK, 300, nothing)
    cv2.createTrackbar(NOME_HIST_N, JANELA_CTRL, tracker.HISTORICO_LARGURA_N, 50, nothing)
    #!# margin_percent removido do slider: fica fixo no valor de __init__ (0.05), não é mais editável aqui.
    #!# MIN_VOTOS_CONFIANCA removido do slider: só afeta se um frame "conta" pro aprendizado da
    #!# largura, sem efeito visual direto no corte — se quiser tunar, edita tracker.MIN_VOTOS_CONFIANCA
    #!# direto no cabo_tracker.py.
    # Canny/Hough continuam hardcoded dentro de _centralizar_cabo. Pra tunar ao vivo, expõe
    # como atributos no __init__ do CaboTracker e adiciona os trackbars aqui.


def aplicar_sliders(tracker):
    tracker.roi_window = max(1, cv2.getTrackbarPos(NOME_ROI, JANELA_CTRL))
    tracker.LARGURA_TOLERANCIA = max(0.01, cv2.getTrackbarPos(NOME_TOLER, JANELA_CTRL) / 100)
    tracker.LARGURA_FALLBACK = max(1, cv2.getTrackbarPos(NOME_FALLBACK, JANELA_CTRL))
    tracker.HISTORICO_LARGURA_N = max(1, cv2.getTrackbarPos(NOME_HIST_N, JANELA_CTRL))


def snapshot_memoria(tracker):
    return {
        "last_center_x": tracker.last_center_x,
        "last_width": tracker.last_width,
        "largura_estimada": tracker.largura_estimada,
        "bootstrap_larguras": list(tracker.bootstrap_larguras),
        "historico_larguras_confiaveis": list(tracker.historico_larguras_confiaveis),
    }


def restaurar_memoria(tracker, memoria):
    tracker.last_center_x = memoria["last_center_x"]
    tracker.last_width = memoria["last_width"]
    tracker.largura_estimada = memoria["largura_estimada"]
    tracker.bootstrap_larguras = list(memoria["bootstrap_larguras"])
    tracker.historico_larguras_confiaveis = list(memoria["historico_larguras_confiaveis"])


def desenhar_texto_com_contorno(img, texto, org, escala, cor, espessura_texto=1, espessura_contorno=2):
    """Contorno preto atrás + texto colorido por cima — legível em qualquer fundo, sem tapar a imagem."""
    cv2.putText(img, texto, org, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), espessura_contorno, cv2.LINE_AA)
    cv2.putText(img, texto, org, cv2.FONT_HERSHEY_SIMPLEX, escala, cor, espessura_texto, cv2.LINE_AA)


def desenhar_overlay(frame, tracker, left, right, legenda_extra=""):
    overlay = frame.copy()
    h, w = overlay.shape[:2]

    if tracker.last_center_x is not None:
        x_start = max(0, tracker.last_center_x - tracker.roi_window)
        x_end = min(w, tracker.last_center_x + tracker.roi_window)
        cv2.rectangle(overlay, (x_start, 0), (x_end, h), COR_ROI, 3)

    for cx in tracker.vertical_lines:
        cor = COR_CANDIDATO_ESQ if cx < tracker.last_center_x else COR_CANDIDATO_DIR
        cv2.line(overlay, (cx, 0), (cx, h), cor, 1)

    cv2.line(overlay, (left, 0), (left, h), COR_CORTE, 2)
    cv2.line(overlay, (right, 0), (right, h), COR_CORTE, 2)

    centro_cabo = (left + right) // 2
    cv2.line(overlay, (centro_cabo, 0), (centro_cabo, h), COR_CENTRO, 1)
    cv2.circle(overlay, (centro_cabo, 20), 5, COR_CENTRO, -1)

    largura = right - left
    n_esq, n_dir = len(tracker.left_candidates), len(tracker.right_candidates)
    if tracker.largura_estimada is not None:
        status_largura = f"largura_estimada={tracker.largura_estimada:.1f}px (mediana/{len(tracker.historico_larguras_confiaveis)})"
    else:
        status_largura = f"calibrando({len(tracker.bootstrap_larguras)}/{tracker.BOOTSTRAP_N})"

    #!# getattr com default: só existe se você já tiver aplicado o self.ultima_correcao no cabo_tracker.py
    #!# (ver explicação sobre a Tolerância). Sem isso aplicado, mostra "?" em vez de quebrar o script.
    correcao = getattr(tracker, "ultima_correcao", "?")

    info = (f"L={left} R={right} W={largura}px | CentroCabo={centro_cabo} | ROI=+-{tracker.roi_window} "
            f"| cand: esq={n_esq} dir={n_dir} | {status_largura} | correcao={correcao}")

    desenhar_texto_com_contorno(overlay, info, (10, h - 15), 0.5, COR_INFO)
    if legenda_extra:
        desenhar_texto_com_contorno(overlay, legenda_extra, (10, h - 40), 0.5, COR_INFO)

    return overlay


def desenhar_legenda_no_painel(painel, modo_pasta):
    linhas = ["CONTROLES:"]
    if modo_pasta:
        linhas += ["n/espaco: proxima (aceita memoria)  |  p: anterior (memoria original)"]
    linhas += ["r: reset  |  s: imprimir valores  |  q/ESC: sair",
               "",
               "CORES:  ciano=ROI  magenta/vermelho=candidatos",
               "verde escuro=corte final  laranja=centro do cabo"]

    y = painel.shape[0] - 16 * len(linhas) - 10
    for linha in linhas:
        desenhar_texto_com_contorno(painel, linha, (8, y), 0.42, COR_LEGENDA_CTRL,
                                     espessura_texto=1, espessura_contorno=2)
        y += 18


def montar_painel_corte(cropped, altura_total):
    """Cola o recorte real (o que de fato viraria imagem de treino/inferência) num
    painel preto, garantindo largura mínima pra caber a legenda."""
    h_crop, w_crop = cropped.shape[:2]
    largura_final = max(w_crop, LARGURA_MIN_PAINEL_CORTE)
    painel = np.zeros((altura_total, largura_final, 3), dtype=np.uint8)

    h_a_colar = min(h_crop, altura_total)
    y_off = max(0, (altura_total - h_crop) // 2)
    x_off = max(0, (largura_final - w_crop) // 2)
    painel[y_off:y_off + h_a_colar, x_off:x_off + w_crop] = cropped[:h_a_colar]

    desenhar_texto_com_contorno(painel, "Corte final", (10, 25), 0.6, (0, 255, 0))
    return painel


def montar_canvas(overlay, cropped, modo_pasta):
    painel_corte = montar_painel_corte(cropped, overlay.shape[0])
    desenhar_legenda_no_painel(painel_corte, modo_pasta)
    return np.hstack([overlay, painel_corte])


def imprimir_valores(tracker):
    print("--- Valores atuais ---")
    print(f"roi_window          = {tracker.roi_window}")
    print(f"margin_percent      = {tracker.margin_percent} (fixo, não é mais slider)")
    print(f"LARGURA_TOLERANCIA  = {tracker.LARGURA_TOLERANCIA}")
    print(f"LARGURA_FALLBACK    = {tracker.LARGURA_FALLBACK}")
    print(f"HISTORICO_LARGURA_N = {tracker.HISTORICO_LARGURA_N}")
    print(f"MIN_VOTOS_CONFIANCA = {tracker.MIN_VOTOS_CONFIANCA} (fixo, não é mais slider)")
    print(f"largura_estimada (agora) = {tracker.largura_estimada}")


def calibrar_camera(image_size=640):
    camera = setup_camera(image_size)
    if hasattr(camera, "start"):
        camera.start()

    tracker = CaboTracker(crop_output_size=None)
    cv2.namedWindow(JANELA_IMG, cv2.WINDOW_NORMAL)
    criar_sliders(tracker)
    cv2.resizeWindow(JANELA_IMG, image_size + LARGURA_MIN_PAINEL_CORTE, image_size + 60)

    while True:
        frame = get_frame(camera, image_size)
        if frame is None:
            print("Erro ao ler frame da câmera.")
            break

        aplicar_sliders(tracker)
        cropped, left, right = tracker._centralizar_cabo(frame, None, debug=False)
        overlay = desenhar_overlay(frame, tracker, left, right)
        canvas = montar_canvas(overlay, cropped, modo_pasta=False)
        cv2.imshow(JANELA_IMG, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('r'):
            tracker.reset()
        elif key == ord('s'):
            imprimir_valores(tracker)

    if hasattr(camera, "stop"):
        camera.stop()
    elif isinstance(camera, cv2.VideoCapture):
        camera.release()
    cv2.destroyAllWindows()


def calibrar_pasta(pasta, image_size=640):
    arquivos = sorted(glob.glob(os.path.join(pasta, "*.jpg")) + glob.glob(os.path.join(pasta, "*.png")))
    if not arquivos:
        print(f"Nenhuma imagem encontrada em: {pasta}")
        return

    tracker = CaboTracker(crop_output_size=None)
    cv2.namedWindow(JANELA_IMG, cv2.WINDOW_NORMAL)
    criar_sliders(tracker)
    print(f"{len(arquivos)} imagens encontradas.")

    idx = 0
    janela_redimensionada = False
    memoria_base = snapshot_memoria(tracker)
    historico_memoria = {0: dict(memoria_base)}

    while True:
        frame = cv2.imread(arquivos[idx])
        if frame is None:
            print(f"Não foi possível ler: {arquivos[idx]}")
            idx = min(idx + 1, len(arquivos) - 1)
            continue

        h, w = frame.shape[:2]
        if not janela_redimensionada:
            cv2.resizeWindow(JANELA_IMG, w + LARGURA_MIN_PAINEL_CORTE, h + 60)
            janela_redimensionada = True

        aplicar_sliders(tracker)
        restaurar_memoria(tracker, memoria_base)
        cropped, left, right = tracker._centralizar_cabo(frame, None, debug=True)

        nome = os.path.basename(arquivos[idx])
        overlay = desenhar_overlay(frame, tracker, left, right,
                                    legenda_extra=f"[{idx+1}/{len(arquivos)}] {nome}")
        canvas = montar_canvas(overlay, cropped, modo_pasta=True)
        cv2.imshow(JANELA_IMG, canvas)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key in (ord('n'), ord(' ')):
            memoria_base = snapshot_memoria(tracker)
            idx = min(idx + 1, len(arquivos) - 1)
            if idx not in historico_memoria:
                historico_memoria[idx] = dict(memoria_base)
            else:
                memoria_base = dict(historico_memoria[idx])
        elif key == ord('p'):
            idx = max(idx - 1, 0)
            memoria_base = dict(historico_memoria.get(idx, snapshot_memoria(CaboTracker())))
        elif key == ord('r'):
            tracker.reset()
            memoria_base = snapshot_memoria(tracker)
            historico_memoria = {idx: dict(memoria_base)}
        elif key == ord('s'):
            imprimir_valores(tracker)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibração interativa do CaboTracker")
    parser.add_argument("--pasta", type=str, default=None,
                         help="Pasta com imagens já salvas (se omitido, usa a câmera ao vivo)")
    parser.add_argument("--image_size", type=int, default=640)
    args = parser.parse_args()

    calibrar_pasta("C:\\Users\\Leonardo\\Downloads\\Programas\\PFC\\src\\novo\\images_2026-07-14_16-47-27", args.image_size)

    '''
    images_2026-07-14_16-48-23 - A
    images_2026-07-14_16-47-27 - A
    images_2026-07-14_16-43-26 - N
    '''
    if args.pasta:
        calibrar_pasta(args.pasta, args.image_size)
    else:
        calibrar_camera(args.image_size)
