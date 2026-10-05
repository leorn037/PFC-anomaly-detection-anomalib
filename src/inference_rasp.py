"""
Inferência leve na Raspberry Pi, usando SÓ o OpenVINO Runtime — sem PyTorch,
sem Anomalib, sem torchvision. Import deste módulo não carrega nenhuma
biblioteca pesada; só cv2, numpy, openvino e utils.Colors.
"""
import socket  
import struct  
import time
import cv2
import numpy as np
from openvino import Core  #!# não openvino.runtime — está deprecated (avisado no seu próprio log)
from utils import Colors
from cabo_tracker import CaboTracker


def preparar_modelo(model_xml_path):
    core = Core()
    compiled_model = core.compile_model(model=str(model_xml_path), device_name="CPU")
    
    output_layer = None
    
    # 1. Tenta achar a saída que se chama explicitamente "anomaly_map"
    for out in compiled_model.outputs:
        if "anomaly_map" in out.any_name.lower():
            output_layer = out
            break
            
    # 2. Se o nome foi apagado no export, busca a saída 2D/4D que seja do tipo FLOAT (f32)
    # Isso evita pegar a pred_mask, que geralmente é do tipo int (i32 ou u8)
    if output_layer is None:
        for out in compiled_model.outputs:
            p_shape = out.get_partial_shape()
            if p_shape.rank.is_static and p_shape.rank.get_length() in [2, 4]:
                if "f32" in out.get_element_type().get_type_name().lower() or "float" in out.get_element_type().get_type_name().lower():
                    output_layer = out
                    break
                    
    # Fallback de segurança
    if output_layer is None:
        output_layer = compiled_model.output(0)

    # Verifica formato de entrada
    input_node = compiled_model.input(0)
    in_p_shape = input_node.get_partial_shape()
    espera_nchw = False
    if in_p_shape.rank.is_static and in_p_shape.rank.get_length() == 4:
        if in_p_shape[1].is_static and in_p_shape[1].get_length() == 3:
            espera_nchw = True

    return compiled_model, output_layer, espera_nchw


def inferir_frame(compiled_model, output_layer, espera_nchw, frame_bgr):
    frame_float = frame_bgr.astype(np.float32) / 255.0
    if espera_nchw:
        input_tensor = np.expand_dims(np.transpose(frame_float, (2, 0, 1)), 0)
    else:
        input_tensor = np.expand_dims(frame_float, 0)

    print("=== DEBUG OPENVINO ===", flush=True)
    print("Shape:", input_tensor.shape, flush=True)
    print("Dtype:", input_tensor.dtype, flush=True)
    print("Contiguous:", input_tensor.flags["C_CONTIGUOUS"], flush=True)
    print("Nbytes:", input_tensor.nbytes, flush=True)
    print("======================", flush=True)


    results = compiled_model([input_tensor])
    print(3)
    # Extrai a matriz da camada selecionada
    anomaly_map = np.squeeze(results[output_layer])
    print(4)    
    # Se o anomaly_map tiver o shape correto (ex: 256x256), o score é o valor máximo dele
    if anomaly_map.ndim >= 2:
        pred_score = float(np.max(anomaly_map))
    else:
        # Caso o modelo realmente só tenha exportado o score global
        pred_score = float(anomaly_map)
        
    return anomaly_map, pred_score

def live_inference_rasp_lite(config, camera, model_xml_path, anomaly_output=None, move_output=None, pc_port=5005):
    """
    Roda inferência de anomalia diretamente na Raspberry Pi, sem depender do
    Anomalib/PyTorch — só o grafo OpenVINO já exportado (model.xml + model.bin).
    """
    compiled_model, output_layer, espera_nchw = preparar_modelo(model_xml_path)

    picam2 = camera
    image_size = config["image_size"]

    tracker = CaboTracker(crop_output_size=image_size)  #!# NOVO: aplicado antes de inferir, faltava isso

    sock_vis = None
    if config.get("network_inference", True):
        sock_vis = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock_vis.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)   #!# NOVO: habilita broadcast nesse socket

    threshold = config.get("anomaly_threshold", 0.5)

    inference_count = 0
    if move_output:
        print(f"[{Colors.YELLOW}ROBÔ{Colors.RESET}] Enviando sinal inicial: MOVER (HIGH)")
        move_output.on()

    print(f"\n{Colors.GREEN}{Colors.BOLD}--- Inferência Nativa OpenVINO (Raspberry Pi) Iniciada ---{Colors.RESET}")
    print(f" Pressione {Colors.YELLOW}Ctrl+C{Colors.RESET} no terminal para encerrar.\n")

    try:
        while True:
            t_start_loop = time.time()

            # 1. Captura do frame
            frame = picam2.capture_array()
            if frame is None:
                print(f"[{Colors.RED}ERRO{Colors.RESET}] Falha ao capturar frame da câmera.")
                continue
            
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 2. Rastreamento e recorte do cabo
            frame_processed = tracker.track(frame_bgr)

            # 3. Executa a inferência OpenVINO
            t_start_inf = time.time()
            #!anomaly_map, pred_score = inferir_frame(compiled_model, output_layer, espera_nchw, frame_processed)
            pred_score = 0.0
            anomaly_map = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2GRAY) * 0.0

            t_inf_duration = time.time() - t_start_inf

            is_anomaly = pred_score >= threshold
            print(4)
            # 4. Lógica de Atuação GPIO e Logs de Decisão
            if is_anomaly:
                if move_output: move_output.off()   #!# só PARA o robô — nada de atuador ainda
                # TODO: acionar o maçarico de verdade aqui quando o pipeline
                # TODO estiver validado. De propósito, hoje só imprime — evita
                # TODO acionar o atuador por engano enquanto ainda há risco de bug
                # TODO na detecção. Trocar esse print por uma chamada real (ex:
                # TODO enviar_pacote('E', 0) ou equivalente) só depois de confiar.
                print(f"{Colors.RED}[DEBUG] ANOMALIA (score={pred_score:.4f}) — robô PARADO, "
                      f"queima NÃO acionada (placeholder).{Colors.RESET}")
            else:
                if move_output: move_output.on()


            # Print mais limpo para frames normais para não inundar o terminal
            print(f"[{Colors.CYAN}INF{Colors.RESET}] Frame {inference_count:04d} | Score: {pred_score:.4f} | Inf: {t_inf_duration:.4f}s | Status: []")

            # 5. Envio UDP para visualização remota no PC
            if sock_vis:
                try:
                    _enviar_visualizacao_udp(sock_vis, pc_port, frame_processed, anomaly_map, pred_score)
                except Exception as net_err:
                    print(f"[{Colors.YELLOW}REDE-AVISO{CV.RESET}] Erro ao enviar UDP: {net_err}")

            inference_count += 1
        
            # ---> RESPIRAÇÃO DA CPU: Evita o travamento da câmera (timeout) <---
            elapsed = time.time() - t_start_loop
            if elapsed < 0.1:  # Garante um intervalo saudável para a placa respirar
                time.sleep(0.1 - elapsed)
    except KeyboardInterrupt:
        print(f"{Colors.YELLOW}Interrompido pelo usuário.{Colors.RESET}")
    finally:
        picam2.stop()
        print(f"{Colors.CYAN}Câmera liberada.{Colors.RESET}")
        if sock_vis:             
            sock_vis.close()     
              
def _enviar_visualizacao_udp(sock, pc_port, frame_bgr, anomaly_map, score):
    """Manda frame + mapa colorido pro PC via UDP, só pra visualização.
    Best-effort: qualquer falha é ignorada — nunca afeta a inferência real."""
    try:
        anomaly_map_norm = (np.clip(anomaly_map, 0, 1) * 255).astype(np.uint8)  #!# escala fixa, não NORM_MINMAX
        anomaly_colorido = cv2.applyColorMap(anomaly_map_norm, cv2.COLORMAP_JET)

        _, frame_jpg = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        _, mapa_jpg = cv2.imencode('.jpg', anomaly_colorido, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

        header = struct.pack("!fII", score, len(frame_jpg), len(mapa_jpg))
        payload = header + frame_jpg.tobytes() + mapa_jpg.tobytes()

        if len(payload) > 65000:   # limite prático de 1 datagrama UDP
            return  # pula esse frame de visualização, sem quebrar nada

        sock.sendto(payload, ('<broadcast>', pc_port))
    except Exception:
        pass
