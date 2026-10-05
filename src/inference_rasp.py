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

    results = compiled_model([input_tensor])
    
    # Extrai a matriz da camada selecionada
    anomaly_map = np.squeeze(results[output_layer])
    
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

    print(f"{Colors.GREEN}Inferência leve (OpenVINO puro) iniciada. Ctrl+C pra sair.{Colors.RESET}")

    try:
        while True:
            start_time = time.time()
            frame = picam2.capture_array()  # (H, W, C), já em BGR (format="BGR888")
            frame = tracker.track(frame)

            anomaly_map, pred_score = inferir_frame(compiled_model, output_layer, espera_nchw, frame)
            is_anomaly = pred_score >= threshold

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

            elapsed = time.time() - start_time
            print(f"[{elapsed:.3f}s] Score: {pred_score:.4f}")

            if sock_vis:
                 _enviar_visualizacao_udp(sock_vis, pc_port, frame, anomaly_map, pred_score)
 

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
