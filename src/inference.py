import torch # Importar torch para converter imagens para tensor
import torchvision.transforms.v2 as v2 # Importar transforms para pré-processar imagens
from PIL import Image
import numpy as np
import time
from utils import Colors
import cv2
from anomalib.deploy import OpenVINOInferencer

ANOMALY_SCORE = 1.0

def predict_image(model, image, transform, image_size):

    if isinstance(model, OpenVINOInferencer):
        # OpenVINO espera Numpy Array, mas sua entrada original é PIL
        image_numpy = np.array(image) if isinstance(image, Image.Image) else image
        
        # O OpenVINO faz o resize/normalize internamente usando metadata
        predictions = model.predict(image=image_numpy)
        
        # Extração dos resultados
        pred_score = predictions.pred_score.item()
        print(f"{pred_score=}")
        
        # Mapa de Calor (Heat Map)
        if predictions.anomaly_map is not None:
            anomaly_map = np.squeeze(predictions.anomaly_map)
        else:
            anomaly_map = np.zeros((image_size, image_size), dtype=np.float32)
        print(f'{anomaly_map.ndim=}')
        # Máscara de Segmentação
        if predictions.pred_mask is not None:
            pred_mask = np.squeeze((predictions.pred_mask.astype(np.uint8) * 255))
        else:
            pred_mask = np.zeros((image_size, image_size), dtype=np.uint8)
        print(f'{pred_mask.ndim=}')

    else:
        input_tensor = transform(image).unsqueeze(0) # Adiciona dimensão de batch
        with torch.no_grad():
            predictions = model.forward(input_tensor.to(model.device))

        # pred_score é um tensor (1,)
        pred_score = predictions.pred_score.cpu().item()

        try:
            # Se o modelo retornar um mapa de anomalia, use-o
            anomaly_map = predictions.anomaly_map.cpu().squeeze().numpy()
        except AttributeError:
            # Se o modelo NÃO retornar, crie um mapa preto (array de zeros)
            anomaly_map = np.zeros((image_size, image_size), dtype=np.float32)
            
        # 3. Extrair ou criar placeholder para a máscara de anomalia
        try:
            # Se o modelo retornar uma máscara, use-a
            pred_mask = predictions.pred_mask.cpu().squeeze().numpy().astype(np.uint8) * 255
        except AttributeError:
            # Se o modelo NÃO retornar, crie uma máscara preta (array de zeros)
            pred_mask = np.zeros((image_size, image_size), dtype=np.uint8)
        
    return image, anomaly_map, pred_score, pred_mask

def apply_pred_mask_on_image(image, pred_mask, color=(0,0,255)):

# O parâmetro alpha ajusta a transparência da máscara sobre a imagem.
# Para destacar ainda mais as bordas, use outline=True e, opcionalmente, altere a espessura do contorno.
# Adapte para coloração diferente conforme a classe de anomalia (se for o caso).

    # Converte imagem para RGB se estiver em formato PIL ou BGR
    if isinstance(image, np.ndarray):
        img = image.copy()
    else:  # Se for PIL, converte para np.ndarray
        img = np.array(image)
        if img.shape[2] == 4:  # Remove canal alpha se existir
            img = img[:, :, :3]

    # Redimensione a máscara para o tamanho da imagem
    pred_mask_resized = cv2.resize(pred_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_bool = pred_mask_resized > 255

    # Se quiser borda, desenhe contorno em volta de regiões anômalas:
    contours, _ = cv2.findContours(pred_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, 1)

    return img

def visualize_imgs(path_dir, model, img_class, image_size):
    """
    Processa e visualiza imagens de anomalia usando OpenCV.
    
    Args:
        path_dir (Path): Caminho do diretório contendo as imagens.
        model: O modelo de detecção de anomalias.
        img_class (str): O nome da classe de imagem (ex: "normal", "anomalia").
        image_size (int): O tamanho da imagem para redimensionamento.
    """
    if hasattr(model, 'eval'): model.eval() # Coloca o modelo em modo de avaliação

    # Certifique-se de que as transformações correspondem às usadas no datamodule
    transform = v2.Compose([
        v2.Resize((image_size, image_size)),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if path_dir.exists() and path_dir.is_dir():
        path_images = sorted(list(path_dir.glob("*.jpg")))
        if not path_images:
            print(f"{Colors.YELLOW}Aviso: Nenhuma imagem .jpg encontrada em {path_dir}{Colors.RESET}")
        
        for i, img_path in enumerate(path_images):
            print(f"\n{Colors.YELLOW}Processando imagem ({img_class}) {i+1}/{len(path_images)}: {img_path}{Colors.RESET}")
            image = Image.open(img_path).convert("RGB")
            original_img, anomaly_map, score, pred_mask = predict_image(model, image, transform, image_size) 

            # --- Visualização com OpenCV ---
            # Converte a imagem original para o formato BGR para exibição com cv2
            original_img_cv2 = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
            original_img_cv2 = apply_pred_mask_on_image(original_img_cv2, pred_mask, color=(0,0,255))

            # Redimensiona o mapa de anomalia para o tamanho da imagem original
            anomaly_map_resized = cv2.resize(anomaly_map, (original_img_cv2.shape[1], original_img_cv2.shape[0]))
            print(2)
            # Normaliza o mapa de anomalia para o intervalo 0-255 e aplica um colormap
            anomaly_map_normalized = (anomaly_map_resized * 255).astype(np.uint8)
            anomaly_map_color = cv2.applyColorMap(anomaly_map_normalized, cv2.COLORMAP_JET)

            # Concatena as duas imagens (original e mapa de anomalia) lado a lado
            combined_img = np.hstack([original_img_cv2, anomaly_map_color])
            
            # Exibe a imagem combinada
            window_name = f"Original ({img_class}) - Score: {score:.4f} | Anomaly Map"
            
            cv2.imshow(window_name, combined_img)

            print(f"{Colors.CYAN}Pressione qualquer tecla para a próxima imagem...{Colors.RESET}")
            cv2.waitKey(0) # Espera por uma tecla

            cv2.destroyAllWindows() # Fecha todas as janelas abertas
            
            expected_status = f"{Colors.GREEN}Baixa (ex: < 0.5){Colors.RESET}"
            print(f"Pontuação de anomalia esperada ({img_class}): {expected_status}, Obtido: {Colors.GREEN}{score:.4f}{Colors.RESET}")
            
    else:
        print(f"{Colors.YELLOW}Aviso: Diretório de imagens normais não encontrado ou não é uma pasta: {path_dir}.{Colors.RESET}")

def live_inference_opencv(model, image_size):
    """
    Realiza inferência em tempo real usando a câmera e exibe os resultados
    usando OpenCV para visualização.

    Args:
        model (torch.nn.Module): O modelo Anomalib carregado para inferência.
        image_size (int): O tamanho para redimensionar a imagem de entrada do modelo.
        device (str): O dispositivo para onde enviar o modelo e os tensores ('cuda' ou 'cpu').
    """
    # 1. Configuração da Câmera
    cap = cv2.VideoCapture(0)  # 0 geralmente se refere à câmera padrão do sistema
    if not cap.isOpened():
        print(f"{Colors.RED}Erro: Não foi possível abrir a câmera. Verifique se a câmera está conectada e disponível.{Colors.RESET}")
        return

    # 2. Pré-processamento: As mesmas transformações usadas no treinamento/inferência do dataset
    transform_for_model = v2.Compose([
        v2.Resize((image_size, image_size)),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model.eval() # Coloca o modelo em modo de avaliação
    print(f"{Colors.GREEN}Iniciando inferência em tempo real com OpenCV. Pressione 'q' para sair.{Colors.RESET}")

    try:
        while True:
            ret, frame = cap.read() # Lê um frame da câmera (BGR)
            if not ret:
                print(f"{Colors.RED}Erro: Não foi possível ler o frame da câmera. Saindo...{Colors.RESET}")
                break

            # Converter o frame OpenCV (BGR) para PIL RGB para o modelo
            frame_rgb_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Clona o frame original para exibir ao lado do mapa de anomalia
            original_frame_display = frame.copy() 

            start_time=time.time()
            original_img, anomaly_map, pred_score, pred_mask = predict_image(model, frame_rgb_pil, transform_for_model, image_size)
            print(f"Tempo: {time.time()-start_time:.4f}, {pred_score:4f} / Score: {check_for_anomaly_by_score(pred_score, 0.5)} / Area: {check_for_anomaly_by_area(pred_mask, 100, 0.5)}")

            original_frame_display = apply_pred_mask_on_image(original_frame_display, pred_mask, color=(0,0,255))

            # Pós-processamento para visualização com OpenCV:
            # Redimensiona o mapa de anomalia para o tamanho do frame original
            # Normaliza o mapa de 0-1 para 0-255 (necessário para cv2.applyColorMap)
            anomaly_map_normalized = (anomaly_map * 255).astype(np.uint8)
            anomaly_map_resized = cv2.resize(anomaly_map_normalized, 
                                             (frame.shape[1], frame.shape[0]), # (largura, altura)
                                             interpolation=cv2.INTER_LINEAR)
            
            # Aplica um mapa de cores (heatmap) para melhor visualização
            anomaly_map_colored = cv2.applyColorMap(anomaly_map_resized, cv2.COLORMAP_JET)

            # Adiciona o score na imagem original para display
            cv2.putText(original_frame_display, 
                        f"Score: {pred_score:.4f}", 
                        (10, 30), # Posição do texto
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1,        # Tamanho da fonte
                        (0, 255, 0), # Cor verde (BGR)
                        2)        # Espessura da linha

            # Concatena as imagens horizontalmente para exibir lado a lado
            # Certifique-se de que ambas as imagens tenham a mesma altura antes de concatenar
            # if anomaly_map_colored.shape[0] != original_frame_display.shape[0]:
            #     # Isso não deve acontecer se redimensionamos corretamente, mas é uma verificação de segurança
            #     min_height = min(anomaly_map_colored.shape[0], original_frame_display.shape[0])
            #     original_frame_display = cv2.resize(original_frame_display, (original_frame_display.shape[1], min_height))
            #     anomaly_map_colored = cv2.resize(anomaly_map_colored, (anomaly_map_colored.shape[1], min_height))

            combined_frame = np.hstack((original_frame_display, anomaly_map_colored))

            # 4. Visualização
            cv2.imshow("Inferência em Tempo Real (Original | Mapa de Anomalia)", combined_frame)

            # Saída ao pressionar 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f"{Colors.YELLOW}Saindo da inferência em tempo real.{Colors.RESET}")
                break

    except Exception as e:
        print(f"{Colors.RED}Ocorreu um erro durante a inferência em tempo real: {e}{Colors.RESET}")

    finally:
        cap.release() # Libera a câmera
        cv2.destroyAllWindows() # Fecha todas as janelas do OpenCV

def live_inference_rasp(model, config, camera):
    """
    Realiza inferência em tempo real usando a Raspberry.

    Args:
        model (torch.nn.Module): O modelo Anomalib carregado para inferência.
        image_size (int): O tamanho para redimensionar a imagem de entrada do modelo.
        device (str): O dispositivo para onde enviar o modelo e os tensores ('cuda' ou 'cpu').
    """
    import os

    # 1. Configuração da Câmera
    picam2 = camera # Cria uma instância do controle da câmera
    image_size = config["image_size"]
    # Configuração de pré-visualização com resolução de 640x480 pixels
    config2 = picam2.create_preview_configuration(main={"size": (image_size, image_size), "format": "BGR888"})
    picam2.configure(config2)

    # Inicia a câmera e espera ficar estabilizada
    picam2.start()

    # 2. Pré-processamento: As mesmas transformações usadas no treinamento/inferência do dataset
    transform_for_model = v2.Compose([
        v2.Resize((image_size, image_size)),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model.eval() # Coloca o modelo em modo de avaliação

    # 3. Configuração do Socket para Envio
    
    if config["websocket"]:
        import socket
        import pickle
        # --- Configurações UDP ---
        UDP_IP = config["pc_ip"] # IP do SEU PC (cliente)! A Raspberry Pi vai enviar para este IP.
        UDP_PORT = 5005        # Porta para onde a Pi vai ENVIAR os dados.
        # Nota: O IP do servidor da Pi é '0.0.0.0' para escutar, mas o cliente precisa do IP real da Pi.
        # Aqui, a Pi é o REMETENTE UDP, então precisa do IP do DESTINATÁRIO (seu PC).

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 

        print(f"{Colors.CYAN}Iniciando servidor UDP na Raspberry Pi. Enviando para {UDP_IP}:{UDP_PORT}{Colors.RESET}")
        print(f"{Colors.GREEN}Iniciando inferência em tempo real na Raspberry. Pressione 'q' para sair.{Colors.RESET}")

    try:
        while True:
            # Captura um frame da câmera como um array
            frame = picam2.capture_array()

            # Clona o frame original para exibir ao lado do mapa de anomalia
            original_frame_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Converter o frame OpenCV (BGR) para PIL RGB para o modelo
            frame_rgb_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            start_time=time.time()
            original_img, anomaly_map, pred_score, pred_mask = predict_image(model, frame_rgb_pil, transform_for_model, image_size)
            print(f"Tempo: {time.time()-start_time:.4f}, {pred_score:4f} / Score: {check_for_anomaly_by_score(pred_score, 0.9)} / Area: {check_for_anomaly_by_area(pred_mask, 100, 0.9)}")
            
            original_frame_display = apply_pred_mask_on_image(original_frame_display, pred_mask, color=(0,0,255))

            # Pós-processamento para visualização com OpenCV:
            # Redimensiona o mapa de anomalia para o tamanho do frame original
            # Normaliza o mapa de 0-1 para 0-255 (necessário para cv2.applyColorMap)
            anomaly_map_normalized = (anomaly_map * 255).astype(np.uint8)
            anomaly_map_resized = cv2.resize(anomaly_map_normalized, 
                                             (frame.shape[1], frame.shape[0]), # (largura, altura)
                                             interpolation=cv2.INTER_LINEAR)
            
            # Aplica um mapa de cores (heatmap) para melhor visualização
            anomaly_map_colored = cv2.applyColorMap(anomaly_map_resized, cv2.COLORMAP_JET)
                        
            if config["websocket"]:
                # Prepara os dados para envio: frame original, mapa colorido e score
                # Usaremos o frame original (BGR) do OpenCV e o mapa de anomalia colorido
                _, original_encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70]) # Qualidade 70 (0-100)
                # anomaly_map_colored é BGR
                _, anomaly_encoded = cv2.imencode('.jpg', anomaly_map_colored, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

                data_to_send = {
                    'original_frame': original_encoded.tobytes(), # Envie bytes JPEG
                    'anomaly_map': anomaly_encoded.tobytes(),     # Envie bytes JPEG
                    'score': pred_score
                }

                # Serializa com pickle
                serialized = pickle.dumps(data_to_send)

                # Divide em pacotes menores se necessário
                for i in range(0, len(serialized), 65536):
                    sock.sendto(serialized[i:i+65536], (UDP_IP, UDP_PORT))

                time.sleep(0.1)
            else:
                # Adiciona o score na imagem original para display
                cv2.putText(original_frame_display, 
                            f"Score: {pred_score:.4f}", 
                            (10, 30), # Posição do texto
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1,        # Tamanho da fonte
                            (0, 255, 0), # Cor verde (BGR)
                            2)        # Espessura da linha

                combined_frame = np.hstack((original_frame_display, anomaly_map_colored))

                # 4. Visualização
                cv2.imshow("Inferencia em Tempo Real (Original | Mapa de Anomalia)", combined_frame)

                # Salva no disco para comparar
                cv2.imwrite("bgr_frame.jpg", frame)  # sem conversão
                cv2.imwrite("rgb_frame.jpg", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # convertido


                # Saída ao pressionar 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print(f"{Colors.YELLOW}Saindo da inferência em tempo real.{Colors.RESET}")
                    break

    except KeyboardInterrupt:
        print(f"{Colors.YELLOW}Ctrl+C detectado. Encerrando ...{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}Ocorreu um erro durante o envio ou inferência: {e}{Colors.RESET}")
    finally:
        picam2.stop()
        if config["websocket"]:
            sock.close()
        if hasattr(cv2, "destroyAllWindows") and os.environ.get("DISPLAY"):
            cv2.destroyAllWindows()
        print(f"{Colors.YELLOW}Câmera e socket liberados.{Colors.RESET}")

def serve_inference_to_pi(model, config, sock, threshold=0.9):
    """
    Recebe um stream de imagens via TCP, executa inferência e envia uma flag de anomalia.

    Args:
        model (torch.nn.Module): O modelo Anomalib carregado para inferência.
        pc_ip (str): IP do PC (geralmente '0.0.0.0' para escutar todas as interfaces).
        pc_port (int): A porta TCP para escutar.
        image_size (int): O tamanho da imagem para inferência.
    """
    import struct
    import socket
    from datetime import datetime
    from pathlib import Path
    import os
    import threading
    import queue

    # --- Configuração de Consenso ---
    CONSECUTIVE_ANOMALY_LIMIT = 1 # Limite de flags seguidas
    anomaly_streak = 0            # Contador de flags consecutivas
    inference_count = 0

    image_size = config["image_size"]

    # CONFIGURAÇÃO DA THREAD
    # Cria a fila. 'maxsize=0' significa tamanho infinito (cuidado com memória RAM)
    image_save_queue = queue.Queue(maxsize=50)

    def worker_save_images():
        """
        Esta função roda em uma linha do tempo paralela (Thread).
        Ela fica em loop infinito esperando algo cair na fila para salvar.
        """
        print(f"[{Colors.GREEN}THREAD{Colors.RESET}] Worker de salvamento iniciado.")
        
        while True:
            # 1. Pega o item da fila (Bloqueia se estiver vazia, esperando chegar algo)
            item = image_save_queue.get()
            
            if item is None: break # Sinal para matar a thread quando fechar o programa
            
            # Desempacota os dados que você mandou
            filepath, image = item
            
            try:
                # 2. A operação lenta acontece AQUI, sem travar o robô
                cv2.imwrite(filepath, image)
                # print(f"Salvo: {filepath}") # Opcional (prints gastam tempo também)
            except Exception as e:
                print(f"Erro ao salvar imagem: {e}")
            finally:
                # Avisa a fila que esse item foi processado
                image_save_queue.task_done()

    # Inicia a Thread assim que o programa começa
    save_thread = threading.Thread(target=worker_save_images, daemon=True)
    save_thread.start()
    # --- Lógica de criação de diretório de inferência ---
    # 1. Obtém a data e hora atuais
    current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 2. Cria o caminho para a nova pasta de inferência
    inference_dir = Path("inference_results") / f"inference_{current_timestamp}"
    # 3. Cria o diretório (se ele não existir)
    os.makedirs(inference_dir, exist_ok=True)
    print(f"[{Colors.GREEN}Diretório de Salvamento{Colors.RESET}] Criando pasta para resultados de inferência: {inference_dir}")

    # Pré-processamento: as mesmas transformações usadas no treinamento
    transform_for_model = v2.Compose([
        v2.Resize((image_size, image_size)),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if hasattr(model, 'eval'): model.eval() # Coloca o modelo em modo de avaliação
    
    if not sock:
        print(f"[{Colors.RED}Erro{Colors.RESET}] Conexão de rede não estabelecida. Abortando coleta.")
        return

    command = "M"
    ack_received = False
    while not ack_received:
        try:
            print(f"[{Colors.CYAN}SYNC-PC{Colors.RESET}] Enviando comando '{command}' para a Pi...")
            sock.sendall(command.encode('utf-8'))
            
            sock.settimeout(None) # Timeout para a resposta ACK
            response = sock.recv(3).decode().strip()
            
            if response == "ACK":
                print(f"[{Colors.GREEN}SYNC{Colors.RESET}] Confirmação (ACK) recebida. Comando '{command}' concluído.")
                ack_received = True
            else:
                print(f"[{Colors.YELLOW}SYNC{Colors.RESET}] Resposta inválida da Pi: {response}. ")
                time.sleep(3)

        except (ConnectionRefusedError, socket.timeout) as e:
            print(f"[{Colors.RED}SYNC-PC{Colors.RESET}] Falha na conexão ao sincronizar 'M': {e}")
        except Exception as e:
            print(f"[{Colors.RED}SYNC-PC{Colors.RESET}] Erro inesperado: {e}")
            return
        
    # Loop principal de inferência
    decision_buffer = None
    try:    
        while True:
            start_time = time.time()
            # 1. Recebe o tamanho da imagem
            image_size_data = sock.recv(4)
            if not image_size_data: 
                print(f"[{Colors.YELLOW}Inferência{Colors.RESET}] Pi encerrou o stream.")
                break
            message_size = struct.unpack("!I", image_size_data)[0]
            
            # 2. Recebe a imagem completa
            image_data = b''
            while len(image_data) < message_size:
                packet = sock.recv(min(message_size - len(image_data), 4096))
                if not packet: break
                image_data += packet
            
            if not image_data: break

# --- RAIO-X DOS BYTES (COLOQUE ISSO AQUI) ---
            print(f"-> Esperado: {message_size} bytes | Recebido: {len(image_data)} bytes")
            if len(image_data) >= 4:
                # Todo JPEG no planeta TEM que começar com \xff\xd8
                print(f"-> Assinatura do Arquivo (Primeiros 4 bytes): {image_data[:4]}")
            
            # Trava de segurança: Se não recebeu tudo, aborta esse frame!
            if len(image_data) < message_size:
                print(f"[{Colors.YELLOW}REDE{Colors.RESET}] Pacote incompleto. Ignorando frame.")
                sock.sendall(b'N') 
                continue
            # -------------------------------------------
            
            print(message_size)
            
            # Transforma os bytes crus da rede diretamente em um array numpy
            np_arr = np.frombuffer(image_data, dtype=np.uint8)
            # Decodifica o JPEG para Matriz de Imagem (BGR)
            decoded_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # 4. Executa a inferência
            original_img_pil = Image.fromarray(cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB))
            _, anomaly_map, pred_score, pred_mask = predict_image(model, original_img_pil, transform_for_model, image_size)
            
            # 5. Obtém a flag de anomalia
            is_anomaly = check_for_anomaly_by_score(pred_score, threshold)
            # is_anomaly = check_for_anomaly_by_area(pred_mask, 100, threshold)

            # !LÓGICA DE DECISÃO E RESPOSTA
            response = b'N'
            # CASO 1: O humano apertou algo no frame anterior?
            if decision_buffer == 'A':
                response = b'A' # Trava total
                decision_buffer = None
                print(f"[{Colors.RED}Operador{Colors.RESET}] Anomalia CONFIRMADA.")
            elif decision_buffer == 'N':
                response = b'N' # Libera
                anomaly_streak = 0 # Reseta contagem da IA
                decision_buffer = None # Consumimos o comando, volta a monitorar
                print(f"[{Colors.YELLOW}Operador{Colors.RESET}] Anomalia REJEITADA (Falso Positivo).")
            else:
                # --- Lógica de Consenso ---
                if is_anomaly:
                    anomaly_streak += 1
                    if anomaly_streak >= CONSECUTIVE_ANOMALY_LIMIT:
                        response = b'P'
                        print("Pausa para confirmação")
                else:
                    anomaly_streak = 0 # Reseta a contagem se for um frame normal

            # Envio imediato da resposta
            sock.sendall(response)

            # Pós-processamento para visualização com OpenCV:
            anomaly_map_normalized = (anomaly_map * 255).astype(np.uint8)
            anomaly_map_resized = cv2.resize(anomaly_map_normalized, 
                                            (decoded_image.shape[1], decoded_image.shape[0]),
                                            interpolation=cv2.INTER_LINEAR)
    
            # Aplica um mapa de cores (heatmap) para melhor visualização
            anomaly_map_colored = cv2.applyColorMap(anomaly_map_resized, cv2.COLORMAP_JET)

            # Salva as imagens
            timestamp = int(time.time() * 1000)
            fname_original = str(inference_dir / f"{inference_count:04d}_{timestamp}_original.jpg")
            fname_map = str(inference_dir / f"{inference_count:04d}_{timestamp}_anomaly_map_{pred_score:.4f}.jpg")
                                  
            try:
                image_save_queue.put((fname_original, decoded_image.copy()), block=False)
                image_save_queue.put((fname_map, anomaly_map_colored.copy()), block=False)
                print(f"Imagens enviadas para fila (Frame {inference_count})")
            except queue.Full:
                print("Aviso: Fila de salvamento cheia, pulando este frame.")
            
            #cv2.imwrite(fname_original, decoded_image)
            #cv2.imwrite(fname_map, anomaly_map_colored)
            #print(f"Imagens salvas para o frame {inference_count} com score {pred_score:.4f}.")

            inference_count += 1

            decoded_image = apply_pred_mask_on_image(decoded_image, pred_mask, color=(0,0,255))

            combined_frame = np.hstack((decoded_image, anomaly_map_colored))
            new_size = 640
            combined_frame = cv2.resize(combined_frame, (2*new_size,new_size), interpolation=cv2.INTER_LINEAR)

            # Desenha status na tela
            color = (0,0,255) if response in [b'P', b'A'] else (0,255,0)
            text = f"Score: {pred_score:.4f}"
            cv2.putText(combined_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            # Adiciona o score na imagem original para display
            cv2.putText(cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB), text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if response == b'P':
                cv2.putText(combined_frame, "ANOMALIA - Confirmar (Y/N)?", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
            
            cv2.imshow("Inferencia (Original | Mapa de Anomalia)", combined_frame)
            
            # LEITURA DE TECLADO 
            key = cv2.waitKey(1) & 0xFF 

            # Saída ao pressionar 'q'
            if key == ord('q'):
                sock.sendall(b'Q')
                print(f"{Colors.YELLOW}Saindo da inferência em tempo real.{Colors.RESET}")
                break
            elif response in [b'P', b'A']:
                # Se apertar tecla, armazena para o PRÓXIMO loop
                if key == ord('y'):
                    decision_buffer = 'A'
                elif key == ord('n'):
                    decision_buffer = 'N'

            is_anomaly = response in [b'P', b'A']
            print(f"[{time.time() - start_time:.2f} s]Score: {pred_score:.4f}, Anomalia: {is_anomaly}. Enviando flag...")

    except socket.timeout:
        print(f"[{Colors.YELLOW}Inferência{Colors.RESET}] Timeout. A Pi parou de enviar frames.")
    except ConnectionResetError:
        print(f"{Colors.YELLOW}Conexão com a Raspberry Pi encerrada. Reiniciando...{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}Erro inesperado: {e}{Colors.RESET}")
        raise e
    finally:
        # Limpeza da Thread
        image_save_queue.put(None)
        save_thread.join()       
        cv2.destroyAllWindows()
    
# Abordagem é a mais simples.
# Pode ser menos eficaz para anomalias pequenas que não elevam significativamente o score total da imagem.
def check_for_anomaly_by_score(pred_score: float, threshold: float = ANOMALY_SCORE) -> bool:
    """
    Verifica se uma imagem contém anomalias com base em seu score de anomalia.

    Args:
        pred_score: O score de anomalia da imagem inteira.
        threshold: O limiar para classificar uma imagem como anômala.

    Returns:
        True se a imagem for anômala, False caso contrário.
    """
    return pred_score >= threshold

# Mais sensível a anomalias localizadas, independentemente do score total da imagem.
# Pode ser sensível a ruído, detectando pequenos grupos de pixels anômalos que não representam um defeito real.
def check_for_anomaly_by_area(pred_mask: np.ndarray, min_area_threshold: int = 100, threshold: float = ANOMALY_SCORE) -> bool:
    """
    Verifica se a área de anomalia na máscara é maior que um limiar de pixels.

    Args:
        pred_mask: A máscara de anomalia binária.
        min_area_threshold: O número mínimo de pixels de anomalia para classificar.

    Returns:
        True se a área de anomalia for maior que o limiar, False caso contrário.
    """
    anomaly_pixels = np.sum(pred_mask > threshold)
    return anomaly_pixels > min_area_threshold
