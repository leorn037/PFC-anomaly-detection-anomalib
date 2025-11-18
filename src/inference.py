import torch # Importar torch para converter imagens para tensor
import torchvision.transforms.v2 as v2 # Importar transforms para pré-processar imagens
from PIL import Image
import numpy as np
import time
from utils import Colors
import cv2

ANOMALY_SCORE = 1.0

def predict_image(model, image, transform, image_size):

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
    model.eval()

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
    import pickle
    from datetime import datetime
    from pathlib import Path
    import os

    # --- Configuração de Consenso ---
    CONSECUTIVE_ANOMALY_LIMIT = 5 # Limite de flags seguidas
    anomaly_streak = 0            # Contador de flags consecutivas
    is_anomaly_confirmed = False  # Flag para o sinal a ser enviado para a Pi
    
    image_size = config["image_size"]

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

    model.eval() # Coloca o modelo em modo de avaliação
    
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
            response = sock.recv(16).decode().strip()
            
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
        
    inference_count = 0
    # Loop principal de inferência

    while True:
        try:    
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
                packet = sock.recv(message_size - len(image_data))
                if not packet: break
                image_data += packet
            
            if not image_data: break
            
            # 3. Deserializa e decodifica a imagem
            encoded_image = pickle.loads(image_data)
            decoded_image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)

            # 4. Executa a inferência
            original_img_pil = Image.fromarray(cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB))
            _, anomaly_map, pred_score, pred_mask = predict_image(model, original_img_pil, transform_for_model, image_size)
            
            # 5. Obtém a flag de anomalia
            is_anomaly = check_for_anomaly_by_score(pred_score, threshold)
            # is_anomaly = check_for_anomaly_by_area(pred_mask, 100, threshold)

            # --- Lógica de Consenso (A Nova Lógica) ---
            if is_anomaly:
                anomaly_streak += 1
                if anomaly_streak >= CONSECUTIVE_ANOMALY_LIMIT:
                    is_anomaly_confirmed = True # Confirma que a anomalia é persistente
            else:
                anomaly_streak = 0 # Reseta a contagem se for um frame normal
                is_anomaly_confirmed = False # Garante que a flag seja desligada imediatamente se o score cair

            # Pós-processamento para visualização com OpenCV:
            anomaly_map_normalized = (anomaly_map * 255).astype(np.uint8)
            anomaly_map_resized = cv2.resize(anomaly_map_normalized, 
                                            (decoded_image.shape[1], decoded_image.shape[0]),
                                            interpolation=cv2.INTER_LINEAR)
    
            # Aplica um mapa de cores (heatmap) para melhor visualização
            anomaly_map_colored = cv2.applyColorMap(anomaly_map_resized, cv2.COLORMAP_JET)

            # Salva as imagens
            timestamp = int(time.time() * 1000)
            cv2.imwrite(str(inference_dir / f"original_{inference_count:04d}_{timestamp}.jpg"), decoded_image)
            cv2.imwrite(str(inference_dir / f"anomaly_map_{inference_count:04d}_{timestamp}_{pred_score:.4f}.jpg"), anomaly_map_colored)
            print(f"Imagens salvas para o frame {inference_count} com score {pred_score:.4f}.")

            inference_count += 1

            # Adiciona o score na imagem original para display
            cv2.putText(cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB), 
                        f"Score: {pred_score:.4f}", 
                        (10, 30), # Posição do texto
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1,        # Tamanho da fonte
                        (0, 255, 0), # Cor verde (BGR)
                        2)        # Espessura da linha

            combined_frame = np.hstack((cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB), anomaly_map_colored))
            
            if is_anomaly_confirmed:
                response = b'P'
                cv2.putText(combined_frame, "ANOMALIA - Confirmar (Y/N)?", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                cv2.imshow("Inferencia em Tempo Real (Original | Mapa de Anomalia)", combined_frame)
                # --- PAUSA (Bloqueante) ---
                # Espera indefinidamente (0) pela tecla 'y' ou 'n'
                key = cv2.waitKey(1) & 0xFF 
                if key == ord('y'):
                    print(f"[{Colors.GREEN}Operador{Colors.RESET}] Anomalia CONFIRMADA.")
                    is_anomaly_confirmed = True # Mantém a decisão
                    response = b'A'
                elif key == ord('N'):
                    print(f"[{Colors.YELLOW}Operador{Colors.RESET}] Anomalia REJEITADA (Falso Positivo).")
                    anomaly_streak = 0
                    is_anomaly_confirmed = False # SOBRESCREVE a decisão do modelo
                    response = b'N'
            else:
                response = b'N'
                # 4. Visualização
                cv2.imshow("Inferencia em Tempo Real (Original | Mapa de Anomalia)", combined_frame)

            # Saída ao pressionar 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f"{Colors.YELLOW}Saindo da inferência em tempo real.{Colors.RESET}")
                break
            
            # 6. Envia a flag de volta para a Raspberry Pi
            ##response = b'A' if is_anomaly_confirmed else b'N'
            sock.sendall(response)

            print(f"[{time.time() - start_time:.2f} s]Score: {pred_score:.4f}, Anomalia: {is_anomaly_confirmed}. Enviando flag...")

        except socket.timeout:
            print(f"[{Colors.YELLOW}Inferência{Colors.RESET}] Timeout. A Pi parou de enviar frames.")
        except ConnectionResetError:
            print(f"{Colors.YELLOW}Conexão com a Raspberry Pi encerrada. Reiniciando...{Colors.RESET}")
            break
        except Exception as e:
            print(f"{Colors.RED}Erro inesperado: {e}{Colors.RESET}")
            raise e
            break
            
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
    return pred_score > threshold

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
