import socket
import os
import struct
import pickle
import cv2
from pathlib import Path
from utils import Colors
import time
from scipy.signal import find_peaks
import numpy as np

def crop_and_resize(
    frame_bgr: np.ndarray, 
    size: int | None, 
    ) -> np.ndarray:
    """
    Localiza o cabo usando detecção de bordas verticais e análise de perfil,
    e o centraliza dentro de um corte quadrado, redimensionado para o tamanho final.

    Args:
        frame_bgr: Imagem de entrada em formato BGR (OpenCV).
        crop_output_size: Tamanho final do lado do quadrado (ex: 256).
        debug_show_steps: Se True, mostra as imagens intermediárias para depuração.

    Returns:
        np.ndarray: O frame final (cortado e redimensionado).
    """
    h, w, _ = frame_bgr.shape

    # --- LIMITES RÍGIDOS PARA AS COORDENADAS X ---
    X_MIN_ALLOWED = 250 #214
    X_MAX_ALLOWED = 392 #437
    Y_CUT = 0
    
    # Reduz a altura da análise para ignorar o chão e o topo (ROI Vertical)
    Y_START_ROI = int(h * 0.2) 
    Y_END_ROI = int(h * 0.8)

    # 3. Encontrar Picos (Bordas)
    PEAK_HEIGHT_THRESHOLD = 60_000 
    PEAK_MIN_DISTANCE = 30
    MIN_CABLE_WIDTH = 125

    frame_roi = frame_bgr[Y_START_ROI:Y_END_ROI, :]
    # 60_000 k_size 5 frame_roi
    # 15_000 k_size 3 frame_bgr

    # 1. Pré-processamento: Tons de Cinza
    gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)

    # Criamos um retângulo largo e baixo. Ele vai "borrar" tudo que estiver perto na horizontal, fechando os buracos entre os fios do cabo.
    # (15, 3) significa 15 pixels de largura por 3 de altura.
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))

    # O 'Morph Close' é uma Dilatação seguida de Erosão. Ele preenche vazios escuros.
    morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_morph)

    # 2. Suavização para reduzir ruído (Gaussian Blur é bom para isso)
    blurred = cv2.GaussianBlur(morph, (5, 5), 0) # Ajuste o tamanho do kernel se necessário
    
    # 3. Detecção de Bordas Verticais (Sobel X):
    # O cabo é um objeto vertical. O operador Sobel na direção X destacará suas bordas verticais.
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5) # Derivada na direção X
    sobelx = cv2.convertScaleAbs(sobelx) # Converte de volta para 8-bit

    # 4. Análise do Perfil Horizontal para encontrar a coluna mais "ativa"
    # Somamos os valores de pixel de cada coluna. O cabo terá um pico de bordas.
    horizontal_profile = np.sum(sobelx, axis=0)  
    
    # Filtra os picos com a altura e distância mínima
    peaks, properties = find_peaks(horizontal_profile, 
                                   height=PEAK_HEIGHT_THRESHOLD, 
                                   distance=PEAK_MIN_DISTANCE)
    
    center_x = w // 2 # Centro padrão de fallback
    center_y = h // 2

    valid_peaks = peaks[(peaks >= X_MIN_ALLOWED) & (peaks <= X_MAX_ALLOWED)]
    

    left_peaks_in_roi = valid_peaks[valid_peaks < center_x]
    if left_peaks_in_roi.size > 0:
        borda_esquerda = left_peaks_in_roi[-1]
    else: 
        borda_esquerda = X_MIN_ALLOWED
            
    # Encontre o pico de maior valor (borda mais forte) na direita permitida
    right_peaks_in_roi = valid_peaks[valid_peaks > center_x]
    if right_peaks_in_roi.size > 0:
        borda_direita = right_peaks_in_roi[0]
    else:
        borda_direita = X_MAX_ALLOWED
    
    if borda_direita - borda_esquerda < MIN_CABLE_WIDTH:
        # Se ficaram muito perto, expande artificialmente
        borda_direita = min(borda_esquerda + MIN_CABLE_WIDTH, X_MAX_ALLOWED)
        borda_esquerda = max(X_MIN_ALLOWED, borda_direita - MIN_CABLE_WIDTH)
        print(f"[AJUSTE] Largura do cabo forçada para o mínimo: {MIN_CABLE_WIDTH} pixels.")

    # Se a borda detectada estiver fora do limite permitido, usamos o limite.
    borda_esquerda = max(X_MIN_ALLOWED, borda_esquerda)
    borda_direita = min(X_MAX_ALLOWED, borda_direita)

    # Verifica se os limites se cruzaram após o clamping
    if borda_esquerda >= borda_direita:
        print(f"[ERRO] Limites de Clamping se cruzaram! Usando limites seguros: ({X_MIN_ALLOWED}, {X_MAX_ALLOWED})")
        borda_esquerda = X_MIN_ALLOWED
        borda_direita = X_MAX_ALLOWED

    center_x = (borda_esquerda + borda_direita) // 2

    # 3. CORTE HORIZONTAL PELAS BORDAS
    # O corte vertical (altura) permanece na imagem inteira, mas pode ser ajustado
    x_crop_start = borda_esquerda+int(0.1*( borda_direita - borda_esquerda))
    x_crop_end = borda_direita-int(0.1*(borda_direita - borda_esquerda))

    cropped_frame = frame_bgr[Y_CUT:h-Y_CUT, x_crop_start:x_crop_end]

    if size: 
        resized_frame = cv2.resize(cropped_frame, (size, size), interpolation=cv2.INTER_LINEAR)
        return resized_frame
    else:
        return cropped_frame

def send_tcp_frame(sock: socket.socket, frame: np.ndarray, quality: int = 70):
    """
    Codifica (JPEG), serializa (pickle) e envia um frame (com cabeçalho de tamanho)
    através de um socket TCP.

    Levanta uma exceção (BrokenPipeError, etc.) se a conexão falhar.

    Args:
        sock (socket.socket): O socket de conexão TCP ativo.
        frame (np.ndarray): O frame de imagem (array NumPy BGR) a ser enviado.
        quality (int): A qualidade do JPEG (0-100).
    """
    if not sock:
        # Se a rede estiver desabilitada (sock is None), não faz nada.
        return

    # 1. Codifica o frame em formato JPEG para compressão
    # Isso reduz o tamanho da imagem antes de enviar pela rede.
    _, encoded_image = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    
    # 2. Serializa os dados (pickle)
    # Converte o array NumPy da imagem codificada em um fluxo de bytes.
    data = pickle.dumps(encoded_image)
    
    # 3. Prepara o cabeçalho de tamanho (4 bytes, unsigned int)
    # Isso informa ao receptor (PC) exatamente quantos bytes ele deve esperar.
    message_size = struct.pack("!I", len(data))
    
    # 4. Envia o tamanho da mensagem (cabeçalho) + os dados (payload)
    # sendall garante que todos os bytes sejam enviados.
    sock.sendall(message_size + data)

def pi_socket(pi_port):
    # Se a rede estiver habilitada, a Pi atua como Servidor
    try:
        host = '0.0.0.0'
        port = pi_port
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((host, port))
        server_sock.listen(1)
        
        print(f"[{Colors.CYAN}Servidor{Colors.RESET}] Aguardando conexão do PC em {host}:{port}...")
        conn, addr = server_sock.accept()
        print(f"[{Colors.GREEN}Servidor{Colors.RESET}] Conexão estabelecida com {addr}.")
        return conn, server_sock
    except Exception as e:
        print(f"[{Colors.RED}Erro{Colors.RESET}] Falha ao iniciar o servidor: {e}. Rodando em modo Offline.")
        return None, None

def pi_connect(pi_ip, pi_port):
        
        while True: # Loop de retry da conexão principal
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2.0) # Timeout para a tentativa de conexão
                print(f"[{Colors.CYAN}Rede-PC{Colors.RESET}] Conectando à Pi em {pi_ip}:{pi_port}...")
                sock.connect((pi_ip, pi_port))
                print(f"[{Colors.GREEN}Rede-PC{Colors.RESET}] Conexão principal estabelecida.")
                return sock # Sucesso, sai do loop de retry
            except (ConnectionRefusedError, socket.timeout, socket.gaierror) as e:
                print(f"[{Colors.RED}Rede-PC{Colors.RESET}] Falha ao conectar ({e}). A Pi está escutando? Tentando novamente em 5s...")
                time.sleep(3)

# Função para receber todas as imagens e salvar
def receive_all_images_and_save(num_images: int, save_path: Path, sock: socket.socket):
    """
    Recebe um número específico de imagens via TCP, decodifica e salva em um diretório.
    
    Args:
        num_images (int): Número total de imagens a serem recebidas.
        save_path (Path): O caminho para o diretório onde as imagens serão salvas.
        pc_port (int): A porta TCP que o PC irá escutar.
    """
    save_path.mkdir(parents=True, exist_ok=True)
    
    # --- ETAPA 1: Sincronização (Lógica da antiga 'send_command_and_wait_ack') ---
    
    if not sock:
        print(f"[{Colors.RED}Erro{Colors.RESET}] Conexão de rede não estabelecida. Abortando coleta.")
        return
    
    command = "P"
    ack_received = False
    while not ack_received:
        try:
            print(f"[{Colors.CYAN}SYNC-PC{Colors.RESET}] Enviando comando '{command}' para a Pi...")
            sock.sendall(command.encode('utf-8'))
            
            sock.settimeout(10.0) # Timeout para a resposta ACK
            response = sock.recv(16).decode().strip()
            
            if response == "ACK":
                print(f"[{Colors.GREEN}SYNC{Colors.RESET}] Confirmação (ACK) recebida. Comando '{command}' concluído.")
                ack_received = True
            else:
                print(f"[{Colors.YELLOW}SYNC{Colors.RESET}] Resposta inválida da Pi: {response}. ")
                time.sleep(3)

        except (ConnectionRefusedError, socket.timeout) as e:
            print(f"[{Colors.RED}SYNC-PC{Colors.RESET}] Falha na conexão ao sincronizar 'P': {e}")
        except Exception as e:
            print(f"[{Colors.RED}SYNC-PC{Colors.RESET}] Erro inesperado: {e}")
            return "DISCONNECTED"

    print(f"{Colors.GREEN}Recebendo {num_images} imagens...{Colors.RESET}")
    print(f"{Colors.GREEN}Pressione '{Colors.CYAN}c{Colors.RESET}' para iniciar a coleta de {num_images} imagens.{Colors.RESET}")
    print(f"{Colors.GREEN}Pressione '{Colors.CYAN}q{Colors.RESET}' para finalizar.{Colors.RESET}")

    saving = False
    img_count = 0

    while True:
        try:
            # Recebe o tamanho da mensagem
            message_size_data = sock.recv(4)
            if not message_size_data: 
                print(f"[{Colors.YELLOW}Coleta-PC{Colors.RESET}] Stream encerrado pela Pi.")
                break
            message_size = struct.unpack("!I", message_size_data)[0]

            # Recebe os dados
            data = b''
            while len(data) < message_size:
                packet = sock.recv(message_size - len(data))
                if not packet: break
                data += packet
            
            # Decodifica e salva a imagem
            encoded_image = pickle.loads(data)
            decoded_image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)

            if decoded_image is None: continue
            
            # Exibe a imagem
            cv2.imshow("Recepcao de Imagens", decoded_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and not saving:
                # Envia o comando para a Pi começar a salvar
                sock.sendall(b'C')
                saving = True
                print(f"[{Colors.GREEN}Coleta-PC{Colors.RESET}] Iniciando salvamento de {num_images} imagens.")
            elif key == ord('q'):
                print(f"{Colors.YELLOW}Saindo...{Colors.RESET}")
                try: sock.sendall(b'Q')
                except Exception: pass
                break

            # Lógica de salvamento, agora controlada pela flag `saving`
            if saving and img_count < num_images:
                filename = save_path / f"img_{img_count:04d}.jpg"
                cv2.imwrite(str(filename), decoded_image)
                img_count += 1
                print(f"{Colors.GREEN}Salvo {img_count}/{num_images}{Colors.RESET}")

                if img_count >= num_images:
                    sock.sendall(b'Q') # Envia 'Q' para parar o salvamento na Pi
                    print(f"{Colors.YELLOW}Coleta finalizada ({num_images} imagens salvas){Colors.RESET}")
                    saving = False  # Para de salvar
                    break
        
        except socket.timeout:
            print(f"[{Colors.YELLOW}Timeout{Colors.RESET}] A Pi parou de enviar frames. Encerrando coleta.")
            continue
        except (ConnectionResetError, BrokenPipeError):
            print(f"{Colors.RED}Conexão com a Raspberry Pi encerrada. Encerrando...{Colors.RESET}")
            break
        except Exception as e:
            print(f"{Colors.RED}Erro inesperado: {e}{Colors.RESET}")
            break
    
    print(f"{Colors.GREEN}Todas as imagens foram recebidas e salvas!{Colors.RESET}")
    cv2.destroyAllWindows()

# Função para enviar o modelo
def send_model_to_pi(model_path: Path, config: dict, model_configs: dict):
    """
    Envia o arquivo do modelo (.ckpt) e suas configurações via socket TCP.

    Args:
        model_path (Path): O caminho para o arquivo do modelo.
        config (dict): O dicionário de configuração principal.
        model_configs (dict): O dicionário de configurações dos modelos.
    """
    pi_ip = config["pi_ip"]
    pi_port = config["pi_port"]
    if not model_path.exists():
        print(f"{Colors.RED}Erro: Arquivo do modelo não encontrado em {model_path}{Colors.RESET}")
        return

    # 1. Leia o arquivo do modelo em formato de bytes
    with open(model_path, 'rb') as f:
        model_bytes = f.read()

    # 2. Reúna todas as informações em um único dicionário
    model_name = config["model_name"]
    payload = {
        'model_name': model_name,
        'model_params': model_configs[model_name]["params"],
        'model_inference_params': model_configs[model_name].get("inference_params", {}),
        'model_file': model_bytes,
    }


    model_name = config["model_name"]
    file_size = os.path.getsize(model_path)
    
    # 1. Prepara o cabeçalho (configurações + info do arquivo)
    header_payload = {
        'model_name': model_name,
        'model_params': model_configs[model_name]["params"],
        'model_inference_params': model_configs[model_name].get("inference_params", {}),
        'file_size': file_size
    }
        
    serialized_header = pickle.dumps(header_payload)
    header_size = struct.pack("!I", len(serialized_header))

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            print(f"{Colors.BLUE}Conectando à Raspberry Pi em {pi_ip}:{pi_port} para enviar o modelo...{Colors.RESET}")
            sock.connect((pi_ip, pi_port))

            # 2. Envia o tamanho e o cabeçalho primeiro
            sock.sendall(header_size + serialized_header)
            
            # 3. Envia o arquivo em blocos diretamente do disco
            print(f"Iniciando envio do arquivo '{model_name}.ckpt' ({file_size} bytes)...")
            with open(model_path, 'rb') as f:
                while True:
                    bytes_read = f.read(4096) # Lê em blocos de 4KB
                    if not bytes_read:
                        break # Fim do arquivo
                    sock.sendall(bytes_read)

            print(f"{Colors.GREEN}Modelo enviado com sucesso! Tamanho: {file_size} bytes.{Colors.RESET}")
            
    except ConnectionRefusedError:
        print(f"{Colors.RED}Erro: Raspberry Pi {pi_ip} recusou a conexão. Verifique se o servidor de recepção está rodando.{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}Erro ao enviar o modelo: {e}{Colors.RESET}")

def receive_model_from_pc(server_port: int, output_dir: str):
    """
    Escuta por um pacote de modelo e configurações, salva o modelo
    e retorna as configurações para o script principal.

    Args:
        server_port (int): A porta TCP para escutar.
        output_dir (str): O diretório onde o arquivo do modelo será salvo.

    Returns:
        dict: Um dicionário com as configurações recebidas ou None em caso de falha.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.bind(('0.0.0.0', server_port))
        server_sock.listen(1)
        
        start_time = time.time()
        print(f"{Colors.BLUE}Aguardando o modelo do PC na porta {server_port}...{Colors.RESET}")
        conn, addr = server_sock.accept()
        
        with conn:
            training_time = time.time()
            print(f"{Colors.GREEN}Conexão aceita de {addr} após {training_time - start_time} segundos. Recebendo modelo...{Colors.RESET}")
            
            # 1. Recebe o tamanho do cabeçalho
            header_size_data = conn.recv(4)
            if not header_size_data:
                print(f"{Colors.RED}Erro: Conexão encerrada antes de receber o cabeçalho.{Colors.RESET}")
                return None
            header_size = struct.unpack("!I", header_size_data)[0]

                        # 2. Recebe o cabeçalho e o desserializa
            header = conn.recv(header_size)
            header_payload = pickle.loads(header)
            
            # 4. Extrai os dados
            model_name = header_payload['model_name']
            model_params = header_payload['model_params']
            model_inference_params = header_payload.get("model_inference_params")

            # Extrai os dados do cabeçalho
            file_size = header_payload['file_size']
            
            file_path = output_path / f"{model_name}.ckpt"
            bytes_received = 0

            # 3. Salva o arquivo em disco em blocos
            print(f"{Colors.BLUE}Recebendo arquivo '{model_name}': 0.00% [0 / ({file_size//1024} KB]{Colors.RESET}", end="\r")
            with open(file_path, 'wb') as f:
                while bytes_received < file_size:
                    chunk = conn.recv(4096) # Recebe dados em blocos
                    if not chunk: break
                    f.write(chunk)
                    bytes_received += len(chunk)

                    # Exibe o progresso a cada 100KB recebidos para não sobrecarregar
                    if bytes_received % 102400 == 0 or bytes_received == file_size:
                        progress = (bytes_received / file_size) * 100
                        print(f"{Colors.BLUE}Recebendo dados: {progress:.2f}% [{bytes_received//1024} / {file_size//1024} KB]{Colors.RESET}", end="\r")
                
            if bytes_received == file_size:
                print(f"\n{Colors.GREEN}Arquivo '{model_name}.ckpt' recebido com sucesso!{Colors.RESET}")
            else:
                print(f"{Colors.YELLOW}Aviso: Conexão encerrada prematuramente. Arquivo pode estar incompleto.{Colors.RESET}")
                return None
            
            receive_time = time.time() - training_time
            print(f"{Colors.GREEN}Modelo e configurações recebidos com sucesso após {receive_time:.2f} segundos!{Colors.RESET}")
            
            # Retorna as configurações para o script principal
            return {
                "model_name": model_name,
                "model_params": model_params,
                "model_inference_params": model_inference_params,
                "ckpt_path": str(file_path)
            }

def live_inference_rasp_to_pc(picam2, conn, anomaly_output = None, move_output = None):
    """
    Captura frames, envia para um PC para inferência e recebe o resultado.

    Args:
        picam2 (Picamera2): Instância da câmera já configurada e iniciada.
        pc_ip (str): Endereço IP do PC.
        pc_port (int): A porta TCP do servidor no PC.
        image_size (int): Tamanho da imagem para captura.
        timeout (int): Tempo máximo em segundos para esperar pela resposta do PC.
    """

    if conn:
        print(f"[{Colors.CYAN}Coleta{Colors.RESET}] Aguardando comando de INÍCIO DA INFERÊNCIA ('M')...")

        try:
            conn.settimeout(None) # Espera indefinidamente pelo comando da fase
            command_bytes = conn.recv(1)
            if not command_bytes: raise ConnectionResetError("PC Desconectou")
            
            command = command_bytes.decode().strip()

            if command == "M":
                conn.sendall(b'ACK')
                print(f"[{Colors.GREEN}SYNC{Colors.RESET}] Comando 'M' recebido. Iniciando inferência.")
            else:
                conn.sendall(b'NACK')
                print(f"[{Colors.RED}SYNC{Colors.RESET}] Dessincronização! Comando '{command}' recebido. Saindo da coleta.")
                return "DISCONNECTED" # Falha na sincronização
        except (ConnectionResetError, BrokenPipeError, socket.timeout) as e:
            print(f"[{Colors.RED}Rede{Colors.RESET}] Conexão perdida durante a sincronização: {e}")
            return "DISCONNECTED"

    if move_output:
        print(f"[{Colors.YELLOW}ROBÔ{Colors.RESET}] Enviando sinal: MOVER")
        move_output.on()     

    try:       
        picam2.start()
        while True:
            # 1. Captura o frame da câmera
            frame = picam2.capture_array()
            frame = crop_and_resize(frame, 
                                    size=None)
            start_time = time.time()

            send_tcp_frame(conn, frame)
            # 4. Espera a resposta do PC
            conn.settimeout(120)
            try:
                response_bytes = conn.recv(1)
                if not response_bytes:
                    print(f"{Colors.YELLOW}Conexão encerrada pelo PC.{Colors.RESET}")
                    break
                
                if response_bytes not in (b'A', b'N',b'P',b'Q'):
                    print(f"[{Colors.RED}Dessincronização!{Colors.RESET}] Resposta inválida recebida do PC: {response_bytes}")
                    print(f"[{Colors.RED}Erro{Colors.RESET}] O PC pode estar em uma fase diferente. Encerrando inferência.")
                    return 'DISCONNECTED'
                
                if response_bytes == b'Q': return print("Q")
                if response_bytes == b'P' and move_output: 
                    print(f"[{Colors.YELLOW}ROBÔ{Colors.RESET}] PAUSADOO")
                    move_output.off()    
                elif move_output:
                    move_output.on()

                is_anomaly = response_bytes == b'A'
                end_time = time.time()

                status = f"{Colors.RED}ANOMALIA DETECTADA!{Colors.RESET}" if is_anomaly else f"{Colors.GREEN}NORMAL{Colors.RESET}"
                
                if anomaly_output is None:
                    # Se a inicialização falhou (por exemplo, falta de permissão ou pino inválido), apenas imprime
                    print(f"[{Colors.YELLOW}GPIO{Colors.RESET}] Aviso: Sinal não enviado (Inicialização do pino falhou).")
                elif is_anomaly:
                    anomaly_output.on() # Define o pino para HIGH (3.3V)
                    print(f"[{Colors.RED}GPIO{Colors.RESET}] Sinal HIGH (ANOMALIA) enviado para o pino GPIO {anomaly_output.pin.number}.")
                else:
                    anomaly_output.off() # Define o pino para LOW (0V)
                    print(f"[{Colors.GREEN}GPIO{Colors.RESET}] Sinal LOW (NORMAL) enviado para o pino GPIO {anomaly_output.pin.number}.")

                print(f"Inferência concluída em {(end_time - start_time):.2f}s. Status: {status}")
            
            except socket.timeout:
                print(f"{Colors.YELLOW}Tempo limite excedido. O PC não respondeu.{Colors.RESET}")
                continue
            
            except Exception as e:
                print(f"{Colors.RED}Erro durante a comunicação: {e}. Encerrando...{Colors.RESET}")
                return "DISCONNECTED"
    
    except ConnectionRefusedError:
        print(f"{Colors.RED}Erro: Conexão recusada.")
    except socket.timeout:
        print(f"{Colors.RED}Erro: Tempo limite excedido ao tentar conectar ao PC.{Colors.RESET}")
    except KeyboardInterrupt:
        print(f"{Colors.YELLOW}Ctrl+C detectado. Encerrando...{Colors.RESET}")

def receive_and_process_data():
    """
    Função que escuta continuamente por datagramas UDP em um socket e processa os dados recebidos.
    Executa na thread principal, sem uso de threads separadas.
    """
    import numpy as np
    UDP_IP = "0.0.0.0"
    UDP_PORT = 5005
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    
    # Configura um timeout para o socket (opcional, mas recomendado)
    # Isso impede que o programa fique bloqueado indefinidamente se a Raspberry Pi parar de enviar dados
    sock.settimeout(5.0) 
    
    data_buffer = bytearray()
    
    print(f"{Colors.CYAN}Aguardando datagramas UDP em {UDP_IP}:{UDP_PORT}{Colors.RESET}")
    print(f"Pressione {Colors.YELLOW}Ctrl+C{Colors.RESET} no terminal para encerrar.")
    
    try:
        while True:
            try:
                # Recebe o datagrama (agora com timeout)
                packet, _ = sock.recvfrom(65536)
                data_buffer.extend(packet)
                
                # Tenta desserializar o conteúdo
                try:
                    data = pickle.loads(data_buffer)
                    data_buffer.clear()
    
                    # Processamento dos dados
                    original_bytes = data['original_frame']
                    anomaly_bytes = data['anomaly_map']
                    score = data['score']
    
                    original_img = cv2.imdecode(np.frombuffer(original_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                    anomaly_img = cv2.imdecode(np.frombuffer(anomaly_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    
                    if original_img is not None and anomaly_img is not None:
                        # Coloca score na imagem original
                        cv2.putText(original_img, f"Score: {score:.4f}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
                        combined_frame = np.hstack((original_img, anomaly_img))
                        cv2.imshow("Inferência em Tempo Real", combined_frame)
    
                        # A função `waitKey` é necessária para atualizar a janela do OpenCV
                        # Se for 1ms, o programa não vai pausar.
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print(f"{Colors.YELLOW}Sinal de parada 'q' detectado. Encerrando...{Colors.RESET}")
                            break
                    else:
                        print(f"{Colors.YELLOW}Aviso: Dados de imagem incompletos recebidos.{Colors.RESET}")
                
                except pickle.UnpicklingError:
                    # Continua acumulando dados se não conseguir desserializar
                    continue
                except Exception as e:
                    print(f"{Colors.RED}Erro inesperado ao processar datagrama UDP: {e}{Colors.RESET}")
            except socket.timeout:
                # Se o timeout for atingido, o loop continua
                continue
    except KeyboardInterrupt:
        print(f"{Colors.YELLOW}Programa encerrado por Ctrl+C.{Colors.RESET}")
    finally:
        cv2.destroyAllWindows()
        sock.close()
        print(f"{Colors.CYAN}Visualização e socket encerrados.{Colors.RESET}")