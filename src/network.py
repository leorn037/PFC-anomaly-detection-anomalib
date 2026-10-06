import socket
import os
import struct
import pickle
import cv2
from pathlib import Path
from utils import Colors
import time
import numpy as np
from cabo_tracker import CaboTracker

def send_tcp_frame(sock: socket.socket, frame: np.ndarray, quality: int = 95):
    """
    Codifica (JPEG) e envia um frame (com cabeçalho de tamanho)
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
    
    # 2. Converte o array numpy diretamente para bytes (Zero Copy overhead)
    data = encoded_image.tobytes()
    
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
                print(f"[{Colors.RED}Rede-PC{Colors.RESET}] Falha ao conectar ({e}). A Pi está escutando? Tentando novamente em 3s...")
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
            response = sock.recv(4).decode().strip()
            
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

            # Recebe os dados brutos da imagem
            data = b''
            sock.settimeout(2.0) # Evita travamento eterno se a rede piscar
            try:
                while len(data) < message_size:
                    packet = sock.recv(message_size - len(data))
                    if not packet: break
                    data += packet
            except socket.timeout:
                print(f"[{Colors.YELLOW}REDE{Colors.RESET}] Timeout. Pacote perdido, pulando...")
                sock.settimeout(None)
                continue
                
            sock.settimeout(None) # Volta ao normal
            
            # Decodifica e salva a imagem
            np_arr = np.frombuffer(data, dtype=np.uint8)
            decoded_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if decoded_image is None: continue
            
            # Exibe a imagem
            resized_image = cv2.resize(decoded_image, (decoded_image.shape[1] * 2, decoded_image.shape[0] * 2), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Recepcao de Imagens", resized_image)

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
    
    print(f"{Colors.GREEN}Todas as imagens foram recebidas e salvas em {save_path}!{Colors.RESET}")
    cv2.destroyAllWindows()

# Função para enviar o modelo
def send_model_to_pi(openvino_dir: Path, sock: socket.socket):
    """
    Envia model.xml + model.bin para a Raspberry Pi via socket TCP.

    Args:
        openvino_dir (Path): Pasta onde estão model.xml e model.bin.
        config (dict): O dicionário de configuração principal (usa pi_ip, pi_port).
    """
    files_to_send = [openvino_dir / "model.xml", openvino_dir / "model.bin"]

    for file_path in files_to_send:
        if not file_path.exists():
            print(f"{Colors.RED}Erro: Arquivo não encontrado em {file_path}{Colors.RESET}")
            return

    try:
        # 2. Envia o tamanho e o cabeçalho primeiro
        sock.sendall(struct.pack("!I", len(files_to_send)))
        
        # 3. Envia o arquivo em blocos diretamente do disco
        for file_path in files_to_send:
            file_size = os.path.getsize(file_path)
            header = f"{file_path.name}|{file_size}".encode()
            sock.sendall(struct.pack("!I", len(header)) + header)

            print(f"Iniciando envio do arquivo '{file_path.name}' ({file_size} bytes)...")
            with open(file_path, 'rb') as f:
                while True:
                    bytes_read = f.read(4096)
                    if not bytes_read:
                        break
                    sock.sendall(bytes_read)
                    
            print(f"{Colors.GREEN}'{file_path.name}' enviado com sucesso! Tamanho: {file_size} bytes.{Colors.RESET}")

        print(f"{Colors.GREEN}Modelo enviado com sucesso! Tamanho: {file_size} bytes.{Colors.RESET}")
        
    except Exception as e:
        print(f"{Colors.RED}Erro ao enviar o modelo: {e}{Colors.RESET}")
def receive_model_from_pc(sock: socket.socket, output_dir: str):
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
    
    try:
        print(f"{Colors.BLUE}Aguardando envio do modelo OpenVINO pela conexão TCP ativa...{Colors.RESET}")

        start_time = time.time()

        # 1. Recebe o tamanho do cabeçalho
        num_files_data = sock.recv(4)
        if not num_files_data:
            print(f"{Colors.RED}Erro: Conexão encerrada antes de receber o cabeçalho.{Colors.RESET}")
            return None
        num_files = struct.unpack("!I", num_files_data)[0]

        xml_path = None
        for _ in range(num_files):
            header_size = struct.unpack("!I", sock.recv(4))[0]
            header = sock.recv(header_size).decode().split('|')
            filename, file_size = header[0], int(header[1])

            file_path = output_path / filename
            bytes_received = 0
            print(f"{Colors.BLUE}Recebendo '{filename}': 0.00% [0 / {file_size//1024} KB]{Colors.RESET}", end="\r")
            with open(file_path, 'wb') as f:
                while bytes_received < file_size:
                    a_ler = min(4096, file_size - bytes_received)
                    chunk = sock.recv(a_ler)
                    if not chunk:
                        break
                    f.write(chunk)
                    bytes_received += len(chunk)
                    if bytes_received % 102400 == 0 or bytes_received == file_size:
                        progress = (bytes_received / file_size) * 100
                        print(f"{Colors.BLUE}Recebendo '{filename}': {progress:.2f}% "
                                f"[{bytes_received//1024} / {file_size//1024} KB]{Colors.RESET}", end="\r")

            if bytes_received != file_size:
                print(f"\n{Colors.YELLOW}Aviso: '{filename}' incompleto ({bytes_received}/{file_size} bytes).{Colors.RESET}")
                return None

            print(f"\n{Colors.GREEN}'{filename}' recebido com sucesso!{Colors.RESET}")
            if filename == "model.xml":
                xml_path = file_path

        receive_time = time.time() - start_time
        print(f"{Colors.GREEN}Todos os arquivos recebidos em {receive_time:.2f}s!{Colors.RESET}")
        return xml_path

    except Exception as e:
        print(f"{Colors.RED}Erro ao receber o modelo via TCP: {e}{Colors.RESET}")
        return None
    
def live_inference_rasp_to_pc(picam2, conn, image_size, anomaly_output = None, move_output = None):
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
        tracker = CaboTracker(crop_output_size=image_size)
        
        while True:
            # 1. Captura o frame da câmera
            start_time = time.time()
            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            frame_bgr = tracker.track(frame_bgr)
            

            send_tcp_frame(conn, frame_bgr)

            # 4. Espera a resposta do PC
            conn.settimeout(5.0)
            try:
                response_bytes = conn.recv(1)
                if not response_bytes:
                    print(f"{Colors.YELLOW}Conexão encerrada pelo PC.{Colors.RESET}")
                    break
                
                if response_bytes ==b'P':
                    if move_output: 
                        status = f"{Colors.YELLOW}Pausado{Colors.RESET}"
                        move_output.off()
                elif response_bytes == b'A':
                    status = f"{Colors.RED}ANOMALIA DETECTADA!{Colors.RESET}"
                    
                    if anomaly_output:
                        anomaly_output.on() # Define o pino para HIGH (3.3V)
                        anomaly_output.off()
                        print(f"[{Colors.RED}GPIO{Colors.RESET}] Sinal HIGH (ANOMALIA) enviado para o pino GPIO {anomaly_output.pin.number}.")
                    else:
                        print(f"[{Colors.YELLOW}GPIO{Colors.RESET}] Aviso: Sinal não enviado (Inicialização do pino falhou).")
                elif response_bytes == b'N':
                    status = f"{Colors.GREEN}NORMAL{Colors.RESET}"
                    if move_output: move_output.on() # Continua andando
                    if anomaly_output: anomaly_output.off()
                    print(f"[{Colors.GREEN}GPIO{Colors.RESET}] Sinal LOW (NORMAL) enviado para o pino GPIO {anomaly_output.pin.number}.")
                    
                elif response_bytes == b'Q':
                    break
                else:
                    print(f"[{Colors.RED}Dessincronização!{Colors.RESET}] Resposta inválida recebida do PC: {response_bytes}")
                    print(f"[{Colors.RED}Erro{Colors.RESET}] O PC pode estar em uma fase diferente. Encerrando inferência.")
                    return 'DISCONNECTED'

                end_time = time.time()
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

def receive_and_process_data(sock):
    """
    Recebe o stream TCP de frames processados e mapas de calor vindos da Raspberry Pi,
    exibindo a visualização combinada em tempo real no PC.
    """
    if not sock:
        print(f"[{Colors.RED}Erro{Colors.RESET}] Socket TCP inválido para recepção.")
        return

    sock.settimeout(None)

    print(f"\n{Colors.GREEN}{Colors.BOLD}--- Servidor de Visualização TCP (PC) Iniciado ---{Colors.RESET}")
    print(f"Pressione {Colors.YELLOW}'q'{Colors.RESET} na janela do OpenCV para encerrar.\n")
    try:
        while True:
            # 1. Recebe o tamanho do frame enviado pela Raspberry Pi (4 bytes)
            message_size_data = sock.recv(4)
            if not message_size_data:
                print(f"[{Colors.YELLOW}Rede{Colors.RESET}] A Raspberry Pi encerrou a conexão.")
                break
                
            message_size = struct.unpack("!I", message_size_data)[0]

            # 2. Recebe os bytes exatos do frame JPEG comprimido
            image_data = bytearray()
            while len(image_data) < message_size:
                packet = sock.recv(min(message_size - len(image_data), 4096))
                if not packet:
                    break
                image_data.extend(packet)

            if not image_data:
                break

            # 3. Decodifica os bytes JPEG de volta para matriz numpy (OpenCV BGR)
            np_arr = np.frombuffer(image_data, dtype=np.uint8)
            combined_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if combined_frame is None:
                continue

            combined_frame = cv2.resize(combined_frame, (640*2, 640), interpolation=cv2.INTER_LINEAR)
            
            # 4. Exibe a janela de visualização em tempo real no PC
            cv2.imshow("Inferencia Nativa - Raspberry Pi (Original | Mapa de Calor)", combined_frame)

            # 5. Verifica se o operador apertou 'q' para sair
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print(f"{Colors.YELLOW}Encerrando visualização a pedido do usuário...{Colors.RESET}")
                try:
                    sock.sendall(b'Q') # Avisa a Pi para parar
                except Exception:
                    pass
                break

    except (ConnectionResetError, BrokenPipeError):
        print(f"\n{Colors.RED}Conexão perdida com a Raspberry Pi.{Colors.RESET}")
    except Exception as e:
        print(f"\n{Colors.RED}Erro inesperado no PC: {e}{Colors.RESET}")
    finally:
        cv2.destroyAllWindows()
        print(f"{Colors.CYAN}Janelas do OpenCV fechadas e recursos liberados.{Colors.RESET}")