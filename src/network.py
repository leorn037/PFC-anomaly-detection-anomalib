import socket
import os
import struct
import pickle
import cv2
from pathlib import Path
from utils import Colors
import queue
import time

def crop_and_resize(image, x0: int, y0: int, x1: int, y1: int, size: int | None):
    """
    Corta e redimensiona uma imagem.

    Args:
        image (Image.Image): A imagem de entrada (objeto Pillow).
        x (int): Coordenada X (esquerda) do canto superior esquerdo para o corte.
        y (int): Coordenada Y (topo) do canto superior esquerdo para o corte.
        size (int): O lado do quadrado para o corte e o novo tamanho de redimensionamento.

    Returns:
        Image.Image: A imagem cortada e redimensionada.
    """

    # 1. Corta a imagem usando fatiamento de arrays (slicing)
    # A sintaxe é [y_start:y_end, x_start:x_end]
    cropped_image = image[y0 : y1, x0 : x1]

    # 2. Redimensiona a imagem para o tamanho desejado
    # cv2.resize espera (width, height)
    if size: 
        resized_image = cv2.resize(cropped_image, (size, size), interpolation=cv2.INTER_LINEAR)
        return resized_image
    else:
        return cropped_image
    
# frame = crop_and_resize(frame, x0=0, y0=0, x1=256, x2=256, size=None)
def sender_thread_func(stop_event, image_queue, pc_ip, pc_port):
    """Função que será executada em uma thread para enviar imagens."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(120.0)
            try:
                print(f"[{Colors.CYAN}Thread de Envio{Colors.RESET}] Conectando ao PC em {pc_ip}:{pc_port}...")
                sock.connect((pc_ip, pc_port))
                print(f"[{Colors.CYAN}Thread de Envio{Colors.RESET}] Conexão estabelecida.")
                
                while not stop_event.is_set():
                    try:
                        # Tenta pegar uma imagem da fila com um timeout
                        frame = image_queue.get(timeout=0.1)
                        

                        # Codifica o frame em formato JPEG para compressão
                        _, encoded_image = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

                        # Serializa os dados e calcula o tamanho
                        data = pickle.dumps(encoded_image)
                        message_size = struct.pack("!I", len(data))

                        # Envia o tamanho da mensagem e os dados
                        sock.sendall(message_size + data)
                        print(f"[{Colors.CYAN}Thread de Envio{Colors.RESET}] Imagem enviada. Tamanho: {len(data)} bytes.")
                    except queue.Empty: # Se a fila estiver vazia, apenas continua o loop e verifica o stop_event
                        continue
                    except (ConnectionError, BrokenPipeError) as e:
                        print(f"[{Colors.RED}Thread de Envio{Colors.RESET}] Erro de conexão: {e}. Encerrando thread...")
                        break
            except (socket.timeout, ConnectionRefusedError):
                print(f"{Colors.RED}Não foi possível conectar ao cliente, enviando imagens desativado.{Colors.RESET}")
                return
    except ConnectionRefusedError:
        print(f"[{Colors.RED}Thread de Envio{Colors.RESET}] Erro: O PC {pc_ip} recusou a conexão. Verifique se o servidor está rodando.")
    except Exception as e:
        print(f"[{Colors.RED}Thread de Envio{Colors.RESET}] Erro inesperado: {e}")

# Função para receber todas as imagens e salvar
def receive_all_images_and_save(num_images: int, save_path: Path, pc_port: int):
    """
    Recebe um número específico de imagens via TCP, decodifica e salva em um diretório.
    
    Args:
        num_images (int): Número total de imagens a serem recebidas.
        save_path (Path): O caminho para o diretório onde as imagens serão salvas.
        pc_port (int): A porta TCP que o PC irá escutar.
    """
    save_path.mkdir(parents=True, exist_ok=True)
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.bind(('0.0.0.0', pc_port))
        server_sock.listen(1)
        
        start_time = time.time()
        print(f"{Colors.BLUE}Aguardando conexão da Raspberry Pi em 0.0.0.0:{pc_port}...{Colors.RESET}")
        conn, addr = server_sock.accept()
        
        with conn:
            connection_time = time.time() - start_time
            print(f"{Colors.GREEN}Conexão aceita de {addr} em {connection_time:.2f} segundos. Recebendo {num_images} imagens...{Colors.RESET}")
            for i in range(num_images):
                # Recebe o tamanho da mensagem
                message_size_data = conn.recv(4)
                if not message_size_data: break
                message_size = struct.unpack("!I", message_size_data)[0]

                # Recebe os dados
                data = b''
                while len(data) < message_size:
                    packet = conn.recv(message_size - len(data))
                    if not packet: break
                    data += packet
                
                # Decodifica e salva a imagem
                encoded_image = pickle.loads(data)
                decoded_image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
                
                if decoded_image is not None:
                    filename = save_path / f"img_{i:04d}.jpg"
                    cv2.imwrite(str(filename), decoded_image)
                    print(f"{Colors.GREEN}Imagem {i+1}/{num_images} recebida e salva em {filename}{Colors.RESET}")
                else:
                    print(f"{Colors.RED}Falha ao decodificar a imagem {i+1}.{Colors.RESET}")

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

def live_inference_rasp_to_pc(picam2, config, timeout: int = 120):
    """
    Captura frames, envia para um PC para inferência e recebe o resultado.

    Args:
        picam2 (Picamera2): Instância da câmera já configurada e iniciada.
        pc_ip (str): Endereço IP do PC.
        pc_port (int): A porta TCP do servidor no PC.
        image_size (int): Tamanho da imagem para captura.
        timeout (int): Tempo máximo em segundos para esperar pela resposta do PC.
    """
    pc_ip = config["pc_ip"]
    pc_port = config["receive_port"]
    print(f"{Colors.GREEN}Conectando ao servidor no PC ({pc_ip}:{pc_port})...{Colors.RESET}")
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout) # Define um timeout para as operações de socket
            sock.connect((pc_ip, pc_port))
            print(f"{Colors.GREEN}Conexão estabelecida com sucesso. Pressione 'Ctrl+C' para sair.{Colors.RESET}")
            picam2.start()
            
            while True:
                # 1. Captura o frame da câmera
                frame = picam2.capture_array()
                crop_and_resize(frame,x0=126,y0=0,x1=421,y1=config["image_size"],size=None)

                
                # 2. Codifica e serializa o frame
                _, encoded_image = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                data = pickle.dumps(encoded_image)
                message_size = struct.pack("!I", len(data))
                
                start_time = time.time()
                
                # 3. Envia o tamanho da mensagem e os dados
                sock.sendall(message_size + data)
                
                # 4. Espera a resposta do PC
                try:
                    response_flag_bytes = sock.recv(1)
                    if not response_flag_bytes:
                        print(f"{Colors.YELLOW}Conexão encerrada pelo PC.{Colors.RESET}")
                        break
                        
                    is_anomaly = response_flag_bytes == b'\x01'
                    end_time = time.time()
                    
                    status = f"{Colors.RED}ANOMALIA DETECTADA!{Colors.RESET}" if is_anomaly else f"{Colors.GREEN}NORMAL{Colors.RESET}"
                    
                    print(f"Inferência concluída em {(end_time - start_time):.2f}s. Status: {status}")
                
                except socket.timeout:
                    print(f"{Colors.YELLOW}Tempo limite de {timeout}s excedido. O PC não respondeu.{Colors.RESET}")
                    continue
                
                except Exception as e:
                    print(f"{Colors.RED}Erro durante a comunicação: {e}. Encerrando...{Colors.RESET}")
                    break
    
    except ConnectionRefusedError:
        print(f"{Colors.RED}Erro: Conexão recusada. O servidor no PC {pc_ip} não está online ou a porta {pc_port}está incorreta.{Colors.RESET}")
    except socket.timeout:
        print(f"{Colors.RED}Erro: Tempo limite excedido ao tentar conectar ao PC.{Colors.RESET}")
    except KeyboardInterrupt:
        print(f"{Colors.YELLOW}Ctrl+C detectado. Encerrando...{Colors.RESET}")
    finally:
        picam2.stop()
        print(f"{Colors.CYAN}Câmera liberada.{Colors.RESET}")

def rasp_wait_flag(config):
    host = '0.0.0.0'
    port = config["receive_port"]
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind((host, port))
        server.listen(1)
        print(f"Aguardando flag em {host}:{port}...")
        conn, addr = server.accept()
        with conn:
            data = conn.recv(1024)
            if data == b"OK":
                print("Flag recebida, continuando...")

def send_flag(config):
    pi_ip = config["pi_ip"]
    port = config["receive_port"]
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((pi_ip, port))
        sock.sendall(b"OK")

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