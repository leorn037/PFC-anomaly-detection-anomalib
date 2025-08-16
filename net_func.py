import socket
import os
import struct
import pickle
import cv2
import numpy as np
from pathlib import Path
from functions import Colors
import queue
from time import time

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
        
        start_time = time()
        print(f"{Colors.BLUE}Aguardando conexão da Raspberry Pi em 0.0.0.0:{pc_port}...{Colors.RESET}")
        conn, addr = server_sock.accept()
        
        with conn:
            connection_time = time() - start_time
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
        
        start_time = time()
        print(f"{Colors.BLUE}Aguardando o modelo do PC na porta {server_port}...{Colors.RESET}")
        conn, addr = server_sock.accept()
        
        with conn:
            training_time = time()
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
            
            receive_time = time() - training_time
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
            
            while True:
                # 1. Captura o frame da câmera
                frame = picam2.capture_array()
                
                # 2. Codifica e serializa o frame
                _, encoded_image = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                data = pickle.dumps(encoded_image)
                message_size = struct.pack("!I", len(data))
                
                start_time = time()
                
                # 3. Envia o tamanho da mensagem e os dados
                sock.sendall(message_size + data)
                
                # 4. Espera a resposta do PC
                try:
                    response_flag_bytes = sock.recv(1)
                    if not response_flag_bytes:
                        print(f"{Colors.YELLOW}Conexão encerrada pelo PC.{Colors.RESET}")
                        break
                        
                    is_anomaly = response_flag_bytes == b'\x01'
                    end_time = time()
                    
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