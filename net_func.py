import socket
import os
import struct
import pickle
import cv2
import numpy as np
from pathlib import Path
from functions import Colors
import queue

def sender_thread_func(stop_event, image_queue, pc_ip, pc_port):
    """Função que será executada em uma thread para enviar imagens."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
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
    except ConnectionRefusedError:
        print(f"[{Colors.RED}Thread de Envio{Colors.RESET}] Erro: O PC recusou a conexão. Verifique se o servidor está rodando.")
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
        
        print(f"{Colors.BLUE}Aguardando conexão da Raspberry Pi em 0.0.0.0:{pc_port}...{Colors.RESET}")
        conn, addr = server_sock.accept()
        
        with conn:
            print(f"{Colors.GREEN}Conexão aceita de {addr}. Recebendo {num_images} imagens...{Colors.RESET}")
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
        pi_ip (str): O IP da Raspberry Pi.
        pi_port (int): A porta TCP que a Pi está escutando.
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
        'file_name': config['file_name']
    }
    
    # 3. Serializa o dicionário completo com pickle
    serialized_payload = pickle.dumps(payload)
    message_size = struct.pack("!I", len(serialized_payload))

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            print(f"{Colors.BLUE}Conectando à Raspberry Pi em {pi_ip}:{pi_port} para enviar o modelo...{Colors.RESET}")
            sock.connect((pi_ip, pi_port))

            # Envia o tamanho total do pacote seguido do pacote de dados
            sock.sendall(message_size + serialized_payload)
            
            with open(model_path, 'rb') as f:
                while True:
                    data = f.read(1024)
                    if not data: break
                    sock.sendall(data)
            
            print(f"{Colors.GREEN}Modelo e configurações enviados com sucesso! Tamanho: {len(serialized_payload)} bytes.{Colors.RESET}")
    except ConnectionRefusedError:
        print(f"{Colors.RED}Erro: Raspberry Pi recusou a conexão. Verifique se o servidor de recepção está rodando.{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}Erro ao enviar o modelo: {e}{Colors.RESET}")

def get_latest_checkpoint(results_path: Path) -> Path:
    """Função auxiliar para encontrar o último checkpoint salvo."""
    if not results_path.exists(): return None
    
    version_dirs = sorted([d for d in results_path.iterdir() if d.is_dir() and d.name.startswith("v")])
    if not version_dirs: return None
    
    latest_version = version_dirs[-1]
    checkpoint_path = latest_version / "weights" / "lightning"
    
    if checkpoint_path.exists():
        ckpt_files = list(checkpoint_path.glob("*.ckpt"))
        if ckpt_files:
            return sorted(ckpt_files)[-1]
    return None


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
        
        print(f"{Colors.BLUE}Aguardando o modelo do PC na porta {server_port}...{Colors.RESET}")
        conn, addr = server_sock.accept()
        
        with conn:
            print(f"{Colors.GREEN}Conexão aceita de {addr}. Recebendo modelo...{Colors.RESET}")
            
            # 1. Recebe o tamanho total do pacote
            message_size_data = conn.recv(4)
            if not message_size_data:
                print(f"{Colors.RED}Erro: Conexão encerrada antes de receber o tamanho do pacote.{Colors.RESET}")
                return None
            message_size = struct.unpack("!I", message_size_data)[0]
            
            # 2. Recebe o pacote completo
            data = b''
            while len(data) < message_size:
                chunk = conn.recv(message_size - len(data))
                if not chunk: break
                data += chunk
            
            if len(data) < message_size:
                print(f"{Colors.YELLOW}Aviso: Conexão encerrada prematuramente. Pacote pode estar incompleto.{Colors.RESET}")
                return None
            
            # 3. Desserializa o pacote
            payload = pickle.loads(data)
            
            # 4. Extrai os dados e salva o arquivo
            model_name = payload['model_name']
            model_params = payload['model_params']
            model_inference_params = payload['model_inference_params']
            model_file_bytes = payload['model_file']
            file_name = payload['file_name']
            
            file_path = output_path / file_name

            # Recebe o arquivo e salva
            with open(file_path, 'wb') as f:
                f.write(model_file_bytes)
            
            print(f"{Colors.GREEN}Modelo e configurações recebidos com sucesso!{Colors.RESET}")
            
            # Retorna as configurações para o script principal
            return {
                "model_name": model_name,
                "model_params": model_params,
                "model_inference_params": model_inference_params,
                "ckpt_path": str(file_path)
            }