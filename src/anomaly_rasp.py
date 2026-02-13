from utils import Colors, CONFIG as config, anomaly_args, print_config_summary
anomaly_args(config,"rasp")
print_config_summary(config, "rasp")
import time
init_time = time.time()
from pathlib import Path
import matplotlib.pyplot as plt

from collect_data import collect_and_split_dataset, setup_camera
from network import receive_model_from_pc, live_inference_rasp_to_pc, pi_socket
from inference import live_inference_rasp, visualize_imgs

try:
    from gpiozero import OutputDevice
    import atexit
except ImportError:
    print(f"[{Colors.YELLOW}Aviso{Colors.RESET}] gpiozero não encontrada. Rodando sem suporte a GPIO.")
    OutputDevice = None # Cria uma classe Falsa para evitar erros

ANOMALY_SIGNAL_PIN = 17 # Linha 1 Coluna 6
MOVE_SIGNAL_PIN = 27    # Pino para PAUSAR ROBÔ

anomaly_output = None
move_output = None

try:
    # 1. Configura o pino de Anomalia (queima)
    anomaly_output = OutputDevice(ANOMALY_SIGNAL_PIN, active_high=True, initial_value=False)
    
    # 2. NOVO: Configura o pino de Pausa
    move_output = OutputDevice(MOVE_SIGNAL_PIN, active_high=True, initial_value=False)

    # Função para garantir que o pino seja desligado ao final do script
    def cleanup_gpio():
        if anomaly_output: anomaly_output.off()
        if move_output: move_output.off()
        time.sleep(1)
        print(f"[{Colors.CYAN}GPIO{Colors.RESET}] Sinais LOW garantidos nos pinos {ANOMALY_SIGNAL_PIN} e {MOVE_SIGNAL_PIN} ao finalizar.")
    
    atexit.register(cleanup_gpio)

except Exception as e:
    print(f"[{Colors.RED}GPIO ERROR{Colors.RESET}] Não foi possível configurar os pinos GPIO: {e}")


def setup_network_server(config):
    """Configura servidor de rede se necessário."""
    if not (config["network_inference"] or config["receive_model"] or config["visual_rasp"]):
        return None, None
    return pi_socket(config["pi_port"])

def collect_dataset(camera, conn, config):
    """Coleta e organiza dataset se configurado."""
    if not config["collect"]:
        print(f"{Colors.YELLOW}Coleta de Imagens Desabilitada.{Colors.RESET}")
        return 
    
    ret = collect_and_split_dataset(
        camera, output_base_dir="data", # Onde o Anomalib espera encontrar os dados
        time_sample=config["time_sample"], # Salvar um frame normal automaticamente a cada 0.5 segundos
        total_frames_to_collect=config["img_n"], # Parar a coleta automática de normais após 200 frames
        image_size=config["collect_img_size"],
        new_size=config["image_size"],
        conn=conn,
        move_output=move_output,
    )
    return ret == "DISCONNECTED"

def receive_model(config, port):
    """Recebe modelo do PC e atualiza configuração."""
    if not config["receive_model"]:
        print(f"{Colors.YELLOW}Recebimento de Modelo Desabilitado.{Colors.RESET}")
        return 
    
    start_time = time.time()
    from models import MODEL_CONFIGS, create_model
    dict_model = receive_model_from_pc(port, config["model_output_dir"])
    receive_time = time.time() - start_time
    
    # Atualiza config
    config['model_name'] = dict_model['model_name']
    MODEL_CONFIGS[config['model_name']]['params'] = dict_model['model_params']
    MODEL_CONFIGS[config['model_name']]['inference_params'] = dict_model['model_inference_params']
    config['ckpt_path'] = dict_model['ckpt_path']
    
    print(f"{Colors.BLUE}Modelo recebido em {receive_time:.2f}s.{Colors.RESET}")

def create_local_model(config):
    """Cria modelo local se não usa inferência em rede."""
    if config["network_inference"]:
        return None
    
    from models import MODEL_CONFIGS, create_model
    model = create_model(config)
    if model is None:
        print(f"{Colors.RED}Falha ao criar modelo.{Colors.RESET}")
        return None
    return model

def run_inference(camera, conn, config, model):
    """Executa inferência baseada no modo."""
    print(f"{Colors.BLUE}Iniciando detecção de anomalias...{Colors.RESET}")
    
    if config["live"]:
        if config["network_inference"] and conn:
            ret = live_inference_rasp_to_pc(
                camera, conn, config["image_size"], anomaly_output, move_output
            )
            return ret == "DISCONNECTED" #!
        else:
            if model is None: 
                print(f"{Colors.RED}Modelo necessário para visualização offline.{Colors.RESET}")
            live_inference_rasp(model, config, camera, anomaly_output)
    else:
        # Visualização offline
        #normal_dir = dataset_root / "test" / "normal"
        normal_dir = Path(config["normal_dir"])
        img_class="Normal"
        visualize_imgs(normal_dir, model, img_class, config["image_size"])

        # --- Processar imagens anômalas ---
        abnormal_dir = Path(config["dataset_root"]) / "test" / "abnormal"
        img_class = "Abnormal"
        visualize_imgs(abnormal_dir, model, img_class, config["image_size"])

        plt.close('all') 

def main(camera):
    print(f"{Colors.GREEN}Iniciando Raspberry Pi Pipeline...{Colors.RESET}")
    
    conn, server_sock = None, None
    
    try:
        # 1. Configuração do Servidor Único
        conn, server_sock = setup_network_server(config)
        
        # 2. Coleta de dataset
        if collect_dataset(camera, conn, config):
            if conn: conn.close()
            if server_sock: server_sock.close()
            return True  # Se desconectar durante a funçãomanda reiniciar a main
        
        # 3. Recebimento de modelo
        receive_model(config, config["pi_port"])
        
        # 4. Criação de modelo local
        model = create_local_model(config)
        if model is None and not config["network_inference"]: # Se mandou a rasp criar modelo e não criou manda reiniciar main
            return True 
        
        # 5. Inferência/Visualização
        if run_inference(camera, conn, config, model):
            if conn: conn.close()
            if server_sock: server_sock.close()
            return True  # Se desconectar durante a função manda reiniciar a main
        
    except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
        print(f"[{Colors.RED}CRÍTICO{Colors.RESET}] Conexão perdida: {e}")
        return True # Sinaliza ao main para reiniciar
    except KeyboardInterrupt:
        print(f"{Colors.YELLOW}Interrompido pelo usuário.{Colors.RESET}")
    finally:
        # O main() é o dono dos recursos de rede e os fecha
        print("Finalizando")
        if conn: 
            conn.close()
        if server_sock: 
            server_sock.close()
        cleanup_gpio()
        print("Pipeline finalizado.")

# TODO: receive_model_from_pc(): websocket e sincronização
# TODO: live_inference_rasp(): anomaly_output e websocket

if __name__ == "__main__":
    print(f"{Colors.BLUE}Bibliotecas importadas em {time.time()-init_time:.2f} segundos.{Colors.RESET}")
    import warnings
    import os

    warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers") # Desativa todos os warnings de depreciação
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="openvino.runtime") # Desativa FutureWarnings

    # Específico para PyTorch Lightning
    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    camera = setup_camera(config["collect_img_size"])

    try:
        while main(camera):
            print(f"[{Colors.MAGENTA}REINICIANDO O FLUXO PRINCIPAL...{Colors.RESET}")
            config["collect"] = True
            time.sleep(5) # Pausa antes de tentar reabrir o servidor
    finally:
        camera.stop()