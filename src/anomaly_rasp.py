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


def main(camera):
    
    print(f"{Colors.GREEN}Iniciando ...{Colors.RESET}")
    
    # --- Configuração do Servidor Único ---
    conn = None
    server_sock = None
    if config["network_inference"] or config["receive_model"] or config["visual_rasp"]:
        conn, server_sock = pi_socket(config["pi_port"])
    # --- Fim da Configuração do Servidor ---

    try:
        # 1. Prepare dataset:
        if config["collect"]: # Novo dataset
            ret = collect_and_split_dataset(
                camera,
                output_base_dir="data",                 # Onde o Anomalib espera encontrar os dados
                time_sample=config["time_sample"],                       # Salvar um frame normal automaticamente a cada 0.5 segundos
                total_frames_to_collect=config["img_n"],             # Parar a coleta automática de normais após 200 frames
                image_size=config["collect_img_size"],
                conn = conn,
                move_output=move_output
            )
            if ret == "DISCONNECTED":
                if conn: conn.close()
                if server_sock: server_sock.close()
                return True

        else: print(f"{Colors.YELLOW}Coleta de Imagens Desabilitada.'{Colors.RESET}")

        if config["receive_model"]:
            start_time = time.time()
            from models import MODEL_CONFIGS, create_model
            dict = receive_model_from_pc(config["pi_port"], config["model_output_dir"])
            receive_model_time = time.time() - start_time
            
            config['model_name'] = model_name = dict['model_name']
            MODEL_CONFIGS[model_name]['params'] = dict['model_params']
            MODEL_CONFIGS[model_name]['inference_params'] = dict['model_inference_params']
            config['ckpt_path'] = dict['ckpt_path']
            print(f"{Colors.BLUE}Recebimento do modelo concuído em {receive_model_time:.2f} segundos.{Colors.RESET}")
        else: 
            print(f"{Colors.YELLOW}Recebimento de Modelo Desabilitado{Colors.RESET}")

        if not config["network_inference"]:
            # 2. Crie o modelo
            from models import MODEL_CONFIGS, create_model
            model = create_model(config)
            if model == None: return

        # --- 6. Verificação e visualização da detecção individual por código ---
        print(f"\n{Colors.BLUE}--- Verificando detecção de anomalias em imagens individuais ---{Colors.RESET}")

        # --- Processar imagens normais ---

        if config["live"]:
                if config["network_inference"] and conn: 
                    ret = live_inference_rasp_to_pc(camera, conn, anomaly_output, move_output)
                    if ret == "DISCONNECTED": 
                        if conn: conn.close()
                        if server_sock: server_sock.close()
                        return True
                else: live_inference_rasp(model, config, camera, anomaly_output)
        else:
            #normal_dir = dataset_root / "test" / "normal"
            normal_dir = Path(config["normal_dir"])
            img_class="Normal"
            visualize_imgs(normal_dir, model, img_class, config["image_size"])

            # --- Processar imagens anômalas ---
            abnormal_dir = Path(config["dataset_root"]) / "test" / "abnormal"
            img_class = "Abnormal"
            visualize_imgs(abnormal_dir, model, img_class, config["image_size"])

            plt.close('all') 
    except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
        print(f"[{Colors.RED}CRÍTICO{Colors.RESET}] Conexão principal perdida: {e}.")
        return 'restart' # Sinaliza ao main para reiniciar
    except KeyboardInterrupt:
        print(f"{Colors.YELLOW}Ctrl+C detectado.{Colors.RESET}")
    finally:
        # O main() é o dono dos recursos de rede e os fecha
        print("Finalizando")
        if conn: conn.close()
        if server_sock: server_sock.close()

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

    # Para OpenVINO especificamente
    os.environ["OPENVINO_PYTHON_WARNINGS"] = "0"

    # Para timm (PyTorch Image Models)
    os.environ["TIMM_IGNORE_DEPRECATED_WARNINGS"] = "1"

    camera = setup_camera(config["collect_img_size"])

    try:
        while main(camera):
            print(f"[{Colors.MAGENTA}REINICIANDO O FLUXO PRINCIPAL...{Colors.RESET}")
            config["collect"] = True
            time.sleep(5) # Pausa antes de tentar reabrir o servidor
    finally:
        camera.stop()

