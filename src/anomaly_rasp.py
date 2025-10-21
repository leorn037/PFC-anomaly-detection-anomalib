from utils import Colors, CONFIG as config, anomaly_args, print_config_summary
anomaly_args(config,"rasp")
print_config_summary(config, "rasp")
import time
init_time = time.time()
from pathlib import Path
import matplotlib.pyplot as plt

from collect_data import collect_and_split_dataset, setup_camera
from network import receive_model_from_pc, live_inference_rasp_to_pc, rasp_wait_flag
from inference import live_inference_rasp, visualize_imgs

def main():
    
    print(f"{Colors.GREEN}Iniciando ...{Colors.RESET}")
    if config["collect"] and (config["on_pc_inference"] or config["receive_model"]): 
        rasp_wait_flag(config)

    # 1. Prepare dataset:
    camera = setup_camera(config["collect_img_size"])
    if config["collect"]: # Novo dataset
        collect_and_split_dataset(
            camera,
            output_base_dir="data",                 # Onde o Anomalib espera encontrar os dados
            time_sample=config["time_sample"],                       # Salvar um frame normal automaticamente a cada 0.5 segundos
            total_frames_to_collect=config["img_n"],             # Parar a coleta automática de normais após 200 frames
            image_size=config["collect_img_size"],
            crop_size=config["crop_x"],
            pc_ip=config["pc_ip"],
            pc_port=config["receive_port"]
        )
    else: f"{Colors.YELLOW}Coleta de Imagens Desabilitada{Colors.RESET}"

    if config["on_pc_inference"]:
         print(f"{Colors.CYAN}Esperando confirmação de treinamento...{Colors.RESET}")
         rasp_wait_flag(config)
    elif config["receive_model"]:
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
         from models import MODEL_CONFIGS, create_model
         f"{Colors.YELLOW}Recebimento de Modelo Desabilitado{Colors.RESET}"

    if not config["on_pc_inference"]:
        # 2. Crie o modelo
        model = create_model(config)
        if model == None: return

    # --- 6. Verificação e visualização da detecção individual por código ---
    print(f"\n{Colors.BLUE}--- Verificando detecção de anomalias em imagens individuais ---{Colors.RESET}")

    # --- Processar imagens normais ---

    if config["live"]:
            if config["on_pc_inference"]: live_inference_rasp_to_pc(camera, config, timeout = 120, ser = ser)
            else: live_inference_rasp(model, config, camera)
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

    main()
