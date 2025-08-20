from utils import Colors, CONFIG as config, anomaly_args, print_config_summary
anomaly_args(config,"pc")
print_config_summary(config, "pc")

from pathlib import Path
import matplotlib.pyplot as plt
import time
init_time = time.time()
from models import MODEL_CONFIGS, setup_datamodule, create_model, train_model, evaluate_model, get_latest_checkpoint
from inference import live_inference_opencv, visualize_imgs, serve_inference_to_pi
from network import receive_all_images_and_save, send_model_to_pi, send_flag, receive_and_process_data


def main():
    print(f"{Colors.GREEN}Iniciando ...{Colors.RESET}")
    if config["collect"] and (config["on_pc_inference"] or config["receive_model"]): 
        send_flag(config)
    # --- Passo 1: Receber todas as imagens da Raspberry Pi ---
    
    receive_path = Path(config["normal_dir"])
    if config["collect"]:
        receive_all_images_and_save(config["img_n"], receive_path, config["receive_port"])
        print(f"{Colors.GREEN}Todas as imagens foram recebidas e salvas!{Colors.RESET}")
    else: print(f"{Colors.YELLOW}Coleta de Imagens Desabilitada{Colors.RESET}")

    # --- Passo 2: Executar o treinamento com as imagens recebidas ---
    dataset_root = Path(config["dataset_root"])
    datamodule = setup_datamodule(config, dataset_root)
    # Crie o modelo
    model = create_model(config)
    if model is None: return

    print(f"{Colors.BLUE}Iniciando treinamento do modelo...{Colors.RESET}")
    engine, training_time = train_model(model, datamodule, config)
    print(f"{Colors.BLUE}Treinamento concluído em {training_time:.2f} segundos.{Colors.RESET}")

    # --- 5. Avaliação com métricas (no conjunto de teste preparado) ---
    print(f"{Colors.BLUE}Iniciando avaliação do modelo no conjunto de teste...{Colors.RESET}")
    test_results, eval_time = evaluate_model(engine, model, datamodule)
    print(f"{Colors.BLUE}Resultados da avaliação concluída em {eval_time:.2f}:{Colors.RESET}")

    # --- Passo 3: Enviar o modelo treinado para a Raspberry Pi ---
    # Encontre o caminho do checkpoint mais recente
    results_path = Path("results") / config["model_name"] / config["folder_name"]
    model_path = get_latest_checkpoint(results_path)

    if not config["receive_model"] and config["live"]:
        send_flag(config)
    elif model_path and config["live"]:
        send_model_to_pi(model_path,config, MODEL_CONFIGS)
    else:
        print(f"{Colors.RED}Erro: Não foi possível encontrar o modelo treinado para enviar.{Colors.RESET}")
    
    #engine, training_time = train_model(model, datamodule, config)
    #engine = Engine(logger=False,accelerator="auto")
    #export_model(engine, model, config)

    # --- 6. Verificação e visualização da detecção individual por código ---
    print(f"\n{Colors.BLUE}--- Verificando detecção de anomalias em imagens individuais ---{Colors.RESET}")

    if config["live"]:
            if config["on_pc_inference"]: serve_inference_to_pi(model, config)
            elif config["websocket"]: receive_and_process_data()
            else: live_inference_opencv(model, config["image_size"])
    else:
        #normal_dir = dataset_root / "test" / "normal"
        normal_dir = Path(config["normal_test_dir"])
        img_class="Normal"
        visualize_imgs(normal_dir, model, img_class, config["image_size"])

        # --- Processar imagens anômalas ---
        abnormal_dir = Path(config["abnormal_test_dir"])
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
