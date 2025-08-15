from pathlib import Path
import matplotlib.pyplot as plt
import time
init_time = time.time()
from functions import Colors, MODEL_CONFIGS, setup_datamodule, create_model, train_model, evaluate_model, get_latest_checkpoint, live_inference_opencv, visualize_imgs
from net_func import receive_all_images_and_save, send_model_to_pi
from picam_collect_dataset import setup_camera

config = {
    # Rede
    "receive_port": 5007,      # Porta para receber imagens
    "pi_port": 5008,           # Porta para enviar o modelo
    "pi_ip": "192.168.15.7",   # IP da Raspberry Pi
    "pc_ip": "192.168.15.3",   # IP do PC

    # Coleta de Imagens
    "collect" : True,
    "time_sample" : 0.2,
    "img_n" : 100,

    # Recebimento do modelo pela rasp
    "receive_model" : True,
    "model_output_dir" : "models/received",

    # Dataset configs
    "dataset_root": '.',
    "image_size": 256, # Defina o tamanho da imagem para redimensionamento
    "batch_size": 8,
    "folder_name": 'Test',
    "normal_dir": "temp_images",#"data/train/normal", # Imagens normais para treinamento
    "abnormal_test_dir": "data/test/abnormal", # Imagens anômalas para teste
    "normal_test_dir": "data/test/normal", # Imagens normais para teste

    # Model configs
    "model_name": 'Padim',
    "ckpt_path": "C:/Users/Leonardo/Downloads/Programas/PFC/results/PatchCore/Test/v95/weights/lightning/model.ckpt", # None, "C:/Users/Leonardo/Downloads/Programas/PFC/weights/onnx/model_onnx.onnx"
    
    "export_type": "onnx",
    
    "operation" : 'Train', # Operação com modelo ('Inference','Train','Continue')

    # Visualização
    "live" : True,
    "websocket" : True,
}

def main():
    print(f"{Colors.GREEN}Iniciando ...{Colors.RESET}")
    # --- Passo 1: Receber todas as imagens da Raspberry Pi ---
    
    receive_path = Path(config["normal_dir"])
    if config["collect"]:
        receive_all_images_and_save(config["img_n"], receive_path, config["receive_port"])
        print(f"{Colors.GREEN}Todas as imagens foram recebidas e salvas!{Colors.RESET}")

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

    if model_path:
        send_model_to_pi(model_path,config, MODEL_CONFIGS)
    else:
        print(f"{Colors.RED}Erro: Não foi possível encontrar o modelo treinado para enviar.{Colors.RESET}")
    
    #engine, training_time = train_model(model, datamodule, config)
    #engine = Engine(logger=False,accelerator="auto")
    #export_model(engine, model, config)

    # --- 6. Verificação e visualização da detecção individual por código ---
    print(f"\n{Colors.BLUE}--- Verificando detecção de anomalias em imagens individuais ---{Colors.RESET}")

    if config["live"]:
            live_inference_opencv(model, config["image_size"])
    else:
        #normal_dir = dataset_root / "test" / "normal"
        normal_dir = Path(config["normal_dir"])
        img_class="Normal"
        visualize_imgs(normal_dir, model, img_class, config["image_size"])

        # --- Processar imagens anômalas ---
        abnormal_dir = dataset_root / "test" / "abnormal"
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
