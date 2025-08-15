from pathlib import Path
import matplotlib.pyplot as plt
import time
init_time = time.time()
from functions import *
from collect_dataset import collect_and_split_dataset

# Adicione a configuração para a porta e o diretório de destino


config = {
    # Recebimento de imagens no pc
    "num_images_to_receive": 100,  # Adicione o número de imagens a receber
    "receive_port": 5007,          # Porta para receber imagens
    "pi_port": 5008,       # Porta para enviar o modelo
    "pi_ip": "192.168.15.5",       # IP da Raspberry Pi

    # Recebimento do modelo pela rasp
    "receive_model_port" : 5008,
    "receive_model" : False,
    "model_output_dir" : "models/received",

    # Collecting images configs
    "collect" : False,
    "time_sample" : 0.2,
    "img_n" : 100,

    # Dataset configs
    "dataset_root": '.',
    "image_size": 256, # Defina o tamanho da imagem para redimensionamento
    "batch_size": 8,
    "folder_name": 'Test',
    "normal_dir": "temp_images",#"data/train/normal", # Imagens normais para treinamento
    "abnormal_test_dir": "data/test/abnormal", # Imagens anômalas para teste
    "normal_test_dir": "data/test/normal", # Imagens normais para teste

    # Model configs
    "model_name": 'DFM',
    "ckpt_path": "C:/Users/Leonardo/Downloads/Programas/PFC/results/PatchCore/Test/v95/weights/lightning/model.ckpt", # None, "C:/Users/Leonardo/Downloads/Programas/PFC/weights/onnx/model_onnx.onnx"
    
    "export_type": "onnx",

    # 
    "operation" : 'Train', # Operação com modelo ('Inference','Train','Continue')
    "live" : True,
    "rasp" : False,
    "websocket" : True,
    "udp_ip" : "192.168.15.5"
}

def main():
    print(f"{Colors.GREEN}Iniciando ...{Colors.RESET}")
    # 1. Prepare dataset:
    if config["collect"]: # Novo dataset
        collect_and_split_dataset(
            output_base_dir="data",                 # Onde o Anomalib espera encontrar os dados
            capture_dir=config["normal_dir"], # Pasta para salvar as imagens brutas da câmera
            time_sample=config["time_sample"],                       # Salvar um frame normal automaticamente a cada 0.5 segundos
            total_frames_to_collect=config["img_n"],             # Parar a coleta automática de normais após 200 frames
            image_size=config["image_size"]
        )

    dataset_root = Path(config["dataset_root"])
    datamodule = setup_datamodule(config, dataset_root)

    # 2. Crie o modelo
    model = create_model(config)
    if model == None: return

    if config["operation"] != "Inference":

        print(f"{Colors.BLUE}Iniciando treinamento do modelo...{Colors.RESET}")

        engine, training_time = train_model(model, datamodule, config)

        print(f"{Colors.BLUE}Treinamento concluído em {training_time:.2f} segundos.{Colors.RESET}")

        # --- 5. Avaliação com métricas (no conjunto de teste preparado) ---
        print(f"{Colors.BLUE}Iniciando avaliação do modelo no conjunto de teste...{Colors.RESET}")

        test_results, eval_time = evaluate_model(engine, model, datamodule)

        print(f"{Colors.BLUE}Resultados da avaliação concluída em {eval_time:.2f}:{Colors.RESET}")
        print(test_results)
    
    #engine, training_time = train_model(model, datamodule, config)
    #engine = Engine(logger=False,accelerator="auto")
    #export_model(engine, model, config)

    # --- 6. Verificação e visualização da detecção individual por código ---
    print(f"\n{Colors.BLUE}--- Verificando detecção de anomalias em imagens individuais ---{Colors.RESET}")

    # --- Processar imagens normais ---

    if config["live"]:
        if config["rasp"]:
            live_inference_rasp(model, config)
        else: 
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
