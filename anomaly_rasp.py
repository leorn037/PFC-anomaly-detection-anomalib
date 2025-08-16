from pathlib import Path
import matplotlib.pyplot as plt
import time
init_time = time.time()
from functions import Colors, MODEL_CONFIGS, create_model, live_inference_rasp, visualize_imgs, anomaly_args
from collect_dataset import collect_and_split_dataset, setup_camera
from net_func import receive_model_from_pc, live_inference_rasp_to_pc

config = {
    # Rede
    "receive_port": 5007,      # Porta para receber imagens
    "pi_port": 5008,           # Porta para enviar o modelo
    "pi_ip": "raspberrypi",   # IP da Raspberry Pi
    "pc_ip": "192.168.15.2", #"leorn037-ACER",   # IP do PC

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
    "model_name": 'DFKDE',
    "ckpt_path": None, # None, "C:/Users/Leonardo/Downloads/Programas/PFC/weights/onnx/model_onnx.onnx"
    
    "export_type": "onnx",

    "operation" : 'Inference', # Operação com modelo ('Inference','Train','Continue')
    "on_pc_inference" : True,

    # Visualização
    "live" : True,
    "websocket" : True,
}


def main():
    anomaly_args(config,"rasp")
    print(f"{Colors.GREEN}Iniciando ...{Colors.RESET}")
    # 1. Prepare dataset:
    camera = setup_camera(config["image_size"])
    if config["collect"]: # Novo dataset
        collect_and_split_dataset(
            camera,
            output_base_dir="data",                 # Onde o Anomalib espera encontrar os dados
            capture_dir=config["normal_dir"], # Pasta para salvar as imagens brutas da câmera
            time_sample=config["time_sample"],                       # Salvar um frame normal automaticamente a cada 0.5 segundos
            total_frames_to_collect=config["img_n"],             # Parar a coleta automática de normais após 200 frames
            image_size=config["image_size"],
            pc_ip=config["pc_ip"],
            pc_port=config["receive_port"]
        )

    if config["receive_model"] and not config["on_pc_inference"]:
        start_time = time.time()
        dict = receive_model_from_pc(config["pi_port"], config["model_output_dir"])
        receive_model_time = time.time() - start_time
        
        config['model_name'] = model_name = dict['model_name']
        MODEL_CONFIGS[model_name]['params'] = dict['model_params']
        MODEL_CONFIGS[model_name]['inference_params'] = dict['model_inference_params']
        config['ckpt_path'] = dict['ckpt_path']
        print(f"{Colors.BLUE}Recebimento do modelo concuído em {receive_model_time:.2f} segundos.{Colors.RESET}")

    if not config["on_pc_inference"]:
        # 2. Crie o modelo
        model = create_model(config)
        if model == None: return

    # --- 6. Verificação e visualização da detecção individual por código ---
    print(f"\n{Colors.BLUE}--- Verificando detecção de anomalias em imagens individuais ---{Colors.RESET}")

    # --- Processar imagens normais ---

    if config["live"]:
            if config["on_pc_inference"]: live_inference_rasp_to_pc(camera, config, timeout = 1)
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
