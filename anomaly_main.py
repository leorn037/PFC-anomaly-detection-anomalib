from pathlib import Path
import matplotlib.pyplot as plt
import time
start_time = time.time()
from functions import *

config = {
    "dataset_root": '.',
    "image_size": 256, # Defina o tamanho da imagem para redimensionamento
    "batch_size": 8,
    "folder_name": 'Test',
    "normal_dir": "temp_images",#"data/train/normal", # Imagens normais para treinamento
    "abnormal_test_dir": "data/test/abnormal", # Imagens anômalas para teste
    "normal_test_dir": "data/test/normal", # Imagens normais para teste

    "model_name": 'PatchCore',
    # "mobilenet_v2" "wide_resnet50_2", "resnet18" efficientnet_b0 com layers ["blocks.2", "blocks.4"]
    "ckpt_path": "C:/Users/Leonardo/Downloads/Programas/PFC/results/Fre/Test/v1/weights/lightning/model.ckpt", # None, "C:/Users/Leonardo/Downloads/Programas/PFC/weights/onnx/model_onnx.onnx"
    "export_type": "onnx",
    "operation" : 'New', # Operação com modelo ('Inference','New','Continue')
    "live" : False,
}

def main():
    print(f"{Colors.GREEN}Iniciando ...{Colors.RESET}")
    # 1. Prepare dataset:
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
    print(f"{Colors.BLUE}Bibliotecas importadas em {time.time()-start_time:.2f} segundos.{Colors.RESET}")
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
