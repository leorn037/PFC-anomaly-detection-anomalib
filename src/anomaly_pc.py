from utils import Colors, CONFIG as config, anomaly_args, print_config_summary


def connect_to_pi(config):
    """Estabelece conexão resiliente com a Raspberry Pi."""
    # Tenta se conectar à Pi (Servidor) com resiliência
    if not (config["network_inference"] or config["receive_model"] or config["visual_rasp"]):
        return None
    return pi_connect(config["pi_ip"], config["pi_port"])

def collect_images(config, sock, receive_path):
    """Coleta e salva imagens da Raspberry Pi."""
    if not config["collect"]:
        print(f"{Colors.YELLOW}Coleta de Imagens Desabilitada.{Colors.RESET}")
        return
    ret = receive_all_images_and_save(config["img_n"], receive_path, sock)
    if ret == 'DISCONNECTED':
        raise ConnectionError("Conexão perdida durante coleta")

def setup_pipeline(config, dataset_root):
    """Configura datamodule e modelo."""
    datamodule = setup_datamodule(config, dataset_root)
    
    # Cria o modelo
    model = create_model(config)
    if model is None:
        raise ValueError("Falha ao criar modelo")
    return datamodule, model

def train_and_evaluate(config, model, datamodule):
    """Executa treinamento e avaliação se configurado."""
    if config["operation_mode"] == 'Train':
        print(f"{Colors.BLUE}Iniciando treinamento...{Colors.RESET}")
        engine, training_time = train_model(model, datamodule, config)
        print(f"{Colors.BLUE}Treinamento concluído em {training_time:.2f}s.{Colors.RESET}")
    
        print(f"{Colors.BLUE}Exportando para OpenVINO...{Colors.RESET}")

        #  Avaliação com métricas (no conjunto de teste preparado)
        if config["evaluate"]:
            print(f"{Colors.BLUE}Iniciando avaliação...{Colors.RESET}")
            test_results, eval_time = evaluate_model(engine, model, datamodule)
            print(f"{Colors.BLUE}Avaliação concluída em {eval_time:.2f}s.{Colors.RESET}")
                
        if config.get("use_openvino", False):
            from anomalib.deploy import ExportType, OpenVINOInferencer
            # Define onde salvar (cria pasta weights/openvino se não existir)
            export_root = Path(engine.trainer.default_root_dir)

            exported_path = engine.export(
                model=model,
                export_type=ExportType.OPENVINO,
                export_root=export_root,
                ckpt_path=None # Usa os pesos atuais na memória
            )

            # Garante que é um objeto Path
            exported_path = Path(exported_path) 

            print(f"Modelo exportado para: {exported_path}")
            
            print(f"{Colors.GREEN}Carregando OpenVINO Inferencer na memória...{Colors.RESET}")

            # Substituímos o modelo PyTorch pesado pelo OpenVINO leve AGORA
            ov_model = OpenVINOInferencer(
                path=exported_path,
                device="CPU" # Força CPU para seu AMD
            )

            return ov_model

    # Se não for modo Train ou não usou OpenVINO, retorna o modelo original (PyTorch)
    return model

def send_trained_model(config):
    """Resgata modelo treinado e envia para Pi se configurado."""

    # Encontre o caminho do checkpoint mais recente
    results_path = Path("results") / config["model_name"] / config["folder_name"]
    model_path = get_latest_checkpoint(results_path)
    
    if model_path and config["receive_model"] and config["live"]:
        send_model_to_pi(model_path, config, MODEL_CONFIGS)

def run_inference(config, model, sock):
    """Executa inferência baseada na configuração."""
    print(f"{Colors.BLUE}Verificando detecção de anomalias...{Colors.RESET}")
    
    if config["live"]:
        if config["network_inference"]:
            serve_inference_to_pi(model, config, sock, threshold=0.9)
        elif config["visual_rasp"]:
            receive_and_process_data()
        else:
            live_inference_opencv(model, config["image_size"])
    else:
        # Teste offline
        normal_dir = Path(config["normal_dir"])
        img_class="Test"
        visualize_imgs(normal_dir, model, img_class, config["image_size"])

        # --- Processar imagens anômalas ---
        abnormal_dir = Path(config["abnormal_test_dir"])
        img_class = "Abnormal"
        visualize_imgs(abnormal_dir, model, img_class, config["image_size"])

        plt.close('all') 

def main():
    print(f"{Colors.GREEN}Iniciando...{Colors.RESET}")
    
    sock = None
    try:
        # 1. Conexão com a Pi
        sock = connect_to_pi(config)
        
        # 2. Receber todas as imagens da Raspberry Pi se configurado
        receive_path = Path(config["normal_dir"])
        collect_images(config, sock, receive_path)
        
        # 3. Pipeline de ML, cria modelo ou recupera um checkpoint
        dataset_root = Path(config["dataset_root"])
        datamodule, model = setup_pipeline(config, dataset_root)
        
        # 4. Executar o treinamento/avaliação com as imagens recebidas
        model = train_and_evaluate(config, model, datamodule)
        
        # 5. Envio do modelo para Raspberry se configurado
        send_trained_model(config)
        
        # 6. Inferência/Visualização
        run_inference(config, model, sock)
        
    except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
        print(f"[{Colors.RED}CRÍTICO{Colors.RESET}] Conexão perdida: {e}")
    except KeyboardInterrupt:
        print(f"{Colors.YELLOW}Interrompido pelo usuário.{Colors.RESET}")
    except Exception as e:
        print(f"[{Colors.RED}CRÍTICO{Colors.RESET}] Erro: {e}")
    finally:
        if sock:
            sock.close()
            print(f"[{Colors.CYAN}Conexão fechada.{Colors.RESET}]")


# TODO: send_model_to_pi(): websocket e sincronização
# TODO: receive_and_process_data(): websocket

if __name__ == "__main__":

    anomaly_args(config,"pc")
    print_config_summary(config, "pc")

    from pathlib import Path
    import matplotlib.pyplot as plt
    import time
    init_time = time.time()
    from models import MODEL_CONFIGS, setup_datamodule, create_model, train_model, evaluate_model, get_latest_checkpoint
    from inference import live_inference_opencv, visualize_imgs, serve_inference_to_pi
    from network import receive_all_images_and_save, send_model_to_pi, pi_connect, receive_and_process_data


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