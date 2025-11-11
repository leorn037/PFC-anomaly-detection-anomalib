from utils import Colors, CONFIG as config, anomaly_args, print_config_summary

def main():
    print(f"{Colors.GREEN}Iniciando ...{Colors.RESET}")

    sock = None
    # Tenta se conectar à Pi (Servidor) com resiliência
    if config["network_inference"] or config["receive_model"] or config["visual_rasp"]:
        sock = pi_connect(config["pi_ip"], config["pi_port"])
    # --- Passo 1: Receber todas as imagens da Raspberry Pi ---
    
    try:
        receive_path = Path(config["normal_dir"])
        if config["collect"]:
            receive_all_images_and_save(config["img_n"], receive_path, sock)
        else: print(f"{Colors.YELLOW}Coleta de Imagens Desabilitada.{Colors.RESET}")

        # --- Passo 2: Executar o treinamento com as imagens recebidas ---
        dataset_root = Path(config["dataset_root"])
        datamodule = setup_datamodule(config, dataset_root)
        # Crie o modelo
        model = create_model(config)
        if model is None: return

        if config["operation_mode"]=='Train':
            print(f"{Colors.BLUE}Iniciando treinamento do modelo...{Colors.RESET}")
            engine, training_time = train_model(model, datamodule, config)
            print(f"{Colors.BLUE}Treinamento concluído em {training_time:.2f} segundos.{Colors.RESET}")

            # --- 5. Avaliação com métricas (no conjunto de teste preparado) ---
            if config["evaluate"]: 
                print(f"{Colors.BLUE}Iniciando avaliação do modelo no conjunto de teste...{Colors.RESET}")
                test_results, eval_time = evaluate_model(engine, model, datamodule)
                print(f"{Colors.BLUE}Resultados da avaliação concluída em {eval_time:.2f}:{Colors.RESET}")

        # --- Passo 3: Enviar o modelo treinado para a Raspberry Pi ---
        # Encontre o caminho do checkpoint mais recente
        results_path = Path("results") / config["model_name"] / config["folder_name"]
        model_path = get_latest_checkpoint(results_path)

        if model_path and config["receive_model"] and config["live"]:
            send_model_to_pi(model_path,config, MODEL_CONFIGS)
        else:
            print(f"{Colors.RED}Erro: Não foi possível encontrar o modelo treinado para enviar.{Colors.RESET}")
        
        # --- 6. Verificação e visualização da detecção individual por código ---
        print(f"\n{Colors.BLUE}--- Verificando detecção de anomalias em imagens individuais ---{Colors.RESET}")

        if config["live"]:
                if config["network_inference"]: serve_inference_to_pi(model, config, sock, threshold=0.9)
                elif config["visual_rasp"]: receive_and_process_data()
                else: live_inference_opencv(model, config["image_size"])
        else:
            normal_dir = Path(config["normal_test_dir"])
            img_class="Test"
            visualize_imgs(normal_dir, model, img_class, config["image_size"])

            # --- Processar imagens anômalas ---
            abnormal_dir = Path(config["abnormal_test_dir"])
            img_class = "Abnormal"
            visualize_imgs(abnormal_dir, model, img_class, config["image_size"])

            plt.close('all') 

    except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
        print(f"[{Colors.RED}CRÍTICO{Colors.RESET}] Conexão com a Pi perdida: {e}.")
    except KeyboardInterrupt:
        print(f"{Colors.YELLOW}Ctrl+C detectado. Encerrando...{Colors.RESET}")
    except Exception as e:
        print(f"[{Colors.RED}CRÍTICO{Colors.RESET}] Erro inesperado no main: {e}.")
    finally:
        if sock:
            sock.close()
            print(f"[{Colors.CYAN}Rede-PC{Colors.RESET}] Conexão com a Pi fechada.")


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
