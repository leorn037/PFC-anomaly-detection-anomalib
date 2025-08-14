from anomalib.data import MVTecAD, Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore, Fastflow, ReverseDistillation, Padim, Dfm, Stfpm, Dfkde, EfficientAd, Fre
from anomalib.pre_processing import PreProcessor
import torch # Importar torch para converter imagens para tensor
import torchvision.transforms.v2 as v2 # Importar transforms para pré-processar imagens
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time
from pathlib import Path
import socket
import pickle
import cv2
from anomalib.data.utils import TestSplitMode

# --- Mapeamento de Modelos e Parâmetros ---
# Este dicionário centraliza todas as configurações dos modelos
MODEL_CONFIGS = {
    "PatchCore": {
        "class": Patchcore,
        "params": {
            "backbone": "resnet10t", 
            "layers": ("layer2", "layer3"), # ["blocks.2", "blocks.4"]
            "coreset_sampling_ratio": 0.1,
            "num_neighbors": 9
        },
        "inference_params": {
            "pre_trained": False # Parâmetro específico para inferência
        }
    },
    "Padim": {
        "class": Padim,
        "params": {
            "backbone": "resnet10t",
            "layers": ["layer1", "layer2", "layer3"],
            "n_features": 100, #resnet18=100
            "pre_trained": True,
        },
        "inference_params": {
            # Padim não precisa de `load_from_checkpoint`
            # Ele treina e faz a inferência ao mesmo tempo
        }
    },
    "DFM": {
        "class": Dfm,
        "params": {
            "backbone": "resnet18", #resnet50
            "layer": "layer3",
            "pre_trained": True
        },
        "inference_params": {

        },
    },
        "DFKDE": {
        "class": Dfkde,
        "params": {
            "backbone": "resnet18",
            "layers": ("layer4",),
            "pre_trained": True
        },
        "inference_params": {

        }
    },
        "STFPM":{
        "class": Stfpm,
        "params": {
                "backbone": "resnet18",
                "layers": ["layer1", "layer2", "layer3"]
        },
        "inference_params": {

        }
    },
    # ----
    "FastFlow": {
        "class": Fastflow,
        "params": {
            "backbone": "resnet18",
            "pre_trained": True,
            "flow_steps": 8,
            "evaluator": False
        },
        "inference_params": {
            "pre_trained": False # Parâmetro específico para inferência
        }
    },
    "ReverseDistillation": {
        "class": ReverseDistillation,
        "params": {
            "backbone": "resnet18",
            "layers": ["layer1", "layer2", "layer3"]
        },
        "inference_params": {
            # Não é necessário 'pre_trained' aqui, pois é um parâmetro
            # opcional para esta classe e o `load_from_checkpoint` cuida disso
        }
    },

    "EfficientAd": {
        "class": EfficientAd,
        "params":{},
        "inference_params": {

        }
    },
    "FRE": {
        "class": Fre,
        "params": {
            "backbone": "resnet18",
            "layer": "layer3",
            "pooling_kernel_size": 2,
            "input_dim": 16384,#65536,
            "pre_trained": True

        }
        #pooling_kernel_size=2, input_dim=65536, latent_dim=220
    }
}

class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'

def setup_datamodule(config, dataset_root): # Recebe um objeto de configuração e retorna o Folder
    
    image_size = config["image_size"] # Defina o tamanho da imagem para redimensionamento

    transform= v2.Compose([
        v2.Resize((image_size, image_size)),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    datamodule = Folder(
        name=config["folder_name"],
        root=None,
        normal_dir=config["normal_dir"],  # Imagens normais para treinamento
        test_split_mode=TestSplitMode.SYNTHETIC, # The Folder datamodule will create training, validation, test and prediction datasets and dataloaders for us.
        #abnormal_dir=config["abnormal_test_dir"], # Imagens anômalas para teste
        #normal_test_dir=config["normal_test_dir"], # Imagens normais para teste
        num_workers=0,
        train_batch_size=config["batch_size"],
        augmentations=transform
    )
    datamodule.setup() # Carrega os datasets

    return datamodule

def create_model(config):
    """
    Cria e retorna um modelo do Anomalib com base na configuração fornecida.
    Funciona para treinamento e inferência.
    """
    model_name = config.get("model_name")
    ckp_path = config.get("ckpt_path")
    operation = config.get("operation")

    image_size = config["image_size"]
    transform= v2.Compose([
        v2.Resize((image_size, image_size)),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    pre_processor = PreProcessor(transform=transform)
    
    # 1. Busca a configuração do modelo no dicionário
    model_info = MODEL_CONFIGS.get(model_name)
    if not model_info:
        print(f"{Colors.RED}Modelo {model_name} não encontrado. Retornando 'None'{Colors.RESET}")
        return None
    
    ModelClass = model_info["class"]
    model_params = model_info["params"]
    
    model = None
    
    # 2. Lógica para Treinamento ou Inferência (unificada)
    if operation != 'Inference':
        # Cria um novo modelo para treinamento
        model = ModelClass(pre_processor=pre_processor, **model_params)
        print(f"{Colors.GREEN}Modelo de treinamento '{model_name}' criado com sucesso.{Colors.RESET}")
        
    elif ckp_path and Path(ckp_path).exists():
        # Carrega um modelo existente para inferência
        print(f"{Colors.CYAN}Carregando o MELHOR modelo de {model_name} para inferência de: {ckp_path}{Colors.RESET}")
        start_time = time.time()
        
        # Junta os parâmetros de treinamento com os de inferência
        load_params = {**model_params, **model_info.get("inference_params", {})}
        
        # A classe Padim não tem o método `load_from_checkpoint`
        if hasattr(ModelClass, 'load_from_checkpoint'):
            model = ModelClass.load_from_checkpoint(
                checkpoint_path=ckp_path,
                **load_params
            )
        else:
            print(f"{Colors.YELLOW}AVISO: Modelo '{model_name}' não suporta carregamento de checkpoint. "
                  "O modelo será criado do zero.{Colors.RESET}")
            model = ModelClass(**model_params)
        
        end_time = time.time()
        load_time = end_time - start_time
        print(f"{Colors.GREEN} Modelo carregado em {load_time:.2f} segundos.{Colors.RESET}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}AVISO: Checkpoint '{ckp_path}' não encontrado para inferência. Não será possível realizar a inferência. Verifique o caminho.{Colors.RESET}")
        return None
    
    return model

def train_model(model, datamodule, config): # Encapsula a lógica de treinamento, incluindo a verificação de checkpoints para retomar.
    
    start_time = time.time() # Registra o tempo de início

    engine = Engine(logger=False,accelerator="auto", max_epochs=10)
    #trainer = Trainer(devices=4, accelerator="gpu", strategy="ddp")

    if config["operation"] == "New": ckpt_path = None
    elif config["operation"] == "Continue": ckpt_path = config["ckpt_path"]
    else: ckpt_path=None
    
    engine.fit(model=model,
            datamodule=datamodule, 
            ckpt_path=ckpt_path, # Caminho para um checkpoint para continuar o treinamento.
            ) 
    
    end_time = time.time()   # Registra o tempo de fim
    training_time = end_time - start_time
    
    return engine, training_time

def evaluate_model(engine, model, datamodule): 
    # Executa a avaliação no conjunto de teste.
    start_time = time.time() # Registra o tempo de início

    test_results = engine.test(model, datamodule=datamodule)

    end_time = time.time()   # Registra o tempo de fim
    eval_time = end_time - start_time

    return test_results, eval_time

def export_model(engine, model, config):
    from anomalib.deploy import ExportType

    model_type=config["export_type"]
    export_type = ExportType.ONNX
    engine.export(model, export_type, export_root='.', model_file_name=f'model_{model_type}',ckpt_path=config["ckpt_path"])

def predict_image(model, image, transform, image_size):

    input_tensor = transform(image).unsqueeze(0) # Adiciona dimensão de batch
    
    with torch.no_grad():
        predictions = model.forward(input_tensor.to(model.device))

    # Extraindo o mapa de anomalia e a pontuação
    # anomaly_map é um tensor (1, 1, H, W) ou (1, H, W)
    anomaly_map = predictions.anomaly_map.cpu().squeeze().numpy()
    
    # pred_score é um tensor (1,)
    pred_score = predictions.pred_score.cpu().item()
    pred_mask = predictions.pred_mask.cpu().squeeze().numpy().astype(np.uint8) * 255 # Converte True/False para 0/255 para visualização
    return np.array(image), anomaly_map, pred_score, pred_mask

def visualize_imgs(path_dir, model, img_class,image_size):
    
    # --- Processar imagens normais ---
    model.eval()

    # Certifique-se de que as transformações correspondem às usadas no datamodule
    transform = v2.Compose([
        v2.Resize((image_size, image_size)),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if path_dir.exists() and path_dir.is_dir():
        path_images = sorted(list(path_dir.glob("*.jpg"))) # Lista todos os .jpg na pasta
        if not path_images:
            print(f"{Colors.YELLOW}Aviso: Nenhuma imagem .jpg encontrada em {path_dir}{Colors.RESET}")
        
        for i, img_path in enumerate(path_images):
            print(f"\n{Colors.YELLOW}Processando imagem ({img_class}) {i+1}/{len(path_images)}: {img_path}{Colors.RESET}")
            image = Image.open(img_path).convert("RGB")
            original_img, anomaly_map, score, pred_mask = predict_image(model, image, transform, image_size) 

            # --- Plotagem para Imagem Normal ---
            fig, axes = plt.subplots(1, 2, figsize=(12, 6)) # Cria figura e 2 eixos

            # Subplot 1: Imagem Original
            axes[0].imshow(original_img)
            axes[0].set_title(f"Original ({img_class})\nScore: {score:.4f}")
            axes[0].axis("off")

            # Subplot 2: Mapa de Anomalia 
            im_map = axes[1].imshow(anomaly_map, cmap="jet") 
            axes[1].set_title("Mapa de Anomalia")
            fig.colorbar(im_map, ax=axes[1])
            axes[1].axis("off")
            
            plt.tight_layout()
            
            plt.show() # Exibe a figura e BLOQUEIA até uma tecla ser pressionada
            print(f"{Colors.CYAN}Pressione qualquer tecla para a próxima imagem ...{Colors.RESET}")
            plt.waitforbuttonpress() # Espera por uma tecla
            plt.close(fig) # Fecha a figura atual para a próxima iteração

            expected_status = f"{Colors.GREEN}Baixa (ex: < 0.5){Colors.RESET}"
            print(f"Pontuação de anomalia esperada ({img_class}): {expected_status}, Obtido: {Colors.GREEN}{score:.4f}{Colors.RESET}")

    else:
        print(f"{Colors.YELLOW}Aviso: Diretório de imagens normais não encontrado ou não é uma pasta: {path_dir}.{Colors.RESET}")

def live_inference_opencv(model, image_size):
    """
    Realiza inferência em tempo real usando a câmera e exibe os resultados
    usando OpenCV para visualização.

    Args:
        model (torch.nn.Module): O modelo Anomalib carregado para inferência.
        image_size (int): O tamanho para redimensionar a imagem de entrada do modelo.
        device (str): O dispositivo para onde enviar o modelo e os tensores ('cuda' ou 'cpu').
    """
    # 1. Configuração da Câmera
    cap = cv2.VideoCapture(0)  # 0 geralmente se refere à câmera padrão do sistema
    if not cap.isOpened():
        print(f"{Colors.RED}Erro: Não foi possível abrir a câmera. Verifique se a câmera está conectada e disponível.{Colors.RESET}")
        return

    # 2. Pré-processamento: As mesmas transformações usadas no treinamento/inferência do dataset
    transform_for_model = v2.Compose([
        v2.Resize((image_size, image_size)),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model.eval() # Coloca o modelo em modo de avaliação
    print(f"{Colors.GREEN}Iniciando inferência em tempo real com OpenCV. Pressione 'q' para sair.{Colors.RESET}")

    try:
        while True:
            ret, frame = cap.read() # Lê um frame da câmera (BGR)
            if not ret:
                print(f"{Colors.RED}Erro: Não foi possível ler o frame da câmera. Saindo...{Colors.RESET}")
                break

            # Converter o frame OpenCV (BGR) para PIL RGB para o modelo
            frame_rgb_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Clona o frame original para exibir ao lado do mapa de anomalia
            original_frame_display = frame.copy() 

            start_time=time.time()
            original_img, anomaly_map, pred_score, pred_mask = predict_image(model, frame_rgb_pil, transform_for_model, image_size)
            print(f"Tempo: {time.time()-start_time:.4f}, {pred_score:4f} / Score: {check_for_anomaly_by_score(pred_score, 0.5)} / Area: {check_for_anomaly_by_area(pred_mask, 100)}")

            # Pós-processamento para visualização com OpenCV:
            # Redimensiona o mapa de anomalia para o tamanho do frame original
            # Normaliza o mapa de 0-1 para 0-255 (necessário para cv2.applyColorMap)
            anomaly_map_normalized = (anomaly_map * 255).astype(np.uint8)
            anomaly_map_resized = cv2.resize(anomaly_map_normalized, 
                                             (frame.shape[1], frame.shape[0]), # (largura, altura)
                                             interpolation=cv2.INTER_LINEAR)
            
            # Aplica um mapa de cores (heatmap) para melhor visualização
            anomaly_map_colored = cv2.applyColorMap(anomaly_map_resized, cv2.COLORMAP_JET)

            # Adiciona o score na imagem original para display
            cv2.putText(original_frame_display, 
                        f"Score: {pred_score:.4f}", 
                        (10, 30), # Posição do texto
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1,        # Tamanho da fonte
                        (0, 255, 0), # Cor verde (BGR)
                        2)        # Espessura da linha

            # Concatena as imagens horizontalmente para exibir lado a lado
            # Certifique-se de que ambas as imagens tenham a mesma altura antes de concatenar
            # if anomaly_map_colored.shape[0] != original_frame_display.shape[0]:
            #     # Isso não deve acontecer se redimensionamos corretamente, mas é uma verificação de segurança
            #     min_height = min(anomaly_map_colored.shape[0], original_frame_display.shape[0])
            #     original_frame_display = cv2.resize(original_frame_display, (original_frame_display.shape[1], min_height))
            #     anomaly_map_colored = cv2.resize(anomaly_map_colored, (anomaly_map_colored.shape[1], min_height))

            combined_frame = np.hstack((original_frame_display, anomaly_map_colored))

            # 4. Visualização
            cv2.imshow("Inferência em Tempo Real (Original | Mapa de Anomalia)", combined_frame)

            # Saída ao pressionar 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f"{Colors.YELLOW}Saindo da inferência em tempo real.{Colors.RESET}")
                break

    except Exception as e:
        print(f"{Colors.RED}Ocorreu um erro durante a inferência em tempo real: {e}{Colors.RESET}")

    finally:
        cap.release() # Libera a câmera
        cv2.destroyAllWindows() # Fecha todas as janelas do OpenCV

def live_inference_rasp(model, image_size,use_websoket):
    """
    Realiza inferência em tempo real usando a Raspberry.

    Args:
        model (torch.nn.Module): O modelo Anomalib carregado para inferência.
        image_size (int): O tamanho para redimensionar a imagem de entrada do modelo.
        device (str): O dispositivo para onde enviar o modelo e os tensores ('cuda' ou 'cpu').
    """
    from picamera2 import Picamera2

    # 1. Configuração da Câmera
    picam2 = Picamera2() # Cria uma instância do controle da câmera

    # Configuração de pré-visualização com resolução de 640x480 pixels
    config = picam2.create_preview_configuration(main={"size": (image_size, image_size), "format": "BGR888"})
    picam2.configure(config)

    # Inicia a câmera e espera ficar estabilizada
    picam2.start()

    # 2. Pré-processamento: As mesmas transformações usadas no treinamento/inferência do dataset
    transform_for_model = v2.Compose([
        v2.Resize((image_size, image_size)),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model.eval() # Coloca o modelo em modo de avaliação

    # 3. Configuração do Socket para Envio
    
    if use_websoket:
        # --- Configurações UDP ---
        UDP_IP = "192.168.15.5" # IP do SEU PC (cliente)! A Raspberry Pi vai enviar para este IP.
        UDP_PORT = 5005        # Porta para onde a Pi vai ENVIAR os dados.
        # Nota: O IP do servidor da Pi é '0.0.0.0' para escutar, mas o cliente precisa do IP real da Pi.
        # Aqui, a Pi é o REMETENTE UDP, então precisa do IP do DESTINATÁRIO (seu PC).

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 

        print(f"{Colors.CYAN}Iniciando servidor UDP na Raspberry Pi. Enviando para {UDP_IP}:{UDP_PORT}{Colors.RESET}")
        print(f"{Colors.GREEN}Iniciando inferência em tempo real na Raspberry. Pressione 'q' para sair.{Colors.RESET}")

    try:
        while True:
            # Captura um frame da câmera como um array
            frame = picam2.capture_array()

            # Clona o frame original para exibir ao lado do mapa de anomalia
            original_frame_display = frame.copy() 

            # Converter o frame OpenCV (BGR) para PIL RGB para o modelo
            frame_rgb_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            start_time=time.time()
            original_img, anomaly_map, pred_score, pred_mask = predict_image(model, frame_rgb_pil, transform_for_model, image_size)
            print(f"Tempo: {time.time()-start_time:.2f}")
            

            # Pós-processamento para visualização com OpenCV:
            # Redimensiona o mapa de anomalia para o tamanho do frame original
            # Normaliza o mapa de 0-1 para 0-255 (necessário para cv2.applyColorMap)
            anomaly_map_normalized = (anomaly_map * 255).astype(np.uint8)
            anomaly_map_resized = cv2.resize(anomaly_map_normalized, 
                                             (frame.shape[1], frame.shape[0]), # (largura, altura)
                                             interpolation=cv2.INTER_LINEAR)
            
            # Aplica um mapa de cores (heatmap) para melhor visualização
            anomaly_map_colored = cv2.applyColorMap(anomaly_map_resized, cv2.COLORMAP_JET)
                        
            if use_websoket:
                # Prepara os dados para envio: frame original, mapa colorido e score
                # Usaremos o frame original (BGR) do OpenCV e o mapa de anomalia colorido
                _, original_encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70]) # Qualidade 70 (0-100)
                # anomaly_map_colored é BGR
                _, anomaly_encoded = cv2.imencode('.jpg', anomaly_map_colored, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

                data_to_send = {
                    'original_frame': original_encoded.tobytes(), # Envie bytes JPEG
                    'anomaly_map': anomaly_encoded.tobytes(),     # Envie bytes JPEG
                    'score': pred_score
                }

                # Serializa com pickle
                serialized = pickle.dumps(data_to_send)

                # Divide em pacotes menores se necessário
                for i in range(0, len(serialized), 65536):
                    sock.sendto(serialized[i:i+65536], (UDP_IP, UDP_PORT))

                time.sleep(0.1)
            else:
                # Adiciona o score na imagem original para display
                cv2.putText(original_frame_display, 
                            f"Score: {pred_score:.4f}", 
                            (10, 30), # Posição do texto
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1,        # Tamanho da fonte
                            (0, 255, 0), # Cor verde (BGR)
                            2)        # Espessura da linha

                combined_frame = np.hstack((original_frame_display, anomaly_map_colored))

                # 4. Visualização
                cv2.imshow("Inferência em Tempo Real (Original | Mapa de Anomalia)", combined_frame)

                # Saída ao pressionar 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print(f"{Colors.YELLOW}Saindo da inferência em tempo real.{Colors.RESET}")
                    break

    except KeyboardInterrupt:
        print(f"{Colors.YELLOW}Ctrl+C detectado. Encerrando ...{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}Ocorreu um erro durante o envio ou inferência: {e}{Colors.RESET}")
    finally:
        picam2.stop()
        if use_websoket:
            sock.close()
        if hasattr(cv2, "destroyAllWindows") and os.environ.get("DISPLAY"):
            cv2.destroyAllWindows()
        print(f"{Colors.YELLOW}Câmera e socket liberados.{Colors.RESET}")

# Abordagem é a mais simples.
# Pode ser menos eficaz para anomalias pequenas que não elevam significativamente o score total da imagem.
def check_for_anomaly_by_score(pred_score: float, threshold: float = 0.5) -> bool:
    """
    Verifica se uma imagem contém anomalias com base em seu score de anomalia.

    Args:
        pred_score: O score de anomalia da imagem inteira.
        threshold: O limiar para classificar uma imagem como anômala.

    Returns:
        True se a imagem for anômala, False caso contrário.
    """
    return pred_score > threshold

# Mais sensível a anomalias localizadas, independentemente do score total da imagem.
# Pode ser sensível a ruído, detectando pequenos grupos de pixels anômalos que não representam um defeito real.
def check_for_anomaly_by_area(pred_mask: np.ndarray, min_area_threshold: int = 100) -> bool:
    """
    Verifica se a área de anomalia na máscara é maior que um limiar de pixels.

    Args:
        pred_mask: A máscara de anomalia binária.
        min_area_threshold: O número mínimo de pixels de anomalia para classificar.

    Returns:
        True se a área de anomalia for maior que o limiar, False caso contrário.
    """
    anomaly_pixels = np.sum(pred_mask > 0)
    return anomaly_pixels > min_area_threshold