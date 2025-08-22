

from anomalib.models import Patchcore, ReverseDistillation, Padim, Dfm, Stfpm, Dfkde, EfficientAd, Fre

import torchvision.transforms.v2 as v2 # Importar transforms para pré-processar imagens
# import matplotlib.pyplot as plt
import time
from pathlib import Path
from utils import Colors


# --- Mapeamento de Modelos e Parâmetros ---
# Este dicionário centraliza todas as configurações dos modelos
MODEL_CONFIGS = {
    "PatchCore": {
        "class": Patchcore,
        "params": {
            "backbone": "resnet18", 
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
            "backbone": 'resnet18',
            "layers": ["layer1", "layer2", "layer3"], # ,
            #"n_features": 100, #resnet18=100
            "pre_trained": True,
        },
        "inference_params": {
            "pre_trained": False
        }
    },
    "DFM": {
        "class": Dfm,
        "params": {
            "backbone": "resnet50", 
            "layer": "layer3",
            "pre_trained": True
        },
        "inference_params": {
            "pre_trained": False

        },
    },
        "DFKDE": { # A nível de imagem, testar com mais imagens
        "class": Dfkde,
        "params": {
            "backbone": "resnet18",
            "layers": ("layer3",),
            "n_pca_components": 16,
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

def setup_datamodule(config, dataset_root): # Recebe um objeto de configuração e retorna o Folder
    
    from anomalib.data import Folder
    from anomalib.data.utils import TestSplitMode
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
        num_workers=4,
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
    from anomalib.pre_processing import PreProcessor
    model_name = config.get("model_name")
    ckp_path = config.get("ckpt_path")
    mode = config.get("mode")

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
    if mode != 'Inference':
        # Cria um novo modelo para treinamento
        model = ModelClass(pre_processor=pre_processor, **model_params)
        print(f"{Colors.GREEN}Modelo de treinamento '{model_name}' criado com sucesso.{Colors.RESET}")
        
    else:
        if not (ckp_path and Path(ckp_path).exists()):
            results_path = Path("results") / config["model_name"] / config["folder_name"]
            ckp_path = get_latest_checkpoint(results_path)
            print(f"{Colors.CYAN}Carregando o ÚLTIMO modelo de {model_name} para inferência de: {ckp_path}{Colors.RESET}")
        else:
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
    
    return model

def train_model(model, datamodule, config): # Encapsula a lógica de treinamento, incluindo a verificação de checkpoints para retomar.
    
    start_time = time.time() # Registra o tempo de início

    from anomalib.engine import Engine
    engine = Engine(logger=False,accelerator="auto", max_epochs=50)
    #trainer = Trainer(devices=4, accelerator="gpu", strategy="ddp")

    if config["mode"] == "New": ckpt_path = None
    elif config["mode"] == "Continue": ckpt_path = config["ckpt_path"]
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

def get_latest_checkpoint(results_path: Path) -> Path:
    """Função auxiliar para encontrar o último checkpoint salvo."""
    if not results_path.exists(): return None
    
    version_dirs = sorted([d for d in results_path.iterdir() if d.is_dir() and d.name.startswith("v")])
    if not version_dirs: return None
    
    latest_version = version_dirs[-1]
    checkpoint_path = latest_version / "weights" / "lightning"
    
    if checkpoint_path.exists():
        ckpt_files = list(checkpoint_path.glob("*.ckpt"))
        if ckpt_files:
            return sorted(ckpt_files)[-1]
    return None

