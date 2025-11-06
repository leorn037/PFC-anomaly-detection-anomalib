
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

CONFIG = {
    # Rede
    "receive_port": 5007,      # Porta para receber imagens
    "pi_port": 5008,           # Porta para enviar o modelo
    "pi_ip": "raspberrypi",   # IP da Raspberry Pi
    "pc_ip": "leorn037-ACER.local", #"leorn037-ACER",   # IP do PC

    # Coleta de Imagens
    "collect" : True, # Coleta de Imagens pela Rasp para treinamento
    "time_sample" : 0.2,
    "img_n" : 100,
    "collect_img_size": 640,

    # Recebimento do modelo pela rasp
    "receive_model" : False, # Envio do PC / Recebimento pela Rasp do modelo
    "model_output_dir" : "models/received",

    # Dataset configs
    "dataset_root": './data',
    "image_size": 256, # Defina o tamanho da imagem para coleta e redimensionamento
    "batch_size": 32,
    "folder_name": 'Barra',
    "normal_dir": "img_barra_nor",#"data/normal", # Imagens normais para treinamento
    "abnormal_test_dir": "img_test", #"data/abnormal", # Imagens anômalas para teste
    "normal_test_dir": "img_test", #"data/test", # Imagens normais para teste

    # Model configs
    "model_name": 'PatchCore',
    "ckpt_path": "C:/Users/Leonardo/Downloads/Programas/PFC/src/results/Patchcore/Cabo/v4/weights/lightning/model.ckpt",
    "evaluate": True, 

    "operation_mode" : 'Train', # Operação com modelo ('Inference','Train','Continue')
    "network_inference" : True, # Executa inferência no PC com imagens da Raspberry

    # Visualização
    "live" : False, # Inferência em tempo real, False: Inferência em imagens salvas
    "websocket" : True, # Envio via websocket da Raspberry para o PC
}

import argparse

# Função de conversão customizada
def str_to_bool(arg):
    """Converte uma string 'True' ou 'False' para o valor booleano correspondente."""
    if isinstance(arg, bool):
        return arg  # Já é um booleano, retorna
    if arg.lower() in ('true', 't', '1'):
        return True
    elif arg.lower() in ('false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Valor booleano esperado: True, False, 1, 0, etc.')
    
def anomaly_args(config, mode="rasp"):
    """
    Cria argumentos de linha de comando para sobrescrever valores de config.

    Args:
        config (dict): Dicionário de configuração padrão.
        mode (str): "pc" ou "rasp" para decidir quais argumentos expor.
    """
    
    # Cria um objeto ArgumentParser
    parser = argparse.ArgumentParser(
        description="Script para detecção de anomalias")

    pc_args = [
        ("pi_ip", str, "IP da Raspberry Pi"),
        ("collect", bool, "Habilita a coleta de imagens (True/False)"),
        ("img_n", int, "Número de imagens a ser capturadas"),
        ("model_name", str, "Nome do modelo para ser treinado"),
        ("operation_mode", str, "Escolhe o modo de operação da detecção ('Train', 'Inference')"),
        ("network_inference", bool, "Habilita a inferência das imagens via PC e retorna resultado para a Rasp"),
    ]

    rasp_args = [
        ("pc_ip", str, "IP do PC para receber dados"),
        ("collect", bool, "Habilita a coleta de imagens (True/False)"),
        ("time_sample", float, "Intervalo entre capturas automáticas de imagens"),
        ("img_n", int, "Número de imagens a ser capturadas"),
        ("receive_model", bool, "Habilita a recepção do modelo via rede (True/False)"),
        ("network_inference", bool, "Habilita a inferência das imagens via PC e retorna resultado para a Rasp"),
    ]

    # Adiciona argumentos específicos de cada modo
    specific_args = pc_args if mode == "pc" else rasp_args
    for key, arg_type, help_text in specific_args:

        if arg_type == bool:
            parser.add_argument(
                f"-{key[0]}",
                f"--{key}", 
                type=str_to_bool, 
                default=config[key], 
                # choices=['baixa', 'normal', 'alta'],
                help=help_text)

        else:
            parser.add_argument(
                f"-{key[0]}",
                f"--{key}", 
                type=arg_type, 
                default=config[key], 
                # choices=['baixa', 'normal', 'alta'],
                help=help_text)

    args = parser.parse_args()

    # Atualiza o config com valores passados no CLI
    for key in vars(args):
        config[key] = getattr(args, key)

    return config


# A classe Colors e o dicionário CONFIG já estão no seu código
# Não é necessário copiá-los novamente.
def print_config_summary(config: dict, mode: str = "rasp"):
    """
    Imprime um resumo visual dos parâmetros de configuração.

    Args:
        config (dict): O dicionário de configuração a ser exibido.
        keys_to_show (list, opcional): Uma lista de chaves a serem mostradas.
                                       Se None, exibe um conjunto padrão de chaves.
    """
    
    # 1. Defina um conjunto padrão de chaves se nenhuma for fornecida
    if mode == "rasp":
        keys_to_show = [
            "pc_ip", "collect", "time_sample", "img_n", 
            "receive_model", "network_inference", "websocket"
            ]
    else: 
        keys_to_show = [ "pi_ip", "collect", "img_n", "model_name", "operation_mode", "network_inference" ]

    # 2. Crie uma string de separação para o cabeçalho
    separator = f"{Colors.CYAN}{'-'*40}{Colors.RESET}"
    
    print(f"\n{separator}")
    print(f"{Colors.BOLD}{Colors.CYAN} Resumo da Configuração {Colors.RESET}")
    print(f"{separator}")

    # 3. Itere sobre as chaves e imprima os valores
    for key in keys_to_show:
        value = config.get(key, f"{Colors.RED}N/A{Colors.RESET}")
        
        # 4. Aplique formatação especial para booleanos
        if isinstance(value, bool):
            if value:
                value_str = f"{Colors.GREEN}{value}{Colors.RESET}"
            else:
                value_str = f"{Colors.RED}{value}{Colors.RESET}"
        else:
            value_str = f"{Colors.YELLOW}{value}{Colors.RESET}"
            
        print(f" {Colors.BOLD}{key:<20}{Colors.RESET}: {value_str}")

    print(f"{separator}\n")
