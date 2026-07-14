import cv2
import os
from pathlib import Path
import numpy as np
from src.cabo_tracker import CaboTracker

def process_image_folder(input_dir: str, output_dir: str, x0: int, y0: int, x1: int, y1: int, size: int | None):
    """
    Processa todas as imagens em uma pasta e salva os resultados em outra.

    Args:
        input_dir (str): Caminho para a pasta de imagens de origem.
        output_dir (str): Caminho para a pasta de destino (será criada se não existir).
        x0, y0, x1, y1 (int): Coordenadas para o corte.
        size (int | None): Tamanho do lado do quadrado final (para redimensionamento).
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 1. Cria a pasta de destino se ela não existir
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Processando imagens de: {input_path.resolve()}")
    print(f"Salvando resultados em: {output_path.resolve()}")

    # 2. Define os formatos de arquivo a serem processados
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # 3. Itera sobre os arquivos
    image_count = 0
    for file_path in input_path.iterdir():
        if file_path.suffix.lower() in allowed_extensions:
            # 3.1. Lê a imagem usando OpenCV
            # cv2.IMREAD_COLOR garante que a imagem é lida em 3 canais (BGR)
            image_np = cv2.imread(str(file_path), cv2.IMREAD_COLOR)

            if image_np is None:
                print(f"[AVISO] Não foi possível ler o arquivo: {file_path.name}")
                continue

            try:
                # 3.2. Aplica o corte e redimensionamento
                processed_image = crop_and_resize(image_np, x0, y0, x1, y1, size)

                # 3.3. Define o caminho de salvamento
                output_file_path = output_path / file_path.name

                # 3.4. Salva a imagem processada
                cv2.imwrite(str(output_file_path), processed_image)
                image_count += 1
                
            except Exception as e:
                print(f"[ERRO] Falha ao processar {file_path.name}: {e}")
                
    print("-" * 30)
    print(f"Processamento concluído. {image_count} imagens processadas com sucesso.")

# --- FUNÇÃO DE PROCESSAMENTO DE PASTA MODIFICADA (Usa a v2) ---
def process_image_folder_dynamic(input_dir: str, output_dir: str, size: int, debug_show_steps: bool = False):
    """
    Processa todas as imagens em uma pasta, centralizando dinamicamente no cabo.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Processando imagens de: {input_path.resolve()}")
    print(f"Salvando resultados em: {output_path.resolve()}")

    allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_count = 0

    tracker = CaboTracker(crop_output_size=size)
    
    for file_path in input_path.iterdir():
        if file_path.suffix.lower() in allowed_extensions:
            image_np = cv2.imread(str(file_path), cv2.IMREAD_COLOR)

            if image_np is None:
                print(f"[AVISO] Não foi possível ler o arquivo: {file_path.name}")
                continue

            print(f"Processando arquivo: {file_path.name}")
            processed_image = tracker.track(image_np, debug_show_steps=debug_show_steps)

            output_file_path = output_path / file_path.name
            cv2.imwrite(str(output_file_path), processed_image)
            image_count += 1
            
                
    print("-" * 30)
    print(f"Processamento concluído. {image_count} imagens processadas com sucesso.")


if __name__ == "__main__":
    # --- CONFIGURAÇÕES DE EXECUÇÃO ---

    INPUT_FOLDER = "inference_results\img_cabo_normal"
    OUTPUT_FOLDER = "teste1" 

    FINAL_SIZE = 256
    DEBUG_MODE = True # Mantenha como True para ver o plot da primeira imagem
    
    # Certifique-se de ter `scipy` instalado: pip install scipy

    process_image_folder_dynamic(
        input_dir=INPUT_FOLDER,
        output_dir=OUTPUT_FOLDER,
        size=FINAL_SIZE,
        debug_show_steps=DEBUG_MODE
    )