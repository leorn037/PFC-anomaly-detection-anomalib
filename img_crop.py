import cv2
import os
from pathlib import Path
import numpy as np

# --- FUNÇÃO DE CORTE E REDIMENSIONAMENTO (ASSUMIMOS QUE ESTÁ CORRETA) ---
def crop_and_resize(image: np.ndarray, x0: int, y0: int, x1: int, y1: int, size: int | None) -> np.ndarray:
    """
    Cortar e redimensionar uma imagem (array NumPy) com base nas coordenadas fornecidas.

    Args:
        image (np.ndarray): O array NumPy da imagem de entrada (OpenCV).
        x0 (int): Coordenada X inicial (coluna).
        y0 (int): Coordenada Y inicial (linha).
        x1 (int): Coordenada X final (coluna).
        y1 (int): Coordenada Y final (linha).
        size (int | None): O tamanho do lado do quadrado final. Se None, apenas corta.

    Returns:
        np.ndarray: A imagem cortada e/ou redimensionada.
    """
    # 1. Corta a imagem usando fatiamento de arrays (slicing)
    # A sintaxe é [y_start:y_end, x_start:x_end]
    cropped_image = image[y0 : y1, x0 : x1]

    # 2. Redimensiona a imagem para o tamanho desejado
    if size is not None: 
        # Garante que o redimensionamento seja para um quadrado (size x size)
        resized_image = cv2.resize(cropped_image, (size, size), interpolation=cv2.INTER_LINEAR)
        return resized_image
    else:
        return cropped_image

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

if __name__ == "__main__":
    # --- CONFIGURAÇÕES DE EXECUÇÃO ---

    # 1. Caminhos das pastas
    INPUT_FOLDER = "../img_2025-08-28_16-37-33"  # Substitua pelo nome da sua pasta de origem
    OUTPUT_FOLDER = "../images_processadas" # Nome da pasta de destino

    # 2. Coordenadas de Corte e Redimensionamento
    # Baseado na sua função, o corte é feito por [y0:y1, x0:x1]
    # Se a imagem original for 640x640 e você quiser o centro 256x256:
    # x0: 192 (coluna inicial), y0: 192 (linha inicial)
    # x1: 448 (coluna final), y1: 448 (linha final)

    FINAL_SIZE = None  # Tamanho final da imagem
    X0_COORD = 180+43
    Y0_COORD = 0
    X1_COORD = 470-58
    Y1_COORD = 640
   

    # CUIDADO: Crie uma pasta 'images_originais' com algumas imagens de teste antes de rodar!

    process_image_folder(
        input_dir=INPUT_FOLDER,
        output_dir=OUTPUT_FOLDER,
        x0=X0_COORD,
        y0=Y0_COORD,
        x1=X1_COORD,
        y1=Y1_COORD,
        size=FINAL_SIZE
    )