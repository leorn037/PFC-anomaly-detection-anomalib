import cv2
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt # Para visualização no debug
from scipy.signal import find_peaks # <--- NOVO: Biblioteca para encontrar picos

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

# --- FUNÇÃO DE DETECÇÃO E CENTRALIZAÇÃO DINÂMICA (REVISADA) ---
def centralizar_cabo_dinamicamente(
    frame_bgr: np.ndarray, 
    crop_output_size: int, 
    debug_show_steps: bool = False
    ) -> np.ndarray:
    """
    Localiza o cabo usando detecção de bordas verticais e análise de perfil,
    e o centraliza dentro de um corte quadrado, redimensionado para o tamanho final.

    Args:
        frame_bgr: Imagem de entrada em formato BGR (OpenCV).
        crop_output_size: Tamanho final do lado do quadrado (ex: 256).
        debug_show_steps: Se True, mostra as imagens intermediárias para depuração.

    Returns:
        np.ndarray: O frame final (cortado e redimensionado).
    """
    h, w, _ = frame_bgr.shape

    # --- LIMITES RÍGIDOS PARA AS COORDENADAS X ---
    X_MIN_ALLOWED = 250 #214
    X_MAX_ALLOWED = 400 #437
    Y_CUT = 0

    # Reduz a altura da análise para ignorar o chão e o topo (ROI Vertical)
    Y_START_ROI = int(h * 0.2) 
    Y_END_ROI = int(h * 0.8)
    
    # 3. Encontrar Picos (Bordas)
    PEAK_HEIGHT_THRESHOLD = 60_000 
    PEAK_MIN_DISTANCE = 30
    MIN_CABLE_WIDTH = 125

    frame_roi = frame_bgr[Y_START_ROI:Y_END_ROI, :]
    # 60_000 k_size 5 frame_roi
    # 15_000 k_size 3 frame_bgr

    # 1. Pré-processamento: Tons de Cinza
    gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)

    # Criamos um retângulo largo e baixo. Ele vai "borrar" tudo que estiver perto na horizontal, fechando os buracos entre os fios do cabo.
    # (15, 3) significa 15 pixels de largura por 3 de altura.
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))

    # O 'Morph Close' é uma Dilatação seguida de Erosão. Ele preenche vazios escuros.
    morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_morph)

    # 2. Suavização para reduzir ruído (Gaussian Blur é bom para isso)
    blurred = cv2.GaussianBlur(morph, (5, 5), 0) # Ajuste o tamanho do kernel se necessário
    
    # 3. Detecção de Bordas Verticais (Sobel X):
    # O cabo é um objeto vertical. O operador Sobel na direção X destacará suas bordas verticais.
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5) # Derivada na direção X
    sobelx = cv2.convertScaleAbs(sobelx) # Converte de volta para 8-bit

    # 4. Análise do Perfil Horizontal para encontrar a coluna mais "ativa"
    # Somamos os valores de pixel de cada coluna. O cabo terá um pico de bordas.
    horizontal_profile = np.sum(sobelx, axis=0)  
    
    # Filtra os picos com a altura e distância mínima
    peaks, properties = find_peaks(horizontal_profile, 
                                   height=PEAK_HEIGHT_THRESHOLD, 
                                   distance=PEAK_MIN_DISTANCE)
    
    center_x = w // 2 # Centro padrão de fallback
    center_y = h // 2
    
    valid_peaks = peaks[(peaks >= X_MIN_ALLOWED) & (peaks <= X_MAX_ALLOWED)]
    

    left_peaks_in_roi = valid_peaks[valid_peaks < center_x]
    if left_peaks_in_roi.size > 0:
        borda_esquerda = left_peaks_in_roi[-1]
    else: 
        borda_esquerda = X_MIN_ALLOWED
            
    # Encontre o pico de maior valor (borda mais forte) na direita permitida
    right_peaks_in_roi = valid_peaks[valid_peaks > center_x]
    if right_peaks_in_roi.size > 0:
        borda_direita = right_peaks_in_roi[0]
    else:
        borda_direita = X_MAX_ALLOWED
    
    if borda_direita - borda_esquerda < MIN_CABLE_WIDTH:
        # Se ficaram muito perto, expande artificialmente
        borda_direita = min(borda_esquerda + MIN_CABLE_WIDTH, X_MAX_ALLOWED)
        borda_esquerda = max(X_MIN_ALLOWED, borda_direita - MIN_CABLE_WIDTH)
        print(f"[AJUSTE] Largura do cabo forçada para o mínimo: {MIN_CABLE_WIDTH} pixels.")


    center_x = (borda_esquerda + borda_direita) // 2

    # 3. CORTE HORIZONTAL PELAS BORDAS
    # O corte vertical (altura) permanece na imagem inteira, mas pode ser ajustado
    x_crop_start = borda_esquerda+int(0.05*( borda_direita - borda_esquerda))
    x_crop_end = borda_direita-int(0.05*(borda_direita - borda_esquerda))

    cropped_frame = frame_bgr[Y_CUT:h-Y_CUT, x_crop_start:x_crop_end]

    final_frame = cropped_frame # cv2.resize(cropped_frame, (crop_output_size, crop_output_size), interpolation=cv2.INTER_LINEAR)

    # 7. DEBUG VISUAL: PLOTAGEM DOS PASSOS
    if debug_show_steps:
        # 7.1 Imagem da Borda e Perfil Horizontal
        fig, axs = plt.subplots(1, 4, figsize=(18, 5))
        
        # Plot 1: Imagem Original + Linhas de Borda (Azul) e Centro (Vermelho)
        axs[0].imshow(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        axs[0].axvline(x=center_x, color='red', linestyle='--', linewidth=1.5) # Linha do centro
        axs[0].axvline(x=x_crop_start if len(peaks) >= 2 else 0, color='blue', linestyle=':', linewidth=1) 
        axs[0].axvline(x=x_crop_end if len(peaks) >= 2 else 0, color='blue', linestyle=':', linewidth=1)
        axs[0].set_title('Original')
        
        # Plot 4: Perfil Horizontal
        axs[1].plot(horizontal_profile, color='black')
        axs[1].plot(peaks, horizontal_profile[peaks], "x", color='red') # Marca os picos detectados
        axs[1].axvline(x=center_x, color='r', linestyle='--', linewidth=1.5) # Linha do centro
        axs[1].axvline(x=x_crop_start if len(peaks) >= 2 else 0, color='blue', linestyle=':', linewidth=1) 
        axs[1].axvline(x=x_crop_end if len(peaks) >= 2 else 0, color='blue', linestyle=':', linewidth=1)
        axs[1].set_title('Perfil Horizontal (Picos = Bordas)')

        # Plot 5: Imagem Cortada Final
        axs[2].imshow(cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB))
        axs[2].set_title(f'Imagem Final ({crop_output_size}x{crop_output_size})')

        # Plot 3: Borda Sobel X (Visualização do que o algoritmo 'vê')
        axs[3].imshow(sobelx, cmap='gray')
        axs[3].set_title('Borda Sobel X')

        plt.tight_layout()
        plt.show() # Exibe as janelas de debug

    return final_frame

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
    
    for file_path in input_path.iterdir():
        if file_path.suffix.lower() in allowed_extensions:
            image_np = cv2.imread(str(file_path), cv2.IMREAD_COLOR)

            if image_np is None:
                print(f"[AVISO] Não foi possível ler o arquivo: {file_path.name}")
                continue

            try:
                processed_image = centralizar_cabo_dinamicamente(image_np, crop_output_size=size, debug_show_steps=debug_show_steps)

                output_file_path = output_path / file_path.name
                cv2.imwrite(str(output_file_path), processed_image)
                image_count += 1
                
            except Exception as e:
                print(f"[ERRO] Falha ao processar {file_path.name}: {e}")
                
    print("-" * 30)
    print(f"Processamento concluído. {image_count} imagens processadas com sucesso.")


if __name__ == "__main__":
    # --- CONFIGURAÇÕES DE EXECUÇÃO ---

    INPUT_FOLDER = "new" 
    OUTPUT_FOLDER = "../images_processadas_dinamicas_v" 

    FINAL_SIZE = 256
    DEBUG_MODE = True # Mantenha como True para ver o plot da primeira imagem
    
    # Certifique-se de ter `scipy` instalado: pip install scipy

    process_image_folder_dynamic(
        input_dir=INPUT_FOLDER,
        output_dir=OUTPUT_FOLDER,
        size=FINAL_SIZE,
        debug_show_steps=DEBUG_MODE
    )