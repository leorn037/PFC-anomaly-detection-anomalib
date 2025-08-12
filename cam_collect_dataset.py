import cv2
import time
import os
import random
import shutil
from pathlib import Path

# --- Códigos ANSI para Cores (para mensagens no console) ---
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

def collect_and_split_dataset(
    output_base_dir: str = "data",
    capture_dir: str = "temp_images", # Para imagens normais automáticas
    train_ratio: float = 1.0,
    camera_idx: int = 0,
    time_sample: float = 0.5,  # Salvar um frame a cada X segundos
    total_frames_to_collect: int = 100, # Total de frames normais a coletar automaticamente
    image_size = 256
):
    """
    Coleta imagens de uma câmera, as salva temporariamente e as divide
    em diretórios de treino/teste para o Anomalib. Permite também salvar
    imagens anormais manualmente durante a coleta.

    Args:
        output_base_dir (str): O diretório base onde os dados 'train' e 'test' serão salvos.
        temp_normal_capture_dir (str): Diretório temporário para salvar imagens normais capturadas automaticamente.
        temp_abnormal_capture_dir (str): Diretório temporário para salvar imagens anormais capturadas manualmente.
        train_ratio (float): A proporção de imagens normais para treinamento (ex: 0.8 para 80%).
        camera_idx (int): O índice da câmera a ser usada (0 para a câmera padrão).
        time_sample (float): Intervalo de tempo em segundos para capturar imagens normais automaticamente.
        total_frames_to_collect (int): Número total de frames normais a serem coletados automaticamente.
    """

    # Definir o caminho para o diretório de captura bruta
    raw_path = Path(capture_dir)
    
    # --- Configuração Inicial e Limpeza de Diretórios Temporários ---
    # Limpa o diretório de captura bruta se ele já existir, garantindo uma nova coleta
    if raw_path.exists():
        print(f"{Colors.YELLOW}Limpando diretório de captura bruta existente: {raw_path}{Colors.RESET}")
        shutil.rmtree(raw_path)
    raw_path.mkdir(parents=True, exist_ok=True)
    
    # Preparar diretórios finais
    train_normal_path = Path(output_base_dir) / "train" / "normal"
    test_normal_path = Path(output_base_dir) / "test" / "normal"
    test_abnormal_path = Path(output_base_dir) / "test" / "abnormal"

    # Criar os diretórios finais se não existirem (sem limpar, para não apagar dados já triados)
    train_normal_path.mkdir(parents=True, exist_ok=True)
    test_normal_path.mkdir(parents=True, exist_ok=True)
    test_abnormal_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{Colors.CYAN}--- Iniciando Coleta de Imagens da Câmera ---{Colors.RESET}")
    print(f"{Colors.BLUE}TODAS as imagens serão salvas em: {raw_path.resolve()}{Colors.RESET}")
    print(f"{Colors.GREEN}{Colors.BOLD}Instruções durante a coleta:{Colors.RESET}")
    print(f"  - O script salvará frames automaticamente a cada {time_sample}s.")
    print(f"  - Pressione '{Colors.CYAN}s{Colors.RESET}' para salvar um frame {Colors.BOLD}MANUALMENTE{Colors.RESET} (além dos automáticos).")
    print(f"  - Pressione '{Colors.CYAN}q{Colors.RESET}' para {Colors.BOLD}parar a coleta.{Colors.RESET}")
    print(f"{Colors.YELLOW}Coletará até {total_frames_to_collect} frames automaticamente.{Colors.RESET}")

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"{Colors.RED}Erro: Não foi possível abrir a câmera {camera_idx}. Verifique a conexão e permissões.{Colors.RESET}")
        return

    init_time = time.time()
    last_capture_auto = init_time
    saved_auto_count = 0
    saved_manual_count = 0

    while True:
        ret, frame = cap.read() # Lê um frame da câmera
        if not ret:
            print(f"{Colors.RED}Erro: Não foi possível ler o frame. A câmera pode ter sido desconectada ou ter um problema. Saindo da coleta.{Colors.RESET}")
            break

        # Mostrar o frame da câmera
        #frame=cv2.resize(frame, (image_size, image_size))
        cv2.imshow("Coletor de Dataset - 's' NORMAL, 'a' ANORMAL, 'q' SAIR", frame)

        current_time = time.time()

        # Lógica para captura automática de imagens
        if saved_auto_count < total_frames_to_collect and (current_time - last_capture_auto >= time_sample):
            timestamp = int(time.time() * 1000)
            filename = os.path.join(raw_path, f"auto_img_{saved_auto_count:04d}_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            print(f"[{int(current_time - init_time)}s] {Colors.GREEN}AUTO salvo: {Path(filename).name}{Colors.RESET}")
            saved_auto_count += 1
            last_capture_auto = current_time

        # Verificar inputs do teclado para salvar manualmente
        key = cv2.waitKey(1) & 0xFF 
        if key == ord('s'):
            timestamp = int(time.time() * 1000)
            filename = os.path.join(raw_path, f"manual_img_{saved_manual_count:04d}_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            saved_manual_count += 1
            print(f"[{int(current_time - init_time)}s] {Colors.YELLOW}MANUAL salvo: {Path(filename).name} ({saved_manual_count} manuais){Colors.RESET}")
        elif key == ord('q'):
            print(f"{Colors.YELLOW}Saindo da coleta de imagens.{Colors.RESET}")
            break

        # Parar se o total de frames automáticos for atingido
        if saved_auto_count >= total_frames_to_collect:
            print(f"{Colors.YELLOW}Total de frames automáticos atingido ({total_frames_to_collect}). Pressione 's' para salvar manualmente ou 'q' para finalizar.{Colors.RESET}")
            # Uma vez que o limite automático é atingido, resetamos a condição para que ele não tente mais salvar automaticamente
            total_frames_to_collect = -1 # Garante que a condição 'saved_auto_count < total_frames_to_collect' seja falsa
            break


    cap.release()          # Libera o recurso da câmera
    cv2.destroyAllWindows() # Fecha a janela do OpenCV

    print(f"{Colors.BOLD}{Colors.GREEN}\nColeta de imagens concluída!{Colors.RESET}")
    print(f"{Colors.BLUE}Foram coletadas {saved_auto_count} imagens automáticas e {saved_manual_count} imagens manuais.{Colors.RESET}")
    print(f"{Colors.CYAN}Todas as imagens estão em: {raw_path.resolve()}{Colors.RESET}")

    print(f"\n{Colors.BOLD}{Colors.RED}--- PRÓXIMOS PASSOS CRÍTICOS (MANUAIS) ---{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.RED}1. Vá até a pasta '{raw_path.resolve()}' e revise CADA IMAGEM.{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.RED}2. Mova as imagens para os diretórios finais com base em sua CLASSIFICAÇÃO:{Colors.RESET}")
    print(f"{Colors.CYAN}   - Imagens NORMAIS para Treino (maioria): {train_normal_path.resolve()}{Colors.RESET}")
    print(f"{Colors.CYAN}   - Imagens NORMAIS para Teste (algumas): {test_normal_path.resolve()}{Colors.RESET}")
    print(f"{Colors.CYAN}   - Imagens ANORMAIS para Teste (todas): {test_abnormal_path.resolve()}{Colors.RESET}")
    print(f"{Colors.YELLOW}   Sugestão: Crie subpastas temporárias dentro de '{raw_path.name}' para organizar antes de mover.{Colors.RESET}")
    print(f"{Colors.MAGENTA}   Exemplo de estrutura desejada após a triagem manual:{Colors.RESET}")
    print(f"     {output_base_dir}/")
    print(f"     ├── train/")
    print(f"     │   └── normal/ (suas imagens normais para treino)")
    print(f"     └── test/")
    print(f"         ├── normal/ (suas imagens normais para teste)")
    print(f"         └── abnormal/ (suas imagens anormais para teste)")
    print(f"\n{Colors.BOLD}{Colors.GREEN}Após mover as imagens, seu dataset estará pronto para o Anomalib!{Colors.RESET}")


if __name__ == "__main__":
    # --- Parâmetros de Configuração ---
    # Ajuste estes valores conforme sua necessidade
    collect_and_split_dataset(
        output_base_dir="data",                 # Onde o Anomalib espera encontrar os dados
        capture_dir="temp_images", # Pasta para salvar as imagens brutas da câmera
        train_ratio=1.0,                        # 80% das imagens normais para treino, 20% para teste
        camera_idx=0,                           # O índice da sua câmera (0 é a padrão, tente 1, 2, etc. se tiver várias)
        time_sample=0.1,                       # Salvar um frame normal automaticamente a cada 0.5 segundos
        total_frames_to_collect=100,             # Parar a coleta automática de normais após 100 frames
        image_size=256
    )