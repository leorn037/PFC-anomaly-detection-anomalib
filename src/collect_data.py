import cv2
import time
import os
import shutil
from pathlib import Path
from datetime import datetime
from utils import Colors
from network import crop_and_resize
import queue

try:
    from picamera2 import Picamera2

    # Se a importação for bem-sucedida, configure a câmera da Raspberry Pi
    print("Usando a câmera da Raspberry Pi.")
    # Aqui vai a sua lógica de setup para a picamera2
    def setup_camera(image_size):
        picam2 = Picamera2()
        # Configurações para a preview e captura de imagem
        config = picam2.create_video_configuration(main={"size": (image_size, image_size)})
        picam2.configure(config)
        picam2.start()

        # Atraso para a câmera estabilizar
        # time.sleep(2)

        # Opcional: Desabilitar o controle automático de exposição e ganho para consistência de imagens
        # picam2.set_controls({"AeEnable": False, "AwbEnable": False})

        # Define o modo de foco para manual com um valor específico
        # picam2.set_controls({"AfMode": picam2.controls.AfModeEnum.Manual, "LensPosition": 2.0})

        # A posição da lente é um valor numérico. O valor '0' é o infinito.
        # Valores maiores movem a lente para mais perto.

        #focus_distance_10cm = 10.0 # Valor de foco em dioptrias (1/m)
        #lens_position = 1000.0 / focus_distance_10cm # Converte para o valor de controle
        
        #print(f"Definindo foco manual para {focus_distance_10cm} cm...")
        #picam2.set_controls({"LensPosition": lens_position})

        return picam2

    def get_frame(camera,image_size):
        # Lógica para pegar o frame da picamera2
        frame = camera.capture_array()
        # Converte a imagem de RGB para BGR para ser compatível com o OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame_bgr  

except ImportError:
    # --- Bloco executado se as bibliotecas da Raspberry Pi não existirem ---
    print("Usando a câmera do PC com OpenCV.")
    
    # Use as suas funções originais para a câmera do PC
    def setup_camera(image_size):
        return cv2.VideoCapture(0)

    def get_frame(camera,image_size):
        ret, frame = camera.read()
        if not ret:
            print(f"{Colors.RED}Erro: Não foi possível ler o frame. A câmera pode ter sido desconectada ou ter um problema. Saindo da coleta.{Colors.RESET}")
            return None
        frame=cv2.resize(frame, (image_size, image_size))
        return frame

def has_gui():
    """Verifica se há uma interface gráfica disponível."""
    return 'DISPLAY' in os.environ and not os.environ['DISPLAY'] == ''

def collect_and_split_dataset(
    camera,
    output_base_dir: str = "data",
    time_sample: float = 0.1,  # Salvar um frame a cada X segundos
    total_frames_to_collect: int = 200, # Total de frames normais a coletar automaticamente
    image_size: int = 256,
    pc_ip = None,
    pc_port = None,
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
    
    # 1. Obtém a data e hora atuais
    current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Configurações para o envio
    IMAGE_QUEUE = queue.Queue(maxsize=5) # Fila com limite de 10 imagens

    # Definir o caminho para o diretório de captura bruta
    raw_path = Path(output_base_dir) / f"images_{current_timestamp}"
    
    raw_path.mkdir(parents=True, exist_ok=False)
    print(f"{Colors.GREEN}Criando novo diretório de captura em: {raw_path}{Colors.RESET}")

    is_headless = not has_gui()

    # --- INÍCIO DA LÓGICA DE THREADS E FILAS ---
    if pc_ip and pc_port:
        import threading
        from network import sender_thread_func
        stop_event = threading.Event()
        sender_thread = threading.Thread(target=sender_thread_func, args=(stop_event, IMAGE_QUEUE, pc_ip, pc_port), daemon=True)
        sender_thread.start()
    # --- FIM DA LÓGICA DE THREADS E FILAS ---

    if is_headless:
        print(f"\n{Colors.CYAN}--- Modo de Coleta Headless (Sem GUI) ---{Colors.RESET}")
        print(f"{Colors.YELLOW}Não foi detectada uma interface gráfica. A coleta será totalmente automática.{Colors.RESET}")
        print(f"{Colors.BLUE}Imagens serão salvas em: {raw_path.resolve()}{Colors.RESET}")
        print(f"{Colors.YELLOW}Coletará um total de {total_frames_to_collect} frames automaticamente.{Colors.RESET}")
    else:
        print(f"\n{Colors.CYAN}--- Iniciando Coleta de Imagens com Picamera2 ---{Colors.RESET}")
        print(f"{Colors.BLUE}Imagens serão salvas em: {raw_path.resolve()}{Colors.RESET}")
        print(f"{Colors.GREEN}{Colors.BOLD}Instruções durante a coleta:{Colors.RESET}")
        print(f"  - O script salvará frames automaticamente a cada {time_sample}s.")
        print(f"  - Pressione '{Colors.CYAN}c{Colors.RESET}' começar a salvar {Colors.BOLD}AUTOMATICAMENTE{Colors.RESET}.")
        print(f"  - Pressione '{Colors.CYAN}s{Colors.RESET}' para salvar um frame {Colors.BOLD}MANUALMENTE{Colors.RESET}.")
        print(f"  - Pressione '{Colors.CYAN}q{Colors.RESET}' para {Colors.BOLD}parar a coleta.{Colors.RESET}")
        print(f"{Colors.YELLOW}Coletará até {total_frames_to_collect} frames automaticamente.{Colors.RESET}")


    init_time = time.time()
    last_capture_auto = init_time
    saved_auto_count = 0
    saved_manual_count = 0
    saving = is_headless

    try:
        while True:
            # Captura a imagem da câmera
            frame_bgr = get_frame(camera,image_size)
            frame_bgr = crop_and_resize(frame_bgr,x0=126,y0=0,x1=421,y1=image_size,size=None)

            if frame_bgr is None:
                print("Erro: Não foi possível ler o frame.")
                break

            current_time = time.time()

            # Lógica para captura automática de imagens
            if saving and saved_auto_count < total_frames_to_collect and (current_time - last_capture_auto >= time_sample):
                # A thread de envio pegará a imagem da fila em segundo plano
                if not IMAGE_QUEUE.full():
                    IMAGE_QUEUE.put(frame_bgr)
                timestamp = int(time.time() * 1000)
                filename = os.path.join(raw_path, f"auto_img_{saved_auto_count:04d}_{timestamp}.jpg")
                cv2.imwrite(filename, frame_bgr)
                print(f"[{int(current_time - init_time)}s] {Colors.GREEN}AUTO salvo: {Path(filename).name}{Colors.RESET}")
                saved_auto_count += 1
                last_capture_auto = current_time

            if not is_headless:
                # Mostrar o frame da câmera
                cv2.imshow("Coletor de Dataset - 's' NORMAL, 'a' ANORMAL, 'q' SAIR", frame_bgr)

                # Verificar inputs do teclado para salvar manualmente
                key = cv2.waitKey(1) & 0xFF 

                if key == ord('s'):
                    timestamp = int(time.time() * 1000)
                    filename = os.path.join(raw_path, f"manual_img_{saved_manual_count:04d}_{timestamp}.jpg")
                    cv2.imwrite(filename, frame_bgr)
                    saved_manual_count += 1
                    print(f"[{int(current_time - init_time)}s] {Colors.YELLOW}MANUAL salvo: {Path(filename).name} ({saved_manual_count} manuais){Colors.RESET}")
                elif key == ord('c') and not saving:
                    saving = True
                    last_capture_auto = time.time() # Reseta o timer para o início
                    print(f"{Colors.GREEN}{Colors.BOLD}Início da coleta automática de imagens!{Colors.RESET}")
                elif key == ord('q'):
                    print(f"{Colors.YELLOW}Saindo da coleta de imagens.{Colors.RESET}")
                    break

            # Parar se o total de frames automáticos for atingido
            if saved_auto_count >= total_frames_to_collect:
                print(f"{Colors.YELLOW}Total de frames automáticos atingido ({total_frames_to_collect}). Pressione 's' para salvar manualmente ou 'q' para finalizar.{Colors.RESET}")
                saving = False
                # Uma vez que o limite automático é atingido, resetamos a condição para que ele não tente mais salvar automaticamente
                total_frames_to_collect = -1 # Garante que a condição 'saved_auto_count < total_frames_to_collect' seja falsa
                break
    finally:
        # --- DESLIGAMENTO SEGURO DA THREAD ---
        if pc_ip and pc_port:
            stop_event.set()
            sender_thread.join(timeout=1)  # Espera no máximo 1 segundo
        if isinstance(camera, cv2.VideoCapture):
            # Se for a câmera do PC, use release()
            camera.release()
        elif hasattr(camera, 'stop'):
            # Se for a câmera da Raspberry Pi, use stop()
            print(f"{Colors.CYAN}Desligando câmera...{Colors.RESET}")
            camera.stop()

    print(f"{Colors.BOLD}{Colors.GREEN}\nColeta de imagens concluída!{Colors.RESET}")
    print(f"{Colors.BLUE}Foram coletadas {saved_auto_count} imagens automáticas e {saved_manual_count} imagens manuais.{Colors.RESET}")
    print(f"{Colors.CYAN}Todas as imagens estão em: {raw_path.resolve()}{Colors.RESET}")


if __name__ == "__main__":
    # --- Parâmetros de Configuração ---
    # Ajuste estes valores conforme sua necessidade
    collect_and_split_dataset(
        output_base_dir="data",                 # Onde o Anomalib espera encontrar os dados
        time_sample=0.1,                       # Salvar um frame normal automaticamente a cada 0.5 segundos
        total_frames_to_collect=100,             # Parar a coleta automática de normais após 200 frames
        image_size=640,
        camera = setup_camera(640)
    )