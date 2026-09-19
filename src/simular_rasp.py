"""
Simula a Raspberry Pi COMPLETA: faz tanto a fase de COLETA (transmite imagens
de uma pasta de treino, como se fosse a câmera do robô capturando continuamente)
quanto a fase de INFERÊNCIA (transmite imagens de uma pasta de teste, aguardando
resposta de anomalia), no mesmo protocolo de rede que a Rasp real usa
(collect_and_split_dataset + live_inference_rasp_to_pc, ambos em network.py).

Roda sem nenhum argumento — edite as constantes na seção CONFIGURAÇÕES logo
abaixo. Também aceita os mesmos valores via linha de comando, se preferir
(útil pra testar rápido sem editar o arquivo).

Uso:
    Terminal 1 (este script, faz o papel da Raspberry Pi):
        python simular_rasp.py

    Terminal 2 (o script real, sem nenhuma mudança):
        python anomaly_pc.py --pi_ip 127.0.0.1
"""
import argparse
import glob
import os
import socket
import time

import cv2

from utils import Colors, CONFIG as config
from network import pi_socket, send_tcp_frame
from cabo_tracker import CaboTracker

# ============================== CONFIGURAÇÕES ===============================
# Edite aqui pra rodar sem passar nenhum argumento na linha de comando.
PASTA_TREINO = "C:/Users/Leonardo/Downloads/Programas/PFC/src/simulacao/normal"    #!# imagens usadas na fase de COLETA (viram o dataset de treino no PC)
PASTA_TESTE = "C:/Users/Leonardo/Downloads/Programas/PFC/src/simulacao/teste_2026-07-14_16-47-27"      #!# imagens usadas na fase de INFERÊNCIA (testa o modelo já treinado)

"""
"C:/Users/Leonardo/Downloads/Programas/PFC/src/simulacao/normal"
"C:/Users/Leonardo/Downloads/Programas/PFC/src/simulacao/abnormal"
"C:/Users/Leonardo/Downloads/Programas/PFC/src/simulacao/teste_2026-07-14_16-47-27"
"C:/Users/Leonardo/Downloads/Programas/PFC/src/simulacao/normal_2026-07-14_16-43-26"
"""

IMAGE_SIZE = config["image_size"]      # tamanho do crop do CaboTracker — precisa bater com o treino do modelo
PI_PORT = config["pi_port"]
DELAY_TREINO = 0.2                     # segundos entre frames na fase de coleta (simula o time_sample real)
DELAY_TESTE = 0.0                      # segundos entre frames na fase de inferência
LOOP_TESTE = False                     # ao chegar no fim da pasta de teste, recomeça do início?
AGUARDAR_CONFIRMACAO_PC = False  # False = Roda contínuo como um vídeo | True = Fica reenviando até o PC liberar
# =============================================================================


def listar_imagens(pasta):
    return sorted(glob.glob(os.path.join(pasta, "*.jpg")) + glob.glob(os.path.join(pasta, "*.png")))


def fase_coleta(conn, tracker, pasta, delay):
    """Espelha collect_and_split_dataset(): espera 'P' (ou 'M', se o PC decidir pular a
    coleta), e se for 'P', transmite frames em loop contínuo — como uma câmera real,
    que nunca "acaba" sozinha — até o PC mandar 'Q'."""
    arquivos = listar_imagens(pasta)
    if not arquivos:
        print(f"{Colors.RED}Pasta de treino vazia ou não encontrada: {pasta}. Pulando fase de coleta.{Colors.RESET}")

    print(f"[{Colors.CYAN}Coleta{Colors.RESET}] Aguardando comando de início ('P') do PC...")
    conn.settimeout(None)
    command_bytes = conn.recv(1)
    if not command_bytes:
        raise ConnectionResetError("PC desconectou antes do handshake de coleta")
    command = command_bytes.decode().strip()

    if command == "M":
        # O PC decidiu pular a coleta (config["collect"]=False do lado dele) e já mandou
        # o comando da fase de inferência direto. Devolve NACK aqui (igual ao real) — o
        # PC vai reenviar 'M' automaticamente, e dessa vez quem responde é fase_inferencia().
        conn.sendall(b'NACK')
        print(f"[{Colors.YELLOW}Coleta{Colors.RESET}] PC pulou a coleta (mandou 'M'). Indo direto pra inferência.")
        return
    elif command != "P":
        conn.sendall(b'NACK')
        print(f"[{Colors.RED}Coleta{Colors.RESET}] Comando inesperado '{command}'. Abortando.")
        return

    if not arquivos:
        conn.sendall(b'NACK')
        return

    conn.sendall(b'ACK')
    print(f"[{Colors.GREEN}Coleta{Colors.RESET}] Comando 'P' recebido. Transmitindo {len(arquivos)} imagens em loop "
          f"até o PC mandar 'Q' (igual a câmera real faria).")

    saving = False
    saved_count = 0
    idx = 0

    while True:
        frame_bgr = cv2.imread(arquivos[idx])
        idx = (idx + 1) % len(arquivos)  #!# cicla infinitamente — uma câmera de verdade nunca "acaba" sozinha
        if frame_bgr is None:
            continue

        frame_bgr = tracker.track(frame_bgr)  # mesmo pré-processamento que o robô real aplicaria

        conn.settimeout(None)
        send_tcp_frame(conn, frame_bgr)
        conn.settimeout(delay if delay > 0 else 0.05)

        try:
            comando = conn.recv(1).decode().strip()
            if comando == "C":
                saving = True
                saved_count = 0
                print(f"\n[{Colors.YELLOW}Coleta{Colors.RESET}] PC começou a salvar imagens.")
            elif comando == "Q":
                print(f"\n[{Colors.YELLOW}Coleta{Colors.RESET}] PC mandou parar ('Q'). Fase de coleta concluída.")
                break
        except socket.timeout:
            pass

        if saving:
            saved_count += 1
            print(f"[{Colors.GREEN}Coleta{Colors.RESET}] Frame {saved_count} enviado.", end="\r")

        if delay > 0:
            time.sleep(delay)


def fase_inferencia(conn, tracker, pasta, delay):
    """Espelha live_inference_rasp_to_pc(): espera 'M', depois transmite as imagens de
    teste em loop contínuo (como o robô real, que nunca para de inferir sozinho) e reage
    às respostas N (normal) / P (pausa) / A (anomalia) / Q (sair).

    Enquanto a resposta for 'P', o robô real estaria fisicamente parado — então o mesmo
    frame é reenviado até o operador confirmar ('A') ou rejeitar ('N') a anomalia, só
    então avançando pra próxima imagem."""
    arquivos = listar_imagens(pasta)
    if not arquivos:
        print(f"{Colors.RED}Pasta de teste vazia ou não encontrada: {pasta}. Encerrando.{Colors.RESET}")
        return

    print(f"[{Colors.CYAN}Inferência{Colors.RESET}] Aguardando comando de início ('M') do PC...")
    conn.settimeout(None)
    command_bytes = conn.recv(1)
    if not command_bytes:
        raise ConnectionResetError("PC desconectou antes do handshake de inferência")
    command = command_bytes.decode().strip()

    if command != "M":
        conn.sendall(b'NACK')
        print(f"[{Colors.RED}Inferência{Colors.RESET}] Comando inesperado '{command}'. Abortando.")
        return

    conn.sendall(b'ACK')
    print(f"[{Colors.GREEN}Inferência{Colors.RESET}] Comando 'M' recebido. Enviando {len(arquivos)} imagens de teste "
          f"em loop contínuo.")

    idx = 0
    while True:
        if idx >= len(arquivos):
            #idx = 0
            break
            print(f"{Colors.YELLOW}Fim da pasta de teste — reiniciando (loop contínuo).{Colors.RESET}")

        frame_bgr = cv2.imread(arquivos[idx])
        if frame_bgr is None:
            idx += 1
            continue

        frame_bgr = tracker.track(frame_bgr)

        start_time = time.time()
        conn.settimeout(None)
        send_tcp_frame(conn, frame_bgr)

        conn.settimeout(5.0)
        response_bytes = conn.recv(1)
        if not response_bytes:
            print(f"{Colors.YELLOW}PC encerrou a conexão.{Colors.RESET}")
            break

        elapsed = time.time() - start_time
        nome = os.path.basename(arquivos[idx])

        if response_bytes == b'P':
            status = f"{Colors.YELLOW}PAUSADO (aguardando confirmação) — reenviando o mesmo frame{Colors.RESET}"
        elif response_bytes == b'A':
            status = f"{Colors.RED}ANOMALIA CONFIRMADA{Colors.RESET}"
        elif response_bytes == b'N':
            status = f"{Colors.GREEN}NORMAL{Colors.RESET}"
        elif response_bytes == b'Q':
            print(f"{Colors.YELLOW}PC pediu para parar ('Q').{Colors.RESET}")
            break
        else:
            print(f"{Colors.RED}Resposta desconhecida do PC: {response_bytes}. Abortando.{Colors.RESET}")
            break

        print(f"[{idx+1}/{len(arquivos)}] {nome} | {elapsed:.2f}s | {status}")

        #!# Só avança quando a anomalia for resolvida (A ou N). Em 'P', o robô real
        #!# estaria parado — repete o mesmo frame até o operador decidir.
        if response_bytes != b'P' or not AGUARDAR_CONFIRMACAO_PC:
            idx += 1

        if delay > 0:
            time.sleep(delay)


def simular_rasp(pasta_treino, pasta_teste, image_size, pi_port, delay_treino, delay_teste):
    conn, server_sock = pi_socket(pi_port)
    if conn is None:
        print(f"{Colors.RED}Falha ao abrir o servidor. Abortando.{Colors.RESET}")
        return

    tracker = CaboTracker(crop_output_size=image_size)

    try:
        fase_coleta(conn, tracker, pasta_treino, delay_treino)
        fase_inferencia(conn, tracker, pasta_teste, delay_teste)
    except (ConnectionResetError, BrokenPipeError) as e:
        print(f"{Colors.RED}Conexão perdida: {e}{Colors.RESET}")
    except KeyboardInterrupt:
        print(f"{Colors.YELLOW}Interrompido pelo usuário.{Colors.RESET}")
    finally:
        conn.close()
        server_sock.close()
        print(f"{Colors.CYAN}Simulação encerrada.{Colors.RESET}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simula a Raspberry Pi completa (coleta + inferência)")
    parser.add_argument("--pasta_treino", type=str, default=PASTA_TREINO)
    parser.add_argument("--pasta_teste", type=str, default=PASTA_TESTE)
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--pi_port", type=int, default=PI_PORT)
    parser.add_argument("--delay_treino", type=float, default=DELAY_TREINO)
    parser.add_argument("--delay_teste", type=float, default=DELAY_TESTE)
    args = parser.parse_args()

    simular_rasp(args.pasta_treino, args.pasta_teste, args.image_size, args.pi_port,
                 args.delay_treino, args.delay_teste)