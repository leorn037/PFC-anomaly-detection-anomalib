import serial
import threading
import time
import sys
import struct 

# Detecta SO e porta automaticamente
def detectar_porta():
    if sys.platform.startswith('win'):  # Windows
        PORTAS_POSSIVEIS = ['COM3', 'COM4', 'COM5', 'COM6', 'COM7']
    else:  # Linux/Mac (Raspberry Pi)
        PORTAS_POSSIVEIS = ['/dev/ttyUSB0', '/dev/ttyACM0', '/dev/ttyUSB1']
    
    for porta in PORTAS_POSSIVEIS:
        try:
            ser = serial.Serial(porta, 115200, timeout=1)
            ser.close()
            return porta
        except:
            continue
    return None

PORTA = detectar_porta()
#PORTA = '/dev/ttyUSB0'
BAUDRATE = 115200
velocidade_atual = 75  # Velocidade atual (0-255)

def ler_prints_esp():
    """Thread ROBUSTA que lê TODOS os prints do ESP32"""
    while True:
        try:
            if esp.in_waiting > 0:
                # Lê BYTES crus e tenta decodificar com ERRO IGNORE
                linha_bytes = esp.readline()
                linha = linha_bytes.decode('utf-8', errors='ignore').strip()
                if linha:
                    print(f"\n[ESP32] {linha}")
        except:
            pass  # Ignora erros de decodificação
        time.sleep(0.01)

# --- Função para Enviar o Pacote Binário ---
def enviar_pacote(tipo_char, valor_int):
    """
    Empacota os dados no formato: [Tipo (1B)] [Valor (2B)] [Fim (1B)]
    Total: 4 Bytes
    """
    try:
        # '<'  = Little Endian (Padrão do ESP32)
        # 'c'  = char (1 byte)
        # 'h'  = short (2 bytes - int16)
        # 'c'  = char (1 byte - fim)
        
        # Garante que o tipo seja bytes (ex: 'V' vira b'V')
        tipo_byte = tipo_char.upper().encode('utf-8') 
        fim_byte = b'\n'
        
        # Cria o pacote
        pacote = struct.pack('<chc', tipo_byte, int(valor_int), fim_byte)
        
        # Envia pela serial
        esp.write(pacote)
        # print(f"DEBUG BINARIO: Enviado Tipo={tipo_char} Valor={valor_int}") # Descomente para debug
        
    except Exception as e:
        print(f"Erro ao empacotar: {e}")

# --- Configuração Serial ---
if PORTA is None:
    print("❌ Nenhuma porta encontrada! Verifique o cabo.")
    sys.exit()

try:
    esp = serial.Serial()
    esp.port = PORTA
    esp.baudrate = BAUDRATE
    esp.timeout = 1
    esp.setDTR(False)
    esp.setRTS(False)
    esp.open()
    print(f"✅ Conectado em {PORTA} @ {BAUDRATE}")
except Exception as e:
    print(f"❌ Erro ao abrir serial: {e}")
    sys.exit()

print(f"Monitorando {PORTA}... (Pressione Ctrl+C pra sair)")

# Inicia thread de leitura
thread_leitura = threading.Thread(target=ler_prints_esp, daemon=True)
thread_leitura.start()
time.sleep(2) # Espera o ESP reiniciar caso necessário

try:
    while True:
        cmd = input("\nComando (e=emerg, l=lib, a=semi, m=manual, q=sair): ").strip().lower()
        if not cmd: continue

        if cmd == 'q': break
        elif cmd.startswith('v'):  # NOVO: v50, v100, v200
            try:
                nova_vel = int(cmd[1:])  # Pega número após 'v'
                if 0 <= nova_vel <= 255:
                    velocidade_atual = nova_vel
                    esp.write(f"v{nova_vel}\n".encode())
                    #todo: enviar_pacote('V', velocidade_atual)
                    print(f"✅ Velocidade: {velocidade_atual}")
                else:
                    print("❌ Velocidade 0-255")
            except:
                print("❌ Use: v50, v100...")
        elif cmd == '+':  # Aumenta 10
            velocidade_atual = min(255, velocidade_atual + 10)
            #todo enviar_pacote('V', velocidade_atual)
            print(f"Aumentando Velocidade: {velocidade_atual}")
        elif cmd == '-':  # Diminui 10
            velocidade_atual = max(0, velocidade_atual - 10)
            #todo enviar_pacote('V', velocidade_atual)
            print(f"Diminuindo Velocidade: {velocidade_atual}")
        elif cmd in ['e', 'l', 'a', 'm']:
            esp.write((cmd + '\n').encode())
            #todo: enviar_pacote(cmd, 0)
            print(f"Enviado: {cmd}")
        else:
            print("❌ Comando desconhecido.")

except KeyboardInterrupt:
    print("\nSaindo...")
finally:
    if esp.is_open:
        esp.close()
    print("Porta fechada")
  


