import serial
import threading
import time
import sys

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

# Configuração CORRETA da porta serial
esp = serial.Serial()
esp.port = PORTA
esp.baudrate = BAUDRATE
esp.timeout = 1
esp.setDTR(False)  # ← IMPORTANTE pro ESP32
esp.setRTS(False)  # ← IMPORTANTE pro ESP32
esp.open()

print(f"Monitorando {PORTA}... (Pressione Ctrl+C pra sair)")

# Inicia thread de leitura
thread_leitura = threading.Thread(target=ler_prints_esp, daemon=True)
thread_leitura.start()
time.sleep(2)

try:
    while True:
        cmd = input("\nComando (e=emerg, l=lib, a=semi, m=manual, q=sair): ").strip().lower()
        
        if cmd == 'q': break
        elif cmd.startswith('v'):  # NOVO: v50, v100, v200
            try:
                nova_vel = int(cmd[1:])  # Pega número após 'v'
                if 0 <= nova_vel <= 255:
                    velocidade_atual = nova_vel
                    esp.write(f"v{nova_vel}\n".encode())
                    print(f"✅ Velocidade: {velocidade_atual}")
                else:
                    print("❌ Velocidade 0-255")
            except:
                print("❌ Use: v50, v100...")
        elif cmd == '+':  # Aumenta 10
            velocidade_atual = min(255, velocidade_atual + 10)
            print(f"✅ Velocidade: {velocidade_atual}")
        elif cmd == '-':  # Diminui 10
            velocidade_atual = max(0, velocidade_atual - 10)
            print(f"✅ Velocidade: {velocidade_atual}")
        else:
            esp.write((cmd + '\n').encode())
            print(f"Enviado: {cmd}")

except KeyboardInterrupt:
    print("\nSaindo...")
finally:
    esp.close()
    print("Porta fechada")
  


