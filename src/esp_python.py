import serial
import threading
import time

PORTA = '/dev/ttyUSB0'
BAUDRATE = 115200

def ler_prints_esp():
    """Thread que lê TODOS os prints do ESP32 em tempo real"""
    while True:
        if esp.in_waiting > 0:
            linha = esp.readline().decode('utf-8').strip()
            if linha:
                print(f"[ESP32] {linha}")
        time.sleep(0.01)  # Não sobrecarrega CPU

esp = serial.Serial(PORTA, BAUDRATE, timeout=1)
print(f"Monitorando {PORTA}...")

# Inicia thread de leitura CONTÍNUA
thread_leitura = threading.Thread(target=ler_prints_esp, daemon=True)
thread_leitura.start()
time.sleep(2)

try:
    while True:
        cmd = input("\nComando (e=emerg, l=lib, a=semi-auto, m=manual, q=sair): ").strip().lower()
        if cmd == 'q': break
        esp.write((cmd + '\n').encode())
        print(f"Enviado: {cmd}")
except KeyboardInterrupt:
    print("\nSaindo...")
finally:
    esp.close()
