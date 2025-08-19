import os
import random
import shutil

def selecionar_e_mover_imagens(pasta_origem, nome_pasta_destino="normal", quantidade=5):
    # Cria a pasta destino dentro da pasta de origem
    pasta_destino = os.path.join(pasta_origem, nome_pasta_destino)
    os.makedirs(pasta_destino, exist_ok=True)

    # Lista todas as imagens (aqui estou considerando extensões comuns)
    extensoes_validas = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
    imagens = [f for f in os.listdir(pasta_origem) if f.lower().endswith(extensoes_validas)]

    if not imagens:
        print("Nenhuma imagem encontrada na pasta.")
        return

    # Se a quantidade for maior que o número de imagens disponíveis, reduz
    quantidade = min(quantidade, len(imagens))

    # Seleciona aleatoriamente
    selecionadas = random.sample(imagens, quantidade)

    # Move as imagens
    for img in selecionadas:
        origem = os.path.join(pasta_origem, img)
        destino = os.path.join(pasta_destino, img)
        shutil.move(origem, destino)
        print(f"Movido: {img}")

    print(f"\n{quantidade} imagens foram movidas para {pasta_destino}")

# Exemplo de uso:
selecionar_e_mover_imagens("C:/Users/Leonardo/Downloads/Programas/PFC/brnch/img_640", quantidade=100)
