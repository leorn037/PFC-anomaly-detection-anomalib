# 1. Imagem Base: Usamos uma imagem Python leve e específica
FROM python:3.10-slim

# 2. Diretório de Trabalho
WORKDIR /app

# 3. Copiar e instalar dependências primeiro (otimiza o cache do Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libxcb1 \
        libxcb-xinerama0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-xfixes0 && \
    rm -rf /var/lib/apt/lists/*


# 4. Copiar o restante do código
COPY . .

# 5. Portas Expostas (Apenas documentação para quem for usar a imagem)
EXPOSE 5007
EXPOSE 5008

# 6. Ponto de Entrada (ENTRYPOINT) e Comando Padrão (CMD)
# Usamos ENTRYPOINT para definir a base da execução e CMD para os argumentos opcionais.
# Isso permite que você execute 'anomalypc.py' e passe argumentos opcionais por fora.

# O ENTRYPOINT garante que 'anomalypc.py' seja o executável principal
ENTRYPOINT ["python", "anomaly_pc.py"]

# O CMD define os argumentos padrão, se você não passar nada no docker run
# Se preferir que o padrão seja rodar sem argumentos, deixe o CMD vazio: []
# CMD ["-m", "PatchCore"] 
# OU, se você não quer nenhum argumento padrão:
CMD []
