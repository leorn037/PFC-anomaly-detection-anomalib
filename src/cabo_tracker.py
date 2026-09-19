import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import time

class CaboTracker:
    def __init__(self, crop_output_size: int | None = None):
        """
        Tracker robusto para cabo de aço com estabilização temporal.
        
        Args:
            crop_output_size: Tamanho final do frame quadrado.
        """
        self.crop_output_size = crop_output_size

        '''self.MIN_CABLE_WIDTH = 80
        self.MAX_CABLE_WIDTH = 150

        self.margin_percent = 0.05  # de cada lado

        # ROI DINÂMICA (MEMÓRIA) ---
        self.last_center_x = None  # Começa vazia pois não sabemos onde está o cabo
        self.roi_window = 150      # Olha 100px para cada lado (Total 200px de largura)
        self.last_width = self.MIN_CABLE_WIDTH'''

        # Largura não é mais fixa: aprendida em tempo real a partir de detecções confiáveis.
        self.largura_estimada = None       # referência adaptativa (None até o bootstrap terminar)
        self.LARGURA_TOLERANCIA = 0.25     # +-25% da largura estimada vira o min/max de sanidade
        self.LARGURA_FALLBACK = 80         # só usado enquanto largura_estimada ainda é None
        self.HISTORICO_LARGURA_N = 15                # tamanho da janela usada pra mediana
        self.historico_larguras_confiaveis = []      # janela deslizante de larguras confiáveis

        self.MIN_VOTOS_CONFIANCA = 2       # exige >=2 linhas de cada lado pra considerar "confiável"

        self.bootstrap_larguras = []       # histórico curto de larguras confiáveis, até calibrar
        self.BOOTSTRAP_N = 5               # quantos frames confiáveis seguidos exige
        self.BOOTSTRAP_TOLERANCIA = 0.15   # variação máxima entre eles pra aceitar como semente

        self.margin_percent = 0.05  # de cada lado

        # ROI DINÂMICA (MEMÓRIA) ---
        self.last_center_x = None
        self.roi_window = 100
        self.last_width = self.LARGURA_FALLBACK   #!# antes: self.MIN_CABLE_WIDTH
     
    def track(self, frame_bgr: np.ndarray, debug_show_steps: bool = False) -> np.ndarray:
        """
        Processa frame único com todas as técnicas de estabilização.
        
        Args:
            frame_bgr: Imagem BGR (OpenCV).
            debug_show_steps: Mostra debug visual.
            
        Returns:
            Frame centralizado e estabilizado.
        """
        
        # 2. FUNÇÃO CENTRALIZAR
        final_frame, left, right = self._centralizar_cabo(frame_bgr, self.crop_output_size, debug=debug_show_steps)

        if debug_show_steps:
            self._debug_visualization(frame_bgr,left, right)
        
        return final_frame
    
    def _centralizar_cabo(self, frame_roi: np.ndarray, crop_output_size: int | None, debug: bool = False):

        h_roi, w_roi, _ = frame_roi.shape

        self.gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)

        # APLICAÇÃO DA ROI DINÂMICA
        if self.last_center_x is not None:
            # Define os limites da janela de busca
            x_start = max(0, self.last_center_x - self.roi_window)
            x_end = min(w_roi, self.last_center_x + self.roi_window)
            
            # Pinta de PRETO (0) tudo que está fora da janela
            # Isso "apaga" a escada e a parede antes do Canny rodar
            self.gray[:, :x_start] = 0  # Zera esquerda
            self.gray[:, x_end:] = 0    # Zera direita
            
        else:
            self.last_center_x = w_roi // 2
            
        # APLICA BLUR APENAS VERTICAL: 
        self.gray = self._linearizar_trancado(self.gray)
        # 1. CANNY (robusto a borrão)
        self.edges = cv2.Canny(self.gray, 50, 100, apertureSize=3) 
    
        # 2. HOUGH LINES VERTICAIS
        lines = cv2.HoughLinesP(self.edges, 1, np.pi/180, threshold=30, 
                            minLineLength=h_roi*0.1, maxLineGap=50)

        self.vertical_lines = []
        self.left_candidates = []   # Bordas esquerdas do cabo
        self.right_candidates = [] 
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # FILTRA só VERTICAIS (|Δy| > 5×|Δx|)
                if abs(y2-y1) > 5 * abs(x2-x1) and abs(x2-x1) < 30:  # Largura cabo
                    center_x = int((x1+x2)//2)
                    if w_roi * 0.2 <= center_x <= w_roi * 0.8:
                        self.vertical_lines.append(center_x)
                        if center_x < self.last_center_x:
                            self.left_candidates.append(center_x)
                        else:
                            self.right_candidates.append(center_x)

        # Confiança alta = múltiplas linhas concordando dos dois lados
        confianca_alta = (len(self.left_candidates) >= self.MIN_VOTOS_CONFIANCA and
                        len(self.right_candidates) >= self.MIN_VOTOS_CONFIANCA)
        largura_bruta = None  # guarda a largura "crua" da Prioridade 1, antes de correções de sanidade

        # Referência de largura: usa a aprendida, ou o fallback enquanto não calibrou
        largura_referencia = self.largura_estimada if self.largura_estimada is not None else self.LARGURA_FALLBACK

        # PRIORIDADE 1: LÓGICA DAS BORDAS
        if self.left_candidates and self.right_candidates:
            # Borda DIREITA das esquerdas = borda ESQUERDA do cabo
            left_border = max(self.left_candidates)
            # Borda ESQUERDA das direitas = borda DIREITA do cabo  
            right_border = min(self.right_candidates)
            
            initial_width = right_border - left_border
            largura_bruta = initial_width  # guarda pra eventual aprendizado, mais abaixo

            # MARGEM DO INTERVALO FINAL
            margin = int(initial_width * self.margin_percent)

            left = int(left_border + margin)
            right = int(right_border - margin)
            
            cabo_encontrado = True

            if debug: print(f"Bordas reais: L={left}, R={right}, largura={right-left}px")

        elif self.left_candidates:
            # SÓ ESQUERDA: usa + MIN_WIDTH à direita
            left_border = max(self.left_candidates)
            left = int(left_border + int(largura_referencia * self.margin_percent))  # Pequena margem
            right = int(left + largura_referencia)
            cabo_encontrado = True
            if debug: print(f"⚠️ SÓ ESQUERDA: L={left}, R={right} (width fixo)")
                
        elif self.right_candidates:
            # SÓ DIREITA: usa MIN_WIDTH à esquerda
            right_border = min(self.right_candidates)
            right = int(right_border - int(largura_referencia * self.margin_percent))  #!# antes: self.MIN_CABLE_WIDTH
            left = int(right - largura_referencia)   
            cabo_encontrado = True
            if debug: print(f"⚠️ SÓ DIREITA: L={left}, R={right} (width fixo)")
                
        else:
            # NENHUM lado
            cabo_encontrado=False
            half_w = self.last_width // 2
            left = max(0, self.last_center_x - half_w)
            right = min(w_roi, self.last_center_x + half_w)
            if debug: print("❌ SEM BORDAS: usando ultima posição")

        largura_atual = right - left

        # min/max relativos à largura aprendida, não constantes fixas
        min_width = largura_referencia * (1 - self.LARGURA_TOLERANCIA)
        max_width = largura_referencia * (1 + self.LARGURA_TOLERANCIA)

        if largura_atual > max_width:
                # Calcula distâncias para o centro esperado
                dist_l = abs(self.last_center_x - left)
                dist_r = abs(self.last_center_x - right)

                if dist_l < dist_r:
                    # A linha da ESQUERDA é a verdadeira (está mais perto do histórico)
                    right = int(left + largura_referencia) # Completa artificialmente a direita
                    if debug: print(f"⚠️ Largura {largura_atual}px > Max. Salvando lado ESQ (Dist {dist_l} vs {dist_r})")
                else:
                    # A linha da DIREITA é a verdadeira
                    left = int(right - largura_referencia) # Completa artificialmente a esquerda
                    if debug: print(f"⚠️ Largura {largura_atual}px > Max. Salvando lado DIR (Dist {dist_r} vs {dist_l})")
                
                cabo_encontrado = True


        if largura_atual < min_width:
            if debug: print(f"🔧 EXPANDINDO (centrado): {left}→{right} ({largura_atual}px)")
            
            ajuste_total = min_width - largura_atual
            
            # DISTÂNCIAS do centro da imagem
            dist_left = abs(self.last_center_x - left)   # Quanto left tá longe do centro
            dist_right = abs(self.last_center_x - right) # Quanto right tá longe do centro
            
            # PRIORIDADE: quem tá MAIS PERTO do centro ganha MAIS expansão
            peso_left = 1.0 / (1 + dist_left * 0.01)   # Perto = peso alto (1.0)
            peso_right = 1.0 / (1 + dist_right * 0.01) # Longe = peso baixo (0.3)
            
            total_peso = peso_left + peso_right
            ajuste_left = int(ajuste_total * (peso_left / total_peso))
            ajuste_right = int(ajuste_total * (peso_right / total_peso))
            
            if debug: print(f"   Pesos: Esq={peso_left:.2f}, Dir={peso_right:.2f}")
            if debug: print(f"   Ajustes: Esq={ajuste_left}px, Dir={ajuste_right}px")
            
            # Aplica expansão direta (respeitando apenas a borda da imagem 0 e w_roi)
            left = max(0, left - ajuste_left)
            right = min(w_roi, right + ajuste_right)
            
            if debug: print(f"✅ FINAL: {left}→{right} ({right-left}px)")

        centro_geometrico = w_roi // 2
        centro_detectado = (left + right) // 2
        limite_desvio = int(w_roi * 0.25)

        if centro_geometrico < left or centro_geometrico > right:
            #o cabo, fisicamente, nunca fica longe do centro da imagem durante
            # operação normal — só quando alguém reposiciona a câmera manualmente. Se o
            # centro detectado está muito longe do centro geométrico, é sinal de falha
            # (travou numa estrutura errada) ou reposicionamento real — nos dois casos,
            # o certo é não confiar nessa detecção e recomeçar a busca a partir do centro.
            if debug:
                print(f"🚨 Centro detectado ({centro_detectado}) muito longe do centro geométrico. "
                    f"Resetando busca.")
            self.last_center_x = centro_geometrico
            self.last_width = self.LARGURA_FALLBACK
            cabo_encontrado = False
            left = max(0, centro_geometrico - self.last_width // 2)
            right = min(w_roi, centro_geometrico + self.last_width // 2)
            
        if debug: print(f"Largura final do corte: {right - left}px")

        if cabo_encontrado:
            # Sucesso: Atualiza memória
            self.last_center_x = (left + right) // 2
            self.last_width = right - left
            # Usa a última largura conhecida centrada na última posição

            # Aprendizado da largura de referência — só a partir de detecções confiáveis
            if confianca_alta and largura_bruta is not None:
                #!# Só deixa a leitura entrar no histórico se ela já é "sã" pelos padrões atuais
                #!# (os mesmos limites usados pra corrigir o corte, min_width/max_width). Isso
                #!# rejeita leituras vindas de situações anômalas (câmera fora do cabo, por
                #!# exemplo) ANTES delas contaminarem a mediana — em vez de deixar entrar e
                #!# só descontaminar depois de N frames bons.
                dentro_da_tolerancia = (self.largura_estimada is None) or (min_width <= largura_bruta <= max_width)

                if dentro_da_tolerancia:
                    if self.largura_estimada is None:
                        self.bootstrap_larguras.append(largura_bruta)
                        if len(self.bootstrap_larguras) > self.BOOTSTRAP_N:
                            self.bootstrap_larguras.pop(0)
                        if len(self.bootstrap_larguras) == self.BOOTSTRAP_N:
                            media = sum(self.bootstrap_larguras) / self.BOOTSTRAP_N
                            if all(abs(w - media) <= media * self.BOOTSTRAP_TOLERANCIA for w in self.bootstrap_larguras):
                                self.largura_estimada = media
                                if debug: print(f"✅ Bootstrap concluído: largura_estimada={self.largura_estimada:.1f}px")
                            elif debug:
                                print("⏳ Bootstrap: últimas medidas inconsistentes, aguardando mais frames")
                    else:
                        # Mediana das últimas N larguras confiáveis, resistente a 1-2 outliers,
                        self.historico_larguras_confiaveis.append(largura_bruta)
                        if len(self.historico_larguras_confiaveis) > self.HISTORICO_LARGURA_N:
                            self.historico_larguras_confiaveis.pop(0)
                        self.largura_estimada = float(np.median(self.historico_larguras_confiaveis))

        cropped = frame_roi[:, left:right]
        if crop_output_size:
            # Cria um "Canvas" (Fundo) preto quadrado do tamanho final
            canvas = np.zeros((crop_output_size, crop_output_size, 3), dtype=np.uint8)
            # Calcula a escala para redimensionar MANTENDO PROPORÇÃO
            h_crop, w_crop = cropped.shape[:2]
            # Descobre qual dimensão limita o redimensionamento (normalmente a altura)
            scale = min(crop_output_size / h_crop, crop_output_size / w_crop)
            # Calcula novas dimensões proporcionais
            new_w, new_h = int(w_crop * scale), int(h_crop * scale)
            resized = cv2.resize(cropped, (new_w, new_h))
            # Calcula offsets para centralizar no canvas
            x_off, y_off = (crop_output_size - new_w) // 2, (crop_output_size - new_h) // 2
            # Cola a imagem redimensionada no centro do canvas
            canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
            cropped = canvas

        return cropped, left, right

    def _linearizar_trancado(self, imagem_gray):
        # Kernel (1, 51) significa: 1px de largura, 51px de altura.
        # Isso borra tudo violentamente na vertical, transformando a senóide numa "coluna".
        # Aumente o 51 se a ondulação for muito "longa".
        kernel_vertical = (1, 51) 
        blur_vertical = cv2.blur(imagem_gray, kernel_vertical)
        return blur_vertical

    def _debug_visualization(self,  frame_roi: np.ndarray, 
                                    left: int, right: int):
        """Debug COMPLETO com 6 painéis + Fallback Sobel."""
            
        h_roi, w_roi = frame_roi.shape[:2]
        
        # LAYOUT: 2x3 com perfil ocupando 2 colunas embaixo
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1])
        
        # Axes com SINCRONIZAÇÃO X
        ax_img  = fig.add_subplot(gs[0, 0])
        ax_gray  = fig.add_subplot(gs[0, 1]) 
        ax_mask = fig.add_subplot(gs[0, 2])
        ax_cont = fig.add_subplot(gs[1, 0])
        ax_supp = fig.add_subplot(gs[1, 2])  # Perfil ocupa última coluna
        
        # === 1. FRAME ORIGINAL COMPLETO ===
        ax_img.imshow(cv2.cvtColor(frame_roi, cv2.COLOR_BGR2RGB))
        ax_img.axvline(left, color='lime', linewidth=1, label=f'Corte Final\n({left}-{right}px)')
        ax_img.axvline(right, color='lime', linewidth=1)
        ax_img.set_title("1. FRAME COMPLETO\n(ROI + Cortes)")
        ax_img.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # === 2. GRAY + CANNY EDGES ===
        ax_gray.imshow(self.gray, cmap='gray')
        ax_gray.set_title("2. gray (pré-Canny)")
        
        # === 3. CANNY EDGES ===
        ax_mask.imshow(self.edges, cmap='gray')
        ax_mask.set_title("3. Canny Edges\n(30,100,aperture=5)")
        
        # === 4. LINHAS VERTICAIS HOUGH ===
        debug_lines = frame_roi.copy()
        left_candidates_debug = []
        right_candidates_debug = []

        for cx in self.vertical_lines:  # ✅ SÓ CX agora
            # Verde: linha vertical no centro detectado
            cv2.line(debug_lines, (cx, 0), (cx, h_roi), (0, 255, 0), 1)
            
            # Azul: largura estimada do cabo (±30px)
            cv2.line(debug_lines, (cx-30, h_roi//2), (cx+30, h_roi//2), (255, 0, 0), 1)
            
            # Classifica visualmente
            if cx < w_roi // 2:
                left_candidates_debug.append(cx)
                cv2.circle(debug_lines, (cx, h_roi//2), 8, (255, 0, 255), -1)  # Magenta = esquerda
            else:
                right_candidates_debug.append(cx)
                cv2.circle(debug_lines, (cx, h_roi//2), 8, (0, 0, 255), -1)   # Vermelho = direita 
        
        # Destaca BORDAS FINAIS
        if self.left_candidates and self.right_candidates:
            left_border = max(self.left_candidates)
            right_border = min(self.right_candidates)
            cv2.rectangle(debug_lines, (left_border, 0), (right_border, h_roi), (0, 255, 255), 2)  # Amarelo = bordas

        ax_cont.imshow(cv2.cvtColor(debug_lines, cv2.COLOR_BGR2RGB))
        ax_cont.set_title(f"4. Hough Lines\nVerde=Centros ({len(self.vertical_lines)})\nMagenta=Esq, Verm=Direita")

        
        # === 6. PERFIS HOUGH + CANNY ===
        support_profile = np.zeros(w_roi)

        for cx in self.vertical_lines:
            left_slice = max(0, int(cx - 25))
            right_slice = min(w_roi, int(cx + 25))

            # Proteção extra para garantir que left < right
            if right_slice > left_slice:
                support_profile[left_slice:right_slice] += 1

        edge_profile = np.sum(self.edges > 0, axis=0) / max(h_roi, 1)

        ax_supp.plot(support_profile, 'b-', linewidth=3, label=f'Hough Lines\nN={len(self.vertical_lines)}')
        
        # Canny COMPLETO (sem corte estranho)
        ax_supp.plot(edge_profile, 'cyan', linewidth=2, alpha=0.8, label='Canny Completo')

        # Linhas de corte FINAL
        ax_supp.axvline(left, color='lime', linewidth=1, label=f'Corte {left}-{right}px')
        ax_supp.axvline(right, color='lime', linewidth=1)

        ax_supp.grid(True, alpha=0.3)
        ax_supp.set_title("6. PERFIS Hough + Canny")
        ax_supp.legend(fontsize='small')
    
        # Título indica MÉTODO usado
        metodo = "Hough ✓" if self.vertical_lines else "Fallback (última posição) ⚠"
        plt.suptitle(f"CABO TRACKER | {metodo} | Largura Final: {right-left}px", 
                    fontsize=16, y=0.98, color='green' if self.vertical_lines else 'orange')
        plt.tight_layout()

        mng = plt.get_current_fig_manager()
        try:
            # Tkinter (mais comum no Windows)
            mng.window.wm_geometry("+0+0")
        except:
            try:
                # Qt
                mng.window.move(0, 0)
            except:
                # WXAgg
                mng.frame.SetPosition((0, 0))
                # Windows
                    
        plt.show()

    def reset(self):
        """Reset estado para nova sessão."""
        self.last_center_x = None
        self.last_width = self.LARGURA_FALLBACK
        self.largura_estimada = None
        self.bootstrap_larguras = []
        self.historico_larguras_confiaveis = []
        print("Tracker resetado")

# === USO ===
# tracker = CaboTracker(crop_output_size=640)
# while True:
#     ret, frame = cap.read()
#     frame_out = tracker.track(frame, debug_show_steps=True)
#     cv2.imshow('Cabo Estabilizado', frame_out)
#     if cv2.waitKey(1) == 27: break
