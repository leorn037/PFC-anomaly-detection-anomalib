import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

class CaboTracker:
    def __init__(self, crop_output_size: int | None = None):
        """
        Tracker robusto para cabo de aço com estabilização temporal.
        
        Args:
            crop_output_size: Tamanho final do frame quadrado.
        """
        self.crop_output_size = crop_output_size

        # --- LIMITES RÍGIDOS PARA AS COORDENADAS X ---
        self.x_min = 220
        self.x_max = 380

        self.MIN_CABLE_WIDTH = 100

        self.margin_percent = 0.05  # de cada lado
        
        # Parâmetros de tracking
        self.alpha_mask = 0.3  # Peso da média exponencial
    
         
    def track(self, frame_bgr: np.ndarray, debug_show_steps: bool = False) -> np.ndarray:
        """
        Processa frame único com todas as técnicas de estabilização.
        
        Args:
            frame_bgr: Imagem BGR (OpenCV).
            debug_show_steps: Mostra debug visual.
            
        Returns:
            Frame centralizado e estabilizado.
        """
        h, w, _ = frame_bgr.shape
        
        # 2. FUNÇÃO CENTRALIZAR
        final_frame, left, right = self._centralizar_cabo(frame_bgr, self.crop_output_size)

        if debug_show_steps:
            self._debug_visualization(frame_bgr,left, right)

        debug_img = frame_bgr.copy()
        h_roi, w_roi = debug_img.shape[:2]
        
        cv2.line(debug_img, (left, 0), (left, h_roi), (0, 255, 0), 3)
        cv2.line(debug_img, (right, 0), (right, h_roi), (0, 255, 0), 3)
        

        return final_frame
    
    def _centralizar_cabo(self, frame_roi: np.ndarray, crop_output_size: int | None):
        """
        === AQUI VOCÊ COLOCA TODO O SEU CÓDIGO DA FUNÇÃO ORIGINAL ===
        Substitua esta implementação placeholder pela sua função completa.
        """
        # SEU CÓDIGO AQUI: HSV, morph, contornos, Sobel fallback, left/right, canvas...
        h_roi, w_roi, _ = frame_roi.shape

        self.gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
        #blurred = cv2.GaussianBlur(self.gray, (3, 3), 0)  # Remove ruído movimento

        # 1. CANNY (robusto a borrão)
        self.edges = cv2.Canny(self.gray, 50, 150, apertureSize=3) 

        # 2. HOUGH LINES VERTICAIS
        lines = cv2.HoughLinesP(self.edges, 1, np.pi/180, threshold=40, 
                            minLineLength=h_roi*0.3, maxLineGap=50) 

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
                        if center_x < w_roi // 2:
                            self.left_candidates.append(center_x)
                        else:
                            self.right_candidates.append(center_x)


        #* print(f"Hough: {len(self.left_candidates)} esquerda(s), {len(self.right_candidates)} direita(s)")

        center_img = w_roi // 2
        tolerance = 40  # Pixels do centro para IGNORAR
        # FILTRA candidatos próximos ao centro (ruído)
        self.left_candidates = [x for x in self.left_candidates if abs(x - center_img) > tolerance]
        self.right_candidates = [x for x in self.right_candidates if abs(x - center_img) > tolerance]

        #* print(f"Hough FILTRADO: {len(self.left_candidates)} Esq, {len(self.right_candidates)} Dir (ignorou ±{tolerance}px centro)")


        # PRIORIDADE 1: LÓGICA DAS BORDAS
        if self.left_candidates and self.right_candidates:
            # Borda DIREITA das esquerdas = borda ESQUERDA do cabo
            left_border = max(self.left_candidates)  
            # Borda ESQUERDA das direitas = borda DIREITA do cabo  
            right_border = min(self.right_candidates)
            initial_width = right_border - left_border

            # MARGEM DO INTERVALO FINAL
            margin = int(initial_width * self.margin_percent)

            left = int(left_border + margin)
            right = int(right_border - margin)
            
            #* print(f"Bordas reais: L={left}, R={right}, largura={right-left}px")

        elif self.left_candidates:
            # SÓ ESQUERDA: usa + MIN_WIDTH à direita
            left_border = max(self.left_candidates)
            left = int(left_border + int(self.MIN_CABLE_WIDTH * self.margin_percent))  # Pequena margem
            right = int(left + self.MIN_CABLE_WIDTH)
            #* print(f"⚠️ SÓ ESQUERDA: L={left}, R={right} (width fixo)")
                
        elif self.right_candidates:
            # SÓ DIREITA: usa MIN_WIDTH à esquerda
            right_border = min(self.right_candidates)
            right = int(right_border - int(self.MIN_CABLE_WIDTH * self.margin_percent))
            left = int(right - self.MIN_CABLE_WIDTH)
            #* print(f"⚠️ SÓ DIREITA: L={left}, R={right} (width fixo)")
                
        else:
            # NENHUM lado
            left, right = self.x_min, self.x_max
            #* print("❌ SEM BORDAS: usando zona completa")

        largura_atual = right - left

        if largura_atual < self.MIN_CABLE_WIDTH:
            #* print(f"🔧 EXPANDINDO (centrado): {left}→{right} ({largura_atual}px)")
            
            ajuste_total = self.MIN_CABLE_WIDTH - largura_atual
            center_cut = (left + right) // 2  # CENTRO do corte atual
            
            # DISTÂNCIAS do centro da imagem
            dist_left = abs(center_img - left)   # Quanto left tá longe do centro
            dist_right = abs(center_img - right) # Quanto right tá longe do centro
            
            # PRIORIDADE: quem tá MAIS PERTO do centro ganha MAIS expansão
            peso_left = 1.0 / (1 + dist_left * 0.01)   # Perto = peso alto (1.0)
            peso_right = 1.0 / (1 + dist_right * 0.01) # Longe = peso baixo (0.3)
            
            total_peso = peso_left + peso_right
            ajuste_left = int(ajuste_total * (peso_left / total_peso))
            ajuste_right = int(ajuste_total * (peso_right / total_peso))
            
            #* print(f"   Pesos: Esq={peso_left:.2f}, Dir={peso_right:.2f}")
            #* print(f"   Ajustes: Esq={ajuste_left}px, Dir={ajuste_right}px")
            
            # Expansão ESQUERDA (prioridade se perto do centro)
            espaco_esq = left - self.x_min
            shift_esq = min(ajuste_left, espaco_esq)
            left -= shift_esq
            
            # Expansão DIREITA
            espaco_dir = self.x_max - right
            shift_dir = min(ajuste_right, espaco_dir)
            right += shift_dir
            
            # Sobra vai pro lado com MAIS ESPAÇO (não pro centro)
            sobra_total = ajuste_total - (shift_esq + shift_dir)
            if sobra_total > 0:
                espaco_restante_esq = left - self.x_min
                espaco_restante_dir = self.x_max - right
                
                if espaco_restante_esq >= espaco_restante_dir:
                    left = max(self.x_min, left - sobra_total)
                    #* print(f"   ↩️ Sobra {sobra_total}px → ESQUERDA (mais espaço)")
                else:
                    right = min(self.x_max, right + sobra_total)
                    #* print(f"   ↪️ Sobra {sobra_total}px → DIREITA (mais espaço)")
            
            #* print(f"✅ FINAL: {left}→{right} ({right-left}px)")

        #* print(f"Largura final do corte: {right - left}px")
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
        
        return canvas, left, right

    def _debug_visualization(self,  frame_roi: np.ndarray, 
                                    left: int, right: int):
        """Debug COMPLETO com 6 painéis + Fallback Sobel."""
            
        h_roi, w_roi = frame_roi.shape[:2]
        
        # LAYOUT: 2x3 com perfil ocupando 2 colunas embaixo
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1])
        
        # Axes com SINCRONIZAÇÃO X
        ax_img  = fig.add_subplot(gs[0, 0])
        ax_hsv  = fig.add_subplot(gs[0, 1]) 
        ax_mask = fig.add_subplot(gs[0, 2])
        ax_cont = fig.add_subplot(gs[1, 0])
        ax_sobel= fig.add_subplot(gs[1, 1])
        ax_supp = fig.add_subplot(gs[1, 2])  # Perfil ocupa última coluna
        
        # === 1. FRAME ORIGINAL COMPLETO ===
        ax_img.imshow(cv2.cvtColor(frame_roi, cv2.COLOR_BGR2RGB))
        ax_img.axvline(left, color='lime', linewidth=1, label=f'Corte Final\n({left}-{right}px)')
        ax_img.axvline(right, color='lime', linewidth=1)
        ax_img.axvspan(self.x_min, self.x_max, color='yellow', alpha=0.2, label='Zona Permitida')
        ax_img.set_title("1. FRAME COMPLETO\n(ROI + Cortes)")
        ax_img.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # === 2. GRAY + CANNY EDGES ===
        ax_hsv.imshow(self.gray, cmap='gray')
        ax_hsv.set_title("2. gray (pré-Canny)")
        
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
            right_slice = min(w_roi, self.x_max)
            support_profile[left_slice:right_slice] += 1

        edge_profile = np.sum(self.edges > 0, axis=0) / max(h_roi, 1)

        ax_supp.plot(support_profile, 'b-', linewidth=3, label=f'Hough Lines\nN={len(self.vertical_lines)}')
        
        # Canny COMPLETO (sem corte estranho)
        ax_supp.plot(edge_profile, 'cyan', linewidth=2, alpha=0.8, label='Canny Completo')

        # Linhas de corte FINAL
        ax_supp.axvline(left, color='lime', linewidth=1, label=f'Corte {left}-{right}px')
        ax_supp.axvline(right, color='lime', linewidth=1)

        # Zona de análise
        ax_supp.axvspan(self.x_min, self.x_max, color='yellow', alpha=0.2, label='Zona Análise (20-80%)')

        ax_supp.grid(True, alpha=0.3)
        ax_supp.set_title("6. PERFIS Hough + Canny")
        ax_supp.legend(fontsize='small')
    
        # Título indica MÉTODO usado
        metodo = "HSV ✓" if self.vertical_lines else "SOBEL ✓"
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
        print("Tracker resetado")

# === USO ===
# tracker = CaboTracker(crop_output_size=640)
# while True:
#     ret, frame = cap.read()
#     frame_out = tracker.track(frame, debug_show_steps=True)
#     cv2.imshow('Cabo Estabilizado', frame_out)
#     if cv2.waitKey(1) == 27: break
