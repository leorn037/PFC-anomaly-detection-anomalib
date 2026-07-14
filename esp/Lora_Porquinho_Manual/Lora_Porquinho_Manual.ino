/* Código desenvolvido para o módulo autônomo de queima
 * Autor: Rafael Lins Nobre e Leonardo Rodrigues Nascimento
 * */

#include "Arduino.h"
#include "LoRaWan_APP.h"


#define RF_FREQUENCY 868000000  // Hz

#define TX_OUTPUT_POWER 14  // dBm

#define LORA_BANDWIDTH 1         // [0: 125 kHz, \
                                 //  1: 250 kHz, \
                                 //  2: 500 kHz, \
                                 //  3: Reserved]
#define LORA_SPREADING_FACTOR 7  // [SF7..SF12]
#define LORA_CODINGRATE 1        // [1: 4/5, \
                                 //  2: 4/6, \
                                 //  3: 4/7, \
                                 //  4: 4/8]
#define LORA_PREAMBLE_LENGTH 8   // Same for Tx and Rx
#define LORA_SYMBOL_TIMEOUT 0    // Symbols
#define LORA_FIX_LENGTH_PAYLOAD_ON false
#define LORA_IQ_INVERSION_ON false


#define RX_TIMEOUT_VALUE 1000
#define BUFFER_SIZE 30  // Define the payload size here

char txpacket[BUFFER_SIZE];
char rxpacket[BUFFER_SIZE];

static RadioEvents_t RadioEvents;
int16_t rssi, rxSize;
bool lora_idle = true;

// --- Pinos de Atuadores e Motores ---
#define ATUADOR_AVANCO 46
#define ATUADOR_RECUO 45

#define MOTOR_AVANCO 41
#define MOTOR_RECUO 42

int PWM = 0, pwm = 0;
bool emergency_stop = false;

#define RPI_PIN_ANOMALIA 6  //
#define RPI_PIN_PAUSA 7     //

// Define a velocidade padrão para a inspeção (ajuste conforme necessário)
#define VELOCIDADE_INSPECAO 75  // Valor de PWM (0-255)
#define TEMPO_MOVER_MACARICO 2000

#define COMPENSACAO_ATRASO_MS 500 // Tempo em ms para o robô recuar e alinhar o maçarico

enum ModoOperacao {
  MANUAL,
  SEMI_AUTOMATICO
};

enum EstadoQueima {
  OCIOSO,        // Robô em inspeção normal ou parado (antes/depois da queima)
  INICIO_RECUO,  // 1. Avançar por X segundos
  AVANCO_ALVO,   // 2. Recuar por 2X segundos
  RECUO_RETORNO  // 3. Avançar por X segundos para voltar à posição original
};

ModoOperacao modoAtual = SEMI_AUTOMATICO;  //MANUAL; // Começa em modo MANUAL por segurança
EstadoQueima estadoQueimaAtual = OCIOSO;

// Variáveis de parâmetro para a rotina de queima (podem ser alteradas via LoRa/config)
int V_VELOCIDADE_PWM = 75;          // Velocidade Constante (V) para a queima (0-255)
unsigned long X_DURACAO_MS = 2000;  // Duração base do movimento (X) em milissegundos

// Variáveis de tempo
unsigned long tempoInicioRotina = 0;

// Quantas vezes a rotina vai repetir?
int NUM_CICLOS_QUEIMA = 1;  
int contadorCiclos = 0;     // Variável interna para contar

// --- Variáveis do Watchdog Failsafe ---
unsigned long ultimoPacoteRecebido = 0;
#define WATCHDOG_TIMEOUT 500  // 500ms. Se não receber nada, para.

// Flag para interrupção (volatile é obrigatório aqui)
volatile bool anomaliaDetectada = false;

// Controle de Rampa
float velocidadeAtual = 0; // Float para suavidade matemática
int velocidadeAlvo = 0;    // Onde queremos chegar

// Rampa de aceleração
const float PASSO_ACELERACAO = 2.5; // Incremento por ciclo
const float PASSO_FREIO = 8.0;      // Freio deve ser mais forte que aceleração

// --- Definição do Pacote Binário (4 Bytes) ---
struct __attribute__((packed)) PacoteComando {
  char tipo;       // 1 Byte: 'E', 'L', 'A', 'M', 'V'
  int16_t valor;   // 2 Bytes: Valor numérico (-32768 a 32767)
  char fim;        // 1 Byte: '\n' (segurança para sincronia)
};

PacoteComando pacote; // Cria a variável global para receber os dados


// --- Funções Auxiliares de Controle ---

void lerSerialBinario() {
  // 1. Verifica se há pelo menos 4 bytes (tamanho da struct) no buffer
  if (Serial.available() >= sizeof(PacoteComando)) {
    
    // 2. Lê os bytes brutos direto para a memória da struct
    // Isso é instantâneo, sem conversão de texto.
    Serial.readBytes((char*)&pacote, sizeof(PacoteComando));

    // 3. Validação de Sincronia
    // Se o último caractere não for '\n', perdemos o alinhamento (ruído ou erro).
    if (pacote.fim != '\n') {
      Serial.println("ERRO: Pacote desalinhado/ruído. Limpando buffer...");
      // Esvazia o buffer até achar o próximo '\n' ou acabar, para tentar realinhar
      while(Serial.available() > 0) Serial.read(); 
      return; // Aborta e tenta na próxima
    }

    // 4. Processamento dos Comandos (Switch Case é mais rápido que if/else)
    switch (pacote.tipo) {
      
      // --- EMERGÊNCIA ---
      case 'E': 
      case 'e':
        Serial.println("CMD BIN: EMERGENCIA");
        emergency_stop = true;
        estadoQueimaAtual = OCIOSO; // Reseta máquina de estados
        
        // Para motores imediatamente
        analogWrite(MOTOR_RECUO, 0);
        analogWrite(MOTOR_AVANCO, 0);

        analogWrite(ATUADOR_AVANCO, 255);
        analogWrite(ATUADOR_RECUO, 0);
        delay(1000); 
        analogWrite(ATUADOR_AVANCO, 0);
        break;

      // --- LIBERAR ---
      case 'L':
      case 'l':
        Serial.println("CMD BIN: LIBERADO");
        emergency_stop = false;
        
        analogWrite(ATUADOR_AVANCO, 0);
        analogWrite(ATUADOR_RECUO, 255);
        analogWrite(MOTOR_RECUO, 0);
        analogWrite(MOTOR_AVANCO, 0);
        delay(1000);
        analogWrite(ATUADOR_RECUO, 0);
        break;

      // --- MODO SEMI-AUTOMÁTICO ---
      case 'A':
      case 'a':
        Serial.println("CMD BIN: MODO SEMI-AUTO");
        modoAtual = SEMI_AUTOMATICO;
        pararTodosMotoresAtuadores();
        break;

      // --- MODO MANUAL ---
      case 'M':
      case 'm':
        Serial.println("CMD BIN: MODO MANUAL");
        modoAtual = MANUAL;
        pararTodosMotoresAtuadores();
        break;

      // --- VELOCIDADE ---
      case 'V':
      case 'v':
        // Aqui usamos o int16_t 'valor' que veio no pacote
        if (pacote.valor >= 0 && pacote.valor <= 255) {
           V_VELOCIDADE_PWM = pacote.valor;
           Serial.print("CMD BIN: Vel ajustada para ");
           Serial.println(V_VELOCIDADE_PWM);
        }
        break;

      default:
        Serial.print("CMD BIN: Comando desconhecido: ");
        Serial.println(pacote.tipo);
        break;
    }
  }
}

// Função chamada IMEDIATAMENTE quando o pino da Rasp sobe (Rising)
void IRAM_ATTR isrAnomalia() {
  anomaliaDetectada = true;
}

void moverRobo(int alvo) {
  velocidadeAlvo = alvo;
  // Nota: Não damos analogWrite aqui! Só definimos a meta.
}

void atualizarRampaMotores() {
  // Se já estamos na velocidade certa, não faz nada
  if ((int)velocidadeAtual == velocidadeAlvo) return;

  // --- Lógica de Aceleração/Desaceleração ---
  if (velocidadeAtual < velocidadeAlvo) {
    // Precisamos acelerar (ou ré para frente)
    if (velocidadeAtual < 0 && velocidadeAlvo > velocidadeAtual) {
       // Estamos indo de ré para parar ou avançar -> É FREIO/REVERSÃO
       velocidadeAtual += PASSO_FREIO;
    } else {
       // Aceleração normal
       velocidadeAtual += PASSO_ACELERACAO;
    }
    
    // Não deixa passar do alvo
    if (velocidadeAtual > velocidadeAlvo) velocidadeAtual = velocidadeAlvo;
  } 
  else if (velocidadeAtual > velocidadeAlvo) {
    // Precisamos desacelerar (ou frente para ré)
    if (velocidadeAtual > 0 && velocidadeAlvo < velocidadeAtual) {
       // Estamos indo de frente para parar ou ré -> É FREIO/REVERSÃO
       velocidadeAtual -= PASSO_FREIO;
    } else {
       // Aceleração em ré (negativo ficando mais negativo)
       velocidadeAtual -= PASSO_ACELERACAO;
    }

    // Não deixa passar do alvo
    if (velocidadeAtual < velocidadeAlvo) velocidadeAtual = velocidadeAlvo;
  }

  // Aplica fisicamente aos motores
  int pwmSaida = (int)velocidadeAtual;

  if (pwmSaida > 0) {
    analogWrite(MOTOR_AVANCO, 0);
    analogWrite(MOTOR_RECUO, pwmSaida);
  } else if (pwmSaida < 0) {
    analogWrite(MOTOR_AVANCO, abs(pwmSaida));
    analogWrite(MOTOR_RECUO, 0);
  } else {
    analogWrite(MOTOR_AVANCO, 0);
    analogWrite(MOTOR_RECUO, 0);
  }
}

void pararTodosMotoresAtuadores() {
  // 1. Para os motores de movimento
  analogWrite(MOTOR_AVANCO, 0);
  analogWrite(MOTOR_RECUO, 0);

  analogWrite(ATUADOR_AVANCO, 0);
  analogWrite(ATUADOR_RECUO, 0);
}

void ativarMacarico(bool ligar) {
  if (ligar) {
    // LIGA = RECUAR para apertar o botão
    Serial.println("Atuador: RECUANDO (Ligando Maçarico)");
    analogWrite(ATUADOR_AVANCO, 0);
    analogWrite(ATUADOR_RECUO, 255);  // Recua

    delay(TEMPO_MOVER_MACARICO);
    analogWrite(ATUADOR_RECUO, 0);  // Para o atuador e mantém pressionado
  } else {
    // DESLIGA = AVANÇAR para soltar o botão
    Serial.println("Atuador: AVANÇANDO (Desligando Maçarico)");
    analogWrite(ATUADOR_RECUO, 0);
    analogWrite(ATUADOR_AVANCO, 255);
    // Tempo necessário para o atuador soltar o botão
    delay(TEMPO_MOVER_MACARICO);
    analogWrite(ATUADOR_AVANCO, 0);  // Para o atuador
  }
}

//ROTINA DE QUEIMA (NÃO-BLOQUEANTE)
void gerenciarRotinaDeQueima() {
  unsigned long tempoDecorrido = millis() - tempoInicioRotina;
  Serial.print("\r");
  Serial.print(tempoDecorrido / 1000);
  switch (estadoQueimaAtual) {

    case OCIOSO:
      break;  // Não faz nada. Espera o gatilho da RPi no loop().

    case INICIO_RECUO:  // Fase 1: Avança por X segundos
      moverRobo(-V_VELOCIDADE_PWM);
      if (tempoDecorrido >= X_DURACAO_MS) {
        tempoInicioRotina = millis();  // Reseta o cronômetro para a próxima fase
        estadoQueimaAtual = AVANCO_ALVO;
        Serial.println("Rotina Queima: FASE 2 -> AVANCO (2X).");
      }
      break;

    case AVANCO_ALVO:  // Fase 2: Recua por 2X segundos
      moverRobo(V_VELOCIDADE_PWM);
      // O tempo total desta fase é 2 * X_DURACAO_MS
      if (tempoDecorrido >= (2 * X_DURACAO_MS)) {
        tempoInicioRotina = millis();  // Reseta o cronômetro para a próxima fase
        estadoQueimaAtual = RECUO_RETORNO;
        Serial.println("Rotina Queima: FASE 3 -> RECUO (X) Retorno.");
      }
      break;

    case RECUO_RETORNO:  // Fase 3: Avança por X segundos para voltar à posição inicial
      moverRobo(-V_VELOCIDADE_PWM);

      if (tempoDecorrido >= X_DURACAO_MS) {
        // Rotina de queima finalizada!
        moverRobo(0);  // Para o robô

        contadorCiclos++;
        if (contadorCiclos < NUM_CICLOS_QUEIMA) {  // Se ainda não atingiu o número desejado, repete
          tempoInicioRotina = millis();
          estadoQueimaAtual = INICIO_RECUO;
          Serial.println("Ciclo novo.");
        } else {

          ativarMacarico(false);  // Desliga o maçarico (com delay interno)

          estadoQueimaAtual = OCIOSO;  // Volta ao estado inicial
          Serial.println("Rotina Queima: FINALIZADA. Robô Parado.");
        }
      }
      break;
  }
}

void OnRxDone(uint8_t *payload, uint16_t size, int16_t rssi, int8_t snr);

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(10);
  Mcu.begin();

  rssi = 0;

  RadioEvents.RxDone = OnRxDone;
  Radio.Init(&RadioEvents);
  Radio.SetChannel(RF_FREQUENCY);
  Radio.SetRxConfig(MODEM_LORA, LORA_BANDWIDTH, LORA_SPREADING_FACTOR,
                    LORA_CODINGRATE, 0, LORA_PREAMBLE_LENGTH,
                    LORA_SYMBOL_TIMEOUT, LORA_FIX_LENGTH_PAYLOAD_ON,
                    0, true, 0, 0, LORA_IQ_INVERSION_ON, true);

  pinMode(ATUADOR_AVANCO, OUTPUT);
  pinMode(ATUADOR_RECUO, OUTPUT);
  pinMode(MOTOR_AVANCO, OUTPUT);
  pinMode(MOTOR_RECUO, OUTPUT);

  // Configura pinos da RPi como entrada
  // Assumindo que a RPi envia HIGH (3.3V) para ativar e LOW (0V) em repouso.
  pinMode(RPI_PIN_ANOMALIA, INPUT_PULLDOWN);
  pinMode(RPI_PIN_PAUSA, INPUT_PULLDOWN);

  // Configura a interrupção: 
  // RISING = dispara quando o sinal vai de LOW para HIGH
  attachInterrupt(digitalPinToInterrupt(RPI_PIN_ANOMALIA), isrAnomalia, RISING);

  pararTodosMotoresAtuadores();  // Garante que tudo começa desligado

  // Inicializa o timer do watchdog
  ultimoPacoteRecebido = millis();
}

void loop() {
  // --- 1. Leitura de Comandos via USB (Raspberry Pi) ---
  if (Serial.available() > 0) {
    lerSerialBinario();
  }
  // 1. Processa comandos de rádio (LoRa)
  if (lora_idle) {
    lora_idle = false;
    Radio.Rx(0);
  }
  Radio.IrqProcess();

  // 2. Lógica de Emergência (tem prioridade MÁXIMA)
  if (emergency_stop) {
    // O comando 'D' em OnRxDone já acionou o atuador e parou os motores.
    // Apenas garantimos que o estado de queima seja resetado.
    estadoQueimaAtual = OCIOSO;

    // 2. Lógica Anti-Spam: Só printa se passou 1 segundo (1000ms)
    static unsigned long timerEmerg = 0; // 'static' memoriza o valor entre loops
        
      if (millis() - timerEmerg > 1000) {
          timerEmerg = millis(); // Atualiza o relógio
          
        // Agora sim, pode printar sem travar
        Serial.println("Parado por botão de emergência!");
      }
    
    return;  // Não faz mais nada até a emergência ser liberada (comando 'C')
  }

  //  --- Verificação do Watchdog (Failsafe) ---
  // Só aplica o watchdog se estivermos em modo MANUAL
  if (modoAtual == MANUAL && !emergency_stop) {
    // Verifica se o tempo desde o último pacote passou do limite
    if (millis() - ultimoPacoteRecebido > WATCHDOG_TIMEOUT) {
      // CONEXÃO PERDIDA!
      Serial.println("FALHA DE SINAL (Watchdog)! Parando motores.");
      pararTodosMotoresAtuadores();

      // Reseta o timer para não ficar spammando o Serial
      // A parada já foi executada.
      ultimoPacoteRecebido = millis();
    }
  }

  // Atualiza os motores suavemente (não bloqueante)
  atualizarRampaMotores();
  // 3. Lógica de Operação principal
  if (modoAtual == SEMI_AUTOMATICO) {
    // --- LÓGICA SEMI-AUTOMÁTICA (Controlada pela RPi) ---

    // Se está no meio de um ciclo de queima, apenas espera o tempo acabar
    if (estadoQueimaAtual != OCIOSO) {
      gerenciarRotinaDeQueima();
      return;
    } else {
      // Se não está queimando

      if (anomaliaDetectada) {
        anomaliaDetectada = false; // Reseta a flag imediatamente
        // ANOMALIA DETECTADA!
        Serial.println("\nAnomalia detectada! Iniciando ciclo de queima.");

        // COMPENSAÇÃO DE ATRASO
        // Para o movimento de inspeção imediatamente
        moverRobo(0);

        // Recua o robô pelo tempo exato do delay da Rasp para centralizar na anomalia
        // Se o robô avança com V_VELOCIDADE_PWM, recuamos com a mesma intensidade
        moverRobo(-V_VELOCIDADE_PWM); 
        delay(COMPENSACAO_ATRASO_MS); 
        moverRobo(0);

      
        ativarMacarico(true);  // Liga o maçarico

        // 3. Inicializa a Máquina de Estados de Queima
        tempoInicioRotina = millis();
        estadoQueimaAtual = INICIO_RECUO;
        Serial.println("Rotina Queima: FASE 1 -> RECUO (X).");
      } else {
        // Lógica de Pausa e Inspeção
      
        bool sinalPausa = (digitalRead(RPI_PIN_PAUSA) == LOW);
      
        if (sinalPausa) {
          // RASP MANDOU PARAR (sem anomalia)
        
          pararTodosMotoresAtuadores();

          // 2. Lógica Anti-Spam: Só printa se passou 1 segundo (1000ms)
          static unsigned long timerPausa = 0; // 'static' memoriza o valor entre loops

          if (millis() - timerPausa > 1000) {
            timerPausa = millis(); // Atualiza o relógio
            
            Serial.println("Pausado");
          }
        } else {
          // NENHUM SINAL (anomalia=LOW, pausa=LOW)
          // Modo de inspeção: movimento constante
          
          //moverRobo(VELOCIDADE_INSPECAO);
          moverRobo(V_VELOCIDADE_PWM);

          // 2. Lógica Anti-Spam: Só printa se passou 1 segundo (1000ms)
          static unsigned long timerInspecao = 0; // 'static' memoriza o valor entre loops
          
          if (millis() - timerInspecao > 1000) {
            timerInspecao = millis(); // Atualiza o relógio
            
            // Agora sim, pode printar sem travar
            Serial.print("Movendo (Inspeção): ");
            Serial.println(V_VELOCIDADE_PWM);
          }
        }
      }
    }
  // else (modoAtual == MANUAL)
  // Em modo manual, nenhuma ação é necessária no loop.
  // A função OnRxDone() trata os comandos de PWM diretamente.
  }
}

void OnRxDone(uint8_t *payload, uint16_t size, int16_t rssi, int8_t snr) {
  // NOVO: Reseta o timer do watchdog! Recebemos um pacote.
  ultimoPacoteRecebido = millis();

  rssi = rssi;
  rxSize = size;
  memcpy(rxpacket, payload, size);
  rxpacket[size] = '\0';
  Radio.Sleep();

  // INTERPRETAR AS MENSAGENS
  if (strcmp(rxpacket, "A") == 0) {
    // Ativa modo Semi-Automático
    modoAtual = SEMI_AUTOMATICO;
    pararTodosMotoresAtuadores();  // Para tudo ao trocar de modo
    Serial.println("Modo: SEMI-AUTOMATICO");
  } else if (strcmp(rxpacket, "B") == 0) {
    // Ativa modo Manual
    modoAtual = MANUAL;
    pararTodosMotoresAtuadores();  // Para tudo ao trocar de modo
    Serial.println("Modo: MANUAL");
  } else if (strcmp(rxpacket, "C") == 0) {
    emergency_stop = false;
    pararTodosMotoresAtuadores();  // Libera emergência e para tudo
    Serial.println("EMERGENCIA LIBERADA");
  } else if (strcmp(rxpacket, "D") == 0) {
    emergency_stop = true;

    analogWrite(ATUADOR_AVANCO, 255);
    analogWrite(ATUADOR_RECUO, 0);
    analogWrite(MOTOR_RECUO, 0);
    analogWrite(MOTOR_AVANCO, 0);
    delay(1000);
    analogWrite(ATUADOR_AVANCO, 0);

  }
  // Só processa PWM se NÃO estiver em emergência E estiver em modo MANUAL
  else if (emergency_stop == false && modoAtual == MANUAL) {
    String rxString = String(rxpacket);  // Converte rxpacket para String
    PWM = rxString.toInt();              // Usa toInt para converter para inteiro

    if (PWM == 0) {
      pararTodosMotoresAtuadores();
    } else if (PWM < -256 || PWM > 256) {
      pwm = PWM / 1000;
      if (PWM > 0) {
        Serial.println("ATUADOR");
        analogWrite(ATUADOR_AVANCO, pwm);
        analogWrite(ATUADOR_RECUO, 0);
      } else if (PWM < 0) {
        analogWrite(ATUADOR_AVANCO, 0);
        analogWrite(ATUADOR_RECUO, abs(pwm));
      }
    } else if (PWM > -256 && PWM < 256) {
      Serial.println("MOTORES");
      if (PWM > 0) {
        analogWrite(MOTOR_AVANCO, PWM);
        analogWrite(MOTOR_RECUO, 0);
      } else if (PWM < 0) {
        analogWrite(MOTOR_AVANCO, 0);
        analogWrite(MOTOR_RECUO, abs(PWM));
      }
    }
  }

  if (modoAtual == MANUAL) {
    Serial.print("PWM recebido: ");
    Serial.println(PWM);
  }

  //Serial.println(rxpacket);

  lora_idle = true;
}
