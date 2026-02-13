#include "LoRaWan_APP.h"
#include "Arduino.h"
#include <Wire.h>  
#include "HT_SSD1306Wire.h"
#include "images.h"


#define RF_FREQUENCY                                868000000 // Hz

#define TX_OUTPUT_POWER                             5        // dBm

#define LORA_BANDWIDTH                              1         // [0: 125 kHz,
                                                              //  1: 250 kHz,
                                                              //  2: 500 kHz,
                                                              //  3: Reserved]
#define LORA_SPREADING_FACTOR                       7         // [SF7..SF12]
#define LORA_CODINGRATE                             1         // [1: 4/5,
                                                              //  2: 4/6,
                                                              //  3: 4/7,
                                                              //  4: 4/8]
#define LORA_PREAMBLE_LENGTH                        8         // Same for Tx and Rx
#define LORA_SYMBOL_TIMEOUT                         0         // Symbols
#define LORA_FIX_LENGTH_PAYLOAD_ON                  false
#define LORA_IQ_INVERSION_ON                        false


#define RX_TIMEOUT_VALUE                            1000
#define BUFFER_SIZE                                 30 // Define the payload size here

char txpacket[BUFFER_SIZE];
//char rxpacket[BUFFER_SIZE];
char ID[BUFFER_SIZE] = "1";

double dataBuffer[3];

bool lora_idle=true;

static RadioEvents_t RadioEvents;
void OnTxDone( void );
void OnTxTimeout( void );

// Definições dos pinos dos botões
const byte PIN_BUTTON_A = 36;
const byte PIN_BUTTON_B = 34;
const byte PIN_BUTTON_C = 0;
const byte PIN_BUTTON_D = 20;

const byte PIN_ANALOG_X = 5;
const byte PIN_ANALOG_Y = 6;

int ID_Motor = 1;

//Configuração OLED
SSD1306Wire  factory_display(0x3c, 500000, SDA_OLED, SCL_OLED, GEOMETRY_128_64, RST_OLED); // addr , freq , i2c group , resolution , rst

void setup() {
    Serial.begin(115200);
    Mcu.begin();

    // Inicialização do display
    factory_display.init();
    factory_display.clear();
    
    // Desenhar o logotipo no display
    factory_display.drawXbm(0, 0, 128, 53, logo_aneel); // Ajuste o tamanho se necessário
    factory_display.display();
    delay(1000); // Manter o logotipo na tela por 5 segundos antes de prosseguir
    factory_display.clear();
  
    factory_display.drawXbm(0, 0, 128, 53, logo_celesc); // Ajuste o tamanho se necessário
    factory_display.display();
    delay(1000); // Manter o logotipo na tela por 5 segundos antes de prosseguir
    factory_display.clear();
  
    factory_display.drawXbm(0, 0, 128, 53, logo_ufu); // Ajuste o tamanho se necessário
    factory_display.display();
    delay(1000); // Manter o logotipo na tela por 5 segundos antes de prosseguir
    factory_display.clear();
  
    //Declaração dos botões
    pinMode(PIN_BUTTON_A, INPUT);
    digitalWrite(PIN_BUTTON_A, HIGH);
  
    pinMode(PIN_BUTTON_B, INPUT);
    digitalWrite(PIN_BUTTON_B, HIGH);
  
    pinMode(PIN_BUTTON_C, INPUT);
    digitalWrite(PIN_BUTTON_C, HIGH);
  
    pinMode(PIN_BUTTON_D, INPUT);
    digitalWrite(PIN_BUTTON_D, HIGH);

    RadioEvents.TxDone = OnTxDone;
    RadioEvents.TxTimeout = OnTxTimeout;
    
    Radio.Init( &RadioEvents );
    Radio.SetChannel( RF_FREQUENCY );
    Radio.SetTxConfig( MODEM_LORA, TX_OUTPUT_POWER, 0, LORA_BANDWIDTH,
                                   LORA_SPREADING_FACTOR, LORA_CODINGRATE,
                                   LORA_PREAMBLE_LENGTH, LORA_FIX_LENGTH_PAYLOAD_ON,
                                   true, 0, 0, LORA_IQ_INVERSION_ON, 3000 ); 

    factory_display.init();
    factory_display.clear();
    factory_display.setFont(ArialMT_Plain_10);
    factory_display.setTextAlignment(TEXT_ALIGN_LEFT);
   }



void loop()
{
  if(lora_idle == true)
  {
    int buttonStateA = digitalRead(PIN_BUTTON_A);
    int buttonStateB = digitalRead(PIN_BUTTON_B);
    int buttonStateC = digitalRead(PIN_BUTTON_C);
    int buttonStateD = digitalRead(PIN_BUTTON_D);
    int joystickX = analogRead(PIN_ANALOG_X);
    int joystickY = analogRead(PIN_ANALOG_Y);

    double countX = 0.0;
    double countY = 0.0;
    double pwm = 0.0;

    countX = joystickX/4095.0;
    countY = joystickY/4095.0;

    Serial.print("CountX: ");
    Serial.print(countX);
    Serial.println(" ");
    Serial.print("CountY: ");
    Serial.print(countY);
    Serial.println(" ");
    
    if (buttonStateA == LOW) {
        sprintf(txpacket, "A");
        delay(250);
    }
    else if (buttonStateB == LOW) {
        sprintf(txpacket, "B");
        delay(250);
    }
    else if (buttonStateC == LOW) {
        sprintf(txpacket, "C");
        delay(250);
    }
    else if (buttonStateD == LOW) {
        sprintf(txpacket, "D");
        ID_Motor = ID_Motor + 1;
        delay(250);
        if(ID_Motor > 3){
          ID_Motor = 1;
        }
        sprintf(ID, "%d" ,ID_Motor);
    }
    else if ((countX > 0.44 && countX <0.47) && (countY > 0.44 && countY <0.47)) {
        factory_display.drawString(0, 15, "Aguardando...");
        sprintf(txpacket, "0");
    }
    else if (countX > 0.47) {
       factory_display.drawString(0, 15, "Direita");

       pwm = ((countX-0.47)/0.53)*255;
       sprintf(txpacket, "%fl", pwm);
    }
    else if (countX < 0.44) {
        factory_display.drawString(0, 15, "Esquerda");
        
        pwm = (1- countX/0.46)*255* (-1);
        sprintf(txpacket, "%fl", pwm);
    }
    else if (countY > 0.47) {
       factory_display.drawString(0, 15, "Avanço");

       pwm = ((countY-0.47)/0.53)*255*1000;
       sprintf(txpacket, "%fl", pwm);
    }
    else if (countY < 0.44) {
        factory_display.drawString(0, 15, "Recuo");
        
        pwm = (1- countY/0.46)*255*(-1)*1000;
        sprintf(txpacket, "%fl", pwm);
    }
    delay(100);
    
    Serial.printf("\r\nSending packet \"%s\" , length %d\r\n", txpacket, strlen(txpacket));
    factory_display.drawString(0, 30, txpacket);
    factory_display.drawString(0, 45, ID);
    factory_display.display();
    factory_display.clear();
    Radio.Send( (uint8_t *)txpacket, strlen(txpacket) ); //send the package out 

    lora_idle = false;
  }
  Radio.IrqProcess( );
}

void OnTxDone( void )
{
  //Serial.println("TX done......");
  lora_idle = true;
}

void OnTxTimeout( void )
{
    Radio.Sleep( );
    Serial.println("TX Timeout......");
    lora_idle = true;
}
