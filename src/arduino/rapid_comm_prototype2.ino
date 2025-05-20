#include <SoftwareSerial.h>

#define RS485_RX 2
#define RS485_TX 3
#define RS485_DE 4
#define BAUD 115200

SoftwareSerial bus(RS485_RX, RS485_TX);
    // while (bus.available()) bus.read();

#define NUM_PINS 10
#define NUM_NODES 18
uint16_t lastValues[NUM_NODES][NUM_PINS];  // One set per node
uint16_t newValues[NUM_PINS];
bool changed[NUM_PINS];
const uint16_t SIMULATED_ANALOG = 1023;
const uint8_t SIMULATED_DIGITAL = 1;
const uint8_t ANALOG_THRESHOLD = 5;

inline void txBegin() { 
    digitalWrite(RS485_DE, HIGH); 
    delayMicroseconds(100); // keep it at 100
}

inline void txEnd() {
    bus.flush();
    digitalWrite(RS485_DE, LOW);
    bus.listen();                     // resume RX
}


void setup() {
    pinMode(RS485_DE, OUTPUT);
    pinMode(8, INPUT);  // Digital Pin 8
    pinMode(9, INPUT);  // Digital Pin 9
    txEnd();
    bus.begin(BAUD);
    // Serial.begin(115200); 
    delay(100);
    // initialize lastValues
    for (uint8_t node = 0; node < NUM_NODES; node++) {
        for (uint8_t pin = 0; pin < NUM_PINS; pin++) {
            lastValues[node][pin] = 0xFFFF;
        }
    }
    

}

void loop() {
    if (!bus.available()) return;
    if (bus.available() < 3) return;
    if (bus.read() != 0xCC) return;
    uint8_t nodeId = bus.read();
    uint8_t command = bus.read();

    if (nodeId < 1 || nodeId > NUM_NODES) return;

    txBegin();

    // Start marker
    bus.write(0xAA);
    bus.write(nodeId);

    if (nodeId >= 6 && nodeId <= 10) {
        bool forceSend = (command == 0x01);
        uint8_t changeCount = 0;
        memset(changed, 0, sizeof(changed));

        for (uint8_t i = 0; i < NUM_PINS; i++) {
            if (i < 5) {
                newValues[i] = analogRead(i);
                if (forceSend || lastValues[nodeId - 1][i] == 0xFFFF ||
                    abs((int)newValues[i] - (int)lastValues[nodeId - 1][i]) > ANALOG_THRESHOLD) {
                    changed[i] = true;
                    changeCount++;
                    lastValues[nodeId - 1][i] = newValues[i];
                }
            } else if (i < 8) {
                newValues[i] = SIMULATED_ANALOG;
                if (forceSend || lastValues[nodeId - 1][i] == 0xFFFF ||
                    abs((int)newValues[i] - (int)lastValues[nodeId - 1][i]) > ANALOG_THRESHOLD) {
                    changed[i] = true;
                    changeCount++;
                    lastValues[nodeId - 1][i] = newValues[i];
                }
            } else {
                newValues[i] = SIMULATED_DIGITAL;
                if (forceSend || lastValues[nodeId - 1][i] == 0xFFFF || newValues[i] != lastValues[nodeId - 1][i]) {
                    changed[i] = true;
                    changeCount++;
                    lastValues[nodeId - 1][i] = newValues[i];
                }
            }
        }

        if (changeCount > 0) {
            bus.write(0xFF);              // Has updates
            bus.write(changeCount);
            for (uint8_t i = 0; i < NUM_PINS; i++) {
                if (!changed[i]) continue;
                bus.write(i);
                if (i < 8) {
                    bus.write(newValues[i] & 0xFF);
                    bus.write((newValues[i] >> 8) & 0xFF);
                } else {
                    bus.write(newValues[i]); // digital: 1 byte
                }
            }
        } else {
            bus.write(0x01); // No changes
        }

    } else {
        bus.write(0x01); // Node not tracked, no changes
    }

    // End marker
    bus.write(0xBB);
    txEnd();
    // delayMicroseconds(100);       // let line settle
    // while (bus.available()) bus.read();  // flush junk
}

