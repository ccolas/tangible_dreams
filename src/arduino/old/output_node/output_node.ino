#include <SoftwareSerial.h>

#define RS485_RX 2
#define RS485_TX 3
#define RS485_DE 4
#define BAUD 115200

SoftwareSerial bus(RS485_RX, RS485_TX);

#define CV_SWITCH 8 // id of switch for CV override

const uint8_t handledNodes[] = {17, 18, 19}; // handled nodes
const uint8_t disabledPins[] = {0, 2, 3, 4, 5, 8, 9}; // pins to simulate
#define NUM_NODES (sizeof(handledNodes) / sizeof(handledNodes[0]))
#define NUM_PINS 10
uint16_t lastValues[NUM_NODES][NUM_PINS];
uint16_t newValues[NUM_PINS];
bool changed[NUM_PINS];
const bool isDiscrete[NUM_PINS] = {
    true, true, true, false, false, false, false, false, // A0–A7
    true, true                                 // D8–D9
};


const uint16_t SIMULATED_ANALOG = 512;
const uint8_t SIMULATED_DIGITAL = 0;
const uint8_t ANALOG_THRESHOLD = 5;

uint16_t thresholdNodeIds[] = {46, 110, 147, 195, 266, 324, 371, 411, 441, 503, 579, 644, 696, 735, 766, 789}; // threshold to recognize nodeIds from voltage dividers
#define NUM_THRESHOLDS (sizeof(thresholdNodeIds) / sizeof(thresholdNodeIds[0]))



bool handlesNode(uint8_t id) {
    for (uint8_t i = 0; i < sizeof(handledNodes) / sizeof(handledNodes[0]); i++) {
        if (handledNodes[i] == id) return true;
    }
    return false;
}

bool isDisabled(uint8_t pinIndex) {
    for (uint8_t i = 0; i < sizeof(disabledPins) / sizeof(disabledPins[0]); i++) {
        if (disabledPins[i] == pinIndex) return true;
    }
    return false;
}

// maps analog value to discrete choices (eg input type, activation function)
uint8_t mapToChoice(uint16_t value, uint8_t numChoices) {
    uint16_t step = 1024 / numChoices;
    uint8_t bin = value / step;
    return (bin >= numChoices) ? numChoices - 1 : bin;
}

// maps analog value discrete choices given thresholds
uint8_t mapToCustomBins(uint16_t value, const uint16_t* thresholds, uint8_t len) {
    for (uint8_t i = 0; i < len; i++) {
        if (value < thresholds[i]) return i;
    }
    return len;  // value >= last threshold
}


inline void txBegin() {
    digitalWrite(RS485_DE, HIGH);
    delayMicroseconds(100); // keep it at 100
}

inline void txEnd() {
    bus.flush();
    digitalWrite(RS485_DE, LOW);
    bus.listen();
}

void setup() {
    pinMode(RS485_DE, OUTPUT);
    pinMode(8, INPUT);
    pinMode(9, INPUT);
    txEnd();
    bus.begin(BAUD);
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

    if (!handlesNode(nodeId)) return;

    txBegin();

    // Start marker
    bus.write(0xAA);
    bus.write(nodeId);

    bool forceSend = (command == 0x01);
    uint8_t changeCount = 0;
    memset(changed, 0, sizeof(changed));
    uint8_t nodeIndex = nodeId - handledNodes[0];  // index of that node in lastValues array

    for (uint8_t i = 0; i < NUM_PINS; i++) {
        if (isDisabled(i)) {
            newValues[i] = (i < 8) ? SIMULATED_ANALOG : SIMULATED_DIGITAL;
        } else if (i < 8) {
            newValues[i] = analogRead(i);
        } else {
            newValues[i] = digitalRead(i);
        }

        // map input voltages to node ids
        if (i < 3) {
            newValues[i] = mapToCustomBins(newValues[i], thresholdNodeIds, NUM_THRESHOLDS);  // map analog pin 5 to 4 bins
        }

        // send if python asks for it, or when there are changes
        if (forceSend || lastValues[nodeIndex][i] == 0xFFFF ||
            (!isDiscrete[i] && abs((int)newValues[i] - (int)lastValues[nodeIndex][i]) > ANALOG_THRESHOLD) ||
            (isDiscrete[i] && newValues[i] != lastValues[nodeIndex][i])){
            changed[i] = true;
            changeCount++;
            lastValues[nodeIndex][i] = newValues[i];
        }
    }

    // // Propagate changes from new inputs to new switches
    if (changed[0] && !changed[3]) changed[3] = true, changeCount++;
    if (changed[1] && !changed[4]) changed[4] = true, changeCount++;
    if (changed[2] && !changed[5]) changed[5] = true, changeCount++;

    // Propagate change from CV override (D8) to input and weight 1 (A0 and A3)
    if (changed[CV_SWITCH] && !changed[0]) changed[0] = true, changeCount++;
    if (changed[CV_SWITCH] && !changed[3]) changed[3] = true, changeCount++;

    // if any changes, send them, if not send special code
    if (changeCount > 0) {
        bus.write(0xFF);
        bus.write(changeCount);
        for (uint8_t i = 0; i < NUM_PINS; i++) {
            if (!changed[i]) continue;
            bus.write(i);
            if (i < 8) {
                bus.write(newValues[i] & 0xFF);
                bus.write((newValues[i] >> 8) & 0xFF);
            } else {
                bus.write(newValues[i]);
            }
        }
    } else {
        bus.write(0x01);  // no changes
    }

    bus.write(0xBB);
    txEnd();
}
