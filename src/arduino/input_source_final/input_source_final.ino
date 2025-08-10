#include <SoftwareSerial.h>

// -------- RS485 config
#define RS485_TX 2   // DI
#define RS485_DE 3   // DE+RE (tied together)
#define RS485_RX 4   // RO
#define BAUD 115200
SoftwareSerial bus(RS485_RX, RS485_TX);

// -------- Node id
#define NODE_ID 3

// -------- Pin mapping (hardware)
#define SOURCE_KNOB   A3   // maps to protocol pinIndex 3 (activation choice)
#define AIN6          A6   // maps to pinIndex 6
#define AIN7          A7   // maps to pinIndex 7
#define ON_OFF_SWITCH 10   // maps to pinIndex 8 (active-low toggle)
#define ON_OFF_LED    12

// -------- Constants
#define NUM_SOURCES 6        // number of activation functions (bins)
#define ANALOG_THRESHOLD 5   // change threshold for real analogs (A6/A7)

// -------- Only these protocol pin indices are active
const uint8_t ACTIVE_PINS[] = {3, 6, 7, 8};   // keep protocol indices
const uint8_t ACTIVE_COUNT = sizeof(ACTIVE_PINS) / sizeof(ACTIVE_PINS[0]);

// Compact state arrays (one node only)
uint16_t lastValues[ACTIVE_COUNT];
uint16_t newValues[ACTIVE_COUNT];
bool     changed[ACTIVE_COUNT];

// -------- Helpers
static inline void txBegin() { digitalWrite(RS485_DE, HIGH); delayMicroseconds(100); }
static inline void txEnd()   { bus.flush(); digitalWrite(RS485_DE, LOW); bus.listen(); }

uint8_t mapToChoice(uint16_t value, uint8_t numChoices) {
  uint16_t step = 1024 / numChoices;
  uint8_t bin = value / step;
  return (bin >= numChoices) ? (numChoices - 1) : bin;
}

// find compact-array index for a given protocol pinIndex (3,6,7,8)
int idxForPin(uint8_t pinIndex) {
  for (uint8_t i = 0; i < ACTIVE_COUNT; i++) if (ACTIVE_PINS[i] == pinIndex) return i;
  return -1;
}

static uint32_t lastSample = 0; // for on/off led update

void setup() {
  pinMode(RS485_DE, OUTPUT);
  pinMode(ON_OFF_SWITCH, INPUT_PULLUP);  // active-low switch
  pinMode(ON_OFF_LED, OUTPUT);

  // A6/A7 are analog-only; no pinMode needed. A3 read as analog.
  txEnd();
  bus.begin(BAUD);
  delay(100);

  // init lastValues to "unset"
  for (uint8_t i = 0; i < ACTIVE_COUNT; i++) lastValues[i] = 0xFFFF;
}

void readInputs() {
  analogRead(SOURCE_KNOB);
  delayMicroseconds(10);            // try 10 µs; raise if needed

  // pinIndex 3: SOURCE_KNOB (A3) mapped to discrete choice
  newValues[idxForPin(3)] = mapToChoice(analogRead(SOURCE_KNOB), NUM_SOURCES);

  // pinIndex 6: A6 raw analog
  newValues[idxForPin(6)] = analogRead(AIN6);

  // pinIndex 7: A7 raw analog
  newValues[idxForPin(7)] = analogRead(AIN7);

  // pinIndex 8: ON_OFF_SWITCH active-low -> 1 when ON (LOW), 0 when OFF (HIGH)
  uint16_t onOff = (digitalRead(ON_OFF_SWITCH) == LOW) ? 1 : 0;
  newValues[idxForPin(8)] = onOff;

}

void loop() {
  // activate on/off led
  if (millis() - lastSample >= 5) {
    lastSample = millis();
    uint8_t onOffNow = (digitalRead(ON_OFF_SWITCH) == LOW) ? HIGH : LOW;
    digitalWrite(ON_OFF_LED, onOffNow);
  }

  // Expect: 0xCC, nodeId, command
  if (bus.available() < 3) return;
  if (bus.read() != 0xCC)  return;

  uint8_t nodeId  = bus.read();
  uint8_t command = bus.read();

  if (nodeId != NODE_ID) return;  // only answer to node 7

  readInputs();

  bool forceSend = (command == 0x01);
  uint8_t changeCount = 0;
  memset(changed, 0, sizeof(changed));

  // Decide what changed
  for (uint8_t i = 0; i < ACTIVE_COUNT; i++) {
    uint8_t pinIndex = ACTIVE_PINS[i];
    bool diff;

    // Treat A6/A7 as analog-thresholded; A3(choice) and D10(on/off) as exact
    if (pinIndex == 6 || pinIndex == 7) {
      diff = (lastValues[i] == 0xFFFF) ||
             (abs((int)newValues[i] - (int)lastValues[i]) > ANALOG_THRESHOLD);
    } else {
      diff = (lastValues[i] != newValues[i]) || (lastValues[i] == 0xFFFF);
    }

    if (forceSend || diff) {
      changed[i] = true;
      changeCount++;
      lastValues[i] = newValues[i];
    }
  }

  // ---- TX response
  txBegin();
  bus.write(0xAA);
  bus.write(nodeId);

  if (changeCount > 0) {
    bus.write(0xFF);
    bus.write(changeCount);

    for (uint8_t i = 0; i < ACTIVE_COUNT; i++) {
      if (!changed[i]) continue;
      uint8_t pinIndex = ACTIVE_PINS[i];
      bus.write(pinIndex);
      if (pinIndex < 8) {  // analog -> 2 bytes little-endian
        bus.write(newValues[i] & 0xFF);
        bus.write((newValues[i] >> 8) & 0xFF);
      } else {             // digital -> 1 byte
        bus.write((uint8_t)newValues[i]);
      }
    }
  } else {
    bus.write(0x01); // no changes
  }

  bus.write(0xBB);
  txEnd();
}
