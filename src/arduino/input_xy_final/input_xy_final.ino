#include <SoftwareSerial.h>

// -------- RS485 config
#define RS485_TX 2   // DI
#define RS485_DE 3   // DE+RE (tied together)
#define RS485_RX 4   // RO
#define BAUD 115200
SoftwareSerial bus(RS485_RX, RS485_TX);

// -------- Node id
#define NODE_ID 4

// -------- Pin mapping (hardware)
#define AIN6          A6   // -> proto 6
#define AIN7          A7   // -> proto 7
#define ON_OFF_SWITCH 10   // -> proto 8 (active-low toggle, with LED)
#define ON_OFF_LED    12
#define AUX_SWITCH     9   // NEW: -> proto 9 (active-low toggle, no LED)

// -------- Constants
#define ANALOG_THRESHOLD 5

// -------- Active protocol pin indices
const uint8_t ACTIVE_PINS[] = {6, 7, 8, 9};  // NEW: added 9
const uint8_t ACTIVE_COUNT = sizeof(ACTIVE_PINS) / sizeof(ACTIVE_PINS[0]);

// -------- State
uint16_t lastValues[ACTIVE_COUNT];
uint16_t newValues[ACTIVE_COUNT];
bool     changed[ACTIVE_COUNT];

static uint32_t lastSample = 0; // for on/off LED refresh

// -------- Helpers
static inline void txBegin() { digitalWrite(RS485_DE, HIGH); delayMicroseconds(50); }
static inline void txEnd()   { bus.flush(); digitalWrite(RS485_DE, LOW); bus.listen(); }

uint8_t mapToChoice(uint16_t value, uint8_t numChoices) {
  uint16_t step = 1024 / numChoices;
  uint8_t bin = value / step;
  return (bin >= numChoices) ? (numChoices - 1) : bin;
}

int idxForPin(uint8_t pinIndex) {
  for (uint8_t i = 0; i < ACTIVE_COUNT; i++) if (ACTIVE_PINS[i] == pinIndex) return i;
  return -1;
}

void setup() {
  pinMode(RS485_DE, OUTPUT);
  digitalWrite(RS485_DE, LOW);   // force driver off ASAP
  pinMode(ON_OFF_SWITCH, INPUT_PULLUP);  // active-low
  pinMode(AUX_SWITCH,    INPUT_PULLUP);  // NEW: active-low
  pinMode(ON_OFF_LED, OUTPUT);

  txEnd();
  bus.begin(BAUD);
  delay(100);

  for (uint8_t i = 0; i < ACTIVE_COUNT; i++) lastValues[i] = 0xFFFF;
  ADCSRA = (ADCSRA & 0xF8) | 0x07;  // prescaler 128 (~100μs per read)
}

void readInputs() {
  // settle A3 mux slightly
  analogRead(AIN6);
  delayMicroseconds(10);

  // proto 6/7: raw analogs
  newValues[idxForPin(6)] = analogRead(AIN6);
  newValues[idxForPin(7)] = analogRead(AIN7);

  // proto 8: D10 on/off (active-low -> 1 when pressed/ON)
  newValues[idxForPin(8)] = (digitalRead(ON_OFF_SWITCH) == LOW) ? 1 : 0;

  // proto 9: D9 aux toggle (active-low -> 1 when ON)
  newValues[idxForPin(9)] = (digitalRead(AUX_SWITCH) == LOW) ? 1 : 0;
}

void loop() {
  // background LED reflects D10 even when not polled
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

  if (nodeId != NODE_ID) return;  // only answer to node 1

  readInputs();

  bool forceSend = (command == 0x01);
  uint8_t changeCount = 0;
  memset(changed, 0, sizeof(changed));

  // Decide what changed
  for (uint8_t i = 0; i < ACTIVE_COUNT; i++) {
    uint8_t pinIndex = ACTIVE_PINS[i];
    bool diff;

    // Threshold analogs: proto 6/7; Exact compare: proto 3,8,9
    if (pinIndex == 6 || pinIndex == 7) {
      diff = (lastValues[i] == 0xFFFF) ||
             (abs((int)newValues[i] - (int)lastValues[i]) > ANALOG_THRESHOLD);
    } else {
      diff = (lastValues[i] == 0xFFFF) || (lastValues[i] != newValues[i]);
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
      if (pinIndex < 8) {
        // analog -> 2 bytes little-endian
        bus.write(newValues[i] & 0xFF);
        bus.write((newValues[i] >> 8) & 0xFF);
      } else {
        // digital -> 1 byte (proto 8 = D10, proto 9 = D9)
        bus.write((uint8_t)newValues[i]);
      }
    }
  } else {
    bus.write(0x01); // no changes
  }

  bus.write(0xBB);
  txEnd();
}
