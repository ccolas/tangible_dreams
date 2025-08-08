#include <SoftwareSerial.h>

// RS485 config
#define RS485_TX 2  // DI
#define RS485_DE 3  // DE + RE
#define RS485_RX 4  // RO
#define BAUD 115200
SoftwareSerial bus(RS485_RX, RS485_TX);

// Node config
const uint8_t handledNodes[] = {7}; 
#define NUM_NODES (sizeof(handledNodes) / sizeof(handledNodes[0]))

// Pin mapping
#define ACTIV_KNOB A6  // activation 
#define ON_OFF_SWITCH 9  
#define CV_SWITCH 10
#define ON_OFF_LED 11
#define CV_LED 12

// Constants
#define NUM_ACTIV 8  // num of activaiton functions
#define NUM_ANALOG 8  
#define NUM_DIGITAL 2
#define NUM_TOTAL (NUM_ANALOG + NUM_DIGITAL)
#define ANALOG_THRESHOLD 5  // threshold for change detection in analog value

// Threshold bins for A0-A2
uint16_t thresholdNodeIds[] = {45, 108, 144, 191, 260, 317, 364, 423, 492, 566, 630, 681, 719, 759, 799};
#define NUM_THRESHOLDS (sizeof(thresholdNodeIds) / sizeof(thresholdNodeIds[0]))

// Storage
uint16_t lastValues[NUM_NODES][NUM_TOTAL];
uint16_t newValues[NUM_TOTAL];
bool changed[NUM_TOTAL];
uint8_t cvState = 0;
bool cvLast = false;

// Helpers
uint8_t mapToChoice(uint16_t value, uint8_t numChoices) {
  uint16_t step = 1024 / numChoices;
  uint8_t bin = value / step;
  return (bin >= numChoices) ? numChoices - 1 : bin;
}

uint8_t mapToCustomBins(uint16_t value, const uint16_t* thresholds, uint8_t len) {
  for (uint8_t i = 0; i < len; i++) {
    if (value < thresholds[i]) return i;
  }
  return len;
}

inline void txBegin() {
  digitalWrite(RS485_DE, HIGH);
  delayMicroseconds(100);
}

inline void txEnd() {
  bus.flush();
  digitalWrite(RS485_DE, LOW);
  bus.listen();
}

bool handlesNode(uint8_t id) {
  for (uint8_t i = 0; i < NUM_NODES; i++) {
    if (handledNodes[i] == id) return true;
  }
  return false;
}

void setup() {
  pinMode(RS485_DE, OUTPUT);
  pinMode(ON_OFF_SWITCH, INPUT_PULLUP);
  pinMode(CV_SWITCH, INPUT_PULLUP);
  pinMode(ON_OFF_LED, OUTPUT);
  pinMode(CV_LED, OUTPUT);

  for (uint8_t i = 0; i < NUM_ANALOG; i++) pinMode(A0 + i, INPUT);
  txEnd();
  bus.begin(BAUD);
  delay(100);

  for (uint8_t node = 0; node < NUM_NODES; node++) {
    for (uint8_t pin = 0; pin < NUM_TOTAL; pin++) {
      lastValues[node][pin] = 0xFFFF;
    }
  }
}

void readInputs() {
  newValues[0] = mapToCustomBins(analogRead(A0), thresholdNodeIds, NUM_THRESHOLDS);
  newValues[1] = mapToCustomBins(analogRead(A1), thresholdNodeIds, NUM_THRESHOLDS);
  newValues[2] = mapToCustomBins(analogRead(A2), thresholdNodeIds, NUM_THRESHOLDS);
  newValues[3] = analogRead(A3);
  newValues[4] = analogRead(A4);
  newValues[5] = analogRead(A5);
  newValues[6] = mapToChoice(analogRead(ACTIV_KNOB), NUM_ACTIV);
  newValues[7] = analogRead(A7);

  newValues[8] = (digitalRead(ON_OFF_SWITCH) == HIGH) ? 1 : 0;

  bool cvNow = (digitalRead(CV_SWITCH) == LOW);
  if (cvNow && !cvLast) cvState ^= 1;
  cvLast = cvNow;
  newValues[9] = cvState;

  digitalWrite(ON_OFF_LED, newValues[8] ? HIGH : LOW);
  digitalWrite(CV_LED, newValues[9] ? HIGH : LOW);
}

void loop() {
  if (!bus.available()) return;
  if (bus.available() < 3) return;
  if (bus.read() != 0xCC) return;

  uint8_t nodeId = bus.read();
  uint8_t command = bus.read();

  if (!handlesNode(nodeId)) return;

  readInputs();

  txBegin();
  bus.write(0xAA);
  bus.write(nodeId);

  bool forceSend = (command == 0x01);
  uint8_t changeCount = 0;
  memset(changed, 0, sizeof(changed));
  uint8_t nodeIndex = nodeId - handledNodes[0];

  // Compute which pins changed
  for (uint8_t i = 0; i < NUM_TOTAL; i++) {
    bool diff = false;
    if (i <= 2 || i == 6 || i >= 8) {
      diff = (lastValues[nodeIndex][i] != newValues[i]);
    } else {
      diff = (lastValues[nodeIndex][i] == 0xFFFF || abs((int)newValues[i] - (int)lastValues[nodeIndex][i]) > ANALOG_THRESHOLD);
    }
    if (forceSend || diff) {
      changed[i] = true;
      changeCount++;
      lastValues[nodeIndex][i] = newValues[i];
    }
  }

  // Propagation logic
  // If CV changes, resend new values of input and weight 1
  if (changed[9]) {
    if (!changed[0]) { changed[0] = true; changeCount++; lastValues[nodeIndex][0] = newValues[0]; }
    if (!changed[3]) { changed[3] = true; changeCount++; lastValues[nodeIndex][3] = newValues[3]; }
  }
  // If input changes, resend the weight value
  if (changed[0] && !changed[3]) { changed[3] = true; changeCount++; lastValues[nodeIndex][3] = newValues[3]; }
  if (changed[1] && !changed[4]) { changed[4] = true; changeCount++; lastValues[nodeIndex][4] = newValues[4]; }
  if (changed[2] && !changed[5]) { changed[5] = true; changeCount++; lastValues[nodeIndex][5] = newValues[5]; }

  if (changeCount > 0) {
    bus.write(0xFF);
    bus.write(changeCount);
    for (uint8_t i = 0; i < NUM_TOTAL; i++) {
      if (!changed[i]) continue;
      bus.write(i);
      if (i < NUM_ANALOG) {
        bus.write(newValues[i] & 0xFF);
        bus.write((newValues[i] >> 8) & 0xFF);
      } else {
        bus.write(newValues[i]);
      }
    }
  } else {
    bus.write(0x01);
  }

  bus.write(0xBB);
  txEnd();
}
