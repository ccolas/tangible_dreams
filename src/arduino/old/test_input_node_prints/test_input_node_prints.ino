#define NUM_ACTIV 8
#define ACTIV_KNOB A6
#define ON_OFF_SWITCH 9
#define CV_SWITCH 10
#define ON_OFF_LED 11
#define CV_LED 12

#define NUM_ANALOG 8
#define NUM_DIGITAL 2
#define NUM_TOTAL (NUM_ANALOG + NUM_DIGITAL)

#define ANALOG_THRESHOLD 5

const char* valueNames[NUM_TOTAL] = {
  "input 1", "input 2", "input 3",
  "weight 1", "weight 2", "weight 3",
  "activation", "scale",
  "on/off", "cv"
};

uint16_t thresholdNodeIds[] = {45, 108, 144, 191, 260, 317, 364, 423, 492, 566, 630, 681, 719, 759, 799}; // threshold to recognize nodeIds from voltage dividers
#define NUM_THRESHOLDS (sizeof(thresholdNodeIds) / sizeof(thresholdNodeIds[0]))

uint16_t lastValues[NUM_TOTAL];
uint16_t newValues[NUM_TOTAL];

uint8_t cvState = 0;
bool cvLast = false;

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

void setup() {
  Serial.begin(115200);
  for (uint8_t i = 0; i < NUM_ANALOG; i++) pinMode(A0 + i, INPUT);
  pinMode(ON_OFF_SWITCH, INPUT_PULLUP);
  pinMode(CV_SWITCH, INPUT_PULLUP);
  pinMode(ON_OFF_LED, OUTPUT);
  pinMode(CV_LED, OUTPUT);
  for (uint8_t i = 0; i < NUM_TOTAL; i++) lastValues[i] = 0xFFFF;
}

void loop() {
  // Read analogs
  newValues[0] = mapToCustomBins(analogRead(A0), thresholdNodeIds, NUM_THRESHOLDS);
  newValues[1] = mapToCustomBins(analogRead(A1), thresholdNodeIds, NUM_THRESHOLDS);
  newValues[2] = mapToCustomBins(analogRead(A2), thresholdNodeIds, NUM_THRESHOLDS);
  newValues[3] = analogRead(A3);
  newValues[4] = analogRead(A4);
  newValues[5] = analogRead(A5);
  newValues[6] = mapToChoice(analogRead(ACTIV_KNOB), NUM_ACTIV);
  newValues[7] = analogRead(A7);

  // Read ON_OFF (D9): released = 1, pressed = 0
  newValues[8] = (digitalRead(ON_OFF_SWITCH) == HIGH) ? 1 : 0;

  // CV toggle logic
  bool cvNow = (digitalRead(CV_SWITCH) == LOW);
  if (cvNow && !cvLast) cvState ^= 1;
  cvLast = cvNow;
  newValues[9] = cvState;

  // Change detection & print
  for (uint8_t i = 0; i < NUM_TOTAL; i++) {
    bool changed = false;
    if (i <= 2 || i == 6 || i >= 8) {  // binning/discrete: exact compare
      changed = (lastValues[i] != newValues[i]);
    } else {  // raw analog with threshold
      changed = (lastValues[i] == 0xFFFF || abs((int)newValues[i] - (int)lastValues[i]) > ANALOG_THRESHOLD);
    }
    if (changed) {
      lastValues[i] = newValues[i];
      Serial.print("new ");
      Serial.print(valueNames[i]);
      Serial.print(": ");
      Serial.println(newValues[i]);
    }
  }

  // Update LEDs
  digitalWrite(ON_OFF_LED, newValues[8] ? HIGH : LOW);
  digitalWrite(CV_LED, newValues[9] ? HIGH : LOW);

  delay(50);
}
