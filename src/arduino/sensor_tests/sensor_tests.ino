const int analogPins[] = {A0, A1, A2, A3, A4, A5, A6, A7};
const int digitalPins[] = {8, 9};

// Define pins to disable (use analog and digital pin numbers directly)
const int disabledPins[] = {A0, A2, A3, A4, A5, 8, 9};  // example: disable A2 and D9

bool isDisabled(int pin) {
  for (int i = 0; i < sizeof(disabledPins) / sizeof(disabledPins[0]); i++) {
    if (pin == disabledPins[i]) return true;
  }
  return false;
}

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < sizeof(digitalPins) / sizeof(digitalPins[0]); i++) {
    pinMode(digitalPins[i], INPUT);
  }
}

void loop() {
  Serial.print("A: ");
  for (int i = 0; i < 8; i++) {
    int pin = analogPins[i];
    int val = isDisabled(pin) ? 512 : analogRead(pin);
    Serial.print(val);
    Serial.print(i < 7 ? ", " : " | ");
  }

  Serial.print("D: ");
  for (int i = 0; i < 2; i++) {
    int pin = digitalPins[i];
    int val = isDisabled(pin) ? 0 : digitalRead(pin);
    Serial.print("D");
    Serial.print(pin);
    Serial.print(": ");
    Serial.print(val);
    if (i < 1) Serial.print(", ");
  }

  Serial.println();
  delay(1000);
}
