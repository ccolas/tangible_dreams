// Pin definitions
const int ledPin = 2;
const int switchPin = 9;

void setup() {
  pinMode(ledPin, OUTPUT);
  pinMode(switchPin, INPUT_PULLUP);
  
  digitalWrite(ledPin, HIGH); // Turn on LED

  Serial.begin(9600);
}

int readStable(uint8_t pin){
  analogRead(pin);                   // throw-away to let S/H settle
  delayMicroseconds(50);
  return analogRead(pin);
}

void loop() {
  int valA0 = readStable(A0);
  int valA1 = readStable(A1);
  int valA2 = readStable(A2);
  int switchState = digitalRead(switchPin);

  Serial.print("pot: ");
  Serial.print(valA0); Serial.print(", ");
  Serial.print(valA1); Serial.print(", ");
  Serial.println(valA2);

  delay(100);
}



