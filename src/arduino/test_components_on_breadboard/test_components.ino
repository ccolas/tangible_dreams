// Pin definitions
const int ledPin = 2;
const int switchPin = 9;
const int analogPin1 = A0;
const int analogPin2 = A7;

void setup() {
  pinMode(ledPin, OUTPUT);
  pinMode(switchPin, INPUT_PULLUP);
  
  digitalWrite(ledPin, HIGH); // Turn on LED

  Serial.begin(9600);
}

void loop() {
  int valA0 = analogRead(analogPin1);
  int valA7 = analogRead(analogPin2);
  int switchState = digitalRead(switchPin);  // LOW = pressed, HIGH = released

  // Print values in CSV format for easy plotting
  Serial.print("jack: ");
  Serial.print(valA0);
  Serial.print(", pot: ");
  Serial.print(valA7);
  Serial.print(", switch: ");
  Serial.println(switchState);

  delay(100); // Adjust refresh rate as needed
}
