// Arduino Nano: A0–A7 analog reads (skip list), switches on D9/D10 active-low,
// LEDs on D11 (D9 state) and D12 (D10 state), print only on switch state change.

const uint8_t LED_D9 = 11;
const uint8_t LED_D10 = 12;

const uint8_t BTN_TOGGLE_ON_PRESS = 9;   // push button, toggles state on press
const uint8_t BTN_DIRECT_STATE    = 10;  // toggle switch, state = reading

// Skip analog pins by index: 0=A0, 1=A1, ... 7=A7
const uint8_t skipPins[] = {};  
const uint8_t skipCount = sizeof(skipPins) / sizeof(skipPins[0]);

const unsigned long DEBOUNCE_MS = 20;

const uint8_t ANALOG_PINS[8] = {A0, A1, A2, A3, A4, A5, A6, A7};

bool d9_state = false;   // latched state for push button
bool d10_state = false;  // mirrors D10 switch reading

bool last_d9_state = false;
bool last_d10_state = false;

int  d9_lastRead = HIGH; // start HIGH because of pullup
unsigned long d9_lastEdgeAt = 0;

bool shouldSkip(uint8_t index) {
  for (uint8_t i = 0; i < skipCount; i++) {
    if (skipPins[i] == index) return true;
  }
  return false;
}

int readStable(uint8_t pin) {
  analogRead(pin);                  // dummy read to charge the S/H cap
  delayMicroseconds(50);            // let it settle (tune 20–100 µs)
  return analogRead(pin);
}


void setup() {
  pinMode(BTN_TOGGLE_ON_PRESS, INPUT_PULLUP); // active-low
  pinMode(BTN_DIRECT_STATE,    INPUT_PULLUP); // active-low

  pinMode(LED_D9, OUTPUT);
  pinMode(LED_D10, OUTPUT);

  Serial.begin(115200);
}

void loop() {
  unsigned long now = millis();

  // --- Handle D9: toggle-on-press (active-low, so press = LOW) ---
  int d9_read = digitalRead(BTN_TOGGLE_ON_PRESS);
  if (d9_read != d9_lastRead) {
    d9_lastEdgeAt = now;
    d9_lastRead = d9_read;
  } else {
    if ((now - d9_lastEdgeAt) >= DEBOUNCE_MS) {
      static int d9_lastStable = HIGH;
      if (d9_read != d9_lastStable) {
        d9_lastStable = d9_read;
        if (d9_read == LOW) { // button pressed
          d9_state = !d9_state;
        }
      }
    }
  }

  // --- Handle D10: direct state from switch (active-low) ---
  d10_state = (digitalRead(BTN_DIRECT_STATE) == LOW);

  // --- Drive LEDs ---
  digitalWrite(LED_D9,  d9_state ? HIGH : LOW);
  digitalWrite(LED_D10, d10_state ? HIGH : LOW);

  // --- Print only when a switch state changes ---
  if (d9_state != last_d9_state || d10_state != last_d10_state) {
    Serial.print("Switch D9 state: "); Serial.print(d9_state ? "ON" : "OFF");
    Serial.print(" | Switch D10 state: "); Serial.println(d10_state ? "ON" : "OFF");
    last_d9_state = d9_state;
    last_d10_state = d10_state;
  }

  // --- Read analogs ---
  for (uint8_t i = 0; i < 8; i++) {
    if (shouldSkip(i)) continue;
    int val = readStable(ANALOG_PINS[i]);  // <— use stable read
    Serial.print("A"); Serial.print(i); Serial.print("="); Serial.print(val);
    Serial.print(" ");
  }
  Serial.println();

  delay(50); // pause between full readings
}
