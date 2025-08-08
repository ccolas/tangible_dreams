#include <SoftwareSerial.h>

// --- RS485
#define RS485_TX 2
#define RS485_DE 3   // DE+RE tied
#define RS485_RX 4
#define BAUD 115200
SoftwareSerial bus(RS485_RX, RS485_TX);

// --- Node id
#define NODE_ID 17

// --- Pin mapping (hardware)
#define AIN0          A0   // -> proto 0 (binned)
#define AIN1          A1   // -> proto 1 (binned)
#define AIN2          A2   // -> proto 2 (binned)
#define AIN3          A3   // -> proto 3 (raw analog)
#define AIN4          A4   // -> proto 4 (raw analog)
#define AIN5          A5   // -> proto 5 (raw analog)
#define AIN6          A6   // -> proto 6 (raw analog)
#define AIN7          A7   // -> proto 7 (raw, analog)
#define ON_OFF_SWITCH 10   // -> proto 8 (active-low toggle)
#define ON_OFF_LED    12
#define CV_SWITCH     9    // push button (active-low) toggles CV state
#define CV_LED        11   // -> proto 9 mirrored on LED

// --- Constants
#define ANALOG_THRESHOLD 5

// --- Active protocol pin indices (0..9 all used)
const uint8_t ACTIVE_PINS[] = {0,1,2,3,4,5,6,7,8,9};
const uint8_t ACTIVE_COUNT = sizeof(ACTIVE_PINS)/sizeof(ACTIVE_PINS[0]);

// --- State
uint16_t lastValues[ACTIVE_COUNT];
uint16_t newValues[ACTIVE_COUNT];
bool     changed[ACTIVE_COUNT];

bool cvLast = false;
uint8_t cvState = 0;

// --- Helpers
static inline void txBegin(){ digitalWrite(RS485_DE, HIGH); delayMicroseconds(100); }
static inline void txEnd(){ bus.flush(); digitalWrite(RS485_DE, LOW); bus.listen(); }

// custom thresholds for A0–A2 binning
uint16_t thresholdNodeIds[] = {24, 70, 111, 149, 200, 276, 340, 394, 463, 547, 638, 719, 796, 877};
#define NUM_THRESHOLDS (sizeof(thresholdNodeIds)/sizeof(thresholdNodeIds[0]))

uint8_t mapToCustomBins(uint16_t v, const uint16_t* th, uint8_t len){
  for(uint8_t i=0;i<len;i++) if(v < th[i]) return i;
  return len;
}

int idxForPin(uint8_t p){
  for(uint8_t i=0;i<ACTIVE_COUNT;i++) if(ACTIVE_PINS[i]==p) return i;
  return -1;
}


void setup(){
  pinMode(RS485_DE, OUTPUT);
  pinMode(ON_OFF_SWITCH, INPUT_PULLUP);
  pinMode(CV_SWITCH, INPUT_PULLUP);
  pinMode(ON_OFF_LED, OUTPUT);
  pinMode(CV_LED, OUTPUT);

  txEnd();
  bus.begin(BAUD);
  delay(100);

  for(uint8_t i=0;i<ACTIVE_COUNT;i++) lastValues[i]=0xFFFF;
}

void readInputs(){
  // A0–A2 -> custom bins (proto 0..2)
  // clear buffer first?
  analogRead(AIN0);
  delayMicroseconds(10);            // try 10 µs; raise if needed
  newValues[idxForPin(0)] = mapToCustomBins(analogRead(AIN0), thresholdNodeIds, NUM_THRESHOLDS);
  newValues[idxForPin(1)] = mapToCustomBins(analogRead(AIN1), thresholdNodeIds, NUM_THRESHOLDS);
  newValues[idxForPin(2)] = mapToCustomBins(analogRead(AIN2), thresholdNodeIds, NUM_THRESHOLDS);

  // A3–A7 raw analog (proto 3..7)
  newValues[idxForPin(3)] = analogRead(AIN3);
  newValues[idxForPin(4)] = analogRead(AIN4);
  newValues[idxForPin(5)] = analogRead(AIN5);
  newValues[idxForPin(6)] = analogRead(AIN6);
  newValues[idxForPin(7)] = analogRead(AIN7);

  // D10 on/off active-low -> proto 8 (1=ON when LOW)
  uint16_t onOff = (digitalRead(ON_OFF_SWITCH)==LOW) ? 1 : 0;
  newValues[idxForPin(8)] = onOff;

  // CV push (D9) toggles cvState -> proto 9
  bool cvNow = (digitalRead(CV_SWITCH)==LOW);
  if(cvNow && !cvLast) cvState ^= 1;
  cvLast = cvNow;
  newValues[idxForPin(9)] = cvState;

  // LEDs
  digitalWrite(ON_OFF_LED, onOff ? HIGH : LOW);
  digitalWrite(CV_LED, cvState ? HIGH : LOW);
}

void loop(){
  if(bus.available() < 3) return;
  if(bus.read() != 0xCC) return;

  uint8_t nodeId  = bus.read();
  uint8_t command = bus.read();
  if(nodeId != NODE_ID) return;

  readInputs();

  bool forceSend = (command==0x01);
  uint8_t changeCount=0;
  memset(changed,0,sizeof(changed));

  for(uint8_t i=0;i<ACTIVE_COUNT;i++){
    uint8_t pinIndex = ACTIVE_PINS[i];
    bool diff;

    // Thresholded analogs: A3–A6 (proto 3..6)
    if(pinIndex==3 || pinIndex==4 || pinIndex==5 || pinIndex==6 || pinIndex==7){
      diff = (lastValues[i]==0xFFFF) ||
             (abs((int)newValues[i] - (int)lastValues[i]) > ANALOG_THRESHOLD);
    } else {
      // Exact compare: A0–A2 bins, A7 choice, D10 on/off, CV state (0/1)
      diff = (lastValues[i]==0xFFFF) || (lastValues[i]!=newValues[i]);
    }

    if(forceSend || diff){
      changed[i]=true;
      changeCount++;
      lastValues[i]=newValues[i];
    }
  }

  txBegin();
  bus.write(0xAA);
  bus.write(nodeId);

  if(changeCount>0){
    bus.write(0xFF);
    bus.write(changeCount);
    for(uint8_t i=0;i<ACTIVE_COUNT;i++){
      if(!changed[i]) continue;
      uint8_t pinIndex = ACTIVE_PINS[i];
      bus.write(pinIndex);
      if(pinIndex < 8){
        bus.write(newValues[i] & 0xFF);
        bus.write((newValues[i] >> 8) & 0xFF);
      } else {
        bus.write((uint8_t)newValues[i]); // 8:on/off, 9:cv state
      }
    }
  } else {
    bus.write(0x01); // no changes
  }

  bus.write(0xBB);
  txEnd();
}
