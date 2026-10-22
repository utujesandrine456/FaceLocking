/*
  face-tacho / esp/tacho.ino
  Wemos D1 mini + ULN2003 + 28BYJ-48 tachometer needle

  Wiring (ULN2003 <- Wemos):
    IN1 -> D1 (GPIO5)
    IN2 -> D2 (GPIO4)
    IN3 -> D5 (GPIO14)
    IN4 -> D6 (GPIO12)
    GND -> G
    +   -> 5V (NOT 3.3V), grounds shared with motor supply

  MQTT:
    face/status  "1" = recognised -> needle HIGH
                 "0" = not recognised -> needle ZERO
    tacho/state  LWT "offline", publish "online" after connect

  Libraries (Arduino Library Manager):
    - PubSubClient by Nick O'Leary
    - AccelStepper by Mike McCauley
    Board: ESP8266 -> LOLIN(WEMOS) D1 R2 & mini
*/

#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <AccelStepper.h>

// ========== FILL THESE IN ==========
const char* WIFI_SSID     = "Main Hall";
const char* WIFI_PASS     = "Meeting@2024";
const char* MQTT_BROKER   = "10.12.73.178";  // PC Wi-Fi IPv4 (ipconfig) — NOT 127.0.0.1, NOT 172.x
const int   MQTT_PORT     = 1883;
const char* MQTT_CLIENT_ID = "esp8266-tacho-01";
// ===================================

const char* TOPIC_STATUS = "face/status";
const char* TOPIC_TACHO  = "tacho/state";

// AccelStepper HALF4WIRE pin order: IN1, IN3, IN2, IN4
// = D1, D5, D2, D6
#define PIN_IN1 D1   // GPIO5
#define PIN_IN3 D5   // GPIO14
#define PIN_IN2 D2   // GPIO4
#define PIN_IN4 D6   // GPIO12

AccelStepper stepper(AccelStepper::HALF4WIRE, PIN_IN1, PIN_IN3, PIN_IN2, PIN_IN4);

// 28BYJ-48: 4096 half-steps per full revolution
const long STEPS_PER_REV = 4096;

/*
  Dial degrees explained (-90 / +180 / -90):

  Imagine the tachometer face:
      LEFT (-90°)     CENTER (0°)     RIGHT (+90°)
           \              |              /
            \             |             /

  - Power-up ZERO = needle at LEFT (-90° mark) = "not recognised"
  - HIGH          = rotate +180° to RIGHT (+90° mark) = "recognised"
  - Return        = rotate back -180° to LEFT

  So the physical sweep is 180 degrees (not 360).
  Steps for 180° = 4096 * 180 / 360 = 2048

  If your dial uses a different sweep (e.g. 240°), change SWEEP_DEG below.
*/
const float SWEEP_DEG = 180.0;  // -90 left -> +90 right
const long  HIGH_POS  = (long)(STEPS_PER_REV * SWEEP_DEG / 360.0);  // ~2048

// Motion tuning (raise if too slow, lower if it skips/vibrates)
const float MAX_SPEED = 600.0;
const float ACCEL     = 300.0;

WiFiClient wifiClient;
PubSubClient mqtt(wifiClient);

unsigned long lastWifiAttempt = 0;
unsigned long lastMqttAttempt = 0;
const unsigned long RECONNECT_MS = 3000;

void moveNeedleHigh() {
  Serial.print("Needle -> HIGH (");
  Serial.print(HIGH_POS);
  Serial.println(" steps)");
  stepper.moveTo(HIGH_POS);
}

void moveNeedleZero() {
  Serial.println("Needle -> ZERO (0)");
  stepper.moveTo(0);
}

void onMqttMessage(char* topic, byte* payload, unsigned int length) {
  char msg[8];
  unsigned int n = length < sizeof(msg) - 1 ? length : sizeof(msg) - 1;
  memcpy(msg, payload, n);
  msg[n] = '\0';

  Serial.print("MQTT ");
  Serial.print(topic);
  Serial.print(" -> ");
  Serial.println(msg);

  if (strcmp(topic, TOPIC_STATUS) != 0) {
    return;
  }

  if (msg[0] == '1') {
    moveNeedleHigh();
  } else {
    moveNeedleZero();
  }
}

void ensureWifi() {
  if (WiFi.status() == WL_CONNECTED) {
    return;
  }
  unsigned long now = millis();
  if (now - lastWifiAttempt < RECONNECT_MS) {
    return;
  }
  lastWifiAttempt = now;

  Serial.print("WiFi connecting to ");
  Serial.println(WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
}

void ensureMqtt() {
  if (WiFi.status() != WL_CONNECTED) {
    return;
  }
  if (mqtt.connected()) {
    return;
  }
  unsigned long now = millis();
  if (now - lastMqttAttempt < RECONNECT_MS) {
    return;
  }
  lastMqttAttempt = now;

  Serial.print("MQTT connecting ");
  Serial.print(MQTT_BROKER);
  Serial.print(":");
  Serial.print(MQTT_PORT);
  Serial.print(" ... ");

  // Last Will: if ESP drops, broker publishes offline (retained)
  if (mqtt.connect(MQTT_CLIENT_ID, TOPIC_TACHO, 1, true, "offline")) {
    Serial.println("ok");
    mqtt.publish(TOPIC_TACHO, "online", true);
    mqtt.subscribe(TOPIC_STATUS);
    Serial.print("Subscribed: ");
    Serial.println(TOPIC_STATUS);
  } else {
    Serial.print("fail rc=");
    Serial.println(mqtt.state());
  }
}

void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println();
  Serial.println("=== face-tacho ESP ===");
  Serial.print("HIGH_POS steps for ");
  Serial.print(SWEEP_DEG);
  Serial.print(" deg = ");
  Serial.println(HIGH_POS);
  Serial.println("Assume needle is at ZERO (-90 dial) at power-up.");

  stepper.setMaxSpeed(MAX_SPEED);
  stepper.setAcceleration(ACCEL);
  stepper.setCurrentPosition(0);  // mechanical zero = dial left / not recognised

  mqtt.setServer(MQTT_BROKER, MQTT_PORT);
  mqtt.setCallback(onMqttMessage);
  mqtt.setKeepAlive(30);

  ensureWifi();
}

void loop() {
  // Never block here — stepper.run() must keep being called
  ensureWifi();
  ensureMqtt();

  if (mqtt.connected()) {
    mqtt.loop();
  }

  stepper.run();
}
