/*
  face-tacho / esp8266_servo_pan.ino
  Wemos D1 mini + ULN2003 + 28BYJ-48 tachometer needle

  Wiring (ULN2003 <- Wemos):
    IN1 -> D1 (GPIO5)
    IN2 -> D2 (GPIO4)
    IN3 -> D5 (GPIO14)
    IN4 -> D6 (GPIO12)
    GND -> G
    +   -> 5V (NOT 3.3V)

  MQTT topic: face/status
    Receives angle as ASCII string, e.g. "90"
    Range  : 0  = full LEFT  (-90 dial)
             90 = CENTER     (needle pointing straight up)
             180= full RIGHT (+90 dial)

  Behaviour:
    - Power-up : needle drives to CENTER (90 deg) and waits.
    - Tracking : Python publishes angle 0-180 based on face position.
                 Needle follows in real time.
    - Lost face: Python stops publishing → needle FREEZES at last position.

  Libraries (Arduino Library Manager):
    - PubSubClient by Nick O'Leary
    - AccelStepper by Mike McCauley
  Board: ESP8266 -> LOLIN(WEMOS) D1 R2 & mini  (upload speed 115200)
*/

#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <AccelStepper.h>


const char* WIFI_SSID      = "Main Hall";
const char* WIFI_PASS      = "Meeting@2024";
const char* MQTT_BROKER    = "10.12.73.141";   
const int   MQTT_PORT      = 1883;
const char* MQTT_CLIENT_ID = "esp8266-tacho-01";
// ============================================================

const char* TOPIC_STATUS = "face/status";   // receives angle 0-180
const char* TOPIC_TACHO  = "tacho/state";   // LWT / online heartbeat

// ---- Stepper pins (AccelStepper HALF4WIRE order: IN1,IN3,IN2,IN4) ----
#define PIN_IN1  5   // D1 → GPIO5
#define PIN_IN3 14   // D5 → GPIO14
#define PIN_IN2  4   // D2 → GPIO4
#define PIN_IN4 12   // D6 → GPIO12

AccelStepper stepper(AccelStepper::HALF4WIRE, PIN_IN1, PIN_IN3, PIN_IN2, PIN_IN4);


const long STEPS_PER_REV  = 4096;
const int  ANGLE_MIN      = 0;
const int  ANGLE_MAX      = 180;
const int  ANGLE_CENTER   = 90;

// Convert angle (0-180) to stepper steps
long angleToSteps(int angle) {
  angle = constrain(angle, ANGLE_MIN, ANGLE_MAX);
  return (long)(angle * STEPS_PER_REV / 360.0f);
}

// ---- Motion tuning ----
const float MAX_SPEED = 800.0;   // half-steps/sec  (raise if needle is slow)
const float ACCEL     = 400.0;   // half-steps/sec² (lower if it skips)

// ---- MQTT / WiFi ----
WiFiClient   wifiClient;
PubSubClient mqtt(wifiClient);

unsigned long lastWifiAttempt = 0;
unsigned long lastMqttAttempt = 0;
const unsigned long RECONNECT_MS = 3000;

// ---------------------------------------------------------------
void moveToAngle(int angle) {
  long steps = angleToSteps(angle);
  Serial.print("[stepper] angle=");
  Serial.print(angle);
  Serial.print("  steps=");
  Serial.println(steps);
  stepper.moveTo(steps);
}

// ---------------------------------------------------------------
void onMqttMessage(char* topic, byte* payload, unsigned int length) {
  if (strcmp(topic, TOPIC_STATUS) != 0) return;

  // Parse angle string (e.g. "90", "0", "180")
  char msg[8] = {0};
  unsigned int n = min(length, (unsigned int)(sizeof(msg) - 1));
  memcpy(msg, payload, n);
  msg[n] = '\0';

  Serial.print("[mqtt] face/status -> \"");
  Serial.print(msg);
  Serial.println("\"");

  int angle = atoi(msg);
  moveToAngle(angle);
}

// ---------------------------------------------------------------
void ensureWifi() {
  if (WiFi.status() == WL_CONNECTED) return;

  unsigned long now = millis();
  if (now - lastWifiAttempt < RECONNECT_MS) return;
  lastWifiAttempt = now;

  Serial.print("[wifi] connecting to ");
  Serial.println(WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
}

// ---------------------------------------------------------------
void ensureMqtt() {
  if (WiFi.status() != WL_CONNECTED) return;
  if (mqtt.connected()) return;

  unsigned long now = millis();
  if (now - lastMqttAttempt < RECONNECT_MS) return;
  lastMqttAttempt = now;

  Serial.print("[mqtt] connecting ");
  Serial.print(MQTT_BROKER);
  Serial.print(":");
  Serial.print(MQTT_PORT);
  Serial.print(" ... ");

  if (mqtt.connect(MQTT_CLIENT_ID, TOPIC_TACHO, 1, true, "offline")) {
    Serial.println("ok");
    mqtt.publish(TOPIC_TACHO, "online", true);
    mqtt.subscribe(TOPIC_STATUS);
    Serial.println("[mqtt] subscribed face/status");
  } else {
    Serial.print("fail rc=");
    Serial.println(mqtt.state());
  }
}

// ---------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println();
  Serial.println("=== face-tacho ESP v2 ===");
  Serial.print("CENTER steps = ");
  Serial.println(angleToSteps(ANGLE_CENTER));

  stepper.setMaxSpeed(MAX_SPEED);
  stepper.setAcceleration(ACCEL);

  // Assume needle is physically at LEFT (0 deg) at power-up.
  // Drive immediately to CENTER and wait for face lock.
  stepper.setCurrentPosition(0);
  moveToAngle(ANGLE_CENTER);   // → 90 deg = center
  Serial.println("[stepper] driving to CENTER (90 deg) ...");

  mqtt.setServer(MQTT_BROKER, MQTT_PORT);
  mqtt.setCallback(onMqttMessage);
  mqtt.setKeepAlive(30);

  ensureWifi();
}

// ---------------------------------------------------------------
void loop() {
  // Keep calling stepper.run() every loop — never block here.
  ensureWifi();
  ensureMqtt();

  if (mqtt.connected()) {
    mqtt.loop();
  }

  stepper.run();
}
