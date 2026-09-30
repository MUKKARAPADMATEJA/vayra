const int sensorPin = 34; // ESP32 Analog Pin
int sensorValue = 0;

void setup() {
  Serial.begin(115200);
  pinMode(sensorPin, INPUT);
  Serial.println("Vayra Sensor Node Initialized.");
}

void loop() {
  sensorValue = analogRead(sensorPin);
  float waterLevelCm = (sensorValue / 4095.0) * 20.0; 
  
  Serial.print("Water Level (cm): ");
  Serial.println(waterLevelCm);
  
  delay(1000);
}
