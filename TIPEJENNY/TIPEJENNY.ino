String incoming;

void setup() {

  Serial.begin(9600);
  if(Serial.available() > 0){
    incoming = Serial.readStringUntil(":");
    Serial.println(incoming);
  }
  
}

void loop() {
  Serial.println("INCOMING : " + incoming);
  delay(100);
}
