#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <Servo.h>

//WiFi Passwords
#define wifiSSID ""
#define wifiPass ""

//Change ssid and password when needed
const char* ssid = wifiSSID;
const char* password = wifiPass;

//Raspberry Pi IP and port, IP will vary
const char* MQTT_SERVER = "";
const int MQTT_PORT = 1883;

//MQTT Topics
const char* moveTopic = "vehicle/Move";

int recvPay[8];
char yPayload[3] = "";
char fBPayload[3] = "";
int stateFB[2];

//Pins
#define Forward D2
#define Backward D1
//Servo that doesn't use L298N Board due to PWM restrictions
#define leftRight D8

//Controls the speed of the Vehicle
#define enFB D7

//Used to indicate vehicle is connected to WiFi
#define readyLED D4

//Define magnitudes of the turn
#define Slight 50
#define Moderate 100
#define Significant 255

/*
 * Define set positions for turning and turn threshholds.
 * Resting angle position of servo set at 91.
 * < 91, bank left (no less than 40)
 * > 91, bank right (no greater than 141)
 */
 #define restingPos 91
 #define sLeft 75
 #define mLeft 65
 #define lLeft 40
 #define sRight 107
 #define mRight 117
 #define lRight 141

 #define carCenter 50
 #define sLeftThresh 47
 #define mLeftThresh 43
 #define sRightThresh 53
 #define mRightThresh 57
 
/*
 * uint8_t array used to set pin state via for loop.
 * Reduces delay and lines of code significantly.
 */
static const uint8_t pinNames[] = {Forward, Backward};

//Servo setup
Servo servo;

//Prototyping
void callback(char* topic, byte* payload, unsigned int length);
void moveRC(int payload[16]);
void laneHandler(int targetCenter);
void setX(int magnitude, int fbState[2]);

WiFiClient espClient;
PubSubClient client(espClient);

void setup() {
    Serial.begin(9600);

    //Pin Setup
    pinMode(Forward, OUTPUT);
    pinMode(Backward, OUTPUT);
    pinMode(readyLED, OUTPUT);
    pinMode(enFB, OUTPUT);

    //Servo Setup
    servo.attach(leftRight);
    servo.write(restingPos);

    digitalWrite(Forward, LOW);
    digitalWrite(Backward, LOW);
    digitalWrite(readyLED, LOW);

    //WiFi/MQTT Setup
    WiFi.begin(ssid, password);

    while (WiFi.status() != WL_CONNECTED)
    {
        delay(500);
        Serial.print(".");
    }
    Serial.print("Connected to ");
    Serial.println(ssid);

    client.setServer(MQTT_SERVER, MQTT_PORT);
    client.setCallback(callback);

    while (!client.connected())
    {
        Serial.println("Connecting to MQTT...");

        if (client.connect(moveTopic))
        {
            Serial.print("Connected to ");
            Serial.println(moveTopic);
            digitalWrite(readyLED, HIGH);
        }
        else
        {
            Serial.print("Failed with state ");
            Serial.println(client.state());
            delay(2000);
        }
    }
}


void callback(char* topic, byte* payload, unsigned int length)
{
    Serial.print("Payload from topic: ");
    Serial.println(topic);
    Serial.print("Payload: ");
    char temp[3];
    /*
     * Parses payload into variables to be interpreted.
     */
     int x = 0;
    for (int i = 0; i < 3; i++)
    {
        recvPay[i] = (int) payload[i];
        Serial.print((char)recvPay[i]);
        yPayload[i] = recvPay[i];
    }
    for (int i = 3; i < 6; i++)
    {
        recvPay[i] = (int) payload[i];
        fBPayload[x] = recvPay[i];
        temp[x] = recvPay[i];
        Serial.print((char)fBPayload[x]);
        x++;
    }
    x = 0;
    for (int i = 6; i < 8; i++)
    {
        recvPay[i] = (int) payload[i];
        Serial.print((char)recvPay[i]);
        stateFB[x] = recvPay[i];
        x++;
    }
    Serial.println("\n");
    Serial.print("Steering Payload:- ");
    Serial.println(yPayload);
    Serial.print("Distance Payload:- ");
    Serial.println(temp);
    Serial.print("Forward/Backward Payload:- ");
    Serial.print((char)stateFB[0]);
    Serial.println((char)stateFB[1]);
    /*
     * Insert vehicle movement commands here.
     * Commands will be executed after payload is received.
     */
     laneHandler(atoi(yPayload));
     setX(atoi(fBPayload), stateFB);
    Serial.println("\n--------------");
}

/*
 * Handles turning for the vehicle.
 * Will compare targetCenter to predefined variables.
 */
void laneHandler(int targetCenter)
{
    if (targetCenter == carCenter || targetCenter == carCenter + 1 || targetCenter == carCenter - 1)
        servo.write(restingPos);
    else if (targetCenter < restingPos && targetCenter >= sLeftThresh)
        servo.write(sLeft);
    else if (targetCenter < sLeftThresh && targetCenter >= mLeftThresh)
        servo.write(mLeft);
    else if (targetCenter < mLeftThresh)
        servo.write(lLeft);
    else if (targetCenter > restingPos && targetCenter <= sRightThresh)
        servo.write(sRight);
    else if (targetCenter > sRightThresh && targetCenter <= mRightThresh)
        servo.write(mRight);
    else if (targetCenter > mRightThresh)
        servo.write(lRight);
}


/*
 * Sets vehicle's speed using PWM.
 * Will need to declare variable at some point.
 */
void setX(int magnitude, int fbState[2])
{
    for (int i = 0; i < 2; i ++)
    {
        if (fbState[i] == '1')
            digitalWrite(pinNames[i], HIGH);
        else
            digitalWrite(pinNames[i], LOW);
    }
    if (magnitude <= 10)
    {
        analogWrite(enFB, Slight);
    }
    else if (magnitude > 10 && magnitude < 20)
    {
        analogWrite(enFB, Moderate);
    }
    else if (magnitude >= 20)
    {
        analogWrite(enFB, Significant);
    }
}


void loop() {
    client.subscribe(moveTopic);
    delay(1);
    client.loop();
}
