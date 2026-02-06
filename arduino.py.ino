#include <Wire.h> 
#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27,20,4);  // set the LCD address to 0x27 for a 16 chars and 2 line display

const int redLEDs[] = {2, 4, 6, 8};    // Red LED pins
const int greenLEDs[] = {3, 5, 7, 9};  // Green LED pins
const int numLEDs = 4;                 // Number of LED pairs
 int greenDuration1 = 10;       // 10 seconds green light duration
int greenDuration2 = 10; 
 int greenDuration3 = 10; 
 int greenDuration4 = 10; 
int  state1=0;
String val1,val2;
int stat1=0,stat2=0;
void setup() 
{
  Serial.begin(9600);
  for (int i = 0; i < numLEDs; i++) {
    pinMode(redLEDs[i], OUTPUT);
    pinMode(greenLEDs[i], OUTPUT);
  }
  
  // Initial state: All Red ON, Green OFF
  for (int i = 0; i < numLEDs; i++) {
    digitalWrite(redLEDs[i], HIGH);
    digitalWrite(greenLEDs[i], LOW);
  }
  lcd.init();                      // initialize the lcd 
  lcd.init();
  // Print a message to the LCD.
  lcd.backlight();
  lcd.setCursor(0,0);
  lcd.print(" Smart Traffic ");
  lcd.setCursor(0,1);
  lcd.print(" Management     ");

  delay(2000); // Optional startup delay
 
}

void loop() {
  
  turnON1();
   
  turnON2();
   
  turnON3();
   
  turnON4();
   


  
}
void delay1(int a)
{
  for(int i=0;i<a;i++)
  {
    delay(1000);
    read1();
  }
}
void read1()
{
  String myString = Serial.readString();
  Serial.println(myString);
    val1=  getValue(myString, ':', 0);
    val2=  getValue(myString, ':', 1);
   String val3=  getValue(myString, ':', 2);
   if(val1.toInt()>3)
   {
      stat1=1;
    greenDuration1 = 15; 
   }
   if(val2.toInt()>3)
   {
    stat2=1;
    greenDuration2= 15; 
   }
  Serial. println(val3);
   if (val3=="siren"&&state1==0)
   {
      lcd.setCursor(0,0);
  lcd.print(" Ambulance      ");
  lcd.setCursor(0,1);
  lcd.print(" Coming      ");
  Serial.println("Emergency");state1=1;
  digitalWrite(greenLEDs[1], LOW);
  digitalWrite(greenLEDs[2], LOW);
  digitalWrite(greenLEDs[0], LOW);
  digitalWrite(redLEDs[3], LOW);
  digitalWrite(greenLEDs[3], HIGH);
  delay1(greenDuration4);
  digitalWrite(greenLEDs[3], LOW);
  digitalWrite(redLEDs[3], HIGH);
   }
}
void turnON1() {
 if( stat1==1)
 {
  greenDuration1=15;
   lcd.setCursor(0,0);
  lcd.print(" More vehicle       ");
  lcd.setCursor(0,1);
  lcd.print(" SIg1 Green on        ");
  stat1=0;
 }
 else
 {
    lcd.setCursor(0,0);
  lcd.print(" Signal 1       ");
  lcd.setCursor(0,1);
  lcd.print(" Green on        ");
 }
  Serial.println('1');
  digitalWrite(redLEDs[0], LOW);
  digitalWrite(greenLEDs[0], HIGH);
  delay1(greenDuration1);
  digitalWrite(greenLEDs[0], LOW);
  digitalWrite(redLEDs[0], HIGH);

  greenDuration1=10;
}

void turnON2() {

  if( stat2==1)
 {
  greenDuration2=15;
  stat2=0;
   lcd.setCursor(0,0);
  lcd.print(" More vehicles       ");
  lcd.setCursor(0,1);
  lcd.print(" SIg2 Green on        ");
 }
 else
 {
   lcd.setCursor(0,0);
  lcd.print(" Signal 2      ");
  lcd.setCursor(0,1);
  lcd.print(" Green on        ");
 }
  Serial.println('2');
  digitalWrite(redLEDs[1], LOW);
  digitalWrite(greenLEDs[1], HIGH);
  delay1(greenDuration2);
  digitalWrite(greenLEDs[1], LOW);
  digitalWrite(redLEDs[1], HIGH);
  greenDuration2=10;
}

void turnON3() {
   lcd.setCursor(0,0);
  lcd.print(" Signal 3       ");
  lcd.setCursor(0,1);
  lcd.print(" Green on        ");
  Serial.println('3');
  digitalWrite(redLEDs[2], LOW);
  digitalWrite(greenLEDs[2], HIGH);
  delay1(greenDuration3);
  digitalWrite(greenLEDs[2], LOW);
  digitalWrite(redLEDs[2], HIGH);
}

void turnON4() {
   lcd.setCursor(0,0);
  lcd.print(" Signal 4       ");
  lcd.setCursor(0,1);
  lcd.print(" Green on        ");
  Serial.println('4');
  digitalWrite(greenLEDs[1], LOW);
  digitalWrite(greenLEDs[2], LOW);
  digitalWrite(greenLEDs[0], LOW);
  digitalWrite(redLEDs[3], LOW);
  digitalWrite(greenLEDs[3], HIGH);
  delay1(greenDuration4);
  digitalWrite(greenLEDs[3], LOW);
  digitalWrite(redLEDs[3], HIGH);
}

String getValue(String data, char separator, int index)
{
    int found = 0;
    int strIndex[] = { 0, -1 };
    int maxIndex = data.length() - 1;

    for (int i = 0; i <= maxIndex && found <= index; i++) {
        if (data.charAt(i) == separator || i == maxIndex) {
            found++;
            strIndex[0] = strIndex[1] + 1;
            strIndex[1] = (i == maxIndex) ? i+1 : i;
        }
    }
    return found > index ? data.substring(strIndex[0], strIndex[1]) : "";
}
