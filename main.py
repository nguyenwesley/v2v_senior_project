# This is a sample Python script.
from __future__ import print_function

import os.path
import pickle

#MQTT Stuff
import time

import paho.mqtt.client as mqtt

import urllib

#Google Auth stuff for Sheets API
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']

###Drive url to Access###
#urllib.request.urlopen("https://script.google.com/macros/s/AKfycbxe3GQaVFcu1N3He549Chgblg1fxXIHp4EbskeulbDaga_q3oo/exec?gps=20&spd=40")
#Use this line of code to simply upload the data. Replace GPS and SPD dummy variables with
#actual modular values.
###END DRIVE URL###

# The ID and range of a sample spreadsheet.
#Also MQTT variables needed for listening and updating spread sheets
server_IP = "10.0.0.73"#This IP is the one used for devices recieiving information through MQTT Data pipes.
dataTopic = "vehicle/Move"#Current established topic for our MQTT 

#Used for googles example code for accessing spread sheet info. Variables were changed, names were kept same for simplicity
#This sheet data is our groups sheet data info along with specific ranges that we wanted modified
EXspreadsheetId = '1ruRFQLasz1iCC2ZW5BVG1tXLvu-DNVu-7bhVyyJROyk'
EXrange = "E1:E3"


#Changing the constructor for on_message by predefining our own routine upon a message recieved within the topic
def on_message(client,userdata,msg):
    print("Message Recieved ->"+msg.topic+" "+ str(msg.payload))#Displays what message was recieved and what data to pull out
    #This if statement ensures we're receiving the correct data package from the same device. Additional fail safe if we use the same topic for different data variables or 
    #something tries to send false packet information
    if len(msg.payload) >= 6:
        temp = chr(msg.payload[3]) + chr(msg.payload[4]) + chr(msg.payload[5])
        #uploads information to the google sheets
        urllib.request.urlopen("https://script.google.com/macros/s/AKfycbxe3GQaVFcu1N3He549Chgblg1fxXIHp4EbskeulbDaga_q3oo/exec?gps="+str(temp)+"&spd=40")
        print("Updated variables in server\n")
    else:
        print("insufficient packet size\n")


def main():
    """Shows basic usage of the Sheets API.
    Prints values from a sample spreadsheet.
    """
    creds = None
    ###Establish MQTT link with previously designated variables(server IP and topic)

    client = mqtt.Client()
    client.on_message = on_message
    client.connect(server_IP, 1883, 60)
    client.subscribe(dataTopic)

 
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('sheets', 'v4', credentials=creds)

    # Call the Sheets API
    sheet = service.spreadsheets()
    client.loop_start()
    #Loops used to constantly check if server side crash is detected by check CRASH box D2
    while(True):
        result = sheet.values().get(spreadsheetId=EXspreadsheetId, range=EXrange).execute()
        values = result.get('values', [])
        
        #Ends loop if nothing found
        if not values:
             print('No data found.')
             client.loop_stop()
             client.disconnect()
        else:
            print(values)
            #If statement to check if box D2 is ever printing true or flase and prints to screen corresponding data. Can be replaced with sound.fx or other
            #means of detection to notify if driver should be wary of a crash and where to be wary of the crash. Reads box and checks if TRUE or FALSE and 
            #outputs corresponding data.
            if str(values[0]) =="['TRUE']":
                print('CRASH DETECTED AT GPS ', (values[2]))

            else:
                print('CURRENT STATUS ',(values[2]))

        #Sleep needed or else GOOGLEAPI quota exceeds alloted amount of requests
        #permitted by GOOGLESAPI, Quota can be increased and different IPs can
        #access different amounts of times. Needs more experimenting for limit testing.
        time.sleep(3)

if __name__ == '__main__':
    main()
