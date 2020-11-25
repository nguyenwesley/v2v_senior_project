# This is a sample Python script.
from __future__ import print_function

import os.path
import pickle

#MQTT Stuff
import time

import paho.mqtt.client as mqtt

import urllib


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
server_IP = "10.0.0.73"#Need ADAMS IP
dataTopic = "vehicle/Move"

EXspreadsheetId = '1ruRFQLasz1iCC2ZW5BVG1tXLvu-DNVu-7bhVyyJROyk'
EXrange = "E1:E3"

def on_message(client,userdata,msg):
    print("Message Recieved ->"+msg.topic+" "+ str(msg.payload))
    if len(msg.payload) >= 6:
        temp = str(msg.payload[3-5])
        urllib.request.urlopen("https://script.google.com/macros/s/AKfycbxe3GQaVFcu1N3He549Chgblg1fxXIHp4EbskeulbDaga_q3oo/exec?gps="+temp+"&spd=40")
        print("Updated variables in server\n")
    else:
        print("insufficient packet size\n")


def main():
    """Shows basic usage of the Sheets API.
    Prints values from a sample spreadsheet.
    """
    creds = None
###

    client = mqtt.Client()
    client.on_message = on_message
    client.connect(server_IP, 1883, 60)
    client.subscribe(dataTopic)

###
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
    while(True):
        result = sheet.values().get(spreadsheetId=EXspreadsheetId, range=EXrange).execute()
        values = result.get('values', [])

        if not values:
             print('No data found.')
             client.loop_stop()
             client.disconnect()
        else:
            print(values)
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
