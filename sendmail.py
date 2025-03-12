import smtplib

def sendmail(msg):
    TO = "rangusushanth31@gmail.com"
    SUBJECT = 'No Mask'
    TEXT = msg

    print(f"Sending email: {TEXT}")
    gmail_sender = "facemaskdetection25@gmail.com"
    gmail_passwd = "efmr knov fors fghk"

    server = None
    try:
        # Use SSL connection
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.ehlo()
        
        # Debug: Verify connection
        print("Server connected:", server.noop()[0])
        
        # Login with credentials
        server.login(gmail_sender, gmail_passwd)
        
        # Construct email
        BODY = '\r\n'.join([
            f'To: {TO}',
            f'From: {gmail_sender}',
            f'Subject: {SUBJECT}',
            '', 
            TEXT
        ])
        
        server.sendmail(gmail_sender, [TO], BODY)
        print('Email sent')
        
    except Exception as e:
        print(f'ERROR: {str(e)}')
        # Add more specific error handling here
    finally:
        if server:
            try:
                server.quit()
            except Exception as e:
                print(f"Cleanup error: {e}")

