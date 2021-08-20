import smtplib

def sendmail(msg):
    TO = " "
    SUBJECT = 'No Mask'
    TEXT =" Person With out mask"
     
    print(TEXT)
    # Gmail Sign In
    gmail_sender = "jayaram4241@gmail.com"
    gmail_passwd = "jayaram123"

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.login(gmail_sender, gmail_passwd)

    BODY = '\r\n'.join(['To: %s' % TO,
                        'From: %s' % gmail_sender,
                        'Subject: %s' % SUBJECT,
                        '', TEXT])

    try:
        server.sendmail(gmail_sender, [TO], BODY)
        print ('email sent')
    except:
        print ('error sending mail')

    server.quit()
