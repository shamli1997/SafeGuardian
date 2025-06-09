
import requests

def send_notification_to_guardian(message):

    try:
        title="Alert : 🚨 Fall Detected!"
        message = f"{title}\n\n{message}"
        requests.post(f"https://ntfy.sh/safeguardian", data=message.encode('utf-8'))
        
        return True
    except Exception as e:
        print(f"Failed to send SMS: {e}")
        return False