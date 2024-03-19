from detection import PersonDetection
from notifications import Notification
import os
from dotenv import load_dotenv
import argparse

def main(capture_index):

    # Load environment variables

    load_dotenv()

    password = os.environ.get("INTRUSALERTS_PASSWORD")
    from_email = os.environ.get("INTRUSALERTS_FROM_EMAIL")
    to_email = os.environ.get("INTRUSALERTS_TO_EMAIL")

    # Instanciate Notification and PersonDetection classes
    email_notification = Notification(from_email, to_email, password)
    detector = PersonDetection(capture_index=capture_index, email_notification=email_notification)

    #Detect
    detector()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the person detection system.")
    parser.add_argument('--capture_index', default=1, help='The index or IP Address of the camera to be used for capture.')
    args = parser.parse_args()

    main(args.capture_index)