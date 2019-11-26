import os
from tqdm import tqdm
from email.parser import HeaderParser
from email.policy import compat32
from email import message_from_string
from util import enron_to_postgres as db

ENRON_MAIL_PATH = "../datasets/enron"
test_path = "/home/moiddes/opt/datasets/enron/white-s/val"


def parsemail(filepath):
    """
    parse the mail. return email object
    """
    with open(filepath) as fp:
        mail = message_from_string(fp.read(), policy=compat32)
        return mail


if __name__ == "__main__":
    falsely_parsed = 0
    for dirpath, dnames, fnames in os.walk(ENRON_MAIL_PATH):
        for file in tqdm(fnames, desc=f'walking {dirpath}'):
            try:
                parsed_mail = parsemail(os.path.join(dirpath, file))
            except UnicodeDecodeError:
                print(f"mail {os.path.join(dirpath, file)} wasn't parsed correctly")
                falsely_parsed += 1
            db.insert_document(parsed_mail.get_payload(), parsed_mail['subject'])
