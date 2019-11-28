from email.policy import compat32
from email import message_from_string


def parsemail(filepath):
    """
    parse the mail. return email object
    """
    with open(filepath) as fp:
        mail = message_from_string(fp.read(), policy=compat32)
        return mail
