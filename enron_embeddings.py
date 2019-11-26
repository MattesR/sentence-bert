


if __name__ == "__main__":
    connections.create_connection(hosts=['localhost:9200'])
    Mail.init()
    el_id = 0
    falsely_parsed = 0
    for dirpath, dnames, fnames in os.walk(ENRON_MAIL_PATH):
        for file in tqdm(fnames, desc=f'walking {dirpath}'):
            try:
                parsed_mail = parsemail(os.path.join(dirpath, file))
            except UnicodeDecodeError:
                print(f"mail {os.path.join(dirpath, file)} wasn't parsed correctly")
                falsely_parsed += 1
            if parsed_mail['to']:
                receiver = parsed_mail['to'].split(',')
            elastic_mail = Mail(sent_date=parsed_mail['date'],
                                sender=parsed_mail['from'],
                                receiver=parsed_mail['to'],
                                subject=parsed_mail['subject'],
                                body=parsed_mail.get_payload(),
                                meta={'id': el_id}
                                )
            reason = elastic_mail.save()