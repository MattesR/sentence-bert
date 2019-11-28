import psycopg2


def insert_document(connection, name):
    document_command = """
    INSERT INTO documents(document_name)
    VALUES
        (%s)
    RETURNING document_id;
    """
    try:
        cur = connection.cursor()
        cur.execute(document_command, (name,))
        document_id = cur.fetchone()[0]
        cur.close()
        connection.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    return document_id


def insert_text_unit(connection, content, position, document_id):
    unit_command = """
        INSERT INTO text_units(content, position, document_id)
        VALUES
            (%s, %s, %s)
        RETURNING unit_id;
        """
    try:
        cur = connection.cursor()
        cur.execute(unit_command, (content, position, document_id))
        unit_id = cur.fetchone()[0]
        cur.close()
        connection.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    return unit_id
