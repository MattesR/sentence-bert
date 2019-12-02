import psycopg2


def insert_document(connection, name):
    """
    Takes a connection and a name and inserts the document into the database.
    returns the database_id of the inserted document
    :param connection: database connection
    :param name: name
    :return: document_id
    """
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
    """
    Inserts a text unit into the database. Needs a connection and the content plus position in the document it came from
    returns it unit_id from the database
    :param connection: database connection
    :param content: text content
    :param position: position inside the document
    :param document_id: document_id of the document
    :return: unit_id from the database
    """
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
