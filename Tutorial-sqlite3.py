import sqlite3

# tutorial urls for sqlite3 with python:
# https://www.tutorialspoint.com/sqlite/sqlite_python.htm
# https://www.sqlitetutorial.net/sqlite-python/

# tutorial for sqlite3 via terminal
# https://www.sqlite.org/quickstart.html
# https://www.sqlitetutorial.net/sqlite-commands/
# Example to start sqlite3,
#         to show available commands,
#         to connect to db, show databases, show tables, show table schema, exit sqlite3
# C:\Users\41792\SqLite3> sqlite3
# sqlite> .help
# sqlite> .open C:\Users\41792\Documents\4) Python-Scripts\Tut_Sqlite3\chinook.db
# sqlite> .databases
# sqlite> .tables
# sqlite> .schema albums
# sqlite> .exit

# tutorial via desktop-app DB Browser for SQLite:
# https://sqlitebrowser.org/dl/
# https://sqlitebrowser.org/


def main():
    print("program to demonstrate sql library")
    print("sqlite3-version: ", sqlite3.version, "\n")

    # 1. open connection
    con = one_connect_to_db('test.db')
    one_delete_all_tables(con)

    # 2. create some tables
    two_create_table(con, "school",
                     '''CREATE TABLE school (
                        ID INT PRIMARY KEY NOT NULL,
                        NAME TEXT NOT NULL,
                        AGE INT NOT NULL,
                        ADDRESS CHAR(50),
                        SALARY REAL);''')
    two_create_table(con, "company",
                     '''CREATE TABLE company
                        (ID INT PRIMARY KEY NOT NULL,
                        NAME TEXT NOT NULL,
                        AGE INT NOT NULL,
                        ADDRESS CHAR(50),
                        SALARY REAL);''')
    two_create_table(con, "projects",
                     """ CREATE TABLE IF NOT EXISTS projects (
                         id integer PRIMARY KEY,
                         name text NOT NULL,
                         begin_date text,
                         end_date text
                         ); """)
    two_create_table(con, "tasks",
                     """CREATE TABLE IF NOT EXISTS tasks (
                        id integer PRIMARY KEY,
                        name text NOT NULL,
                        priority integer,
                        status_id integer NOT NULL,
                        project_id integer NOT NULL,
                        begin_date text NOT NULL,
                        end_date text NOT NULL,
                        FOREIGN KEY (project_id) REFERENCES projects (id)
                        );""")
    two_show_all_tables(con)

    # 3. insert records into tables
    three_insert_into_table_company(con)
    #
    project_record_1 = ('Cool App with SQLite & Python', '2015-01-01', '2015-01-30')
    project_record_id_1 = three_insert_into_table_project(con, project_record_1)
    print("inserted project_record_id: ", project_record_id_1)
    project_records = three_show_all_projects_records(con)
    #
    task_record_1 = ('Analyze the requirements of the app', 1, 1, project_record_id_1, '2015-01-01', '2015-01-02')
    task_record_2 = ('Confirm with user about the top requirements', 1, 1, project_record_id_1, '2015-01-03', '2015-01-05')
    task_record_id_1 = three_insert_into_table_task(con, task_record_1)
    task_record_id_2 = three_insert_into_table_task(con, task_record_2)
    print("inserted task_record_ids: ", task_record_id_1)
    print("inserted task_record_ids: ", task_record_id_2)
    three_show_all_tasks_records(con)
    #
    sql_1 = "SELECT name, priority FROM tasks WHERE priority=?"; params_1 = (1,)
    four_select_from_where(con, sql_1, params_1)
    sql_2 = "SELECT name FROM tasks WHERE project_id=?"; params_2 = (1,)
    four_select_from_where(con, sql_2, params_2)
    print()
    #
    task_record_update_1 = (2, '2015-01-04', '2015-01-06', 2)
    five_update_record_in_table_task(con, task_record_update_1)
    five_show_all_tasks_records(con)
    #
    six_delete_record_in_table_task(con, task_record_id_1)
    six_delete_all_records_in_table_task(con)
    #
    seven_close_connection_to_db(con)


def one_connect_to_db(db):
    connection = sqlite3.connect(db)
    print("one_connect_to_db: Database has been opened successfully")
    return connection


def one_delete_all_tables(connection):
    tables = one_get_all_tables_(connection)
    one_delete_all_tables_(connection, tables)
    print("one_delete_all_tables: Database tables have been deleted successfully\n")


def one_get_all_tables_(connection):
    cursor = connection.cursor()
    # cursor.execute("""select name from sqlite_master where type='table';""")
    cursor.execute("SELECT name FROM sqlite_schema WHERE type='table';")
    tables = cursor.fetchall()
    cursor.close()
    return tables


def one_delete_all_tables_(connection, tables):
    cursor = connection.cursor()
    TABLE_PARAMETER = "{TABLE_PARAMETER}"
    for table, in tables:
        sql = f"DROP TABLE {TABLE_PARAMETER};".replace("{TABLE_PARAMETER}", table)
        cursor.execute(sql)
    cursor.close()


def two_create_table(connection, tablename, sql_command):
    cursor = connection.cursor()
    try:
        cursor.execute(sql_command)
        print("two_create_table: Database-Table ", tablename,  "  has been created successfully")
    except:
        print("two_create_table: Database-Table ", tablename,  " not created.")
    cursor.close()


def two_show_all_tables(connection):
    tables = one_get_all_tables_(connection)
    print("two_show_all_tables: ", tables, "\n")


def three_insert_into_table_company(connection):
    cursor = connection.cursor()
    cursor.execute("INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY) \
          VALUES (1, 'Paul', 32, 'California', 20000.00 )")
    cursor.execute("INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY) \
          VALUES (2, 'Allen', 25, 'Texas', 15000.00 )")
    cursor.execute("INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY) \
          VALUES (3, 'Teddy', 23, 'Norway', 20000.00 )")
    cursor.execute("INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY) \
          VALUES (4, 'Mark', 25, 'Rich-Mond ', 65000.00 )")
    connection.commit()
    print("three_insert_into_table_company: the four records created successfully")
    cursor.close()


def three_insert_into_table_project(connection, project):
    cursor = connection.cursor()
    sql = ''' INSERT INTO projects(name,begin_date,end_date)
                  VALUES(?,?,?) '''
    cursor.execute(sql, project)
    connection.commit()
    print("three_insert_into_table_project: record created successfully")
    #cursor.close()
    return cursor.lastrowid


def three_insert_into_table_task(connection, task):
    cursor = connection.cursor()
    sql = ''' INSERT INTO tasks(name,priority,status_id,project_id,begin_date,end_date)
                  VALUES(?,?,?,?,?,?) '''
    cursor.execute(sql, task)
    connection.commit()
    print("three_insert_into_table_task: record created successfully")
    #cursor.close()
    return cursor.lastrowid


def three_show_all_projects_records(connection):
    cursor = connection.cursor()
    # cursor.execute("""select name from sqlite_master where type='table';""")
    cursor.execute("SELECT * FROM projects")
    records = cursor.fetchall()
    cursor.close()
    print("three_show_all_projects_records: ", records, "\n")
    return records


def three_show_all_tasks_records(connection):
    cursor = connection.cursor()
    # cursor.execute("""select name from sqlite_master where type='table';""")
    cursor.execute("SELECT * FROM tasks")
    records = cursor.fetchall()
    cursor.close()
    print("three_show_all_tasks_records: ", records, "\n")
    return records


def four_select_from_where(connection, sql, parameters):
    cursor = connection.cursor()
    cursor.execute(sql, parameters)
    records = cursor.fetchall()
    cursor.close()
    print("four_select_from_where: ", records)
    return records


def five_update_record_in_table_task(connection, task):
    cursor = connection.cursor()
    sql = '''   UPDATE tasks
                SET priority = ? ,
                    begin_date = ? ,
                    end_date = ?
                WHERE id = ?'''
    cursor.execute(sql, task)
    connection.commit()
    print("five_update_record_in_table_task: record updated successfully")
    cursor.close()

def five_show_all_tasks_records(con):
    three_show_all_tasks_records(con)


def six_delete_record_in_table_task(connection, id):
    cursor = connection.cursor()
    sql = 'DELETE FROM tasks WHERE id=?'
    cursor.execute(sql, (id,))
    connection.commit()
    print("six_delete_record_in_table_task: record deleted successfully")
    cursor.close()


def six_delete_all_records_in_table_task(connection):
    cursor = connection.cursor()
    sql = 'DELETE FROM tasks'
    cursor.execute(sql)
    connection.commit()
    print("six_delete_all_records_in_table_task: all records deleted successfully\n")
    cursor.close()


def seven_close_connection_to_db(connection):
    connection.close()
    print("seven_close_connection_to_db: Database has been closed successfully")


if __name__ == '__main__':
    if 1:
        main()