import io

import mysql.connector
from PIL import Image
from mysql.connector import Error


def show_data():
    connection = None
    try:
        # 建立数据库连接
        connection = mysql.connector.connect(
            host='192.168.206.128',
            user='root',
            password='1234',
            database='detection_data',
            connection_timeout=3
        )

        if connection.is_connected():
            cursor = connection.cursor()
            cursor.execute("select database();")
            record = cursor.fetchone()[0]
            print("You're connected to database: ", record)

            cursor.execute("select result, savetime from data;")
            result = cursor.fetchall()
            for row in result:
                print(row)

    except Error as e:
        print("Error while connecting to MySQL", e)

    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()


def insert_into_database(string, bimgdata):
    connection = None
    try:
        # 建立数据库连接
        connection = mysql.connector.connect(
            host='192.168.206.128',
            user='root',
            password='1234',
            database='detection_data',
            connection_timeout=3
        )

        if connection.is_connected():
            cursor = connection.cursor()
            cursor.execute("select database();")
            record = cursor.fetchone()[0]
            print("You're connected to database: ", record)

            # 执行查询
            insert_query = "INSERT INTO data (result, image_data) VALUES (%s, %s)"
            cursor.execute(insert_query, (string, bimgdata))
            connection.commit()

    except Error as e:
        print("Error while connecting to MySQL", e)

    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()


def delete_from_database(pic_id):
    connection = None
    try:
        # 建立数据库连接
        connection = mysql.connector.connect(
            host='192.168.206.128',
            user='root',
            password='1234',
            database='detection_data',
            connection_timeout=3
        )

        if connection.is_connected():
            cursor = connection.cursor()
            cursor.execute("select database();")
            record = cursor.fetchone()[0]
            print("You're connected to database: ", record)

            # 执行查询
            delete_query = "DELETE FROM data WHERE pic_id = %s"
            cursor.execute(delete_query, (pic_id,))
            connection.commit()
            cursor.execute("select * from data;")
            result = cursor.fetchall()
            for row in result:
                print(row)

    except Error as e:
        print("Error while connecting to MySQL", e)

    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")


def draw_from_database(id):
    connection = None
    try:
        # 建立数据库连接
        connection = mysql.connector.connect(
            host='192.168.206.128',
            user='root',
            password='1234',
            database='detection_data',
            connection_timeout=3
        )

        if connection.is_connected():
            cursor = connection.cursor()
            cursor.execute("select database();")
            record = cursor.fetchone()[0]
            print("You're connected to database: ", record)

            # 执行查询
            draw_query = "SELECT image_data FROM data where id = %s;"
            cursor.execute(draw_query, (id,))
            result = cursor.fetchall()
            return result

    except Error as e:
        print("Error while connecting to MySQL", e)

    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()


if __name__ == "__main__":
    show_data()
    # with open("C:\\Users\\14485\\Pictures\\Screenshots\\gt_7.jpg", "rb") as file:
    #     binary_data = file.read()
    #     insert_into_database("gzy", binary_data)
    image = Image.open(io.BytesIO(draw_from_database(3)[0][0]))
    image.show()
