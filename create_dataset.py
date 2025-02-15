import sqlite3
import csv


def load_data(db_file, data_count: int = None):
    sql = f"SELECT title, part, vendor, product, version FROM categorized_cpes WHERE part='a' ORDER BY RANDOM()"
    if data_count is not None:
        sql += f" LIMIT {data_count}"
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(sql)
    columns = [desc[0] for desc in cursor.description]
    datas = [
        {key: (value.replace("\\", "") if isinstance(value, str) else value) for key, value in zip(columns, row)}
        for row in cursor.fetchall()
    ]
    conn.close()
    return datas


def save_csv(datas, output_file):
    with open(output_file, "w", encoding="utf-8", newline='') as f:
        names = ['title', 'part', 'vendor', 'product', 'version']
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        writer.writerows(datas)


def main():
    db_file = "datas/cpe.sqlite3"
    output_file = "datas/categorized_cpes_150000.csv"

    data = load_data(db_file, data_count=150000)
    save_csv(data, output_file=output_file)

    print("Dataset saved.")


if __name__ == "__main__":
    main()