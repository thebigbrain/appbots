import dataset


_db: dataset.Database | None = None


def get_db(db_name="app_bot"):
    global _db
    if _db is None:
        _db = dataset.connect(f'mysql://root:root@192.168.10.115/{db_name}')
    return _db


def get_table(table_name: str) -> dataset.Table:
    db = get_db()
    return db.get_table(table_name)
