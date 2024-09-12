from appbots.core.db import get_db


def get_anno(anno_id: int):
    db = get_db()
    return db.get_table('bot_memories').find_one(id=anno_id)


def get_tag_annotations(limit: int = 100):
    db = get_db()
    tag_annotations = db.get_table('tag_annotations')
    return tag_annotations.find(_limit=limit, order_by="-anno_id")


def update_tag_prediction(anno_id: str, prediction: str):
    db = get_db()
    tag_annotations = db.get_table('tag_annotations')
    tag_annotations.upsert(dict(id=anno_id, prediction=prediction), ['id'])