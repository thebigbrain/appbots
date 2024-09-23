from appbots.core.db import get_table


def get_bot_memo(anno_id: int):
    bot_memo_table = get_table('bot_memories')
    return bot_memo_table.find_one(id=anno_id)


def get_tag_annotations(limit: int = 100):
    tag_annotations = get_table('tag_annotations')
    return tag_annotations.find(_limit=limit, order_by="-anno_id")


def update_tag_prediction(anno_id: str, prediction: str):
    tag_annotations = get_table('tag_annotations')
    tag_annotations.upsert(dict(id=anno_id, prediction=prediction), ['id'])