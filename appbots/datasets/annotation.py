from appbots.core.db import get_table, get_db


def get_bot_memos():
    bot_memo_table = get_table('bot_memories')
    return list(bot_memo_table.find(_limit=100, order_by="-id"))


def get_not_bboxes_generated_memos(limit=100):
    result_iter = get_db().query(
        f'SELECT * from bot_memories WHERE bboxes_generated IS NULL OR bboxes_generated = 0 ORDER BY id DESC LIMIT {limit}'
    )
    return list(result_iter)


def get_bot_memo(anno_id: int):
    bot_memo_table = get_table('bot_memories')
    return bot_memo_table.find_one(id=anno_id)


def get_tag_annotations(limit: int = 100):
    tag_annotations = get_table('tag_annotations')
    return tag_annotations.find(_limit=limit, order_by="-anno_id")


def update_tag_prediction(anno_id: str, prediction: str):
    tag_annotations = get_table('tag_annotations')
    tag_annotations.upsert(dict(id=anno_id, prediction=prediction), ['id'])
