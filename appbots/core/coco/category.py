from typing import TypedDict, OrderedDict

from appbots.core.db import get_table


class CocoCategory(TypedDict):
    id: int
    name: str
    supercategory: str


class CocoCategoryUtil:
    _cat_dict: dict = None
    categories: list[CocoCategory] = []

    @classmethod
    def build_coco_category(cls, data: dict):
        name = data.get("name") or ""
        supercategory = data.get("supercategory") or ""
        return CocoCategory(id=data.get('id'), name=name, supercategory=supercategory)

    @classmethod
    def load_categories(cls):
        if cls._cat_dict:
            return cls._cat_dict

        cls._cat_dict = dict()

        categories = get_table('coco_categories').all()
        for category in categories:
            cat_id = category.get('id')
            coco_cat = cls.build_coco_category(category)
            cls._cat_dict[cat_id] = coco_cat
            cls.categories.append(coco_cat)

        return cls.categories

    @classmethod
    def get(cls, category_id) -> CocoCategory:
        cls.load_categories()
        return cls._cat_dict.get(category_id)

