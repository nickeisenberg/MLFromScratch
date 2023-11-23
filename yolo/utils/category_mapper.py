import json

def category_mapper(annote_json) -> dict:
    if isinstance(annote_json, str):
        with open(annote_json, 'r') as oj:
            annotations = json.load(oj)
    elif isinstance(annote_json, dict):
        annotations = annote_json
    else:
        print("annotations wernt loaded")
        return {}

    cat_ids = {}
    for annote in annotations['annotations']:
        id = annote['category_id']
        if id not in cat_ids.keys():
            cat_ids[id] = len(cat_ids)

    return cat_ids

