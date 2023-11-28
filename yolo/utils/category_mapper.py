import json

class AnnotationTransformer:
    def __init__(self, annotations, instructions={}):
        self.annotations = annotations 
        self.instructions = instructions
        self._annotations()
        self._categories()
        if len(instructions) > 0:
            self._transform_annotations()
            self._categories()
        self._category_mapper()

    def _annotations(self):
        if isinstance(self.annotations, str):
            with open(self.annotations, 'r') as oj:
                self.annotations = json.load(oj)
        elif isinstance(self.annotations, dict):
            self.annotations = self.annotations
        else:
            print("annotations wernt loaded")
            return None

    def _categories(self):
        self.id_category = {}
        for annote in self.annotations['annotations']:
            cat_id  = annote['category_id']

            for catinfo in self.annotations['categories']:
                if catinfo['id'] == cat_id:
                    if cat_id not in self.id_category.keys():
                        self.id_category[cat_id] = catinfo['name']

        self.category_id = {}
        for id, cat in self.id_category.items():
            if cat not in self.category_id:
                self.category_id[cat] = [id]
            else:
                self.category_id[cat].append(id)

    def _transform_annotations(self):
        new_annotations = {'images': [], 'annotations': [], 'categories': []}
        image_ids = []
        for annote in self.annotations['annotations']:
            annote_cat_id = annote['category_id']
            annote_cat = self.id_category[annote_cat_id]
            if annote_cat in self.instructions.keys():
                if self.instructions[annote_cat] == 'ignore':
                    continue
                else:
                    new_annotations['annotations'].append(annote)
                    image_ids.append(annote['image_id'])
            else:
                new_annotations['annotations'].append(annote)
                image_ids.append(annote['image_id'])

        for image in self.annotations['images']:
            if image['id'] not in image_ids:
                continue
            else:
                new_annotations['images'].append(image)

        for category in self.annotations['categories']:
            if category['name'] in self.instructions.keys():
                if self.instructions[category['name']] == 'ignore':
                    continue
                else:
                    new_annotations['categories'].append(
                        {
                            'id': category['id'], 
                            'name': self.instructions[category['name']], 
                        }
                    )
            else:
                new_annotations['categories'].append(category)

        self.annotations = new_annotations

    def _category_mapper(self):
        self.cat_mapper = {}
        for key, (_, ids) in enumerate(self.category_id.items()):
            for id in ids:
                self.cat_mapper[id] = key
