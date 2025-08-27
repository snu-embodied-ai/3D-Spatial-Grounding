NUM_LABELS = 41  # Will be converted to 20 as defined in IGNORE_LABELS.

CLASS_LABELS = ('wall', 'floor', 
                'cabinet', 'bed', 
                'chair', 'sofa', 
                'table', 'door', 
                'window', 'bookshelf', 
                'picture', 'counter', 
                'desk', 'curtain', 
                'refrigerator', 'shower curtain', 
                'toilet', 'sink', 
                'bathtub', 'otherfurniture')
VALID_CLASS_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
VALID_CLASS_ID_TO_LABEL = dict(zip(VALID_CLASS_IDS, CLASS_LABELS))
IGNORE_LABELS = tuple(set(range(NUM_LABELS)) - set(VALID_CLASS_IDS))


CLASS_LABELS_INSTANCE = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter',
                             'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
VALID_CLASS_IDS_INSTANCE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
IGNORE_LABELS_INSTANCE = tuple(set(range(NUM_LABELS)) - set(VALID_CLASS_IDS_INSTANCE))