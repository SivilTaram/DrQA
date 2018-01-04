import os
import json

all_lines = []


class Article:
    def __init__(self, text):
        data = json.loads(text, encoding='utf8')
        self.title = data['title']
        self.url = data['url']
        paragraphs = data['text'].split('\n')
        # 去除空串
        self.paragraphs = list(filter(lambda x: x, paragraphs))
        self.id = data['id']


def merge_documents(dir_name):
    global all_lines
    files = os.listdir(dir_name)
    for f in files:
        file_path = os.path.join(dir_name, f)
        if os.path.isdir(file_path):
            merge_documents(file_path)
        else:
            f_ = open(file_path, encoding='utf8')
            all_lines += f_.readlines()


merge_documents('WikiDocuments')

for line in all_lines:
    article = Article(line)
    with open('WikiDocuments/{0}.txt'.format(article.id), 'w', encoding='utf8') as f:
        f.writelines(json.dumps(article.__dict__))
