import os.path
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, NUMERIC

schema = Schema(docID=NUMERIC(stored=True), contents=TEXT)
index_dir = "index"

if not os.path.exists(index_dir):
    os.makedirs(index_dir)

ix = create_in(index_dir, schema)

writer = ix.writer()

with open('./doc/document.text', 'r') as f:
    text = f.read()
    docs = text.split('********************************************\n')[:-1]
    for doc in docs:
        br = doc.find('Document')
        try:
            docID = int(doc[br:].split('\n')[0].split(' ')[-1])
        except:
            print(doc)
            print(br)
            raise ValueError
        doc_text = '\n'.join(doc[br:].split('\n')[1:])
        writer.add_document(docID=docID, contents=doc_text)

writer.commit()
print("make index!")