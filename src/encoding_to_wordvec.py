from gensim.models import Word2Vec
import os
import pickle
import sys

if __name__=="__main__":
    data = []
    path=sys.argv[1]
    for root, dirs, files in os.walk(path):
        for file in files:
            if not file.startswith('.') and file.endswith('.txt'):
                print('Processing {}'.format(file))
                with open(os.path.join(root, file)) as f:
                    data.append(f.read().split())

    print(len(data))

    model = Word2Vec(
        data, size=300, window=30, min_count=1, workers=4, iter=100)

    model_name = 'word2vec.model'
    print('Saving model to {}'.format(model_name))
    model.save(model_name)

    wv = {}
    for word in model.wv.vocab:
        wv[word] = model.wv[word]
        print('{} - {}'.format(word, model.wv.most_similar(word)))

    print('Saving pickle to wv.pickle')
    pickle.dump(wv, open('wv.pickle', 'wb'))

    print('num words: {}'.format(len(wv)))