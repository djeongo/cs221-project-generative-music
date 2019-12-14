from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse
import numpy as np

notes = ['C', 'C#', 'D', 'D#', 'E-', 'E', 'F', 'G-', 'F#', 'G', 'G#', 'A-', 'A', 'B-', 'B']
hsv = plt.get_cmap('hsv')
colors = hsv(np.linspace(0, 1.0, len(notes)))


def to_color(word):
    if word[:-1] in notes:
        return np.array([colors[notes.index(word[:-1])]])
    return 'gray'

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='weights-4/word2vec.model')
    ARGS = parser.parse_args()

    model = Word2Vec.load(ARGS.model)
    labels = []
    tokens = []
    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=4, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    c = []
    for i, value in enumerate(new_values):
        x.append(value[0])
        y.append(value[1])
        c.append(to_color(labels[i]))

    plt.figure(figsize=(8, 8)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i], c=c[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.title('TSNE plot of Word2Vec embeddings of notes and rests')
    plt.show()
    plt.grid()
    
    plt.tight_layout()

    plt.savefig('plots/word2vec-tosne.png')

    print(model.wv.most_similar('D6'))