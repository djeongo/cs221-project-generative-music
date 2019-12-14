from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt

sparsity_files = [
    'plots-similarity/sparsity.csv.npy',
    'plots-similarity-batch-generated-bach/sparsity.csv.npy',
    'plots-similarity-baseline-bach/sparsity.csv.npy',
    'plots-similarity-chopin/sparsity.csv.npy',
    'plots-similarity-batch-generated-chopin/sparsity.csv.npy',
    'plots-similarity-baseline-chopin/sparsity.csv.npy',
]

label_map = {
    'plots-similarity/sparsity.csv.npy':'Bach',
    'plots-similarity-batch-generated-bach/sparsity.csv.npy':'Bach (LSTM)',
    'plots-similarity-baseline-bach/sparsity.csv.npy':'Bach (Baseline)',
    'plots-similarity-chopin/sparsity.csv.npy':'Chopin',
    'plots-similarity-batch-generated-chopin/sparsity.csv.npy':'Chopin (LSTM)',
    'plots-similarity-baseline-chopin/sparsity.csv.npy':'Chopin (Baseline)',
}

styles = {
        'plots-similarity/sparsity.csv.npy':'b',
        'plots-similarity-baseline-bach/sparsity.csv.npy':'b:',
        'plots-similarity-batch-generated-bach/sparsity.csv.npy':'b--',
        'plots-similarity-chopin/sparsity.csv.npy':'r',
        'plots-similarity-baseline-chopin/sparsity.csv.npy':'r:',
        'plots-similarity-batch-generated-chopin/sparsity.csv.npy':'r--'
    }

for sparsity_file in sparsity_files:
    sparsity = np.load(sparsity_file)
    kde = gaussian_kde(sparsity)

    min_val, max_val = 0, 0.15
    x = np.linspace(min_val, max_val, 100)
    pdf = kde.evaluate(x)

    plt.plot(x, pdf, styles[sparsity_file], label=label_map[sparsity_file])

plt.xlabel('Sparsity')
plt.title('Distribution of sparsity of self-similarity matrices')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('plots/sparsity.png')