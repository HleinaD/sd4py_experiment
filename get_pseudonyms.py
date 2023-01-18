import numpy as np
import pandas as pd

numbers = np.arange(999)
np.random.seed(42)
np.random.shuffle(numbers)
pseudonyms = []
for i, num in enumerate(numbers):
    if i % 2 == 0:
        pseudonyms.append("E{}E".format(str(num).zfill(3)))
    else:
        pseudonyms.append("B{}B".format(str(num).zfill(3)))

pd.Series(pseudonyms).to_csv('/home/daniel/Documents/private-EPA-2/experiment_drafting/pseudonyms.txt', header=False, index=False)
