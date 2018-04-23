import pandas as pd
import numpy as np

gci_compe_path = "../../../../weblab/weblab_datascience/competitions/"

train = pd.read_csv(gci_compe_path + "wine_quality/train.csv").values
print(train)
