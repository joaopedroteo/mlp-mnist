#!/usr/bin/env python
# coding: utf-8

# # Multilayer Perceptron MNIST

# ## Imports

# In[19]:


# get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
import random
import pprint

import numpy as np
import pandas as pd
import seaborn as sn

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.datasets.base import get_data_home 
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt


# ## Baixando e configurando o dataset

# In[20]:


#Ignorando os warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
print('Baixando MNIST data')
mnist_dataset = fetch_mldata('MNIST original')
print('Download concluído.')

data = mnist_dataset.data
target = mnist_dataset.target
#mages = mnist_dataset.images

tamanho_dataset_teste = 0.2

data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=tamanho_dataset_teste, random_state=42)

print(f"Training dataset size: {len(data_train)}")
print(f"Test dataset size: {len(target_train)}")


# Alguns números do dataset:

# In[21]:


fig, ax = plt.subplots(2,5)
for i, ax in enumerate(ax.flatten()):
    im_idx = np.argwhere(target == i)[0]
    plottable_image = np.reshape(data[im_idx], (28, 28))
    ax.imshow(plottable_image, cmap='gray_r')


# In[22]:


fig, ax = plt.subplots(1)
im_idx = 34567
plottable_image = np.reshape(data[im_idx], (28, 28))
ax.imshow(plottable_image, cmap='gray_r')


# In[23]:


# for index, value in enumerate(data[im_idx]):
    # if index % 28 == 0: print("\n")
    # print("{0:0=3d} ".format(value), end="")


# In[24]:


classificador_MLP = MLPClassifier(hidden_layer_sizes=(64,128,64,10), verbose=True, alpha=0.0001, tol=1e-4)

print("Fitting model")
classificador_MLP.fit(data_train,target_train)
print("Fitting model ended")


# In[25]:


print("Começando testes")
predictions = classificador_MLP.predict(data_test)
print("Fim dos testes")


# In[26]:


print(confusion_matrix(target_test,predictions))
cm = confusion_matrix(target_test,predictions)
df_cm = pd.DataFrame(cm, index = [i for i in "0123456789"],
                  columns = [i for i in "0123456789"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)


# In[27]:


print("Training set score: %f" % classificador_MLP.score(data_train, target_train))
print("Test set score: %f" % classificador_MLP.score(data_test, target_test))

