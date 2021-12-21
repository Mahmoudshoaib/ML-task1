import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer

ds = pd.read_csv("Wuzzuf_Jobs.csv")

x= ds.iloc[0:5,0:5] #some of dataset

print(x)

#---------------------

print(ds.info()) #structure

print(ds.describe()) #summary

#---------------------
 
ds.sort_values(by=["Title", "Company", "Location", "Type", "Level", "YearsExp", "Country", "Skills"], inplace=True)
 
ds.drop_duplicates(inplace=True)

y = ds.iloc[:, :].values

imp_mean = SimpleImputer(missing_values=None, strategy='most_frequent')

imp_mean = imp_mean.fit(y[:, :])

y[:, :] = imp_mean.transform(y[:, :])

print(y)

#---------------------
a = ds.iloc[:, [1]]

b = a.value_counts()

print(b)

plt.pie(b)

plt.show()

#---------------------

z = ds.iloc[:100, [0]]

c = z.value_counts()

print(c)

fig = plt.figure(figsize=(40,10))

c.plot.bar(color='#f0f',width=.5,fontsize=25)

plt.xlabel("Jobs")

plt.ylabel("No. of workers")

plt.title("Most Jobs")

plt.show()

#---------------------

q = ds.iloc[:100, [2]]

w = q.value_counts()

print(w)

w.plot.bar(color='#fca', width=0.5, fontsize=10)

plt.xlabel("Areas")

plt.ylabel("No. of people")

plt.title("Most Areas")

plt.show()

#---------------------

