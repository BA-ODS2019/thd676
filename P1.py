#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 13:40:24 2020

@author: dpbmac-05
"""

#opgave 2)
#a) henter dataset
import pandas as pd

import numpy as np
pd.options.display.max_rows = 900

titanic=pd.read_csv('titanic.csv')

#b)#viser at der ikke mangler data
print(titanic.isnull().sum())


#opgave 3) beskriver datasettet 

titanic.shape
titanic.size
titanic.dtypes
print('Datasettet Titanic har antal columns og rækker ',titanic.shape, 'og',titanic.size, 'antal værdier.')
print('Og der er disse typer data',titanic.dtypes)
print('Middelværdi alder er ',titanic['Age'].mean())
print('Yngste rejsende var',titanic['Age'].min(),'år gammel')
print('Ældste rejsende var',titanic['Age'].max(),'år gammel')

#antal døde
number_travelers = 887
number_survivers = 342
number_dead = (number_travelers - number_survivers)
print('Der var ', number_dead,'mænd, kvinder og børn, der omkom på rejsen' )

#Opgave 4 - viser at der ens efternavne i datasettet.
Names = (titanic['Name'].str.split(' ', n = 2, expand = True))
first_Names = (Names[1])
last_Names = (Names [2])
last_Names.duplicated(keep='first')

#Opgave 5 pivot tabel
#pivot tabel der viser antal overlevende på hhv første anden og tredie klasse.
pivot=pd.pivot_table(titanic,'Survived', ['Pclass'])

print (pivot)

# skaber en pivotabel vist i barchart med antal overlevende på de tre klasser.
pivot=pd.pivot_table(titanic,'Survived', ['Pclass'],aggfunc= np.sum).plot.bar()

print (pivot)

# skaber en pivotabel vist i barchart med antal overlevende på de tre klasser.
pivot=pd.pivot_table(titanic,'Survived', ['Sex'],aggfunc= np.sum).plot.bar()

print (pivot)


