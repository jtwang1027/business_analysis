''''
Using sales data generated from Square app to quantify sales
'''

import pandas as pd
import numpy as np
from collections import Counter

filename=r'C:\Users\jwang\Desktop\quickly\items_transactions\items-190315.csv'

items= pd.read_csv(filename)


mod=items['Modifiers Applied']
#remove null before splitting
mod=mod[mod.notnull()]

[x.strip() for x in mod.split(',')]

all=mod.apply(lambda x: x.split(','))
all=all.tolist()
flat_list = [item for sublist in all for item in sublist]

opt=[y.strip() for y in flat_list]

top=['Honey Boba', 'Crystal Boba','Strawberry Hearts','Coffee Jelly','Popping Boba: Mango',
   'Mango Stars',  'Lychee Jelly', 'Grass Jelly',  'Egg Pudding', 'Green Apple Jelly','Red Bean',
   'Rainbow Jelly']

#keep only if in toppings list (top)
only=[x for x in opt if x in top]

Counter(only).keys()
Counter(only).values()

