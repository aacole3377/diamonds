import numpy as np
import pandas as pd
import random as ran

np.set_printoptions(precision=3)
np.set_printoptions(threshold=3)
np.set_printoptions(suppress=True)

df = pd.read_csv('preprocessed.csv', index_col=[0])


chosen_idx = np.random.choice(219595, replace=False, size=43000)

df2 = df.iloc[chosen_idx]
df2 = df2.sort_index()

cond = df2.index


df.drop(cond, inplace=True)
df2.reset_index(drop=True,inplace=True)
a = 1399
b = 8299
c = 499999

num_classes=3
classOne = 0
classTwo = 0
classThree = 0



for i in df['total_sales_price']:
    if i <= a:
        classOne += 1
    elif i <= b:
        classTwo += 1
    elif i <= c:
        classThree += 1


df2['chosen_class'] = 'z'

df2['actual_class']=df2['total_sales_price'].apply(lambda x: 'a' if x<=a else ('b' if x<=b else( 'c' )))

cutOne = df.loc[df['total_sales_price'].between(0, a, inclusive='both'), 'cut'].value_counts()
cutTwo = df.loc[df['total_sales_price'].between(a, b, inclusive='right'), 'cut'].value_counts()
cutThree = df.loc[df['total_sales_price'].between(b, c, inclusive='right'), 'cut'].value_counts()

colorOne = df.loc[df['total_sales_price'].between(0, a, inclusive='both'), 'color'].value_counts()
colorTwo = df.loc[df['total_sales_price'].between(a, b, inclusive='right'), 'color'].value_counts()
colorThree = df.loc[df['total_sales_price'].between(b, c, inclusive='right'), 'color'].value_counts()

clarityOne = df.loc[df['total_sales_price'].between(0, a, inclusive='both'), 'clarity'].value_counts()
clarityTwo = df.loc[df['total_sales_price'].between(a, b, inclusive='right'), 'clarity'].value_counts()
clarityThree = df.loc[df['total_sales_price'].between(b, c, inclusive='right'), 'clarity'].value_counts()

cut_qualityOne = df.loc[df['total_sales_price'].between(0, a, inclusive='both'), 'cut_quality'].value_counts()
cut_qualityTwo = df.loc[df['total_sales_price'].between(a, b, inclusive='right'), 'cut_quality'].value_counts()
cut_qualityThree = df.loc[df['total_sales_price'].between(b, c, inclusive='right'), 'cut_quality'].value_counts()

symmetryOne = df.loc[df['total_sales_price'].between(0, a, inclusive='both'), 'symmetry'].value_counts()
symmetryTwo = df.loc[df['total_sales_price'].between(a, b, inclusive='right'), 'symmetry'].value_counts()
symmetryThree = df.loc[df['total_sales_price'].between(b, c, inclusive='right'), 'symmetry'].value_counts()

polishOne = df.loc[df['total_sales_price'].between(0, a, inclusive='both'), 'polish'].value_counts()
polishTwo = df.loc[df['total_sales_price'].between(a, b, inclusive='right'), 'polish'].value_counts()
polishThree = df.loc[df['total_sales_price'].between(b, c, inclusive='right'), 'polish'].value_counts()

girdle_minOne = df.loc[df['total_sales_price'].between(0, a, inclusive='both'), 'girdle_min'].value_counts()
girdle_minTwo = df.loc[df['total_sales_price'].between(a, b, inclusive='right'), 'girdle_min'].value_counts()
girdle_minThree = df.loc[df['total_sales_price'].between(b, c, inclusive='right'), 'girdle_min'].value_counts()

girdle_maxOne = df.loc[df['total_sales_price'].between(0, a, inclusive='both'), 'girdle_max'].value_counts()
girdle_maxTwo = df.loc[df['total_sales_price'].between(a, b, inclusive='right'), 'girdle_max'].value_counts()
girdle_maxThree = df.loc[df['total_sales_price'].between(b, c, inclusive='right'), 'girdle_max'].value_counts()

train_size = 176596

accuracy = 0
num = 0
carat_weight_varOne = df.loc[df['total_sales_price'].between(0, a, inclusive='right'), 'carat_weight'].var()
carat_weight_meanOne = df.loc[df['total_sales_price'].between(0, a, inclusive='right'), 'carat_weight'].mean()

meas_width_varOne = df.loc[df['total_sales_price'].between(0, a, inclusive='both'), 'meas_width'].var()
meas_width_meanOne = df.loc[df['total_sales_price'].between(0, a, inclusive='both'), 'meas_width'].mean()

carat_weight_varTwo = df.loc[df['total_sales_price'].between(a, b, inclusive='right'), 'carat_weight'].var()
carat_weight_meanTwo = df.loc[df['total_sales_price'].between(a, b, inclusive='right'), 'carat_weight'].mean()

meas_width_varTwo = df.loc[df['total_sales_price'].between(a, b, inclusive='right'), 'meas_width'].var()
meas_width_meanTwo = df.loc[df['total_sales_price'].between(a, b, inclusive='right'), 'meas_width'].mean()

carat_weight_varThree = df.loc[df['total_sales_price'].between(b, c, inclusive='right'), 'carat_weight'].var()
carat_weight_meanThree = df.loc[df['total_sales_price'].between(b, c, inclusive='right'), 'carat_weight'].mean()

meas_width_varThree = df.loc[df['total_sales_price'].between(b, c, inclusive='right'), 'meas_width'].var()
meas_width_meanThree = df.loc[df['total_sales_price'].between(b, c, inclusive='right'), 'meas_width'].mean()



def count(dataset, value):
    try:
        return dataset[value]
    except:
        return 0

test_size=0
for test in df2.values:
    print('Index: ' + str(num))


    prob1 = np.log((classOne + 1) / (train_size + num_classes)) + np.log(
        (count(cutOne,test[0]) + 1) / (classOne + df['cut'].unique().size)) + np.log(
        (count(colorOne,test[1]) + 1) / (classOne + df['color'].unique().size)) + np.log(
        (count(clarityOne,test[2]) + 1) / (classOne + df['clarity'].unique().size)) + np.log(
        (count(cut_qualityOne,test[4]) + 1) / (classOne + df['cut_quality'].unique().size)) + np.log(
        (count(symmetryOne,test[5]) + 1) / (classOne + df['symmetry'].unique().size)) + np.log(
        (count(polishOne,test[6]) + 1) / (classOne + df['polish'].unique().size)) +  np.log(
        (count(girdle_minOne,test[8]) + 1) / (classOne + df['girdle_min'].unique().size)) + np.log(
        (count(girdle_maxOne,test[9]) + 1) / (classOne + df['girdle_max'].unique().size))

    prob2 = np.log((classTwo + 1) / (train_size + num_classes)) + np.log(
        (count(cutTwo, test[0]) + 1) / (classTwo + df['cut'].unique().size)) + np.log(
        (count(colorTwo, test[1]) + 1) / (classTwo + df['color'].unique().size)) + np.log(
        (count(clarityTwo, test[2]) + 1) / (classTwo + df['clarity'].unique().size)) + np.log(
        (count(cut_qualityTwo, test[4]) + 1) / (classTwo + df['cut_quality'].unique().size)) +  np.log(
        (count(symmetryTwo, test[5]) + 1) / (classTwo + df['symmetry'].unique().size)) + np.log(
        (count(polishTwo, test[6]) + 1) / (classTwo + df['polish'].unique().size)) +  np.log(
        (count(girdle_minTwo, test[8]) + 1) / (classTwo + df['girdle_min'].unique().size)) + np.log(
        (count(girdle_maxTwo, test[9]) + 1) / (classTwo + df['girdle_max'].unique().size))

    prob3 = np.log((classThree + 1) / (train_size + num_classes)) + np.log(
        (count(cutThree, test[0]) + 1) / (classThree + df['cut'].unique().size)) + np.log(
        (count(colorThree, test[1]) + 1) / (classThree + df['color'].unique().size)) + np.log(
        (count(clarityThree, test[2]) + 1) / (classThree + df['clarity'].unique().size)) + np.log(
        (count(cut_qualityThree, test[4]) + 1) / (classThree + df['cut_quality'].unique().size)) +  np.log(
        (count(symmetryThree, test[5]) + 1) / (classThree + df['symmetry'].unique().size)) + np.log(
        (count(polishThree, test[6]) + 1) / (classThree + df['polish'].unique().size)) + np.log(
        (count(girdle_minThree, test[8]) + 1) / (classThree + df['girdle_min'].unique().size)) + np.log(
        (count(girdle_maxThree, test[9]) + 1) / (classThree + df['girdle_max'].unique().size))
    if(test[3]!=0):
        carat_weightOne = (1 / np.sqrt(2 * np.pi * carat_weight_varOne)) * np.exp(
            -((test[3] - carat_weight_meanOne) ** 2) / (2 * carat_weight_varOne))
        carat_weightTwo = (1 / np.sqrt(2 * np.pi * carat_weight_varTwo)) * np.exp(
            -((test[3] - carat_weight_meanTwo) ** 2) / (2 * carat_weight_varTwo))
        carat_weightThree = (1 / np.sqrt(2 * np.pi * carat_weight_varThree)) * np.exp(
            -((test[3] - carat_weight_meanThree) ** 2) / (2 * carat_weight_varThree))
        prob1 += np.log(carat_weightOne)
        prob2 +=np.log(carat_weightTwo)
        prob3 +=np.log(carat_weightThree)
    if (test[7] != 0):
        meas_widthOne = (1 / np.sqrt(2 * np.pi * meas_width_varOne)) * np.exp(
            -((test[7] - meas_width_meanOne) ** 2) / (2 * meas_width_varOne))
        meas_widthTwo = (1 / np.sqrt(2 * np.pi * meas_width_varTwo)) * np.exp(
            -((test[7] - meas_width_meanTwo) ** 2) / (2 * meas_width_varTwo))
        meas_widthThree = (1 / np.sqrt(2 * np.pi * meas_width_varThree)) * np.exp(
            -((test[7] - meas_width_meanThree) ** 2) / (2 * meas_width_varThree))
        prob1 +=  np.log(meas_widthOne)
        prob2 += np.log(meas_widthTwo)
        prob3 +=  np.log(meas_widthThree)

    probs = [prob1, prob2, prob3]
    chosen_prob = np.max(probs)

    if chosen_prob == prob1:
        df2.at[num,'chosen_class']='a'
        if test[10] <= a:
            accuracy += 1
    elif chosen_prob == prob2:
        df2.at[num,'chosen_class']='b'
        if b >= test[10] > a:
            accuracy += 1
    elif chosen_prob == prob3:
        df2.at[num,'chosen_class']='c'
        if c >= test[10] > b:
            accuracy += 1

    test_size+=1.0
    print("Accuracy " + str(accuracy / test_size))
    print(df2['chosen_class'].loc[num])
    print(test[12])

    num += 1
df.to_csv('trainset16.csv')
df2.to_csv('testset16.csv')
