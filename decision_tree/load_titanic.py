# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import pickle

titanicDF = pd.read_csv('./data/train.csv')
test  = pd.read_csv('./data/test.csv')
# titanicDF = titanicDF.append( test , ignore_index = True )

sexList = list(titanicDF.Sex)

pclassList = list(titanicDF.Pclass)

ageList = pd.DataFrame()
ageList = titanicDF.Age.fillna( int(titanicDF.Age.mean()) )
ageList = list(ageList)
ageList = [i//10*10 for i in ageList]

fareList = pd.DataFrame()
fareList = titanicDF.Age.fillna( titanicDF.Age.mean() )
fareList = list(fareList)
fareList = [i//10*10 for i in fareList]

titleList = pd.DataFrame()
titleList['Title'] = titanicDF['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())
Title_Dictionary = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"

                    }
titleList[ 'Title' ] = titleList.Title.map( Title_Dictionary )
titleList = titleList.Title

cabin = pd.DataFrame()
cabin['Cabin'] = titanicDF.Cabin.fillna( 'U' )
cabin['Cabin'] = cabin[ 'Cabin' ].map( lambda c : c[0] )
cabinList = list(cabin.Cabin)

family = pd.DataFrame()
family['FamilySize'] = titanicDF['Parch']+ titanicDF['SibSp'] + 1
familySizeList = list(family.FamilySize)

embarked = pd.DataFrame()
embarked['Embarked'] = titanicDF.Embarked.fillna('C')
embarkedList = list(embarked.Embarked)

featureName= ['Sex', 'Pclass', 'Age', 'Fare', 'Title', 'Cabin', 'FamilySize', 'Embarked']
featureList = []
dataListRaw = [sexList, pclassList, ageList, fareList, titleList,\
 cabinList, familySizeList, embarkedList]

for i in range(len(featureName)):
    LSet = sorted(list(set(dataListRaw[i])))
    featureList.append([featureName[i],LSet])

survivalList = list(titanicDF.Survived)
dataSize = len(survivalList)
dataList = []
for j in range(dataSize):
    row = []
    for i in range(len(featureName)):
        row.append(featureList[i][1].index(dataListRaw[i][j]))
    row.append(survivalList[j])
    # print(row)
    dataList.append(row)


with open('Titanic.data', 'wb') as filehandle:
    pickle.dump(dataList, filehandle)

with open('Titanic_feature_list.data', 'wb') as filehandle:
    pickle.dump(featureList, filehandle)


# print(type(dList))

# if __name__ == '__main__':







































