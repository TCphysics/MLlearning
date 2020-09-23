from sklearn.linear_model import LogisticRegression
import numpy as np
import re
# Model = LogisticRegression()
# Model.fit(X_train, y_train)
# Model.score(X_train,y_train)
# # Equation coefficient and Intercept
# Print(‘Coefficient’,model.coef_)
# Print(‘Intercept’,model.intercept_)
# # Predict Output
# Predicted = Model.predict(x_test)

fileName = 'cleveland.data'
fileData = open(fileName,encoding='latin-1').read()
regEx = re.compile('\\W+')
wordList = regEx.split(fileData)
# print(len(wordList))
# quit()
print(wordList[:100])
quit()
patientData = []
temp = []
for w in list(fileData):
    if w == 'name':
       patientData.append(temp)
       temp = []
    elif w == ' ':
        continue
    else:
        temp.append(int(w))

patientData = np.array(patientData)
print(patientData.shape) 
