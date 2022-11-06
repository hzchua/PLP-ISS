import numpy as np
import pickle


# Loading the Multinomial Naive Bayes model and CountVectorizer object 
filename = 'mnb_model_v3.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('cv-model_v3.pkl','rb'))

def predict(input_new):
  data = input_new
  vect = cv.transform(data).toarray()
  prediction = classifier.predict(vect)
  return prediction

#input_new = ['food and taste are really bad.' ] # free text input
#input_new = ['A bit expensive but lunch specials are reasonable. Excellent food and very good taste.' ] # Positive (rating as 5) review from project dataset  - predict as Positive
#input_new = ['You can\'t go wrong at this place -the whole menu rocks!' ] # Positive (rating as 5) review from project dataset  - predict as Negative
#input_new = ['This place doesn\'t respect reservations. It\'s loud, over priced and will not make you full.'] # Negative(rating as 1) review from project dataset - predict as Positive
#input_new = ['Don\t waste your time or your money! This restaurant has forgotten everything that made them what they were.  Poor to slow service. Overpriced everything. Management that is so bad I am at a loss for words.  Plenty of other options in Temecula, choose one of them. '] # Negative(rating as 1) review from project dataset - predict as Negative
#input_new_neg = ['This place used to be good but now it is disgusting and dirty stay away unless you want to get mugged or buy crack on the way out.  This is a very dangerous block.  you will never find parking anyway.  Go to Chutneys across the st.  At least it is clean and they never made me sick'] # Negative(rating as 1) review from project dataset - predict as Negative

#load and model vectorization 
input_new = ['This place doesn\'t respect reservations. It\'s loud, over priced and will not make you full.']

result = predict(input_new )
print(result)