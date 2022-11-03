import numpy as np
import pickle, os
import datetime
import openpyxl
import pandas as pd

def predict(input_new, filename='models/mnb_model.pkl', cv_file='models/cv-model.pkl'):
  classifier = pickle.load(open(filename, 'rb'))
  cv = pickle.load(open((cv_file),'rb'))
  data = input_new
  vect = cv.transform(data).toarray()
  prediction = classifier.predict(vect)

  return prediction

def res_sent(reviews, res_name, newDB = "data/newDB.xlsx"):
  review = [reviews]
  result = predict(review)
  curr_time = datetime.datetime.now()
  df = pd.DataFrame(data={"Restaurant Name": res_name,
                          "Date of review": curr_time, 
                          "Review": reviews,
                          "Sentiments": result})
  
  newDB_file = os.path.join(os.getcwd(), newDB)

  if os.path.exists(newDB_file):
    wb = openpyxl.load_workbook(newDB_file)
    sheet = wb.active
    
    for _, row in df.iterrows():
      sheet.append(row.values.tolist())
        
    wb.save(newDB_file)
    
  else:
    df.to_excel(newDB_file, index=False)

  return "Sense-R: Rating is {}" .format(result[0])


if __name__ == "__main__":
  review = ['A bit expensive but lunch specials are reasonable. Excellent food and very good taste.' ]
  result = res_sent(review)
  print(result)