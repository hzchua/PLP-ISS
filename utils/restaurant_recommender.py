import pandas as pd
import pickle

def res_rec(userID="110357160800000000000"):
  ## Use default userID
  df1=pd.read_excel('data/df1.xlsx')
  bestSent=df1.sort_values(['rating'],ascending=False).groupby(['gPlusUserId'],as_index=False)[['gPlusUserId','name','rating']].first()
  restaurantChosen=bestSent["name"].iloc[bestSent.index[bestSent["gPlusUserId"]== userID]].values[0]
  restaurant_index = mapping[restaurantChosen]
  #get similarity values with other restaurants
  #similarity_score is the list of index and similarity matrix
  similarity_score = list(enumerate(similarity_matrix[restaurant_index]))
  #sort in descending order the similarity score of movie inputted with all the other movies
  similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)
  # Get the scores of the 15 most similar movies. Ignore the first movie.
  similarity_score = similarity_score[1:4]
  #return movie names using the mapping series
  restaurant_index = [i[0] for i in similarity_score]
  top3_recommendation =  (pd.DataFrame({'Name':df1['name'].iloc[restaurant_index],'Categories':df1['categories'].iloc[restaurant_index]}))
  return top3_recommendation

if __name__ == "__main__":
  with open('models/similarityMatrix.pkl') as pickle_file:
    content = pickle.load(pickle_file)
  # df1 = pickle.load('models/similarityMatrix.pkl')
  # df1.to_excel("similarityMatrix.xlsx", index=False)
  print(content)