# PLP-ISS (Intelligent System Sense-R)

The models folder should look like this:
```
models/
  restaurant_improvements.pkl
  restaurant_sentiments.pkl
  restaurant_recommender.pkl
  
```

The utils folder should look like this:
```
utils/
  restaurant_improvements.py
  restaurant_sentiments.py
  restaurant_recommender.py
```

main.py will contains the chatbot codes to call the individual model / functions as and when triggered by the users.

### restaurant_improvements.py

The script should contain a function for main.py to call upon, e.g.:
```
def res_imp(restaurant_name):  
.................... 
.................... 
.................... 
return improvement_list
```
### restaurant_sentiments.py

The script should contain a function for main.py to call upon, e.g.:
```
def res_sent(restaurant_name):  
.................... <br>
.................... <br>
.................... <br>
return sentiment_scores
```

### restaurant_recommender.py

The script should contain a function for main.py to call upon, e.g.:
```
def res_rec():  
.................... <br>
.................... <br>
.................... <br>
return top3_recommendation
```

  
