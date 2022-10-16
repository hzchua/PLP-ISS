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
....................
....................
....................
return sentiment_scores
```

### restaurant_recommender.py

The script should contain a function for main.py to call upon, e.g.:
```
def res_rec():  
....................
....................
....................
return top3_recommendation
```

### Installation

**Step 1: Get the repository**

Using git clone 
```
git clone <Github Repo URL>
```
**Step 2: Create a Virtual Environment**

Enter folder and create a new environment to sandbox your developmental workspace (with Command Prompt)
```
py -3.7 -m venv "YOUR_ENV_NAME"
"YOUT_ENV_NAME"\Scripts\activate
```

**Step 3: Download the Models**

**Download pretrained & trained models and replace the pipeline_models folder**: <link>

The folder structure will look like this:
```
models/


```

**Step 4: Install Dependencies**

Install dependencies via the following command
```
pip install -r requirements.txt
```
