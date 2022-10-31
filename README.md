# PLP-ISS (Intelligent System Sense-R) - Restaurant Virtual Assistant

## **SECTION 1: INTRODUCTION**

Most restaurants are lacking an intelligent system that can analyse customer feedback and reviews to understand what the restaurant is still lacking in or what areas are necessary to be improved upon, in order to increase the customer retention rate. And on the customer side, they are also lacking an intelligent recommender system that can recommend restaurant choices to them based on top recommendations by other customers, or individual preferences or based on general sentiments of customers to the restaurant.

Therefore, our team of five decided to resolve these issues by developing an intelligent virtual assistant for both the customer and the restaurant owner who is able to provide reliable restaurant recommendations, general sentiments of customers towards the restaurant, as well as suggestions for restaurant improvements. With that, the team introduce Intelligent Sentiment Sense-R (ISS).



## **SECTION 2: OBJECTIVES** 

The objective of the project is to create an intelligent system to recommend restaurants owners the different aspects of restaurants that require improvement, to take on the role of a restaurant recommender for customers and provide general sentiments of customers. The system will include:

•	Aspect-Based Sentiment Analysis to identify the different areas of the restaurants, e.g., food quality or ambience, that requires further improvements

•	General Sentiment Analysis to extract the general customers’ sentiments towards a specific restaurant

•	Recommender System to produce the top recommended restaurants for customers to choose from

•	And lastly, a chatbot or virtual assistant to incorporate the above-mentioned capabilities into a single platform



## **SECTION 3: CREDITS / PROJECT CONTRIBUTION**




## **SECTION 4: PROJECT DESCRIPTION**

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




## **SECTION 5: Installation**


**Step 1: Get the repository**

Using git clone 
```
git clone <Github Repo URL>
```
**Step 2: Create a Virtual Environment**

Enter folder and create a new environment to sandbox your developmental workspace (with Command Prompt)
```
cd <Github Repo Folder>           (Change directory to Github Repo Folder directory)
py -3.7 -m venv "YOUR_ENV_NAME"   (Create virtual environment)
"YOUR_ENV_NAME"\Scripts\activate  (Avtivate virtual environment)
```

**Step 3: Download the Models**

**Download pretrained & trained models and replace the pipeline_models folder**: https://drive.google.com/drive/folders/1u3bPZd1WBmtSgn_pE4fM2Grdih0Uz506?usp=share_link

The folder structure will look like this:
```
models/
    conv_model.pt
    int_model.pt
```

**Step 4: Install Dependencies**

Install dependencies via the following command
```
pip install -r requirements.txt
```




## **SECTION 6: User Guide**



**Step 1: Open Terminal**

```
Open Command Prompt
```

**Step 2: Change directory to Project Folder**

```
cd <Github Repo Folder>
```

**Step 3: Activate virtual environment** 

"YOUR_ENV_NAME" == Name of the virtual environment

```
"YOUR_ENV_NAME"\Scripts\activate**
```

**Step 4: Enter Command**

```
python chatbot.py
```
