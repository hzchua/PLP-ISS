# PLP-ISS (Intelligent System Sense-R) - Restaurant Virtual Assistant

 <p align="center" width="100%">
    <img width="33%" src="https://cdn.icon-icons.com/icons2/3347/PNG/512/chef_cook_man_avatar_icon_210168.png"> 
</p>
  

## **SECTION 1: INTRODUCTION**

<p>Most restaurants are lacking an intelligent system that can analyse customer feedback and reviews to understand what the restaurant is still lacking in or what areas are necessary to be improved upon, in order to increase the customer retention rate. And on the customer side, they are also lacking an intelligent recommender system that can recommend restaurant choices to them based on top recommendations by other customers, or individual preferences or based on general sentiments of customers to the restaurant.

Therefore, our team of five decided to resolve these issues by developing an intelligent virtual assistant for both the customer and the restaurant owner who is able to provide reliable restaurant recommendations, general sentiments of customers towards the restaurant, as well as suggestions for restaurant improvements. With that, the team introduce Intelligent Sentiment Sense-R (ISS). </p>



## **SECTION 2: OBJECTIVES** 

The objective of the project is to create an intelligent system to recommend restaurants owners the different aspects of restaurants that require improvement, to take on the role of a restaurant recommender for customers and provide general sentiments of customers. The system will include:

•	Aspect-Based Sentiment Analysis to identify the different areas of the restaurants, e.g., food quality or ambience, that requires further improvements

•	General Sentiment Analysis to extract the general customers’ sentiments towards a specific restaurant

•	Recommender System to produce the top recommended restaurants for customers to choose from

•	And lastly, a chatbot or virtual assistant to incorporate the above-mentioned capabilities into a single platform



## **SECTION 3: CREDITS / PROJECT CONTRIBUTION**

| Official Full Name  | Student ID (MTech Applicable) | Work Items (Who Did What)  |
| ------------- |:-------------:| -----|
| CHUA HAO ZI      | A0229960W | 1. Market Research & Literature Review </br> 2. System Architecture Design </br> 3. Chatbot Design and Creation </br> 4. Report Writing </br> 5. Creation of System Demo Video </br> 6. System Integration |
| FRANSISCA THERESIA       | A0213540W | 1. Market Research & Literature Review </br> 2. Text Summarization Modelling  </br> 3. Threshold Filtering using VADER Sentiment Analysis </br> 4. Report Writing |
| LI XIAOXIA | A0229975J  | 1. Market Research & Literature Review </br> 2. Sentiment Polarity Detection Modelling </br> 3. Report Writing |
| TAN WEE HAN | A0125244N | 1. Market Research & Literature Review </br> 2. Improvements Recommender with Aspect-Based Sentiment Analysis (ABSA) </br> 3. Report Writing |
| FAN YUECHEN | A0141009W | 1. Market Research & Literature Review </br> 2. Restaurant Recommendation System </br> 3. Report Writing </br> 4. Creation of System Demo Video|


## **SECTION 4: Installation**


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

**Download pretrained & trained models and replace the models folder**: https://drive.google.com/file/d/1HEo40R1AwKciPEAAmsU-4v7YfWHy-bRI/view?usp=share_link

The folder structure will look like this:
```
models/
    conv_model.pt
    cv-model.pkl
    int_model.pt
    mappingTable.pkl
    mnb_model.pkl
    similiarityMatrix.npy
    word2vec.model
```

**Step 4: Install Dependencies**

Install dependencies via the following command
```
pip install -r requirements.txt
```


## **SECTION 5: User Guide**


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
