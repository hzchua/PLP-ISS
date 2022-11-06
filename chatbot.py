import torch
from utils.data import src_data, check_restaurant
from sentence_transformers import SentenceTransformer
from utils.mod_func import get_response, BERT_Arch, get_intent
from utils.res_imp import res_imp
from utils.restaurant_sentiments import res_sent
from utils.restaurant_recommender import res_rec

print("Sense-R is waking up.....")

############# Parameters & Initialize Model ###################
datafile = 'data/df_CA.json'
conv_model_path = "models/conv_model.pt"
int_model_path = "models/int_model.pt"

############# Chatbot model ##################################
loaded_int_model = torch.load(int_model_path)
loaded_conv_model = torch.load(conv_model_path)

sent_model = SentenceTransformer('all-distilroberta-v1')

df, res_list = src_data(datafile)   #unique list of restaurant names from dataset

###############################################################

print("Sense-R: Good day, Sense-R at your service.")
print("Sense-R: If you wish to exit the chat, pls key in 'exit'")
user_input = input("Sense-R: How may I help you?\nUser: ")

while user_input != "exit":
    intent = get_intent(user_input, loaded_int_model)
    
    if intent == "restaurant improvements":
        restaurant = check_restaurant(res_list, sent_model)
        search_term = input("Sense-R: Is there any aspect that you are interested in?\nUser: ")
        print("Sense-R is looking through customer reviews.... Pls wait a while....")
        output = res_imp(restaurant, search_term)   ## function for restaurant improvement
        user_input = input("Sense-R: Are there any other things that I can help you with?\nUser: ")
        if user_input.lower() in ["n", "no", "nope"]:
            user_input = "exit"
        
    elif intent == "customer sentiments":
        restaurant = check_restaurant(res_list, sent_model)
        review_input = input("Sense-R: May I know what is your review for this restaurant?\nUser: ")
        output = res_sent(review_input,restaurant)
        print(output)
        user_input = input("Sense-R: Are there any other things that I can help you with?\nUser: ")
        if user_input.lower() in ["n", "no", "nope"]:
            user_input = "exit"
        
    elif intent == "restaurant recommender":
        output = res_rec()
        print(output)
        user_input = input("Sense-R: Are there any other things that I can help you with?\nUser: ")
        if user_input.lower() in ["n", "no", "nope"]:
            user_input = "exit"
    
    elif intent == "conversation":
        bot_out = get_response(user_input, loaded_conv_model)
        print("Sense-R: {}".format(bot_out))
        user_input = input("User: ")
     
     
print("Sense-R: Thank you and goodbye.")


