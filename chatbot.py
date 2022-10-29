import torch
from utils.data import src_data, check_restaurant
from sentence_transformers import SentenceTransformer
from utils.mod_func import get_response, BERT_Arch, get_intent

print("Sense-R is waking up.....")

############# Parameters & Initialize Model ###################
datafile = 'data/df_CA.json'
conv_model_path = "models/conv_model.pt"
int_model_path = "models/int_model.pt"

############# Chatbot model ##################################
loaded_int_model = torch.load(int_model_path)
loaded_conv_model = torch.load(conv_model_path)

sent_model = SentenceTransformer('all-distilroberta-v1')

_, res_list = src_data(datafile)   #unique list of restaurant names from dataset

###############################################################

print("Sense-R: Good day, Sense-R at your service.")
print("Sense-R: If you wish to exit the chat, pls key in 'exit'")
user_input = input("Sense-R: How may I help you?\nUser: ")

while user_input != "exit":
    intent = get_intent(user_input, loaded_int_model)
    
    if intent == "restaurant improvements":
        restaurant = check_restaurant(res_list, sent_model)
        # output = res_imp(restaurant)   ## function for restaurant improvement
        if restaurant != None:
            print("Sense-r: Areas of improvements include:")
            print("1) Price\n2) Ambience\n3) Service")
        user_input = input("Sense-R: Are there any other things that I can help you with?\nUser: ")
        if user_input.lower() in ["n", "no", "nope"]:
            user_input = "exit"
        
    elif intent == "customer sentiments":
        restaurant = check_restaurant(res_list, sent_model)
        # output = res_sent(res_name)
        # print(output)
        if restaurant != None:
            print("Sense-r: Sentiments of customers towards this restaurant is:")
            print("Happy (60%), Upset (30%), Angry (10%)")
        user_input = input("Sense-R: Are there any other things that I can help you with?\nUser: ")
        if user_input.lower() in ["n", "no", "nope"]:
            user_input = "exit"
        
    elif intent == "restaurant recommender":
        # output = res_rec()
        # print(output)
        print("Sense-r: Top Recommendations include:")
        print("1) Revolution Cafe\n2) Claim Jumper\n3) Black Bear Diner")
        user_input = input("Sense-R: Are there any other things that I can help you with?\nUser: ")
        if user_input.lower() in ["n", "no", "nope"]:
            user_input = "exit"
    
    elif intent == "conversation":
        # bot_out = intent_output(conv_model, user_input, conversation_labels, 0.2)
        bot_out = get_response(user_input, loaded_conv_model)
        print("Sense-R: {}".format(bot_out))
        user_input = input("User: ")
     
     
print("Sense-R: Thank you and goodbye.")


