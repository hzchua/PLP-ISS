from utils.data import src_data
import pandas as pd
from sentence_transformers import SentenceTransformer
# import transformers
from sklearn.metrics.pairwise import cosine_similarity
# transformers.logging.set_verbosity_error()

print("Sense-R is waking up.....")

############# Parameters & Initialize Model ###################
sent_model = SentenceTransformer('all-distilroberta-v1')
# intent_model = XXXX
_, res_list = src_data()   #unique list of restaurant names from dataset

################# Functions #######################################

def check_restaurant(res_name, res_list, sent_model):
    res_name = input("Sense-R: May I know which restaurant are you looking at?\nUser: ")
    
    res_name = res_name.strip()
    res_name_enc = sent_model.encode(res_name)
    res_list_enc = sent_model.encode(res_list)
    
    # print(res_name_enc)
    # print(res_list_enc)
        
    sim_score = cosine_similarity([res_name_enc],res_list_enc).tolist()
    max_value = max(sim_score[0])
    # print(max_value)
    df= pd.DataFrame({"Restaurant":res_list, "Score":sim_score[0]})
    df.sort_values(by=["Score"], ascending=False, inplace=True)
    top_5 = df.head(5)["Restaurant"].tolist()
    restaurant = res_list[sim_score[0].index(max_value)]
    
    # print(top_5)
    # df.to_excel("score.xlsx")
    # print(restaurant)
    
    confirmation = input("Sense-R: Are you referring to {}. Pls indicate Y/N.\nUser: ".format(restaurant))
    while confirmation.lower() not in ["y", "n", "yes", "no", "nope", "yeah", "yeap"]:
        confirmation = input("Sense-R: Invalid Response. Pls indicate Y/N.\nUser: ")    
    if confirmation.lower() in ["n", "no", "nope"]:
        print("Sense-R: Apologies, we are unable to find a match in our database.")
        print("Sense-R: Below is our top 5 closet match for your reference:") 
        print("1) {}\n2) {}\n3) {}\n4) {}\n5) {}".format(top_5[0], top_5[1], top_5[2], top_5[3], top_5[4]))
        print("Sense-R: Is the restaurant you are looking for in the list?")
        user_answer = input("Sense-R: If so, pls input the number that correspond to your choice, else input 'NA'.\nUser: ")
        if user_answer.lower().strip() in ["na", "no","n", "n/a"]:
            print("Sense-R: Apologies, the restaurant you are searching for is not in our database.")  
            restaurant = None
        elif int(user_answer.strip()) in range(1,6):
            restaurant = top_5[int(user_answer.strip())-1]
            print("Sense-R: You have chosen {}.".format(restaurant))
        else:
            restaurant = None
            
    return restaurant


##################################################

print("Sense-R: Good day, Sense-R at your service.")
print("Sense-R: If you wish to exit the chat, pls key in 'exit'")
user_input = input("Sense-R: How may I help you?\nUser: ")

while user_input != "exit":
    ## intent = model(user_input)
    ## intent in [restaurant improvements, customer sentiments, restaurant recommender, exception]
    intent = user_input
    res_name = "Macs"
    
    count = 1
    while intent == "exception":
        user_input = input("Senser-R: Apologies, could you rephrase your question?\nUser: ")
        ## intent = model(user_input)
        count += 1
        if count == 3:
            print("Sense-R: Pardon my incapability. I am ashamed as I am unable to assist you.")
            user_input = "exit"
            break
    
    if intent == "restaurant improvements":
        restaurant = check_restaurant(res_name, res_list, sent_model)
        # output = res_imp(restaurant)   ## function for restaurant improvement
        if restaurant != None:
            print("Sense-r: Areas of improvements include:")
            print("1) Price\n2) Ambience\n3) Service")
        user_input = input("Sense-R: Are there any other things that I can help you with?\nUser: ")
        if user_input.lower() in ["n", "no", "nope"]:
            user_input = "exit"
        
    elif intent == "customer sentiments":
        restaurant = check_restaurant(res_name, res_list, sent_model)
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
     
     
print("Sense-R: Thank you and goodbye.")


