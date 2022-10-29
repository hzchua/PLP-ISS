import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def src_data(datafile):

    with open(datafile) as json_file:     
        data = json_file.readlines()
        data = list(map(json.loads, data)) 

    df = pd.DataFrame(data)
    res_list = df["name"].unique().tolist()

    return df, res_list


def check_restaurant(res_list, sent_model):
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

if __name__ == "__main__":
    datafile = '../data/df_CA.json'
    src_data(datafile)
