import pandas as pd
from fuzzywuzzy import process
from gensim.models import Word2Vec
model = Word2Vec.load("models/word2vec.model")

def res_imp(restaurant, search_term = ''):
    # read absa results
    result = pd.read_json('data/result.json')
    
    # confirm restaurant name if restaurant name doesn't match
    name = process.extractOne(restaurant, set(result['name']))
    if name[1] < 95:
        ratios = process.extract(name[0], set(result['name']))
        print('We found the following restaurants: \n')
        print([x[0] for x in ratios])
        restaurant = input("Sense-R: Which restaurant are you referring to?\nUser: ")
        name = process.extractOne(restaurant, set(result['name']))
    
    print("For " + str(name[0]) + ":")
    
    # get on results for restaurant
    restaurant_result = result.loc[result['name'] == restaurant]
    # unpack lists in dataframe
    output = restaurant_result.explode(['aspect', 'sentiment', 'confidence'])

    # aggregate sentiments by restaurant
    pos = (output['sentiment']=='Positive').sum()
    neu = (output['sentiment']=='Neutral').sum()
    neg = (output['sentiment']=='Negative').sum()

    if search_term in ['', 'no', 'none']:    
        # print opinions counts for restaurant
        pos = (output['sentiment']=='Positive').sum()
        neu = (output['sentiment']=='Neutral').sum()
        neg = (output['sentiment']=='Negative').sum()
        print('There were ' + str(pos) + ' positive opinions, ' + str(neu) + ' neutral opinions, and ' + str(neg) + ' negative opinions about ' + restaurant)
    else:
        # one-hot encode sentiment column
        output = pd.concat([output, pd.get_dummies(output.sentiment)], axis=1).drop('sentiment', axis=1)
        # aggregate sentiments by aspect
        if 'Neutral' in output.columns:
            aggregate = output.groupby("aspect").agg(
                positive = ('Positive', sum),
                neutral = ('Neutral', sum),
                negative = ('Negative', sum)
            )
        else:
            aggregate = output.groupby("aspect").agg(
                positive = ('Positive', sum),
                negative = ('Negative', sum)
            )
        # find top 3 closest aspect 
        cos_sim = {}
        for index in aggregate.index:
            score = []
            for s_word in search_term.split():
                s_word = process.extractOne(s_word, model.wv.index_to_key)
                for word in index.split():
                    word = process.extractOne(word, model.wv.index_to_key)
                    score.append(model.wv.similarity(s_word[0], word[0]))
            cos_sim[index] = sum(score) / len(score)
        closest_match = sorted(cos_sim, key=cos_sim.get, reverse=True)[:3]
        
        for search_term in closest_match:
            # give summary of opinions
            search_result = aggregate.loc[aggregate.index == search_term]
            pos = str(search_result['positive'][0])
            neu = str(0)
            if 'neutral' in search_result.columns:
                neu = str(search_result['neutral'][0])
            neg = str(search_result['negative'][0])
            print('There were ' + pos + ' positive opinions, ' + neu + ' neutral opinions, and ' + neg + ' negative opinions about ' + search_term)
