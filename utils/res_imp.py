import pandas as pd
import openai
import random

openai.api_key = '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'

def res_imp(df, restaurant, search_term = ''):
    results = []
    reviews = df.loc[df['name'] == restaurant]['reviewText']
    
    for review in reviews:
        # write a prompt for GPT-3 to extract aspect-opinion pairs with 3 examples
        prompt = "Extract aspect opinion pairs from the text and format output in JSON:\n"\
                    + "\nThis product is good but the battery doesn't last. Does what it should do. It's lightweight and very easy to use. Well worth the money. " \
                    + "[{\"aspect\": \"Overall satisfaction\", \"opinion\": \"product is good\", \"sentiment\": \"positive\"}, "\
                    + "{\"aspect\": \"Battery\", \"opinion\": \"battery doesn't last\", \"sentiment\": \"negative\"}, "\
                    + "{\"aspect\": \"Performance\", \"opinion\": \"Does what it should do\", \"sentiment\": \"positive\"}, "\
                    + "{\"aspect\": \"Weight\", \"opinion\": \"lightweight\", \"sentiment\": \"positive\"}, "\
                    + "{\"aspect\": \"Usability\", \"opinion\": \"very easy to use\", \"sentiment\": \"positive\"}, "\
                    + "{\"aspect\": \"Value for money\", \"opinion\": \"Well worth the money\", \"sentiment\": \"positive\"}]\n"\
                    + "A bit expensive but lunch specials are reasonable. Excellent food. "\
                    + "[{\"aspect\": \"Price\", \"opinion\": \"A bit expensive\", \"sentiment\": \"negative\"}, "\
                    + "{\"aspect\": \"Lunch specials\", \"opinion\": \"reasonable\", \"sentiment\": \"positive\"}, "\
                    + "{\"aspect\": \"Food\", \"opinion\": \"Excellent food\", \"sentiment\": \"positive\"}]\n"\
                    + "A casual friendly spot that where nothing short of exceptional goes into their homemade extracts and sauces. " \
                    + "The breakfast egg sandwich on a rosemary croissant and aioli dressing is a favorite. " \
                    + "[{\"aspect\": \"Ambiance\", \"opinion\": \"casual friendly spot\", \"sentiment\": \"positive\"}, "\
                    + "{\"aspect\": \"Homemade extracts and sauces\", \"opinion\": \"nothing short of exceptional\", \"sentiment\": \"positive\"}, "\
                    + "{\"aspect\": \"Breakfast egg sandwich\", \"opinion\": \"favourite\", \"sentiment\": \"positive\"}, "\
                    + "{\"aspect\": \"Rosemary croissant\", \"opinion\": \"favourite\", \"sentiment\": \"positive\"}, "\
                    + "{\"aspect\": \"Aioli dressing\", \"opinion\": \"favourite\", \"sentiment\": \"positive\"}]\n"\
                    + "Delicious burgers as you'd expect from other Umani Burger locations. Long lines nextdoor from CREAM overflow in front of the building while dining. "\
                    + "[{\"aspect\": \"Burgers\", \"opinion\": \"Delicious\", \"sentiment\": \"positive\"}, "\
                    + "{\"aspect\": \"Ambiance\", \"opinion\": \"long lines nextdoor from CREAM overflow in front of the building while dining\", \"sentiment\": \"negative\"}]\n"\
                    + "##\n"\
                    + '\"' + review + '\"'

        # get the response from GPT-3
        response = openai.Completion.create(
            model = 'text-davinci-002',
            prompt = prompt,
            temperature = 0,
            max_tokens = 256,
            top_p = 1,
            frequency_penalty = 0,
            presence_penalty = 0
        )
        # format json and add to result list
        result = eval(response.choices[0].text)
        results.append(result)
    
    # flatten list
    output = [item for result in results for item in result]

    # make dataframe
    output = pd.DataFrame(output)

    # one-hot encode sentiment column
    output = pd.concat([output, pd.get_dummies(output.sentiment)], axis=1).drop('sentiment', axis=1)

    # aggregate sentiments by aspect
    aggregate = output.groupby("aspect").agg(
        positive = ('positive', sum),
        neutral = ('neutral', sum),
        negative = ('negative', sum)
    )

    if search_term in ['', 'no', 'none']:    
        # print opinions counts for restaurant
        total_pos = sum(aggregate['positive'])
        total_neu = sum(aggregate['neutral'])
        total_neg = sum(aggregate['negative'])
        print('Sense-R: There were ' + str(total_pos) + ' positive opinions, ' + str(total_neu) + ' neutral opinions, and ' + str(total_neg) + ' negative opinions about ' + restaurant)
        # give random positive, neutral and negative comment
        opins_pos = output.loc[output['positive'] == 1, ['aspect', 'opinion']].iloc[[random.randint(0, total_pos-1)]]
        opins_neu = output.loc[output['neutral'] == 1, ['aspect', 'opinion']].iloc[[random.randint(0, total_neu-1)]]
        opins_neg = output.loc[output['negative'] == 1, ['aspect', 'opinion']].iloc[[random.randint(0, total_neg-1)]]
        asp_pos, opin_pos = opins_pos['aspect'].iloc[0], opins_pos['opinion'].iloc[0]
        asp_neu, opin_neu = opins_neu['aspect'].iloc[0], opins_neu['opinion'].iloc[0]
        asp_neg, opin_neg = opins_neg['aspect'].iloc[0], opins_neg['opinion'].iloc[0]
        print('Sense-R: There was a positive comment about ' + asp_pos + ' saying: ' + opin_pos + ',')
        print('a neutral comment about ' + asp_neu + ' saying: ' + opin_neu + ',')
        print('and a negative comment about ' + asp_neg + ' saying: ' + opin_neg + '.')
    else:
        # give summary of opinions
        search_result = aggregate.loc[aggregate.index == search_term]
        pos = str(search_result['positive'][0])
        neu = str(search_result['neutral'][0])
        neg = str(search_result['negative'][0])
        print('Sense-R: There were ' + pos + ' positive opinions, ' + neu + ' neutral opinions, and ' + neg + ' negative opinions about ' + search_term)
        # give the opinions
        opins = output.loc[output['aspect'] == search_term]['opinion']
        print('Sense-R: The opinions for ' + search_term + ' were:')
        num = 1
        for opin in opins:
            if num == len(opins):
                print(str(num) +") " + opin)
            else:
                print(str(num) +") " + opin + '\n')
                num += 1