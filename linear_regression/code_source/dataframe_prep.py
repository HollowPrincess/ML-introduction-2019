import numpy as np

def get_cols_names(data):
    #get and fit features names
    features_names=np.array(['page_likes_num',#1
                   'page_checkins',#2
                   'page_talking_about',#3
                    'page_cat',#4
                    'page_statistics',#5-29 #mean, avg etc.
                    'comments_num_before_base_time',#30
                    'comments_num_in_last_24_hours',#31 #last day
                    'comments_num_in_last_48_to_24_hours',#32 #day before last
                    'comments_num_in_first_24_hours',#33
                    'comments_difference_in_last_two_days', #34 (32-31)
                    'base_time', #35
                    'character_num_in_post', #36
                    'share_num',#37
                    'post_promotion', #38 binary
                    'h_local', #39 This describes the H hrs, for which we have the target variable/ comments received. 
                    'post_published_weekday', #40-46 This represents the day(Sunday...Saturday) on which the post was published. 
                    'base_ditetime_weekday', #47-53 This represents the day(Sunday...Saturday) on selected base Date/Time. 
                    'target' #54 The no of comments in next H hrs(H is given in Feature no 39).                
                   ])

    for index in range(5,29):
        features_names=np.insert(features_names, index, features_names[4]+'_'+str(index-4))

    weekday=('sunday', 'monday','tuesday', 'wednesday', 'thursday', 'friday', 'saturday')    

    for index in range(40,47):
        features_names=np.insert(features_names,index, features_names[39]+'_'+ weekday[index-40])
    features_names=np.delete(features_names, 39)

    for index in range(47,54):
        features_names=np.insert(features_names,index, features_names[46]+'_'+ weekday[index-47])
    features_names=np.delete(features_names, 46)

    data.columns=features_names
    return data

def drop_cols(data):
    cols=['post_promotion']
    return data.drop(columns=cols)