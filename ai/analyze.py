from translate import *

threshold = 0.8

def analyze_from_query(text:str):
    query_eng = lug_to_en(query_eng)
    
    classifier_prediction = class_predict(query_eng)
    llm_prediction = llm_predict(query_eng)
    
    if classifier_prediction["probability"] >= threshold and llm_prediction["class"] == classifier_prediction["class"]:
        pass
    else:
        pass
        
    return