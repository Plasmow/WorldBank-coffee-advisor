from ai.translate import *

threshold = 0.8

def analyze_from_query(text:str, clarify_answer=None):
    query_eng = lug_to_en(text)
    
    classifier_prediction = class_predict(query_eng)
    llm_prediction = llm_predict(query_eng)
    
    if classifier_prediction["probability"] >= threshold and llm_prediction["class"] == classifier_prediction["class"]:
        pass
    else:
        pass
        
    return {
        "lang":"",
        "text_en":"",
        "label":"", # healthy, leaf_rust, phoma, other
        "proba":"",
        "llm_label":"",
        "decision":"", # answer, clarify, escalate
        "template_id":"", # Healthy, Rust, Phoma, clarify, unsure
        "reason":""
    }