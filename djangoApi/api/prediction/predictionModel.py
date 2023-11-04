import pickle

def predictLoanType(predictionData):

    loaded_model = pickle.load(open('C:\\Users\\lenovo\\OneDrive\\lenovo g3 backup\\Desktop\\Kaiburr\\djangoApi\\api\\prediction\\consumer_complaint_model', 'rb'))
    fitted_Vectorizer = pickle.load(open('C:\\Users\\lenovo\\OneDrive\\lenovo g3 backup\\Desktop\\Kaiburr\\djangoApi\\api\\prediction\\fittedVectorizer', 'rb'))
    pred = loaded_model.predict(fitted_Vectorizer.transform([predictionData]))
    return pred


