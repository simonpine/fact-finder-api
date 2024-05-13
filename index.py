from flask import Flask, jsonify, request
import pickle
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins='https://www.factfinder.today')

model = pickle.load(open('./trained_model_XGBoost.pkl', 'rb'))
preprocessor = pickle.load(open('./preprocessor.pkl', 'rb'))




@app.route('/')
def home():
    return 'Is working'

@app.route('/test')
def test():
    dfForTest = pd.DataFrame([[''' Luke Mosseau The Warren County Finance and Budget Committee passed a resolution that will give the county greater flexibility to spend what is left of the $12 million the county received in federal ARPA money. The resolution attempts to make sure that ARPA funds that have gone unused make it back to the county’s general fund. During the COVID-19 Pandemic, Warren County was awarded roughly $12 million from ARPA, the American Rescue Plan Act, according to Warren County Administrator John Taflan. As groups such as local housing and healthcare providers came forward with requests to use that money, the county voted to “obligate” the funds to those groups, meaning under federal rules that “obligated” money is under contract with one of the groups or projects. If the county does not obligate all of the money, or if the group ends up not spending their money by the end of 2026, the federal government could require that the money is returned. The Wednesday May 8 resolution would move the unspent money into the general fund at the end of 2024. “We want to ensure that what’s not being obligated and/or expended by any ARPA recipient has a means to come back and stay here,” County Attorney Larry Elmen said. The county still has $3.3 million from the federal government that is obligated to groups and projects, but has either not yet been distributed to those entities or the entities have gotten the money but have not yet spent it, Taflan said. The resolution that passed this week replaces one approved at the end of last year. That resolution had said that any unspent ARPA money would have gone to mental health and homelessness issues at the end of this year. However, roughly $400,000 has already been obligated for mental health issues and has not yet been spent. “We think there are adequate resources for mental health and wellness,” Taflan said. “What we would like to do is ensure that money that is obligated is going to be spent by asking the people who have received the ARPA funding, if they’re confident enough that they’re going to spend the money by 2026.” The Warren County Planning Department reaches out to organizations like Glens Falls Hospital and Ascend Mental Wellness to review the progress of ARPA expenditure. “On a quarterly basis we reach out to everyone that’s received funding and we get a project status update to understand where they’re at with their project,” said Ethan Gaddy, Warren County planner. “We collect all the receipts from expenditures, have them self-report any obstacles to spending their funds, and then that helps us stay on top of where they are.” “This process has served us well because we have a clearer idea of where people are at with their projects and understanding where people are at with expenditures past 2024,” Gaddy added. If there are no plans to spend that money, Warren County would like to recoup the funds from these organizations—a complicated process since the funds are under contract. Recouping would involve going through the county attorney’s office and the county planning department, as well as coming to an agreement with the ARPA recipient, according to Taflan. The county established an ARPA committee that allocate those funds over a three-year period. The ARPA committee ended at the end of 2023 and all ARPA responsibility fell to the Warren County Finance and Budget Committee. The resolution to transfer unspent or un-obligated ARPA funds to the general fund at the end of 2024 will need to go before the Warren County Board of Supervisors for final approval.''', '''County moves to recalculate how ARPA funding is spent''']],
                  columns=['text', 'title'])
    # def removeStopwordsAndLower(text):
    #     stop_words = set(stopwords.words('english')) 
    #     words = text.lower().split() 
    #     filtered_words = [word for word in words if word not in stop_words] 
    #     return ' '.join(filtered_words)
    def lemaAndStem(text):
        stemmer = SnowballStemmer("english")
        normalized_text = []
        for word in text.split():
            stemmed_word = stemmer.stem(word)
            normalized_text.append(stemmed_word)
        return ' '.join(normalized_text).replace(',', '')
    # dfForTest['text'] = dfForTest['text'].apply(removeStopwordsAndLower)
    # dfForTest['title'] = dfForTest['title'].apply(removeStopwordsAndLower)
    dfForTest['text'] = dfForTest['text'].apply(lemaAndStem)
    dfForTest['title'] = dfForTest['title'].apply(lemaAndStem)

    dfForTest = preprocessor.transform(dfForTest)
    res = model.predict_proba(dfForTest)
    return {'FakePosibility' :str(round(res[0][0], 4)),
            'RealPosibility' :str(round(res[0][1], 4))
            }

@app.route('/predict', methods=["POST"])
def predict():
    data = request.get_json()

    # return({'Res': type(data['text'])})
    # dfForTest = pd.read_json(data)

    title = data['title']
    text = data['text']
    dfForTest = pd.DataFrame([[text, title]],
                  columns=['text', 'title'])
    # return(f"{title}, {text}, {type(title)}")
    # dfForTest = pd.DataFrame([[text, title]],
    #               columns=['text', 'title'])
    def lemaAndStem(text):
        stemmer = SnowballStemmer("english")
        normalized_text = []
        for word in text.split():
            stemmed_word = stemmer.stem(word)
            normalized_text.append(stemmed_word)
        return ' '.join(normalized_text).replace(',', '')

    dfForTest['text'] = dfForTest['text'].apply(lemaAndStem)
    dfForTest['title'] = dfForTest['title'].apply(lemaAndStem)

    dfForTest = preprocessor.transform(dfForTest)
    res = model.predict_proba(dfForTest)

    return {'FakePosibility' :str(round(res[0][0], 4)),
            'RealPosibility' :str(round(res[0][1], 4))
            }