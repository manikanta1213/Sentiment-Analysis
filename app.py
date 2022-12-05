from flask import Flask, render_template, request
import pickle

app = Flask(__name__, template_folder='templates')

def predict_sentiment_emotion(text):
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw,maxlen=200)
    model = pickle.load(open('model.pkl','rb'))
    prediction = int(model.predict(tw).round().item())
    if prediction==0:
        return "User is Happy About the Flight Journey"
    else:
        return "User is unhappy about the flight journey"



@app.route("/", methods=['GET','POST'])
def predict():
    output=""
    if request.method == "POST":
       sentence = request.form.get("review")
       output = predict_sentiment_emotion(sentence)
    #    return "<h1>"+str(output)+"<h1>"
    return render_template("main.html", data=output)


if __name__ == "main":
    app.run(debug=True)