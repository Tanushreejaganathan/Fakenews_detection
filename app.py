from flask import Flask, render_template, request
import pickle
from groq import Groq

app = Flask(__name__)

# Load your trained model and vectorizer once at startup
with open("model/nb_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("model/nb_vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# Initialize Groq client with your API key
client = Groq(api_key="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")

def get_explanation(text, prediction, probability):
    prompt = f"""
    You are a fake news detection expert.

    Example:
    Text: "Drinking bleach cures COVID-19"
    Prediction: Fake News
    Probability: 0.99
    Explanation: This claim is false and dangerous. Health organizations like WHO and CDC have warned against it. No scientific evidence supports it.

    Now analyze this article:

    Text: "{text}"
    Prediction: {prediction}
    Probability: {probability}

    Explain clearly why it was classified this way:
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful AI that explains fake news predictions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("⚠️ Error from Groq API:", str(e))
        return f"⚠️ Could not generate AI explanation. Based on the model, this was classified as '{prediction}' with probability {probability}."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/explain", methods=["POST"])
def explain():
    text = request.form["news_text"]
    text_vector = vectorizer.transform([text])
    prediction_label = model.predict(text_vector)[0]
    prediction_prob = model.predict_proba(text_vector)[0]

    prob_real = float(prediction_prob[1])
    prob_fake = float(prediction_prob[0])

    if prediction_label == 1:
        prediction_text = "Fake News"
        final_prob = prob_real
    else:
        prediction_text = "Real News"
        final_prob = prob_fake

    explanation = get_explanation(text, prediction_text, round(final_prob, 2))

    return render_template(
        "result.html",
        news_text=text,
        prediction=prediction_text,
        probability=round(final_prob, 2),
        explanation=explanation
    )

if __name__ == "__main__":
    app.run(debug=True)
