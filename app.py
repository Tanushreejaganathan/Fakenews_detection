from flask import Flask, render_template, request, session, redirect, url_for
from groq import Groq
import json

app = Flask(__name__)
app.secret_key = "super_secret_key"

# 🔑 Replace with your real Groq API key
client = Groq(api_key="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")


def get_ai_judgement(text: str):
    prompt = f"""
You are a fake news detection expert.
Analyze the news text and decide whether it is REAL or FAKE.

Respond ONLY in valid JSON (no extra text):
{{
  "prediction": "Real" or "Fake",
  "probability": number between 0 and 1,
  "explanation": "Detailed explanation of why it is Real or Fake."
}}

Text: "{text}"
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # ✅ VALID MODEL
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("⚠️ Groq API error:", e)
        return None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/explain", methods=["POST"])
def explain():
    text = request.form["news_text"]

    ai_response = get_ai_judgement(text)

    if ai_response:
        try:
            ai_result = json.loads(ai_response)
            prediction_text = "Real News" if ai_result["prediction"].lower() == "real" else "Fake News"
            final_prob = round(float(ai_result.get("probability", 0)), 2)
            explanation = ai_result["explanation"]
        except Exception:
            prediction_text = "Error"
            final_prob = 0.0
            explanation = "Invalid AI response."
    else:
        prediction_text = "Error"
        final_prob = 0.0
        explanation = "No response from AI."

    # ✅ Always include all 4 keys — even for errors
    history_item = {
        "text": text,
        "prediction": prediction_text,
        "probability": final_prob,        # always a float
        "explanation": explanation
    }
    session.setdefault("history", []).insert(0, history_item)
    session.modified = True

    return render_template(
        "result.html",
        news_text=text,
        prediction=prediction_text,
        probability=final_prob,
        explanation=explanation,
        history=session.get("history", [])
    )


@app.route("/history/<int:index>")
def view_history(index):
    history = session.get("history", [])
    if 0 <= index < len(history):
        item = history[index]
        # ✅ Pass full_text as news_text
        return render_template(
            "result.html",
            news_text=item["text"],
            prediction=item["prediction"],
            probability=item["probability"],
            explanation=item["explanation"],
            history=history
        )
    return redirect(url_for("home"))


@app.post("/clear_history")
def clear_history():
    session.pop("history", None)
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)