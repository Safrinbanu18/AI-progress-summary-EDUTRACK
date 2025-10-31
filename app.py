from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression
import google.generativeai as genai

app = Flask(__name__)

# Configure Gemini API key
genai.configure(api_key="AIzaSyCBAJunapsyRhFFNMNSJhO419DTS0UUvik")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get JSON data from frontend
        data = request.json
        activities = data['activities']

        # Create DataFrame for 7 days
        df = pd.DataFrame({
            'Day': [f'Day {i+1}' for i in range(7)],
            'Score': activities
        })

        # Train ML model (predict trend)
        X = [[i] for i in range(7)]
        y = activities
        model = LinearRegression()
        model.fit(X, y)
        prediction = model.predict([[7]])[0]

        # Use Gemini to summarize
        avg_score = sum(activities) / len(activities)
        prompt = f"""
        You are an educational AI coach.
        Here is a student's 7-day productivity data: {activities}.
        Average score: {avg_score:.2f}
        Predicted next day score: {prediction:.2f}.
        Write a short, motivating summary describing the student's performance trend,
        strengths, and one key improvement suggestion.
        """

        # Use Gemini 2.5 model for summarization
        model_gemini = genai.GenerativeModel(model_name="gemini-2.5-flash")
        response = model_gemini.generate_content(prompt)
        summary = response.text

        return jsonify({
            'summary': summary,
            'activities': activities
        })  

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
