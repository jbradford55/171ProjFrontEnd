from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# loading model
model = joblib.load("mi_best_model.pkl")

FEATURE_NAMES = [
    'ca_0.0', 'thal_7.0', 'cp_4.0', 'asymptomatic_chest_pain',
    'age_sex', 'combined_risk_score', 'oldpeak', 'thalach',
    'thalach_exang', 'thal_3.0'
]

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            # grabbing form data
            ca_0_0 = float(request.form.get("ca_0_0", 0.0))
            
            # Handle radio button values
            thal_7_0 = float(request.form.get("thal_7_0", 0.0))
            cp_4_0 = float(request.form.get("cp_4_0", 0.0))
            asymptomatic_chest_pain = float(request.form.get("asymptomatic_chest_pain", 0.0))
            
            age_sex = float(request.form.get("age_sex", 0.0))
            combined_risk_score = float(request.form.get("combined_risk_score", 0.0))
            oldpeak = float(request.form.get("oldpeak", 0.0))
            thalach = float(request.form.get("thalach", 0.0))
            thalach_exang = float(request.form.get("thalach_exang", 0.0))
            
            thal_3_0 = float(request.form.get("thal_3_0", 0.0))

            # making prediction
            sample = np.array([[ca_0_0, thal_7_0, cp_4_0, asymptomatic_chest_pain,
                                age_sex, combined_risk_score, oldpeak, thalach,
                                thalach_exang, thal_3_0]])
            prediction = model.predict(sample)[0]
            probability = model.predict_proba(sample)[0, 1]

            # More detailed risk assessment
            mi_risk = "High Risk" if prediction == 1 else "Low Risk"
            prob_percentage = f"{probability:.2%}"

            return render_template(
                "index.html",
                submitted=True,
                prediction=mi_risk,
                probability=prob_percentage
            )
        except Exception as e:
            print(e)
            return render_template(
                "index.html",
                submitted=False,
                error="Error: Please ensure all fields contain valid numerical values and try again."
            )

    # if GET, just render the blank form
    return render_template("index.html", submitted=False)

if __name__ == "__main__":
    app.run(debug=True)
