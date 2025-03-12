from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

def engineer_features(df):
    df_out = df.copy()
    df_out['age_over_55'] = (df_out['age'] >= 55).astype(int)
    df_out['asymptomatic_chest_pain'] = (df_out['cp'] == 4).astype(int)
    df_out['typical_angina'] = (df_out['cp'] == 1).astype(int)
    df_out['hypertension'] = (df_out['trestbps'] >= 140).astype(int)
    df_out['high_cholesterol'] = (df_out['chol'] > 240).astype(int)
    df_out['borderline_cholesterol'] = (
        (df_out['chol'] > 200) & (df_out['chol'] <= 240)
    ).astype(int)
    df_out['low_max_hr'] = (df_out['thalach'] < 150).astype(int)
    df_out['exercise_angina_risk'] = df_out['exang']
    df_out['significant_st_depression'] = (df_out['oldpeak'] > 1.5).astype(int)
    df_out['flat_st_slope'] = (df_out['slope'] == 2).astype(int)
    df_out['downsloping_st'] = (df_out['slope'] == 3).astype(int)
    df_out['age_sex'] = df_out['age'] * df_out['sex']
    df_out['thalach_exang'] = df_out['thalach'] * (1 - df_out['exang'])
    df_out['combined_risk_score'] = (
        df_out['age_over_55'] +
        df_out['hypertension'] +
        df_out['high_cholesterol'] +
        df_out['fbs'] +
        df_out['exercise_angina_risk'] +
        df_out['significant_st_depression']
    )
    return df_out

app = Flask(__name__)

# Load complete model dictionary
model_dict = joblib.load("mi_complete_model.pkl")

# Extract items
model = model_dict["model"]       
preprocessor = model_dict["preprocessor"]  
selector = model_dict["selector"]          
feature_names = model_dict["feature_names"] 
print("[DEBUG] Loaded model:", type(model))
print("[DEBUG] Loaded preprocessor:", type(preprocessor))
print("[DEBUG] Loaded selector:", type(selector))
print("[DEBUG] Feature names:", feature_names)

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            # Gather the 13 raw inputs from the form
            age = float(request.form.get("age", 0.0))
            sex = float(request.form.get("sex", 0.0))
            cp = float(request.form.get("cp", 0.0))
            trestbps = float(request.form.get("trestbps", 0.0))
            chol = float(request.form.get("chol", 0.0))
            fbs = float(request.form.get("fbs", 0.0))
            restecg = float(request.form.get("restecg", 0.0))
            thalach = float(request.form.get("thalach", 0.0))
            exang = float(request.form.get("exang", 0.0))
            oldpeak = float(request.form.get("oldpeak", 0.0))
            slope = float(request.form.get("slope", 0.0))
            ca = float(request.form.get("ca", 0.0))
            thal = float(request.form.get("thal", 0.0))

            # Create a single-row DataFrame with these raw features
            patient_df = pd.DataFrame([{
                'age': age,
                'sex': sex,
                'cp': cp,
                'trestbps': trestbps,
                'chol': chol,
                'fbs': fbs,
                'restecg': restecg,
                'thalach': thalach,
                'exang': exang,
                'oldpeak': oldpeak,
                'slope': slope,
                'ca': ca,
                'thal': thal
            }])

            print("[DEBUG] Raw patient data:\n", patient_df)

            # Engineer additional features
            patient_df = engineer_features(patient_df)
            print("[DEBUG] After feature engineering:\n", patient_df)

            # Apply preprocessor and selector
            processed = preprocessor.transform(patient_df)
            selected = selector.transform(processed)
            print("[DEBUG] Processed shape:", processed.shape)
            print("[DEBUG] Selected shape:", selected.shape)

            # 5. Predict with the final model
            prediction = model.predict(selected)[0]
            probability = model.predict_proba(selected)[0, 1]

            print("[DEBUG] prediction:", prediction)
            print("[DEBUG] probability:", probability)

            # Convert numeric prediction to text
            mi_risk = "High Risk" if prediction == 1 else "Low Risk"
            prob_percentage = f"{probability:.2%}"

            # Render the results
            return render_template(
                "index.html",
                submitted=True,
                prediction=mi_risk,
                probability=prob_percentage
            )
        except Exception as e:
            print("[DEBUG] Error during prediction:", e)
            return render_template(
                "index.html",
                submitted=False,
                error="Error: Please ensure all fields are valid numeric values."
            )

    # If GET, just render the blank form
    return render_template("index.html", submitted=False)

if __name__ == "__main__":
    app.run(debug=True)
