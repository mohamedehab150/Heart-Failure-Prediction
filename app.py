import gradio as gr
import pandas as pd
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import joblib


def handle_outliers(X):
    X_df = pd.DataFrame(X).copy()
    for col in X_df.columns:
        lower = X_df[col].quantile(0.01)
        upper = X_df[col].quantile(0.99)
        X_df[col] = X_df[col].clip(lower, upper)
    return X_df
    
model = joblib.load("pipeline_with_smote.pkl")

def predict_heart_failure(age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
                          high_blood_pressure, platelets, serum_creatinine, serum_sodium,
                          sex, smoking, time):
    # Create dataframe for prediction
    input_data = pd.DataFrame([[age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
                                high_blood_pressure, platelets, serum_creatinine, serum_sodium,
                                sex, smoking, time]],
                              columns=['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
                                       'ejection_fraction', 'high_blood_pressure', 'platelets',
                                       'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time'])

  
    prediction = model.predict(input_data)[0]
    proba_safe = model.predict_proba(input_data)[0][0]   # class 0
    proba_death = model.predict_proba(input_data)[0][1]  # class 1

    
    if prediction == 1:
        result_html = f"""
        <div style='background-color:#ffcccc; padding:15px; border-radius:10px; text-align:center;'>
            <h2 style='color:red;'>💀 خطر: احتمالية الوفاة مرتفعة</h2>
            <p style='font-size:18px;color:black';>نسبة الوفاة: <b style='font-size:18px;color:black';>{proba_death*100:.1f}%</b></p>
            <p style='font-size:18px;color:black';'>نسبة النجاة: <b style='font-size:18px;color:black';>{proba_safe*100:.1f}%</b></p>
        </div>
        """
    else:
        result_html = f"""
        <div style='background-color:#ccffcc; padding:15px; border-radius:10px; text-align:center;'>
            <h2 style='color:green;'>✅ المنطقة الآمنة</h2>
            <p style='font-size:18px;color:black';'>نسبة النجاة: <b style='font-size:18px;color:black';>{proba_safe*100:.1f}%</b></p>
            <p style='font-size:18px;color:black';'>نسبة الوفاة: <b style='font-size:18px;color:black';>{proba_death*100:.1f}%</b></p>
        </div>
        """

    return result_html


iface = gr.Interface(
    fn=predict_heart_failure,
    inputs=[
        gr.Number(label="Age"),
        gr.Radio([0, 1], label="Anaemia"),
        gr.Number(label="Creatinine Phosphokinase"),
        gr.Radio([0, 1], label="Diabetes"),
        gr.Number(label="Ejection Fraction"),
        gr.Radio([0, 1], label="High Blood Pressure"),
        gr.Number(label="Platelets"),
        gr.Number(label="Serum Creatinine"),
        gr.Number(label="Serum Sodium"),
        gr.Radio([0, 1], label="Sex (0=female, 1=male)"),
        gr.Radio([0, 1], label="Smoking"),
        gr.Number(label="Follow-up Time")
    ],
    outputs=gr.HTML(label="Prediction"),
    title="🔍 Heart Failure Prediction",
    description="ادخل البيانات الطبية للحصول على توقع إحتمالية الوفاة والنجاة"
)

iface.launch()
