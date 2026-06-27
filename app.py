import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load('artifacts/XGBoost_model.pkl')
le    = joblib.load('artifacts/label_encoder.pkl')

def main():
    st.set_page_config(initial_sidebar_state="expanded")
    st.title('Credit Score Prediction')
    st.info("This app predicts a customer's credit score (Good, Standard, or Poor) based on their financial profile, account information, and payment behavior.")
    st.warning("👈 Start by filling in the customer's basic information in the sidebar on the left, then complete the account and payment details below before clicking **Make Prediction**.")

    #Sidebar
    with st.sidebar:
        st.header("Customer Information")
        age = st.number_input('Age', 14, 100, 30)
        occupation = st.selectbox('Occupation', ['Accountant', 'Architect', 'Developer', 'Doctor', 'Engineer', 'Entrepreneur', 'Journalist', 'Lawyer', 'Manager', 'Mechanic', 'Media_Manager', 'Musician', 'Scientist', 'Teacher', 'Writer'])
        month = st.selectbox('Month', ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
        annual_income = st.number_input('Annual Income', 0.0, 300000.0, 50000.0, step=1000.0)
        monthly_inhand_salary = st.number_input('Monthly Inhand Salary', 0.0, 20000.0, 4000.0, step=100.0)

    #Main Form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)  #create 2 equal width columns
        with col1:
            st.subheader("Accounts & Loans")
            num_bank_accounts = st.number_input('Number of Bank Accounts', 0, 20, 5)
            num_credit_card = st.number_input('Number of Credit Cards', 0, 20, 5)
            num_of_loan = st.number_input('Number of Loans', 0, 15, 3)
            num_loan_types = st.number_input('Number of Loan Types', 0, 9, 2)
            interest_rate = st.number_input('Interest Rate (%)', 1, 50, 12)
            outstanding_debt = st.number_input('Outstanding Debt', 0.0, 5000.0, 1000.0, step=10.0)
            credit_mix = st.selectbox('Credit Mix', ['Bad', 'Standard', 'Good'])
            credit_history_years = st.number_input('Credit History (Years)', 0, 33, 15)
            credit_history_months = st.number_input('Credit History (Months)', 0, 11, 6)

        with col2:
            st.subheader("Payment Behavior")
            delay_from_due_date = st.number_input('Delay from Due Date (days)', -5, 70, 10)
            num_of_delayed_payment = st.number_input('Number of Delayed Payments', 0, 50, 10)
            num_credit_inquiries = st.number_input('Number of Credit Inquiries', 0, 50, 5)
            changed_credit_limit = st.number_input('Changed Credit Limit', -10.0, 50.0, 10.0, step=0.5)
            credit_utilization_ratio = st.number_input('Credit Utilization Ratio', 20.0, 50.0, 32.0, step=0.5)
            total_emi_per_month = st.number_input('Total EMI per Month', 0.0, 5000.0, 100.0, step=10.0)
            amount_invested_monthly = st.number_input('Amount Invested Monthly', 0.0, 2000.0, 100.0, step=10.0)
            monthly_balance = st.number_input('Monthly Balance', 0.0, 2000.0, 300.0, step=10.0)
            payment_of_min_amount = st.selectbox('Payment of Minimum Amount', ['Yes', 'No'])
            payment_behaviour_map = {
                "Low spent, Small value payments":"Low_spent_Small_value_payments",
                "Low spent, Medium value payments":"Low_spent_Medium_value_payments",
                "Low spent, Large value payments":"Low_spent_Large_value_payments",
                "High spent, Small value payments":"High_spent_Small_value_payments",
                "High spent, Medium value payments":"High_spent_Medium_value_payments",
                "High spent, Large value payments":"High_spent_Large_value_payments"
            }
            payment_behaviour_display = st.selectbox('Payment Behaviour', list(payment_behaviour_map.keys()))

        submitted = st.form_submit_button('Make Prediction')

    if submitted:
        month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}

        features = {
            "Month": month_map[month],
            "Age": int(age),
            "Annual_Income": annual_income,
            "Monthly_Inhand_Salary": monthly_inhand_salary,
            "Num_Bank_Accounts": int(num_bank_accounts),
            "Num_Credit_Card": int(num_credit_card),
            "Interest_Rate": int(interest_rate),
            "Num_of_Loan": int(num_of_loan),
            "Delay_from_due_date": int(delay_from_due_date),
            "Num_of_Delayed_Payment": int(num_of_delayed_payment),
            "Changed_Credit_Limit": changed_credit_limit,
            "Num_Credit_Inquiries": int(num_credit_inquiries),
            "Outstanding_Debt": outstanding_debt,
            "Credit_Utilization_Ratio": credit_utilization_ratio,
            "Credit_History_Age": credit_history_years * 12 + credit_history_months,
            "Total_EMI_per_month": total_emi_per_month,
            "Amount_invested_monthly": amount_invested_monthly,
            "Monthly_Balance": monthly_balance,
            "Num_Loan_Types": int(num_loan_types),
            "Credit_Mix": credit_mix,
            "Occupation": occupation,
            "Payment_of_Min_Amount": payment_of_min_amount,
            "Payment_Behaviour": payment_behaviour_map[payment_behaviour_display]
        }

        result = make_prediction(features)

        credit_score = result["credit_score"]
        probabilities = result["probabilities"]

        st.subheader("Prediction Results")
        if credit_score == "Good":
            st.success(f'Credit Score: {credit_score}')
            st.markdown("**This customer shows a strong credit profile.**")
        elif credit_score == "Standard":
            st.warning(f'Credit Score: {credit_score}')
            st.markdown("**This customer shows an average credit profile.**")
        else:
            st.error(f'Credit Score: {credit_score}')
            st.markdown("**This customer shows a high-risk credit profile.**")

        st.subheader("Class Probabilities")
        prob_df = pd.DataFrame({
            "Probability": {
                "Good": probabilities[0],
                "Poor": probabilities[1],
                "Standard": probabilities[2]
            }
        })
        st.bar_chart(prob_df)

def make_prediction(features):
    df = pd.DataFrame([features])
    prediction = model.predict(df)
    credit_label = le.inverse_transform(prediction)[0]
    probabilities = model.predict_proba(df)[0]
    return {'credit_score': credit_label, 'probabilities': probabilities}


if __name__ == '__main__':
    main()
