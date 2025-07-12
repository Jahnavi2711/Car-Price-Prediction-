import pandas as pd
import datetime
import xgboost as xgb
import streamlit as st

# Custom CSS for light blue background and styled header
st.markdown(
    """
    <style>
    .stApp {
        background-color: #e6f2ff;  /* Light blue background */
        color: #000000;
    }

    .custom-header {
        background-color: #445580;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-size: 32px;
        font-weight: 700;
        box-shadow: 2px 4px 10px rgba(0,0,0,0.2);
        margin-bottom: 20px;
    }

    .logo-header {
        display: flex;
        align-items: center;
        gap: 15px;
        justify-content: center;
    }

    .logo-header img {
        width: 50px;
        height: 50px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    # Logo and gradient heading
    logo_url = "https://cdn-icons-png.flaticon.com/512/743/743007.png"  # Example logo URL
    st.markdown(f"""
        <div class='logo-header'>
            <img src='{logo_url}' />
            <div class='custom-header'>Car Price Prediction</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🚗 Are you planning to sell your car?\n#### 💡 Let's estimate its market value:")

    # Load model
    model = xgb.XGBRegressor()
    model.load_model('xgb_model.json')

    # Input fields
    p1 = st.number_input("Ex-showroom price of the car (in lakhs)", 2.5, 25.0, step=1.0)
    p2 = st.number_input("Distance completed by the car in kilometers", 100, 5000000, step=100)

    s1 = st.selectbox("Fuel type of the car", ('Petrol', 'Diesel', 'CNG'))
    p3 = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}[s1]

    s2 = st.selectbox("Are you a Dealer or Individual?", ('Dealer', 'Individual'))
    p4 = {'Dealer': 0, 'Individual': 1}[s2]

    s3 = st.selectbox("Transmission type", ('Manual', 'Automatic'))
    p5 = {'Manual': 0, 'Automatic': 1}[s3]

    p6 = st.slider("How many previous owners?", 0, 3)

    date_time = datetime.datetime.now()
    years = st.number_input("Year the car was purchased", 1990, date_time.year)
    p7 = date_time.year - years

    # DataFrame for prediction
    data_new = pd.DataFrame({
        'Present_Price': p1,
        'Driven_kms': p2,
        'Fuel_Type': p3,
        'Selling_type': p4,
        'Transmission': p5,
        'Owner': p6,
        'Age': p7
    }, index=[0])

    try:
        if st.button('Predict'):
            pred = model.predict(data_new)
            if pred[0] > 0:
                st.markdown(
                   f"""
                   <div style='
                    background-color: #007acc;
                    padding: 16px;
                    border-radius: 10px;
                    color: white;
                    font-size: 20px;
                    font-weight: bold;
                    text-align: center;
                    box-shadow: 2px 4px 10px rgba(0,0,0,0.2);'>
                    ✅ Estimated Selling Price: ₹ {pred[0]:.2f} lakhs
                   </div>
                   """,
                 unsafe_allow_html=True
                )

            else:
                st.warning("⚠️ It seems the car cannot be sold.")
    except Exception as e:
        st.warning("Something went wrong. Please try again!")
        st.error(str(e))

if __name__ == "__main__":
    main()
