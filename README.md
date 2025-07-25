🏠 House Price Prediction using Machine Learning


This project aims to build a machine learning model that predicts house prices based on various features like location, number of rooms, population, and more. It includes data preprocessing, model training, evaluation, and deployment using \*\*Streamlit\*\*.

🚀 Demo

Want to try it out?  
Run this in your terminal:

```bash
streamlit run app.py

📂 Project Structure

bash
house-price-prediction/
├── app.py              # Streamlit web app
├── main.py             # Training script
├── housing.csv         # Dataset
├── .gitignore          # Ignoring model files
├── model.pkl (ignored) # Trained ML model (not pushed)
├── pipeline.pkl (ignored)


🧠 Features

End-to-end machine learning pipeline

Data cleaning and preprocessing

Model training using Random Forest / Gradient Boosting

Real-time prediction using Streamlit interface


⚙️ Technologies Used

Python
Pandas, NumPy
Scikit-learn
Streamlit
Matplotlib \& Seaborn (for EDA)


📊 Dataset

The dataset contains housing data including:
Median income
Housing age
Rooms per household
Population
Latitude / Longitude

(Note: You can replace this with real Indian city data if needed.)

📦 How to Run
bash
\# Install dependencies
pip install -r requirements.txt
\# Train the model

python main.py

\# Launch the app

streamlit run app.py

💡 Future Improvements

Integrate Google Maps API for address-based predictions
Add feature selection and tuning option
Deploy to Streamlit Cloud or Heroku


📬 Contact

Made with ❤️ by Rahul Sahu

