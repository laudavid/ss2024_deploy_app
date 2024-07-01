# Demo 1: Deploy a Sentiment Analysis app using Streamlit 

![Alt text](images/app_demo.PNG)



## 1. Streamlit tutorials
- Deploy a Streamlit app: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app
- Manage streamlit secrets: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
- Connect app to external data sources: https://docs.streamlit.io/develop/tutorials/databases
- Deploy to HuggingFace spaces: https://medium.com/@imanuelyosi/deploy-your-streamlit-web-app-using-hugging-face-7b9cddb11148
<br>

## 2. Useful commands

Launch a streamlit app locally <br>
```python 
streamlit run app.py
```
<br>

Build a `requirments.txt` file for deployment
```python
pip3 freeze > requirements.txt
```