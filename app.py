import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from flask import Flask, request, render_template, flash,redirect, url_for
from flask.globals import request, session
from keras_preprocessing.sequence import pad_sequences
import nltk 
from nltk.corpus import stopwords
from keras_preprocessing.text import tokenizer_from_json
import json 
import emoji
from textblob import TextBlob
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt



stopwords_list = set(stopwords.words('english'))

maxlen = 100

# Load model
model_path ='lstm_model.h5'
pretrained_lstm_model = load_model(model_path)

# Loading
with open('tokenizer.json') as f:
    data = json.load(f)
    loaded_tokenizer = tokenizer_from_json(data)

# creating function for data cleaning
from b2_preprocessing_function import CustomPreprocess
custom = CustomPreprocess()

#mongodb 
import pymongo
client=pymongo.MongoClient("mongodb://localhost:27017")
db=client['project5']
collection=db['collection5']

app = Flask(__name__)
app.secret_key = 'secret'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/explore')
def explore():
    return render_template('explore.html')

@app.route('/response')
def response():
    return render_template('sresponse.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    query_asis = [str(x) for x in request.form.values()]
#     query_list = []
#     query_list.append(query_asis)
    
    # Preprocess review text with earlier defined preprocess_text function
    query_processed_list = []
    for query in query_asis:
        query_processed = custom.preprocess_text(query)
        query_processed_list.append(query_processed)
        

    query_tokenized = loaded_tokenizer.texts_to_sequences(query_processed_list)
    query_padded = pad_sequences(query_tokenized, padding='post', maxlen=maxlen)
    
    query_sentiments = pretrained_lstm_model.predict(query_padded)
    
    if query_sentiments[0][0]>0.5:
        #add to mongodb
        dict={'review':query,'sentiment':'positive'}
        collection.insert_one(dict)
    else:
        #add to mongodb
        dict={'review':query,'sentiment':'negative'}
        collection.insert_one(dict)
    
        
    #emoji sentiment
    def extract_emojis(text):
        return ''.join(c for c in text if c in emoji.EMOJI_DATA)

    def predict_sentiment_from_emojis(emojis):
        if emojis:
            analysis = TextBlob(emojis)
            # Sentiment polarity ranges from -1 to 1
            polarity = analysis.sentiment.polarity
            if polarity > 0:
                return "Positive sentiment"
            elif polarity < 0:
                return "Negative sentiment"
            else:
                return "Neutral sentiment"
        else:
            return "No emojis found"
        
    
    user_text=query
    # print(user_text)
    emojis = extract_emojis(user_text)
    sentiment_prediction = predict_sentiment_from_emojis(emojis)
    # print(sentiment_prediction)
    
    

    if query_sentiments[0][0]>0.5:
        return render_template('sresponse.html', prediction_text=f"Result: Positive Review & on the basis of emoji: {sentiment_prediction}")
        
    else:
        return render_template('sresponse.html', prediction_text=f"Result: Negative Review & on the basis of emoji: {sentiment_prediction}")


fixed_username = 'admin'
fixed_password = '123'

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == fixed_username and password == fixed_password:
            # Redirect to admin dashboard or perform any admin-specific tasks
            return redirect(url_for('admin_dashboard'))
        else:
            error_message = 'Invalid username or password. Please try again.'

    return render_template('admin.html', error_message=error_message if 'error_message' in locals() else None)
 
   
@app.route('/admindashboard')
def admin_dashboard():
    # number of reviews and sentiments
    csv_file_path = 'record.csv'
    df = pd.read_csv(csv_file_path)
    num_records_review = len(df['review'])
    num_records_sentiment = len(df['sentiment'])
    
    # chart one 
    df = pd.read_csv('record.csv')
    sentiment_counts = df['sentiment'].value_counts()
    colors = {'positive': 'green', 'negative': 'red'}
    plt.figure()
    plt.bar(sentiment_counts.index, sentiment_counts, color=[colors[s] for s in sentiment_counts.index])
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Reviews')
    plt.title('Sentiment Distribution in Reviews')
    plt.yticks(range(int(sentiment_counts.max()) + 1))
    canvas = FigureCanvas(plt.gcf())  # Create canvas from the figure
    canvas.print_png('static/sentiment_distribution.png')
    plt.close()  # Close the figure to release resources
    
    
    #chart two
    nltk.download('stopwords')
    df = pd.read_csv('record.csv')
    text = ' '.join(df['review'].dropna())
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    word_freq = FreqDist(filtered_words)
    top_words = word_freq.most_common(20)
    words, frequencies = zip(*top_words)
    total_words = len(filtered_words)
    most_use_threshold = total_words * 0.02  # 2% of total words
    low_use_threshold = total_words * 0.005  # 0.5% of total words
    categories = ['most use' if freq > most_use_threshold else 'normal use' if freq > low_use_threshold else 'low use' for freq in frequencies]
    colors = {'most use': 'green', 'normal use': 'orange', 'low use': 'red'}
    plt.figure(figsize=(12, 6))
    bars = plt.bar(words, frequencies, color=[colors[category] for category in categories])
    legend_labels = [plt.Rectangle((0, 0), 1, 1, color=colors[category]) for category in colors]
    plt.legend(legend_labels, colors.keys())
    plt.title('Categorized Words in Reviews')
    plt.xlabel('Words')
    plt.ylabel('Frequency of Words')
    plt.xticks(rotation=45, ha='right')
    canvas = FigureCanvas(plt.gcf())  # Create canvas from the figure
    canvas.print_png('static/review_distribution.png')
    plt.close()  # Close the figure to release resources

    
    return render_template('admindashboard.html',num_records_review=num_records_review,num_records_sentiment=num_records_sentiment)

@app.route('/record',methods=['GET','POST'])
def record():
    all_records=collection.find()
    make_list=list(all_records)
    df=pd.DataFrame(make_list)
    # df.to_html('templates\\record.html',columns=['review','sentiment'],index=False)
    df.to_csv('record.csv',columns=['review','sentiment'],index=False)
    df = pd.read_csv('record.csv')
    df['Number'] = range(1, len(df) + 1)
    columns_order = ['Number'] + [col for col in df.columns if col not in ['Number']]
    df = df[columns_order]
    table_html = df.to_html(classes='table table-striped', index=False)
    
    
    # return render_template('record.html')
    return render_template('record.html', table_html=table_html)


if __name__ == '__main__':
    app.run(debug=True)
