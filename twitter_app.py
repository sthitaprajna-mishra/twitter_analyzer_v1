# Streamlit
import streamlit as st 

# Stream live tweets
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler, Stream, API, Cursor, TweepError

# Data cleaning and preprocessing
import re
import copy
import numpy as np
import pandas as pd

# Visualization libraries
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# NLP libraries
import nltk
nltk.download("punkt")
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Additional libraries
import os
from PIL import Image

# # # TWITTER CLIENT # # #

class TwitterClient():
    
    def __init__(self, twitter_user = None):
        self.auth = TwitterAuthenticator().authenticate_twitter_app()
        self.twitter_client = API(self.auth)
        self.twitter_user = twitter_user

    def get_img_url(self, screen_name):
        return self.twitter_client.get_user(screen_name).profile_image_url_https

    def get_id(self, screen_name):
        return self.twitter_client.get_user(screen_name).id

    def get_name(self, screen_name):
        return self.twitter_client.get_user(screen_name).name

    def get_follower_count(self, screen_name):
        return self.twitter_client.get_user(screen_name).followers_count

    def get_verif_status(self, screen_name):
        return self.twitter_client.get_user(screen_name).verified

    def get_creation_date(self, screen_name):
        return self.twitter_client.get_user(screen_name).created_at

    def get_desc(self, screen_name):
        return self.twitter_client.get_user(screen_name).description

    def get_status_count(self, screen_name):
        return self.twitter_client.get_user(screen_name).statuses_count

    def get_twitter_client_api(self):
        return self.twitter_client
        
    def get_user_timeline_tweets(self, num_tweets):
        tweets = []
        for tweet in Cursor(self.twitter_client.user_timeline, id = self.twitter_user).items(num_tweets):
            tweets.append(tweet)
        return tweets
    
    def get_friend_list(self, num_friends):
        friend_list = []
        for friend in Cursor(self.twitter_client.friends, id = self.twitter_user).items(num_friends):
            friend_list.append(friend)
        return friend_list
    
    def get_home_timeline_tweets(self, num_tweets):
        home_timeline_tweets = []
        for tweet in Cursor(self.twitter_client.home_timeline, id = self.twitter_user).items(num_tweets):
            home_timeline_tweets.append(tweet)
        return home_timeline_tweets

# # # TWITTER AUTHENTICATOR # # #
class TwitterAuthenticator():
    
    def authenticate_twitter_app(self):
            auth = OAuthHandler(os.environ['API_KEY'], os.environ['API_SECRET'])
            auth.set_access_token(os.environ['ACCESS_TOKEN'], os.environ['ACCESS_SECRET'])
            return auth

# # # TWITTER STREAM LISTENER # # # 

class TwitterListener(StreamListener):
    """
    This is a basic listener class that just prints tweets to stdout
    """
    
    def __init__(self, fetched_tweets_filename):
        self.fetched_tweets_filename = fetched_tweets_filename
    
    def on_data(self, data):
        try:
            print(data)
            with open(fetched_tweets_filename, 'a') as f:
                f.write(data)
            return True
        except BaseException as e:
            print("Error on_data: {}".format(e))
        return True
    
    def on_error(self, status):
        if status == 420:
            # returning False on_data method in case rate limit occurs
            return False
        print(status)


# # # TWITTER STREAMER # # # 

class TwitterStreamer():
    """
    Class for streaming and processing live tweets
    """
    def __init__(self):
        self.twitter_authenticator = TwitterAuthenticator()
        
    
    def stream_tweets(self, fetched_tweets_filename, hash_tag_list):
        # This handles Twitter authentication and the connection to the Twitter Streaming API
        listener = TwitterListener(fetched_tweets_filename)
        
        auth = self.twitter_authenticator.authenticate_twitter_app()

        stream = Stream(auth, listener)

        stream.filter(track = hash_tag_list)


class TweetAnalyzer():
    """
    Functionality for analyzing and categorizing content from tweets.
    """
    
    def clean_tweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def analyze_sentiment(self, tweet):
        analysis = TextBlob(self.clean_tweet(tweet))
        
        if analysis.sentiment.polarity > 0:
            return 1
        elif analysis.sentiment.polarity == 0:
            return 0
        else:
            return -1
    
    def tweets_to_data_frame(self, tweets):
        df = pd.DataFrame(data = [tweet.text for tweet in tweets], columns = ['tweets'])
        df['id'] = np.array([tweet.id for tweet in tweets])
        df['len'] = np.array([len(tweet.text) for tweet in tweets])
        df['date'] = np.array([tweet.created_at for tweet in tweets])
        df['source'] = np.array([tweet.source for tweet in tweets])
        df['likes'] = np.array([tweet.favorite_count for tweet in tweets])
        df['retweets'] = np.array([tweet.retweet_count for tweet in tweets])
        
        return df


twitter_client = TwitterClient()
tweet_analyzer = TweetAnalyzer()

api = twitter_client.get_twitter_client_api()


# # # USER INTERFACE # # # 

# Use the full page instead of a narrow central column
# st.beta_set_page_config(layout="wide")

# Sidebar Heading
st.sidebar.write("""

# Analyze Twitter Profile 

This app will give you a complete analysis of any Twitter profile.

	""")

st.sidebar.header("User Input")

# Accepts user data
user_id = st.sidebar.text_input("Enter Twitter username: ", "realDonaldTrump")
tweet_count = st.sidebar.slider("Enter number of tweets to analyze: ", 10, 3200, 100)

st.sidebar.write("""

	---

	""")

st.sidebar.subheader("Basic Profile Info")

img_display = st.sidebar.checkbox("Enlarge profile picture")
account_created = st.sidebar.checkbox("Account creation date")
description = st.sidebar.checkbox("Description")
status_count = st.sidebar.checkbox("Status count")

st.sidebar.write("""

	---

	""")

st.sidebar.subheader("Sentiment Analysis")

user_width = st.sidebar.slider("Choose width of wordclouds: ", 300, 800, 700)
user_height = st.sidebar.slider("Choose height of wordclouds: ", 200, 500, 400)
user_font = st.sidebar.slider("Choose max fontsize of wordclouds: ", 50, 150, 80)

# Extract message from Tweepy errors
def getExceptionMessage(msg):
    words = msg.split(' ')

    errorMsg = ""
    for index, word in enumerate(words):
        if index not in [0,1,2]:
            errorMsg = errorMsg + ' ' + word
    errorMsg = errorMsg.rstrip("\'}]")
    errorMsg = errorMsg.lstrip(" \'")

    return errorMsg

# Convert tweets into dataframe
try:
	tweets = api.user_timeline(screen_name = user_id, count = tweet_count)
except TweepError as e:
    st.write(getExceptionMessage(e.reason))
    raise Exception('API code: ', e.api_code)
    
df = tweet_analyzer.tweets_to_data_frame(tweets)
df['sentiment'] = np.array([tweet_analyzer.analyze_sentiment(tweet) for tweet in df['tweets']])


# Get and display user profile picture
user_img = twitter_client.get_img_url(user_id)
index = user_img.find("_normal.jpg")
user_img_big = user_img[:index] + "_bigger.jpg"
user_img_orig = user_img[:index] + ".jpg"

st.write("""

	# Basic Profile Info

	""")
st.write("""

	### Profile Picture

	""")

if img_display:
	st.markdown("![Profile Picture]({})".format(user_img_orig))
else:
	st.markdown("![Profile Picture]({})".format(user_img_big))
	
col1, col2 = st.beta_columns(2)
col1.write("""

### Profile ID

	""")

col1.write(twitter_client.get_id(user_id))
col1.write("""

### Profile Name

	""")

col1.write(twitter_client.get_name(user_id))
col2.write("""

### Account Verification Status

	""")

col2.write(twitter_client.get_verif_status(user_id))
col2.write("""

### Total Followers

	""")

col2.write(twitter_client.get_follower_count(user_id))

if account_created:
	col2.write("""

	### Account Created On

		""")

	col2.write(twitter_client.get_creation_date(user_id))

if description:
	col2.write("""

	### User Description

		""")

	col2.write(twitter_client.get_desc(user_id))	

if status_count:
	col2.write("""

	### Status Count

		""")

	col2.write(twitter_client.get_status_count(user_id))	

st.write("""

	---

	""")

st.title("Tweet Stats")

st.write("""

	### Average length of all tweets

	""")
st.write(np.mean(df['len']))

col1, col2 = st.beta_columns(2)

col1.write("""

	### Most likes on a tweet

	""")
col1.write(np.max(df['likes']))

col2.write("""

	### Most retweets on a tweet

	""")
col2.write(np.max(df['retweets']))

col1.write("""

	### Least likes on a tweet

	""")
col1.write(np.min(df['likes']))

col2.write("""

	### Least retweets on a tweet

	""")
col2.write(np.min(df['retweets']))

dates = []

for i in df['date']:
    dates.append(i.date())


col1, col2 = st.beta_columns(2)

col1.write("""

## Likes (Time Series)

	""")


likes_df = pd.DataFrame({'date' : dates, 'likes' : df['likes']})
likes_df = likes_df.set_index('date')
col1.line_chart(likes_df)

col2.write("""

## Retweets (Time Series)

	""")


rt_df = pd.DataFrame({'date' : dates, 'retweets' : df['retweets']})
rt_df = rt_df.set_index('date')
col2.line_chart(rt_df)

st.write("""

## Likes vs Retweets (Time Series)

	""")


both_df = pd.DataFrame({'date' : dates, 'likes' : df['likes'], 'retweets' : df['retweets']})
both_df = both_df.set_index('date')
st.line_chart(both_df)

st.write("""

## Tweet Frequency

	""")
freq = {} 
for item in dates: 
    if (item in freq): 
        freq[item] += 1
    else: 
        freq[item] = 1

freq_df = pd.DataFrame({'date' : freq.keys(), 'tweet_count' : freq.values()})
freq_df = freq_df.set_index('date')
st.area_chart(freq_df)

# # # # # # # SENTIMENT ANALYSIS # # # # # # # 

st.title("Sentiment Analysis")


# # # DATA CLEANING # # #

# Creating a copy for this purpose
data = copy.deepcopy(df)
data.drop(['id', 'len', 'date', 'source', 'likes', 'retweets'],axis = 1,inplace = True)


# Removing Twitter Handles (@user)
data['Clean_TweetText'] = data['tweets'].str.replace("RT @", "") 

# Removing links
data['Clean_TweetText'] = data['Clean_TweetText'].str.replace(r"http\S+", "")

# Removing Punctuations, Numbers, and Special Characters
data['Clean_TweetText'] = data['Clean_TweetText'].str.replace("[^a-zA-Z]", " ")

# Remove stop words
stopwords = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])

def remove_stopwords(text):
    clean_text=' '.join([word for word in text.split() if word not in stopwords])
    return clean_text

data['Clean_TweetText'] = data['Clean_TweetText'].apply(lambda text : remove_stopwords(text.lower()))

# Text Tokenization and Normalization
data['Clean_TweetText'] = data['Clean_TweetText'].apply(lambda x: nltk.word_tokenize(x))

# Now let’s stitch these tokens back together
data['Clean_TweetText'] = data['Clean_TweetText'].apply(lambda x: ' '.join([w for w in x]))

# Removing small words
data['Clean_TweetText'] = data['Clean_TweetText'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

# Getting rid of image tweets
data["Clean_TweetText"].replace("", np.nan, inplace = True)
data.dropna(inplace=True)

# WordCloud Function
def create_wordcloud(df_name, u_width, u_height, u_font):
	all_words = ' '.join([text for text in df_name['Clean_TweetText']])
	wordcloud = WordCloud(width=u_width,height=u_height,random_state=21,max_font_size=u_font).generate(all_words)
	return st.image(wordcloud.to_array())

# TOTAL WORLD CLOUD #

st.write("""

    ## Word Cloud (All Tweets)

    """)
create_wordcloud(data, user_width, user_height, user_font)

# POSITIVE WORLD CLOUD #

# Create dataframe containing positive tweets only
pos_df = data[data["sentiment"] == 1]

st.write("""

    ## Word Cloud (Positive Tweets)

    """)
create_wordcloud(pos_df, user_width, user_height, user_font)

# NEGATIVE WORLD CLOUD #

# Create dataframe containing negative tweets only
neg_df = data[data["sentiment"] == -1]

st.write("""

    ## Word Cloud (Negative Tweets)

    """)
create_wordcloud(neg_df, user_width, user_height, user_font)


# VaderSentiment Analysis

analyzer = SentimentIntensityAnalyzer()

def sentiment(text):
    vs = analyzer.polarity_scores(text)
    if (vs["pos"] - vs["neg"] == 0) or vs["neu"] == 1:
        return "Meh"
    if not vs["neg"] > 0.15:
        if vs["pos"] - vs["neg"] >= 0.3:
            return "Positive"
        if vs["pos"] - vs["neg"] >= 0:
            return "A bit positive"
    if not vs["pos"] > 0.15:
        if vs["neg"] - vs["pos"] >= 0.3:
            return "Negative"
        if vs["neg"] - vs["pos"] >= 0:
            return "A bit negative"
    return "Meh"

bit_pos = 0
pos = 0
meh = 0
bit_neg = 0
neg = 0

for i in data["Clean_TweetText"]:
    ans = sentiment(i)
    if ans == "A bit positive":
        bit_pos += 1
    elif ans == "Positive":
        pos += 1
    elif ans == "Meh":
        meh += 1 
    elif ans == "A bit negative":
        bit_neg += 1
    else:
        neg += 1
    
# Create a List of Labels for x-axis
x = ["A bit positive", "Positive", "Meh", "A bit negative", "Negative"]

# Create a List of Values (Same Length as Names List)
y = [bit_pos, pos, meh, bit_neg, neg]

st.write("""

    ## Sentiment Breakdown of Tweets (Pie Chart)

    """)
fig = px.pie(df, values= y, names= x)
st.plotly_chart(fig)

st.write("""

    ## Sentiment Breakdown of Tweets (Bar Chart)

    """)
fig = go.Figure(data=[
        go.Bar(name='Sentiment', x= x, y= y)])
st.plotly_chart(fig)

st.balloons()
