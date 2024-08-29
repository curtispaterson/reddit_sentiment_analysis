import snowflake.connector
import praw
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

# Fetch Snowflake connection parameters from environment variables
SNOWFLAKE_ACCOUNT = os.getenv('SNOWFLAKE_ACCOUNT')
SNOWFLAKE_USER = os.getenv('SNOWFLAKE_USER')
SNOWFLAKE_PASSWORD = os.getenv('SNOWFLAKE_PASSWORD')
SNOWFLAKE_WAREHOUSE = os.getenv('SNOWFLAKE_WAREHOUSE')
SNOWFLAKE_DATABASE = os.getenv('SNOWFLAKE_DATABASE')
SNOWFLAKE_SCHEMA = os.getenv('SNOWFLAKE_SCHEMA')

# Fetch Reddit API credentials from environment variables
CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')
USER_AGENT = os.getenv('USER_AGENT')

# Initialize Reddit API client
reddit = praw.Reddit(client_id=CLIENT_ID,
                     client_secret=CLIENT_SECRET,
                     user_agent=USER_AGENT)

# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

def get_snowflake_connection():
    """
    This function attempts to establish a connection to the Snowflake database using the provided
    credentials and connection parameters. If the connection is successful, it prints a confirmation message
    and returns the connection object. If the connection fails, it prints an error message and returns None.
    """
    try:
        conn = snowflake.connector.connect(
            user=SNOWFLAKE_USER,
            password=SNOWFLAKE_PASSWORD,
            account=SNOWFLAKE_ACCOUNT,
            warehouse=SNOWFLAKE_WAREHOUSE,
            database=SNOWFLAKE_DATABASE,
            schema=SNOWFLAKE_SCHEMA
        )
        print("Connection to Snowflake successful!")
        return conn
    except snowflake.connector.errors.Error as e:
        print(f"Connection to Snowflake failed: {e}")
        return None

def fetch_hot_posts(num_posts=10):
    """
    This function uses the Reddit API (via PRAW) to fetch the specified number of hot posts from the Reddit front page.
    Each post's title, score, creation time, and body text are extracted and stored in a list of dictionaries.
    """
    posts = []

    for submission in reddit.front.hot(limit=num_posts):
        posts.append({
            'title': submission.title,
            'score': submission.score,
            'created': submission.created_utc,
            'text': submission.selftext  # The body of the post
        })

    return posts

def analyze_sentiment(posts):
    """
    This function takes a list of post dictionaries and converts it into a Pandas DataFrame.
    It then uses the VADER sentiment analysis tool to calculate the sentiment scores for both the title
    and body of each post, adding these scores as new columns in the DataFrame.
    """
    df = pd.DataFrame(posts)

    # Analyze sentiment for the post title and body using VADER
    df['title_sentiment'] = df['title'].apply(lambda title: analyzer.polarity_scores(title)['compound'])
    df['body_sentiment'] = df['text'].apply(lambda text: analyzer.polarity_scores(text)['compound'])

    return df

def save_to_snowflake(df, table_name="REDDIT_SENTIMENT_ANALYSIS"):
    """
    This function saves the sentiment analysis results to a Snowflake table. It first establishes a connection
    to Snowflake, then iterates over the DataFrame rows to insert each record into the specified table. The
    'created' timestamp is converted to a string format that Snowflake can parse before insertion.
    The function handles errors during the insertion process and ensures that the connection is closed afterward.
    """
    conn = get_snowflake_connection()
    if conn:
        cursor = conn.cursor()
        try:
            # Convert the 'created' column to a string in the format Snowflake expects
            df['created'] = df['created'].apply(lambda x: pd.to_datetime(x, unit='s').strftime('%Y-%m-%d %H:%M:%S'))

            # Insert data into Snowflake
            for _, row in df.iterrows():
                insert_query = f"""
                INSERT INTO {SNOWFLAKE_SCHEMA}.{table_name} (TITLE, SCORE, CREATED, TEXT, TITLE_SENTIMENT, BODY_SENTIMENT)
                VALUES (%s, %s, %s, %s, %s, %s);
                """
                cursor.execute(insert_query, (
                    row['title'],
                    row['score'],
                    row['created'],  # Now a string
                    row['text'],
                    row['title_sentiment'],
                    row['body_sentiment']
                ))

            conn.commit()
            print("Data successfully inserted into Snowflake.")
        except snowflake.connector.errors.Error as e:
            print(f"Failed to insert data into Snowflake: {e}")
        finally:
            cursor.close()
            conn.close()

def main():
    """
    This function orchestrates the process of fetching hot posts from Reddit, analyzing their sentiment,
    and saving the results to a Snowflake table. It first fetches the posts, analyzes the sentiment,
    and then attempts to save the results. If no posts are found, it prints an appropriate message.
    """
    posts = fetch_hot_posts()
    if posts:
        sentiment_df = analyze_sentiment(posts)
        save_to_snowflake(sentiment_df)
    else:
        print("No hot posts found.")

if __name__ == "__main__":
    main()