import os
import re
import ast
import json
import time
from itertools import chain
import snscrape.modules.twitter as sntwitter
import tweepy
import torch

script_dir = os.path.dirname(__file__)
data_path = os.path.join(script_dir, "data/auth.json")
twitter_auth_data = open(data_path).read()
twitter_auth_data_json = json.loads(twitter_auth_data)

access_token = twitter_auth_data_json["access_token"]
access_token_secret = twitter_auth_data_json["access_token_secret"]
consumer_key = twitter_auth_data_json["consumer_key"]
consumer_secret = twitter_auth_data_json["consumer_secret"]

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

def eval_code(code):
    parsed = ast.parse(code, mode='eval')
    fixed = ast.fix_missing_locations(parsed)
    compiled = compile(fixed, '<string>', 'eval')
    return eval(compiled)

def collect_tweet_replies(tweet_id, max_num_replies):
    replies_ids = []

    for reply in sntwitter.TwitterSearchScraper(
            query=f"conversation_id:{tweet_id} (filter:safe OR -filter:safe)").get_items():
        replies_ids.append(reply.id)

    batch_size_replies = 50
    n_chunks_repl = int((len(replies_ids) - 1) // batch_size_replies + 1)

    replies = []
    i = 0
    while i < n_chunks_repl:

        if i > 0 and i % 300 == 0:
            # if batch number exceed 300 request could fail
            time.sleep(60)

        if max_num_replies <= i*batch_size_replies:
            # if too many replies
            break

        if i != n_chunks_repl - 1:
            batch = replies_ids[i * batch_size_replies:(i + 1) * batch_size_replies]
        else:
            batch = replies_ids[i * batch_size_replies:]

        print(f"Processing REPLIES batch n° {i + 1}/{n_chunks_repl} ...")
        
        list_of_tw_status_reply = api.lookup_statuses(batch, tweet_mode="extended")
        
        replies_batch = []
        for status_reply in list_of_tw_status_reply:
            # print(status_reply)
            if hasattr(status_reply, 'full_text'):
                reply = {
                    "id": status_reply.id,
                    "username": status_reply.user.screen_name,
                    "date": status_reply.created_at,
                    "text": status_reply.full_text.replace('\n', ' ')
                }
                print(reply)
                replies_batch.append(reply)
        i += 1
        replies.append(replies_batch)

    return list(chain.from_iterable(replies))


def get_replies(url):
    # Get the replies to a tweet
    # url: https://twitter.com/realDonaldTrump/status/879058680109879296
    tweet_id = url.split("/")[-1]
    replies = collect_tweet_replies(tweet_id, max_num_replies=100)
    return replies

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)

url_pattern = re.compile('(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})')
tag_pattern = re.compile('(@[\w\d\_]+)')
punc_pattern = re.compile('[“”()\.\":;!@#$%^&*\?\']')
#stop_words =  set(stopwords.words('indonesian'))

with open(os.path.join(script_dir, 'data/slang_word.txt')) as f:
    slang = f.read()
    slang = eval_code(slang)

def clean_text(tweet):
    #use regex to remove url
    tweet = url_pattern.sub('', tweet)
    #use regex to remove username
    tweet = tag_pattern.sub('', tweet)
    #use regex to remove emoji
    tweet = emoji_pattern.sub(r'', tweet)
    tweet = tweet.lower()
    tweet = tweet.replace('  ',' ')
    tweet = tweet.replace(' , ',', ')
    #use regex to remove punctuation
    tweet = punc_pattern.sub('', tweet)
    words = tweet.split()
    #replace slang word
    for i, word in enumerate(words):
        if word in slang.keys():
            words[i] = slang[word]
        
    words = [w.lower() for w in words]
    
    return " ".join(words)

def tokenize(tokenizer, source_text):
    source_text = " ".join(source_text.split())

    source = tokenizer.encode_plus(
        source_text,
        max_length=1024,
        pad_to_max_length=True,
        truncation=True,
        return_attention_mask = True,
        padding="max_length",
        return_tensors="pt",
    )

    source_ids = source["input_ids"]
    source_mask = source["attention_mask"]

    return {
        "source_ids": source_ids.to(dtype=torch.long),
        "source_mask": source_mask.to(dtype=torch.long)}

def sumarize(model, tokenizer, data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids = data['source_ids'].to(device, dtype = torch.long)
    mask = data['source_mask'].to(device, dtype = torch.long)

    generated_ids = model.generate(
        input_ids = ids,
        attention_mask = mask, 
        max_length=150, 
        num_beams=2,
        repetition_penalty=2.5, 
        length_penalty=1.0, 
        early_stopping=True
        )
    preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

    return preds[0]

def summarize_replies(tokenizer, model, url):
    # Summarize the replies
    # replies: list of replies
    # return: list of summarized replies
    fulltext = ''
    replies = get_replies(url)
    for reply in replies:
        fulltext += reply['text'] + '.\n'
    fulltext = clean_text(fulltext)

    tokenized_data = tokenize(tokenizer, fulltext)
    preds = sumarize(model, tokenizer, tokenized_data)

    return preds
