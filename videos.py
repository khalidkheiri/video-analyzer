import streamlit as st
from pytube import YouTube
import whisper
from pydub import AudioSegment
from pydub.utils import make_chunks
import os
from collections import Counter
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle, Wedge, Rectangle
import arabic_reshaper
from bidi.algorithm import get_display
from textblob import TextBlob
from googletrans import Translator
from googleapiclient.discovery import build
import pandas as pd
import numpy as np
import re
import datetime
from streamlit_extras.metric_cards import style_metric_cards

st.set_page_config(page_title="Video Analyzer",
                   page_icon=None,
                   layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(
    """
<style>
[data-testid="stMetricValue"] {
    font-size: 50px;
}
[data-testid="stMarkdownContainer"] > p {
    font-size: 20px;
    font-weight: bold;
}
</style>
""",
    unsafe_allow_html=True,
)


def degree_range(n):
    start = np.linspace(0, 180, n + 1, endpoint=True)[0:-1]
    end = np.linspace(0, 180, n + 1, endpoint=True)[1::]
    mid_points = start + ((end - start) / 2.)
    return np.c_[start, end], mid_points
def rot_text(ang):
    rotation = np.degrees(np.radians(ang) * np.pi / np.pi - np.radians(90))
    return rotation
def gauge(labels=['LOW', 'MEDIUM', 'HIGH', 'VERY HIGH', 'EXTREME'], \
            colors='jet_r', arrow=1, title='', fname=False):


    N = len(labels)

    if arrow > N:
        raise Exception("\n\nThe category ({}) is greated than \
        the length\nof the labels ({})".format(arrow, N))


    if isinstance(colors, str):
        cmap = cm.get_cmap(colors, N)
        cmap = cmap(np.arange(N))
        colors = cmap[::-1, :].tolist()
    if isinstance(colors, list):
        if len(colors) == N:
            colors = colors[::-1]
        else:
            raise Exception("\n\nnumber of colors {} not equal \
            to number of categories{}\n".format(len(colors), N))


    fig, ax = plt.subplots()

    ang_range, mid_points = degree_range(N)

    labels = labels[::-1]

    patches = []
    for ang, c in zip(ang_range, colors):
        # sectors
        patches.append(Wedge((0.,0.), .4, *ang, facecolor='w', lw=2))
        # arcs
        patches.append(Wedge((0., 0.), .4, *ang, width=0.10, facecolor=c, lw=2, alpha=0.5))

    foo = [ax.add_patch(p) for p in patches]


    for mid, lab in zip(mid_points, labels):
        ax.text(0.35 * np.cos(np.radians(mid)), 0.35 * np.sin(np.radians(mid)), lab, \
                horizontalalignment='center', verticalalignment='center', fontsize=14, \
                fontweight='bold', rotation=rot_text(mid))

    r = Rectangle((-0.4, -0.1), 0.8, 0.1, facecolor='w', lw=2)
    ax.add_patch(r)

    ax.text(0, -0.05, title, horizontalalignment='center', \
            verticalalignment='center', fontsize=22, fontweight='bold')

    pos = mid_points[abs(arrow - N)]

    ax.arrow(0, 0, 0.225 * np.cos(np.radians(pos)), 0.225 * np.sin(np.radians(pos)), \
                width=0.04, head_width=0.09, head_length=0.1, fc='k', ec='k')

    ax.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
    ax.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))
    plt.title("Video Score", fontsize = 20)
    ax.set_frame_on(False)
    ax.axes.set_xticks([])
    ax.axes.set_yticks([])
    ax.axis('equal')
    #plt.tight_layout()
    col2.write(fig)
    if fname:
        fig.savefig(fname, dpi=200)


st.title("Video Analysis")

link = st.sidebar.text_input("Youtube Video URL:")
if link:
    video_id = link.split("?")[1].split("&")[0].split("=")[1]

api_key = "AIzaSyC8BsyJF1-qSPeOh3mWZRiBw8Df-YIDMCI"
service = build("youtube", "v3", developerKey=api_key)

def get_all_comments(service, video_id):
    request = service.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100
    )
    response = request.execute()

    comments = []
    for item in response["items"]:
        comment = item["snippet"]["topLevelComment"]
        comments.append({
            "authorDisplayName": comment["snippet"]["authorDisplayName"],
            "textDisplay": comment["snippet"]["textDisplay"],
            "likeCount": comment["snippet"]["likeCount"],
            "publishedAt": comment["snippet"]["publishedAt"],
            "updatedAt": comment["snippet"]["updatedAt"],
        })

    while (1 == 1):
        try:
            nextPageToken = response["nextPageToken"]
        except KeyError:
            break
        nextPageToken = response["nextPageToken"]
        nextRequest = service.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        pageToken=nextPageToken
        )
        response = nextRequest.execute()
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]
            comments.append({
                "authorDisplayName": comment["snippet"]["authorDisplayName"],
                "textDisplay": comment["snippet"]["textDisplay"],
                "likeCount": comment["snippet"]["likeCount"],
                "publishedAt": comment["snippet"]["publishedAt"]
            })

    df = pd.DataFrame(comments)
    return df
def get_duration_time(x):
    hours_pattern = re.compile(r"(\d+)H")
    minutes_pattern = re.compile(r"(\d+)M")
    seconds_pattern = re.compile(r"(\d+)S")

    hours = hours_pattern.search(x)
    minutes = minutes_pattern.search(x)
    seconds = seconds_pattern.search(x)

    hours = int(hours.group(1)) if hours else 0
    minutes = int(minutes.group(1)) if minutes else 0
    seconds = int(seconds.group(1)) if seconds else 0

    return datetime.time(hours,minutes,seconds)
def get_video_details(service, video_id):
    request = service.videos().list(
        part=["snippet","statistics","contentDetails"],
        id=video_id
    )
    response = request.execute()

    for item in response["items"]:
        data = dict(title = item["snippet"]["title"],
                    channel_title = item["snippet"]["channelTitle"],
                    tags = item["snippet"]["tags"],
                    duration = get_duration_time(item["contentDetails"]["duration"]),
                    view_count = item["statistics"]["viewCount"],
                    like_count = item["statistics"]["likeCount"],
                    comment_count = item["statistics"]["commentCount"])
    return data

btn = st.sidebar.button("Analyze")
# Download the Video and Divide it into Segments, and get transcripts text Variable
if btn:
    bar_1 = st.empty()
    bar_1 = st.progress(0)
    progress_status_1 = st.empty()
    try:
        yt = YouTube(link)
    except:
        st.error("Connection error")

    yt.streams.filter(file_extension="mp3")
    stream = yt.streams.get_by_itag(139)
    stream.download("", "GoogleImagen.mp3")
    
    audio_file_path = "GoogleImagen.mp3"
    chunk_length_ms = 120000

    audio = AudioSegment.from_file(audio_file_path)
    chunks = make_chunks(audio, chunk_length_ms)

    model = whisper.load_model("base")
    transcripts = ""
    for index,chunk in enumerate(chunks):
        percent = int((index+1)/len(chunks)*100)
        progress_status_1.write(str(percent) + "%")
        bar_1.progress(percent)

        chunk_name = "chunk.mp3"
        chunk.export(chunk_name, format="mp3")
        result = model.transcribe(chunk_name, fp16=False)
        text = result["text"]
        transcripts += text
        os.remove(chunk_name)

    os.remove("GoogleImagen.mp3")
    
####################################################################################################################

### Sidebar
if btn:
    width = st.sidebar.slider("plot width", 1, 25, 8)
    height = st.sidebar.slider("plot height", 1, 25, 5)

    st.sidebar.markdown("---")

    detail = get_video_details(service, video_id)

    st.sidebar.markdown(f"Title: **{detail['title']}**")
    st.sidebar.markdown(f"Channel Title: **{detail['channel_title']}**")
    st.sidebar.markdown(f"Duration: **{detail['duration']}**")
    st.sidebar.markdown(f"Tags: **{detail['tags']}**")


### Row 1
    x1 = int(detail["view_count"])
    x2 = int(detail["like_count"])
    x3 = int(detail["comment_count"])

# ### Row2
# Draw the Most Words Count in Bar Chart
    if transcripts:
        stopwords = ['في', 'من', 'هذا', 'أن', 'على','ما','ان','عن','أو','إن','لم','فيه']
        querywords = transcripts.split()
        resultwords  = [word for word in querywords if word not in stopwords]

        result = Counter(resultwords).most_common(8)

        words = []
        count = []
        for key,val in result:
            words.append(key)
            count.append(val)

        xx = []
        for word in words:
            xx.append(get_display(arabic_reshaper.reshape(word)))

    # For Sentiment Analysis
    translator = Translator()

    df = get_all_comments(service, video_id)
    df['textDisplay'] = df['textDisplay'].str.replace(r'<[^<>]*>', '', regex=True)


    for index,text in enumerate(df['textDisplay']):
        try:
            df.loc[index,"translation"] = translator.translate(text, dest="en").text
            df.loc[index,"score"] = round(TextBlob(df.loc[index,"translation"]).sentiment.polarity,4)*100
        except:
            df.loc[index,"translation"] = ""
            df.loc[index,"score"] = 0
            continue

        percent = int((index+1)/df['textDisplay'].size*100)
        progress_status_1.write(str(percent) + "%")
        bar_1.progress(percent)
    bar_1.empty()
    progress_status_1.empty()

    df["publishedAt"] = df["publishedAt"].apply(lambda a: a.split("T")[0])

    df["category"] = df["score"].apply(lambda a: "High Positive" if a>=70 and a<=100 else
                                       "Mid Positive" if a>=40 and a<70 else
                                       "Positive" if a>=10 and a<40 else
                                       "Neutral" if a<10 and a>-10 else
                                       "Negative" if a>=-40 and a<=-10 else
                                       "Mid Negative" if a>=-70 and a<-40 else
                                       "High Negative" if a>=-100 and a<-70 else None)
    score_mean = round(df["score"].mean(),0)

    aa = 7 if score_mean>=70 and score_mean<=100 else \
        6 if score_mean>=40 and score_mean<70 else \
        5 if score_mean>=10 and score_mean<40 else \
        4 if score_mean<10 and score_mean>-10 else \
        3 if score_mean>=-40 and score_mean<=-10 else \
        2 if score_mean>=-70 and score_mean<-40 else \
        1 if score_mean>=-100 and score_mean<-70 else None


### Row 3
    cats = ["High Negative","Mid Negative","Negative","Neutral","Positive","Mid Positive","High Positive"]
    values = [0,0,0,0,0,0,0]
    for index,c in enumerate(cats):
        for cat, value in df["category"].value_counts().items():
            if c == cat:
                values[index]=value

### Row 4
    df[["year", "month", "day"]] = df["publishedAt"].str.split("-", expand=True)

    grp = df.groupby(["day","month"])

    like_monthly = grp["likeCount"].sum()
    score_monthly = grp["score"].mean()
    comment_monthly = df.groupby(["day","month"]).size()

    x_axis = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16',
              '17','18','19','20','21','22','23','24','25','26','27','28','29','30','31',
              '01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16',
              '17','18','19','20','21','22','23','24','25','26','27','28','29','30','31',
              '01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16',
              '17','18','19','20','21','22','23','24','25','26','27','28','29','30','31',
              '01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16',
              '17','18','19','20','21','22','23','24','25','26','27','28','29','30','31',
              '01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16',
              '17','18','19','20','21','22','23','24','25','26','27','28','29','30','31',
              '01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16',
              '17','18','19','20','21','22','23','24','25','26','27','28','29','30','31',
              '01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16',
              '17','18','19','20','21','22','23','24','25','26','27','28','29','30','31',
              '01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16',
              '17','18','19','20','21','22','23','24','25','26','27','28','29','30','31',
              '01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16',
              '17','18','19','20','21','22','23','24','25','26','27','28','29','30','31',
              '01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16',
              '17','18','19','20','21','22','23','24','25','26','27','28','29','30','31',
              '01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16',
              '17','18','19','20','21','22','23','24','25','26','27','28','29','30','31',
              '01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16',
              '17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']

    y_axis = ['01','02','03','04','05','06','07','08','09','10','11','12',
              '01','02','03','04','05','06','07','08','09','10','11','12',
              '01','02','03','04','05','06','07','08','09','10','11','12',
              '01','02','03','04','05','06','07','08','09','10','11','12',
              '01','02','03','04','05','06','07','08','09','10','11','12',
              '01','02','03','04','05','06','07','08','09','10','11','12',
              '01','02','03','04','05','06','07','08','09','10','11','12',
              '01','02','03','04','05','06','07','08','09','10','11','12',
              '01','02','03','04','05','06','07','08','09','10','11','12',
              '01','02','03','04','05','06','07','08','09','10','11','12',
              '01','02','03','04','05','06','07','08','09','10','11','12',
              '01','02','03','04','05','06','07','08','09','10','11','12',
              '01','02','03','04','05','06','07','08','09','10','11','12',
              '01','02','03','04','05','06','07','08','09','10','11','12',
              '01','02','03','04','05','06','07','08','09','10','11','12',
              '01','02','03','04','05','06','07','08','09','10','11','12',
              '01','02','03','04','05','06','07','08','09','10','11','12',
              '01','02','03','04','05','06','07','08','09','10','11','12',
              '01','02','03','04','05','06','07','08','09','10','11','12',
              '01','02','03','04','05','06','07','08','09','10','11','12',
              '01','02','03','04','05','06','07','08','09','10','11','12',
              '01','02','03','04','05','06','07','08','09','10','11','12',
              '01','02','03','04','05','06','07','08','09','10','11','12',
              '01','02','03','04','05','06','07','08','09','10','11','12',
              '01','02','03','04','05','06','07','08','09','10','11','12',
              '01','02','03','04','05','06','07','08','09','10','11','12',
              '01','02','03','04','05','06','07','08','09','10','11','12',
              '01','02','03','04','05','06','07','08','09','10','11','12',
              '01','02','03','04','05','06','07','08','09','10','11','12',
              '01','02','03','04','05','06','07','08','09','10','11','12',
              '01','02','03','04','05','06','07','08','09','10','11','12']

    another_x = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16',
                 '17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']
    another_y = ['01','02','03','04','05','06','07','08','09','10','11','12']

    likes = []
    scores = []
    comments = []

    for x in another_x:
        for y in another_y:
            likes.append(like_monthly.get((x, y),0))

    for x in another_x:
        for y in another_y:
            scores.append(score_monthly.get((x, y),0))

    for x in another_x:
        for y in another_y:
            comments.append(comment_monthly.get((x, y),0))

    
########### OUTPUTS

### Row 1
    col1, col2, col3 = st.columns(3, gap="large")
    col1.metric(label="Views", value=f"{x1:,}")
    col2.metric(label="Like", value=f"{x2:,}")
    col3.metric(label="Comments", value=f"{x3:,}")
    style_metric_cards()

    st.markdown("---")
### Row 2
    col1, col2 = st.columns(2, gap="medium")
    fig = plt.figure(figsize=(width, height))
    a = plt.barh(xx,count)
    plt.bar_label(a)
    col1.write(fig)

    fig = plt.figure()
    gauge(labels=['High\nNeg.', 'Mid\nNeg.', 'Neg.', 'Neutral', 'Pos.', 'Mid\nPos.', 'High\nPos.'], \
        colors='RdBu', arrow=aa, title=f"{str(score_mean)}%")

    st.markdown("---")
### Row 3
    col1, col2 = st.columns(2, gap="medium")
    col1.dataframe(df[["authorDisplayName","textDisplay","likeCount","publishedAt","score", "category"]])

    fig = plt.figure()
    a = plt.bar(cats,values, color = ['#0c8a30','#12c445','#28fc65','#917214','#fc4128','#c22f1b','#912314'], width = 0.5)
    plt.bar_label(a)
    plt.gcf().autofmt_xdate()
    col2.write(fig)

    st.markdown("---")
### Row 4
    fig = plt.figure(figsize=(14, 4))
    plt.style.use('seaborn-v0_8')
    plt.scatter(x_axis, y_axis, s=likes, c=comments, cmap="Greens", edgecolors="black", linewidths=1, alpha=0.75)
    cbar = plt.colorbar()
    cbar.set_label("# Comments")
    plt.title("Bubble Size Represents the Number of Likes")
    plt.xlabel("Days")
    plt.ylabel("Months")
    st.write(fig)


