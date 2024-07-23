# https://pypi.org/project/SoccerNet/

from SoccerNet.Downloader import SoccerNetDownloader

# Path to store data
mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="dataset/soccernet")

# Download SoccerNet videos (require password from NDA to download videos)
# Check https://www.soccer-net.org/data for more details.
mySoccerNetDownloader.password = input("Password for videos? (contact the author):\n")


# Download labels
mySoccerNetDownloader.downloadGames(files=["Labels-v2.json"], split=["valid"])


# Download videos
# 224p Videos
mySoccerNetDownloader.downloadGames(files=["1_224p.mkv", "2_224p.mkv"], split=["valid"])

# 720p Videos
#mySoccerNetDownloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv"], split=["valid"])

# Download frames
#mySoccerNetDownloader.downloadGames(files=["Frames-v3.zip"], split=["train","valid","test"], task="frames")


# Download replay labels
#mySoccerNetDownloader.downloadGames(files=["Labels-cameras.json"], split=["valid"])

#mySoccerNetDownloader.downloadDataTask(task="reid", split=["valid"])





