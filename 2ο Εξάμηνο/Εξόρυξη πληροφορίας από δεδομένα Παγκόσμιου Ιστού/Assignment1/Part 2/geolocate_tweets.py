import pickle
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

# Files needed in this part
from files_folders_paths import (path_to_raw_tweets,
                                 path_to_os_tweets_predictions,
                                 path_to_geoloc_graph)


# Load file containing tweets' geolocation informatations
#df = pickle.load(open(path_to_raw_tweets, "rb"))[['coordinates']]
df = pickle.load(open(path_to_raw_tweets, "rb"))[['coordinates', 'user_location',
                'place_country_code', 'place_full_name','place_coordinates']]

'''
columns available in raw tweets:
['created_at', 'coordinates', 'text', 'retweet_count', 'favorite_count',
       'user_id_str', 'user_screen_name', 'user_followers_count',
       'user_favourites_count', 'user_statuses_count', 'user_listed_count',
       'user_friends_count', 'user_location', 'user_time_zone',
       'user_utc_offset', 'entities_hashtags', 'entities_urls',
       'entities_user_mentions', 'place_country_code', 'place_full_name',
       'place_coordinates']
'''

# Checking ALL fields containing geolocation information
# Checking root coordinates
print('\nThere are %d tweets with root coordinates information' %df.coordinates.count())

# Checking user location
count_loc = 0
for item in df.user_location:
    if item != '':
        count_loc += 1

print('There are %d tweets with location information' %count_loc)

# Checking place_country_code
print('There are %d tweets with place_country_code information' %df.place_country_code.count())

# Checking place_full_name
print('There are %d tweets with place_full_name information' %df.place_full_name.count())

# Checking place_coordinates
print('There are %d tweets with place_coordinates information' %df.place_coordinates.count())

# Get indices of Positive and Negative classified tweets
sentiment = pickle.load(open(path_to_os_tweets_predictions, "rb"))
pos_idx = list(sentiment[sentiment['vote'] == 1].index)
neg_idx = list(sentiment[sentiment['vote'] == 0].index)


# Structures to keep tweets's coordinates according to their assigned sentiment
pos_x_long = []
pos_y_lat = []
neg_x_long = []
neg_y_lat = []


# Scan all tweets for coordinates informations. If existing, keep them to appropriate
# structure according to tweet's sentiment classification.
i = 0
for coord in df.coordinates:
    if coord is not None:
        if(i in pos_idx):
            pos_x_long.append(coord['coordinates'][0])
            pos_y_lat.append(coord['coordinates'][1])
        elif(i in neg_idx):
            neg_x_long.append(coord['coordinates'][0])
            neg_y_lat.append(coord['coordinates'][1])
    i+=1


# Set basic map's parameters
m = Basemap(projection='merc',
            llcrnrlat = -45,
            llcrnrlon = -130,
            urcrnrlat = 70,
            urcrnrlon = 158,
            resolution = 'l'
            )
m.drawcoastlines()
m.fillcontinents(color='whitesmoke',lake_color='lavender')
m.drawcountries()
m.drawmapboundary(fill_color='lavender')
##m.drawstates(color='b')
##m.drawcounties(color='darkred')
#m.etopo()
#m.bluemarble()


# Map (long, lat) to (x, y) for plotting
neg_x_lons, neg_y_lats = m(neg_x_long, neg_y_lat)
pos_x_lons, pos_y_lats = m(pos_x_long, pos_y_lat)
# Set size of export graph
plt.gcf().set_size_inches(15,15)
# Plot points over map. Two different sets of coordinates for the two sentiments
m.plot(neg_x_lons, neg_y_lats, 'ro', markersize=6)
m.plot(pos_x_lons, pos_y_lats, 'o', color='cyan', markersize=6)
# Set plot's title
plt.title('Tweets\' World Distribution', fontsize=20)
# Save graph
plt.savefig(path_to_geoloc_graph)
# Display graph
plt.show()



