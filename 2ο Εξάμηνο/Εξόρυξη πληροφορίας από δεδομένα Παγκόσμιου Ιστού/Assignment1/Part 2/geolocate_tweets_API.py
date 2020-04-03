import pickle
#pip install opencage
from opencage.geocoder import OpenCageGeocode
from time import time

# Files needed in this part
from files_folders_paths import (path_to_raw_tweets,
                                 path_to_geocoder_results)


# Load file containing tweets' geolocation informations
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
print('There are %d tweets with root coordinates information' %df.coordinates.count())

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


# Indices of tweets having coordinates information
coord_idxs = []
i = 0
for coord in df.coordinates:
    if coord is not None:
        coord_idxs.append(i)
    i += 1


###### Using OpenCage Geocoder to obtain coordinates.
    
# I will take advantage of informations on fields user_location, place_full_name
# and place_country_code
# Combine information from these three fields to create a list of addresses
def combine(dataframe):
    combi = []
    for tweet in dataframe.iterrows():
        cur = []
        # Choose tweets without coordinates information
        if(tweet[0] in coord_idxs):
            combi.append('')
            continue
        else:
            if(tweet[1].user_location is not None):
                cur.append(tweet[1].user_location)
            if(tweet[1].place_full_name is not None):
                cur.append(tweet[1].place_full_name)
            if(tweet[1].place_country_code is not None):
                cur.append(tweet[1].place_country_code)
            combi.append(' '.join(cur))
    return(combi)


# The list the addresses to check. The list will be stored on disk as a tuple,
# together with any already obtained previous results.
# Check if the list already exist along with any previous results.
try:
    combi,previous_results = pickle.load(open(path_to_geocoder_results, "rb"))
except:
    # If not, create a new one and save it on disk
    print('Creating a new list of addresses to be checked...')
    combi = combine(df)
    previous_results = []
    pickle.dump((combi, previous_results), open(path_to_geocoder_results, "wb"))
    print('Done!')


# Procedure to check num (n_places) of addresses from a list (combination), using
# a particular key. Takes about 90 min to check 1000 places
def get_coords(combination, p_results, n_places, geokey):
    geocoder = OpenCageGeocode(geokey)
    print('\nStart searching coordinates for %d new places' %n_places)
    t0 = time()
    t1 = time()
    # Num of allready examined places
    previous = len(p_results)
    count = 0
    # Searh only non examined places
    for place in combination[previous:]:
        if(count % 100 == 0):
            print('\tExamined %d new places in %d sec' %(count, int(time()-t1)))
            t1 = time()
        if(count == n_places):
            break
        if(len(place) > 0):
            result = geocoder.geocode(place)
            # If an actual result is returned
            if(result):
                long = result[0]['geometry']['lng']
                lat = result[0]['geometry']['lat']
                p_results.append([long,lat])
            count+=1
        else:
            p_results.append('')
    print('Total time %d secs' %int(time()-t0))
    return(p_results)


# Available keys. Service allows only 2500 address checks per key per day. So the
# only practical solution is to use multiple keys
keys = [
#        '60657a2b2f774931a3d31fa94465b178',
#        'f9c52765acbb49b5929c25c5144c00e1',
#        '68cbcff98581430b97564c10aa0f9451',
#        '9c6cc4517fb9455ebb4938505deb619c',
#        '5b3391c56bff433089080cc4133535d8',
#        'dfedb5bb0c5f4f45ada1eca81496c46b'
    ]

# Num of new places to examine
new_places = 2499

if(1):
    for key in keys:
        # Load the list of addresses and previous checked ones
        combi,previous_results = pickle.load(open(path_to_geocoder_results, "rb"))
        print('Already examined %d places for coordinates' %len(previous_results))
        # Update with coordinates for new places
        upd_results = get_coords(combi, previous_results, new_places, key)
        print('Examined %d places for coordinates' %len(upd_results))
        # Save updated results
        pickle.dump((combi, upd_results), open(path_to_geocoder_results, "wb"))
    


# Play a sound at the end of procedure
# Requires sox package (sudo apt install sox)
import os
duration = 2  # seconds
freq = 440  # Hz
os.system('play -nq -t alsa synth %d sine %d' %(duration, freq))


# Read a message at the end of procedure
# Requires dispatcher package (sudo apt install speech-dispatcher)
import os
os.system('spd-say "Your program has finished. I repeat, your program has finished"')


######## At the end of procedure save total coordinations on df 

