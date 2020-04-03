import pandas as pd
import numpy as np
import csv


# Primary keys from parcel's table
parsel_id = pd.read_csv('parcels_id.csv', header=None).iloc[:][0]

## Create data for people table
# I will create 10.000 people and assign them to random parsel_id
# data must be at the form (people_name, parsel_id)
id_choices = np.random.choice(parsel_id, size=10000, replace=True)
peoples = []
for i in range(10001):
    if i == 0:
        continue
    name = 'people_'+str(i)
    peoples.append([name,id_choices[i-1]])

# Save data as .csv file
with open("peoples.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(peoples)


## Create data for plants table
# I will create 15 plants. Data must be at the form (plant_name, season)
seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
season_choices = list(np.random.choice(seasons, size=15, replace=True))
plants_names = []
plants = []
for i in range(16):
    if i == 0:
        continue
    plant = 'plant_'+str(i)
    plants_names.append(plant)
    plants.append([plant,str(season_choices[i-1])])

# Save data as .csv file
with open("plants.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(plants)


## Create data for annual_performances table
# For each parsel I will choose 3 random plants and a random performance for
# each plant. Data must be at the form (id, parsel_id, plant_id, performance)
an_perf = []
idx = 1
for parcel in parsel_id:
    plants = list(np.random.choice(plants_names, size=3, replace=False))
    for plant in plants:
        perf = np.random.choice(range(100,1000), size=1)
        an_perf.append([idx,parcel,str(plant),perf[0]])
        idx+=1

# Save data as .csv file
with open("annual_performance.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(an_perf)


## Create data for parc_cult_plants table
# For each parsel I will choose 3 random plants. Data must be at the form
# (parsel_id, plant_id)
p_c_p = []
for parcel in parsel_id:
    plants = list(np.random.choice(plants_names, size=3, replace=False))
    for plant in plants:
        p_c_p.append([parcel,str(plant)])

# Save data as .csv file
with open("parc_cult_plants.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(p_c_p)


