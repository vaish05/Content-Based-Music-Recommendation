import numpy as np
import scipy.spatial.distance as scspdi
import matplotlib.pyplot as plt
import pandas as pd
import pickle


choice = "CNN - Small"
#Each choice usese a different set of files, the following if-else block loads the corresponding files
if choice == "CNN - Small":
    with open('Data\\x_test_sm_genre_cnn.csv', 'rb') as f: data_test = np.load(f)
    with open('Data\\x_train__sm_genre_cnn.csv', 'rb') as f: data_train = np.load(f)
    with open('Genre\\y_train_sm.pkl', 'rb') as f: genre_train = np.array(pickle.load(f))
    with open('Genre\\y_test_sm.pkl', 'rb') as f: genre_test = np.array(pickle.load(f))
    with open('Label\\file_test_sm.pkl', 'rb') as f: label_test = np.array(pickle.load(f))
    with open('Label\\file_train_sm.pkl', 'rb') as f: label_train = np.array(pickle.load(f))

elif choice == "CNN - Medium":
    with open('Data\\x_test_md_genre_cnn1.csv', 'rb') as f: data_test = np.load(f)
    with open('Data\\x_train_md_genre_cnn1.csv', 'rb') as f: data_train = np.load(f)
    with open('Genre\\y_train_md.pkl', 'rb') as f: genre_train = np.array(pickle.load(f))
    with open('Genre\\y_test_md.pkl', 'rb') as f: genre_test = np.array(pickle.load(f))
    with open('Label\\file_test_md.pkl', 'rb') as f: label_test = np.array(pickle.load(f))
    with open('Label\\file_train_md.pkl', 'rb') as f: label_train = np.array(pickle.load(f))

elif choice == "RNN - Small":
    with open('Data\\X_test_genre_sm_30.csv', 'rb') as f: data_test = np.load(f)
    with open('Data\\X_train_genre_sm_30.csv', 'rb') as f: data_train = np.load(f)
    with open('Genre\\y_train_sm_30.pkl', 'rb') as f: genre_train = np.array(pickle.load(f))
    with open('Genre\\y_test_sm_30.pkl', 'rb') as f: genre_test = np.array(pickle.load(f))
    with open('Label\\file_test_sm_30.pkl', 'rb') as f: label_test = np.array(pickle.load(f))
    with open('Label\\file_train_sm_30.pkl', 'rb') as f: label_train = np.array(pickle.load(f))

elif choice == "RNN - Medium":
    with open('Data\\X_test_genre_2.csv', 'rb') as f: data_test = np.load(f)
    with open('Data\\X_train_genre_2.csv', 'rb') as f: data_train = np.load(f)
    with open('Genre\\y_train_md.pkl', 'rb') as f: genre_train = np.array(pickle.load(f))
    with open('Genre\\y_test_md.pkl', 'rb') as f: genre_test = np.array(pickle.load(f))
    with open('Label\\file_test_md.pkl', 'rb') as f: label_test = np.array(pickle.load(f))
    with open('Label\\file_train_md.pkl', 'rb') as f: label_train = np.array(pickle.load(f))


# Function to find the 5 most similar songs to a given song
def measure(dist):
    minim = {}
    c = 0
    label = {}
    for i in enumerate(genre_train): #Creates a dictionary mapping each song name to a genre
        label[i[0]] = label_train[i[0]], i[1]
    minim = {}
    for i in range(dist.shape[0]): #Loop to get the 5 most similar songs
        for j in dist[i].argsort()[:5]:
            if i in minim:
                minim[i].append(label[j])
            else:
                minim[i] = [label[j]]
    return minim


#euclid = np.zeros((X_test.shape[0], X_train.shape[0]))
cos = np.zeros((data_test.shape[0], data_train.shape[0]))
for i in range(data_test.shape[0]):
    for j in range(data_train.shape[0]):
        #euclid[i, j] = scspdi.euclidean(X_test[i], X_train[j])
        cos[i, j] = scspdi.cosine(data_test[i], data_train[j]) #Finds the Cosine Similarity between each song


#euc_match = measure(euclid)
cos_match = measure(cos)

lab = {}
for i in enumerate(genre_test):
        lab[i[0]] = label_test[i[0]], i[1]
#match_df = pd.DataFrame(list(euc_match.items()), columns = ['Selected Song', 'Similar Songs'])
match_df = pd.DataFrame(list(cos_match.items()), columns = ['Selected Song', 'Similar Songs']) #Creating a Dataframe 
genre_cnt = {}
c = 0
for i in match_df['Similar Songs']: #To keep track of the labels of the similar songs
    for j in i:
        if c in genre_cnt:
            genre_cnt[c].append(j[1])
        else:
            genre_cnt[c] = [j[1]]
    c += 1
gen = {}
c = 0
for i in match_df['Selected Song']: #Converting list to string
    gen[c] = ' - '.join(lab[c])
    c += 1
match_df['Score'] = pd.Series(genre_cnt, index = match_df.index)
match_df['Selected Song'] = pd.Series(gen, index = match_df.index)
score = {}
c = 0
for i in match_df['Score']: #Finding Score in the range 0 to 1
    for j in i:
        if j == match_df['Selected Song'][c][match_df['Selected Song'][c].find('-') + 2:]:
            if c in score:
                score[c] += 1
            else:
                score[c] = 1
    if c in score:
        score[c] = score[c] / 5
    else:
        score[c] = 0
    c += 1
match_df['Score'] = pd.Series(score, index = match_df.index)
sim = {}
c = 0
for i in match_df['Similar Songs']: #Converting list to string
    for j in i:
        if c in sim:
            sim[c].append(' - '.join(j))
        else:
            sim[c] = [' - '.join(j)]
    c += 1
match_df['Similar Songs'] = pd.Series(sim, index = match_df.index)


print (match_df)

np.mean(match_df['Score'])#, np.std(match_df['Score'])


#Cherry-picking the most best-looking values
pick = match_df['Selected Song'][match_df['Score'] == 0.6]
for i, j in gen.items():
    if j == pick.iloc[9]:
        index = i
#1, 4, 9
print (pick)


labels = list(np.unique(genre_train))
sizes = data_test[index, ]
txt = []
for i in enumerate(sizes): #Creating the legend for the donut plot
    txt.append(labels[i[0]] + ' - ' + str(round(i[1] * 100, 3)) + '%')


#Code for the donut plot
fig, ax = plt.subplots(figsize = (15, 10), subplot_kw = dict(aspect = "equal")) 

cmap = plt.get_cmap('viridis')
colors = [cmap(i) for i in np.linspace(0, 1, len(labels))]
wedges, texts = ax.pie(sizes, colors = colors, wedgeprops = dict(width = 0.5), startangle = 90)

bbox_props = dict(boxstyle = "square, pad = 0.3", fc = "w", ec = "k", lw = 0.72)

ax.set_title("Genre Membership for track - " + match_df['Selected Song'][index][:match_df['Selected Song'][index].find('-') - 1] 
             + " which is a " + match_df['Selected Song'][index][match_df['Selected Song'][index].find('-') + 2:] + " song.")

legend = ax.legend(wedges, txt,
          prop={'size': 15},
          loc = "center left",
          bbox_to_anchor = (1, 0, 0.5, 1))
legend.set_title("Genre Membership", prop = {'size':'xx-large'})

plt.show()




f = [match_df['Selected Song'][index]]
for i in match_df['Similar Songs'][index]:
    f.append(i)
print (f) #Print out the top 5 similar songs


#Create a playlist folder
import os, shutil

if "Small" in choice: #Only looking at small because my system had only the small dataset
    src = "fma_small"
    dest = "playlist"
    if os.path.isfile(dest):
        os.unlink(dest)
    f = [match_df['Selected Song'][index][:match_df['Selected Song'][index].find('-') - 1]]
    for i in match_df['Similar Songs'][index]:
        f.append(i[:i.find('-') - 1])
    print(f)
    for file in f:
        full_path = 'fma_small\\'+file[:3]+'\\'+file
        shutil.copy(full_path, dest)
        print(file)
