import pandas as pd
import numpy as np
#from tqdm import *
from collections import defaultdict
import mne
import seaborn as sns
import matplotlib.pyplot as plt

def label_TUH(dataFrame=False, window=[0, 0], header=None,channel=None):  # saveDir=os.getcwd(),
    df=dataFrame
    within_con0 = (pd.to_numeric(df['t_start']) <= window[0]) & (window[0] <= pd.to_numeric(df['t_end']))
    within_con1 = (pd.to_numeric(df['t_start']) <= window[1]) & (window[1] <= pd.to_numeric(df['t_end']))

    if channel:
        low_char = {'FP1': 'Fp1', 'FP2': 'Fp2', 'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz'}
        #If channel name is in dictionary, name is changed to small end character.
        for i in range(len(df['channel'])):
            if df['channel'][i] in low_char.keys():
                df['channel'][i] = low_char[df['channel'][i]]

        label_TUH = df[(pd.to_numeric(df['t_start']).between(window[0], window[1]) |
                       pd.to_numeric(df['t_end']).between(window[0], window[1]) |
                       (within_con0 & within_con1))
                       & (df['channel']==channel)
                        & (df['label'].to_numpy()=='elec')]
                        #Handle double labels in the solveLabelChannelRelations function, so only keeping
                        # these in case we would like other checks.
                        # |(df['label'].to_numpy()=='musc_elec')|(df['label'].to_numpy()=='eyem_elec')|
                          # (df['label'].to_numpy()=='shiv_elec')|(df['label'].to_numpy()=='chew_elec'))]
                        #(np.asarray(chan_names)==np.asarray(channel)).tolist()
    else:
        label_TUH = df[df['t_start'].between(window[0], window[1]) |
                   df['t_end'].between(window[0], window[1]) |
                   (within_con0 & within_con1)]
    return_list = label_TUH.to_numpy().tolist()  # Outputter kun listen af label navne i vinduet, fx ["eyem", "null"]
    if return_list==[]:
        return_list=['null']
    elif channel:
        return_list=['elec']
    return return_list


# The function "annotate_TUH()" takes a raw signal and a anntot for a csv file with annotations/labels in it.
# The annotations are read and added to the raw signal. The function is mainly made for the purpose of making
# plots with the artifacts showing.
def annotate_TUH(raw,df=None):
    t_start=df['t_start'].to_numpy().astype(float)
    dura=df['t_end'].to_numpy().astype(float)-t_start
    labels=df['label'].to_numpy().tolist()
    chan_names=df['channel'].to_numpy().tolist()
    t_start=t_start.tolist()
    dura=dura.tolist()

    delete=[]
    low_char={'FP1':'Fp1', 'FP2':'Fp2', 'FZ':'Fz', 'CZ':'Cz', 'PZ':'Pz'}
    for i in range(len(chan_names)):
        # Loop through all channel names in reverse order, so if something is removed it does not affect other index.
        # Change certain channels to have smaller letters:
        if chan_names[i] in low_char:
            chan_names[i]=low_char[chan_names[i]]

        # If channel names are not in the raw info their are removed from an annotation:
        if chan_names[i] not in raw.ch_names:
            delete.append(i)


    #removes every annotation that cannot be handled backwards:
    for ele in sorted(delete,reverse=True):
        print(f"Annotation {labels[ele]} on non-existing channel {chan_names[ele]} removed from annotations.")
        del t_start[ele], dura[ele],labels[ele],chan_names[ele]

    chan_names=[[ch] for ch in chan_names]

    anno=mne.Annotations(onset=t_start,
                            duration=dura,
                              description=labels,
                                ch_names=chan_names)

    raw_anno=raw.set_annotations(anno)
    return raw_anno


def solveLabelChannelRelation(annoPath, header = None, plot=False):
    print(f"Annotation: {annoPath}")
    df = pd.read_csv(annoPath, sep=",", skiprows=6, header=header)
    if len(df.columns)==5:
        # first row in this setup is just the column names so, first column is removed:
        df = df.drop(index=0)
    else:
        #if this setup (6 columns), the first column contains file names, that we don't care about
        df = df.drop(0,axis=1)
    df.columns = ['channel', 'start_time', 'stop_time', 'label', 'confidence']

    # Find all double labels eg. "eyem_elec" and split them to two seperate annotations:
    double_label_temp_df=pd.DataFrame(columns=['channel', 'start_time', 'stop_time', 'label'])
    double_labels=df[df['label'].str.len()==9]
    if not double_labels.empty:
        for i in double_labels.index:
            label1,label2=df['label'][i].split('_')

            rows_two_labels = pd.DataFrame({'channel': [df['channel'][i],df['channel'][i]], 'start_time': [df['start_time'][i],df['start_time'][i]],
                                             'stop_time': [df['stop_time'][i],df['stop_time'][i]], 'label': [label1, label2]})


            double_label_temp_df = pd.concat([double_label_temp_df, rows_two_labels], ignore_index=True)

            df = df.drop(index=i)
        #Join the df with removed doublelabels with the dataframe with the separated single annotations:
        df=pd.concat([df, rows_two_labels], ignore_index=True)

    #Creating data frame:
    anno_df=pd.DataFrame(columns=['channel','t_start','t_end','label'])

    if plot:
        # Should we drop the NaN values?
        plotmat = np.zeros((len(df), int(round(df['stop_time'].max()*256,0))))

        for i in range(len(df)):
            plotmat[i, int(round(df['start_time'][i] * 256, 0)):int(round(df['stop_time'][i] * 256, 0))] = 1

    #checking every entry in label data:
    for i in df.index:
        chan1, chan2=df['channel'][i].split('-')
        # Only check row against rows further down:
        temp = df[i+1:]
        # Only rows with same label:
        temp = temp[temp['label'] == df['label'][i]]

        # Only overlap in time:
        temp_time = temp[((df['start_time'][i]<=temp['start_time']) & (temp['start_time']<=df['stop_time'][i])) |
                         ((df['start_time'][i]<=temp['stop_time']) & (temp['stop_time']<=df['stop_time'][i])) |
                         ((temp['start_time']<df['start_time'][i]) & (df['stop_time'][i]<temp['stop_time']))]

        for k in temp_time.index:
            #check if first channel is a match with one in the new channel pair:

            channel = None
            if chan1 in temp_time['channel'][k].split('-'):
                channel = chan1
            elif chan2 in temp_time['channel'][k].split('-'):
                channel = chan2
            if channel in [chan1, chan2]:
                t_start = max(df['start_time'][i], temp_time['start_time'][k])
                t_end = min(df['stop_time'][i], temp_time['stop_time'][k])

                #Find all entries in the new annotation dataframe where there is a match of label and found channel (two
                # first checks). Then check that there is an overlap in time (three checks of overlap: start time within
                # interval, end time within interval or comparison signal k is larger and lies around the signal i.)
                duplicates=anno_df[ ((df['label'][i]==anno_df['label']) &
                         (channel==anno_df['channel'])  &
                        (((t_start<=anno_df['t_start']) & (anno_df['t_start']<=t_end)) |
                         ((t_start<=anno_df['t_end']) & (anno_df['t_end']<=t_end)) |
                         ((anno_df['t_start']<t_start) & (t_end<anno_df['t_end']))))]

                if not duplicates.empty:
                    new_t_start = min(duplicates['t_start'].to_numpy().tolist()+[t_start])
                    new_t_end = max(duplicates['t_end'].to_numpy().tolist()+[t_start])

                    #delete overlapping rows from behind so the indexes are not confused:
                    for n in range(len(duplicates)):
                        index=duplicates.index[-n]
                        anno_df=anno_df.drop(index=index)

                    #Wait to concatenate new row to dataframe, since the indexes are ignored, meaning the duplicates
                    # get different indexes and cannot be removed unless this order is used.
                    anno_new = pd.DataFrame({'channel': [channel], 't_start': [new_t_start],
                                             't_end': [new_t_end], 'label': [df['label'][i]]})
                    if plot:
                        # Heatmap matrix
                        plotmat[i, int(round(anno_new['t_start'] * 256, 0)):int(round(anno_new['t_end'] * 256, 0))] = 2
                        plotmat[k, int(round(anno_new['t_start'] * 256, 0)):int(round(anno_new['t_end'] * 256, 0))] = 2
                    anno_df = pd.concat([anno_df, anno_new], ignore_index=True)
                # if no duplicates/overlaps found, then just save annotation for channel:
                else:
                    anno_new = pd.DataFrame({'channel': [channel], 't_start': [t_start],
                                             't_end': [t_end], 'label': [df['label'][i]]})
                    if plot:
                        plotmat[i, int(round(anno_new['t_start'] * 256, 0)):int(round(anno_new['t_end'] * 256, 0))] = 2
                        plotmat[k, int(round(anno_new['t_start'] * 256, 0)):int(round(anno_new['t_end'] * 256, 0))] = 2
                    anno_df = pd.concat([anno_df,anno_new],ignore_index=True)

    if plot:
        sns.heatmap(plotmat)
        plt.show()

    print(anno_df)
    return anno_df

def labelChannels(annoPath, header = None):
    df = pd.read_csv(annoPath, sep=",", skiprows=6, header=header)

    # Split pairs into single channels
    channel_pairs = df[1].to_numpy().tolist()
    channel_pairs = [n.split('-') for n in channel_pairs]

    # Creating data frame:
    anno_df = pd.DataFrame(columns=['channel', 't_start', 't_end', 'label'])

    anno_dict = defaultdict(lambda: (0, 0))

    # Checking every entry in label data:
    for i in range(len(channel_pairs)):
        # Check if label is the same in the two rows, eg. 'elec'=='elec':
        # Create two variables, one for each channel in the pair:
        chan1, chan2 = channel_pairs[i]
        for k in range(i+1,len(channel_pairs)):
            #Check if label is the same in the two rows, eg. 'elec'=='elec':
            if df[4][i] == df[4][k]:
                #Add both time frames to anno_dict
               anno_dict[chan1] = (df[2][i], df[3][i])
               anno_dict[chan2] = (df[2][i], df[3][i])



if __name__ == "__main__":
    path = "../TUH_data_sample/131/00013103/s001_2015_09_30/00013103_s001_t000.csv"

    solveLabelChannelRelation(annoPath=path, header=None, plot=True)

    #labelChannels(annoPath=path)