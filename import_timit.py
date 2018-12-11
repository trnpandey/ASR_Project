#!/usr/bin/env python
'''
MT2017006 Aditi Baghel
MT2017012 Akshita
MT2017037 Deepali Shinde
MT2017128 Tarun Pandey
'''
'''
    NAME    : LDC TIMIT Dataset
    URL     : https://catalog.ldc.upenn.edu/ldc93s1
    HOURS   : 5
    TYPE    : Read - English
    AUTHORS : Garofolo, John, et al.
    TYPE    : LDC Membership
    LICENCE : LDC User Agreement
'''

import errno
import os
from os import path
import sys
import tarfile
import fnmatch
import pandas as pd
import subprocess
import argparse
from mapping import phone_maps
import python_speech_features as psf
import scipy.io.wavfile as wav
import numpy as np
timit_phone_map = phone_maps(mapping_file="kaldi_60_48_39.map")

def clean(word_new):
    # LC ALL & strip punctuation which are not required
    new_word = word_new.lower().replace('.', '')
    new_word = new_word.replace(',', '')
    new_word = new_word.replace(';', '')
    new_word = new_word.replace('"', '')
    new_word = new_word.replace('!', '')
    new_word = new_word.replace('?', '')
    new_word = new_word.replace(':', '')
    new_word = new_word.replace('-', '')
    return new

def compute_mfcc(wav_file, n_delta=0):
    mfcc_feat = psf.mfcc(wav_file)
    if(n_delta == 0):
        return(mfcc_feat)
    elif(n_delta == 1):
        return(np.hstack((mfcc_feat, psf.delta(mfcc_feat,1))))
    elif(n_delta == 2):
        return(np.hstack((mfcc_feat, psf.delta(mfcc_feat,1), psf.delta(mfcc_feat, 2))))
    else:
        return 0

def read_transcript(full_wav):
    trans_file = full_wav[:-8] + ".PHN"
    with open(trans_file, "r") as file:
        trans = file.readlines()
    durations = [ele.strip().split(" ")[:-1] for ele in trans]
    durations_int = []
    for duration in durations:
        durations_int.append([int(duration[0]), int(duration[1])])
    trans = [ele.strip().split(" ")[-1] for ele in trans]
    trans = [timit_phone_map.map_symbol_reduced(symbol=phoneme) for phoneme in trans]
    # trans = " ".join(trans)
    return trans, durations_int

def _preprocess_data(args):
    target = args.timit
    preprocessed = args.preprocessed
    print("Preprocessing data")
    print(preprocessed)
    # Assume data is downloaded from LDC - https://catalog.ldc.upenn.edu/ldc93s1
    # We convert the .WAV (NIST sphere format) into MSOFT .wav
    # creates _rif.wav as the new .wav file
    if(preprocessed):
        print("Data is already preprocessed, just gonna read it")
    full_wavs = []
    full_wavs_train = []
    full_wavs_test = []
    features_test={}
    for root, dirnames, filenames in os.walk(target):
        for filename in fnmatch.filter(filenames, "*.WAV"):
            sph_file = os.path.join(root, filename)
            wav_file = os.path.join(root, filename)[:-4] + "_rif.wav"
            full_wavs.append(wav_file)
            if("TEST" in wav_file):
                full_wavs_test.append(wav_file)
            else:
                full_wavs_train.append(wav_file)
            print("converting {} to {}".format(sph_file, wav_file))
            if(~preprocessed):
                subprocess.check_call(["sox", sph_file, wav_file])

    print("Preprocessing Complete")
    print("Building features")

    mfcc_features = []
    mfcc_labels = []

    # with open("train_wavs", "r") as file:
    #     full_wavs = file.readlines()
    # full_wavs = [ele.strip() for ele in full_wavs]

    for full_wav in full_wavs:
        print("Computing features for file: ", full_wav)

        trans, durations = read_transcript(full_wav = full_wav)
        n_delta = int(args.n_delta)
        labels = []

        (sample_rate,wav_file) = wav.read(full_wav)
        mfcc_feats = compute_mfcc(wav_file[durations[0][0]:durations[0][1]], n_delta=n_delta)

        for i in range(len(mfcc_feats)):
                labels.append(trans[0])
        for index, chunk in enumerate(durations[1:]):
            mfcc_feat = compute_mfcc(wav_file[chunk[0]:chunk[1]], n_delta=n_delta)
            mfcc_feats = np.vstack((mfcc_feats, mfcc_feat))
            for i in range(len(mfcc_feat)):
                labels.append(trans[index])
        mfcc_features.extend(mfcc_feats)
        mfcc_labels.extend(labels)
    #Possibly separate features phone-wise and dump them? (np.where() could be used)
    timit_df = pd.DataFrame()
    timit_df["features"] = mfcc_features
    timit_df["labels"] = mfcc_labels
    
    if(n_delta==0):
        timit_df.to_hdf("./features/mfcc/timit.hdf", "timit")
    elif(n_delta==1):
        timit_df.to_hdf("./features/mfcc_delta/timit.hdf", "timit")
    elif(n_delta==2):
        timit_df.to_hdf("./features/mfcc_delta_delta/timit.hdf", "timit")

    mfcc_features_train=[]
    mfcc_labels_train=[]
    for full_wav in full_wavs_train:
        print("Computing features for file: ", full_wav)
        trans, durations = read_transcript(full_wav = full_wav)
        n_delta = int(args.n_delta)
        labels = []
        (sample_rate,wav_file) = wav.read(full_wav)
        mfcc_feats = compute_mfcc(wav_file[durations[0][0]:durations[0][1]], n_delta=n_delta)
        for i in range(len(mfcc_feats)):
            labels.append(trans[0])
        for index, chunk in enumerate(durations[1:]):
            mfcc_feat = compute_mfcc(wav_file[chunk[0]:chunk[1]], n_delta=n_delta)
            mfcc_feats = np.vstack((mfcc_feats, mfcc_feat))
            for i in range(len(mfcc_feat)):
                labels.append(trans[index])
        mfcc_features_train.extend(mfcc_feats)
        mfcc_labels_train.extend(labels)



    timit_df = pd.DataFrame()
    timit_df["features"] = mfcc_features_train
    timit_df["labels"] = mfcc_labels_train



    if(n_delta==0):
        timit_df.to_hdf("./features/mfcc/timit_train.hdf", "timit")
    elif(n_delta==1):
        timit_df.to_hdf("./features/mfcc_delta/timit_train.hdf", "timit")
    elif(n_delta==2):
        timit_df.to_hdf("./features/mfcc_delta_delta/timit_train.hdf", "timit")



    mfcc_features_test=[]
    mfcc_labels_test=[]
    mfcc_region_test=[]
    mfcc_speaker_test=[]
    mfcc_sentence_test=[]
    for full_wav in full_wavs_test:
        print("Computing features for file: ", full_wav)
        dir_list=full_wav.split("/")
        trans, durations = read_transcript(full_wav = full_wav)
        n_delta = int(args.n_delta)
        labels = []
        temp_region=[]
        temp_speaker=[]
        temp_sentence=[]
        (sample_rate,wav_file) = wav.read(full_wav)
        mfcc_feats = compute_mfcc(wav_file[durations[0][0]:durations[0][1]], n_delta=n_delta)
        for i in range(len(mfcc_feats)):
            labels.append(trans[0])
            temp_region.append(dir_list[-3])
            temp_speaker.append(dir_list[-2])
            temp_sentence.append((dir_list[-1].split("_"))[0])
        for index, chunk in enumerate(durations[1:]):
            mfcc_feat = compute_mfcc(wav_file[chunk[0]:chunk[1]], n_delta=n_delta)
            mfcc_feats = np.vstack((mfcc_feats, mfcc_feat))
            for i in range(len(mfcc_feat)):
                labels.append(trans[index])
                temp_region.append(dir_list[-3])
                temp_speaker.append(dir_list[-2])
                temp_sentence.append((dir_list[-1].split("_"))[0])
        mfcc_features_test.extend(mfcc_feats)
        mfcc_labels_test.extend(labels)
        mfcc_region_test.extend(temp_region)
        mfcc_speaker_test.extend(temp_speaker)
        mfcc_sentence_test.extend(temp_sentence)

    timit_df = pd.DataFrame()
    timit_df["features"] = mfcc_features_test
    timit_df["labels"] = mfcc_labels_test
    timit_df["region"]=mfcc_region_test
    timit_df["speaker"]=mfcc_speaker_test
    timit_df["sentence"]=mfcc_sentence_test

    if(n_delta==0):
        timit_df.to_hdf("./features/mfcc/timit_test.hdf", "timit")
    elif(n_delta==1):
        timit_df.to_hdf("./features/mfcc_delta/timit_test.hdf", "timit")
    elif(n_delta==2):
        timit_df.to_hdf("./features/mfcc_delta_delta/timit_test.hdf", "timit")

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--timit', type=str, default="./",
                       help='TIMIT root directory')
    parser.add_argument('--n_delta', type=str, default="0",
                       help='Number of delta features to compute')
    parser.add_argument('--preprocessed', type=bool, default=False,
                       help='Set to True if already preprocessed')

    args = parser.parse_args()
    print(args)
    print("TIMIT path is: ", args.timit)
    _preprocess_data(args)
    print("Completed")
df = pd.read_hdf("./features/mfcc/timit_train.hdf")
df_delta=pd.read_hdf("./features/mfcc_delta/timit_train.hdf")
df_dd=pd.read_hdf("./features/mfcc_delta_delta/timit_train.hdf")
print(df.head())
features = np.array(df["features"].tolist())
labels = np.array(df["labels"].tolist())
features_delta=np.array(df_delta["features"].tolist())
features_delta_labels=np.array(df_delta["labels"].tolist())
features_dd=np.array(df_dd["features"].tolist())
features_dd_labels=np.array(df_dd["labels"].tolist())
unique_labels=[]
for ite in labels:
	if(ite not in unique_labels):
		unique_labels.append(ite)

number_of_examples=len(labels)
separated={}
separated_d={}
separated_dd={}
for l in unique_labels:
	if(l==""):
		l="space"
	separated[l]=pd.DataFrame()
	separated_d[l]=pd.DataFrame()
	separated_dd[l]=pd.DataFrame()
for i in range(0,number_of_examples):
	if(labels[i]==""):
		labels[i]="space"
	if(features_delta_labels[i]==""):
		features_delta_labels[i]="space"
	if(features_dd_labels[i]==""):
		features_dd_labels[i]="space"
	s1=pd.Series(features[i])
	separated[labels[i]]=separated[labels[i]].append(s1,ignore_index=True)
	s2=pd.Series(features_delta[i])
	separated_d[features_delta_labels[i]] = separated_d[features_delta_labels[i]].append(s2,ignore_index=True)
	s3=pd.Series(features_dd[i])
	separated_dd[features_dd_labels[i]]=separated_dd[features_dd_labels[i]].append(s3,ignore_index=True)

for k in separated:
	fname=str(k)+"_train.hdf"
	separated[k].to_hdf("./features2/mfcc/"+fname, k)
	fname=str(k)+"_train.hdf"
	separated_d[k].to_hdf("./features2/mfcc_delta/"+fname,k)
	fname=str(k)+"_train.hdf"
	separated_dd[k].to_hdf("./features2/mfcc_delta_delta/"+fname,k)






lns=open("unique_labels.txt",'w')
for l in unique_labels:
	if(l==""):
		lns.write("space\n")
	else:
		lns.write(str(l)+"\n")

#Training code
