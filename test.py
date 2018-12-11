'''
MT2017006 Aditi Baghel
MT2017012 Akshita
MT2017037 Deepali Shinde
MT2017128 Tarun Pandey
'''
import numpy as numpy
import pandas as pd
import os
import pickle


def get_features(data_new):
	features=[]
	for i in range(0,len(data_new)):
		instance=data_new['features'][i].tolist()
		features.append(instance)
	feature1=pd.DataFrame([subject for subject in features])
	return feature1

def classify(models,models_w,test_data,column):
	print("With Energy Coefficients\n")
	mixture_result={}
	pers={}
	for phoneme in unique_labels:
		scores=(models[phoneme]).score_samples(test_data)
		mixture_result[phoneme]=scores
	for i in range(0,len(mixture_result['uh'])):
		max_label='sil'
		max_prob=-float('inf')
		for k in unique_labels:
			if(max_prob<(mixture_result[k])[i]):
				max_prob=(mixture_result[k])[i]
				max_label=k
		file1=test_mfcc_region[i]+"_"+test_mfcc_speaker[i]+"_"+test_mfcc_sentence[i]
		if(max_label==test_mfcc_labels[i]):
			
			if(file1 in pers):
				(pers[file1])['count']=(pers[file1])['count']+1
			else:
				pers[file1]={}
				(pers[file1])['count']=1
				(pers[file1])['total']=0

		elif(max_label=='space' and test_mfcc_labels[i]==''):

			if(file1 in pers):
				(pers[file1])['count']=(pers[file1])['count']+1

			else:
				pers[file1]={}
				(pers[file1])['count']=1
				(pers[file1])['total']=0

		if(file1 in pers):
			(pers[file1])['total']=(pers[file1])['total']+1
		else:
			pers[file1]={}
			(pers[file1])['total']=1
			(pers[file1])['count']=0
	count=0
	total=0
	for sen in pers:
		accuracy=(pers[sen])['count']/(pers[sen])['total']
		accuracy=1-accuracy
		print(str(sen) + "---->"+(str(accuracy))+"\n")
		count=count+(pers[sen])['count']
		total=total+(pers[sen])['total']

	print("Frame Accuracy------->", str(count/total))
	print("Without Energy Coefficients\n")
	mixture_result={}
	pers={}
	for phoneme in unique_labels:
		scores=(models_w[phoneme]).score_samples(test_data.drop(columns=column,axis=1))
		mixture_result[phoneme]=scores
	for i in range(0,len(mixture_result['uh'])):
		max_label='sil'
		max_prob=-float('inf')
		for k in unique_labels:
			if(max_prob<(mixture_result[k])[i]):
				max_prob=(mixture_result[k])[i]
				max_label=k
		file1=test_mfcc_region[i]+"_"+test_mfcc_speaker[i]+"_"+test_mfcc_sentence[i]
		if(max_label==test_mfcc_labels[i]):
			
			if(file1 in pers):
				(pers[file1])['count']=(pers[file1])['count']+1
			else:
				pers[file1]={}
				(pers[file1])['count']=1
				(pers[file1])['total']=0

		elif(max_label=='space' and test_mfcc_labels[i]==''):

			if(file1 in pers):
				(pers[file1])['count']=(pers[file1])['count']+1

			else:
				pers[file1]={}
				(pers[file1])['count']=1
				(pers[file1])['total']=0

		if(file1 in pers):
			(pers[file1])['total']=(pers[file1])['total']+1
		else:
			pers[file1]={}
			(pers[file1])['total']=1
			(pers[file1])['count']=0
	count=0
	total=0
	for sen in pers:
		accuracy=(pers[sen])['count']/(pers[sen])['total']
		accuracy=1-accuracy
		print(str(sen) + "---->"+(str(accuracy))+"\n")
		count=count+(pers[sen])['count']
		total=total+(pers[sen])['total']

	print("Frame Accuracy------->", str(count/total))


mfcc_models={}
mfcc_models_w={}
mfcc_delta_models={}
mfcc_delta_models_w={}
mfcc_delta_delta_models={}
mfcc_delta_delta_models_w={}

lns = open("unique_labels.txt",'r')
unique_labels=[]

for line in lns:
	mfcc_models[(line.split('\n'))[0]]={}
	mfcc_models_w[(line.split('\n'))[0]]={}
	unique_labels.append((line.split('\n'))[0])

# change model names as per train.py
for phoneme in unique_labels:
	(mfcc_models[phoneme])['2']=pickle.load(open("./models/"+phoneme+"/ec_mfcc_2.pkl", 'rb'))
	(mfcc_models[phoneme])['4']=pickle.load(open("./models/"+phoneme+"/ec_mfcc_4.pkl", 'rb'))
	(mfcc_models[phoneme])['8']=pickle.load(open("./models/"+phoneme+"/ec_mfcc_8.pkl", 'rb'))
	(mfcc_models[phoneme])['16']=pickle.load(open("./models/"+phoneme+"/ec_mfcc_16.pkl", 'rb'))
	(mfcc_models[phoneme])['32']=pickle.load(open("./models/"+phoneme+"/ec_mfcc_32.pkl", 'rb'))
	(mfcc_models[phoneme])['64']=pickle.load(open("./models/"+phoneme+"/ec_mfcc_64.pkl", 'rb'))
	(mfcc_models[phoneme])['128']=pickle.load(open("./models/"+phoneme+"/ec_mfcc_128.pkl", 'rb'))
	(mfcc_models[phoneme])['256']=pickle.load(open("./models/"+phoneme+"/ec_mfcc_256.pkl", 'rb'))

	(mfcc_models_w[phoneme])['2']=pickle.load(open("./models/"+phoneme+"/wec_mfcc_2.pkl", 'rb'))
	(mfcc_models_w[phoneme])['4']=pickle.load(open("./models/"+phoneme+"/wec_mfcc_4.pkl", 'rb'))
	(mfcc_models_w[phoneme])['8']=pickle.load(open("./models/"+phoneme+"/wec_mfcc_8.pkl", 'rb'))
	(mfcc_models_w[phoneme])['16']=pickle.load(open("./models/"+phoneme+"/wec_mfcc_16.pkl", 'rb'))
	(mfcc_models_w[phoneme])['32']=pickle.load(open("./models/"+phoneme+"/wec_mfcc_32.pkl", 'rb'))
	(mfcc_models_w[phoneme])['64']=pickle.load(open("./models/"+phoneme+"/wec_mfcc_64.pkl", 'rb'))
	(mfcc_models_w[phoneme])['128']=pickle.load(open("./models/"+phoneme+"/wec_mfcc_128.pkl", 'rb'))
	(mfcc_models_w[phoneme])['256']=pickle.load(open("./models/"+phoneme+"/wec_mfcc_256.pkl", 'rb'))
	
	mfcc_delta_models[phoneme]=pickle.load(open("./models/"+phoneme+"/ec_delta.pkl", 'rb'))
	mfcc_delta_models_w[phoneme]=pickle.load(open("./models/"+phoneme+"/wec_delta.pkl", 'rb'))

	mfcc_delta_delta_models[phoneme]=pickle.load(open("./models/"+phoneme+"/ec_delta_delta.pkl", 'rb'))
	mfcc_delta_delta_models_w[phoneme]=pickle.load(open("./models/"+phoneme+"/wec_delta_delta.pkl", 'rb'))
df=pd.read_hdf("./features/mfcc/timit_test.hdf")
df_d=pd.read_hdf("./features/mfcc_delta/timit_test.hdf")
df_dd=pd.read_hdf("./features/mfcc_delta_delta/timit_test.hdf")
test_mfcc=get_features(df)
test_mfcc_labels=df['labels']
test_mfcc_region=df['region']
test_mfcc_speaker=df['speaker']
test_mfcc_sentence=df['sentence']
test_mfcc_delta=get_features(df_d)
test_mfcc_delta_labels=df_d['labels']
test_mfcc_delta_delta=get_features(df_dd)
test_mfcc_delta_delta_labels=df_dd['labels']

mixture_list=['2','4','8','16','32','64','128','256']
for mixture in mixture_list:
	print(str(mixture)+"\n")
	print("With Energy Coefficients\n")
	mixture_result={}
	pers={}
	for phoneme in unique_labels:
		scores=(mfcc_models[phoneme])[mixture].score_samples(test_mfcc)
		mixture_result[phoneme]=scores
	for i in range(0,len(mixture_result['uh'])):
		max_label='sil'
		max_prob=-float('inf')
		for k in unique_labels:
			if(max_prob<(mixture_result[k])[i]):
				max_prob=(mixture_result[k])[i]
				max_label=k
		file1=test_mfcc_region[i]+"_"+test_mfcc_speaker[i]+"_"+test_mfcc_sentence[i]
		if(max_label==test_mfcc_labels[i]):
			
			if(file1 in pers):
				(pers[file1])['count']=(pers[file1])['count']+1
			else:
				pers[file1]={}
				(pers[file1])['count']=1
				(pers[file1])['total']=0

		elif(max_label=='space' and test_mfcc_labels[i]==''):

			if(file1 in pers):
				(pers[file1])['count']=(pers[file1])['count']+1

			else:
				pers[file1]={}
				(pers[file1])['count']=1
				(pers[file1])['total']=0

		if(file1 in pers):
			(pers[file1])['total']=(pers[file1])['total']+1
		else:
			pers[file1]={}
			(pers[file1])['total']=1
			(pers[file1])['count']=0
	count=0
	total=0
	for sen in pers:
		accuracy=(pers[sen])['count']/(pers[sen])['total']
		accuracy=1-accuracy
		print(str(sen) + "---->"+(str(accuracy))+"\n")
		count=count+(pers[sen])['count']
		total=total+(pers[sen])['total']

	print("Frame Accuracy------->", str(count/total))
	print("Without Energy Coefficients\n")
	mixture_result={}
	pers={}
	for phoneme in unique_labels:
		scores=(mfcc_models_w[phoneme])[mixture].score_samples(test_mfcc.drop(columns=[0],axis=1))
		mixture_result[phoneme]=scores
	for i in range(0,len(mixture_result['uh'])):
		max_label='sil'
		max_prob=-float('inf')
		for k in unique_labels:
			if(max_prob<(mixture_result[k])[i]):
				max_prob=(mixture_result[k])[i]
				max_label=k
		file1=test_mfcc_region[i]+"_"+test_mfcc_speaker[i]+"_"+test_mfcc_sentence[i]
		if(max_label==test_mfcc_labels[i]):
			
			if(file1 in pers):
				(pers[file1])['count']=(pers[file1])['count']+1
			else:
				pers[file1]={}
				(pers[file1])['count']=1
				(pers[file1])['total']=0

		elif(max_label=='space' and test_mfcc_labels[i]==''):

			if(file1 in pers):
				(pers[file1])['count']=(pers[file1])['count']+1

			else:
				pers[file1]={}
				(pers[file1])['count']=1
				(pers[file1])['total']=0

		if(file1 in pers):
			(pers[file1])['total']=(pers[file1])['total']+1
		else:
			pers[file1]={}
			(pers[file1])['total']=1
			(pers[file1])['count']=0
	count=0
	total=0
	for sen in pers:
		accuracy=(pers[sen])['count']/(pers[sen])['total']
		accuracy=1-accuracy
		print(str(sen) + "---->"+(str(accuracy))+"\n")
		count=count+(pers[sen])['count']
		total=total+(pers[sen])['total']

	print("Frame Accuracy------->", str(count/total))
print("mfcc_delta\n")
classify(mfcc_delta_models, mfcc_delta_models_w,test_mfcc_delta,[0,13])
print("mfcc_delta_delta\n")
classify(mfcc_delta_delta_models, mfcc_delta_delta_models_w,[0,13,26])














