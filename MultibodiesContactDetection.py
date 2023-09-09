# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, LSTM, Bidirectional, Dropout,GRU, Concatenate
import bvh
import os
import numpy as np
import math 
import pandas as pd
import pickle
import shutil


class Attention(tf.keras.Model):
  def __init__(self,return_sequences=False):
    super(Attention, self).__init__()

    self.return_sequences = return_sequences
    self.V = Dense(1)

  def call(self, values, query): 

    hidden_with_time_axis = tf.expand_dims(query, 2)    
    score = self.V(tf.matmul(values,hidden_with_time_axis))
    attention_weights = tf.nn.softmax(score, axis=1)
    context = attention_weights * values
    if not self.return_sequences:
        context = tf.reduce_sum(context, axis=1)

    return context





contact_parts_to_index = {
    'BACK' : 0

    ,'LUARM' : 1
    ,'LLARM' : 2
    ,'LHAND' : 3

    ,'RUARM' : 4
    ,'RLARM' : 5
    ,'RHAND' : 6

    ,'LULEG' : 7
    ,'LLLEG' : 8
    ,'LFOOT' : 9

    ,'RULEG' : 10
    ,'RLLEG' : 11
    ,'RFOOT' : 12

    ,'HIP' : 13

    ,'LKNEE' : 14
    ,'RKNEE' : 15

    ,'LELBOW' : 16
    ,'RELBOW' : 17
    }


use_contact_parts = [
    'BACK',
    'LLARM',
    'LHAND',
    'RLARM',
    'RHAND',
    'LULEG',
    'LFOOT',
    'RULEG',
    'RFOOT',
    'HIP',
    ]

feature_parts_definition = [ 
    #'Head',
    'LeftArm',
    'LeftForeArm',
    'LeftHand', 
    'RightArm',
    'RightForeArm',
    'RightHand',
    'LeftUpLeg',
    'LeftLeg',
    'LeftFoot',
    'RightUpLeg',
    'RightLeg',
    'RightFoot'
    ]

feature_dot_pair = [
    ['LeftShoulder','LeftArm','LeftForeArm'],
    ['LeftArm','LeftForeArm','LeftHand'],#
    ['LeftArm','LeftShoulder','LeftHand','LeftShoulder'],#

    ['RightShoulder','RightArm','RightForeArm'],
    ['RightArm','RightForeArm','RightHand'],
    ['RightArm','RightShoulder','RightHand','RightShoulder'],

    ['Spine3','Hips','LeftLeg','LeftUpLeg'],
    ['LeftUpLeg','LeftLeg','LeftFoot'],
    ['LeftUpLeg','LeftLeg','LeftFoot','LeftUpLeg'],

    ['Spine3','Hips','RightLeg','RightUpLeg'],
    ['RightUpLeg','RightLeg','RightFoot'],
    ['RightUpLeg','RightLeg','RightFoot','RightUpLeg'],

    #['Spine','Hips','LeftUpLeg'],
    #['Spine','Hips','RightUpLeg']
    ]


dump_idx_to_idx = {
    0:[0,0]
    ,1:[0,1]
    ,2:[0,2]
    ,3:[0,3]
    ,4:[0,4]
    ,5:[0,5]
    ,6:[2,0]
    ,7:[2,1]
    ,8:[2,4]
    ,9:[2,5]
    ,10:[3,0]
    ,11:[3,1]
    ,12:[3,4]
    ,13:[3,5]
    ,14:[4,0]
    ,15:[4,1]
    ,16:[4,4]
    ,17:[4,5]
    ,18:[5,0]
    ,19:[5,1]
    ,20:[5,2]
    ,21:[5,3]
    ,22:[5,4]
    ,23:[5,5]
    }


DirList = [            
            'data/train/1door',#21
            'data/train/2posCabinet/2posCabinetR',#21
            'data/train/4posChair/4posChairArms',#21
            'data/train/4posChair/4posChairBack',#21
            'data/train/4posChair/4posChairHand',#21
            'data/train/6Water/6WaterRight',#21
            'data/train/7Call/7CallRight',#21
            'data/train/9Walk/9WalkFourLeft',#21
            'data/train/9Walk/9WalkFourRight',#21
            'data/train/9Walk/9WalkOneLeft',#21
            'data/train/9Walk/9WalkOneRight',#21
            ]

character_scale = 0.005


def read_contact_file(file):
    contact_info = []    
    f = open(file,'r')
    lines = f.readlines()
    for line in lines:
        data = [float(x) for x in line.strip().split(',')]        
        contact_info.append(data)
    f.close()    
    return contact_info

def build_feature_from_motion(motion):
    
    f_motion = []
    
    for i in range(0,motion.frames):
        motion.updateFrame(i,character_scale)
                    
        #distance from Hips to each parts.
        f_frame = []
        p_hip = motion.getJoint('Hips').worldpos
        for j, parts in enumerate(feature_parts_definition):
            p = motion.getJoint(parts).worldpos
            dis = np.linalg.norm(p - p_hip)                        
            f_frame.append(dis)

        #
        for feature in feature_parts_definition:
            p0 = motion.getJoint(feature).worldpos
            p1 = p0
            p2 = p0
            parent = motion.getJoint(feature).parent
            max_distance = 0.0
            while  parent!= None:
                p2 = parent.worldpos
                max_distance +=  float(np.linalg.norm(p2-p1))
                p1 = p2
                parent = parent.parent
            f_frame.append(round(float(np.linalg.norm(p0-p1))/max_distance,3))
               

        #angular velocity
        for parts in feature_parts_definition:
            for j in range(0,3):
                if(i!=0):
                    f_frame.append(round(math.radians(motion.getJoint(parts).frames[i][j] - motion.getJoint(parts).frames[i-1][j])/motion.frameTime,5))
                else:#0 at first frame.
                    f_frame.append(0)

        #dot product
        for q, parts in enumerate(feature_dot_pair):
            v0 = np.array([])
            v1 = np.array([])

            if len(parts)==3:
                v0 = motion.getJoint(parts[0]).worldpos-motion.getJoint(parts[1]).worldpos
                v1 = motion.getJoint(parts[2]).worldpos-motion.getJoint(parts[1]).worldpos
            else:
                v0 = motion.getJoint(parts[0]).worldpos-motion.getJoint(parts[1]).worldpos
                v1 = motion.getJoint(parts[2]).worldpos-motion.getJoint(parts[3]).worldpos

            angle = np.arccos(np.dot(v0,v1)/(np.linalg.norm(v0)*np.linalg.norm(v1)))

            f_frame.append(angle)
                        
        f_motion.append(f_frame)   

    return f_motion

def removeAllFile(filePath):
    if os.path.exists(filePath):
        for file in os.scandir(filePath):
            os.remove(file.path) 


if __name__ == "__main__":

    np.set_printoptions(precision=4,suppress=True)
    
    num_fold = 7

    data_set = []

    for i in range(num_fold):
        data_set.append([])

    num_file = 21
    num_file = num_file - num_file%num_fold
    num_in_fold = num_file/num_fold

    

    for folder in DirList:
        print('Loading data:',folder)
        f_list = os.listdir(folder)
        f_list = [s for s in f_list if '.bvh' in s]

        if len(f_list)>num_file:
            f_list = f_list[:num_file]

        fold_idx = 0

        for i, f in enumerate(f_list):

            pickfile = folder+'/'+f.replace('bvh','pickle')

            if os.path.exists(pickfile):
                data_set[fold_idx].append(pickle.load(open(pickfile, 'rb')))
            else:
                dic_data = {
                    'm_file':folder+'/'+f,
                    'c_file':folder+'/'+f.replace('bvh','csv'),
                    'mc_file':'data/result/'+f,
                    'cc_file':'data/result/'+f.replace('bvh','csv'),
                    'feature':build_feature_from_motion(bvh.Skeleton(folder+'/'+f,character_scale)),
                    'contact':read_contact_file(folder+'/'+f.replace('bvh','csv'))
                    }
                data_set[fold_idx].append(dic_data)

                with open(pickfile,'wb') as fw:
                    pickle.dump(dic_data, fw)

            if i%num_in_fold == (num_in_fold-1):
                fold_idx = fold_idx + 1

    data_set = np.array(data_set)
    
    
    seq_length = 32

    smothing_frame = 3
    
    
    #train data: fold index, motion index, data index, sequence num, feature num
    for fold in data_set:
        for data in fold:
            xdata = []
            ydata = []

            xdata_i = []
            ydata_i = []

            f = np.array(data['feature'])    
            c = np.array(data['contact'])

            if smothing_frame!=0:
                for part in use_contact_parts:
                    for i in range(smothing_frame,len(c)-smothing_frame):
                        if (c[i-1,contact_parts_to_index[part]]==0 and c[i,contact_parts_to_index[part]]==1):
                            for j in range(-smothing_frame,smothing_frame+1,1):
                                c[i+j,contact_parts_to_index[part]] = 1 / (1 + math.exp(-j))

                        elif(c[i-1,contact_parts_to_index[part]]==1 and c[i,contact_parts_to_index[part]]==0):
                            for j in range(-smothing_frame,smothing_frame+1,1):
                                c[i+j,contact_parts_to_index[part]] = 1 / (1 + math.exp(j))


            for i in range(1, len(f) - seq_length+1):                
                x = f[i:i + seq_length, :]
                xdata.append(x)
                xdata_i.append(np.flip(x,axis=0))

                y = np.array([])
                y_i = np.array([])

                #Extract only use_contact_parts from contact data.
                for part in use_contact_parts:
                    y =np.append(y,c[i+seq_length-1, contact_parts_to_index[part]])  
                    y_i =np.append(y_i,c[i-1, contact_parts_to_index[part]])

                ydata.append(y)
                ydata_i.append(y_i)                

            data['x'] = xdata
            data['y'] = ydata

            data['x_i'] = xdata_i
            data['y_i'] = ydata_i

    x_dim = len(data_set[0][0]['feature'][0])
    y_dim = len(use_contact_parts)

    train_fold_idx = [0,1,2,3,4,5]
    test_fold_idx = [6]
    
    isTraining = True
    isEvaluating = False
    doGenerateMotion = False

    epochs = 15
    batch_size = 32
    learning_rate = 0.001
    
    drop_rate = 0.35

    trainX = []
    trainY = [] 


    for i in train_fold_idx:
        for data in data_set[i]:            
            for x in data['x']:                
                trainX.append(x)                                
            for y in data['y']:                
                trainY.append(y)

    trainX = np.array(trainX)
    trainY = np.array(trainY)
    
    trainable_components={
        'f':True,
        'BACK':True,
        'LLARM':True,
        'LHAND':True,
        'RLARM':True,
        'RHAND':True,
        'LULEG':True,
        'LFOOT':True,
        'RULEG':True,
        'RFOOT':True,
    }
    

    model_file = 'model/new.hdf5'
    weight_file = 'model/new.hdf5'

    model_feature = keras.Sequential()
    model_parts = keras.Sequential()

        

    if os.path.exists(model_file):
        del model_parts#
        model_parts = keras.models.load_model(model_file)
        model_parts.summary()
        
    else:
        f_input = Input(shape=(seq_length,x_dim),name='f_input')


        f_l0 = Bidirectional(LSTM(seq_length, return_sequences=True,name='f_l0'))(f_input)
        f_l1 = Bidirectional(LSTM(seq_length, return_sequences=True,name='f_l1'))(f_l0)
        f_l2 = Bidirectional(LSTM(seq_length, return_sequences=True,name='f_l2'))(f_l1)

        f_l3, f_h, _, b_h, _ = Bidirectional(LSTM(seq_length, return_sequences=True, return_state=True,name='f_output'))(f_l2)#, return_sequences=True
        
        state_h = Concatenate()([f_h, b_h]) # 
        attention = Attention(return_sequences=False) # Attention Layer

        context_vector = attention(f_l3, state_h)
        
        #f_output = Bidirectional(LSTM(seq_length, return_sequences=True, return_state=True,name='f_output'))(f_l2)#, return_sequences=True
        model_feature = Model(inputs=f_input,outputs=context_vector)

        
        
        #define part model...
        def get_part_model(name):
            part_l0 = Dense(8, name=name+'_l0')(model_feature.output)
            part_l1 = Dense(8, name=name+'_l1')(part_l0)
            part_l2 = Dense(8, name=name+'_l2')(part_l1)
            part_l3 = Dense(8, name=name+'_l3')(part_l2)
            part_l4 = Dense(8, name=name+'_l4')(part_l3)                
            part_output = Dense(1, activation='sigmoid',name=name+'_output')(part_l4)
            return part_output

        model_outputs = []

        for part in use_contact_parts:
            model_outputs.append(get_part_model(part))                
            
        model_parts = Model(inputs=[f_input],outputs=model_outputs)

        if os.path.exists(weight_file):
            model_parts.load_weights(weight_file, by_name=True)

        

    if isTraining:
         
        filepath = "model/{epoch:02d}-{loss:.4f}.hdf5"


        def scheduler(epoch, lr):
            if epoch < 10:
                return lr
            else:
                return lr * tf.math.exp(-0.1)

        callback_chkM = keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, mode='min')
        callback_lrsc = keras.callbacks.LearningRateScheduler(scheduler)
        callback_reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.0001)
        
        #print weight for debugging
        print_weights = keras.callbacks.LambdaCallback(on_epoch_end=lambda batch, logs: print(model_parts.get_layer('RFOOT_output').get_weights()))

        
        #
        for layers in model_parts.layers:
            layer_name = layers.name.split('_')[0]
            if layer_name in trainable_components:
                layers.trainable = trainable_components[layer_name]

        callback_list = [callback_chkM]
        
        model_parts.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate = learning_rate))
        #activations = keract.get_activations(model, image,layer_names='block1_conv1')
        history = model_parts.fit(trainX,list(np.transpose(trainY)), epochs=epochs, batch_size=batch_size,use_multiprocessing=True, callbacks=callback_list)#

        with open('trainHistoryDict', 'wb') as file_pi:
                pickle.dump(history.history, file_pi)

    if isEvaluating:
        eval_dataX = []
        eval_dataY = []
        
        for f_idx in test_fold_idx:
            for data in data_set[f_idx]:            
                for x in data['x']:
                    eval_dataX.append(x)

                for y in data['y']:
                    eval_dataY.append(y)
                

        eval_dataX = np.array(eval_dataX)
        eval_dataY = np.array(eval_dataY)

        
        min_loss_model = [1.0, '']
        for file_name in os.listdir('model'):            
            if '.hdf5' in file_name:
                print('Model: ',file_name)                
                model_parts.load_weights('model/'+file_name, by_name=True)
                model_parts.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate = learning_rate))
                #model.summary()
                hist = model_parts.evaluate(eval_dataX, [eval_dataY[:,0],eval_dataY[:,1],eval_dataY[:,2],eval_dataY[:,3]],use_multiprocessing=True,verbose=0)
                print('model/'+file_name,(hist[1]+hist[2]))
                if min_loss_model[0]>(hist[1]+hist[2]):
                    min_loss_model[0] = (hist[1]+hist[2])
                    min_loss_model[1] = 'model/'+file_name
                    print('MIN Model:',min_loss_model[1],min_loss_model[0])
        
        print('Final MIN Model:',min_loss_model[1],min_loss_model[0])
        model_parts.load_weights(min_loss_model[1], by_name=True)
        model_parts.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate = learning_rate))
        

    if doGenerateMotion:
        removeAllFile('data/result')

        for f_idx in test_fold_idx:
            for motion in data_set[f_idx]:
                result = np.array(model_parts.predict(np.array(motion['x'])))
        
                if len(result.shape)>2:
                    result = result.reshape(result.shape[0],result.shape[1])
        
                print(motion['m_file'])        
        
                if result.shape[0] < result.shape[1]:
                    result = np.transpose(result)

                debug_data = result

                np.savetxt(motion['cc_file'].replace('.csv','_temp.csv'),np.hstack((np.round(result,3), np.array(motion['y']))),delimiter=",")

                result = np.round(result)
            
                contact_data = np.zeros((result.shape[0],len(contact_parts_to_index))).tolist()        

                for i, frame in enumerate(contact_data):
                    for j in range(len(use_contact_parts)):                
                        frame[contact_parts_to_index[use_contact_parts[j]]] = result[i][j]
    
                for i in range(seq_length):
                    contact_data.insert(0,[0 for i in range(len(contact_parts_to_index))])

                pd.DataFrame(contact_data).to_csv(motion['cc_file'],header=False,index = False) 
                shutil.copy(motion['m_file'],motion['mc_file'])